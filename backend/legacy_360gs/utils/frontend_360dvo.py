"""360DVO pose-prior adapters for the panoramic frontend.

The offline adapter reads a TUM trajectory exported by
``external_baselines/360DVO``.  The online adapter wraps the DPVO object used by
360DVO and advances it one ERP frame at a time inside the SLAM frontend.
"""

from __future__ import annotations

import math
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rotation

from backend.legacy_360gs.utils.erp_geometry import SLAM_TO_PFGS360_AXES


@dataclass
class PosePrior360DVO:
    w2c: np.ndarray
    info: dict


def _read_tum_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"TUM trajectory rows must be 't x y z qx qy qz qw': {path}"
                )
            rows.append([float(v) for v in parts])
    if not rows:
        raise ValueError(f"TUM trajectory is empty: {path}")

    arr = np.asarray(rows, dtype=np.float64)
    stamps = arr[:, 0].copy()
    c2w = []
    for row in arr:
        tx, ty, tz, qx, qy, qz, qw = row[1:].tolist()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        c2w.append(T)
    return stamps, np.stack(c2w, axis=0)


def _axis_conversion_matrix(mode: str) -> np.ndarray:
    mode = str(mode or "identity").lower()
    if mode in {"identity", "none", "slam"}:
        return np.eye(4, dtype=np.float64)
    if mode in {"dvo_to_360uav_slam", "360dvo_to_slam", "dvo_to_slam"}:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return T
    if mode in {
        "pfgs360_to_slam",
        "slam_to_pfgs360",
        "opengl_to_slam",
        "slam_to_opengl",
    }:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = SLAM_TO_PFGS360_AXES
        return T
    raise ValueError(
        "Unsupported Training.frontend_360dvo.axis_conversion "
        f"'{mode}'. Use identity, pfgs360_to_slam, or dvo_to_360uav_slam."
    )


def _sample_depth_nearest(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sampled = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    sampled[valid] = depth[v[valid], u[valid]].astype(np.float32)
    return sampled


def _robust_dap_to_dvo_scale(
    dvo_depth: np.ndarray,
    dap_depth: np.ndarray,
    valid: np.ndarray,
    *,
    min_count: int,
    value_min: float,
    value_max: float,
    trim_quantile: float = 0.1,
    mad_max: float = 0.35,
    previous_scale: float | None = None,
    jump_max: float = 0.0,
) -> tuple[Optional[float], dict]:
    """Estimate a positive scale that maps DAP depth into 360DVO depth."""
    dvo = np.asarray(dvo_depth, dtype=np.float32).reshape(-1)
    dap = np.asarray(dap_depth, dtype=np.float32).reshape(-1)
    valid_np = np.asarray(valid, dtype=bool).reshape(-1)
    valid_np &= (
        np.isfinite(dvo)
        & np.isfinite(dap)
        & (dvo > 1e-6)
        & (dap > 1e-6)
    )
    ratios = dvo[valid_np] / np.maximum(dap[valid_np], 1e-6)
    ratios = ratios[
        np.isfinite(ratios)
        & (ratios > 0.0)
        & (ratios >= float(value_min))
        & (ratios <= float(value_max))
    ]
    stats = {
        "dap_to_dvo_ratio_count": int(ratios.size),
        "dap_to_dvo_ratio_trimmed_count": 0,
        "dap_to_dvo_ratio_median": float("nan"),
        "dap_to_dvo_ratio_mad_norm": float("nan"),
        "dap_to_dvo_reject_reason": "",
    }
    if ratios.size < int(min_count):
        stats["dap_to_dvo_reject_reason"] = "few_valid_ratios"
        return None, stats

    trim_q = float(np.clip(trim_quantile, 0.0, 0.45))
    if trim_q > 0.0:
        lo, hi = np.quantile(ratios, [trim_q, 1.0 - trim_q])
        ratios = ratios[(ratios >= lo) & (ratios <= hi)]
    stats["dap_to_dvo_ratio_trimmed_count"] = int(ratios.size)
    if ratios.size < int(min_count):
        stats["dap_to_dvo_reject_reason"] = "few_trimmed_ratios"
        return None, stats

    scale_obs = float(np.median(ratios))
    mad_norm = float(np.median(np.abs(ratios - scale_obs)) / max(abs(scale_obs), 1e-6))
    stats.update(
        {
            "dap_to_dvo_ratio_median": float(scale_obs),
            "dap_to_dvo_ratio_mad_norm": float(mad_norm),
        }
    )
    if not np.isfinite(scale_obs) or scale_obs <= 0.0:
        stats["dap_to_dvo_reject_reason"] = "bad_scale"
        return None, stats
    if float(mad_max) > 0.0 and mad_norm > float(mad_max):
        stats["dap_to_dvo_reject_reason"] = "ratio_mad"
        return None, stats
    if (
        previous_scale is not None
        and np.isfinite(float(previous_scale))
        and float(previous_scale) > 0.0
        and float(jump_max) > 1.0
    ):
        jump = max(scale_obs / float(previous_scale), float(previous_scale) / scale_obs)
        stats["dap_to_dvo_ratio_jump"] = float(jump)
        if jump > float(jump_max):
            stats["dap_to_dvo_reject_reason"] = "ratio_jump"
            return None, stats
    stats["dap_to_dvo_reject_reason"] = ""
    return float(np.clip(scale_obs, value_min, value_max)), stats


def _robust_weighted_median(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.median(values))
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return float(np.median(values))
    cdf = np.cumsum(weights) / total
    return float(values[np.searchsorted(cdf, 0.5, side="left")])


class SparseDepth360DVO:
    """Optional 360DVO sparse-depth cache used for metric scale recovery.

    Supported formats:
      - one ``.npz`` with ``uv`` and ``depth`` arrays shaped ``(F, N, 2)`` and
        ``(F, N)``, plus optional ``weight``.
      - one ``.npz`` with per-frame keys ``uv_<idx>``, ``depth_<idx>``.
      - a directory containing ``<idx>.npz`` or ``frame_<idx:06d>.npz`` files.
    Depth is expected to be in the same arbitrary 360DVO scale as the TUM
    trajectory translation.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self._npz = None
        self._is_dir = False
        if not path:
            return
        p = Path(path)
        self._is_dir = p.is_dir()
        if p.is_file():
            self._npz = np.load(str(p), allow_pickle=True)

    def _load_from_npz(self, idx: int):
        if self._npz is None:
            return None
        z = self._npz
        uv_key = f"uv_{idx}"
        depth_key = f"depth_{idx}"
        weight_key = f"weight_{idx}"
        if uv_key in z and depth_key in z:
            return (
                np.asarray(z[uv_key], dtype=np.float32),
                np.asarray(z[depth_key], dtype=np.float32),
                np.asarray(z[weight_key], dtype=np.float32) if weight_key in z else None,
            )
        if "uv" in z and "depth" in z:
            uv = np.asarray(z["uv"], dtype=np.float32)
            depth = np.asarray(z["depth"], dtype=np.float32)
            if uv.ndim >= 3 and idx < uv.shape[0] and idx < depth.shape[0]:
                weight = None
                if "weight" in z:
                    weight_all = np.asarray(z["weight"], dtype=np.float32)
                    if weight_all.ndim >= 2 and idx < weight_all.shape[0]:
                        weight = weight_all[idx]
                return uv[idx], depth[idx], weight
        return None

    def _load_from_dir(self, idx: int):
        if not self.path or not self._is_dir:
            return None
        root = Path(self.path)
        candidates = [
            root / f"{idx}.npz",
            root / f"{idx:06d}.npz",
            root / f"frame_{idx:06d}.npz",
            root / f"frame_{idx:04d}.npz",
        ]
        for path in candidates:
            if path.is_file():
                z = np.load(str(path), allow_pickle=True)
                if "uv" in z and "depth" in z:
                    weight = np.asarray(z["weight"], dtype=np.float32) if "weight" in z else None
                    return (
                        np.asarray(z["uv"], dtype=np.float32),
                        np.asarray(z["depth"], dtype=np.float32),
                        weight,
                    )
        return None

    def get(self, idx: int):
        if not self.path:
            return None
        return self._load_from_dir(idx) if self._is_dir else self._load_from_npz(idx)


class Frontend360DVO:
    def __init__(self, config: dict, save_dir: Optional[str] = None):
        training_cfg = config.get("Training", {})
        cfg = training_cfg.get("frontend_360dvo", {}) or {}
        self.enabled = str(training_cfg.get("frontend_mode", "spherical")).lower() in {
            "360dvo",
            "hybrid",
        }
        self.save_dir = save_dir
        self.trajectory_path = cfg.get("tum_trajectory") or cfg.get("trajectory_path")
        self.axis_conversion = cfg.get("axis_conversion", "identity")
        self.index_offset = int(cfg.get("index_offset", config.get("Dataset", {}).get("begin", 0) or 0))
        self.configured_scale = float(cfg.get("scale", 1.0))
        self.scale_ema_alpha = float(cfg.get("scale_ema_alpha", 0.2))
        self.min_sparse_points = int(cfg.get("min_sparse_points", 24))
        self.scale_min = float(cfg.get("scale_min", 1e-3))
        self.scale_max = float(cfg.get("scale_max", 1e3))
        self.depth_min = float(cfg.get("depth_min", training_cfg.get("ransac", {}).get("depth_min", 0.5)))
        self.depth_max = float(
            cfg.get(
                "depth_max",
                training_cfg.get(
                    "dap_depth_max_valid",
                    training_cfg.get("ransac", {}).get("depth_max", 80.0),
                ),
            )
        )
        self.scale = self.configured_scale
        self._warned_no_sparse = False
        self._stamps = None
        self._c2w_rel = None
        self.sparse = SparseDepth360DVO(cfg.get("sparse_depth_path") or cfg.get("sparse_depth_dir"))

        if self.enabled and self.trajectory_path:
            self._load_trajectory()

    @property
    def available(self) -> bool:
        return self._c2w_rel is not None and len(self._c2w_rel) > 0

    def _load_trajectory(self) -> None:
        if not os.path.isfile(self.trajectory_path):
            raise FileNotFoundError(
                f"360DVO TUM trajectory not found: {self.trajectory_path}"
            )
        stamps, c2w = _read_tum_trajectory(self.trajectory_path)
        A = _axis_conversion_matrix(self.axis_conversion)
        if not np.allclose(A, np.eye(4)):
            A_inv = np.linalg.inv(A)
            c2w = np.stack([A @ T @ A_inv for T in c2w], axis=0)

        ref = np.linalg.inv(c2w[0])
        self._c2w_rel = np.stack([ref @ T for T in c2w], axis=0)
        self._stamps = stamps

    def maybe_update_scale(self, frame_idx: int, mono_depth: Optional[np.ndarray]) -> dict:
        if mono_depth is None:
            return {"scale": float(self.scale), "scale_source": "configured_no_mono"}
        if not self.sparse.path:
            if not self._warned_no_sparse:
                self._warned_no_sparse = True
            return {"scale": float(self.scale), "scale_source": "configured_no_sparse"}

        sparse = self.sparse.get(frame_idx + self.index_offset)
        if sparse is None:
            return {"scale": float(self.scale), "scale_source": "configured_missing_sparse"}

        uv, dvo_depth, weight = sparse
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        dvo_depth = np.asarray(dvo_depth, dtype=np.float32).reshape(-1)
        if uv.shape[0] != dvo_depth.shape[0]:
            return {"scale": float(self.scale), "scale_source": "bad_sparse_shape"}

        mono = np.asarray(mono_depth, dtype=np.float32)
        if mono.ndim == 3:
            mono = mono[0]
        mono_sampled = _sample_depth_nearest(mono, uv)
        valid = (
            np.isfinite(mono_sampled)
            & np.isfinite(dvo_depth)
            & (mono_sampled > self.depth_min)
            & (mono_sampled < self.depth_max)
            & (dvo_depth > 1e-6)
        )
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float32).reshape(-1)
            if weight.shape[0] == valid.shape[0]:
                valid &= np.isfinite(weight) & (weight > 0.0)
            else:
                weight = None
        if int(valid.sum()) < self.min_sparse_points:
            return {
                "scale": float(self.scale),
                "scale_source": "configured_few_sparse",
                "sparse_valid": int(valid.sum()),
            }

        ratios = mono_sampled[valid] / np.maximum(dvo_depth[valid], 1e-6)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size < self.min_sparse_points:
            return {"scale": float(self.scale), "scale_source": "configured_bad_ratios"}
        lo, hi = np.quantile(ratios, [0.1, 0.9])
        keep = (ratios >= lo) & (ratios <= hi)
        ratios = ratios[keep]
        valid_weight = weight[valid][keep] if weight is not None else None
        scale_obs = _robust_weighted_median(ratios, valid_weight)
        if not np.isfinite(scale_obs):
            return {"scale": float(self.scale), "scale_source": "configured_nan_scale"}
        scale_obs = float(np.clip(scale_obs, self.scale_min, self.scale_max))
        alpha = float(np.clip(self.scale_ema_alpha, 0.0, 1.0))
        self.scale = (1.0 - alpha) * float(self.scale) + alpha * scale_obs
        return {
            "scale": float(self.scale),
            "scale_observed": float(scale_obs),
            "scale_source": "sparse_mono_depth",
            "sparse_valid": int(valid.sum()),
        }

    def get_prior(
        self,
        frame_idx: int,
        mono_depth: Optional[np.ndarray] = None,
        **_: object,
    ) -> Optional[PosePrior360DVO]:
        if not self.available:
            return None
        traj_idx = frame_idx + self.index_offset
        if traj_idx < 0 or traj_idx >= len(self._c2w_rel):
            return None
        scale_info = self.maybe_update_scale(frame_idx, mono_depth)
        c2w = self._c2w_rel[traj_idx].copy()
        c2w[:3, 3] *= float(self.scale)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        prev_idx = max(traj_idx - 1, 0)
        prev_c2w = self._c2w_rel[prev_idx].copy()
        prev_c2w[:3, 3] *= float(self.scale)
        prev_w2c = np.linalg.inv(prev_c2w).astype(np.float32)
        rel = w2c.astype(np.float64) @ np.linalg.inv(prev_w2c.astype(np.float64))
        t_norm = float(np.linalg.norm(rel[:3, 3]))
        info = {
            "success": True,
            "source": "360dvo_tum",
            "trajectory_index": int(traj_idx),
            "timestamp": float(self._stamps[traj_idx]),
            "t_norm": t_norm,
            "inlier_ratio": 1.0,
            "valid_depth_ratio": 1.0,
            "matches": 0,
            "inliers": 0,
            "mean_ang_deg": 0.0,
            "median_ang_deg": 0.0,
            **scale_info,
        }
        return PosePrior360DVO(w2c=w2c, info=info)


class Online360DVOFrontend:
    """Online 360DVO/DPVO frontend.

    The heavy 360DVO imports and CUDA model allocation are intentionally lazy:
    ``FrontEnd`` is a multiprocessing process, and loading DPVO in ``__init__``
    would touch CUDA in the parent process before spawn.
    """

    mode = "online"

    def __init__(self, config: dict, save_dir: Optional[str] = None):
        training_cfg = config.get("Training", {})
        cfg = training_cfg.get("frontend_360dvo", {}) or {}
        repo_root = Path(__file__).resolve().parents[1]
        external_root = cfg.get("external_root") or str(
            repo_root / "external_baselines" / "360DVO"
        )
        self.external_root = Path(external_root)
        self.network_path = str(
            cfg.get("network")
            or cfg.get("network_path")
            or (self.external_root / "360dvo.pth")
        )
        self.config_path = str(
            cfg.get("config")
            or cfg.get("config_path")
            or (self.external_root / "config" / "360.yaml")
        )
        self.axis_conversion = cfg.get("axis_conversion", "dvo_to_360uav_slam")
        self.input_color_order = str(cfg.get("input_color_order", "rgb_to_bgr")).lower()
        self.device = str(cfg.get("device", "cuda"))
        self.save_dir = save_dir

        self.pose_scale_policy = str(
            cfg.get("pose_scale_policy", "align_pose_to_depth")
        ).lower()
        if self.pose_scale_policy in {"raw", "raw_dvo", "none", "identity"}:
            self.pose_scale_policy = "raw_dvo"
        elif self.pose_scale_policy in {"fixed", "constant"}:
            self.pose_scale_policy = "fixed"
        elif self.pose_scale_policy in {
            "align_pose_to_depth",
            "scale_pose_to_depth",
            "dap_to_pose",
        }:
            self.pose_scale_policy = "align_pose_to_depth"
        else:
            raise ValueError(
                "Unsupported Training.frontend_360dvo.pose_scale_policy "
                f"'{self.pose_scale_policy}'. Use raw_dvo, fixed, or "
                "align_pose_to_depth."
            )

        self.depth_scale_policy = str(
            cfg.get("depth_scale_policy", "none")
        ).lower()
        if self.depth_scale_policy in {
            "align_depth_to_dvo",
            "dvo",
            "dvo_scale",
            "scale_depth_to_dvo",
        }:
            self.depth_scale_policy = "align_depth_to_dvo"
        elif self.depth_scale_policy in {
            "bootstrap_init_only",
            "init_bootstrap",
            "bootstrap",
        }:
            self.depth_scale_policy = "bootstrap_init_only"
        elif self.depth_scale_policy in {"none", "off", "disabled"}:
            self.depth_scale_policy = "none"
        else:
            raise ValueError(
                "Unsupported Training.frontend_360dvo.depth_scale_policy "
                f"'{self.depth_scale_policy}'. Use align_depth_to_dvo, "
                "bootstrap_init_only, or none."
            )
        self.use_sparse_prior_for_mapping = bool(
            cfg.get("use_sparse_prior_for_mapping", False)
        )

        self.configured_scale = float(cfg.get("scale", 1.0))
        self.scale = self.configured_scale
        self.scale_ema_alpha = float(cfg.get("scale_ema_alpha", 0.2))
        self.scale_min = float(cfg.get("scale_min", 1e-3))
        self.scale_max = float(cfg.get("scale_max", 1e3))
        self.min_sparse_points = int(cfg.get("min_sparse_points", 24))
        self.depth_min = float(cfg.get("depth_min", 0.5))
        self.depth_max = float(
            cfg.get(
                "depth_max",
                training_cfg.get(
                    "dap_depth_max_valid",
                    training_cfg.get("ransac", {}).get("depth_max", 80.0),
                ),
            )
        )
        self.scale_stable_frames = int(cfg.get("online_scale_stable_frames", 3))
        self.unstable_max_dt_m = float(cfg.get("online_unstable_max_dt_m", 0.5))
        self.max_dt_m = float(cfg.get("online_max_dt_m", 0.0))
        self.enable_pose_clamp = bool(
            cfg.get(
                "online_enable_pose_clamp",
                self.pose_scale_policy == "align_pose_to_depth",
            )
        )
        self.scale_requires_initialized = bool(
            cfg.get("online_scale_requires_initialized", True)
        )
        self.depth_scale = float(cfg.get("depth_scale", 1.0))
        self.depth_scale_ema_alpha = float(
            cfg.get("depth_scale_ema_alpha", self.scale_ema_alpha)
        )
        self.depth_scale_min = float(cfg.get("depth_scale_min", 1e-4))
        self.depth_scale_max = float(cfg.get("depth_scale_max", 1e4))
        self.depth_scale_stable_frames = int(
            cfg.get("depth_scale_stable_frames", self.scale_stable_frames)
        )
        self.dap_dvo_scale_update = str(
            cfg.get("dap_dvo_scale_update", "bootstrap_only")
        ).lower()
        if self.dap_dvo_scale_update in {"ema", "online", "continuous", "always"}:
            self.dap_dvo_scale_update = "ema_after_bootstrap"
        elif self.dap_dvo_scale_update in {"bootstrap", "bootstrap_init_only"}:
            self.dap_dvo_scale_update = "bootstrap_only"
        self.dap_dvo_scale_ema_alpha = float(
            cfg.get("dap_dvo_scale_ema_alpha", self.depth_scale_ema_alpha)
        )
        self.dap_dvo_scale_trim_quantile = float(
            cfg.get("dap_dvo_scale_trim_quantile", 0.1)
        )
        self.dap_dvo_scale_mad_max = float(
            cfg.get("dap_dvo_scale_mad_max", 0.35)
        )
        self.dap_dvo_scale_ratio_jump_max = float(
            cfg.get("dap_dvo_scale_ratio_jump_max", 2.0)
        )
        self.dap_dvo_scale_bootstrap_frames = max(
            1, int(cfg.get("init_depth_bootstrap_frames", 1))
        )
        self.depth_scale_requires_initialized = bool(
            cfg.get("depth_scale_requires_initialized", self.scale_requires_initialized)
        )
        self.debug_visualize_depth_scale = bool(
            cfg.get("debug_visualize_depth_scale", False)
        )
        self.depth_scale_vis_interval = max(
            1, int(cfg.get("depth_scale_vis_interval", 1))
        )
        self.depth_scale_vis_point_radius = max(
            1, int(cfg.get("depth_scale_vis_point_radius", 3))
        )
        self.warmup_confidence = float(cfg.get("online_warmup_confidence", 0.1))
        self.unstable_confidence = float(cfg.get("online_unstable_confidence", 0.5))
        self.stable_confidence = float(cfg.get("online_stable_confidence", 1.0))

        self._dpvo = None
        self._dpvo_cfg = None
        self._pops = None
        self._reference_c2w_raw: Optional[np.ndarray] = None
        self._last_w2c: Optional[np.ndarray] = None
        self._last_converted_c2w: Optional[np.ndarray] = None
        self._last_sparse_prior: Optional[dict] = None
        self._stable_observations = 0
        self._stable_depth_observations = 0
        self._processed_frame_ids: set[int] = set()
        self._scale_update_frame_ids: set[int] = set()
        self._depth_scale_frozen = False
        self._depth_scale_bootstrap_finalized = False
        self._depth_scale_bootstrap_observations: list[float] = []
        self._depth_scale_observation_count = 0
        self._last_scale_info = {
            "scale": float(self.scale),
            "scale_source": "configured",
            "scale_stable": self.pose_scale_policy in {"raw_dvo", "fixed"},
            "pose_scale": float(self.configured_scale),
            "pose_scale_policy": self.pose_scale_policy,
            "depth_scale": float(self.depth_scale),
            "dap_to_dvo_scale": float(self.depth_scale),
            "depth_scale_policy": self.depth_scale_policy,
            "depth_scale_source": "configured",
            "dap_to_dvo_scale_source": "configured",
            "depth_scale_stable": False,
            "dap_to_dvo_scale_stable": False,
            "dap_dvo_scale_update": self.dap_dvo_scale_update,
        }
        self._A = _axis_conversion_matrix(self.axis_conversion)
        self._A_inv = np.linalg.inv(self._A)
        self._debug_pose_dir = (
            Path(save_dir) / "online_360dvo" if save_dir is not None else None
        )
        self._debug_pose_frames: set[int] = set()

    @property
    def available(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._dpvo is not None:
            return
        if not self.external_root.is_dir():
            raise FileNotFoundError(f"360DVO root not found: {self.external_root}")
        if not os.path.isfile(self.network_path):
            raise FileNotFoundError(f"360DVO network not found: {self.network_path}")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"360DVO config not found: {self.config_path}")
        root_str = str(self.external_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from dpvo.config import cfg as dpvo_cfg  # noqa: WPS433
        from dpvo.dpvo import DPVO  # noqa: WPS433
        from dpvo import projective_ops as pops  # noqa: WPS433

        self._dpvo_cfg = dpvo_cfg.clone()
        self._dpvo_cfg.merge_from_file(self.config_path)
        self._dpvo_class = DPVO
        self._pops = pops

    def _image_to_dpvo(self, erp_image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(erp_image):
            raise TypeError("Online360DVOFrontend expects a torch ERP image tensor.")
        img = erp_image.detach()
        if img.ndim != 3:
            raise ValueError(f"ERP image must be CxHxW, got shape {tuple(img.shape)}")
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[0] != 3:
            raise ValueError(f"ERP image must have 3 channels, got {img.shape[0]}")
        img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        if self.input_color_order in {"rgb_to_bgr", "bgr"}:
            img = img[[2, 1, 0], :, :]
        return img.contiguous().to(self.device, non_blocking=True)

    @staticmethod
    def _intrinsics_for_erp(height: int, width: int, device: str) -> torch.Tensor:
        intrinsics = np.array(
            [
                width / (2.0 * math.pi),
                -height / math.pi,
                width / 2.0,
                height / 2.0,
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(intrinsics).to(device)

    @staticmethod
    def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
        pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(pose7[3:7]).as_matrix()
        T[:3, 3] = pose7[:3]
        return T

    def _refresh_traj(self) -> None:
        self._dpvo.traj = {}
        for i in range(int(self._dpvo.n)):
            self._dpvo.traj[int(self._dpvo.tstamps_[i].item())] = self._dpvo.poses_[i]

    def _current_raw_c2w(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._dpvo is None or int(self._dpvo.counter) <= 0:
            return None
        self._refresh_traj()
        query_idx = int(frame_idx)
        if query_idx not in self._dpvo.traj and query_idx not in self._dpvo.delta:
            query_idx = int(self._dpvo.counter) - 1
        if query_idx < 0:
            return None
        pose_se3 = self._dpvo.get_pose(query_idx)
        pose = pose_se3.inv().data.detach().float().cpu().numpy()
        if pose.ndim > 1:
            pose = pose.reshape(-1, 7)[0]
        return self._pose7_to_matrix(pose)

    def _dpvo_storage_index_for_frame(self, frame_idx: Optional[int]) -> int:
        if self._dpvo is None or int(self._dpvo.n) <= 0:
            return -1
        if frame_idx is None:
            return int(self._dpvo.n) - 1
        try:
            target = int(frame_idx)
            tstamps = self._dpvo.tstamps_[: int(self._dpvo.n)].detach().cpu().numpy()
            hits = np.flatnonzero(tstamps.astype(np.int64) == target)
            if hits.size > 0:
                return int(hits[-1])
        except Exception:
            pass
        return int(self._dpvo.n) - 1

    def _pose_translation_scale(self) -> float:
        if self.pose_scale_policy in {"raw_dvo", "fixed"}:
            return float(self.configured_scale)
        return float(self.scale)

    @staticmethod
    def _filtered_median_ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
        *,
        min_count: int,
        value_min: float,
        value_max: float,
    ) -> Optional[float]:
        ratios = numerator / np.maximum(denominator, 1e-6)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size < min_count:
            return None
        lo, hi = np.quantile(ratios, [0.1, 0.9])
        ratios = ratios[(ratios >= lo) & (ratios <= hi)]
        if ratios.size < min_count:
            return None
        ratio = float(np.median(ratios))
        if not np.isfinite(ratio):
            return None
        return float(np.clip(ratio, value_min, value_max))

    @staticmethod
    def _ratio_summary(
        numerator: np.ndarray,
        denominator: np.ndarray,
        *,
        min_count: int = 1,
        prefix: str = "ratio",
    ) -> dict:
        ratios = numerator / np.maximum(denominator, 1e-6)
        ratios = ratios[np.isfinite(ratios)]
        out = {f"{prefix}_count": int(ratios.size)}
        if ratios.size <= 0:
            return out
        q10, q50, q90 = np.quantile(ratios, [0.1, 0.5, 0.9])
        out.update(
            {
                f"{prefix}_p10": float(q10),
                f"{prefix}_median": float(q50),
                f"{prefix}_p90": float(q90),
                f"{prefix}_mean": float(np.mean(ratios)),
            }
        )
        filtered = ratios[(ratios >= q10) & (ratios <= q90)]
        out[f"{prefix}_filtered_count"] = int(filtered.size)
        if filtered.size >= max(1, int(min_count)):
            out[f"{prefix}_filtered_median"] = float(np.median(filtered))
        return out

    def _update_scale_from_sparse(
        self,
        uv_erp: np.ndarray,
        dvo_depth: np.ndarray,
        dvo_valid: np.ndarray,
        mono_depth: Optional[np.ndarray],
        mono_valid_mask: Optional[np.ndarray] = None,
    ) -> tuple[dict, Optional[np.ndarray]]:
        pose_stable = self.pose_scale_policy in {"raw_dvo", "fixed"}
        scale_info = {
            "scale": float(self._pose_translation_scale()),
            "scale_source": self.pose_scale_policy,
            "scale_stable": bool(pose_stable),
            "pose_scale": float(self._pose_translation_scale()),
            "pose_scale_policy": self.pose_scale_policy,
            "depth_scale": float(self.depth_scale),
            "depth_scale_policy": self.depth_scale_policy,
            "depth_scale_source": "configured",
            "depth_scale_stable": self._stable_depth_observations
            >= self.depth_scale_stable_frames,
            "sparse_valid": int(dvo_valid.sum()),
        }
        mono_sampled = None
        initialized = bool(getattr(self._dpvo, "is_initialized", False))
        needs_pose_scale = self.pose_scale_policy == "align_pose_to_depth"
        needs_depth_scale = self.depth_scale_policy in {
            "align_depth_to_dvo",
            "bootstrap_init_only",
        } and not self._depth_scale_frozen

        if self.scale_requires_initialized and needs_pose_scale and not initialized:
            self._stable_observations = 0
            scale_info.update(
                {
                    "scale": float(self.scale),
                    "scale_source": "dpvo_warmup_random_depth",
                    "scale_stable": False,
                    "pose_scale": float(self.scale),
                }
            )
        if (
            self.depth_scale_requires_initialized
            and needs_depth_scale
            and not initialized
        ):
            self._stable_depth_observations = 0
            scale_info.update(
                {
                    "depth_scale": float(self.depth_scale),
                    "depth_scale_source": "dpvo_warmup_random_depth",
                    "depth_scale_stable": False,
                }
            )

        if mono_depth is None or (not needs_pose_scale and not needs_depth_scale):
            return scale_info, mono_sampled

        if (
            (needs_pose_scale and self.scale_requires_initialized and not initialized)
            and (
                needs_depth_scale
                and self.depth_scale_requires_initialized
                and not initialized
            )
        ):
            return scale_info, mono_sampled

        mono = np.asarray(mono_depth, dtype=np.float32)
        if mono.ndim == 3:
            mono = mono[0]
        mono_sampled = _sample_depth_nearest(mono, uv_erp)
        depth_valid = (
            np.isfinite(mono_sampled)
            & (mono_sampled > self.depth_min)
            & (mono_sampled < self.depth_max)
            & dvo_valid
        )
        if mono_valid_mask is not None:
            valid_mask = np.asarray(mono_valid_mask, dtype=np.float32)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[0]
            if valid_mask.shape == mono.shape:
                sampled_valid = _sample_depth_nearest(valid_mask, uv_erp) > 0.5
                depth_valid &= sampled_valid
        scale_info["sparse_valid"] = int(depth_valid.sum())
        if int(depth_valid.sum()) < self.min_sparse_points:
            if needs_pose_scale:
                self._stable_observations = 0
                scale_info.update(
                    {
                        "scale": float(self._pose_translation_scale()),
                        "scale_source": "sparse_few_valid",
                        "scale_stable": bool(pose_stable),
                        "pose_scale": float(self._pose_translation_scale()),
                    }
                )
            if needs_depth_scale:
                self._stable_depth_observations = 0
                scale_info.update(
                    {
                        "depth_scale": float(self.depth_scale),
                        "depth_scale_source": "sparse_few_valid",
                        "depth_scale_stable": False,
                    }
                )
            return scale_info, mono_sampled

        mono_valid = mono_sampled[depth_valid]
        dvo_valid_depth = dvo_depth[depth_valid]
        scale_info.update(
            self._ratio_summary(
                dvo_valid_depth,
                mono_valid,
                min_count=self.min_sparse_points,
                prefix="depth_scale_ratio_dvo_over_dap",
            )
        )

        if needs_pose_scale and (
            not self.scale_requires_initialized or initialized
        ):
            pose_obs = self._filtered_median_ratio(
                mono_valid,
                dvo_valid_depth,
                min_count=self.min_sparse_points,
                value_min=self.scale_min,
                value_max=self.scale_max,
            )
            if pose_obs is None:
                self._stable_observations = 0
                scale_info.update(
                    {
                        "scale": float(self.scale),
                        "scale_source": "sparse_bad_ratios",
                        "scale_stable": False,
                        "pose_scale": float(self.scale),
                    }
                )
            else:
                alpha = float(np.clip(self.scale_ema_alpha, 0.0, 1.0))
                self.scale = (1.0 - alpha) * float(self.scale) + alpha * pose_obs
                self._stable_observations += 1
                stable = self._stable_observations >= self.scale_stable_frames
                scale_info.update(
                    {
                        "scale": float(self.scale),
                        "scale_observed": float(pose_obs),
                        "scale_source": "sparse_dap_depth"
                        if stable
                        else "sparse_dap_depth_unstable",
                        "scale_stable": bool(stable),
                        "pose_scale": float(self.scale),
                    }
                )

        if needs_depth_scale and (
            not self.depth_scale_requires_initialized or initialized
        ):
            previous_for_jump = (
                float(self.depth_scale)
                if self._depth_scale_bootstrap_finalized
                else None
            )
            depth_obs, dap_dvo_stats = _robust_dap_to_dvo_scale(
                dvo_valid_depth,
                mono_valid,
                min_count=self.min_sparse_points,
                value_min=self.depth_scale_min,
                value_max=self.depth_scale_max,
                valid=np.ones_like(dvo_valid_depth, dtype=bool),
                trim_quantile=self.dap_dvo_scale_trim_quantile,
                mad_max=self.dap_dvo_scale_mad_max,
                previous_scale=previous_for_jump,
                jump_max=self.dap_dvo_scale_ratio_jump_max,
            )
            scale_info.update(dap_dvo_stats)
            if depth_obs is None:
                if not self._depth_scale_bootstrap_finalized:
                    self._stable_depth_observations = 0
                reject_reason = str(
                    dap_dvo_stats.get("dap_to_dvo_reject_reason", "bad_ratios")
                )
                scale_info.update(
                    {
                        "depth_scale": float(self.depth_scale),
                        "dap_to_dvo_scale": float(self.depth_scale),
                        "depth_scale_source": f"dvo_sparse_depth_reused_{reject_reason}",
                        "dap_to_dvo_scale_source": f"dvo_sparse_depth_reused_{reject_reason}",
                        "depth_scale_stable": bool(self._depth_scale_bootstrap_finalized),
                        "dap_to_dvo_scale_stable": bool(self._depth_scale_bootstrap_finalized),
                    }
                )
            else:
                self._depth_scale_observation_count += 1
                self._stable_depth_observations += 1
                if not self._depth_scale_bootstrap_finalized:
                    self._depth_scale_bootstrap_observations.append(float(depth_obs))
                    self.depth_scale = float(
                        np.median(self._depth_scale_bootstrap_observations)
                    )
                    source = "dvo_sparse_depth_bootstrap"
                else:
                    alpha = (
                        self.dap_dvo_scale_ema_alpha
                        if self.dap_dvo_scale_update == "ema_after_bootstrap"
                        else self.depth_scale_ema_alpha
                    )
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    self.depth_scale = (
                        (1.0 - alpha) * float(self.depth_scale)
                        + alpha * float(depth_obs)
                    )
                    source = (
                        "dvo_sparse_depth_ema"
                        if self.dap_dvo_scale_update == "ema_after_bootstrap"
                        else "dvo_sparse_depth"
                    )
                stable = (
                    self._depth_scale_bootstrap_finalized
                    or
                    self._stable_depth_observations
                    >= self.depth_scale_stable_frames
                )
                scale_info.update(
                    {
                        "depth_scale": float(self.depth_scale),
                        "dap_to_dvo_scale": float(self.depth_scale),
                        "depth_scale_observed": float(depth_obs),
                        "dap_to_dvo_scale_observed": float(depth_obs),
                        "depth_scale_source": source
                        if stable
                        else f"{source}_unstable",
                        "dap_to_dvo_scale_source": source
                        if stable
                        else f"{source}_unstable",
                        "depth_scale_stable": bool(stable),
                        "dap_to_dvo_scale_stable": bool(stable),
                    }
                )

        scale_info["scale"] = float(self._pose_translation_scale())
        scale_info["pose_scale"] = float(self._pose_translation_scale())
        if self.pose_scale_policy in {"raw_dvo", "fixed"}:
            scale_info["scale_source"] = self.pose_scale_policy
            scale_info["scale_stable"] = True
        scale_info["depth_scale"] = float(self.depth_scale)
        scale_info["dap_to_dvo_scale"] = float(self.depth_scale)
        scale_info["dap_dvo_scale_update"] = self.dap_dvo_scale_update
        scale_info.setdefault("dap_to_dvo_scale_source", scale_info.get("depth_scale_source", "-"))
        scale_info.setdefault("dap_to_dvo_scale_stable", scale_info.get("depth_scale_stable", False))
        scale_info.update(
            self._ratio_summary(
                mono_valid * float(self.depth_scale),
                dvo_valid_depth,
                min_count=self.min_sparse_points,
                prefix="aligned_dap_over_dvo",
            )
        )
        return scale_info, mono_sampled

    def _extract_sparse_prior(
        self,
        mono_depth: Optional[np.ndarray],
        *,
        frame_idx: Optional[int] = None,
        update_scale: bool = True,
        mono_valid_mask: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        if self._dpvo is None or int(self._dpvo.n) <= 0:
            return None
        latest = self._dpvo_storage_index_for_frame(frame_idx)
        if latest < 0:
            return None
        patches = self._dpvo.patches_[latest].detach().float()
        p = int(patches.shape[-1]) // 2
        uv = patches[:, :2, p, p].detach().cpu().numpy().astype(np.float32)
        disp = patches[:, 2, p, p].detach().cpu().numpy().astype(np.float32)
        res = float(getattr(self._dpvo, "RES", 1))
        uv_erp = uv * res
        dvo_depth = 1.0 / np.maximum(disp, 1e-6)
        valid = np.isfinite(dvo_depth) & (dvo_depth > 1e-6)

        if update_scale and not self._depth_scale_frozen:
            scale_info, mono_sampled = self._update_scale_from_sparse(
                uv_erp, dvo_depth, valid, mono_depth, mono_valid_mask=mono_valid_mask
            )
            if frame_idx is not None:
                self._scale_update_frame_ids.add(int(frame_idx))
            self._last_scale_info = scale_info
        else:
            scale_info = dict(self._last_scale_info)
            mono_sampled = None
            if mono_depth is not None:
                mono = np.asarray(mono_depth, dtype=np.float32)
                if mono.ndim == 3:
                    mono = mono[0]
                mono_sampled = _sample_depth_nearest(mono, uv_erp)

        confidence = np.zeros_like(dvo_depth, dtype=np.float32)
        confidence[valid] = 1.0
        pose_scale = float(self._pose_translation_scale())
        depth_scale = float(self.depth_scale)
        sparse_prior = {
            "uv": uv_erp.astype(np.float32),
            "depth_raw": dvo_depth.astype(np.float32),
            "depth_dvo": (dvo_depth * pose_scale).astype(np.float32),
            "depth_metric": (dvo_depth * pose_scale).astype(np.float32),
            "mono_depth_aligned_to_dvo": (
                (mono_sampled * depth_scale).astype(np.float32)
                if mono_sampled is not None
                else None
            ),
            "mono_depth_sampled": (
                mono_sampled.astype(np.float32) if mono_sampled is not None else None
            ),
            "valid": valid.astype(bool),
            "confidence": confidence,
            "scale": pose_scale,
            "pose_scale": pose_scale,
            "depth_scale": depth_scale,
            "dap_to_dvo_scale": depth_scale,
            "scale_info": dict(scale_info),
            "use_for_mapping": bool(self.use_sparse_prior_for_mapping),
            "source": "online_360dvo",
        }
        self._last_sparse_prior = sparse_prior
        return sparse_prior

    def finalize_depth_scale_bootstrap(self, source: str = "dvo_bootstrap") -> None:
        if not self._depth_scale_bootstrap_observations:
            self._last_scale_info = dict(self._last_scale_info)
            self._last_scale_info.update(
                {
                    "depth_scale": float(self.depth_scale),
                    "dap_to_dvo_scale": float(self.depth_scale),
                    "depth_scale_stable": False,
                    "dap_to_dvo_scale_stable": False,
                    "dap_dvo_scale_update": self.dap_dvo_scale_update,
                }
            )
            return
        self.depth_scale = float(np.median(self._depth_scale_bootstrap_observations))
        self._depth_scale_bootstrap_finalized = True
        self._last_scale_info = dict(self._last_scale_info)
        self._last_scale_info.update(
            {
                "depth_scale": float(self.depth_scale),
                "dap_to_dvo_scale": float(self.depth_scale),
                "depth_scale_source": str(source),
                "dap_to_dvo_scale_source": str(source),
                "depth_scale_stable": True,
                "dap_to_dvo_scale_stable": True,
                "dap_dvo_scale_update": self.dap_dvo_scale_update,
            }
        )

    def freeze_depth_scale(self, source: str = "dvo_bootstrap_frozen") -> None:
        self.finalize_depth_scale_bootstrap(source=source)
        self._depth_scale_frozen = True

    @staticmethod
    def _erp_image_to_uint8(erp_image: torch.Tensor) -> np.ndarray:
        img = erp_image.detach().float().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in {1, 3, 4}:
            img = img[:3].permute(1, 2, 0)
        arr = img.numpy()
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return (np.clip(arr[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def _finite_percentile_range(values: np.ndarray, default=(0.0, 1.0)):
        vals = np.asarray(values, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return tuple(float(v) for v in default)
        lo, hi = np.percentile(vals, [5.0, 95.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            med = float(np.median(vals))
            lo, hi = med - 1.0, med + 1.0
        return float(lo), float(hi)

    def _draw_sparse_points(
        self,
        rgb: np.ndarray,
        uv: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
        *,
        label: str,
        value_range: tuple[float, float] | None = None,
        cmap_name: int | None = None,
    ) -> np.ndarray:
        import cv2

        canvas = rgb.copy()
        h, w = canvas.shape[:2]
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        valid &= np.isfinite(values)
        if value_range is None:
            value_range = self._finite_percentile_range(values[valid])
        lo, hi = value_range
        denom = max(float(hi) - float(lo), 1e-6)
        norm = np.clip((values - float(lo)) / denom, 0.0, 1.0)
        cmap = (
            getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            if cmap_name is None
            else cmap_name
        )
        colors = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cmap)
        radius = int(self.depth_scale_vis_point_radius)
        for idx in np.flatnonzero(valid):
            x = int(np.round(uv[idx, 0])) % w
            y = int(np.clip(np.round(uv[idx, 1]), 0, h - 1))
            color = tuple(int(c) for c in colors[idx, 0].tolist())
            cv2.circle(canvas, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        text = f"{label} [{float(lo):.3g},{float(hi):.3g}] n={int(valid.sum())}"
        cv2.putText(
            canvas,
            text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _save_depth_scale_debug(
        self,
        frame_idx: int,
        erp_image: torch.Tensor,
        sparse_prior: Optional[dict],
        scale_info: dict,
    ) -> None:
        if not self.debug_visualize_depth_scale:
            return
        if self._debug_pose_dir is None or erp_image is None or sparse_prior is None:
            return
        if int(frame_idx) % int(self.depth_scale_vis_interval) != 0:
            return
        try:
            import cv2

            out_dir = self._debug_pose_dir / "depth_scale_vis"
            out_dir.mkdir(parents=True, exist_ok=True)
            rgb = self._erp_image_to_uint8(erp_image)
            uv = np.asarray(sparse_prior.get("uv"), dtype=np.float32).reshape(-1, 2)
            dvo_depth = np.asarray(
                sparse_prior.get("depth_dvo", sparse_prior.get("depth_raw")),
                dtype=np.float32,
            ).reshape(-1)
            mono_sampled = sparse_prior.get("mono_depth_sampled")
            mono_aligned = sparse_prior.get("mono_depth_aligned_to_dvo")
            valid = np.asarray(sparse_prior.get("valid"), dtype=bool).reshape(-1)
            if mono_sampled is None or mono_aligned is None:
                mono_sampled = np.full_like(dvo_depth, np.nan, dtype=np.float32)
                mono_aligned = np.full_like(dvo_depth, np.nan, dtype=np.float32)
            else:
                mono_sampled = np.asarray(mono_sampled, dtype=np.float32).reshape(-1)
                mono_aligned = np.asarray(mono_aligned, dtype=np.float32).reshape(-1)
            valid_scale = (
                valid
                & np.isfinite(dvo_depth)
                & np.isfinite(mono_sampled)
                & (dvo_depth > 1e-6)
                & (mono_sampled > 1e-6)
            )
            depth_range = self._finite_percentile_range(
                np.concatenate([dvo_depth[valid_scale], mono_aligned[valid_scale]])
                if valid_scale.any()
                else dvo_depth[valid]
            )
            aligned_over_dvo = mono_aligned / np.maximum(dvo_depth, 1e-6)
            log2_ratio = np.log2(np.clip(aligned_over_dvo, 1e-3, 1e3))
            ratio_range = (-1.0, 1.0)
            panel_dvo = self._draw_sparse_points(
                rgb,
                uv,
                dvo_depth,
                valid_scale,
                label="DPVO sparse depth",
                value_range=depth_range,
            )
            panel_dap = self._draw_sparse_points(
                rgb,
                uv,
                mono_aligned,
                valid_scale,
                label="DAP depth x scale",
                value_range=depth_range,
            )
            panel_ratio = self._draw_sparse_points(
                rgb,
                uv,
                log2_ratio,
                valid_scale,
                label="log2((DAP x scale)/DPVO)",
                value_range=ratio_range,
                cmap_name=cv2.COLORMAP_COOLWARM
                if hasattr(cv2, "COLORMAP_COOLWARM")
                else cv2.COLORMAP_JET,
            )
            canvas = np.concatenate([rgb, panel_dvo, panel_dap, panel_ratio], axis=1)
            canvas = np.ascontiguousarray(canvas[:, :, ::-1])
            aligned_med = scale_info.get("aligned_dap_over_dvo_median", float("nan"))
            raw_med = scale_info.get(
                "depth_scale_ratio_dvo_over_dap_filtered_median",
                scale_info.get("depth_scale_ratio_dvo_over_dap_median", float("nan")),
            )
            label = (
                f"frame={int(frame_idx)} depth_scale={float(scale_info.get('depth_scale', self.depth_scale)):.5f} "
                f"obs={float(scale_info.get('depth_scale_observed', raw_med)):.5f} "
                f"aligned_med={float(aligned_med):.3f} "
                f"stable={bool(scale_info.get('depth_scale_stable', False))} "
                f"source={scale_info.get('depth_scale_source', '-')}"
            )
            cv2.putText(
                canvas,
                label,
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if canvas.shape[1] > 3200:
                scale = 3200 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(out_dir / f"frame_{int(frame_idx):04d}.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            json_payload = {
                "frame_idx": int(frame_idx),
                "depth_scale": float(scale_info.get("depth_scale", self.depth_scale)),
                "depth_scale_observed": float(
                    scale_info.get("depth_scale_observed", raw_med)
                )
                if np.isfinite(float(scale_info.get("depth_scale_observed", raw_med)))
                else None,
                "depth_scale_source": str(scale_info.get("depth_scale_source", "-")),
                "depth_scale_stable": bool(scale_info.get("depth_scale_stable", False)),
                "sparse_valid": int(scale_info.get("sparse_valid", int(valid_scale.sum()))),
                "aligned_dap_over_dvo_median": float(aligned_med)
                if np.isfinite(float(aligned_med))
                else None,
                "aligned_dap_over_dvo_p10": scale_info.get("aligned_dap_over_dvo_p10"),
                "aligned_dap_over_dvo_p90": scale_info.get("aligned_dap_over_dvo_p90"),
                "depth_scale_ratio_dvo_over_dap_median": scale_info.get(
                    "depth_scale_ratio_dvo_over_dap_median"
                ),
                "depth_scale_ratio_dvo_over_dap_p10": scale_info.get(
                    "depth_scale_ratio_dvo_over_dap_p10"
                ),
                "depth_scale_ratio_dvo_over_dap_p90": scale_info.get(
                    "depth_scale_ratio_dvo_over_dap_p90"
                ),
            }
            with open(out_dir / "depth_scale_debug.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(json_payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            print(f"[Online360DVO] depth scale visualization failed: {exc}", flush=True)

    def _convert_and_scale_pose(
        self,
        c2w_raw: np.ndarray,
        previous_w2c: Optional[np.ndarray],
    ) -> np.ndarray:
        if self._reference_c2w_raw is None:
            self._reference_c2w_raw = c2w_raw.copy()
        c2w_rel = np.linalg.inv(self._reference_c2w_raw) @ c2w_raw
        c2w = self._A @ c2w_rel @ self._A_inv
        c2w[:3, 3] *= float(self._pose_translation_scale())
        self._last_converted_c2w = c2w.astype(np.float64).copy()
        w2c = np.linalg.inv(c2w).astype(np.float32)

        stable = bool(self._last_scale_info.get("scale_stable", False))
        if self.enable_pose_clamp and previous_w2c is not None and not stable:
            dt = float(np.linalg.norm(w2c[:3, 3] - previous_w2c[:3, 3]))
            if dt > self.unstable_max_dt_m > 0:
                ratio = self.unstable_max_dt_m / max(dt, 1e-6)
                w2c[:3, 3] = previous_w2c[:3, 3] + ratio * (
                    w2c[:3, 3] - previous_w2c[:3, 3]
                )
                self._last_scale_info["translation_clamped"] = True
                self._last_scale_info["translation_clamp_reason"] = "scale_unstable"
                self._last_scale_info["translation_unclamped_dt_m"] = dt
        if self.enable_pose_clamp and previous_w2c is not None and self.max_dt_m > 0:
            dt = float(np.linalg.norm(w2c[:3, 3] - previous_w2c[:3, 3]))
            if dt > self.max_dt_m:
                ratio = self.max_dt_m / max(dt, 1e-6)
                w2c[:3, 3] = previous_w2c[:3, 3] + ratio * (
                    w2c[:3, 3] - previous_w2c[:3, 3]
                )
                self._last_scale_info["translation_clamped"] = True
                self._last_scale_info["translation_clamp_reason"] = "max_dt"
                self._last_scale_info["translation_unclamped_dt_m"] = dt
        return w2c

    def _append_tum_pose(self, filename: str, frame_idx: int, c2w: np.ndarray) -> None:
        if self._debug_pose_dir is None:
            return
        self._debug_pose_dir.mkdir(parents=True, exist_ok=True)
        T = np.asarray(c2w, dtype=np.float64)
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        row = (
            f"{float(frame_idx):.6f} "
            f"{T[0, 3]:.9f} {T[1, 3]:.9f} {T[2, 3]:.9f} "
            f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
        )
        with open(self._debug_pose_dir / filename, "a", encoding="utf-8") as f:
            f.write(row)

    def _append_debug_poses(
        self,
        frame_idx: int,
        raw_c2w: np.ndarray,
        converted_c2w: Optional[np.ndarray],
    ) -> None:
        if self._debug_pose_dir is None or frame_idx in self._debug_pose_frames:
            return
        self._append_tum_pose("raw_dpvo_c2w.tum", frame_idx, raw_c2w)
        if converted_c2w is not None:
            self._append_tum_pose("converted_dvo_scale_c2w.tum", frame_idx, converted_c2w)
        self._debug_pose_frames.add(int(frame_idx))

    def _fallback_prior(
        self,
        frame_idx: int,
        previous_w2c: Optional[np.ndarray],
        reason: str,
    ) -> PosePrior360DVO:
        if previous_w2c is not None:
            w2c = previous_w2c.astype(np.float32).copy()
        else:
            w2c = np.eye(4, dtype=np.float32)
        info = {
            "success": True,
            "source": "online_360dvo_warmup",
            "initialized": False,
            "confidence": float(self.warmup_confidence),
            "scale": float(self._pose_translation_scale()),
            "pose_scale": float(self._pose_translation_scale()),
            "pose_scale_policy": self.pose_scale_policy,
            "scale_source": reason,
            "scale_stable": False,
            "depth_scale": float(self.depth_scale),
            "dap_to_dvo_scale": float(self.depth_scale),
            "depth_scale_policy": self.depth_scale_policy,
            "depth_scale_source": reason,
            "dap_to_dvo_scale_source": reason,
            "depth_scale_stable": False,
            "dap_to_dvo_scale_stable": False,
            "dap_dvo_scale_update": self.dap_dvo_scale_update,
            "t_norm": 0.0,
            "rel_rot_deg": 0.0,
            "trajectory_index": int(frame_idx),
            "inlier_ratio": 0.0,
            "valid_depth_ratio": 0.0,
            "matches": 0,
            "inliers": 0,
            "mean_ang_deg": 999.0,
            "median_ang_deg": 999.0,
        }
        return PosePrior360DVO(w2c=w2c, info=info)

    def get_prior(
        self,
        frame_idx: int,
        mono_depth: Optional[np.ndarray] = None,
        mono_valid_mask: Optional[np.ndarray] = None,
        erp_image: Optional[torch.Tensor] = None,
        previous_w2c: Optional[np.ndarray] = None,
        **_: object,
    ) -> Optional[PosePrior360DVO]:
        self._ensure_loaded()
        if erp_image is None:
            raise ValueError("Online 360DVO mode requires erp_image for each frame.")
        image = self._image_to_dpvo(erp_image)
        _, height, width = image.shape
        intrinsics = self._intrinsics_for_erp(height, width, self.device)
        if self._dpvo is None:
            self._dpvo = self._dpvo_class(
                self._dpvo_cfg,
                self.network_path,
                ht=height,
                wd=width,
                viz=False,
            )
        frame_key = int(frame_idx)
        is_new_frame = frame_key not in self._processed_frame_ids
        if is_new_frame:
            with torch.no_grad():
                self._dpvo(frame_key, image, intrinsics)
            self._processed_frame_ids.add(frame_key)
        sparse_prior = self._extract_sparse_prior(
            mono_depth,
            frame_idx=frame_key,
            update_scale=(
                frame_key not in self._scale_update_frame_ids
                and not self._depth_scale_frozen
            ),
            mono_valid_mask=mono_valid_mask,
        )
        self._save_depth_scale_debug(
            frame_idx,
            erp_image,
            sparse_prior,
            self._last_scale_info,
        )
        c2w_raw = self._current_raw_c2w(frame_idx)
        if c2w_raw is None:
            return self._fallback_prior(frame_idx, previous_w2c, "dpvo_no_pose")
        w2c = self._convert_and_scale_pose(c2w_raw, previous_w2c)
        self._append_debug_poses(frame_idx, c2w_raw, self._last_converted_c2w)

        if previous_w2c is not None:
            rel = w2c.astype(np.float64) @ np.linalg.inv(previous_w2c.astype(np.float64))
            t_norm = float(np.linalg.norm(rel[:3, 3]))
            trace = np.clip((np.trace(rel[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
            rel_rot_deg = float(np.degrees(np.arccos(trace)))
        else:
            t_norm = 0.0
            rel_rot_deg = 0.0

        initialized = bool(getattr(self._dpvo, "is_initialized", False))
        stable = bool(self._last_scale_info.get("scale_stable", False))
        depth_stable = bool(self._last_scale_info.get("depth_scale_stable", False))
        confidence = (
            self.stable_confidence
            if initialized and stable
            else self.unstable_confidence
            if initialized
            else self.warmup_confidence
        )
        info = {
            "success": True,
            "source": "online_360dvo",
            "initialized": initialized,
            "confidence": float(confidence),
            "t_norm": t_norm,
            "rel_rot_deg": rel_rot_deg,
            "trajectory_index": int(frame_idx),
            "inlier_ratio": float(confidence),
            "valid_depth_ratio": 1.0 if initialized or depth_stable else 0.0,
            "matches": int(getattr(self._dpvo, "M", 0)),
            "inliers": int(self._last_scale_info.get("sparse_valid", 0)),
            "mean_ang_deg": 0.0,
            "median_ang_deg": 0.0,
            **self._last_scale_info,
        }
        self._last_w2c = w2c.copy()
        return PosePrior360DVO(w2c=w2c, info=info)

    def get_sparse_prior(self, frame_idx: int, mono_depth: Optional[np.ndarray] = None):
        return self._last_sparse_prior
