"""PanoDROID front-end plus panoramic Gaussian backend SLAM runner."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from backend.pano_gs import NeuralScaffoldPanoMap, PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper
from frontend.pano_droid.adapter import build_frontend_from_config
from frontend.pano_droid.dataset import discover_erp_images, discover_ob3d_images, load_erp_image, load_ob3d_camera_c2w
from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from frontend.pano_droid.spherical_ba import se3_exp, skew
from frontend.pano_vggt.grid_utils import feature_uv_to_image_uv
from geometry.pose import relative_c2w
from geometry.trajectory_metrics import c2w_trajectory_metrics
from mapping.gaussian_initializer import GaussianInitializer, GaussianSeedBatch


def _deep_merge_config(base: dict, override: dict) -> dict:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge_config(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    resolved_base = Path(base_path)
    if not resolved_base.is_absolute():
        resolved_base = config_path.parent / resolved_base
    return _deep_merge_config(load_config(resolved_base), config)


def _se3_log(T: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithm matching ``se3_exp``'s ``[tx, ty, tz, rx, ry, rz]`` convention."""

    mat = T.detach().float()
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {tuple(mat.shape)}")
    R = mat[:3, :3]
    t = mat[:3, 3]
    cos_theta = ((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    if float(theta.detach().cpu()) < 1.0e-5:
        omega = 0.5 * vee
    else:
        omega = theta / (2.0 * torch.sin(theta).clamp_min(1.0e-8)) * vee
    K = skew(omega)
    theta_w = torch.linalg.norm(omega).clamp_min(1.0e-8)
    theta2 = theta_w * theta_w
    eye = torch.eye(3, device=mat.device, dtype=mat.dtype)
    if float(theta_w.detach().cpu()) < 1.0e-5:
        V = eye + 0.5 * K + (1.0 / 6.0) * (K @ K)
    else:
        V = eye + ((1.0 - torch.cos(theta_w)) / theta2) * K
        V = V + ((theta_w - torch.sin(theta_w)) / (theta2 * theta_w)) * (K @ K)
    rho = torch.linalg.solve(V, t)
    return torch.cat([rho, omega], dim=0)


def _se3_blend_pose(source_c2w: torch.Tensor, target_c2w: torch.Tensor, alpha: float) -> torch.Tensor:
    """Move ``source_c2w`` toward ``target_c2w`` by ``alpha`` on SE(3)."""

    source = source_c2w.detach().float()
    target = target_c2w.detach().float()
    a = float(alpha)
    if a >= 1.0:
        return target.clone()
    if a <= 0.0:
        return source.clone()
    delta = target @ torch.linalg.inv(source)
    xi = _se3_log(delta).to(device=source.device, dtype=source.dtype)
    return (se3_exp(a * xi) @ source).detach()


def _load_panocity_gt_poses(root: str) -> dict[str, np.ndarray]:
    root_path = Path(root)
    block_dir = root_path.parent if root_path.name == "pano_images" else root_path
    image_dir = block_dir / "pano_images"
    if not image_dir.is_dir():
        return {}
    pose_files = sorted(block_dir.glob("*poses*.json"))
    if not pose_files:
        return {}
    pose_path = next((p for p in pose_files if ".1." not in p.name), pose_files[0])
    with open(pose_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", payload)
    if not isinstance(frames, list):
        return {}
    out: dict[str, np.ndarray] = {}
    for frame in frames:
        name = frame.get("name")
        mat = frame.get("transformation_matrix")
        if name is None or mat is None:
            continue
        c2w = np.asarray(mat, dtype=np.float32)
        if c2w.shape != (4, 4):
            continue
        image_path = image_dir / str(name)
        out[str(image_path.resolve())] = c2w
        out[str(name)] = c2w
    return out


def iter_sequence_frames(config: dict) -> Iterable[PanoFrame]:
    ds_cfg = config.get("Dataset", {})
    if ds_cfg.get("synthetic", False):
        from frontend.pano_droid.dataset import SyntheticPanoPairDataset

        ds = SyntheticPanoPairDataset(
            length=int(ds_cfg.get("synthetic_length", 4)),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
        )
        yielded_first = False
        for idx in range(len(ds)):
            sample = ds[idx]
            if not yielded_first:
                yielded_first = True
                yield PanoFrame(image=sample["image0"], timestamp=float(idx), frame_id=idx)
            yield PanoFrame(image=sample["image1"], timestamp=float(idx + 1), frame_id=idx + 1)
        return

    root = ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.dataset_path is required unless Dataset.synthetic=true.")
    dataset_type = str(ds_cfg.get("type", ds_cfg.get("dataset_type", "")) or "").lower()
    if dataset_type in {"ob3d", "ob3d_pfgs360", "pfgs360_ob3d"}:
        files = discover_ob3d_images(
            root,
            scene=ds_cfg.get("scene", ds_cfg.get("sequence")),
            split=str(ds_cfg.get("split", "Egocentric")),
        )
    else:
        files = discover_erp_images(root, sequence=ds_cfg.get("sequence"))
    begin = int(ds_cfg.get("begin", 0))
    end = ds_cfg.get("end")
    files = files[begin:end]
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    resize = (int(h), int(w)) if h is not None and w is not None else None
    gt_poses = {} if dataset_type in {"ob3d", "ob3d_pfgs360", "pfgs360_ob3d"} else _load_panocity_gt_poses(root)
    for local_idx, path in enumerate(files):
        frame_id = begin + local_idx
        gt = gt_poses.get(str(Path(path).resolve()))
        if gt is None:
            gt = gt_poses.get(Path(path).name)
        if gt is None and dataset_type in {"ob3d", "ob3d_pfgs360", "pfgs360_ob3d"}:
            gt = load_ob3d_camera_c2w(path)
        meta = {"path": path}
        if gt is not None:
            meta["gt_c2w"] = torch.from_numpy(gt).float()
        yield PanoFrame(
            image=load_erp_image(path, resize=resize),
            timestamp=float(frame_id),
            frame_id=frame_id,
            meta=meta,
        )


def _scalar_to_rgb(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = valid & np.isfinite(values)
    if not bool(valid.any()):
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(values[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0.0
    return (255.0 * rgb).astype(np.uint8)


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().cpu().float().clamp(0.0, 1.0)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected source image as 3xHxW, got {tuple(img.shape)}")
    arr = (255.0 * img.permute(1, 2, 0).numpy()).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _cubemap_faces_to_pil(faces: torch.Tensor) -> Image.Image:
    tensor = faces.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim != 4 or tensor.shape[0] != 6 or tensor.shape[1] != 3:
        raise ValueError(f"Expected cubemap faces as 6x3xSxS, got {tuple(tensor.shape)}")
    tiles = [_image_tensor_to_pil(tensor[i]) for i in range(6)]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles) + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    labels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    for label, tile in zip(labels, tiles):
        canvas.paste(tile, (x, 24))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        x += tile.width
    return canvas


def _scalar_tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().cpu().float()
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected scalar image as HxW or 1xHxW, got {tuple(tensor.shape)}")
    valid = torch.isfinite(tensor)
    return Image.fromarray(_scalar_to_rgb(tensor.numpy(), valid.numpy()), mode="RGB")


def _mask_tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().cpu().bool()
    while tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected mask image as HxW, 1xHxW, or Bx1xHxW, got {tuple(value.shape)}")
    arr = tensor.numpy().astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def _inverse_depth_to_pil(inverse_depth: torch.Tensor) -> Image.Image:
    inv = inverse_depth.detach().cpu().float()
    if inv.ndim == 3:
        inv = inv[0]
    if inv.ndim != 2:
        raise ValueError(f"Expected inverse depth as HxW or 1xHxW, got {tuple(inv.shape)}")
    valid = torch.isfinite(inv) & (inv > 1e-6)
    depth = torch.zeros_like(inv)
    depth[valid] = 1.0 / inv[valid].clamp_min(1e-6)
    return Image.fromarray(_scalar_to_rgb(depth.numpy(), valid.numpy()), mode="RGB")


def _resize_to_max_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.size[0] <= int(max_width):
        return image
    scale = float(max_width) / float(image.size[0])
    return image.resize((int(max_width), max(1, int(image.size[1] * scale))), Image.BILINEAR)


def _pose_xyz_from_meta(frame: PanoFrame) -> np.ndarray | None:
    meta = frame.meta or {}
    gt = meta.get("gt_c2w")
    if gt is None:
        return None
    gt_t = torch.as_tensor(gt).detach().cpu().float()
    if gt_t.shape != (4, 4):
        return None
    return gt_t[:3, 3].numpy()


def _align_xyz_umeyama_sim3(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, bool]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if int(valid.sum()) < 3:
        return pred, False
    src = pred[valid]
    tgt = gt[valid]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    cov = src_c.T @ tgt_c / max(1, src.shape[0])
    u, s, vh = np.linalg.svd(cov)
    rot = vh.T @ u.T
    if np.linalg.det(rot) < 0:
        vh = vh.copy()
        vh[-1] *= -1.0
        rot = vh.T @ u.T
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s) / max(var_src, 1e-8))
    trans = tgt_mean - scale * (rot @ src_mean)
    return scale * (pred @ rot.T) + trans, True


def _compute_ape_translation(
    pred: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], bool]:
    """Return Sim(3)-aligned predictions and APE translation per frame."""

    aligned, sim3_aligned = _align_xyz_umeyama_sim3(pred, gt)
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.isfinite(aligned).all(axis=1) & np.isfinite(gt).all(axis=1)
    ape = np.full((len(aligned),), np.nan, dtype=np.float32)
    ape[valid] = np.linalg.norm(aligned[valid] - gt[valid], axis=1)
    finite = np.isfinite(ape)
    metrics: dict[str, float] = {}
    if finite.any():
        vals = ape[finite].astype(np.float64)
        metrics = {
            "rmse": float(np.sqrt(np.mean(vals * vals))),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
        }
    return aligned, ape, metrics, sim3_aligned


def _align_xyz_for_plot(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    aligned, _ = _align_xyz_umeyama_sim3(pred, gt)
    return aligned


def _xyz_to_y_up_plot(xyz: np.ndarray) -> np.ndarray:
    """Map project coordinates to Matplotlib coordinates with original Y vertical."""

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        return xyz.reshape(-1, 3)
    return np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return 0.0
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _summarize_dense_ba_stats(frontend) -> dict:
    """Aggregate internal PanoVGGT-M3 dense BA diagnostics without changing public outputs."""

    history = list(getattr(frontend, "dense_ba_stats_history", []) or [])
    last = getattr(frontend, "last_dense_ba_stats", None)
    enabled = bool(getattr(last, "enabled", False)) if last is not None else bool(history)
    if not history:
        return {
            "enabled": enabled,
            "shadow_mode": bool(getattr(last, "shadow_mode", True)) if last is not None else True,
            "chunks": 0,
            "successes": 0,
            "fallbacks": 0,
            "success_ratio": 0.0,
            "used_refined": 0,
            "used_refined_ratio": 0.0,
            "fallback_reasons": {},
            "mean_angular_residual_deg": 0.0,
            "mean_initial_angular_residual_deg": 0.0,
            "mean_valid_factor_ratio": 0.0,
            "mean_pose_update": 0.0,
            "max_pose_update": 0.0,
            "mean_pose_rot_update_deg": 0.0,
            "mean_depth_update": 0.0,
            "max_depth_update": 0.0,
            "pose_only_solver_ratio": 0.0,
            "mean_used_factors": 0.0,
            "mean_pose_variables": 0.0,
            "mean_depth_variables": 0.0,
            "mean_pose_solve_sec": 0.0,
            "time_budget_stops": 0,
        }

    successes = [item for item in history if bool(getattr(item, "success", False))]
    fallbacks = [item for item in history if not bool(getattr(item, "success", False))]
    reasons: dict[str, int] = {}
    for item in fallbacks:
        reason = str(getattr(item, "fallback_reason", None) or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    pose_update = [float(getattr(item, "pose_update_norm", {}).get("mean", 0.0)) for item in history]
    pose_update_max = [float(getattr(item, "pose_update_norm", {}).get("max", 0.0)) for item in history]
    pose_rot_update = [float(getattr(item, "pose_update_norm", {}).get("rot_max_deg", 0.0)) for item in history]
    depth_update = [float(getattr(item, "depth_update_norm", {}).get("mean", 0.0)) for item in history]
    depth_update_max = [float(getattr(item, "depth_update_norm", {}).get("max", 0.0)) for item in history]
    pose_only_count = sum(int(str(getattr(item, "solver_mode", "")) == "pose_only_factor_graph") for item in history)
    used_factors = [float(getattr(item, "used_factors", 0)) for item in history]
    pose_variables = [float(getattr(item, "num_pose_variables", 0)) for item in history]
    depth_variables = [float(getattr(item, "num_depth_variables", 0)) for item in history]
    pose_solve_sec = [float(getattr(item, "pose_solve_sec", 0.0)) for item in history]
    time_budget_stops = sum(int(bool(getattr(item, "stopped_by_time_budget", False))) for item in history)
    chunk_count = len(history)
    used_refined = sum(int(bool(getattr(item, "used_refined", False))) for item in history)
    return {
        "enabled": True,
        "shadow_mode": bool(getattr(history[-1], "shadow_mode", True)),
        "chunks": int(chunk_count),
        "successes": int(len(successes)),
        "fallbacks": int(len(fallbacks)),
        "success_ratio": float(len(successes) / chunk_count) if chunk_count else 0.0,
        "used_refined": int(used_refined),
        "used_refined_ratio": float(used_refined / chunk_count) if chunk_count else 0.0,
        "fallback_reasons": reasons,
        "mean_angular_residual_deg": _finite_mean([float(getattr(item, "mean_residual_deg", 0.0)) for item in history]),
        "mean_initial_angular_residual_deg": _finite_mean([float(getattr(item, "initial_mean_residual_deg", 0.0)) for item in history]),
        "mean_valid_factor_ratio": _finite_mean([float(getattr(item, "valid_factor_ratio", 0.0)) for item in history]),
        "mean_pose_update": _finite_mean(pose_update),
        "max_pose_update": float(max(pose_update_max)) if pose_update_max else 0.0,
        "mean_pose_rot_update_deg": _finite_mean(pose_rot_update),
        "mean_depth_update": _finite_mean(depth_update),
        "max_depth_update": float(max(depth_update_max)) if depth_update_max else 0.0,
        "pose_only_solver_ratio": float(pose_only_count / chunk_count) if chunk_count else 0.0,
        "mean_used_factors": _finite_mean(used_factors),
        "mean_pose_variables": _finite_mean(pose_variables),
        "mean_depth_variables": _finite_mean(depth_variables),
        "mean_pose_solve_sec": _finite_mean(pose_solve_sec),
        "time_budget_stops": int(time_budget_stops),
    }


def _flatten_dense_ba_summary(summary: dict) -> dict[str, float | int | bool]:
    return {
        f"dense_ba_{key}": value
        for key, value in summary.items()
        if isinstance(value, (bool, int, float))
    }


_COMPACT_SLAM_WANDB_KEYS = frozenset(
    {
        "slam/frame_id",
        "slam/anchors",
        "slam/keyframes",
        "slam/status",
        "backend/loss",
        "backend/trajectory_vs_gt",
        "backend/render_vs_gt_panorama",
        "backend/kf_opt_loss",
        "backend/kf_opt_psnr",
        "backend/kf_render_opt",
        "backend/kf_depth_opt",
        "backend/post_opt_window_frames",
        "backend/post_opt_window_depths",
        "backend/post_opt_window_id",
        "backend/post_opt_frame_count",
        "backend/post_opt_mean_loss",
        "backend/post_opt_mean_psnr",
        "backend/sky_pruned",
        "m3/chunk",
        "m3/ba_success",
        "m3/valid_factor_ratio",
        "m3/residual_drop_deg",
        "m3/match_lines",
        "m3/sky_mask",
        "mapping/depth_insertion",
        "mapping/keyframe_frame_id",
        "mapping/keyframe_inserted_gaussians",
        "mapping/new_gaussians_inserted",
    }
)
_COMPACT_SLAM_WANDB_PREFIXES = ("mapping/depth_insertion_",)


class SlamRuntimeLogger:
    """Runtime W&B and local visualization logger for online SLAM runs."""

    def __init__(self, config: dict, output_dir: Path) -> None:
        wb_cfg = config.get("WeightsAndBiases", {})
        vis_cfg = config.get("Visualization", {})
        self.output_dir = output_dir
        self.log_every = max(1, int(wb_cfg.get("log_every", vis_cfg.get("log_every", 10))))
        self.m3_log_every = max(1, int(vis_cfg.get("m3_log_every", self.log_every)))
        self.m3_max_matches = max(1, int(vis_cfg.get("m3_max_matches", 80)))
        self.save_kf_opt = bool(vis_cfg.get("save_kf_opt", True))
        self.kf_opt_log_every = max(1, int(vis_cfg.get("kf_opt_log_every", 1)))
        self.kf_opt_max_width = max(320, int(vis_cfg.get("kf_opt_max_width", 1920)))
        self.post_opt_all_frames = bool(vis_cfg.get("post_opt_all_frames", False))
        self.post_opt_log_depth = bool(vis_cfg.get("post_opt_log_depth", False))
        self.log_keyframes = bool(wb_cfg.get("log_keyframes", True))
        self.results_cfg = config.get("Results", {})
        self.runtime_log_preset = str(
            wb_cfg.get("runtime_log_preset", wb_cfg.get("log_preset", "")) or ""
        ).strip().lower()
        self.compact_slam_wandb = self.runtime_log_preset in {"compact", "compact_slam", "minimal", "minimal_slam"}
        self.log_image_paths = bool(wb_cfg.get("log_image_paths", not self.compact_slam_wandb))
        self.log_keyframe_inserted_gaussians = bool(
            wb_cfg.get("log_keyframe_inserted_gaussians", self.compact_slam_wandb)
        )
        mode = str(wb_cfg.get("mode") or "online")
        self.wandb_enabled = bool(wb_cfg.get("enabled", False)) and mode != "disabled"
        self.save_local = bool(vis_cfg.get("save_local", self.wandb_enabled))
        self.visualization_dir = output_dir / "visualizations"
        if self.save_local:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.run = None
        self._wandb = None
        self.wandb_mode = mode
        self.wandb_init_error: str | None = None
        self._step = 0
        self._last_m3_chunk_logged: int | None = None
        self._kf_opt_count = 0
        self._frontend_pose_history: list[tuple[int, np.ndarray]] = []
        self._backend_pose_history: list[tuple[int, np.ndarray]] = []
        self._gt_pose_history: list[tuple[int, np.ndarray]] = []

        if self.wandb_enabled:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "WeightsAndBiases.enabled=true requires the 'wandb' package. "
                    "Install it or set WeightsAndBiases.mode=disabled."
                ) from exc
            self._wandb = wandb
            init_kwargs = {
                "project": str(wb_cfg.get("project") or "360Droid-splat"),
                "entity": wb_cfg.get("entity") or None,
                "name": wb_cfg.get("run_name") or None,
                "dir": str(output_dir),
                "config": config,
                "tags": wb_cfg.get("tags") or None,
                "group": wb_cfg.get("group") or None,
            }
            try:
                self.run = wandb.init(mode=mode, **init_kwargs)
            except Exception as exc:
                if mode != "online":
                    raise
                self.wandb_init_error = repr(exc)
                self.wandb_mode = "offline"
                self.run = wandb.init(mode="offline", **init_kwargs)

    @property
    def run_url(self) -> str | None:
        if self.run is None:
            return None
        url = getattr(self.run, "url", None)
        return str(url) if url else None

    def _wandb_step(self, step: int | None = None) -> int:
        return max(1, int(self._step if step is None else step))

    def _should_log_wandb_key(self, key: str) -> bool:
        if not self.compact_slam_wandb:
            return True
        if key.endswith("_png"):
            return False
        if key in _COMPACT_SLAM_WANDB_KEYS:
            return True
        return any(key.startswith(prefix) for prefix in _COMPACT_SLAM_WANDB_PREFIXES)

    def _filter_wandb_payload(self, payload: dict) -> dict:
        if not self.compact_slam_wandb:
            return payload
        return {key: value for key, value in payload.items() if self._should_log_wandb_key(str(key))}

    def _log_wandb_payload(self, payload: dict, *, step: int | None = None) -> None:
        if self.run is None:
            return
        filtered = self._filter_wandb_payload(payload)
        if filtered:
            self.run.log(filtered, step=self._wandb_step(step))

    def observe(
        self,
        output: FrontendOutput,
        source_frame: PanoFrame,
        *,
        anchor_count: int,
        keyframe_count: int,
        backend_loss: float | None,
        backend_pose_c2w: torch.Tensor | None = None,
        backend_render_pkg: dict | None = None,
        m3_debug: dict | None = None,
    ) -> None:
        self._step += 1
        pose = output.pose_c2w.detach().cpu().float()
        if pose.shape == (4, 4):
            self._frontend_pose_history.append((int(output.frame_id), pose[:3, 3].numpy()))
        backend_pose = backend_pose_c2w.detach().cpu().float() if backend_pose_c2w is not None else None
        if backend_pose is not None and backend_pose.shape == (4, 4):
            self._backend_pose_history.append((int(output.frame_id), backend_pose[:3, 3].numpy()))
        gt_xyz = _pose_xyz_from_meta(source_frame)
        if gt_xyz is not None:
            self._gt_pose_history.append((int(output.frame_id), gt_xyz))

        payload: dict[str, float | int | str] = {
            "slam/frame_id": int(output.frame_id),
            "slam/keyframe": int(bool(output.is_keyframe)),
            "slam/keyframe_score": float(output.keyframe_score),
            "slam/pose_confidence": float(output.pose_confidence),
            "slam/anchors": int(anchor_count),
            "slam/keyframes": int(keyframe_count),
            "slam/status": str(output.tracking_status),
        }
        if output.ba_residual is not None:
            payload["slam/ba_residual"] = float(output.ba_residual)
        if backend_loss is not None:
            payload["backend/loss"] = float(backend_loss)
        if output.valid_world_points_mask is not None:
            valid_world = output.valid_world_points_mask.detach().cpu().bool()
            payload["frontend/valid_world_points"] = int(valid_world.sum().item())
        if output.world_points is not None:
            payload["frontend/world_points_finite"] = int(torch.isfinite(output.world_points).all(dim=-1).sum().item())
        if isinstance(m3_debug, dict):
            alignment = m3_debug.get("alignment")
            if isinstance(alignment, dict):
                for source, target in (
                    ("overlap_points", "m3/alignment_overlap_points"),
                    ("history_points", "m3/alignment_history_points"),
                    ("overlap_alignment_points", "m3/overlap_alignment_points"),
                    ("history_alignment_points", "m3/history_alignment_points"),
                    ("scale", "m3/alignment_scale"),
                    ("alignment_scale", "m3/alignment_scale"),
                    ("residual", "m3/alignment_residual"),
                    ("alignment_rmse", "m3/alignment_rmse"),
                    ("inlier_ratio", "m3/alignment_inlier_ratio"),
                ):
                    value = alignment.get(source)
                    if value is not None:
                        payload[target] = float(value)

        self._log_wandb_payload(payload, step=self._step)
        self._observe_m3_debug(m3_debug)

        if not self._should_visualize(output):
            return
        image_payload = {}
        depth_path = None
        if output.inverse_depth is not None:
            depth_path = self._save_depth_panel(output, source_frame)
            if self.run is not None and self._wandb is not None:
                image_payload["frontend/depth"] = self._wandb.Image(str(depth_path))
                image_payload["slam/depth"] = self._wandb.Image(str(depth_path))
        frontend_traj_path = self._save_trajectory_panel(
            output,
            kind="frontend",
            pred_history=self._frontend_pose_history,
        )
        backend_traj_path = self._save_trajectory_panel(
            output,
            kind="backend",
            pred_history=self._backend_pose_history,
        )
        backend_rgb_path = None
        backend_depth_path = None
        if backend_render_pkg is not None:
            backend_rgb_path = self._save_backend_render_panel(output, source_frame, backend_render_pkg)
            backend_depth_path = self._save_backend_depth_panel(output, backend_render_pkg)
        if self.run is not None and self._wandb is not None:
            image_payload["frontend/trajectory_vs_gt"] = self._wandb.Image(str(frontend_traj_path))
            image_payload["backend/trajectory_vs_gt"] = self._wandb.Image(str(backend_traj_path))
            image_payload["slam/trajectory"] = self._wandb.Image(str(frontend_traj_path))
            if depth_path is not None:
                image_payload["slam/depth_png"] = str(depth_path)
            if backend_rgb_path is not None:
                image_payload["backend/render_vs_gt_panorama"] = self._wandb.Image(str(backend_rgb_path))
                image_payload["backend/render_vs_gt_png"] = str(backend_rgb_path)
            if backend_depth_path is not None:
                image_payload["backend/render_depth"] = self._wandb.Image(str(backend_depth_path))
                image_payload["backend/render_depth_png"] = str(backend_depth_path)
            image_payload["frontend/trajectory_png"] = str(frontend_traj_path)
            image_payload["backend/trajectory_png"] = str(backend_traj_path)
            self._log_wandb_payload(image_payload, step=self._step)

    def _observe_m3_debug(self, m3_debug: dict | None) -> None:
        if not m3_debug:
            return
        stats = m3_debug.get("stats")
        if stats is None or not bool(getattr(stats, "enabled", False)):
            return
        chunk_index = int(m3_debug.get("chunk_index", -1))
        if self._last_m3_chunk_logged == chunk_index:
            return
        self._last_m3_chunk_logged = chunk_index

        initial = float(getattr(stats, "initial_mean_residual_deg", 0.0))
        mean = float(getattr(stats, "mean_residual_deg", 0.0))
        payload: dict[str, float | int] = {
            "m3/chunk": int(chunk_index),
            "m3/ba_success": int(bool(getattr(stats, "success", False))),
            "m3/valid_factor_ratio": float(getattr(stats, "valid_factor_ratio", 0.0)),
            "m3/residual_drop_deg": float(initial - mean),
        }
        self._log_wandb_payload(payload, step=self._step)

        should_log_images = self.save_local and (
            chunk_index == 0 or chunk_index % self.m3_log_every == 0
        )
        if not should_log_images:
            return
        paths = self._save_m3_debug_images(m3_debug, chunk_index=chunk_index)
        if self.run is not None and self._wandb is not None and paths:
            image_payload = {}
            if paths.get("match_lines") is not None:
                image_payload["m3/match_lines"] = self._wandb.Image(str(paths["match_lines"]))
            if paths.get("sky_prob") is not None:
                image_payload["m3/sky_prob"] = self._wandb.Image(str(paths["sky_prob"]))
            if image_payload:
                self._log_wandb_payload(image_payload, step=self._step)

    def observe_keyframe_opt(self, diagnostic, *, step: int | None = None) -> None:
        """Save and log post-optimization keyframe render diagnostics."""

        if diagnostic is None:
            return
        self._kf_opt_count += 1
        frame_id = int(getattr(diagnostic, "frame_id"))
        log_step = self._wandb_step(step)
        payload: dict[str, float | int] = {
            "backend/kf_opt_frame_id": frame_id,
            "backend/kf_opt_loss": float(getattr(diagnostic, "loss", 0.0)),
            "backend/kf_opt_psnr": float(getattr(diagnostic, "psnr", 0.0)),
            "backend/kf_opt_anchor_count": int(getattr(diagnostic, "anchor_count", 0)),
        }
        self._log_wandb_payload(payload, step=log_step)

        should_log_image = self.save_kf_opt and (
            self._kf_opt_count == 1 or self._kf_opt_count % self.kf_opt_log_every == 0
        )
        if not should_log_image:
            return

        render_panel = self._make_keyframe_opt_render_panel(diagnostic)
        depth_panel = self._make_keyframe_opt_depth_panel(diagnostic)
        render_path = None
        depth_path = None
        if self.save_local:
            render_path = self._save_keyframe_opt_image(
                render_panel,
                self.output_dir / "kf_renders_opt",
                frame_id=frame_id,
            )
            depth_path = self._save_keyframe_opt_image(
                depth_panel,
                self.output_dir / "kf_depths_opt",
                frame_id=frame_id,
            )

        if self.run is not None and self._wandb is not None:
            image_payload = {
                "backend/kf_render_opt": self._wandb.Image(str(render_path) if render_path is not None else render_panel),
                "backend/kf_depth_opt": self._wandb.Image(str(depth_path) if depth_path is not None else depth_panel),
            }
            if render_path is not None:
                image_payload["backend/kf_render_opt_png"] = str(render_path)
            if depth_path is not None:
                image_payload["backend/kf_depth_opt_png"] = str(depth_path)
            self._log_wandb_payload(image_payload, step=log_step)

    def observe_post_optimized_window(
        self,
        diagnostics: list,
        *,
        window_id: int,
        step: int | None = None,
    ) -> None:
        """Log every frame rendered after a successful spherical window update."""

        if not self.post_opt_all_frames:
            return
        valid = [diagnostic for diagnostic in diagnostics if diagnostic is not None]
        if not valid:
            return

        log_step = self._wandb_step(step)
        render_media = []
        depth_media = []
        losses: list[float] = []
        psnrs: list[float] = []
        for diagnostic in valid:
            frame_id = int(getattr(diagnostic, "frame_id"))
            loss = float(getattr(diagnostic, "loss", 0.0))
            psnr = float(getattr(diagnostic, "psnr", 0.0))
            losses.append(loss)
            psnrs.append(psnr)
            caption = (
                f"window={int(window_id)} frame={frame_id} "
                f"loss={loss:.4f} PSNR={psnr:.2f}dB "
                f"anchors={int(getattr(diagnostic, 'anchor_count', 0))}"
            )

            render_panel = self._make_keyframe_opt_render_panel(diagnostic)
            render_path = None
            if self.save_local:
                render_path = self._save_post_opt_window_image(
                    render_panel,
                    window_id=int(window_id),
                    frame_id=frame_id,
                    suffix="render_vs_gt",
                )
            if self.run is not None and self._wandb is not None:
                render_media.append(
                    self._wandb.Image(
                        str(render_path) if render_path is not None else render_panel,
                        caption=caption,
                    )
                )

            if self.post_opt_log_depth:
                depth_panel = self._make_keyframe_opt_depth_panel(diagnostic)
                depth_path = None
                if self.save_local:
                    depth_path = self._save_post_opt_window_image(
                        depth_panel,
                        window_id=int(window_id),
                        frame_id=frame_id,
                        suffix="depth",
                    )
                if self.run is not None and self._wandb is not None:
                    depth_media.append(
                        self._wandb.Image(
                            str(depth_path) if depth_path is not None else depth_panel,
                            caption=caption,
                        )
                    )

        payload: dict[str, object] = {
            "backend/post_opt_window_id": int(window_id),
            "backend/post_opt_frame_count": int(len(valid)),
            "backend/post_opt_mean_loss": float(np.mean(losses)),
            "backend/post_opt_mean_psnr": float(np.mean(psnrs)),
        }
        if render_media:
            payload["backend/post_opt_window_frames"] = render_media
        if depth_media:
            payload["backend/post_opt_window_depths"] = depth_media
        self._log_wandb_payload(payload, step=log_step)

    def observe_keyframe_inserted_gaussians(
        self,
        *,
        frame_id: int,
        inserted_count: int,
        step: int | None = None,
    ) -> None:
        if not self.log_keyframe_inserted_gaussians:
            return
        self._log_wandb_payload(
            {
                "mapping/keyframe_frame_id": int(frame_id),
                "mapping/keyframe_inserted_gaussians": int(inserted_count),
                "mapping/new_gaussians_inserted": int(inserted_count),
            },
            step=step,
        )

    def observe_keyframe_decision(self, decision: dict, *, step: int | None = None) -> None:
        if self.run is None:
            return
        payload: dict[str, float | int | str] = {
            "kf/frame_id": int(decision.get("frame_id", -1)),
            "kf/accepted": int(bool(decision.get("accepted", False))),
            "kf/keyframe_score": float(decision.get("keyframe_score", 0.0)),
        }
        for source, target in (
            ("frame_mean_pair_conf", "kf/frame_mean_pair_conf"),
            ("low_pair_conf_ratio", "kf/low_pair_conf_ratio"),
            ("match_coverage", "kf/match_coverage"),
            ("translation_delta", "kf/translation_delta"),
            ("median_depth", "kf/median_depth"),
            ("translation_depth_ratio", "kf/translation_depth_ratio"),
            ("m3_keyframe_score", "kf/m3_keyframe_score"),
            ("map_coverage_deficit", "kf/map_coverage_deficit"),
            ("matching_uncertainty", "kf/matching_uncertainty"),
            ("graph_connectivity_deficit", "kf/graph_connectivity_deficit"),
            ("parallax_score", "kf/parallax_score"),
            ("keyframe_gap", "kf/keyframe_gap"),
        ):
            value = decision.get(source)
            if value is not None:
                payload[target] = float(value)
        quantiles = decision.get("pair_conf_quantiles")
        if isinstance(quantiles, dict):
            for key, value in quantiles.items():
                payload[f"kf/pair_conf_{key}"] = float(value)
        reasons = decision.get("reasons") or []
        if reasons:
            payload["kf/reason"] = ",".join(str(item) for item in reasons)
        self._log_wandb_payload(payload, step=step)

    def observe_profile(self, profile: dict, *, step: int | None = None) -> None:
        if self.run is None:
            return
        event = str(profile.get("event", "runtime")).replace(" ", "_")
        payload: dict[str, float] = {}
        for key, value in profile.items():
            if key == "event" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                payload[f"profile/{event}/{key}"] = float(value)
        if payload:
            self._log_wandb_payload(payload, step=step)

    def observe_new_gaussians(
        self,
        *,
        frame_id: int,
        image: torch.Tensor,
        source_hw: tuple[int, int] | None,
        requested_idx: torch.Tensor | None,
        inserted_idx: torch.Tensor | None,
        stats: dict[str, float | int] | None = None,
        step: int | None = None,
    ) -> Path | None:
        if source_hw is None or requested_idx is None or inserted_idx is None:
            return None
        height, width = int(source_hw[0]), int(source_hw[1])
        if height <= 0 or width <= 0:
            return None
        requested = torch.zeros(height * width, dtype=torch.bool)
        inserted = torch.zeros(height * width, dtype=torch.bool)
        req = requested_idx.detach().cpu().long()
        ins = inserted_idx.detach().cpu().long()
        req = req[(req >= 0) & (req < requested.numel())]
        ins = ins[(ins >= 0) & (ins < inserted.numel())]
        if req.numel():
            requested[req] = True
        if ins.numel():
            inserted[ins] = True
        requested_mask = requested.view(height, width)
        inserted_mask = inserted.view(height, width)
        candidate_only = requested_mask & ~inserted_mask

        rgb = _image_tensor_to_pil(image)
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.BILINEAR)
        overlay = np.asarray(rgb).astype(np.float32)
        cand_np = candidate_only.numpy()
        ins_np = inserted_mask.numpy()
        overlay[cand_np] = 0.45 * overlay[cand_np] + 0.55 * np.array([255.0, 210.0, 30.0])
        overlay[ins_np] = 0.35 * overlay[ins_np] + 0.65 * np.array([255.0, 30.0, 30.0])
        overlay_img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8), mode="RGB")
        req_img = Image.fromarray((requested_mask.numpy().astype(np.uint8) * 255), mode="L").convert("RGB")
        ins_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        ins_rgb[..., 0] = inserted_mask.numpy().astype(np.uint8) * 255
        ins_img = Image.fromarray(ins_rgb, mode="RGB")

        title_h = 30
        w, h = rgb.size
        canvas = Image.new("RGB", (4 * w, h + title_h), "white")
        for idx, panel in enumerate((rgb, req_img, ins_img, overlay_img)):
            canvas.paste(panel, (idx * w, title_h))
        draw = ImageDraw.Draw(canvas)
        labels = ("rgb", "candidate seeds", "inserted seeds", "overlay")
        for idx, label in enumerate(labels):
            draw.text((idx * w + 8, 8), label, fill=(0, 0, 0))
        stat_text = (
            f"KF {int(frame_id):06d} requested={int(req.numel())} inserted={int(ins.numel())} "
            f"ratio={float(inserted_mask.float().mean()):.4f}"
        )
        draw.text((8, h + title_h - 18), stat_text, fill=(0, 0, 0))
        canvas = _resize_to_max_width(canvas, self.kf_opt_max_width)

        path = None
        if self.save_local:
            out_dir = self.visualization_dir / "new_gaussians"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"frame_{int(frame_id):06d}.png"
            canvas.save(path)
        if self.run is not None:
            payload: dict[str, float | int | str] = {
                "mapping/new_gaussians_requested": int(req.numel()),
                "mapping/new_gaussians_inserted": int(ins.numel()),
                "mapping/new_gaussians_inserted_mask_ratio": float(inserted_mask.float().mean()),
            }
            if stats:
                for key, value in stats.items():
                    payload[f"mapping/new_gaussians_{key}"] = float(value)
            if self._wandb is not None:
                payload["mapping/new_gaussians"] = self._wandb.Image(str(path) if path is not None else canvas)
                if path is not None:
                    payload["mapping/new_gaussians_png"] = str(path)
            self._log_wandb_payload(payload, step=step)
        return path

    def observe_depth_insertion_diagnostic(
        self,
        *,
        frame_id: int,
        image: torch.Tensor,
        source_hw: tuple[int, int] | None,
        inserted_idx: torch.Tensor | None,
        diagnostic,
        stats: dict[str, float | int] | None = None,
        step: int | None = None,
    ) -> Path | None:
        if diagnostic is None:
            return None
        if source_hw is not None:
            height, width = int(source_hw[0]), int(source_hw[1])
        else:
            height, width = int(image.shape[-2]), int(image.shape[-1])
        if height <= 0 or width <= 0:
            return None

        def scalar_hw(value) -> torch.Tensor | None:
            if not torch.is_tensor(value):
                return None
            tensor = value.detach().cpu().float()
            while tensor.ndim > 2:
                tensor = tensor[0]
            if tensor.ndim != 2:
                return None
            if tuple(tensor.shape) != (height, width):
                tensor = F.interpolate(
                    tensor.view(1, 1, *tensor.shape),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            return tensor

        def mask_hw(value) -> torch.Tensor:
            if not torch.is_tensor(value):
                return torch.zeros(height, width, dtype=torch.bool)
            tensor = value.detach().cpu().bool()
            while tensor.ndim > 2:
                tensor = tensor[0]
            if tensor.ndim != 2:
                return torch.zeros(height, width, dtype=torch.bool)
            if tuple(tensor.shape) != (height, width):
                tensor = F.interpolate(
                    tensor.float().view(1, 1, *tensor.shape),
                    size=(height, width),
                    mode="nearest",
                )[0, 0].bool()
            return tensor

        def scalar_panel(value: torch.Tensor | None, label: str, *, require_positive: bool = True) -> Image.Image:
            if value is None:
                panel = Image.new("RGB", (width, height), "black")
                ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
                return panel
            valid = torch.isfinite(value)
            if require_positive:
                valid = valid & (value > 0.0)
            panel = Image.fromarray(_scalar_to_rgb(value.numpy(), valid.numpy()), mode="RGB")
            ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
            return panel

        rgb = _image_tensor_to_pil(image)
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.BILINEAR)
        render_depth = scalar_hw(getattr(diagnostic, "render_depth", None))
        predicted_depth = scalar_hw(getattr(diagnostic, "predicted_depth", None))
        missing = mask_hw(getattr(diagnostic, "missing_mask", None))
        depth_mismatch = mask_hw(getattr(diagnostic, "depth_mismatch_mask", None))
        need_insert = mask_hw(getattr(diagnostic, "render_bad_mask", None))

        inserted_mask = torch.zeros(height * width, dtype=torch.bool)
        if inserted_idx is not None:
            ins = inserted_idx.detach().cpu().long()
            ins = ins[(ins >= 0) & (ins < inserted_mask.numel())]
            if ins.numel():
                inserted_mask[ins] = True
        inserted_mask = inserted_mask.view(height, width)

        need_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        miss_np = missing.numpy()
        mismatch_np = depth_mismatch.numpy()
        need_np = need_insert.numpy()
        need_rgb[need_np] = np.array([90, 90, 90], dtype=np.uint8)
        need_rgb[miss_np] = np.array([255, 210, 30], dtype=np.uint8)
        need_rgb[mismatch_np] = np.array([255, 40, 40], dtype=np.uint8)
        need_panel = Image.fromarray(need_rgb, mode="RGB")
        ImageDraw.Draw(need_panel).text((8, 8), "inconsistent mask", fill=(255, 255, 255))

        overlay = np.asarray(rgb).astype(np.float32)
        overlay[need_np] = 0.55 * overlay[need_np] + 0.45 * np.array([255.0, 210.0, 30.0])
        ins_np = inserted_mask.numpy()
        overlay[ins_np] = 0.35 * overlay[ins_np] + 0.65 * np.array([255.0, 30.0, 30.0])
        overlay_panel = Image.fromarray(overlay.clip(0, 255).astype(np.uint8), mode="RGB")
        ImageDraw.Draw(overlay_panel).text((8, 8), "rgb overlay: red=inserted", fill=(255, 255, 255))

        render_label = "render depth before insertion" if render_depth is not None else "no prior render depth"
        pred_label = "frontend predicted depth aligned"
        scale = float(getattr(diagnostic, "depth_scale", 1.0))
        shift = float(getattr(diagnostic, "depth_shift", 0.0))
        panels = (
            scalar_panel(render_depth, render_label),
            scalar_panel(predicted_depth, pred_label),
            need_panel,
            overlay_panel,
        )

        title_h = 30
        canvas = Image.new("RGB", (len(panels) * width, height + title_h), "white")
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * width, title_h))
        draw = ImageDraw.Draw(canvas)
        stats = stats or {}
        seed_stats = (
            f"dense={int(stats.get('dense_seed_candidates', 0))} "
            f"mask_seed={int(stats.get('insert_mask_seed_candidates', 0))} "
            f"vox_seed={int(stats.get('voxel_seed_candidates', 0))} "
            f"new={int(stats.get('replace_newly_inserted', int(inserted_mask.sum().item())))} "
            f"pred={int(stats.get('pred_depth_generated_seeds', 0))} "
            f"bad_depth={int(stats.get('pred_depth_invalid_pixels', 0))} "
            f"fuse_old={int(stats.get('replace_fused_existing', 0))} "
            f"fuse_dup={int(stats.get('replace_fused_new_duplicate', 0))} "
            f"compact={int(stats.get('replace_compacted', 0))}"
        )
        draw.text(
            (8, 8),
            (
                f"KF {int(frame_id):06d} need={int(need_insert.sum().item())} "
                f"missing={int(missing.sum().item())} depth={int(depth_mismatch.sum().item())} "
                f"inserted={int(inserted_mask.sum().item())} {seed_stats}"
            ),
            fill=(0, 0, 0),
        )
        canvas = _resize_to_max_width(canvas, self.kf_opt_max_width)

        path = None
        if self.save_local:
            out_dir = self.visualization_dir / "depth_insertion"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"frame_{int(frame_id):06d}.png"
            canvas.save(path)
        if self.run is not None:
            payload: dict[str, float | int | str] = {
                "mapping/depth_insertion_need_pixels": int(need_insert.sum().item()),
                "mapping/depth_insertion_missing_pixels": int(missing.sum().item()),
                "mapping/depth_insertion_depth_mismatch_pixels": int(depth_mismatch.sum().item()),
                "mapping/depth_insertion_inserted_pixels": int(inserted_mask.sum().item()),
                "mapping/depth_insertion_depth_scale": float(scale),
                "mapping/depth_insertion_depth_shift": float(shift),
            }
            for key, value in stats.items():
                payload[f"mapping/depth_insertion_{key}"] = float(value)
            if self._wandb is not None:
                payload["mapping/depth_insertion"] = self._wandb.Image(str(path) if path is not None else canvas)
                if path is not None:
                    payload["mapping/depth_insertion_png"] = str(path)
            self._log_wandb_payload(payload, step=step)
        return path

    def observe_sky_mask(
        self,
        *,
        frame_id: int,
        sky_mask: torch.Tensor | None,
        step: int | None = None,
    ) -> Path | None:
        if sky_mask is None:
            return None
        try:
            image = _mask_tensor_to_pil(sky_mask)
        except Exception:
            return None
        path = None
        if self.save_local:
            out_dir = self.visualization_dir / "sky_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"frame_{int(frame_id):06d}.png"
            image.save(path)
        if self.run is not None and self._wandb is not None:
            self._log_wandb_payload(
                {"m3/sky_mask": self._wandb.Image(str(path) if path is not None else image)},
                step=step,
            )
        return path

    def save_keyframe_diagnostic(self, diagnostic, *, render_dir: Path, depth_dir: Path) -> tuple[Path | None, Path | None]:
        if diagnostic is None:
            return None, None
        render_panel = self._make_keyframe_opt_render_panel(diagnostic)
        depth_panel = self._make_keyframe_opt_depth_panel(diagnostic)
        frame_id = int(getattr(diagnostic, "frame_id"))
        render_path = self._save_keyframe_opt_image(render_panel, render_dir, frame_id=frame_id)
        depth_path = self._save_keyframe_opt_image(depth_panel, depth_dir, frame_id=frame_id)
        return render_path, depth_path

    def log_image_file(self, key: str, path: Path, *, step: int | None = None) -> None:
        if self.run is None or self._wandb is None:
            return
        if not self._should_log_wandb_key(key):
            return
        try:
            payload = {key: self._wandb.Image(str(path))}
            if self.log_image_paths:
                payload[f"{key}_png"] = str(path)
            self._log_wandb_payload(payload, step=step)
        except Exception:
            return

    def observe_rendered_overlap_alignment(
        self,
        diagnostic: dict[str, torch.Tensor] | None,
        *,
        window_id: int,
        step: int | None = None,
    ) -> Path | None:
        if not diagnostic:
            return None

        frame_ids_tensor = diagnostic.get("frame_ids")
        frame_ids = (
            [int(value) for value in frame_ids_tensor.detach().cpu().reshape(-1)]
            if torch.is_tensor(frame_ids_tensor)
            else [None]
        )
        frame_count = max(1, len(frame_ids))

        def frame_tensor(value: torch.Tensor, frame_index: int) -> torch.Tensor:
            tensor = value.detach().cpu()
            if (
                frame_count > 1
                and tensor.ndim >= 3
                and int(tensor.shape[0]) == frame_count
            ):
                tensor = tensor[frame_index]
            while tensor.ndim > 2:
                tensor = tensor[0]
            return tensor

        def scalar_panel(
            name: str,
            label: str,
            *,
            frame_index: int,
            positive: bool = False,
        ) -> Image.Image:
            value = diagnostic.get(name)
            if not torch.is_tensor(value):
                panel = Image.new("RGB", (640, 320), "black")
            else:
                tensor = frame_tensor(value, frame_index).float()
                valid = torch.isfinite(tensor)
                if positive:
                    valid &= tensor > 0.0
                panel = Image.fromarray(
                    _scalar_to_rgb(tensor.numpy(), valid.numpy()),
                    mode="RGB",
                )
            ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
            return panel

        def mask_overlay(frame_index: int, label: str) -> Image.Image:
            sky = frame_tensor(diagnostic["sky_mask"], frame_index).bool()
            valid = frame_tensor(diagnostic["valid_mask"], frame_index).bool()
            inlier = frame_tensor(diagnostic["inlier_mask"], frame_index).bool()
            rgb = np.zeros((*sky.shape, 3), dtype=np.uint8)
            rgb[valid.numpy()] = np.array([64, 128, 255], dtype=np.uint8)
            rgb[sky.numpy()] = np.array([255, 160, 0], dtype=np.uint8)
            rgb[inlier.numpy()] = np.array([0, 255, 64], dtype=np.uint8)
            panel = Image.fromarray(rgb, mode="RGB")
            ImageDraw.Draw(panel).text(
                (8, 8),
                f"{label} mask: orange=sky blue=valid green=inlier",
                fill=(255, 255, 255),
            )
            return panel

        panels: list[Image.Image] = []
        for frame_index, frame_id in enumerate(frame_ids):
            prefix = (
                f"frame {frame_id}"
                if frame_id is not None
                else "shared frame"
            )
            panels.extend(
                [
                    scalar_panel(
                        "local_depth",
                        f"{prefix}: current depth",
                        frame_index=frame_index,
                        positive=True,
                    ),
                    scalar_panel(
                        "aligned_local_depth",
                        f"{prefix}: scale-aligned current depth",
                        frame_index=frame_index,
                        positive=True,
                    ),
                    scalar_panel(
                        "global_depth",
                        f"{prefix}: previous depth",
                        frame_index=frame_index,
                        positive=True,
                    ),
                    scalar_panel(
                        "relative_error",
                        f"{prefix}: relative depth error",
                        frame_index=frame_index,
                    ),
                    scalar_panel(
                        "local_alpha",
                        f"{prefix}: current alpha",
                        frame_index=frame_index,
                    ),
                    scalar_panel(
                        "global_alpha",
                        f"{prefix}: previous alpha",
                        frame_index=frame_index,
                    ),
                    mask_overlay(frame_index, prefix),
                    Image.new("RGB", (640, 320), "black"),
                ]
            )
        target_size = panels[0].size
        panels = [
            panel
            if panel.size == target_size
            else panel.resize(target_size, Image.Resampling.NEAREST)
            for panel in panels
        ]
        width, height = target_size
        canvas = Image.new(
            "RGB",
            (4 * width, 2 * frame_count * height),
            "black",
        )
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 4) * width, (index // 4) * height))
        canvas = _resize_to_max_width(canvas, max(960, self.kf_opt_max_width))
        path = None
        if self.save_local:
            directory = self.visualization_dir / "rendered_overlap_alignment"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"window_{int(window_id):06d}.png"
            canvas.save(path)
        if self.run is not None and self._wandb is not None:
            payload = {
                "backend/rendered_overlap_alignment": self._wandb.Image(
                    str(path) if path is not None else canvas
                )
            }
            if path is not None and self.log_image_paths:
                payload["backend/rendered_overlap_alignment_png"] = str(path)
            self._log_wandb_payload(payload, step=step)
        return path

    def log_artifact_file(self, path: Path) -> None:
        if self.run is None:
            return
        try:
            self.run.save(str(path), base_path=str(self.output_dir))
        except Exception:
            return

    def _make_keyframe_opt_render_panel(self, diagnostic) -> Image.Image:
        target = _image_tensor_to_pil(getattr(diagnostic, "target"))
        render = _image_tensor_to_pil(getattr(diagnostic, "render"))
        if render.size != target.size:
            render = render.resize(target.size, Image.BILINEAR)
        w, h = target.size
        canvas = Image.new("RGB", (2 * w, h + 28), "white")
        canvas.paste(target, (0, 28))
        canvas.paste(render, (w, 28))
        draw = ImageDraw.Draw(canvas)
        phase = str(getattr(diagnostic, "phase", None) or "post-opt")
        label = (
            f"KF {int(getattr(diagnostic, 'frame_id')):04d} [{phase}] "
            f"loss={float(getattr(diagnostic, 'loss', 0.0)):.4f} "
            f"PSNR={float(getattr(diagnostic, 'psnr', 0.0)):.2f}dB "
            f"anchors={int(getattr(diagnostic, 'anchor_count', 0))}"
        )
        draw.text((8, 7), "target panorama", fill=(0, 0, 0))
        draw.text((w + 8, 7), label, fill=(0, 0, 0))
        return _resize_to_max_width(canvas, self.kf_opt_max_width)

    def _make_keyframe_opt_depth_panel(self, diagnostic) -> Image.Image:
        render_depth = getattr(diagnostic, "depth", None)
        target_depth = getattr(diagnostic, "target_depth", None)
        if not torch.is_tensor(render_depth) and not torch.is_tensor(target_depth):
            image = Image.new("RGB", (900, 480), "white")
            ImageDraw.Draw(image).text((20, 20), "no optimized keyframe depth", fill=(0, 0, 0))
            return image

        def depth_panel(value, label: str) -> Image.Image:
            if not torch.is_tensor(value):
                panel = Image.new("RGB", (900, 480), "black")
                ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
                return panel
            depth_t = value.detach().cpu().float()
            while depth_t.ndim > 2:
                depth_t = depth_t[0]
            if depth_t.ndim != 2:
                panel = Image.new("RGB", (900, 480), "black")
                ImageDraw.Draw(panel).text((8, 8), f"bad depth shape: {tuple(value.shape)}", fill=(255, 255, 255))
                return panel
            valid = torch.isfinite(depth_t) & (depth_t > 1e-6)
            panel = Image.fromarray(_scalar_to_rgb(depth_t.numpy(), valid.numpy()), mode="RGB")
            ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
            return panel

        target_panel = depth_panel(target_depth, "frontend keyframe depth")
        render_panel = depth_panel(render_depth, "render depth")
        if render_panel.size != target_panel.size:
            render_panel = render_panel.resize(target_panel.size, Image.BILINEAR)
        w, h = target_panel.size
        canvas = Image.new("RGB", (2 * w, h), "white")
        canvas.paste(target_panel, (0, 0))
        canvas.paste(render_panel, (w, 0))
        return _resize_to_max_width(canvas, self.kf_opt_max_width)

    def _save_keyframe_opt_image(self, image: Image.Image, directory: Path, *, frame_id: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        fmt = str(self.results_cfg.get("kf_render_format", "png")).lower().strip()
        ext = "jpg" if fmt in {"jpg", "jpeg"} else "png"
        path = directory / f"kf_{int(frame_id):04d}.{ext}"
        if ext == "jpg":
            quality = max(1, min(100, int(self.results_cfg.get("kf_jpeg_quality", 95))))
            image.convert("RGB").save(path, quality=quality)
        else:
            image.save(path)
        return path

    def _save_post_opt_window_image(
        self,
        image: Image.Image,
        *,
        window_id: int,
        frame_id: int,
        suffix: str,
    ) -> Path:
        directory = self.visualization_dir / "post_opt" / f"window_{int(window_id):06d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"frame_{int(frame_id):06d}_{suffix}.png"
        image.save(path)
        return path

    def _save_m3_debug_images(self, m3_debug: dict, *, chunk_index: int) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {"match_lines": None, "sky_prob": None}
        images = m3_debug.get("images")
        image_hw = m3_debug.get("image_hw")
        feature_hw = m3_debug.get("feature_hw")
        sky_prob = m3_debug.get("sky_prob")
        graph = m3_debug.get("factor_graph")
        if image_hw is None and torch.is_tensor(images):
            image_hw = tuple(int(v) for v in images.shape[-2:])

        if torch.is_tensor(sky_prob) and sky_prob.numel():
            sky_path = self.visualization_dir / f"m3_chunk_{int(chunk_index):06d}_sky_prob.png"
            _scalar_tensor_to_pil(sky_prob[0, 0]).save(sky_path)
            paths["sky_prob"] = sky_path

        if graph is not None and torch.is_tensor(images) and image_hw is not None and feature_hw is not None:
            match_path = self._save_m3_match_lines(
                images.float(),
                graph,
                image_hw=tuple(int(v) for v in image_hw),
                feature_hw=tuple(int(v) for v in feature_hw),
                chunk_index=chunk_index,
            )
            paths["match_lines"] = match_path
        return paths

    def _save_m3_match_lines(
        self,
        images: torch.Tensor,
        graph,
        *,
        image_hw: tuple[int, int],
        feature_hw: tuple[int, int],
        chunk_index: int,
    ) -> Path | None:
        factors = list(getattr(graph, "factors", []) or [])
        if not factors:
            return None
        factor = factors[0]
        src_idx = int(getattr(factor, "src", 0))
        tgt_idx = int(getattr(factor, "tgt", min(1, int(images.shape[0]) - 1)))
        if src_idx >= int(images.shape[0]) or tgt_idx >= int(images.shape[0]):
            return None
        if tuple(images.shape[-2:]) != tuple(image_hw):
            images = F.interpolate(images, size=image_hw, mode="bilinear", align_corners=False)
        src_image = _image_tensor_to_pil(images[src_idx])
        tgt_image = _image_tensor_to_pil(images[tgt_idx])
        canvas = Image.new("RGB", (src_image.width + tgt_image.width, max(src_image.height, tgt_image.height)))
        canvas.paste(src_image, (0, 0))
        canvas.paste(tgt_image, (src_image.width, 0))
        draw = ImageDraw.Draw(canvas)

        valid = factor.valid_mask.detach().cpu().bool().reshape(-1)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()[: self.m3_max_matches]
        if valid_idx.numel() == 0:
            return None
        src_uv = feature_uv_to_image_uv(factor.src_uv.detach().cpu()[valid_idx], feature_hw, image_hw)
        tgt_uv = feature_uv_to_image_uv(factor.tgt_uv.detach().cpu()[valid_idx], feature_hw, image_hw)
        for src, tgt in zip(src_uv.tolist(), tgt_uv.tolist()):
            sx, sy = float(src[0]), float(src[1])
            tx, ty = src_image.width + float(tgt[0]), float(tgt[1])
            draw.line([(sx, sy), (tx, ty)], fill=(64, 220, 120), width=1)
            draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(255, 80, 80))
            draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=(80, 180, 255))

        path = self.visualization_dir / f"m3_chunk_{int(chunk_index):06d}_match_lines.png"
        _resize_to_max_width(canvas, 1800).save(path)
        return path

    def _should_visualize(self, output: FrontendOutput) -> bool:
        if not self.save_local:
            return False
        return self._step == 1 or self._step % self.log_every == 0 or (
            self.log_keyframes and bool(output.is_keyframe)
        )

    def _save_depth_panel(self, output: FrontendOutput, source_frame: PanoFrame) -> Path:
        rgb = _image_tensor_to_pil(source_frame.image)
        depth = _inverse_depth_to_pil(output.inverse_depth)
        if depth.size != rgb.size:
            depth = depth.resize(rgb.size, Image.BILINEAR)
        w, h = rgb.size
        canvas = Image.new("RGB", (2 * w, h + 26), "white")
        canvas.paste(rgb, (0, 26))
        canvas.paste(depth, (w, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "source ERP", fill=(0, 0, 0))
        draw.text((w + 8, 6), "pred depth", fill=(0, 0, 0))
        canvas = _resize_to_max_width(canvas, 1600)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_depth.png"
        canvas.save(path)
        return path

    def _save_trajectory_panel(
        self,
        output: FrontendOutput,
        *,
        kind: str,
        pred_history: list[tuple[int, np.ndarray]],
    ) -> Path:
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_{kind}_trajectory_vs_gt.png"
        legacy_path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_trajectory.png" if kind == "frontend" else None
        positions = np.asarray([p for _, p in pred_history], dtype=np.float32)
        frame_ids = np.asarray([fid for fid, _ in pred_history], dtype=np.float32)
        if positions.size == 0:
            image = Image.new("RGB", (900, 640), "white")
            ImageDraw.Draw(image).text((20, 20), "no valid trajectory", fill=(0, 0, 0))
            image.save(path)
            if legacy_path is not None:
                shutil.copyfile(path, legacy_path)
            return path
        gt_by_id = {fid: xyz for fid, xyz in self._gt_pose_history}
        gt_positions = np.asarray([gt_by_id.get(int(fid), np.full(3, np.nan)) for fid in frame_ids], dtype=np.float32)
        has_gt = bool(np.isfinite(gt_positions).all(axis=1).any())
        positions_plot = positions
        ape_errors = None
        ape_metrics: dict[str, float] = {}
        sim3_aligned = False
        if has_gt:
            positions_plot, ape_errors, ape_metrics, sim3_aligned = _compute_ape_translation(
                positions,
                gt_positions,
            )
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            fig = plt.figure(figsize=(8.5, 6.2), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            plot_pred = _xyz_to_y_up_plot(positions_plot)
            all_for_limits = [plot_pred]
            valid_gt = np.zeros((len(frame_ids),), dtype=bool)
            plot_gt = None
            if has_gt:
                valid_gt = np.isfinite(gt_positions).all(axis=1)
                plot_gt_all = _xyz_to_y_up_plot(gt_positions)
                plot_gt = plot_gt_all[valid_gt]
                all_for_limits.append(plot_gt)
                ax.plot(
                    plot_gt[:, 0],
                    plot_gt[:, 1],
                    plot_gt[:, 2],
                    color="#6b7280",
                    linestyle="--",
                    linewidth=1.6,
                    label="GT",
                )
            if len(plot_pred) >= 2:
                segments = np.stack([plot_pred[:-1], plot_pred[1:]], axis=1)
                if ape_errors is not None and np.isfinite(ape_errors).any():
                    endpoint_errors = np.stack([ape_errors[:-1], ape_errors[1:]], axis=1)
                    finite_counts = np.isfinite(endpoint_errors).sum(axis=1)
                    segment_errors = np.divide(
                        np.nansum(endpoint_errors, axis=1),
                        np.maximum(finite_counts, 1),
                    )
                    segment_errors[finite_counts == 0] = np.nan
                    finite = np.isfinite(segment_errors)
                    if finite.any():
                        fill = float(np.nanmedian(segment_errors[finite]))
                        segment_errors = np.where(finite, segment_errors, fill)
                    else:
                        segment_errors = np.zeros((len(segments),), dtype=np.float32)
                    line_collection = Line3DCollection(segments, cmap="turbo", linewidth=2.2)
                    line_collection.set_array(segment_errors.astype(np.float32))
                    line_collection.set_label(f"{kind} pred")
                    ax.add_collection3d(line_collection)
                    fig.colorbar(
                        line_collection,
                        ax=ax,
                        shrink=0.75,
                        pad=0.08,
                        label="APE translation",
                    )
                else:
                    line_collection = Line3DCollection(segments, colors="#1f77b4", linewidth=2.2)
                    line_collection.set_label(f"{kind} pred")
                    ax.add_collection3d(line_collection)
            else:
                ax.plot(
                    plot_pred[:, 0],
                    plot_pred[:, 1],
                    plot_pred[:, 2],
                    color="#1f77b4",
                    linewidth=2.2,
                    label=f"{kind} pred",
                )
            ax.scatter(plot_pred[0, 0], plot_pred[0, 1], plot_pred[0, 2], c="limegreen", s=64, label="start")
            ax.scatter(plot_pred[-1, 0], plot_pred[-1, 1], plot_pred[-1, 2], c="red", s=64, label="latest")
            limits_pts = np.concatenate([arr for arr in all_for_limits if arr.size], axis=0)
            center = 0.5 * (limits_pts.min(axis=0) + limits_pts.max(axis=0))
            radius = max(float((limits_pts.max(axis=0) - limits_pts.min(axis=0)).max()) * 0.55, 1e-3)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            if has_gt:
                align_text = "Sim(3) Umeyama" if sim3_aligned else "unaligned (<3 GT matches)"
                metric_text = ""
                if ape_metrics:
                    metric_text = (
                        f"\nAPE trans. {align_text}: "
                        f"RMSE={ape_metrics['rmse']:.3f}, "
                        f"mean={ape_metrics['mean']:.3f}, "
                        f"max={ape_metrics['max']:.3f}"
                    )
                ax.set_title(f"{kind} trajectory APE vs GT{metric_text}")
            else:
                ax.set_title(f"{kind} trajectory")
            ax.view_init(elev=24, azim=-58)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(path, facecolor="white")
            plt.close(fig)
        except Exception:
            self._save_topdown_trajectory(path, positions_plot)
        if legacy_path is not None:
            shutil.copyfile(path, legacy_path)
        return path

    def _save_backend_render_panel(
        self,
        output: FrontendOutput,
        source_frame: PanoFrame,
        render_pkg: dict,
    ) -> Path:
        target = _image_tensor_to_pil(source_frame.image)
        render = _image_tensor_to_pil(render_pkg["render"])
        if render.size != target.size:
            render = render.resize(target.size, Image.BILINEAR)
        w, h = target.size
        canvas = Image.new("RGB", (2 * w, h + 26), "white")
        canvas.paste(target, (0, 26))
        canvas.paste(render, (w, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "target panorama", fill=(0, 0, 0))
        draw.text((w + 8, 6), "backend render", fill=(0, 0, 0))
        canvas = _resize_to_max_width(canvas, 1600)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_backend_render_vs_gt.png"
        canvas.save(path)
        return path

    def _save_backend_depth_panel(self, output: FrontendOutput, render_pkg: dict) -> Path:
        depth = render_pkg.get("depth")
        if depth is None:
            image = Image.new("RGB", (900, 480), "white")
            ImageDraw.Draw(image).text((20, 20), "no backend depth", fill=(0, 0, 0))
        else:
            image = _inverse_depth_to_pil(depth.detach().clamp_min(1e-6).reciprocal())
            image = _resize_to_max_width(image, 1200)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_backend_render_depth.png"
        image.save(path)
        return path

    def log_final_backend_trajectory(self, poses: list[tuple[int, torch.Tensor]], *, step: int) -> str | None:
        if not poses or not self.save_local:
            return None
        history = [(int(fid), pose.detach().cpu().float()[:3, 3].numpy()) for fid, pose in poses if pose.shape == (4, 4)]
        if not history:
            return None
        dummy = FrontendOutput(
            frame_id=history[-1][0],
            timestamp=float(history[-1][0]),
            pose_c2w=torch.eye(4),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=False,
            ba_residual=None,
            tracking_status="final_backend_trajectory",
        )
        path = self._save_trajectory_panel(dummy, kind="backend_final", pred_history=history)
        if self.run is not None and self._wandb is not None:
            self._log_wandb_payload(
                {"backend/final_trajectory_vs_gt": self._wandb.Image(str(path))},
                step=self._step + 1,
            )
        return str(path)

    def observe_backend_snapshot(self, snapshot, *, step: int) -> None:
        """Log asynchronous legacy backend snapshots."""

        poses = getattr(snapshot, "poses_c2w", {}) or {}
        for frame_id, pose in sorted(poses.items()):
            pose_cpu = pose.detach().cpu().float()
            if pose_cpu.shape != (4, 4):
                continue
            self._backend_pose_history = [
                item for item in self._backend_pose_history if item[0] != int(frame_id)
            ]
            self._backend_pose_history.append((int(frame_id), pose_cpu[:3, 3].numpy()))
        if not self.save_local or not poses:
            return
        latest_id = int(max(poses))
        dummy = FrontendOutput(
            frame_id=latest_id,
            timestamp=float(latest_id),
            pose_c2w=torch.eye(4),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=True,
            ba_residual=None,
            tracking_status=f"backend_snapshot_{getattr(snapshot, 'tag', 'unknown')}",
        )
        path = self._save_trajectory_panel(dummy, kind="backend", pred_history=self._backend_pose_history)
        if self.run is not None and self._wandb is not None:
            payload = {
                "backend/trajectory_vs_gt": self._wandb.Image(str(path)),
                "backend/trajectory_png": str(path),
            }
            render_path = getattr(snapshot, "render_path", None)
            if render_path:
                payload["backend/render_vs_gt_panorama"] = self._wandb.Image(str(render_path))
                payload["backend/render_vs_gt_png"] = str(render_path)
            depth_path = getattr(snapshot, "depth_path", None)
            if depth_path:
                payload["backend/render_depth"] = self._wandb.Image(str(depth_path))
                payload["backend/render_depth_png"] = str(depth_path)
            self._log_wandb_payload(payload, step=self._step + 1)

    @staticmethod
    def _save_topdown_trajectory(path: Path, positions: np.ndarray) -> None:
        image = Image.new("RGB", (900, 640), "white")
        draw = ImageDraw.Draw(image)
        pts = positions[:, [0, 2]]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = np.maximum(mx - mn, 1e-3)
        norm = (pts - mn) / span
        xy = np.stack([40 + norm[:, 0] * 820, 600 - norm[:, 1] * 560], axis=-1)
        draw.line([tuple(p) for p in xy], fill=(31, 119, 180), width=3)
        for p in xy:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(31, 119, 180))
        draw.text((20, 20), "top-down trajectory fallback", fill=(0, 0, 0))
        image.save(path)

    def finish(self, summary: dict | None = None) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(summary)
        self.run.finish()


class PanoDroidGSSlamSystem:
    """Small orchestration layer matching the original SLAM staging."""

    def __init__(self, config: dict) -> None:
        self.config = config
        runtime_mode = str(config.get("Runtime", {}).get("mode", "sync_mvp")).lower()
        self._delegate = None
        if runtime_mode == "legacy_online":
            from system.legacy_online_slam import PanoVGGTLegacyOnlineSlamSystem

            self._delegate = PanoVGGTLegacyOnlineSlamSystem(config)
            return
        self._apply_feedforward_window_frontend_defaults(config)
        self.frontend = build_frontend_from_config(config)
        mapping_cfg = config.get("Mapping", {})
        frontend_mode = str(config.get("Frontend", {}).get("mode", "graph")).lower()
        default_seed_source = "world_points_only" if frontend_mode == "panovggt_long" else "depth_pose"
        novel_cfg = (mapping_cfg.get("NovelGaussianInsertion", {}) or {})
        self.initializer = GaussianInitializer(
            max_seeds_per_keyframe=int(mapping_cfg.get("max_seeds_per_keyframe", 2048)),
            min_confidence=float(mapping_cfg.get("min_depth_confidence", 0.15)),
            sky_mask_enable=bool(mapping_cfg.get("sky_mask_enable", False)),
            sky_mask_top_ratio=float(mapping_cfg.get("sky_mask_top_ratio", 0.58)),
            sky_mask_min_blue=float(mapping_cfg.get("sky_mask_min_blue", 0.35)),
            sky_mask_blue_margin=float(mapping_cfg.get("sky_mask_blue_margin", 0.05)),
            sky_mask_cloud_brightness=float(mapping_cfg.get("sky_mask_cloud_brightness", 0.72)),
            sky_mask_cloud_saturation=float(mapping_cfg.get("sky_mask_cloud_saturation", 0.22)),
            sky_mask_texture_threshold=float(mapping_cfg.get("sky_mask_texture_threshold", 0.08)),
            sky_mask_source=str(mapping_cfg.get("sky_mask_source", "heuristic")),
            voxel_sizes=tuple(config.get("Hierarchical", {}).get("voxel_size_lis", [0.12, 0.45, 1.8])),
            seed_source=str(mapping_cfg.get("seed_source", default_seed_source)),
            insertion_strategy=str(novel_cfg.get("strategy", "legacy")),
            pfgs360_voxel_size=float(novel_cfg.get("voxel_size", 0.12)),
            pfgs360_gaussian_scale_mode=str(novel_cfg.get("gaussian_scale_mode", "voxel")),
            pfgs360_gaussian_scale_factor=float(novel_cfg.get("gaussian_scale_factor", 1.25)),
            pfgs360_gaussian_scale_min=float(novel_cfg.get("gaussian_scale_min", 0.008)),
            pfgs360_gaussian_scale_max=float(novel_cfg.get("gaussian_scale_max", 0.08)),
            pfgs360_gaussian_scale_lat_cos_min=float(novel_cfg.get("gaussian_scale_lat_cos_min", 0.25)),
            temporal_pair_conf_min=float(novel_cfg.get("temporal_pair_conf_min", 0.70)),
        )
        map_cfg = config.get("MapRepresentation", {}) if isinstance(config, dict) else {}
        map_mode = str(map_cfg.get("mode", "anchor_scaffold_panorama") or "anchor_scaffold_panorama").lower()
        if map_mode == "anchor_scaffold_panorama":
            self.map = PanoGaussianMap(config=config)
        elif map_mode == "neural_anchor_scaffold_panorama":
            self.map = NeuralScaffoldPanoMap(config=config)
        else:
            raise ValueError(
                "Unsupported MapRepresentation.mode "
                f"{map_mode!r}; expected 'anchor_scaffold_panorama' or 'neural_anchor_scaffold_panorama'."
            )
        render_cfg = config.get("Renderer", {})
        self.renderer = PFGS360Renderer(
            config=config,
            extra_gsplat360_roots=list(render_cfg.get("extra_gsplat360_roots", [])),
            allow_fallback=bool(render_cfg.get("allow_smoke_fallback", True)),
        )
        self.mapper = PanoGaussianMapper(
            self.map,
            renderer=self.renderer,
            lr=float(config.get("Mapping", {}).get("lr", 2e-3)),
        )
        self.spherical_selfi_global_backend = None
        spherical_global_cfg = config.get("SphericalSelfiGlobalBackend", {})
        if isinstance(spherical_global_cfg, dict) and bool(spherical_global_cfg.get("enabled", False)):
            if not isinstance(self.map, PanoGaussianMap):
                raise ValueError(
                    "SphericalSelfiGlobalBackend requires MapRepresentation.mode='anchor_scaffold_panorama'."
                )
            from backend.pano_gs.spherical_selfi_global import SphericalSelfiGlobalBackend

            # Runtime Fibonacci/window/sky settings are the shared source of
            # truth.  Backend-local values remain valid explicit overrides.
            spherical_global_cfg = dict(spherical_global_cfg)
            graph_cfg = dict(spherical_global_cfg.get("global_graph", {}) or {})
            runtime_cfg = dict(config.get("SphericalSelfiRuntime", {}) or {})
            fibonacci_cfg = dict(runtime_cfg.get("fibonacci", {}) or {})
            window_cfg = dict(runtime_cfg.get("window", {}) or {})
            sky_cfg = dict(runtime_cfg.get("sky", {}) or {})
            inherited = {
                "fibonacci_seed": fibonacci_cfg.get("seed"),
                "fibonacci_oversample_factor": fibonacci_cfg.get("oversample_factor"),
                "min_depth": fibonacci_cfg.get("min_depth"),
                "max_depth": fibonacci_cfg.get("max_depth"),
                "expected_overlap_frames": window_cfg.get("expected_overlap_frames"),
                "enforce_exact_overlap": window_cfg.get("enforce_exact_overlap"),
                "sky_threshold": sky_cfg.get("threshold"),
            }
            for key, value in inherited.items():
                if value is not None:
                    graph_cfg.setdefault(key, value)
            spherical_global_cfg["global_graph"] = graph_cfg
            voxel_cfg = dict(config.get("VoxelAnchorRefiner", {}) or {})
            voxel_refiner_enabled = bool(voxel_cfg.get("enabled", False))
            spherical_global_cfg["_voxel_anchor_refiner_enabled"] = (
                voxel_refiner_enabled
            )
            if voxel_refiner_enabled:
                rendered_cfg = dict(
                    spherical_global_cfg.get("rendered_overlap_alignment", {}) or {}
                )
                dedup_cfg = dict(
                    spherical_global_cfg.get("insertion_dedup", {}) or {}
                )
                if not bool(rendered_cfg.get("enabled", False)):
                    raise ValueError(
                        "VoxelAnchorRefiner requires "
                        "SphericalSelfiGlobalBackend.rendered_overlap_alignment.enabled=true"
                    )
                if not bool(dedup_cfg.get("enabled", False)):
                    raise ValueError(
                        "VoxelAnchorRefiner requires "
                        "SphericalSelfiGlobalBackend.insertion_dedup.enabled=true"
                    )
                voxel_sizes = tuple(
                    float(value)
                    for value in voxel_cfg.get(
                        "voxel_sizes", (0.04, 0.08, 0.16, 0.32)
                    )
                )
                fusion_sizes = tuple(
                    float(value)
                    for value in (
                        spherical_global_cfg.get("voxel_fusion", {}) or {}
                    ).get("voxel_sizes", (0.04, 0.08, 0.16, 0.32))
                )
                if voxel_sizes != fusion_sizes:
                    raise ValueError(
                        "VoxelAnchorRefiner.voxel_sizes must exactly match "
                        "SphericalSelfiGlobalBackend.voxel_fusion.voxel_sizes"
                    )
                adapter_dim = int(voxel_cfg.get("adapter_dim", 24))
                head_cfg = dict(config.get("head", {}) or {})
                if int(head_cfg.get("feature_dim", 24)) != adapter_dim:
                    raise ValueError(
                        "VoxelAnchorRefiner.adapter_dim must match head.feature_dim"
                    )
                if int(voxel_cfg.get("iterations", 3)) != 3:
                    raise ValueError(
                        "VoxelAnchorRefiner requires exactly three refinement iterations"
                    )
                if int(head_cfg.get("rgb_sh_degree", 2)) != 2:
                    raise ValueError(
                        "VoxelAnchorRefiner requires head.rgb_sh_degree=2"
                    )
                if int(spherical_global_cfg.get("rgb_sh_degree", 2)) != 2:
                    raise ValueError(
                        "VoxelAnchorRefiner requires "
                        "SphericalSelfiGlobalBackend.rgb_sh_degree=2"
                    )
                if bool(render_cfg.get("allow_smoke_fallback", True)):
                    raise ValueError(
                        "VoxelAnchorRefiner formal integration requires the real "
                        "gsplat360 renderer (Renderer.allow_smoke_fallback=false)"
                    )

            self.spherical_selfi_global_backend = SphericalSelfiGlobalBackend(
                self.map,
                mapper=self.mapper,
                renderer=self.renderer,
                config=spherical_global_cfg,
            )

    @staticmethod
    def _apply_feedforward_window_frontend_defaults(config: dict) -> None:
        frontend_mode = str(config.get("Frontend", {}).get("mode", "graph")).lower()
        if frontend_mode != "panovggt_long":
            return
        backend_cfg = config.get("BackendOptimization", {})
        if not isinstance(backend_cfg, dict):
            return
        ff_cfg = backend_cfg.get("FeedForwardWindow", {})
        ff_enabled = isinstance(ff_cfg, dict) and bool(ff_cfg.get("enabled", False))
        chunk_enabled = bool(backend_cfg.get("optimize_after_every_chunk", False))
        if not ff_enabled and not chunk_enabled:
            return
        history_keyframes = max(
            0,
            int(
                ff_cfg.get(
                    "history_keyframes",
                    backend_cfg.get("recent_keyframe_observation_frames", 2),
                )
                if isinstance(ff_cfg, dict)
                else backend_cfg.get("recent_keyframe_observation_frames", 2)
            ),
        )
        pano_cfg = config.setdefault("PanoVGGT", {})
        if not isinstance(pano_cfg, dict):
            return
        joint_cfg = pano_cfg.setdefault("JointInference", {})
        if isinstance(joint_cfg, dict):
            joint_cfg["max_history_frames"] = history_keyframes

    def _backend_feedback_cfg(self) -> dict:
        cfg = self.config.get("BackendFeedback", {})
        return cfg if isinstance(cfg, dict) else {}

    def _backend_feedback_enabled(self) -> bool:
        return bool(self._backend_feedback_cfg().get("enabled", False))

    @staticmethod
    def _pose_det(pose: torch.Tensor) -> float:
        return float(torch.linalg.det(pose[:3, :3].detach().float()).cpu())

    def _backend_feedback_hard_gate(
        self,
        *,
        frame_id: int,
        pose_c2w: torch.Tensor,
        metrics: dict,
        first_keyframe_id: int | None,
        registered_keyframes: set[int],
    ) -> dict:
        cfg = self._backend_feedback_cfg()
        min_steps = int(cfg.get("min_optimization_steps", 1))
        require_loss = bool(cfg.get("require_finite_loss", True))
        reject_first = bool(cfg.get("reject_first_keyframe_pose_feedback", True))
        min_det = float(cfg.get("min_rotation_det", 0.5))
        max_det = float(cfg.get("max_rotation_det", 1.5))
        steps_raw = metrics.get("steps", 0.0)
        steps = float(steps_raw.detach().cpu()) if torch.is_tensor(steps_raw) else float(steps_raw or 0.0)
        loss_raw = metrics.get("loss")
        loss = float(loss_raw.detach().cpu()) if torch.is_tensor(loss_raw) else (float(loss_raw) if loss_raw is not None else None)
        decision = {
            "frame_id": int(frame_id),
            "accepted": False,
            "reason": "unknown",
            "steps": steps,
            "loss": loss,
            "is_first_keyframe": bool(first_keyframe_id is not None and int(frame_id) == int(first_keyframe_id)),
            "det_rotation": None,
        }
        if int(frame_id) not in registered_keyframes:
            decision["reason"] = "not_registered_keyframe"
            return decision
        if decision["is_first_keyframe"] and reject_first:
            decision["reason"] = "first_keyframe_rejected"
            return decision
        if steps < float(min_steps):
            decision["reason"] = "insufficient_optimization_steps"
            return decision
        if require_loss and (loss is None or not np.isfinite(loss)):
            decision["reason"] = "nonfinite_loss"
            return decision
        if not torch.is_tensor(pose_c2w) or tuple(pose_c2w.shape) != (4, 4):
            decision["reason"] = "invalid_pose_shape"
            return decision
        pose = pose_c2w.detach().float()
        if not bool(torch.isfinite(pose).all()):
            decision["reason"] = "nonfinite_pose"
            return decision
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
        if not bool(torch.allclose(pose[3], expected_bottom, atol=1.0e-4, rtol=1.0e-4)):
            decision["reason"] = "invalid_homogeneous_row"
            return decision
        det = self._pose_det(pose)
        decision["det_rotation"] = det
        if not np.isfinite(det) or det < min_det or det > max_det:
            decision["reason"] = "invalid_rotation_determinant"
            return decision
        decision["accepted"] = True
        decision["reason"] = "accepted"
        return decision

    def _collect_backend_feedback_updates(
        self,
        metrics: dict,
    ) -> tuple[dict[int, torch.Tensor], list[dict]]:
        if not self._backend_feedback_enabled():
            return {}, []
        cfg = self._backend_feedback_cfg()
        registered = {int(kf.frame_id) for kf in self.mapper.keyframes}
        window_keyframes = set(int(fid) for fid in getattr(self.mapper.stats, "last_window_keyframes", []))
        eligible = window_keyframes if window_keyframes else registered
        first_id = int(self.mapper.keyframes[0].frame_id) if self.mapper.keyframes else None
        alpha = float(cfg.get("blend_alpha", 1.0))
        current_poses = getattr(self.frontend, "pose_by_frame", {})
        updates: dict[int, torch.Tensor] = {}
        decisions: list[dict] = []
        for frame_id, refined_pose in self.mapper.refined_keyframe_poses():
            fid = int(frame_id)
            if fid not in eligible:
                continue
            decision = self._backend_feedback_hard_gate(
                frame_id=fid,
                pose_c2w=refined_pose,
                metrics=metrics,
                first_keyframe_id=first_id,
                registered_keyframes=registered,
            )
            if decision["accepted"]:
                current = current_poses.get(fid) if isinstance(current_poses, dict) else None
                if torch.is_tensor(current) and tuple(current.shape) == (4, 4):
                    update = _se3_blend_pose(current.detach().cpu(), refined_pose.detach().cpu(), alpha)
                else:
                    update = refined_pose.detach().cpu().float()
                updates[fid] = update
                decision["blend_alpha"] = alpha
            decisions.append(decision)
        return updates, decisions

    def _apply_backend_feedback_updates(self, updates: dict[int, torch.Tensor]) -> int:
        if not updates:
            return 0
        apply_updates = getattr(self.frontend, "apply_backend_pose_updates", None)
        if not callable(apply_updates):
            self.mapper.stats.notes.append("backend feedback skipped: frontend has no apply_backend_pose_updates")
            return 0
        try:
            apply_updates(
                updates,
                update_last_keyframe_anchor=bool(
                    self._backend_feedback_cfg().get("update_last_keyframe_anchor", True)
                ),
            )
        except TypeError:
            apply_updates(updates)
        return len(updates)

    def run(self, *, max_frames: int | None = None) -> dict:
        if self._delegate is not None:
            return self._delegate.run(max_frames=max_frames)
        self.frontend.initialize({"config": self.config})
        results_cfg = self.config.get("Results", {})
        output_dir = Path(results_cfg.get("save_dir", "outputs/pano_droid_gs_slam"))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = SlamRuntimeLogger(self.config, output_dir)
        refine_steps = int(self.config.get("Mapping", {}).get("refine_steps_per_keyframe", 0))
        mapping_cfg = self.config.get("Mapping", {})
        novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
        novel_cfg = novel_cfg if isinstance(novel_cfg, dict) else {}
        replace_fuse_enabled = (
            bool(novel_cfg.get("enabled", False))
            and str(novel_cfg.get("strategy", "legacy")).lower() == "pfgs360_replace_fuse"
        )
        first_chunk_multiframe_init = bool(novel_cfg.get("first_chunk_multiframe_init", False))
        insert_keyframe_policy = str(novel_cfg.get("insert_keyframe_policy", "frontend") or "frontend").lower()
        insert_keyframe_block_size = max(
            1,
            int(
                novel_cfg.get(
                    "insert_keyframe_block_size",
                    (self.config.get("PanoVGGT", {}) or {}).get("chunk_size", 1),
                )
            ),
        )
        force_chunk_block_insertions = bool(
            replace_fuse_enabled and insert_keyframe_policy in {"new_block_last", "chunk_block_last"}
        )
        bootstrap_cfg = mapping_cfg.get("BootstrapOptimization", {}) if isinstance(mapping_cfg, dict) else {}
        bootstrap_enabled = bool(bootstrap_cfg.get("enabled", False))
        bootstrap_steps = int(bootstrap_cfg.get("first_keyframe_steps", 0))
        bootstrap_save_every = max(1, int(bootstrap_cfg.get("save_every", 25)))
        backend_cfg = self.config.get("BackendOptimization", {})
        feedforward_cfg = backend_cfg.get("FeedForwardWindow", {}) if isinstance(backend_cfg, dict) else {}
        feedforward_cfg = feedforward_cfg if isinstance(feedforward_cfg, dict) else {}
        feedforward_window_enabled = bool(feedforward_cfg.get("enabled", False)) or bool(
            backend_cfg.get("optimize_after_every_chunk", False)
        )
        frontend_mode = str(self.config.get("Frontend", {}).get("mode", "") or "").lower()
        resplat_fusion_cfg = self.config.get("ReSplatFusion", {})
        resplat_fusion_cfg = resplat_fusion_cfg if isinstance(resplat_fusion_cfg, dict) else {}
        resplat_direct_fusion_enabled = bool(resplat_fusion_cfg.get("enabled", False)) and frontend_mode in {
            "pano_resplat_online",
            "panoresplat_online",
            "resplat_online",
        }
        spherical_selfi_global_enabled = bool(
            self.spherical_selfi_global_backend is not None
            and getattr(self.spherical_selfi_global_backend, "enabled", False)
        )
        resplat_global_cfg = backend_cfg.get("ReSplatGlobal", {}) if isinstance(backend_cfg, dict) else {}
        resplat_global_cfg = resplat_global_cfg if isinstance(resplat_global_cfg, dict) else {}
        resplat_global_iters = int(resplat_global_cfg.get("iters", 20))
        non_keyframe_steps = int(backend_cfg.get("non_keyframe_steps", 0))
        pano_cfg = self.config.get("PanoVGGT", {})
        keyframe_anchor_cfg = pano_cfg.get("KeyframeAnchor", {}) if isinstance(pano_cfg, dict) else {}
        decision_logging_enabled = bool(keyframe_anchor_cfg.get("enabled", False))
        decision_path = output_dir / "keyframe_decisions.jsonl"
        decision_file = open(decision_path, "w", encoding="utf-8") if decision_logging_enabled else None
        feedback_cfg = self._backend_feedback_cfg()
        feedback_logging_enabled = self._backend_feedback_enabled() and bool(feedback_cfg.get("log_decisions", True))
        feedback_path = output_dir / "backend_feedback_decisions.jsonl"
        feedback_file = open(feedback_path, "w", encoding="utf-8") if feedback_logging_enabled else None
        profile_cfg = self.config.get("RuntimeProfiling", {})
        profiling_enabled = bool(profile_cfg.get("enabled", False)) if isinstance(profile_cfg, dict) else False
        profile_path = output_dir / str(profile_cfg.get("path", "runtime_profile.jsonl")) if profiling_enabled else None
        profile_file = open(profile_path, "w", encoding="utf-8") if profile_path is not None else None
        keyframe_decision_count = 0
        backend_feedback_decision_count = 0
        backend_feedback_applied_count = 0
        last_profiled_frontend_chunk: int | None = None
        last_optimized_frontend_chunk: int | None = None
        last_feedforward_metrics: dict = {}
        recent_feedforward_chunks: list[tuple[int | None, list[int]]] = []
        frame_cache: dict[int, PanoFrame] = {}
        final_frame_records: dict[int, dict] = {}
        local_ba_window_records: list[dict] = []
        chunk_keyframe_anchor_frame_id: int | None = None
        frame_count = 0
        keyframes = 0
        resplat_fusion_count = 0
        last_status = None
        frontend_sky_required = str(mapping_cfg.get("sky_mask_source", "heuristic") or "heuristic").lower() in {
            "panovggt",
            "panovggt_head",
            "pano_vggt",
            "m3",
            "m3_head",
        }

        def write_profile(event: str, **values) -> None:
            if not profiling_enabled:
                return
            profile = {"event": str(event), **values}
            if profile_file is not None:
                profile_file.write(json.dumps(profile, sort_keys=True) + "\n")
                profile_file.flush()
            logger.observe_profile(profile)

        def frontend_sky_mask_for_frame(frame_id: int, image: torch.Tensor) -> torch.Tensor | None:
            getter = getattr(self.frontend, "sky_mask_for_frame", None)
            height, width = int(image.shape[-2]), int(image.shape[-1])
            mask = getter(int(frame_id), image_size=(height, width)) if callable(getter) else None
            if mask is None and frontend_sky_required:
                raise RuntimeError(
                    f"frame {int(frame_id)}: Mapping.sky_mask_source=panovggt_head requires PanoVGGT sky head mask."
                )
            if mask is None and bool(mapping_cfg.get("sky_mask_enable", False)):
                mask = self.initializer._configured_sky_mask(
                    image,
                    (height, width),
                    device=torch.device("cpu"),
                    insertion_hints=None,
                )
            return None if mask is None else mask.detach().cpu().bool()

        def current_frontend_chunk_frame_ids_full() -> list[int]:
            ids: list[int] = []
            profile = getattr(self.frontend, "last_profile", None)
            if isinstance(profile, dict) and "frame_start" in profile and "frame_end" in profile:
                try:
                    start = int(profile.get("frame_start"))
                    end = int(profile.get("frame_end"))
                    if end >= start:
                        ids.extend(range(start, end + 1))
                except (TypeError, ValueError):
                    ids.clear()
            debug = getattr(self.frontend, "last_m3_debug", None)
            if isinstance(debug, dict):
                for fid in debug.get("frame_ids", ()):
                    if fid is not None and int(fid) not in ids:
                        ids.append(int(fid))
            return [int(fid) for fid in ids]

        def backend_effective_keyframe_flag(out: FrontendOutput) -> bool:
            nonlocal chunk_keyframe_anchor_frame_id
            if not force_chunk_block_insertions:
                return bool(out.is_keyframe)
            ids = current_frontend_chunk_frame_ids_full()
            if chunk_keyframe_anchor_frame_id is None:
                if ids:
                    chunk_keyframe_anchor_frame_id = int(min(ids))
                else:
                    chunk_keyframe_anchor_frame_id = int(out.frame_id)
            rel = int(out.frame_id) - int(chunk_keyframe_anchor_frame_id)
            return bool(rel >= 0 and ((rel + 1) % int(insert_keyframe_block_size) == 0))

        def world_points_from_inverse_depth(
            inverse_depth: torch.Tensor,
            pose_c2w: torch.Tensor,
            image_hw: tuple[int, int],
        ) -> torch.Tensor:
            height, width = int(image_hw[0]), int(image_hw[1])
            inv = inverse_depth.detach().float()
            if inv.ndim == 2:
                inv = inv.unsqueeze(0)
            elif inv.ndim == 3 and int(inv.shape[0]) != 1:
                inv = inv[:1]
            if tuple(inv.shape[-2:]) != (height, width):
                inv = F.interpolate(inv.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
            depth = inv.clamp_min(1.0e-6).reciprocal()
            pixels = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
            bearings = erp_pixel_to_bearing(pixels, height, width).to(device=depth.device, dtype=depth.dtype)
            cam_points = bearings * depth[0].unsqueeze(-1)
            pose = pose_c2w.detach().to(device=depth.device, dtype=depth.dtype)
            if tuple(pose.shape) != (4, 4):
                raise ValueError(f"Expected pose_c2w as 4x4, got {tuple(pose.shape)}")
            return (cam_points.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3].view(1, 3)).reshape(height, width, 3)

        def synthetic_frontend_output_for_seed(
            *,
            frame_id: int,
            source_frame: PanoFrame,
            template: FrontendOutput,
        ) -> FrontendOutput | None:
            fid = int(frame_id)
            pose_by_frame = getattr(self.frontend, "pose_by_frame", {})
            depth_by_frame = getattr(self.frontend, "depth_by_frame", {})
            conf_by_frame = getattr(self.frontend, "conf_by_frame", {})
            pose = template.pose_c2w if fid == int(template.frame_id) else (
                pose_by_frame.get(fid) if isinstance(pose_by_frame, dict) else None
            )
            inv = template.inverse_depth if fid == int(template.frame_id) else (
                depth_by_frame.get(fid) if isinstance(depth_by_frame, dict) else None
            )
            conf = template.depth_confidence if fid == int(template.frame_id) else (
                conf_by_frame.get(fid) if isinstance(conf_by_frame, dict) else None
            )
            if pose is None or inv is None:
                return None
            height, width = int(source_frame.image.shape[-2]), int(source_frame.image.shape[-1])
            if fid == int(template.frame_id) and template.world_points is not None:
                points = template.world_points.detach().cpu().float()
                if points.ndim == 4 and int(points.shape[0]) == 1:
                    points = points[0]
            else:
                points = world_points_from_inverse_depth(inv, pose, (height, width)).detach().cpu().float()
            inv_t = inv.detach().cpu().float()
            if inv_t.ndim == 2:
                inv_t = inv_t.unsqueeze(0)
            elif inv_t.ndim == 3 and int(inv_t.shape[0]) != 1:
                inv_t = inv_t[:1]
            if tuple(inv_t.shape[-2:]) != (height, width):
                inv_t = F.interpolate(inv_t.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
            conf_t = None
            if conf is not None:
                conf_t = conf.detach().cpu().float()
                if conf_t.ndim == 2:
                    conf_t = conf_t.unsqueeze(0)
                elif conf_t.ndim == 3 and int(conf_t.shape[0]) != 1:
                    conf_t = conf_t[:1]
                if tuple(conf_t.shape[-2:]) != (height, width):
                    conf_t = F.interpolate(conf_t.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
            if tuple(points.shape[:2]) != (height, width):
                points = F.interpolate(
                    points.permute(2, 0, 1).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)
            valid = torch.isfinite(points).all(dim=-1, keepdim=False).unsqueeze(0) & torch.isfinite(inv_t) & (inv_t > 0.0)
            if conf_t is not None:
                valid = valid & torch.isfinite(conf_t) & (conf_t > 0.0)
            return FrontendOutput(
                frame_id=fid,
                timestamp=float(source_frame.timestamp),
                pose_c2w=pose.detach().cpu().float(),
                relative_pose=None,
                pose_confidence=float(template.pose_confidence),
                inverse_depth=inv_t.detach().cpu().float(),
                depth_confidence=None if conf_t is None else conf_t.detach().cpu().float(),
                spherical_flow=None,
                keyframe_score=float(template.keyframe_score),
                is_keyframe=fid == int(template.frame_id),
                ba_residual=template.ba_residual,
                tracking_status=template.tracking_status,
                world_points=points.detach().cpu().float(),
                world_points_confidence=None if conf_t is None else conf_t.detach().cpu().float(),
                valid_world_points_mask=valid.detach().cpu().bool(),
            )

        def concatenate_seed_batches(
            batches: list[GaussianSeedBatch],
            *,
            frame_id: int,
        ) -> GaussianSeedBatch:
            nonempty = [batch for batch in batches if len(batch) > 0]
            if not nonempty:
                empty = torch.zeros(0)
                return GaussianSeedBatch(
                    xyz=empty.view(0, 3),
                    rgb=empty.view(0, 3),
                    confidence=empty,
                    scale=empty,
                    level=torch.zeros(0, dtype=torch.int8),
                    frame_id=int(frame_id),
                )
            return GaussianSeedBatch(
                xyz=torch.cat([batch.xyz.detach().cpu() for batch in nonempty], dim=0),
                rgb=torch.cat([batch.rgb.detach().cpu() for batch in nonempty], dim=0),
                confidence=torch.cat([batch.confidence.detach().cpu() for batch in nonempty], dim=0),
                scale=torch.cat([batch.scale.detach().cpu() for batch in nonempty], dim=0),
                level=torch.cat([batch.level.detach().cpu().to(torch.int8) for batch in nonempty], dim=0),
                frame_id=int(frame_id),
                source_flat_idx=None,
                source_hw=None,
                insert_enabled=(
                    torch.cat([batch.insert_enabled.detach().cpu().bool() for batch in nonempty], dim=0)
                    if all(batch.insert_enabled is not None for batch in nonempty)
                    else None
                ),
                insert_score=(
                    torch.cat([batch.insert_score.detach().cpu() for batch in nonempty], dim=0)
                    if all(batch.insert_score is not None for batch in nonempty)
                    else None
                ),
                grid_coord=(
                    torch.cat([batch.grid_coord.detach().cpu().to(torch.int32) for batch in nonempty], dim=0)
                    if all(batch.grid_coord is not None for batch in nonempty)
                    else None
                ),
            )

        def first_chunk_multiframe_seed_batch(
            template: FrontendOutput,
            source_frame: PanoFrame,
        ) -> GaussianSeedBatch | None:
            if not (replace_fuse_enabled and first_chunk_multiframe_init):
                return None
            if int(getattr(self.mapper.stats, "n_keyframes", 0)) != 0:
                return None
            source_frames: dict[int, PanoFrame] = {int(template.frame_id): source_frame}
            source_frames.update({int(fid): frame for fid, frame in frame_cache.items()})
            batches: list[GaussianSeedBatch] = []
            used_ids: list[int] = []
            for fid in current_frontend_chunk_frame_ids_full():
                frame = source_frames.get(int(fid))
                if frame is None:
                    continue
                output = synthetic_frontend_output_for_seed(frame_id=int(fid), source_frame=frame, template=template)
                if output is None:
                    continue
                frame_sky_mask = frontend_sky_mask_for_frame(int(fid), frame.image)
                hints = {"sky_mask": frame_sky_mask.detach().cpu().bool()} if frame_sky_mask is not None else None
                batch = self.initializer.from_frontend_output(
                    output,
                    frame.image,
                    insertion_hints=hints,
                    first_keyframe=True,
                )
                if len(batch) > 0:
                    batches.append(batch)
                    used_ids.append(int(fid))
            if not batches:
                return None
            self.mapper.stats.notes.append(
                f"frame {int(template.frame_id)}: first chunk initialization used frames {used_ids}"
            )
            return concatenate_seed_batches(batches, frame_id=int(template.frame_id))

        def remember_final_frame(out: FrontendOutput, source_frame: PanoFrame, sky_mask: torch.Tensor | None) -> None:
            gt_pose = None
            meta = source_frame.meta or {}
            if meta.get("gt_c2w") is not None:
                gt_pose = torch.as_tensor(meta["gt_c2w"]).detach().cpu().float()
            final_frame_records[int(out.frame_id)] = {
                "image": source_frame.image.detach().cpu().float(),
                "pose_c2w": out.pose_c2w.detach().cpu().float(),
                "gt_c2w": gt_pose,
                "sky_mask": None if sky_mask is None else sky_mask.detach().cpu().bool(),
            }

        def update_frontend_graph_window_hint(out) -> None:
            if not bool(backend_cfg.get("use_frontend_graph_window", False)):
                return
            setter = getattr(self.mapper, "set_frontend_graph_window_ids", None)
            if not callable(setter):
                return
            ids: list[int] = []
            debug = getattr(self.frontend, "last_m3_debug", None)
            if isinstance(debug, dict):
                ids.extend(int(fid) for fid in debug.get("recent_history_ids", ()) if fid is not None)
                alignment = debug.get("alignment")
                if isinstance(alignment, dict):
                    ids.extend(int(fid) for fid in alignment.get("history_ids", ()) if fid is not None)
            decisions = getattr(self.frontend, "keyframe_decision_history", None)
            if isinstance(decisions, list):
                for decision in reversed(decisions):
                    if int(decision.get("frame_id", -1)) != int(out.frame_id):
                        continue
                    ids.extend(int(fid) for fid in decision.get("recent_history_ids", ()) if fid is not None)
                    anchor_id = decision.get("anchor_frame_id")
                    if anchor_id is not None:
                        ids.append(int(anchor_id))
                    break
            ids.append(int(out.frame_id))
            setter(ids)

        def drain_keyframe_decisions(*, step: int | None = None) -> None:
            nonlocal keyframe_decision_count
            if decision_file is None:
                return
            pop_decisions = getattr(self.frontend, "pop_keyframe_decisions", None)
            if not callable(pop_decisions):
                return
            for decision in pop_decisions():
                decision_file.write(json.dumps(decision, sort_keys=True) + "\n")
                decision_file.flush()
                keyframe_decision_count += 1
                logger.observe_keyframe_decision(decision, step=step)

        def drain_frontend_keyframe_graph_pose_updates() -> int:
            pop_updates = getattr(self.frontend, "pop_keyframe_graph_pose_updates", None)
            apply_updates = getattr(self.mapper, "apply_frontend_pose_updates", None)
            if not callable(pop_updates) or not callable(apply_updates):
                return 0
            updates = pop_updates()
            if not updates:
                return 0
            return int(apply_updates(updates))

        def recent_chunk_observation_chunks() -> int:
            return max(
                1,
                int(
                    backend_cfg.get(
                        "recent_chunk_observation_chunks",
                        feedforward_cfg.get("recent_chunk_observation_chunks", 1),
                    )
                ),
            )

        def current_frontend_chunk_index() -> int | None:
            profile = getattr(self.frontend, "last_profile", None)
            if not isinstance(profile, dict) or "chunk_index" not in profile:
                return None
            try:
                return int(profile.get("chunk_index"))
            except (TypeError, ValueError):
                return None

        def remember_recent_feedforward_chunk(current_ids: list[int]) -> list[int]:
            ids: list[int] = []
            for fid in current_ids:
                value = int(fid)
                if value not in ids:
                    ids.append(value)
            if not ids:
                return []
            chunk_key = current_frontend_chunk_index()
            replaced = False
            if chunk_key is not None:
                for idx, (key, _) in enumerate(recent_feedforward_chunks):
                    if key == chunk_key:
                        recent_feedforward_chunks[idx] = (chunk_key, ids)
                        replaced = True
                        break
            if not replaced:
                if chunk_key is None and recent_feedforward_chunks and recent_feedforward_chunks[-1][1] == ids:
                    recent_feedforward_chunks[-1] = (None, ids)
                else:
                    recent_feedforward_chunks.append((chunk_key, ids))
            chunk_limit = recent_chunk_observation_chunks()
            stored_chunk_limit = max(2, chunk_limit)
            if len(recent_feedforward_chunks) > stored_chunk_limit:
                del recent_feedforward_chunks[: len(recent_feedforward_chunks) - stored_chunk_limit]
            out: list[int] = []
            for _, chunk_ids in recent_feedforward_chunks[-chunk_limit:]:
                for fid in chunk_ids:
                    value = int(fid)
                    if value not in out:
                        out.append(value)
            return out

        def feedforward_debug_window(output_ids: list[int]) -> tuple[list[int], list[int]]:
            current_limit = max(
                1,
                int(
                    backend_cfg.get(
                        "current_chunk_observation_frames",
                        feedforward_cfg.get("current_chunk_observation_frames", 4),
                    )
                ),
            )
            history_limit = max(
                0,
                int(
                    backend_cfg.get(
                        "recent_keyframe_observation_frames",
                        feedforward_cfg.get("history_keyframes", 2),
                    )
                ),
            )
            debug = getattr(self.frontend, "last_m3_debug", None)
            current_ids: list[int] = []
            profile = getattr(self.frontend, "last_profile", None)
            if isinstance(profile, dict) and "frame_start" in profile and "frame_end" in profile:
                try:
                    start = int(profile.get("frame_start"))
                    end = int(profile.get("frame_end"))
                    if end >= start:
                        current_ids.extend(range(start, end + 1))
                except (TypeError, ValueError):
                    current_ids.clear()
            if isinstance(debug, dict):
                for fid in debug.get("frame_ids", ()):
                    if fid is not None and int(fid) not in current_ids:
                        current_ids.append(int(fid))
            if not current_ids:
                if output_ids:
                    end = max(int(fid) for fid in output_ids)
                    current_ids.extend(range(max(0, end - current_limit + 1), end + 1))
            current_ids = current_ids[-current_limit:]
            history_ids = [int(kf.frame_id) for kf in self.mapper.keyframes[-history_limit:]] if history_limit > 0 else []
            return current_ids, history_ids

        def recent_backend_scope_frame_ids() -> list[int]:
            chunks: list[list[int]] = []
            for _, chunk_ids in recent_feedforward_chunks[-2:]:
                ids: list[int] = []
                for fid in chunk_ids:
                    value = int(fid)
                    if value not in ids:
                        ids.append(value)
                if ids:
                    chunks.append(ids)
            current_ids: list[int] = []
            for fid in current_frontend_chunk_frame_ids_full():
                value = int(fid)
                if value not in current_ids:
                    current_ids.append(value)
            if current_ids and current_ids not in chunks:
                chunks.append(current_ids)
            chunks = chunks[-2:]
            out: list[int] = []
            for chunk_ids in chunks:
                for fid in chunk_ids:
                    value = int(fid)
                    if value not in out:
                        out.append(value)
            return out

        def recent_chunk_keyframe_ids_for_backend_scope() -> list[int]:
            ids = recent_backend_scope_frame_ids()
            registered = {int(kf.frame_id) for kf in self.mapper.keyframes}
            return [int(fid) for fid in ids if int(fid) in registered]

        def register_cached_feedforward_observations(current_ids: list[int], history_ids: list[int]) -> int:
            pose_by_frame = getattr(self.frontend, "pose_by_frame", {})
            depth_by_frame = getattr(self.frontend, "depth_by_frame", {})
            conf_by_frame = getattr(self.frontend, "conf_by_frame", {})
            registered_keyframes = {int(kf.frame_id) for kf in self.mapper.keyframes}
            count = 0
            for frame_id in [*history_ids, *current_ids]:
                fid = int(frame_id)
                if fid in self.mapper.observations:
                    continue
                source_frame = frame_cache.get(fid)
                if source_frame is None:
                    continue
                pose = pose_by_frame.get(fid) if isinstance(pose_by_frame, dict) else None
                inv = depth_by_frame.get(fid) if isinstance(depth_by_frame, dict) else None
                conf = conf_by_frame.get(fid) if isinstance(conf_by_frame, dict) else None
                if pose is None or inv is None:
                    continue
                sky_mask = frontend_sky_mask_for_frame(fid, source_frame.image)
                self.mapper.register_observation_values(
                    frame_id=fid,
                    image=source_frame.image,
                    c2w=pose,
                    inverse_depth=inv,
                    depth_confidence=conf,
                    is_keyframe=fid in registered_keyframes,
                    sky_mask=sky_mask,
                )
                count += 1
            return count

        def optimize_feedforward_after_batch(outputs: list[FrontendOutput]) -> None:
            nonlocal backend_feedback_decision_count, backend_feedback_applied_count, last_feedforward_metrics
            nonlocal last_optimized_frontend_chunk
            if resplat_direct_fusion_enabled or spherical_selfi_global_enabled:
                return
            if not feedforward_window_enabled or not outputs:
                return
            chunk_index = current_frontend_chunk_index()
            neural_anchor_mode = str(getattr(self.map, "map_mode", "")).lower() == "neural_anchor_scaffold_panorama"
            if neural_anchor_mode and chunk_index is not None and chunk_index == last_optimized_frontend_chunk:
                return
            output_ids = [int(out.frame_id) for out in outputs]
            current_ids, history_ids = feedforward_debug_window(output_ids)
            current_ids = remember_recent_feedforward_chunk(current_ids)
            active_keyframe_ids = recent_chunk_keyframe_ids_for_backend_scope()
            section_start = time.perf_counter()
            registered_count = register_cached_feedforward_observations(current_ids, history_ids)
            register_sec = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            metrics = self.mapper.optimize_feedforward_window(
                current_frame_ids=current_ids,
                history_frame_ids=history_ids,
                chunk_index=chunk_index,
                active_keyframe_ids=active_keyframe_ids,
            )
            optimize_sec = float(time.perf_counter() - section_start)
            if not metrics:
                return
            if neural_anchor_mode and chunk_index is not None and float(metrics.get("steps", 0.0)) > 0.0:
                last_optimized_frontend_chunk = int(chunk_index)
            last_feedforward_metrics = dict(metrics)
            diagnostic_step = max(1, int(logger._step) + 1)
            logger._log_wandb_payload(
                {
                    "backend/sky_pruned": float(metrics.get("sky_pruned", 0.0)),
                    "backend/sky_compacted": float(metrics.get("sky_compacted", 0.0)),
                },
                step=diagnostic_step,
            )
            for out in outputs:
                if not bool(getattr(out, "is_keyframe", False)):
                    continue
                try:
                    diagnostic = self.mapper.render_keyframe_diagnostic(int(out.frame_id))
                    logger.observe_keyframe_opt(diagnostic, step=diagnostic_step)
                except Exception as exc:
                    self.mapper.stats.notes.append(
                        f"frame {int(out.frame_id)}: post-chunk keyframe render failed: {exc!r}"
                    )
            section_start = time.perf_counter()
            feedback_updates, feedback_decisions = self._collect_backend_feedback_updates(metrics)
            feedback_collect_sec = float(time.perf_counter() - section_start)
            for feedback_decision in feedback_decisions:
                if feedback_file is not None:
                    feedback_file.write(json.dumps(feedback_decision, sort_keys=True) + "\n")
                    feedback_file.flush()
                backend_feedback_decision_count += 1
            section_start = time.perf_counter()
            backend_feedback_applied_count += self._apply_backend_feedback_updates(feedback_updates)
            feedback_apply_sec = float(time.perf_counter() - section_start)
            write_profile(
                "backend_feedforward_window",
                current_frame_count=float(len(current_ids)),
                history_frame_count=float(len(history_ids)),
                active_scope_keyframe_count=float(len(active_keyframe_ids)),
                registered_cached_observations=float(registered_count),
                register_sec=register_sec,
                optimize_sec=optimize_sec,
                feedback_collect_sec=feedback_collect_sec,
                feedback_apply_sec=feedback_apply_sec,
                **{
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and str(key).startswith("profile_")
                },
            )

        def drain_resplat_artifacts() -> None:
            nonlocal last_feedforward_metrics, resplat_fusion_count, keyframes
            if not resplat_direct_fusion_enabled:
                return
            consume = getattr(self.frontend, "consume_resplat_artifacts", None)
            if not callable(consume):
                return
            artifacts = consume()
            for artifact in artifacts:
                frame_ids = [int(fid) for fid in getattr(artifact, "frame_ids", ())]
                window_id = int(getattr(artifact, "window_id", resplat_fusion_count))
                state = getattr(artifact, "final_state", None)
                if state is None or not frame_ids:
                    continue
                section_start = time.perf_counter()
                fusion_stats = self.mapper.fuse_resplat_state(
                    state,
                    frame_ids=frame_ids,
                    config=resplat_fusion_cfg,
                    window_id=window_id,
                )
                fusion_sec = float(time.perf_counter() - section_start)
                section_start = time.perf_counter()
                metrics = self.mapper.optimize_resplat_global_window(
                    frame_ids=frame_ids,
                    iters=resplat_global_iters,
                )
                optimize_sec = float(time.perf_counter() - section_start)
                last_feedforward_metrics = dict(metrics)
                resplat_fusion_count += 1
                keyframes = int(getattr(self.mapper.stats, "n_keyframes", keyframes))
                step = max(1, int(logger._step) + 1)
                logger._log_wandb_payload(
                    {
                        "backend/resplat_window_id": int(window_id),
                        "backend/resplat_fused": int(fusion_stats.get("fused", 0)),
                        "backend/resplat_inserted": int(fusion_stats.get("inserted", 0)),
                        "backend/resplat_skipped": int(fusion_stats.get("skipped", 0)),
                        "backend/resplat_global_steps": float(metrics.get("steps", 0.0)),
                        "backend/resplat_global_loss": float(metrics.get("loss", 0.0)),
                    },
                    step=step,
                )
                write_profile(
                    "backend_resplat_direct_fusion",
                    window_id=float(window_id),
                    frame_count=float(len(frame_ids)),
                    frame_start=float(frame_ids[0]),
                    frame_end=float(frame_ids[-1]),
                    fused=float(fusion_stats.get("fused", 0)),
                    inserted=float(fusion_stats.get("inserted", 0)),
                    skipped=float(fusion_stats.get("skipped", 0)),
                    anchors_after=float(fusion_stats.get("anchors_after", self.map.anchor_count())),
                    optimize_steps=float(metrics.get("steps", 0.0)),
                    optimize_loss=float(metrics.get("loss", 0.0)),
                    fusion_sec=fusion_sec,
                    optimize_sec=optimize_sec,
                )

        pending_spherical_selfi_geometry_updates: dict[int, object] = {}
        spherical_selfi_local_inverse_depth: dict[int, torch.Tensor] = {}
        spherical_selfi_global_pose: dict[int, torch.Tensor] = {}
        spherical_selfi_geometry_by_frame: dict[int, object] = {}

        def apply_spherical_selfi_geometry_updates(
            updates: dict[int, object], outputs: list[FrontendOutput]
        ) -> None:
            if not updates:
                return
            self.mapper.apply_frontend_geometry_updates(updates)
            for frame_id, update in updates.items():
                spherical_selfi_geometry_by_frame[int(frame_id)] = update
                spherical_selfi_global_pose[int(frame_id)] = torch.as_tensor(
                    getattr(update, "pose_c2w")
                ).detach().cpu().float()
            ready_ids = {int(output.frame_id) for output in outputs}
            pending_spherical_selfi_geometry_updates.update(
                {
                    int(frame_id): update
                    for frame_id, update in updates.items()
                    if int(frame_id) in ready_ids
                }
            )
            for index, output in enumerate(outputs):
                frame_id = int(output.frame_id)
                update = pending_spherical_selfi_geometry_updates.get(frame_id)
                if update is None:
                    continue
                pose = torch.as_tensor(getattr(update, "pose_c2w")).detach().cpu().float()
                depth_scale = float(getattr(update, "depth_scale"))
                inverse_depth = output.inverse_depth
                if output.inverse_depth is not None:
                    spherical_selfi_local_inverse_depth.setdefault(
                        frame_id, output.inverse_depth.detach().cpu().float().clone()
                    )
                    inverse_depth = spherical_selfi_local_inverse_depth[frame_id] / depth_scale
                world_points = output.world_points
                source = frame_cache.get(frame_id)
                if inverse_depth is not None and source is not None:
                    world_points = world_points_from_inverse_depth(
                        inverse_depth,
                        pose,
                        (int(source.image.shape[-2]), int(source.image.shape[-1])),
                    ).detach().cpu().float()
                relative_pose = output.relative_pose
                previous_pose = spherical_selfi_global_pose.get(frame_id - 1)
                if previous_pose is not None:
                    relative_pose = relative_c2w(previous_pose, pose).detach().cpu().float()
                outputs[index] = replace(
                    output,
                    pose_c2w=pose,
                    relative_pose=relative_pose,
                    inverse_depth=inverse_depth,
                    world_points=world_points,
                )
                pending_spherical_selfi_geometry_updates.pop(frame_id, None)

        def drain_spherical_selfi_windows(outputs: list[FrontendOutput]) -> None:
            if spherical_selfi_global_enabled:
                consume = getattr(self.frontend, "consume_local_gaussian_windows", None)
                if not callable(consume):
                    raise RuntimeError(
                        "SphericalSelfiGlobalBackend is enabled but the frontend does not expose "
                        "consume_local_gaussian_windows()."
                    )
                for packet in consume():
                    section_start = time.perf_counter()
                    try:
                        result = (
                            self.spherical_selfi_global_backend.process_packet(
                                packet
                            )
                        )
                    except Exception as exc:
                        elapsed = float(time.perf_counter() - section_start)
                        try:
                            consume_overlap = getattr(
                                self.spherical_selfi_global_backend,
                                "consume_rendered_overlap_diagnostic",
                                None,
                            )
                            if callable(consume_overlap):
                                logger.observe_rendered_overlap_alignment(
                                    consume_overlap(),
                                    window_id=int(packet.window_id),
                                    step=max(1, int(logger._step) + 1),
                                )
                            logger._log_wandb_payload(
                                {
                                    "backend/selfi_window_id": int(
                                        packet.window_id
                                    ),
                                    "backend/selfi_window_failed": 1,
                                    "backend/selfi_failure": repr(exc),
                                    "backend/selfi_failure_seconds": elapsed,
                                },
                                step=max(1, int(logger._step) + 1),
                            )
                            write_profile(
                                "backend_spherical_selfi_window_failure",
                                window_id=float(packet.window_id),
                                total_sec=elapsed,
                            )
                        except Exception:
                            pass
                        raise
                    elapsed = float(time.perf_counter() - section_start)
                    fusion_profile = {
                        f"fusion_{key}": float(value)
                        for key, value in result.fusion.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                    gpu_memory_mb = (
                        float(torch.cuda.memory_allocated(self.map.get_xyz.device) / (1024.0 * 1024.0))
                        if self.map.get_xyz.device.type == "cuda"
                        else 0.0
                    )
                    gpu_peak_memory_mb = (
                        float(
                            torch.cuda.max_memory_allocated(self.map.get_xyz.device)
                            / (1024.0 * 1024.0)
                        )
                        if self.map.get_xyz.device.type == "cuda"
                        else 0.0
                    )
                    write_profile(
                        "backend_spherical_selfi_window",
                        window_id=float(result.window_id),
                        aligned=float(result.aligned),
                        loops=float(result.loop_accepted),
                        requested=float(result.fusion.get("requested", 0)),
                        window_compacted=float(result.fusion.get("window_compacted", 0)),
                        deduplicated=float(result.fusion.get("deduplicated", 0)),
                        anchors_after=float(result.fusion.get("anchors_after", self.map.anchor_count())),
                        moved=float(result.correction.get("moved", 0)),
                        gpu_memory_mb=gpu_memory_mb,
                        gpu_peak_memory_mb=gpu_peak_memory_mb,
                        total_sec=elapsed,
                        **fusion_profile,
                    )
                    wandb_payload = {
                        "backend/selfi_window_id": int(result.window_id),
                        "backend/selfi_loop_accepted": int(result.loop_accepted),
                        "backend/selfi_window_compacted": int(result.fusion.get("window_compacted", 0)),
                        "backend/selfi_global_anchors": int(result.fusion.get("anchors_after", self.map.anchor_count())),
                        "backend/selfi_graph_objective": 0.0 if result.graph is None else float(result.graph.final_objective),
                        "backend/selfi_graph_initial_objective": 0.0 if result.graph is None else float(result.graph.initial_objective),
                        "backend/selfi_graph_objective_reduction": 0.0 if result.graph is None else float(result.graph.initial_objective - result.graph.final_objective),
                        "backend/selfi_graph_iterations": 0 if result.graph is None else int(result.graph.iterations),
                        "backend/selfi_graph_pcg_iterations": 0 if result.graph is None else int(result.graph.pcg_iterations),
                        "backend/selfi_graph_pcg_relative_residual": 0.0 if result.graph is None else float(result.graph.pcg_relative_residual),
                        "backend/selfi_graph_reason": "none" if result.graph is None else str(result.graph.reason),
                        "backend/selfi_graph_final_damping": 0.0 if result.graph is None else float(result.graph.final_damping),
                        "backend/selfi_graph_gain_ratio": 0.0 if result.graph is None else float(result.graph.gain_ratio),
                        "backend/selfi_graph_rejected_trials": 0 if result.graph is None else int(result.graph.rejected_trials),
                        "backend/selfi_global_ba_scheduled": int(bool(result.diagnostics.get("global_ba_scheduled", False))),
                        "backend/selfi_map_saturated": int(result.fusion.get("map_saturated", 0)),
                        "backend/selfi_gpu_memory_mb": gpu_memory_mb,
                        "backend/selfi_gpu_peak_memory_mb": gpu_peak_memory_mb,
                    }
                    alignment_diag = dict(result.diagnostics.get("alignment", {}) or {})
                    boundary_diag = dict(result.diagnostics.get("boundary_factor", {}) or {})
                    for metric_name, source_name in (
                        ("chunk_scale_normalization", "chunk_scale_normalization"),
                        ("canonical_rotation_mismatch_deg", "canonical_rotation_mismatch_deg"),
                        ("canonical_translation_mismatch", "canonical_translation_mismatch"),
                        ("overlap_residual", "overlap_residual"),
                        ("overlap_inlier_ratio", "overlap_inlier_ratio"),
                        ("shared_scale", "shared_scale"),
                        ("absolute_scale", "absolute_scale"),
                        ("s_shared", "s_shared"),
                        ("s_absolute", "s_absolute"),
                        ("scale_c", "c"),
                        ("rendered_valid_points", "valid_points"),
                        ("rendered_inlier_ratio", "inlier_ratio"),
                        (
                            "rendered_median_relative_error",
                            "median_relative_error",
                        ),
                        ("rendered_p90_relative_error", "p90_relative_error"),
                        ("rendered_render_seconds", "render_seconds"),
                        ("rendered_scale_solve_seconds", "scale_solve_seconds"),
                        ("alignment_seconds", "alignment_seconds"),
                        ("measurement_scale", "measurement_scale"),
                        ("measurement_rotation_deg", "measurement_rotation_deg"),
                        (
                            "measurement_translation_norm",
                            "measurement_translation_norm",
                        ),
                        (
                            "full_sim3_current_covariance_ratio",
                            "full_sim3_current_covariance_ratio",
                        ),
                        (
                            "full_sim3_previous_covariance_ratio",
                            "full_sim3_previous_covariance_ratio",
                        ),
                        (
                            "full_sim3_raw_current_covariance_ratio",
                            "full_sim3_raw_current_covariance_ratio",
                        ),
                        (
                            "full_sim3_raw_previous_covariance_ratio",
                            "full_sim3_raw_previous_covariance_ratio",
                        ),
                        (
                            "full_sim3_train_inlier_ratio",
                            "full_sim3_train_inlier_ratio",
                        ),
                        (
                            "full_sim3_holdout_inlier_ratio",
                            "full_sim3_holdout_inlier_ratio",
                        ),
                        (
                            "full_sim3_holdout_median_residual",
                            "full_sim3_holdout_median_residual",
                        ),
                        (
                            "full_sim3_rotation_correction_deg",
                            "full_sim3_rotation_correction_deg",
                        ),
                        (
                            "full_sim3_translation_correction",
                            "full_sim3_translation_correction",
                        ),
                        (
                            "fallback_train_inlier_ratio",
                            "fallback_train_inlier_ratio",
                        ),
                        (
                            "fallback_holdout_inlier_ratio",
                            "fallback_holdout_inlier_ratio",
                        ),
                    ):
                        value = alignment_diag.get(source_name)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            wandb_payload[f"backend/selfi_{metric_name}"] = float(value)
                    wandb_payload["backend/selfi_alignment_accepted"] = int(
                        bool(alignment_diag.get("accepted", False))
                    )
                    wandb_payload["backend/selfi_full_sim3_accepted"] = int(
                        bool(alignment_diag.get("full_sim3_accepted", False))
                    )
                    wandb_payload["backend/selfi_fallback_accepted"] = int(
                        bool(alignment_diag.get("fallback_accepted", False))
                    )
                    wandb_payload["backend/selfi_alignment_method"] = str(
                        alignment_diag.get("alignment_method", "none")
                    )
                    for source_name in (
                        "full_sim3_current_singular_values",
                        "full_sim3_previous_singular_values",
                        "full_sim3_raw_current_singular_values",
                        "full_sim3_raw_previous_singular_values",
                        "per_frame_valid_points",
                        "per_frame_inlier_ratio",
                        "full_sim3_per_frame_inlier_ratio",
                        "full_sim3_per_frame_median_residual",
                        "full_sim3_shared_rotation_errors_deg",
                        "full_sim3_shared_center_errors",
                        "fallback_per_frame_inlier_ratio",
                        "fallback_shared_rotation_errors_deg",
                        "fallback_shared_center_errors",
                        "pose_frame_scale",
                        "pose_frame_rotation_deg",
                        "pose_frame_translation_norm",
                    ):
                        values = alignment_diag.get(source_name)
                        if not isinstance(values, (list, tuple)):
                            continue
                        for frame_index, value in enumerate(values):
                            if isinstance(value, (int, float)) and not isinstance(
                                value, bool
                            ):
                                wandb_payload[
                                    f"backend/selfi_{source_name}_{frame_index}"
                                ] = float(value)
                    pose_translations = alignment_diag.get(
                        "pose_frame_translation"
                    )
                    if isinstance(pose_translations, (list, tuple)):
                        for frame_index, translation in enumerate(
                            pose_translations
                        ):
                            if not isinstance(translation, (list, tuple)):
                                continue
                            for axis, value in zip("xyz", translation):
                                if isinstance(value, (int, float)) and not isinstance(
                                    value, bool
                                ):
                                    wandb_payload[
                                        "backend/selfi_pose_frame_"
                                        f"translation_{axis}_{frame_index}"
                                    ] = float(value)
                    for metric_name in (
                        "raw_boundary_matches",
                        "hard_gated_boundary_matches",
                        "sky_rejected",
                    ):
                        value = boundary_diag.get(metric_name)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            wandb_payload[f"backend/selfi_{metric_name}"] = float(value)
                    wandb_payload.update(
                        {
                            f"backend/selfi_{key}": float(value)
                            for key, value in result.fusion.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                        }
                    )
                    logger._log_wandb_payload(
                        wandb_payload,
                        step=max(1, int(logger._step) + 1),
                    )
                    consume_overlap = getattr(
                        self.spherical_selfi_global_backend,
                        "consume_rendered_overlap_diagnostic",
                        None,
                    )
                    if callable(consume_overlap):
                        logger.observe_rendered_overlap_alignment(
                            consume_overlap(),
                            window_id=result.window_id,
                            step=max(1, int(logger._step) + 1),
                        )
            consume_ba = getattr(self.frontend, "consume_local_ba_diagnostics", None)
            if callable(consume_ba):
                for diagnostic in consume_ba():
                    ba_diagnostic = diagnostic.get("ba_diagnostics") or {}
                    def finite_optional(value):
                        if value is None:
                            return None
                        scalar = float(value)
                        return scalar if np.isfinite(scalar) else None

                    def finite_sequence(value):
                        if not isinstance(value, (list, tuple)):
                            return []
                        return [
                            float(item)
                            for item in value
                            if isinstance(item, (int, float)) and np.isfinite(float(item))
                        ]

                    gradient_norms = finite_sequence(ba_diagnostic.get("gradient_norms"))
                    pose_step_norms = finite_sequence(ba_diagnostic.get("pose_step_norms"))
                    depth_step_norms = finite_sequence(ba_diagnostic.get("depth_step_norms"))
                    trial_gain_ratios = finite_sequence(ba_diagnostic.get("trial_gain_ratios"))
                    published_pose_twist_norms = finite_sequence(
                        ba_diagnostic.get("published_pose_twist_norms")
                    )
                    published_translation_update_norms = finite_sequence(
                        ba_diagnostic.get("published_translation_update_norms")
                    )
                    published_rotation_update_deg = finite_sequence(
                        ba_diagnostic.get("published_rotation_update_deg")
                    )
                    affine_accepted = ba_diagnostic.get("depth_affine_accepted") or []
                    depth_affine_frames = ba_diagnostic.get("depth_affine_frames") or []

                    record = {
                        "window_id": int(diagnostic["window_id"]),
                        "frame_ids": [int(value) for value in diagnostic["frame_ids"]],
                        "matcher": str(diagnostic["matcher"]),
                        "accepted": bool(diagnostic["accepted"]),
                        "num_factors": int(diagnostic["num_factors"]),
                        "initial_median_residual_deg": diagnostic.get("initial_median_residual_deg"),
                        "final_median_residual_deg": diagnostic.get("final_median_residual_deg"),
                        "matching_sec": float(diagnostic["matching_sec"]),
                        "ba_sec": float(diagnostic["ba_sec"]),
                        "reason": ba_diagnostic.get("reason"),
                        "initial_objective": finite_optional(
                            ba_diagnostic.get("initial_objective")
                        ),
                        "final_objective": finite_optional(
                            ba_diagnostic.get("final_objective")
                        ),
                        "accepted_steps": ba_diagnostic.get("accepted_steps"),
                        "stage1_accepted": bool(
                            ba_diagnostic.get("stage1_accepted", diagnostic["accepted"])
                        ),
                        "stage1_reason": ba_diagnostic.get("stage1_reason"),
                        "stage2_attempted": bool(
                            ba_diagnostic.get("stage2_attempted", False)
                        ),
                        "stage2_accepted": bool(
                            ba_diagnostic.get("stage2_accepted", False)
                        ),
                        "stage2_reason": ba_diagnostic.get("stage2_reason"),
                        "pre_filter_factors": int(
                            ba_diagnostic.get("pre_filter_factors", diagnostic["num_factors"])
                        ),
                        "angular_inliers": int(
                            ba_diagnostic.get("angular_inliers", diagnostic["num_factors"])
                        ),
                        "angular_outliers": int(
                            ba_diagnostic.get("angular_outliers", 0)
                        ),
                        "sim3_candidates": int(
                            ba_diagnostic.get("sim3_candidates", diagnostic["num_factors"])
                        ),
                        "sim3_outliers": int(
                            ba_diagnostic.get("sim3_outliers", 0)
                        ),
                        "post_filter_inliers": int(
                            ba_diagnostic.get("post_filter_inliers", diagnostic["num_factors"])
                        ),
                        "post_filter_inlier_ratio": finite_optional(
                            ba_diagnostic.get("post_filter_inlier_ratio")
                        ),
                        "jacobian_mode": ba_diagnostic.get("jacobian_mode"),
                        "max_factor_jacobian_norm": finite_optional(
                            ba_diagnostic.get("max_factor_jacobian_norm")
                        ),
                        "analytic_autodiff_max_abs": finite_optional(
                            ba_diagnostic.get("analytic_autodiff_max_abs")
                        ),
                        "final_damping": finite_optional(ba_diagnostic.get("final_damping")),
                        "gradient_norms": gradient_norms,
                        "pose_step_norms": pose_step_norms,
                        "depth_step_norms": depth_step_norms,
                        "trial_gain_ratios": trial_gain_ratios,
                        "published_pose_updated": bool(
                            ba_diagnostic.get("published_pose_updated", False)
                        ),
                        "published_pose_twist_norms": published_pose_twist_norms,
                        "published_translation_update_norms": (
                            published_translation_update_norms
                        ),
                        "published_rotation_update_deg": published_rotation_update_deg,
                        "depth_affine_accepted": [bool(value) for value in affine_accepted],
                        "depth_affine_frames": [
                            dict(value)
                            for value in depth_affine_frames
                            if isinstance(value, dict)
                        ],
                        "depth_parameterization": ba_diagnostic.get(
                            "depth_parameterization"
                        ),
                        "normalized_depth_shift": finite_sequence(
                            ba_diagnostic.get("normalized_depth_shift")
                        ),
                        "depth_shift": finite_sequence(
                            ba_diagnostic.get("depth_shift")
                        ),
                        "validation_passed": bool(
                            ba_diagnostic.get("validation_passed", False)
                        ),
                        "validation_support_ok": bool(
                            ba_diagnostic.get("validation_support_ok", False)
                        ),
                        "validation_angular_ok": bool(
                            ba_diagnostic.get("validation_angular_ok", False)
                        ),
                        "validation_sim3_ok": bool(
                            ba_diagnostic.get("validation_sim3_ok", False)
                        ),
                        "validation_initial_median_deg": finite_optional(
                            ba_diagnostic.get("validation_initial_median_deg")
                        ),
                        "validation_final_median_deg": finite_optional(
                            ba_diagnostic.get("validation_final_median_deg")
                        ),
                    }
                    local_ba_window_records.append(record)
                    local_ba_payload = {
                        "local_ba/window_id": record["window_id"],
                        "local_ba/accepted": int(record["accepted"]),
                        "local_ba/valid_factors": record["num_factors"],
                        "local_ba/matching_sec": record["matching_sec"],
                        "local_ba/ba_sec": record["ba_sec"],
                        "local_ba/accepted_steps": int(record["accepted_steps"] or 0),
                        "local_ba/stage1_accepted": int(record["stage1_accepted"]),
                        "local_ba/stage2_attempted": int(record["stage2_attempted"]),
                        "local_ba/stage2_accepted": int(record["stage2_accepted"]),
                        "local_ba/pre_filter_factors": record["pre_filter_factors"],
                        "local_ba/angular_inliers": record["angular_inliers"],
                        "local_ba/angular_outliers": record["angular_outliers"],
                        "local_ba/sim3_candidates": record["sim3_candidates"],
                        "local_ba/sim3_outliers": record["sim3_outliers"],
                        "local_ba/post_filter_inliers": record["post_filter_inliers"],
                        "local_ba/affine_accepted_frames": int(
                            sum(record["depth_affine_accepted"])
                        ),
                        "local_ba/published_pose_updated": int(
                            record["published_pose_updated"]
                        ),
                        "local_ba/validation_passed": int(
                            record["validation_passed"]
                        ),
                    }
                    optional_scalars = {
                        "local_ba/max_factor_jacobian_norm": record["max_factor_jacobian_norm"],
                        "local_ba/analytic_autodiff_max_abs": record["analytic_autodiff_max_abs"],
                        "local_ba/final_damping": record["final_damping"],
                        "local_ba/post_filter_inlier_ratio": record[
                            "post_filter_inlier_ratio"
                        ],
                        "local_ba/gradient_norm": gradient_norms[-1] if gradient_norms else None,
                        "local_ba/pose_step_norm": max(pose_step_norms) if pose_step_norms else None,
                        "local_ba/depth_step_norm": max(depth_step_norms) if depth_step_norms else None,
                        "local_ba/lm_gain_ratio": (
                            float(np.mean(trial_gain_ratios)) if trial_gain_ratios else None
                        ),
                        "local_ba/published_pose_twist_norm_max": (
                            max(published_pose_twist_norms[1:])
                            if len(published_pose_twist_norms) > 1
                            else None
                        ),
                        "local_ba/published_translation_update_max": (
                            max(published_translation_update_norms[1:])
                            if len(published_translation_update_norms) > 1
                            else None
                        ),
                        "local_ba/published_rotation_update_deg_max": (
                            max(published_rotation_update_deg[1:])
                            if len(published_rotation_update_deg) > 1
                            else None
                        ),
                        "local_ba/validation_initial_median_deg": record[
                            "validation_initial_median_deg"
                        ],
                        "local_ba/validation_final_median_deg": record[
                            "validation_final_median_deg"
                        ],
                    }
                    local_ba_payload.update(
                        {key: value for key, value in optional_scalars.items() if value is not None}
                    )
                    if record["initial_median_residual_deg"] is not None:
                        local_ba_payload["local_ba/initial_residual_deg"] = float(
                            record["initial_median_residual_deg"]
                        )
                    if record["final_median_residual_deg"] is not None:
                        local_ba_payload["local_ba/final_residual_deg"] = float(
                            record["final_median_residual_deg"]
                        )
                    logger._log_wandb_payload(
                        local_ba_payload,
                        step=max(1, int(logger._step) + 1),
                    )
            if spherical_selfi_global_enabled:
                graph_geometry_updates = (
                    self.spherical_selfi_global_backend.pop_frame_geometry_updates()
                )
                # Historical observations must follow graph-loop corrections too;
                # replacing only the not-yet-emitted FrontendOutput would leave the
                # photometric replay cameras at stale poses.
                apply_spherical_selfi_geometry_updates(graph_geometry_updates, outputs)

        def optimize_spherical_selfi_windows(outputs: list[FrontendOutput]) -> None:
            nonlocal last_feedforward_metrics
            if not spherical_selfi_global_enabled:
                return
            section_start = time.perf_counter()
            metrics = self.spherical_selfi_global_backend.run_pending_map_optimization()
            if not metrics:
                return
            joint_geometry_updates = self.spherical_selfi_global_backend.pop_frame_geometry_updates()
            apply_spherical_selfi_geometry_updates(joint_geometry_updates, outputs)
            last_feedforward_metrics = dict(metrics)
            successful_steps = float(metrics.get("steps", 0.0))
            rolled_back = float(metrics.get("window_rollback", 0.0)) > 0.0
            if logger.post_opt_all_frames and successful_steps > 0.0 and not rolled_back:
                window_id = int(metrics.get("spherical_selfi_window_id", -1))
                diagnostics = []
                for frame_id in self.mapper.stats.last_window_observations:
                    try:
                        diagnostic = self.mapper.render_keyframe_diagnostic(int(frame_id))
                    except Exception as exc:
                        self.mapper.stats.notes.append(
                            f"window {window_id} frame {int(frame_id)}: post-opt render failed: {exc!r}"
                        )
                        continue
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                logger.observe_post_optimized_window(
                    diagnostics,
                    window_id=window_id,
                    step=max(1, int(logger._step)),
                )
            write_profile(
                "backend_spherical_selfi_map_optimization",
                optimize_steps=float(metrics.get("steps", 0.0)),
                optimize_loss=float(metrics.get("loss", 0.0)),
                total_sec=float(time.perf_counter() - section_start),
            )

        def process_output(out) -> None:
            nonlocal keyframes, last_status, backend_feedback_decision_count, backend_feedback_applied_count
            nonlocal last_profiled_frontend_chunk
            process_start = time.perf_counter()
            output_profile: dict[str, float | int] = {
                "frame_id": int(out.frame_id),
                "is_keyframe": int(bool(out.is_keyframe)),
            }
            metrics: dict = {}
            last_status = out.tracking_status
            source_frame = frame_cache.get(int(out.frame_id))
            if source_frame is None:
                self.mapper.stats.notes.append(f"frame {out.frame_id}: missing source frame for frontend output")
                output_profile["missing_source_frame"] = 1
                output_profile["total_sec"] = float(time.perf_counter() - process_start)
                write_profile("process_output", **output_profile)
                return
            backend_image = source_frame.image
            if resplat_direct_fusion_enabled:
                image_for_frame = getattr(self.frontend, "image_for_frame", None)
                cached_image = image_for_frame(int(out.frame_id)) if callable(image_for_frame) else None
                if torch.is_tensor(cached_image):
                    backend_image = cached_image
                    output_profile["resplat_backend_image_override"] = 1
            frontend_is_keyframe = bool(out.is_keyframe)
            effective_is_keyframe = backend_effective_keyframe_flag(out)
            output_profile["frontend_is_keyframe"] = int(frontend_is_keyframe)
            output_profile["effective_is_keyframe"] = int(effective_is_keyframe)
            if force_chunk_block_insertions:
                output_profile["insert_keyframe_policy_active"] = 1
                output_profile["insert_keyframe_block_size"] = int(insert_keyframe_block_size)
            if effective_is_keyframe != frontend_is_keyframe:
                out = replace(out, is_keyframe=bool(effective_is_keyframe))
            output_profile["is_keyframe"] = int(bool(out.is_keyframe))
            sky_mask = frontend_sky_mask_for_frame(int(out.frame_id), backend_image)
            remember_final_frame(out, replace(source_frame, image=backend_image), sky_mask)
            if (feedforward_window_enabled or resplat_direct_fusion_enabled or spherical_selfi_global_enabled) and out.inverse_depth is not None:
                section_start = time.perf_counter()
                self.mapper.register_observation(
                    out,
                    backend_image,
                    is_keyframe=bool(out.is_keyframe),
                    sky_mask=sky_mask,
                )
                if spherical_selfi_global_enabled:
                    geometry_update = spherical_selfi_geometry_by_frame.get(int(out.frame_id))
                    local_inverse = spherical_selfi_local_inverse_depth.get(int(out.frame_id))
                    if geometry_update is not None and local_inverse is not None:
                        self.mapper.set_spherical_selfi_observation_geometry(
                            int(out.frame_id),
                            target_depth_local=local_inverse.clamp_min(1.0e-6).reciprocal(),
                            depth_scale=float(getattr(geometry_update, "depth_scale")),
                            owner_window_id=int(
                                getattr(
                                    geometry_update,
                                    "depth_owner_window_id",
                                    getattr(geometry_update, "owner_window_id"),
                                )
                            ),
                            depth_confidence=out.depth_confidence,
                            sky_mask=sky_mask,
                        )
                output_profile["mapper_register_observation_sec"] = float(time.perf_counter() - section_start)
            output_wandb_step = max(1, int(logger._step) + 1)
            logger.observe_sky_mask(frame_id=int(out.frame_id), sky_mask=sky_mask, step=output_wandb_step)
            frontend_profile = getattr(self.frontend, "last_profile", None)
            if isinstance(frontend_profile, dict):
                chunk_index = int(frontend_profile.get("chunk_index", -1))
                if chunk_index >= 0 and chunk_index != last_profiled_frontend_chunk:
                    last_profiled_frontend_chunk = chunk_index
                    chunk_profile: dict[str, float | int] = {}
                    for key, value in frontend_profile.items():
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            chunk_profile[str(key)] = float(value)
                    write_profile("frontend_chunk", **chunk_profile)
            backend_loss = None
            keyframe_opt_diagnostic = None
            neural_anchor_mode = str(getattr(self.map, "map_mode", "")).lower() == "neural_anchor_scaffold_panorama"
            if (
                not resplat_direct_fusion_enabled
                and not spherical_selfi_global_enabled
                and out.is_keyframe
                and (out.inverse_depth is not None or (neural_anchor_mode and out.world_points is not None))
            ):
                section_start = time.perf_counter()
                first_chunk_multiframe_used = False
                if neural_anchor_mode:
                    empty = torch.zeros(0, device=source_frame.image.device, dtype=source_frame.image.dtype)
                    seeds = GaussianSeedBatch(
                        xyz=empty.view(0, 3),
                        rgb=empty.view(0, 3),
                        confidence=empty,
                        scale=empty,
                        level=torch.zeros(0, dtype=torch.long, device=source_frame.image.device),
                        frame_id=int(out.frame_id),
                        insert_score=empty,
                        grid_coord=torch.zeros(0, 3, dtype=torch.int32, device=source_frame.image.device),
                    )
                else:
                    seeds = first_chunk_multiframe_seed_batch(out, source_frame)
                    first_chunk_multiframe_used = seeds is not None
                    if seeds is None:
                        consume_hints = getattr(self.frontend, "consume_insertion_hints", None)
                        insertion_hints = consume_hints(int(out.frame_id)) if callable(consume_hints) else None
                        if sky_mask is not None:
                            insertion_hints = dict(insertion_hints or {})
                            insertion_hints["sky_mask"] = sky_mask.detach().cpu().bool()
                        seeds = self.initializer.from_frontend_output(
                            out,
                            source_frame.image,
                            insertion_hints=insertion_hints,
                            first_keyframe=int(getattr(self.mapper.stats, "n_keyframes", 0)) == 0,
                        )
                output_profile["seed_init_sec"] = float(time.perf_counter() - section_start)
                output_profile["first_chunk_multiframe_init"] = int(first_chunk_multiframe_used)
                output_profile["seed_candidates"] = int(len(seeds))
                replace_delete_keyframe_ids = recent_chunk_keyframe_ids_for_backend_scope()
                output_profile["replace_delete_scope_keyframes"] = int(len(replace_delete_keyframe_ids))
                section_start = time.perf_counter()
                if self.mapper.uses_joint_optimization or neural_anchor_mode:
                    inserted_count = self.mapper.insert_keyframe(
                        seeds,
                        out,
                        image=source_frame.image,
                        sky_mask=sky_mask,
                        insert_occupancy_radius_voxels_override=0.0 if first_chunk_multiframe_used else None,
                        compact_after_insert=bool(first_chunk_multiframe_used),
                        replace_delete_keyframe_ids=replace_delete_keyframe_ids,
                    )
                else:
                    inserted_count = self.mapper.insert_keyframe(seeds, out)
                output_profile["mapper_insert_keyframe_sec"] = float(time.perf_counter() - section_start)
                output_profile["inserted_gaussians"] = float(inserted_count)
                logger.observe_keyframe_inserted_gaussians(
                    frame_id=int(out.frame_id),
                    inserted_count=int(inserted_count),
                    step=output_wandb_step,
                )
                section_start = time.perf_counter()
                output_profile["frontend_keyframe_graph_pose_updates"] = float(
                    drain_frontend_keyframe_graph_pose_updates()
                )
                output_profile["frontend_keyframe_graph_pose_update_sec"] = float(time.perf_counter() - section_start)
                output_profile["missing_seed_candidates"] = int(
                    getattr(self.mapper.stats, "last_missing_seed_candidates", 0)
                )
                output_profile["depth_mismatch_seed_candidates"] = int(
                    getattr(self.mapper.stats, "last_depth_mismatch_seed_candidates", 0)
                )
                output_profile["skipped_missing_budget"] = int(
                    getattr(self.mapper.stats, "last_skipped_missing_budget", 0)
                )
                output_profile["skipped_depth_mismatch_budget"] = int(
                    getattr(self.mapper.stats, "last_skipped_depth_mismatch_budget", 0)
                )
                output_profile["dense_seed_candidates"] = int(
                    getattr(self.mapper.stats, "last_dense_seed_candidates", 0)
                )
                output_profile["insert_mask_seed_candidates"] = int(
                    getattr(self.mapper.stats, "last_insert_mask_seed_candidates", 0)
                )
                output_profile["voxel_seed_candidates"] = int(
                    getattr(self.mapper.stats, "last_voxel_seed_candidates", 0)
                )
                output_profile["replace_fused_existing"] = int(
                    getattr(self.mapper.stats, "last_replace_fused_existing", 0)
                )
                output_profile["replace_fused_new_duplicate"] = int(
                    getattr(self.mapper.stats, "last_replace_fused_new_duplicate", 0)
                )
                output_profile["replace_newly_inserted"] = int(
                    getattr(self.mapper.stats, "last_replace_newly_inserted", 0)
                )
                output_profile["replace_deleted"] = int(
                    getattr(self.mapper.stats, "last_replace_deleted", 0)
                )
                output_profile["replace_compacted"] = int(
                    getattr(self.mapper.stats, "last_replace_compacted", 0)
                )
                output_profile["pred_depth_generated_seeds"] = int(
                    getattr(self.mapper.stats, "last_pred_depth_generated_seeds", 0)
                )
                output_profile["pred_depth_invalid_pixels"] = int(
                    getattr(self.mapper.stats, "last_pred_depth_invalid_pixels", 0)
                )
                output_profile["insert_mask_pixels"] = int(
                    getattr(self.mapper.stats, "last_insert_mask_pixels", 0)
                )
                output_profile["anchor_count_before_insert"] = int(
                    getattr(self.mapper.stats, "last_anchor_count_before_insert", 0)
                )
                output_profile["anchor_count_after_insert"] = int(
                    getattr(self.mapper.stats, "last_anchor_count_after_insert", 0)
                )
                output_profile["neural_insert_total_sec"] = float(
                    getattr(self.mapper.stats, "last_neural_insert_total_sec", 0.0)
                )
                output_profile["neural_insert_accept_sec"] = float(
                    getattr(self.mapper.stats, "last_neural_insert_accept_sec", 0.0)
                )
                output_profile["neural_insert_append_sec"] = float(
                    getattr(self.mapper.stats, "last_neural_insert_append_sec", 0.0)
                )
                output_profile["neural_insert_compact_sec"] = float(
                    getattr(self.mapper.stats, "last_neural_insert_compact_sec", 0.0)
                )
                keyframes += 1
                novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
                insertion_stats = {
                    "kept": int(inserted_count),
                    "skipped_voxel": int(getattr(self.mapper.stats, "last_skipped_voxel", 0)),
                    "skipped_budget": int(getattr(self.mapper.stats, "last_skipped_budget", 0)),
                    "render_missing_pixels": int(getattr(self.mapper.stats, "last_render_missing_pixels", 0)),
                    "render_depth_mismatch_pixels": int(
                        getattr(self.mapper.stats, "last_render_depth_mismatch_pixels", 0)
                    ),
                    "render_bad_pixels": int(getattr(self.mapper.stats, "last_render_bad_pixels", 0)),
                    "missing_seed_candidates": int(
                        getattr(self.mapper.stats, "last_missing_seed_candidates", 0)
                    ),
                    "depth_mismatch_seed_candidates": int(
                        getattr(self.mapper.stats, "last_depth_mismatch_seed_candidates", 0)
                    ),
                    "skipped_missing_budget": int(
                        getattr(self.mapper.stats, "last_skipped_missing_budget", 0)
                    ),
                    "skipped_depth_mismatch_budget": int(
                        getattr(self.mapper.stats, "last_skipped_depth_mismatch_budget", 0)
                    ),
                    "replace_deleted": int(getattr(self.mapper.stats, "last_replace_deleted", 0)),
                    "replace_fused": int(getattr(self.mapper.stats, "last_replace_fused", 0)),
                    "replace_compacted": int(getattr(self.mapper.stats, "last_replace_compacted", 0)),
                    "dense_seed_candidates": int(getattr(self.mapper.stats, "last_dense_seed_candidates", 0)),
                    "insert_mask_seed_candidates": int(
                        getattr(self.mapper.stats, "last_insert_mask_seed_candidates", 0)
                    ),
                    "voxel_seed_candidates": int(getattr(self.mapper.stats, "last_voxel_seed_candidates", 0)),
                    "replace_fused_existing": int(
                        getattr(self.mapper.stats, "last_replace_fused_existing", 0)
                    ),
                    "replace_fused_new_duplicate": int(
                        getattr(self.mapper.stats, "last_replace_fused_new_duplicate", 0)
                    ),
                    "replace_newly_inserted": int(
                        getattr(self.mapper.stats, "last_replace_newly_inserted", 0)
                    ),
                    "pred_depth_generated_seeds": int(
                        getattr(self.mapper.stats, "last_pred_depth_generated_seeds", 0)
                    ),
                    "pred_depth_invalid_pixels": int(
                        getattr(self.mapper.stats, "last_pred_depth_invalid_pixels", 0)
                    ),
                    "insert_mask_pixels": int(
                        getattr(self.mapper.stats, "last_insert_mask_pixels", 0)
                    ),
                    "anchor_count_before_insert": int(
                        getattr(self.mapper.stats, "last_anchor_count_before_insert", 0)
                    ),
                    "anchor_count_after_insert": int(
                        getattr(self.mapper.stats, "last_anchor_count_after_insert", 0)
                    ),
                }
                save_new_gaussian_vis = bool(novel_cfg.get("save_visualization", False)) or (
                    bool(getattr(self.mapper, "pfgs360_replace_fuse_enabled", False))
                    and bool(novel_cfg.get("save_depth_insertion_visualization", False))
                )
                if save_new_gaussian_vis:
                    section_start = time.perf_counter()
                    logger.observe_new_gaussians(
                        frame_id=int(out.frame_id),
                        image=source_frame.image,
                        source_hw=getattr(self.mapper, "last_source_hw", None),
                        requested_idx=getattr(self.mapper, "last_requested_source_flat_idx", None),
                        inserted_idx=getattr(self.mapper, "last_inserted_source_flat_idx", None),
                        stats=insertion_stats,
                        step=output_wandb_step,
                    )
                    output_profile["new_gaussian_visualization_sec"] = float(time.perf_counter() - section_start)
                if bool(novel_cfg.get("save_depth_insertion_visualization", False)):
                    section_start = time.perf_counter()
                    logger.observe_depth_insertion_diagnostic(
                        frame_id=int(out.frame_id),
                        image=source_frame.image,
                        source_hw=getattr(self.mapper, "last_source_hw", None),
                        inserted_idx=getattr(self.mapper, "last_inserted_source_flat_idx", None),
                        diagnostic=getattr(self.mapper, "last_depth_insertion_diagnostic", None),
                        stats=insertion_stats,
                        step=output_wandb_step,
                    )
                    output_profile["depth_insertion_visualization_sec"] = float(time.perf_counter() - section_start)
                if self.mapper.uses_joint_optimization:
                    if bootstrap_enabled and keyframes == 1 and bootstrap_steps > 0:
                        init_dir = output_dir / "init_vis" / f"frame_{int(out.frame_id):06d}"
                        init_dir.mkdir(parents=True, exist_ok=True)

                        def save_bootstrap(step: int, diagnostic) -> None:
                            render_panel = logger._make_keyframe_opt_render_panel(diagnostic)
                            depth_panel = logger._make_keyframe_opt_depth_panel(diagnostic)
                            render_path = init_dir / f"iter_{int(step):04d}_render.png"
                            depth_path = init_dir / f"iter_{int(step):04d}_depth.png"
                            render_panel.save(render_path)
                            depth_panel.save(depth_path)
                            if logger.run is not None and logger._wandb is not None:
                                logger._log_wandb_payload(
                                    {
                                        "backend/bootstrap_render": logger._wandb.Image(str(render_path)),
                                        "backend/bootstrap_depth": logger._wandb.Image(str(depth_path)),
                                        "backend/bootstrap_step": int(step),
                                    },
                                    step=output_wandb_step,
                                )

                        section_start = time.perf_counter()
                        metrics = self.mapper.bootstrap_latest_keyframe(
                            steps=bootstrap_steps,
                            diagnostic_callback=save_bootstrap,
                            diagnostic_every=bootstrap_save_every,
                        )
                        output_profile["backend_bootstrap_call_sec"] = float(time.perf_counter() - section_start)
                        try:
                            section_start = time.perf_counter()
                            init_ply = output_dir / "point_cloud" / "init" / f"frame_{int(out.frame_id):06d}.ply"
                            self.map.save_ply(init_ply)
                            logger.log_artifact_file(init_ply)
                            init_alias = output_dir / "point_cloud" / "init" / "point_cloud.ply"
                            self.map.save_ply(init_alias)
                            logger.log_artifact_file(init_alias)
                            output_profile["save_init_ply_sec"] = float(time.perf_counter() - section_start)
                        except Exception as exc:
                            self.mapper.stats.notes.append(f"frame {out.frame_id}: init ply save failed: {exc!r}")
                        if bool(results_cfg.get("save_skybox_previews", False)) and self.map.has_skybox:
                            try:
                                section_start = time.perf_counter()
                                sky_dir = output_dir / "skybox"
                                sky_dir.mkdir(parents=True, exist_ok=True)
                                preview = self.map.skybox_erp_preview(
                                    height=int(results_cfg.get("skybox_preview_height", 256)),
                                    width=int(results_cfg.get("skybox_preview_width", 512)),
                                )
                                if preview is not None:
                                    preview_path = sky_dir / f"init_frame_{int(out.frame_id):06d}_erp.png"
                                    _image_tensor_to_pil(preview).save(preview_path)
                                    logger.log_image_file("skybox/init_erp_preview", preview_path, step=output_wandb_step)
                                output_profile["save_init_skybox_preview_sec"] = float(time.perf_counter() - section_start)
                            except Exception as exc:
                                self.mapper.stats.notes.append(
                                    f"frame {out.frame_id}: init skybox preview save failed: {exc!r}"
                        )
                    elif not feedforward_window_enabled:
                        update_frontend_graph_window_hint(out)
                        section_start = time.perf_counter()
                        metrics = self.mapper.optimize_after_keyframe()
                        output_profile["backend_optimize_after_keyframe_call_sec"] = float(time.perf_counter() - section_start)
                    backend_loss = metrics.get("loss")
                    if not feedforward_window_enabled:
                        section_start = time.perf_counter()
                        feedback_updates, feedback_decisions = self._collect_backend_feedback_updates(metrics)
                        output_profile["backend_feedback_collect_sec"] = float(time.perf_counter() - section_start)
                        for feedback_decision in feedback_decisions:
                            if feedback_file is not None:
                                feedback_file.write(json.dumps(feedback_decision, sort_keys=True) + "\n")
                                feedback_file.flush()
                            backend_feedback_decision_count += 1
                        section_start = time.perf_counter()
                        backend_feedback_applied_count += self._apply_backend_feedback_updates(feedback_updates)
                        output_profile["backend_feedback_apply_sec"] = float(time.perf_counter() - section_start)
                        try:
                            section_start = time.perf_counter()
                            keyframe_opt_diagnostic = self.mapper.render_keyframe_diagnostic(int(out.frame_id))
                            output_profile["backend_keyframe_diagnostic_sec"] = float(time.perf_counter() - section_start)
                        except Exception as exc:
                            self.mapper.stats.notes.append(
                                f"frame {out.frame_id}: keyframe optimized render failed: {exc!r}"
                            )
                elif refine_steps > 0:
                    section_start = time.perf_counter()
                    metrics = self.mapper.refine_on_keyframe(
                        image=source_frame.image,
                        c2w=out.pose_c2w,
                        steps=refine_steps,
                        sky_mask=sky_mask,
                    )
                    output_profile["backend_refine_keyframe_call_sec"] = float(time.perf_counter() - section_start)
                    backend_loss = metrics.get("loss")
            elif (
                self.mapper.uses_joint_optimization
                and non_keyframe_steps > 0
                and not feedforward_window_enabled
                and not resplat_direct_fusion_enabled
                and not spherical_selfi_global_enabled
            ):
                section_start = time.perf_counter()
                metrics = self.mapper.optimize_frame_observation(
                    image=source_frame.image,
                    c2w=out.pose_c2w,
                    steps=non_keyframe_steps,
                    phase="non_keyframe",
                )
                output_profile["backend_non_keyframe_call_sec"] = float(time.perf_counter() - section_start)
                backend_loss = metrics.get("loss")
            for key, value in metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and str(key).startswith("profile_"):
                    output_profile[str(key)] = float(value)
            backend_pose = self.mapper.refined_pose_c2w(int(out.frame_id))
            render_pose = backend_pose if backend_pose is not None else out.pose_c2w.detach().cpu()
            backend_render_pkg = None
            if self.map.anchor_count() > 0:
                try:
                    section_start = time.perf_counter()
                    backend_render_pkg = self.mapper.render_view(
                        image=source_frame.image,
                        c2w=render_pose,
                        sky_mask=sky_mask,
                    )
                    output_profile["backend_render_view_sec"] = float(time.perf_counter() - section_start)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"frame {out.frame_id}: backend visualization render failed: {exc!r}")
            section_start = time.perf_counter()
            logger.observe(
                out,
                source_frame,
                anchor_count=self.map.anchor_count(),
                keyframe_count=keyframes,
                backend_loss=backend_loss,
                backend_pose_c2w=backend_pose,
                backend_render_pkg=backend_render_pkg,
                m3_debug=getattr(self.frontend, "last_m3_debug", None),
            )
            output_profile["logger_observe_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            logger.observe_keyframe_opt(keyframe_opt_diagnostic, step=output_wandb_step)
            output_profile["logger_keyframe_opt_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            drain_keyframe_decisions(step=output_wandb_step)
            output_profile["drain_keyframe_decisions_sec"] = float(time.perf_counter() - section_start)
            output_profile["total_sec"] = float(time.perf_counter() - process_start)
            write_profile("process_output", **output_profile)

        def save_final_all_frame_renders() -> dict | None:
            enabled = bool(
                results_cfg.get(
                    "render_final_all_frames",
                    results_cfg.get("save_final_all_frame_renders", False),
                )
            )
            if not enabled:
                return None
            root = output_dir / "final_all_frames"
            panel_dir = root / "render_vs_gt"
            panel_dir.mkdir(parents=True, exist_ok=True)
            records = final_frame_records
            if not records:
                metrics = {"render_count": 0, "mean_psnr": None, "ate_rmse": None, "ate_count": 0}
                with open(root / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                return {"root": str(root), "metrics": metrics}

            per_frame: list[dict] = []
            psnrs: list[float] = []
            pred_xyz: list[np.ndarray] = []
            gt_xyz: list[np.ndarray] = []
            trajectory_frame_ids: list[int] = []
            predicted_poses: list[torch.Tensor] = []
            target_poses: list[torch.Tensor] = []
            pose_by_frame: dict[int, torch.Tensor] = {}
            for frame_id in sorted(records):
                rec = records[int(frame_id)]
                pose = self.mapper.refined_pose_c2w(int(frame_id))
                if pose is None:
                    pose = rec["pose_c2w"]
                pose = pose.detach().cpu().float()
                pose_by_frame[int(frame_id)] = pose
                gt_pose = rec.get("gt_c2w")
                if (
                    torch.is_tensor(gt_pose)
                    and tuple(gt_pose.shape) == (4, 4)
                    and tuple(pose.shape) == (4, 4)
                ):
                    trajectory_frame_ids.append(int(frame_id))
                    predicted_poses.append(pose)
                    target_poses.append(gt_pose.detach().cpu().float())
                    pred_xyz.append(pose[:3, 3].numpy())
                    gt_xyz.append(gt_pose.detach().cpu().float()[:3, 3].numpy())
            for frame_id in sorted(records):
                rec = records[int(frame_id)]
                image = rec["image"]
                pose = pose_by_frame[int(frame_id)]
                sky_mask = rec.get("sky_mask")
                try:
                    pkg = self.mapper.render_view(image=image, c2w=pose, sky_mask=sky_mask)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"frame {int(frame_id)}: final all-frame render failed: {exc!r}")
                    continue
                if pkg is None or not torch.is_tensor(pkg.get("render")):
                    continue
                render = pkg["render"].detach().cpu().float().clamp(0.0, 1.0)
                target = image.detach().cpu().float().clamp(0.0, 1.0)
                if tuple(render.shape[-2:]) != tuple(target.shape[-2:]):
                    render = F.interpolate(
                        render.unsqueeze(0),
                        size=tuple(int(v) for v in target.shape[-2:]),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                mse = torch.mean((render - target).square()).clamp_min(1.0e-12)
                psnr = float((-10.0 * torch.log10(mse)).item())
                psnrs.append(psnr)

                target_img = _image_tensor_to_pil(target)
                render_img = _image_tensor_to_pil(render)
                w, h = target_img.size
                canvas = Image.new("RGB", (2 * w, h + 26), "white")
                canvas.paste(target_img, (0, 26))
                canvas.paste(render_img, (w, 26))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 6), "target panorama", fill=(0, 0, 0))
                draw.text((w + 8, 6), f"final render PSNR={psnr:.2f}dB", fill=(0, 0, 0))
                canvas = _resize_to_max_width(canvas, int(results_cfg.get("final_all_frames_max_width", 1600)))
                panel_path = panel_dir / f"frame_{int(frame_id):06d}.png"
                canvas.save(panel_path)
                per_frame.append({"frame_id": int(frame_id), "psnr": psnr, "render_vs_gt": str(panel_path)})

            ate_metrics: dict[str, float] = {}
            if len(pred_xyz) >= 1 and len(pred_xyz) == len(gt_xyz):
                _, _, ate_metrics, _ = _compute_ape_translation(
                    np.asarray(pred_xyz, dtype=np.float32),
                    np.asarray(gt_xyz, dtype=np.float32),
                )
            trajectory_metrics: dict[str, float] = {}
            if len(predicted_poses) >= 2:
                trajectory_metrics = c2w_trajectory_metrics(
                    torch.stack(predicted_poses, dim=0),
                    torch.stack(target_poses, dim=0),
                )
            trajectory_payload = {
                "pose_convention": "c2w",
                "frame_ids": trajectory_frame_ids,
                "predicted_c2w": [pose.tolist() for pose in predicted_poses],
                "target_c2w": [pose.tolist() for pose in target_poses],
                "metrics": trajectory_metrics,
            }
            trajectory_path = root / "trajectory.json"
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(trajectory_payload, f, indent=2)
            metrics = {
                "render_count": int(len(per_frame)),
                "mean_psnr": float(np.mean(psnrs)) if psnrs else None,
                "ate_rmse": ate_metrics.get("rmse"),
                "ate_count": int(len(pred_xyz)),
                **trajectory_metrics,
            }
            payload = {
                "metrics": metrics,
                "frames": per_frame,
                "trajectory": str(trajectory_path),
            }
            with open(root / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger._log_wandb_payload(
                {
                    f"final_pose/{key}": value
                    for key, value in trajectory_metrics.items()
                },
                step=max(1, int(logger._step) + 1),
            )
            return {"root": str(root), "metrics": metrics}

        def save_final_artifacts() -> dict:
            artifacts: dict[str, object] = {
                "final_ply": None,
                "final_checkpoint": None,
                "final_mlp_state": None,
                "final_keyframe_render_count": 0,
                "final_all_frames": None,
                "final_skybox_erp_preview": None,
                "final_skybox_faces": None,
            }
            if bool(results_cfg.get("save_final_ply", False)):
                try:
                    ply_path = output_dir / "point_cloud" / "final" / "point_cloud.ply"
                    self.map.save_ply(ply_path)
                    artifacts["final_ply"] = str(ply_path)
                    logger.log_artifact_file(ply_path)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"final ply save failed: {exc!r}")
            if bool(results_cfg.get("save_final_checkpoint", False)):
                try:
                    ckpt_path = output_dir / "checkpoints" / "final_gaussian_map.pt"
                    self.map.save_checkpoint(ckpt_path)
                    artifacts["final_checkpoint"] = str(ckpt_path)
                    logger.log_artifact_file(ckpt_path)
                    neural_cfg = self.config.get("NeuralScaffold", {}) if isinstance(self.config, dict) else {}
                    if bool(neural_cfg.get("save_mlp", False)) and hasattr(self.map, "save_mlp_state"):
                        mlp_path = ckpt_path.parent / "mlp_state.pth"
                        self.map.save_mlp_state(mlp_path)
                        artifacts["final_mlp_state"] = str(mlp_path)
                        logger.log_artifact_file(mlp_path)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"final checkpoint save failed: {exc!r}")
            if bool(results_cfg.get("save_final_keyframe_renders", False)):
                stride = max(1, int(results_cfg.get("final_keyframe_render_stride", 1)))
                max_count = int(results_cfg.get("final_keyframe_render_max", 0))
                selected = self.mapper.keyframes[::stride]
                if max_count > 0 and len(selected) > max_count:
                    selected = selected[-max_count:]
                render_dir = output_dir / "final_kf_renders"
                depth_dir = output_dir / "final_kf_depths"
                saved = 0
                for keyframe in selected:
                    try:
                        diagnostic = self.mapper.render_keyframe_diagnostic(int(keyframe.frame_id))
                        render_path, depth_path = logger.save_keyframe_diagnostic(
                            diagnostic,
                            render_dir=render_dir,
                            depth_dir=depth_dir,
                        )
                        if render_path is not None:
                            saved += 1
                            logger.log_image_file("backend/final_kf_render", render_path)
                        if depth_path is not None:
                            logger.log_image_file("backend/final_kf_depth", depth_path)
                    except Exception as exc:
                        self.mapper.stats.notes.append(
                            f"frame {keyframe.frame_id}: final keyframe render save failed: {exc!r}"
                        )
                artifacts["final_keyframe_render_count"] = int(saved)
            if bool(results_cfg.get("save_skybox_previews", False)) and self.map.has_skybox:
                try:
                    sky_dir = output_dir / "skybox"
                    sky_dir.mkdir(parents=True, exist_ok=True)
                    preview = self.map.skybox_erp_preview(
                        height=int(results_cfg.get("skybox_preview_height", 256)),
                        width=int(results_cfg.get("skybox_preview_width", 512)),
                    )
                    if preview is not None:
                        preview_path = sky_dir / "final_erp_preview.png"
                        _image_tensor_to_pil(preview).save(preview_path)
                        artifacts["final_skybox_erp_preview"] = str(preview_path)
                        logger.log_image_file("skybox/final_erp_preview", preview_path)
                    faces = self.map.get_skybox_faces
                    if torch.is_tensor(faces):
                        faces_path = sky_dir / "final_cubemap_faces.png"
                        _cubemap_faces_to_pil(faces).save(faces_path)
                        artifacts["final_skybox_faces"] = str(faces_path)
                        logger.log_image_file("skybox/final_cubemap_faces", faces_path)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"final skybox preview save failed: {exc!r}")
            final_all_frames = save_final_all_frame_renders()
            if final_all_frames is not None:
                artifacts["final_all_frames"] = final_all_frames
            return artifacts

        try:
            for frame in iter_sequence_frames(self.config):
                if max_frames is not None and frame_count >= int(max_frames):
                    break
                frame_start = time.perf_counter()
                frame_cache[int(frame.frame_id)] = frame
                section_start = time.perf_counter()
                out = self.frontend.track(frame)
                frontend_track_sec = float(time.perf_counter() - section_start)
                last_status = out.tracking_status
                pop_ready = getattr(self.frontend, "pop_ready_outputs", None)
                section_start = time.perf_counter()
                outputs = pop_ready() if callable(pop_ready) else [out]
                pop_ready_sec = float(time.perf_counter() - section_start)
                drain_spherical_selfi_windows(outputs)
                for ready in outputs:
                    process_output(ready)
                optimize_spherical_selfi_windows(outputs)
                drain_resplat_artifacts()
                for ready in outputs:
                    frame_cache.pop(int(ready.frame_id), None)
                optimize_feedforward_after_batch(outputs)
                write_profile(
                    "input_frame",
                    frame_id=int(frame.frame_id),
                    frontend_track_sec=frontend_track_sec,
                    pop_ready_sec=pop_ready_sec,
                    ready_outputs=float(len(outputs)),
                    total_sec=float(time.perf_counter() - frame_start),
                )
                frame_count += 1

            flush = getattr(self.frontend, "flush", None)
            if callable(flush):
                section_start = time.perf_counter()
                flushed = 0
                flushed_outputs = []
                for ready in flush():
                    flushed += 1
                    flushed_outputs.append(ready)
                drain_spherical_selfi_windows(flushed_outputs)
                for ready in flushed_outputs:
                    process_output(ready)
                optimize_spherical_selfi_windows(flushed_outputs)
                drain_resplat_artifacts()
                for ready in flushed_outputs:
                    frame_cache.pop(int(ready.frame_id), None)
                optimize_feedforward_after_batch(flushed_outputs)
                write_profile("frontend_flush", flushed_outputs=float(flushed), total_sec=float(time.perf_counter() - section_start))

            section_start = time.perf_counter()
            if spherical_selfi_global_enabled:
                spherical_final = self.spherical_selfi_global_backend.finalize()
                final_geometry_updates = self.spherical_selfi_global_backend.pop_frame_geometry_updates()
                apply_spherical_selfi_geometry_updates(final_geometry_updates, [])
                final_metrics = {
                    key: float(value)
                    for key, value in spherical_final.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            else:
                final_metrics = self.mapper.finalize_optimization()
            final_metrics["profile_backend_finalize_call_sec"] = float(time.perf_counter() - section_start)
            write_profile("finalize_optimization", **{
                key: float(value)
                for key, value in final_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and str(key).startswith("profile_")
            })
            if decision_file is not None:
                decision_file.close()
            if feedback_file is not None:
                feedback_file.close()
            final_backend_traj = logger.log_final_backend_trajectory(
                self.mapper.refined_keyframe_poses(),
                step=frame_count,
            )
            final_artifacts = save_final_artifacts()
            final_all_frames = final_artifacts.get("final_all_frames")
            final_all_metrics = (
                final_all_frames.get("metrics")
                if isinstance(final_all_frames, dict)
                and isinstance(final_all_frames.get("metrics"), dict)
                else {}
            )
            dense_ba_summary = _summarize_dense_ba_stats(self.frontend)
            local_ba_path = output_dir / "local_ba_windows.json"
            with open(local_ba_path, "w", encoding="utf-8") as f:
                json.dump(local_ba_window_records, f, indent=2)
            accepted_local_ba = sum(int(record["accepted"]) for record in local_ba_window_records)
            local_ba_summary = {
                "windows": int(len(local_ba_window_records)),
                "accepted": int(accepted_local_ba),
                "accepted_ratio": (
                    float(accepted_local_ba / len(local_ba_window_records))
                    if local_ba_window_records
                    else 0.0
                ),
                "mean_valid_factors": (
                    float(np.mean([record["num_factors"] for record in local_ba_window_records]))
                    if local_ba_window_records
                    else 0.0
                ),
                "mean_matching_sec": (
                    float(np.mean([record["matching_sec"] for record in local_ba_window_records]))
                    if local_ba_window_records
                    else 0.0
                ),
                "mean_ba_sec": (
                    float(np.mean([record["ba_sec"] for record in local_ba_window_records]))
                    if local_ba_window_records
                    else 0.0
                ),
            }
            summary = {
                "frames": frame_count,
                "keyframes": keyframes,
                "anchors": self.map.anchor_count(),
                "last_tracking_status": last_status,
                "map_mode": self.map.map_mode,
                "renderer": self.config.get("Training", {}).get("panorama_render_mode", "pfgs360_gsplat"),
                "pose_conventions": {
                    "dataset_input": (
                        "w2c"
                        if str(self.config.get("Dataset", {}).get("type", "")).lower() == "ob3d"
                        else str(self.config.get("Dataset", {}).get("pose_convention", "c2w"))
                    ),
                    "internal": "c2w",
                    "camera_axes": "+X right, +Y down, +Z forward",
                },
                "backend_last_loss": self.mapper.stats.last_loss,
                "backend_last_phase": self.mapper.stats.last_phase,
                "backend_optimization_steps": self.mapper.stats.optimization_steps,
                "backend_pose_delta_norm": self.mapper.stats.last_pose_delta_norm,
                "backend_last_window_size": self.mapper.stats.last_window_size,
                "backend_last_window_keyframes": self.mapper.stats.last_window_keyframes,
                "backend_last_active_keyframes": self.mapper.stats.last_active_keyframes,
                "backend_last_window_observations": self.mapper.stats.last_window_observations,
                "backend_last_feedforward_current_frames": self.mapper.stats.last_feedforward_current_frames,
                "backend_last_feedforward_history_frames": self.mapper.stats.last_feedforward_history_frames,
                "backend_last_sampled_keyframes": self.mapper.stats.last_sampled_keyframes,
                "backend_last_trainable_pose_count": self.mapper.stats.last_trainable_pose_count,
                "backend_last_feedforward_opacity_resets": self.mapper.stats.last_feedforward_opacity_resets,
                "backend_last_feedforward_pruned": self.mapper.stats.last_feedforward_pruned,
                "backend_last_replace_deleted": self.mapper.stats.last_replace_deleted,
                "backend_last_replace_fused": self.mapper.stats.last_replace_fused,
                "backend_last_replace_compacted": self.mapper.stats.last_replace_compacted,
                "backend_last_dense_seed_candidates": self.mapper.stats.last_dense_seed_candidates,
                "backend_last_insert_mask_seed_candidates": self.mapper.stats.last_insert_mask_seed_candidates,
                "backend_last_voxel_seed_candidates": self.mapper.stats.last_voxel_seed_candidates,
                "backend_last_replace_fused_existing": self.mapper.stats.last_replace_fused_existing,
                "backend_last_replace_fused_new_duplicate": self.mapper.stats.last_replace_fused_new_duplicate,
                "backend_last_replace_newly_inserted": self.mapper.stats.last_replace_newly_inserted,
                "backend_resplat_fusion_count": int(resplat_fusion_count),
                "backend_last_resplat_fused": self.mapper.stats.last_resplat_fused,
                "backend_last_resplat_inserted": self.mapper.stats.last_resplat_inserted,
                "backend_last_resplat_skipped": self.mapper.stats.last_resplat_skipped,
                "backend_last_pred_depth_generated_seeds": self.mapper.stats.last_pred_depth_generated_seeds,
                "backend_last_pred_depth_invalid_pixels": self.mapper.stats.last_pred_depth_invalid_pixels,
                "backend_last_insert_mask_pixels": self.mapper.stats.last_insert_mask_pixels,
                "backend_last_anchor_count_before_insert": self.mapper.stats.last_anchor_count_before_insert,
                "backend_last_anchor_count_after_insert": self.mapper.stats.last_anchor_count_after_insert,
                "backend_last_sky_pruned": self.mapper.stats.last_sky_pruned,
                "backend_last_sky_compacted": self.mapper.stats.last_sky_compacted,
                "backend_last_feedforward_metrics": last_feedforward_metrics,
                "backend_final_metrics": final_metrics,
                "final_all_frames_ate_rmse": final_all_metrics.get("ate_rmse"),
                "final_all_frames_se3_ate_rmse": final_all_metrics.get("se3_ate_rmse"),
                "final_all_frames_rpe_delta_1_translation_rmse": final_all_metrics.get("rpe_delta_1_translation_rmse"),
                "final_all_frames_rpe_delta_1_rotation_mean_deg": final_all_metrics.get("rpe_delta_1_rotation_mean_deg"),
                "final_all_frames_scale_drift_percent": final_all_metrics.get("scale_drift_percent"),
                "final_all_frames_trajectory_metrics": {
                    key: value
                    for key, value in final_all_metrics.items()
                    if key not in {"render_count", "mean_psnr", "ate_count"}
                },
                "final_all_frames_mean_psnr": final_all_metrics.get("mean_psnr"),
                "backend_final_trajectory_png": final_backend_traj,
                "artifacts": final_artifacts,
                "dense_ba": dense_ba_summary,
                "local_ba": local_ba_summary,
                "local_ba_windows_path": str(local_ba_path),
                "spherical_selfi_runtime_config": (
                    self.config.get("SphericalSelfiRuntime", {})
                    if spherical_selfi_global_enabled
                    else None
                ),
                "spherical_selfi_global_backend_config": (
                    self.config.get("SphericalSelfiGlobalBackend", {})
                    if spherical_selfi_global_enabled
                    else None
                ),
                "spherical_selfi_processed_windows": (
                    len(self.spherical_selfi_global_backend.results)
                    if spherical_selfi_global_enabled
                    else 0
                ),
                **_flatten_dense_ba_summary(dense_ba_summary),
                "wandb_run_url": logger.run_url,
                "wandb_mode": logger.wandb_mode,
                "wandb_init_error": logger.wandb_init_error,
                "visualization_dir": str(logger.visualization_dir) if logger.save_local else None,
                "keyframe_decisions_path": str(decision_path) if decision_logging_enabled else None,
                "keyframe_decision_count": int(keyframe_decision_count),
                "backend_feedback_path": str(feedback_path) if feedback_logging_enabled else None,
                "backend_feedback_decision_count": int(backend_feedback_decision_count),
                "backend_feedback_applied_count": int(backend_feedback_applied_count),
                "runtime_profile_path": str(profile_path) if profile_path is not None else None,
                "notes": self.mapper.stats.notes,
            }
            with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if profile_file is not None and not profile_file.closed:
                profile_file.close()
            logger.finish(summary)
            return summary
        except BaseException as exc:
            if decision_file is not None and not decision_file.closed:
                decision_file.close()
            if feedback_file is not None and not feedback_file.closed:
                feedback_file.close()
            if profile_file is not None and not profile_file.closed:
                profile_file.close()
            logger.finish({"failed": True, "error": repr(exc)})
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pano_droid_gs_slam.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("WeightsAndBiases", {})["enabled"] = True
    if args.wandb_mode is not None:
        cfg.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    if args.run_name:
        cfg.setdefault("WeightsAndBiases", {})["run_name"] = args.run_name
    system = PanoDroidGSSlamSystem(cfg)
    print(json.dumps(system.run(max_frames=args.max_frames), indent=2))


if __name__ == "__main__":
    main()
