import random
import time
import json
import yaml

import torch
import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os

from backend.legacy_360gs.gaussian_splatting.gaussian_renderer import render
from backend.legacy_360gs.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from backend.legacy_360gs.utils.camera_utils import PanoramaCamera
from backend.legacy_360gs.utils.panoramic_renderer import render_panorama_for_config
from backend.legacy_360gs.utils.logging_utils import Log
from backend.legacy_360gs.utils.multiprocessing_utils import (
    GAUSSIAN_CPU_ONLY_ATTRS,
    clone_obj_to_device,
    move_obj_to_device_,
    pack_queue_message,
    unpack_queue_message,
)
from backend.legacy_360gs.utils.pose_utils import update_pose
from backend.legacy_360gs.utils.slam_utils import (
    get_loss_mapping,
    _charbonnier,
    _get_panorama_supervision,
    erp_top_latitude_mask,
    robust_relative_depth_loss,
)
from backend.legacy_360gs.utils.slam_accel import EmaStabilityTracker, schedule_budget
from backend.legacy_360gs.utils.anchor_hash_index import AnchorHashIndex
from backend.legacy_360gs.utils.pano_consistency import (
    build_dia_insert_mask,
    cap_dia_insert_mask,
    cap_insert_mask_by_score,
    depth_projection_support_mask,
)
from backend.legacy_360gs.utils.submap_manager import SubmapManager, normalize_window_order


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.map_mode = config.get("MapRepresentation", {}).get(
            "mode", "legacy_gaussian_panorama"
        )
        self.use_anchor_scaffold = self.map_mode == "anchor_scaffold_panorama"
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.theta = 0
        self.enable_fastgs_erp = False
        self.fastgs_vcd_only = True
        self.fastgs_vcp_warmup_kfs = 8
        self.fastgs_debug_log_scores = True
        # Phase 4: online acceleration
        self._stability_tracker = EmaStabilityTracker(config)
        self._current_overlap: float = 1.0
        self.enable_submap = bool(config["Training"].get("enable_submap", True))
        self.anchor_hash_index = AnchorHashIndex() if self.use_anchor_scaffold else None
        self.submaps = {}
        self.active_submap_id = -1
        self._submap_manager = SubmapManager(
            interval=int(config["Training"].get("submap_kf_interval", 10)),
            overlap_kfs=int(config["Training"].get("submap_overlap_kfs", 3)),
        )
    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.global_BA_itr_num = self.config["Training"]["global_BA_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.early_prune_kf_count = int(
            self.config["Training"].get("early_prune_kf_count", 3)
        )
        self.early_gaussian_th = float(
            self.config["Training"].get("early_gaussian_th", 0.05)
        )
        self.early_disable_opacity_reset = bool(
            self.config["Training"].get("early_disable_opacity_reset", True)
        )
        self.window_size = self.config["Training"]["window_size"]
        self.enable_fastgs_erp = bool(
            self.config["Training"].get("enable_fastgs_erp", False)
        )
        self.fastgs_vcd_only = bool(
            self.config["Training"].get("fastgs_vcd_only", True)
        )
        self.fastgs_vcp_warmup_kfs = int(
            self.config["Training"].get("fastgs_vcp_warmup_kfs", 8)
        )
        self.fastgs_debug_log_scores = bool(
            self.config["Training"].get("fastgs_debug_log_scores", True)
        )
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def _append_jsonl(self, filename: str, payload: dict) -> None:
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_yaml(self, filename: str, payload: dict) -> None:
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    def _append_anchor_debug(
        self,
        frame_idx: int | None = None,
        render_loss: float | None = None,
        sky_loss: float | None = None,
    ) -> None:
        if not self.use_anchor_scaffold:
            return
        if not hasattr(self.gaussians, "get_anchor_debug_stats"):
            return
        payload = {
            "frame_idx": int(frame_idx) if frame_idx is not None else -1,
            "iteration": int(self.iteration_count),
        }
        payload.update(self.gaussians.get_anchor_debug_stats())
        if render_loss is not None:
            payload["render_loss"] = float(render_loss)
        if sky_loss is not None:
            payload["sky_loss"] = float(sky_loss)
        self._append_jsonl("anchor_debug.jsonl", payload)

    def _rebuild_anchor_hash_tables(
        self,
        *,
        resolve_collisions: bool = False,
        snap_xyz_to_voxel: bool = False,
    ) -> dict:
        if not self.use_anchor_scaffold or self.gaussians is None:
            return {}
        if self.anchor_hash_index is None:
            self.anchor_hash_index = AnchorHashIndex()
        stats = self.anchor_hash_index.rebuild_from_model(
            self.gaussians,
            resolve_collisions=resolve_collisions,
            snap_xyz_to_voxel=snap_xyz_to_voxel,
        )
        if hasattr(self.gaussians, "_last_hash_stats") and isinstance(
            self.gaussians._last_hash_stats, dict
        ):
            self.gaussians._last_hash_stats.update(stats)
        return stats

    def _ensure_anchor_hash_ready(self) -> None:
        if not self.use_anchor_scaffold or self.gaussians is None:
            return
        if self.anchor_hash_index is None:
            self.anchor_hash_index = AnchorHashIndex()
        if self.anchor_hash_index.is_stale(self.gaussians):
            self._rebuild_anchor_hash_tables(
                resolve_collisions=False,
                snap_xyz_to_voxel=False,
            )

    def _anchor_fastgs_active(self) -> bool:
        return bool(self.use_anchor_scaffold and self.enable_fastgs_erp)

    def _skip_anchor_explicit_densify_stats(self) -> bool:
        return self._anchor_fastgs_active()

    def _anchor_dia_active(self) -> bool:
        return bool(
            self.use_anchor_scaffold
            and self.config["Training"].get("enable_depth_inlier_densify", False)
        )

    def _dvo_depth_insertion_active(self) -> bool:
        training_cfg = self.config.get("Training", {}) if self.config else {}
        if not self.use_anchor_scaffold:
            return False
        frontend_mode = str(training_cfg.get("frontend_mode", "spherical")).lower()
        dvo_cfg = training_cfg.get("frontend_360dvo", {}) or {}
        insert_cfg = training_cfg.get("dvo_insertion", {}) or {}
        depth_policy = str(dvo_cfg.get("depth_scale_policy", "none")).lower()
        default_enabled = (
            frontend_mode == "360dvo"
            and str(dvo_cfg.get("mode", "offline_tum")).lower() == "online"
            and depth_policy
            in {
                "align_depth_to_dvo",
                "dvo",
                "dvo_scale",
                "scale_depth_to_dvo",
            }
        )
        return bool(insert_cfg.get("enabled", default_enabled))

    def _build_dvo_depth_insert_mask(self, viewpoint, depth_np, *, align_stats=None):
        training_cfg = self.config["Training"]
        insert_cfg = training_cfg.get("dvo_insertion", {}) or {}
        dvo_cfg = training_cfg.get("frontend_360dvo", {}) or {}
        shape = tuple(depth_np.shape)

        def _empty_stats(reason: str, extra: dict | None = None):
            stats = {
                "event": "dia_anchor_insert",
                "score_mode": "dvo_depth_novelty",
                "dvo_insertion": True,
                "dvo_insert_no_ratio_cap": False,
                "dvo_insert_skip_reason": str(reason),
                "valid_pixels": int(np.isfinite(depth_np).sum()),
                "candidate_pixels": 0,
                "candidate_pixels_before_cap": 0,
                "max_insert_pixels": 0,
                "insert_pixels": 0,
                "insert_ratio": 0.0,
                "capped": False,
            }
            if extra:
                stats.update(extra)
            return np.zeros_like(depth_np, dtype=bool), stats

        dvo_info = getattr(viewpoint, "dvo_depth_scale_info", None)
        require_scale_info = bool(insert_cfg.get("require_depth_scale_info", True))
        if require_scale_info and not isinstance(dvo_info, dict):
            return _empty_stats("missing_dvo_depth_scale_info")
        dvo_info = dvo_info if isinstance(dvo_info, dict) else {}
        raw_depth_scale = dvo_info.get("depth_scale", None)
        if raw_depth_scale is None:
            raw_depth_scale = getattr(viewpoint, "dvo_depth_scale", 1.0)
        try:
            depth_scale = float(raw_depth_scale)
        except (TypeError, ValueError):
            depth_scale = float("nan")
        if require_scale_info and (
            not np.isfinite(depth_scale) or depth_scale <= 0.0
        ):
            return _empty_stats(
                "invalid_dvo_depth_scale",
                {"dvo_depth_scale": float(depth_scale)},
            )
        min_sparse = int(
            insert_cfg.get(
                "min_sparse_points",
                dvo_cfg.get("min_sparse_points", 0),
            )
        )
        sparse_valid = int(dvo_info.get("sparse_valid", 0))
        if bool(insert_cfg.get("require_sparse_depth", False)) and sparse_valid < min_sparse:
            return _empty_stats(
                "insufficient_dvo_sparse_depth",
                {
                    "dvo_sparse_valid": int(sparse_valid),
                    "dvo_min_sparse_points": int(min_sparse),
                },
            )

        min_depth = float(insert_cfg.get("min_depth", 0.01))
        max_depth = float(
            insert_cfg.get(
                "max_depth",
                training_cfg.get(
                    "dap_depth_max_valid",
                    training_cfg.get("ransac", {}).get("depth_max", 80.0),
                ),
            )
        )
        valid_insert = (
            np.isfinite(depth_np)
            & (depth_np > min_depth)
            & (depth_np < max_depth)
        )
        sky_mask = self._as_hw_bool_mask(getattr(viewpoint, "erp_sky_mask", None), shape)
        if sky_mask is not None:
            valid_insert &= ~sky_mask
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        valid_region = self._as_hw_bool_mask(region_masks.get("valid", None), shape)
        if valid_region is not None:
            valid_insert &= valid_region
        valid_before_novelty = valid_insert.copy()

        novelty_mode = "depth_conflict"
        novelty_attr = "kf_depth_conflict_mask"
        novelty_mask = self._as_hw_bool_mask(getattr(viewpoint, novelty_attr, None), shape)
        if novelty_mask is None:
            return _empty_stats("missing_kf_depth_conflict_mask")
        valid_insert = valid_insert & novelty_mask

        max_insert_ratio = float(
            insert_cfg.get(
                "max_insert_ratio",
                training_cfg.get("dvo_insert_max_insert_ratio", 0.10),
            )
        )
        raw_max_pixels = insert_cfg.get(
            "max_insert_pixels",
            training_cfg.get("dvo_insert_max_insert_pixels", 30000),
        )
        max_insert_pixels = None if raw_max_pixels is None else int(raw_max_pixels)
        apply_cap = bool(insert_cfg.get("apply_cap", True))
        cap_score_attr = str(insert_cfg.get("cap_score_attr", "kf_depth_conflict_score"))
        score_map = None
        score_like = getattr(viewpoint, cap_score_attr, None)
        if score_like is not None:
            if isinstance(score_like, torch.Tensor):
                score_map = score_like.detach().cpu().numpy()
            else:
                score_map = np.asarray(score_like)
            if score_map.ndim == 3:
                score_map = score_map[0]
            if tuple(score_map.shape) != shape:
                score_map = None
        candidate_before_cap = valid_insert.copy()
        if apply_cap:
            valid_insert, cap_stats = cap_insert_mask_by_score(
                candidate_before_cap,
                valid_before_novelty,
                max_insert_ratio=max_insert_ratio,
                max_insert_pixels=max_insert_pixels,
                score_map=score_map,
            )
        else:
            cap_stats = {
                "candidate_pixels_before_cap": int(candidate_before_cap.sum()),
                "valid_pixels": int(valid_before_novelty.sum()),
                "max_insert_ratio": float(max_insert_ratio),
                "max_insert_pixels": -1,
                "max_insert_pixels_by_ratio": -1,
                "max_insert_pixels_by_absolute": int(max_insert_pixels)
                if max_insert_pixels is not None
                else -1,
                "insert_pixels": int(candidate_before_cap.sum()),
                "insert_ratio": float(candidate_before_cap.mean())
                if candidate_before_cap.size
                else 0.0,
                "capped": False,
                "cap_score_used": False,
            }

        stats = {
            "event": "dia_anchor_insert",
            "score_mode": "dvo_depth_novelty",
            "dvo_insertion": True,
            "dvo_insert_no_ratio_cap": False,
            "valid_pixels": int(cap_stats.get("valid_pixels", int(valid_before_novelty.sum()))),
            "candidate_pixels": int(
                cap_stats.get("candidate_pixels_before_cap", int(candidate_before_cap.sum()))
            ),
            "candidate_pixels_before_cap": int(
                cap_stats.get("candidate_pixels_before_cap", int(candidate_before_cap.sum()))
            ),
            "max_insert_ratio": float(cap_stats.get("max_insert_ratio", max_insert_ratio)),
            "max_insert_pixels": int(cap_stats.get("max_insert_pixels", -1)),
            "max_insert_pixels_by_ratio": int(
                cap_stats.get("max_insert_pixels_by_ratio", -1)
            ),
            "max_insert_pixels_by_absolute": int(
                cap_stats.get("max_insert_pixels_by_absolute", -1)
            ),
            "insert_pixels": int(cap_stats.get("insert_pixels", int(valid_insert.sum()))),
            "insert_ratio": float(
                cap_stats.get(
                    "insert_ratio",
                    float(valid_insert.mean()) if valid_insert.size else 0.0,
                )
            ),
            "capped": bool(cap_stats.get("capped", False)),
            "dvo_cap_score_attr": str(cap_score_attr),
            "dvo_cap_score_used": bool(cap_stats.get("cap_score_used", False)),
            "dvo_apply_cap": bool(apply_cap),
            "dvo_depth_scale": float(depth_scale),
            "dap_to_dvo_scale": float(dvo_info.get("dap_to_dvo_scale", depth_scale)),
            "dvo_depth_scale_source": str(dvo_info.get("depth_scale_source", "-")),
            "dap_to_dvo_scale_source": str(
                dvo_info.get(
                    "dap_to_dvo_scale_source",
                    dvo_info.get("depth_scale_source", "-"),
                )
            ),
            "dvo_depth_scale_stable": bool(dvo_info.get("depth_scale_stable", False)),
            "dap_to_dvo_scale_stable": bool(
                dvo_info.get(
                    "dap_to_dvo_scale_stable",
                    dvo_info.get("depth_scale_stable", False),
                )
            ),
            "dap_to_dvo_ratio_mad_norm": float(
                dvo_info.get("dap_to_dvo_ratio_mad_norm", float("nan"))
            ),
            "dap_to_dvo_ratio_count": int(dvo_info.get("dap_to_dvo_ratio_count", 0)),
            "dvo_sparse_valid": int(sparse_valid),
            "dvo_min_sparse_points": int(min_sparse),
            "dvo_used_novelty_mask": True,
            "dvo_require_novelty_mask": True,
            "dvo_novelty_mode": str(novelty_mode),
            "dvo_novelty_attr": str(novelty_attr),
            "dvo_min_depth": float(min_depth),
            "dvo_max_depth": float(max_depth),
            "sky_pixels": int(sky_mask.sum()) if sky_mask is not None else 0,
            "valid_region_pixels": int(valid_region.sum()) if valid_region is not None else 0,
            "novelty_pixels": int(novelty_mask.sum()) if novelty_mask is not None else 0,
        }
        if isinstance(align_stats, dict):
            stats.update(
                {
                    "dap_scale_frame": float(align_stats.get("scale_frame", 1.0)),
                    "dap_align_pixels": int(align_stats.get("align_pixels", 0)),
                    "dap_align_reliable": bool(align_stats.get("reliable", False)),
                }
            )
        Log(
            f"[DVO insert] frame={getattr(viewpoint, 'uid', -1)} "
            f"pixels={int(valid_insert.sum())} "
            f"candidates={int(candidate_before_cap.sum())} "
            f"cap={int(stats['max_insert_pixels'])} "
            f"capped={bool(stats['capped'])} "
            f"score_cap={bool(stats['dvo_cap_score_used'])} "
            f"depth_scale={float(depth_scale):.5f} "
            f"source={stats['dvo_depth_scale_source']} "
            f"sparse={int(sparse_valid)} "
            f"novelty={str(novelty_mode)}",
            tag="BackEnd",
        )
        return valid_insert.astype(bool), stats

    def _neural_sky_alpha_loss(self, render_pkg, viewpoint):
        """Splatfacto-W style alpha loss for the neural sky background.

        On sky-mask pixels we want the neural background MLP to drive the
        colour. Penalising the rendered GS opacity there pushes structure
        Gaussians out of the sky region, eliminating the "speckle" artifacts
        that the legacy SkyBox-anchor stack produced. Always returns a scalar
        tensor (or 0.0 when not applicable) so call sites can just ``+=`` it.
        """
        cfg = self.config["Training"]
        if not bool(cfg.get("enable_neural_sky_bg", False)):
            return 0.0
        w = float(cfg.get("neural_sky_alpha_loss_weight", 0.0))
        if w <= 0.0:
            return 0.0
        opacity = None
        if isinstance(render_pkg, dict):
            opacity = render_pkg.get("opacity")
        if opacity is None or not torch.is_tensor(opacity):
            return 0.0
        H = int(opacity.shape[-2])
        W = int(opacity.shape[-1])
        from backend.legacy_360gs.utils.panoramic_renderer import _get_body_cam_sky_mask

        sky_mask = _get_body_cam_sky_mask(
            viewpoint, H, W, device=opacity.device, dtype=opacity.dtype
        )
        if sky_mask is None:
            return 0.0
        sky_mask = sky_mask.to(device=opacity.device, dtype=opacity.dtype).view_as(opacity)
        denom = sky_mask.sum().clamp_min(1.0)
        l_alpha = (opacity.clamp(0.0, 1.0) ** 2 * sky_mask).sum() / denom
        return w * l_alpha

    def _build_xyz_freeze_mask(self) -> torch.Tensor | None:
        """Construct the per-anchor mask whose xyz gradient should be zeroed
        before ``optimizer.step()``.

        Default (post-fix) behaviour:
          * Only sky anchors are frozen (when ``SkyBox.freeze_xyz`` is True).
          * Structure anchors are free to move and learn their position.

        Legacy / ablation behaviour (when ``Training.enable_structure_xyz_freeze``
        is True): every anchor's xyz is frozen, matching the old global freeze
        that produced the ``fixedxyz`` runs.
        """
        if self.gaussians is None or not hasattr(self.gaussians, "_xyz"):
            return None
        n = int(self.gaussians.get_xyz.shape[0])
        if n <= 0:
            return None
        device = self.device
        cfg_train = self.config.get("Training", {})
        cfg_sky = self.config.get("SkyBox", {})
        if bool(cfg_train.get("enable_structure_xyz_freeze", False)):
            return torch.ones((n,), dtype=torch.bool, device=device)
        mask = torch.zeros((n,), dtype=torch.bool, device=device)
        sky_mask_attr = getattr(self.gaussians, "_is_sky_anchor", None)
        if (
            bool(cfg_sky.get("freeze_xyz", True))
            and sky_mask_attr is not None
            and sky_mask_attr.shape[0] == n
        ):
            mask |= sky_mask_attr.to(device=device, dtype=torch.bool)
        return mask

    def _freeze_xyz_gradients(self, freeze_mask: torch.Tensor | None = None) -> int:
        """Zero out ``_xyz.grad`` for the rows selected by ``freeze_mask``.

        Backwards compatible: when ``freeze_mask`` is ``None`` the call is a
        no-op (sky-only freeze is requested through ``_build_xyz_freeze_mask``).
        """
        if self.gaussians is None or not hasattr(self.gaussians, "_xyz"):
            return 0
        xyz = self.gaussians._xyz
        if xyz.grad is None:
            return 0
        if freeze_mask is None:
            return 0
        if freeze_mask.numel() == 0 or freeze_mask.shape[0] != xyz.shape[0]:
            return 0
        if not bool(freeze_mask.any().item()):
            return 0
        xyz.grad[freeze_mask] = 0
        return int(freeze_mask.sum().item())

    def _mask_inactive_anchor_gradients(
        self,
        current_kf_id: int | None,
        current_window=None,
    ) -> dict:
        """Zero non-xyz gradients of protected & non-locally-active anchors.

        This is gated by ``Training.enable_protected_anchor_freeze`` so callers
        can do clean A/B experiments. The statistics dict is always populated
        so ``backend_perf.jsonl`` keeps reporting the underlying counts.

        When ``current_window`` is provided, the protection rule is
        "anchors whose birth_frame is NOT in the current mapping window",
        which is the more permissive Fix2 behaviour. Otherwise the legacy
        ``birth < current_kf_id`` rule is used.
        """
        empty = {
            "n_protected_static": 0,
            "n_local_active": 0,
            "n_frozen_grad_zeroed": 0,
        }
        if (
            not self.use_anchor_scaffold
            or self.gaussians is None
            or not hasattr(self.gaussians, "get_protected_anchor_mask")
            or not hasattr(self.gaussians, "get_local_active_anchor_mask")
        ):
            return empty
        protected = self.gaussians.get_protected_anchor_mask(
            current_window=current_window,
            current_kf_id=current_kf_id,
            device=self.device,
        )
        local_active = self.gaussians.get_local_active_anchor_mask(device=self.device)
        n_local_active = int(local_active.sum().item()) if local_active.numel() else 0
        if protected.numel() == 0 or protected.shape[0] != self.gaussians.get_xyz.shape[0]:
            return {
                "n_protected_static": 0,
                "n_local_active": n_local_active,
                "n_frozen_grad_zeroed": 0,
            }
        n_protected = int(protected.sum().item())
        freeze_mask = protected & (~local_active)
        n_frozen = int(freeze_mask.sum().item())
        apply_freeze = bool(
            self.config["Training"].get("enable_protected_anchor_freeze", False)
            and self.config["Training"].get("enable_local_anchor_freeze", True)
        )
        if apply_freeze and n_frozen > 0:
            for attr in (
                "_anchor_feat",
                "_offsets",
                "_scaling",
                "_rotation",
                "_opacity",
                "_features_dc",
                "_features_rest",
            ):
                tensor = getattr(self.gaussians, attr, None)
                if tensor is None:
                    continue
                if tensor.grad is None:
                    continue
                if tensor.grad.shape[0] != freeze_mask.shape[0]:
                    continue
                tensor.grad[freeze_mask] = 0
        return {
            "n_protected_static": n_protected,
            "n_local_active": n_local_active,
            "n_frozen_grad_zeroed": n_frozen if apply_freeze else 0,
        }

    def _anchor_fastgs_warmup_ready(self, *, init: bool = False) -> bool:
        if init or not self._anchor_fastgs_active():
            return False
        if self.gaussians is None or self.background is None:
            return False
        if int(self.gaussians.get_xyz.shape[0]) <= 0:
            return False
        return len(self.viewpoints) >= self.fastgs_vcp_warmup_kfs

    def _build_anchor_fastgs_insert_mask(self, viewpoint, *, init: bool = False):
        if not self._anchor_fastgs_warmup_ready(init=init):
            return None, None
        if not isinstance(viewpoint, PanoramaCamera):
            return None, None
        from backend.legacy_360gs.utils.fastgs_erp import build_fastgs_metric_map_erp

        with torch.no_grad():
            metric_map, _, metric_stats = build_fastgs_metric_map_erp(
                viewpoint,
                self.gaussians,
                self.background,
                self.config,
            )
        return metric_map.detach().cpu().numpy().astype(bool), metric_stats

    def _viewpoint_w2c_np(self, viewpoint):
        return (
            getWorld2View2(viewpoint.R, viewpoint.T)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _as_hw_bool_mask(self, mask_like, shape):
        if mask_like is None:
            return None
        if isinstance(mask_like, torch.Tensor):
            mask_np = mask_like.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_like)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if tuple(mask_np.shape) != tuple(shape):
            return None
        return mask_np.astype(bool)

    def _aligned_ref_depth_and_valid(self, ref_viewpoint, shape):
        depth = getattr(ref_viewpoint, "mono_depth", None)
        if depth is None:
            return None, None
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[0]
        if tuple(depth_np.shape) != tuple(shape):
            return None, None
        training_cfg = self.config["Training"]
        depth_max = float(
            training_cfg.get(
                "dap_depth_max_valid",
                training_cfg.get("ransac", {}).get("depth_max", 80.0),
            )
        )
        valid = np.isfinite(depth_np) & (depth_np > 0.01) & (depth_np < depth_max)
        sky_mask = self._as_hw_bool_mask(getattr(ref_viewpoint, "erp_sky_mask", None), shape)
        if sky_mask is not None:
            valid &= ~sky_mask
        region_masks = getattr(ref_viewpoint, "erp_region_masks", None) or {}
        valid_region = self._as_hw_bool_mask(region_masks.get("valid", None), shape)
        if valid_region is not None:
            valid &= valid_region
        return depth_np, valid

    def _select_dia_multiview_refs(self, viewpoint, shape):
        training_cfg = self.config["Training"]
        current_uid = int(getattr(viewpoint, "uid", -1))
        ref_kfs = int(training_cfg.get("dia_mv_ref_kfs", 4))
        require_aligned = bool(training_cfg.get("dia_mv_require_aligned_refs", True))
        candidate_ids = []
        for kf_id in list(self.current_window):
            kf_int = int(kf_id)
            if kf_int != current_uid and kf_int in self.viewpoints:
                candidate_ids.append(kf_int)
        if not candidate_ids:
            candidate_ids = [
                int(kf_id)
                for kf_id in self.viewpoints.keys()
                if int(kf_id) != current_uid
            ]
        if current_uid >= 0:
            previous = [kf_id for kf_id in candidate_ids if kf_id < current_uid]
            if previous:
                candidate_ids = previous
        candidate_ids = sorted(set(candidate_ids), key=lambda kf_id: abs(current_uid - int(kf_id)))

        refs = []
        for kf_id in candidate_ids:
            ref = self.viewpoints.get(int(kf_id))
            if not isinstance(ref, PanoramaCamera):
                continue
            align_stats = getattr(ref, "dap_depth_align_stats", None)
            if require_aligned and not (isinstance(align_stats, dict) and bool(align_stats.get("reliable", False))):
                continue
            depth_np, valid_np = self._aligned_ref_depth_and_valid(ref, shape)
            if depth_np is None or valid_np is None:
                continue
            refs.append((int(kf_id), ref, depth_np, valid_np))
            if len(refs) >= ref_kfs:
                break
        return refs

    def _apply_dia_multiview_gate(self, viewpoint, depth_np, base_mask):
        training_cfg = self.config["Training"]
        if not bool(training_cfg.get("dia_multiview_gate_enabled", False)):
            return base_mask.astype(bool), {
                "dia_mv_enabled": False,
                "dia_mv_ref_count": 0,
                "dia_mv_supported_pixels": int(base_mask.sum()),
                "dia_mv_rejected_pixels": 0,
            }
        base_mask = np.asarray(base_mask, dtype=bool)
        if not base_mask.any():
            return base_mask, {
                "dia_mv_enabled": True,
                "dia_mv_ref_count": 0,
                "dia_mv_supported_pixels": 0,
                "dia_mv_rejected_pixels": 0,
            }
        refs = self._select_dia_multiview_refs(viewpoint, base_mask.shape)
        min_refs = int(training_cfg.get("dia_mv_min_refs", training_cfg.get("dia_mv_min_support", 2)))
        min_support = int(training_cfg.get("dia_mv_min_support", 2))
        if len(refs) < min_refs:
            return np.zeros_like(base_mask, dtype=bool), {
                "dia_mv_enabled": True,
                "dia_mv_ref_count": int(len(refs)),
                "dia_mv_min_refs": int(min_refs),
                "dia_mv_min_support": int(min_support),
                "dia_mv_supported_pixels": 0,
                "dia_mv_rejected_pixels": int(base_mask.sum()),
                "dia_mv_skipped_insufficient_refs": True,
            }

        src_w2c = self._viewpoint_w2c_np(viewpoint)
        support_count = np.zeros_like(base_mask, dtype=np.int16)
        valid_projected_total = 0
        rel_sum = 0.0
        rel_count = 0
        rel_thresh = float(training_cfg.get("dia_mv_depth_rel_thresh", 0.15))
        ref_ids = []
        for kf_id, ref, ref_depth, ref_valid in refs:
            ref_ids.append(int(kf_id))
            ref_w2c = self._viewpoint_w2c_np(ref)
            supported, ref_stats = depth_projection_support_mask(
                depth_np,
                src_w2c,
                ref_depth,
                ref_w2c,
                src_valid=base_mask,
                ref_valid=ref_valid,
                rel_thresh=rel_thresh,
            )
            support_count += supported.astype(np.int16)
            valid_projected = int(ref_stats.get("valid_projected", 0))
            valid_projected_total += valid_projected
            if valid_projected > 0:
                rel_sum += float(ref_stats.get("mean_rel_valid", 0.0)) * valid_projected
                rel_count += valid_projected

        gated = base_mask & (support_count >= min_support)
        return gated, {
            "dia_mv_enabled": True,
            "dia_mv_ref_count": int(len(refs)),
            "dia_mv_ref_ids": ref_ids,
            "dia_mv_min_support": int(min_support),
            "dia_mv_depth_rel_thresh": float(rel_thresh),
            "dia_mv_supported_pixels": int(gated.sum()),
            "dia_mv_rejected_pixels": int((base_mask & ~gated).sum()),
            "dia_mv_valid_projected": int(valid_projected_total),
            "dia_mv_mean_rel_valid": float(rel_sum / max(rel_count, 1)),
        }

    def _build_anchor_dia_insert_mask(self, viewpoint, depth_map, *, init: bool = False):
        if init or not self._anchor_dia_active():
            return None, None
        if not isinstance(viewpoint, PanoramaCamera) or depth_map is None:
            return None, None
        if self.gaussians is None or int(self.gaussians.get_xyz.shape[0]) <= 0:
            return None, None
        warmup_kfs = int(
            self.config["Training"].get(
                "dia_densify_warmup_kfs", self.fastgs_vcp_warmup_kfs
            )
        )
        if len(self.viewpoints) < warmup_kfs:
            return None, None
        depth_np = np.asarray(depth_map, dtype=np.float32)
        align_stats = getattr(viewpoint, "dap_depth_align_stats", None)
        if self._dvo_depth_insertion_active():
            return self._build_dvo_depth_insert_mask(
                viewpoint,
                depth_np,
                align_stats=align_stats,
            )
        if (
            bool(self.config["Training"].get("dia_disable_when_depth_align_unreliable", False))
            and isinstance(align_stats, dict)
            and not bool(align_stats.get("reliable", True))
        ):
            Log(
                f"[DIA] frame={getattr(viewpoint, 'uid', -1)} skipped: "
                f"unreliable DAP/render alignment "
                f"(scale_frame={float(align_stats.get('scale_frame', 1.0)):.3f}, "
                f"align_pixels={int(align_stats.get('align_pixels', 0))})",
                tag="BackEnd",
            )
            zero_mask = np.zeros_like(depth_np, dtype=bool)
            return zero_mask, {
                "event": "dia_anchor_insert",
                "score_mode": "anchor_dia",
                "valid_pixels": int(np.isfinite(depth_np).sum()),
                "candidate_pixels": 0,
                "insert_pixels": 0,
                "insert_ratio": 0.0,
                "capped": False,
                "dia_skipped_unreliable_align": True,
                "dap_scale_frame": float(align_stats.get("scale_frame", 1.0)),
                "dap_align_pixels": int(align_stats.get("align_pixels", 0)),
                "dap_align_reliable": False,
            }

        valid_insert = np.isfinite(depth_np) & (depth_np > 0.01)
        depth_max = float(
            self.config["Training"].get(
                "dap_depth_max_valid",
                self.config["Training"].get("ransac", {}).get("depth_max", 80.0),
            )
        )
        valid_insert &= depth_np < depth_max
        def _mask_to_numpy(mask_like):
            if mask_like is None:
                return None
            if isinstance(mask_like, torch.Tensor):
                mask_np = mask_like.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask_like)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            return mask_np.astype(bool)

        sky_mask = _mask_to_numpy(getattr(viewpoint, "erp_sky_mask", None))
        if sky_mask is not None and tuple(sky_mask.shape) == tuple(depth_np.shape):
            valid_insert &= ~sky_mask
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        valid_region = _mask_to_numpy(region_masks.get("valid", None))
        if valid_region is not None:
            if valid_region.shape == depth_np.shape:
                valid_insert &= valid_region
        kf_novelty_mask = _mask_to_numpy(getattr(viewpoint, "kf_novelty_mask", None))
        if (
            bool(self.config["Training"].get("use_kf_novelty_mask_for_anchor_insert", True))
            and kf_novelty_mask is not None
            and kf_novelty_mask.shape == depth_np.shape
        ):
            valid_insert &= kf_novelty_mask

        with torch.no_grad():
            render_pkg = render_panorama_for_config(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                config=self.config,
                theta=torch.zeros(1, 3, device=self.device),
                rho=torch.zeros(1, 3, device=self.device),
            )
        if render_pkg is None or render_pkg.get("depth", None) is None:
            return None, None
        render_depth = render_pkg["depth"].detach().squeeze(0).cpu().numpy().astype(np.float32)
        opacity = render_pkg.get("opacity", None)
        opacity_np = (
            opacity.detach().squeeze(0).cpu().numpy().astype(np.float32)
            if opacity is not None else None
        )
        mask, stats = build_dia_insert_mask(
            render_depth,
            depth_np,
            valid_insert=valid_insert,
            opacity=opacity_np,
            rel_thresh=float(self.config["Training"].get("dia_densify_depth_rel_thresh", 0.10)),
            opacity_min=float(self.config["Training"].get("dia_densify_opacity_min", 0.15)),
            max_insert_ratio=float(self.config["Training"].get("dia_densify_max_insert_ratio", 0.25)),
            apply_cap=not bool(self.config["Training"].get("dia_multiview_gate_enabled", False)),
            far_depth_start=self.config["Training"].get("dia_far_depth_start", None),
            far_rel_thresh_mult=float(self.config["Training"].get("dia_far_rel_thresh_mult", 1.0)),
            far_max_insert_ratio=self.config["Training"].get("dia_far_max_insert_ratio", None),
            disable_insert_beyond=self.config["Training"].get("dia_disable_insert_beyond", None),
        )
        if bool(self.config["Training"].get("dia_multiview_gate_enabled", False)):
            base_candidate_pixels = int(mask.sum())
            mask, mv_stats = self._apply_dia_multiview_gate(viewpoint, depth_np, mask)
            capped_mask, cap_stats = cap_dia_insert_mask(
                mask,
                valid_insert,
                max_insert_ratio=float(
                    self.config["Training"].get("dia_densify_max_insert_ratio", 0.25)
                ),
                depth_map=depth_np,
                far_depth_start=self.config["Training"].get("dia_far_depth_start", None),
                far_max_insert_ratio=self.config["Training"].get("dia_far_max_insert_ratio", None),
                disable_insert_beyond=self.config["Training"].get("dia_disable_insert_beyond", None),
            )
            stats.update(
                {
                    "base_candidate_pixels": int(base_candidate_pixels),
                    "gated_candidate_pixels": int(mask.sum()),
                    **mv_stats,
                    "candidate_pixels_before_cap": int(cap_stats.get("candidate_pixels_before_cap", int(mask.sum()))),
                    "max_insert_pixels": int(cap_stats.get("max_insert_pixels", 0)),
                    "insert_pixels": int(cap_stats.get("insert_pixels", int(capped_mask.sum()))),
                    "far_insert_pixels": int(cap_stats.get("far_insert_pixels", 0)),
                    "far_valid_pixels": int(cap_stats.get("far_valid_pixels", stats.get("far_valid_pixels", 0))),
                    "far_max_insert_pixels": int(cap_stats.get("far_max_insert_pixels", 0)),
                    "beyond_insert_suppressed_pixels": int(
                        cap_stats.get(
                            "beyond_insert_suppressed_pixels",
                            stats.get("beyond_insert_suppressed_pixels", 0),
                        )
                    ),
                    "insert_ratio": float(cap_stats.get("insert_ratio", float(capped_mask.mean()))),
                    "capped": bool(cap_stats.get("capped", False)),
                }
            )
            mask = capped_mask
        stats.update({"event": "dia_anchor_insert", "score_mode": "anchor_dia"})
        if isinstance(align_stats, dict):
            stats.update(
                {
                    "dap_scale_frame": float(align_stats.get("scale_frame", 1.0)),
                    "dap_align_pixels": int(align_stats.get("align_pixels", 0)),
                    "dap_align_reliable": bool(align_stats.get("reliable", False)),
                }
            )
        return mask.astype(bool), stats

    def _append_anchor_fastgs_insert_debug(
        self,
        frame_idx: int,
        metric_stats: dict | None,
        growth_stats: dict | None,
    ) -> None:
        if not (self._anchor_fastgs_active() or self._anchor_dia_active()) or not self.fastgs_debug_log_scores:
            return
        if metric_stats is None:
            return
        event = {
            "event": "fastgs_vcd_insert",
            "iteration": int(self.iteration_count),
            "frame_idx": int(frame_idx),
            "score_mode": "anchor",
            "fastgs_vcd_only": bool(self.fastgs_vcd_only),
        }
        event.update(metric_stats)
        if isinstance(growth_stats, dict):
            event.update(
                {
                    "n_voxel_candidates": int(growth_stats.get("n_voxel_candidates", 0)),
                    "n_insert_enabled_candidates": int(
                        growth_stats.get("n_insert_enabled_candidates", 0)
                    ),
                    "n_structure_hash_hits": int(
                        growth_stats.get("n_structure_hash_hits", 0)
                    ),
                    "n_structure_hash_near_hits": int(
                        growth_stats.get("n_structure_hash_near_hits", 0)
                    ),
                    "n_structure_hash_misses": int(
                        growth_stats.get("n_structure_hash_misses", 0)
                    ),
                    "n_structure_hash_new": int(
                        growth_stats.get("n_structure_hash_new", 0)
                    ),
                    "n_hash_hit_thaw": int(growth_stats.get("n_hash_hit_thaw", 0)),
                    "n_hash_miss_new": int(growth_stats.get("n_hash_miss_new", 0)),
                    "n_local_thaw": int(growth_stats.get("n_local_thaw", 0)),
                    "n_local_new": int(growth_stats.get("n_local_new", 0)),
                    "n_protected_static": int(growth_stats.get("n_protected_static", 0)),
                    "n_frozen_static": int(growth_stats.get("n_frozen_static", 0)),
                    "n_structure_vcd_suppressed": int(
                        growth_stats.get("n_structure_vcd_suppressed", 0)
                    ),
                    "n_sky_hash_new": int(growth_stats.get("n_sky_hash_new", 0)),
                    "dia_anchor_evidence_obs": int(
                        growth_stats.get("dia_anchor_evidence_obs", 0)
                    ),
                    "dia_anchor_inconsistent_hits": int(
                        growth_stats.get("dia_anchor_inconsistent_hits", 0)
                    ),
                    "dia_anchor_depth_inconsistent_hits": int(
                        growth_stats.get("dia_anchor_depth_inconsistent_hits", 0)
                    ),
                    "dia_anchor_far_depth_valid": int(
                        growth_stats.get("dia_anchor_far_depth_valid", 0)
                    ),
                    "dia_anchor_far_depth_inconsistent_hits": int(
                        growth_stats.get("dia_anchor_far_depth_inconsistent_hits", 0)
                    ),
                    "dia_anchor_strong_prune_hits": int(
                        growth_stats.get("dia_anchor_strong_prune_hits", 0)
                    ),
                    "dia_anchor_replacement_hits": int(
                        growth_stats.get("dia_anchor_replacement_hits", 0)
                    ),
                    "dia_anchor_reset_evidence_hits": int(
                        growth_stats.get("dia_anchor_reset_evidence_hits", 0)
                    ),
                    "dia_anchor_sky_floater_hits": int(
                        growth_stats.get("dia_anchor_sky_floater_hits", 0)
                    ),
                    "dia_anchor_supported": int(
                        growth_stats.get("dia_anchor_supported", 0)
                    ),
                    "dia_anchor_protected_evidence_blocked": int(
                        growth_stats.get("dia_anchor_protected_evidence_blocked", 0)
                    ),
                    "dia_anchor_rel_err_p50": float(
                        growth_stats.get("dia_anchor_rel_err_p50", 0.0)
                    ),
                    "dia_anchor_rel_err_p90": float(
                        growth_stats.get("dia_anchor_rel_err_p90", 0.0)
                    ),
                    "dia_anchor_rel_err_p95": float(
                        growth_stats.get("dia_anchor_rel_err_p95", 0.0)
                    ),
                    "dia_anchor_far_rel_err_p90": float(
                        growth_stats.get("dia_anchor_far_rel_err_p90", 0.0)
                    ),
                }
            )
        self._append_jsonl("fastgs_debug.jsonl", event)

    def _annotate_frontend_kf_render(self, frame_idx: int, growth_stats: dict) -> None:
        if self.save_dir is None or not isinstance(growth_stats, dict):
            return
        try:
            import cv2
            from backend.legacy_360gs.utils.kf_render_io import kf_render_extension

            render_dir = os.path.join(self.save_dir, "kf_renders")
            ext = kf_render_extension(self.config.get("Results", {}))
            image_path = os.path.join(render_dir, f"kf_{int(frame_idx):04d}{ext}")
            if not os.path.exists(image_path):
                return
            canvas = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if canvas is None:
                return

            n_new = int(growth_stats.get("n_structure_hash_new", 0)) + int(
                growth_stats.get("n_sky_hash_new", 0)
            )
            n_conflict = int(growth_stats.get("n_structure_hash_merged", 0))
            n_exact = int(growth_stats.get("n_structure_hash_hits", 0))
            n_near = int(growth_stats.get("n_structure_hash_near_hits", 0))
            n_suppressed = int(growth_stats.get("n_structure_vcd_suppressed", 0))
            label = (
                f"KF {int(frame_idx):04d} | anchors new={n_new} "
                f"conflict/merged={n_conflict} (exact={n_exact}, near={n_near}) "
                f"suppressed={n_suppressed}"
            )
            header_h = min(54, canvas.shape[0])
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                label,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(image_path, canvas)
        except Exception as exc:
            Log(f"[KFRender] annotate frame {frame_idx} failed: {exc}", tag="BackEnd")

    def _save_anchor_insert_mask_visualization(
        self,
        frame_idx: int,
        viewpoint,
        insert_mask,
        stats: dict | None = None,
    ) -> None:
        if not bool(self.config["Training"].get("debug_visualize_anchor_insert", False)):
            return
        if self.save_dir is None or insert_mask is None or viewpoint is None:
            return
        try:
            import cv2

            out_dir = os.path.join(self.save_dir, "anchor_insert_mask_vis")
            os.makedirs(out_dir, exist_ok=True)
            rgb = (
                viewpoint.original_image.detach().permute(1, 2, 0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
            mask_np = np.asarray(insert_mask, dtype=bool)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            overlay = rgb.copy()
            overlay[mask_np] = (
                0.25 * overlay[mask_np] + 0.75 * np.array([255, 255, 0])
            ).astype(np.uint8)
            canvas = np.ascontiguousarray(np.concatenate([rgb, overlay], axis=1)[:, :, ::-1])
            label = f"insert_ratio={float(mask_np.mean()):.4f}"
            if stats:
                label += f" pixels={int(stats.get('insert_pixels', int(mask_np.sum())))}"
                if "dap_scale_frame" in stats:
                    label += f" scale={float(stats.get('dap_scale_frame', 1.0)):.3f}"
                if stats.get("dia_skipped_unreliable_align", False):
                    label += " skipped_align"
            cv2.putText(
                canvas,
                label,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if canvas.shape[1] > 1920:
                scale = 1920 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                os.path.join(out_dir, f"frame_{int(frame_idx):04d}.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
        except Exception as exc:
            Log(f"[AnchorInsertVis] frame {frame_idx} save failed: {exc}", tag="BackEnd")

    def _project_world_points_to_erp(self, xyz: np.ndarray, viewpoint, h: int, w: int):
        if xyz is None or int(np.asarray(xyz).shape[0]) <= 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, empty, np.zeros((0,), dtype=bool)
        xyz = np.asarray(xyz, dtype=np.float64)
        R_np = viewpoint.R.float().cpu().numpy().astype(np.float64)
        T_np = viewpoint.T.float().cpu().numpy().astype(np.float64)
        pts_cam = (R_np @ xyz.T).T + T_np
        radius = np.linalg.norm(pts_cam, axis=1)
        valid = np.isfinite(radius) & (radius > 1e-6)
        radius_safe = np.maximum(radius, 1e-12)
        x = pts_cam[:, 0] / radius_safe
        y = pts_cam[:, 1] / radius_safe
        z = pts_cam[:, 2] / radius_safe
        lam = np.arctan2(x, z)
        phi = np.arcsin(np.clip(y, -1.0, 1.0))
        u = float(w) * (lam / (2.0 * np.pi) + 0.5) - 0.5
        v = float(h) * (phi / np.pi + 0.5) - 0.5
        valid &= np.isfinite(u) & np.isfinite(v) & (v >= 0.0) & (v <= h - 1.0)
        return (
            u.astype(np.float32),
            v.astype(np.float32),
            radius.astype(np.float32),
            valid.astype(bool),
        )

    def _save_new_anchor_visualization(
        self,
        frame_idx: int,
        viewpoint,
        growth_stats: dict | None,
        insert_mask=None,
    ) -> None:
        if not bool(self.config["Training"].get("debug_visualize_new_anchors", False)):
            return
        if (
            self.save_dir is None
            or viewpoint is None
            or self.gaussians is None
            or not isinstance(growth_stats, dict)
        ):
            return
        try:
            import cv2

            out_dir = os.path.join(self.save_dir, "new_anchor_vis")
            os.makedirs(out_dir, exist_ok=True)

            rgb = (
                viewpoint.original_image.detach().permute(1, 2, 0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
            h, w = rgb.shape[:2]

            candidate_overlay = rgb.copy()
            mask_np = None
            if insert_mask is not None:
                mask_np = np.asarray(insert_mask, dtype=bool)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                if tuple(mask_np.shape) != (h, w):
                    mask_np = None
            if mask_np is not None:
                candidate_overlay[mask_np] = (
                    0.25 * candidate_overlay[mask_np]
                    + 0.75 * np.array([255, 255, 0], dtype=np.float32)
                ).astype(np.uint8)

            start = int(growth_stats.get("new_structure_start", -1))
            count = int(growth_stats.get("new_structure_count", 0))
            n_total = int(self.gaussians.get_xyz.shape[0])
            if start < 0 or count <= 0 or start >= n_total:
                xyz = np.zeros((0, 3), dtype=np.float32)
            else:
                end = min(start + count, n_total)
                xyz = (
                    self.gaussians.get_xyz[start:end]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            u, v, depth_cam, visible = self._project_world_points_to_erp(xyz, viewpoint, h, w)
            point_mask = np.zeros((h, w), dtype=np.uint8)
            if visible.any():
                xs = np.mod(np.round(u[visible]).astype(np.int64), w)
                ys = np.clip(np.round(v[visible]).astype(np.int64), 0, h - 1)
                point_mask[ys, xs] = 255
                radius_px = max(
                    1,
                    int(self.config["Training"].get("debug_new_anchor_point_radius", 2)),
                )
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * radius_px + 1, 2 * radius_px + 1),
                )
                point_mask = cv2.dilate(point_mask, kernel, iterations=1)

            point_overlay = rgb.copy()
            point_pixels = point_mask > 0
            if point_pixels.any():
                point_overlay[point_pixels] = (
                    0.20 * point_overlay[point_pixels]
                    + 0.80 * np.array([255, 0, 255], dtype=np.float32)
                ).astype(np.uint8)

            canvas = np.ascontiguousarray(
                np.concatenate([rgb, candidate_overlay, point_overlay], axis=1)[:, :, ::-1]
            )
            label = (
                f"frame={int(frame_idx)} "
                f"new={int(xyz.shape[0])} visible={int(visible.sum())} "
                f"insert_enabled={int(growth_stats.get('n_insert_enabled_candidates', 0))} "
                f"miss_new={int(growth_stats.get('n_hash_miss_new', 0))} "
                f"suppressed={int(growth_stats.get('n_structure_vcd_suppressed', 0))}"
            )
            cv2.putText(
                canvas,
                label,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "original | insertion candidates | actual new anchors",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if canvas.shape[1] > 2400:
                scale = 2400 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                os.path.join(out_dir, f"frame_{int(frame_idx):04d}.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            if bool(self.config["Training"].get("debug_save_new_anchor_npz", True)):
                level = np.zeros((xyz.shape[0],), dtype=np.int8)
                if (
                    xyz.shape[0] > 0
                    and hasattr(self.gaussians, "_anchor_level")
                    and self.gaussians._anchor_level.shape[0] >= start + xyz.shape[0]
                ):
                    level = (
                        self.gaussians._anchor_level[start : start + xyz.shape[0]]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int8)
                    )
                np.savez_compressed(
                    os.path.join(out_dir, f"frame_{int(frame_idx):04d}_new_anchors.npz"),
                    xyz=xyz,
                    uv=np.stack([u, v], axis=1).astype(np.float32)
                    if u.size
                    else np.zeros((0, 2), dtype=np.float32),
                    depth_cam=depth_cam.astype(np.float32),
                    visible=visible.astype(bool),
                    level=level,
                )
        except Exception as exc:
            Log(f"[NewAnchorVis] frame {frame_idx} save failed: {exc}", tag="BackEnd")

    def _gaussian_ratio_regularization(self, init_phase: bool = False):
        """Anisotropy regulariser with two modes (selectable via config):

        mode="hinge"  (SC-OmniGS Eq.17, default):
            L = mean(relu(s_max/s_min - gamma))
            A soft hinge that adds zero penalty below the threshold and
            grows linearly above it.  gamma=10 (config: aniso_gamma).

        mode="log"  (legacy):
            L = smooth_l1(clamp(log(s_max/s_min) - log(soft_ratio), 0))
            Log-space penalty, tuned by gaussian_ratio_soft / _weight.
        """
        scaling = self.gaussians.get_scaling
        if init_phase:
            sky_mask = getattr(self.gaussians, "_is_sky", None)
            if sky_mask is not None and sky_mask.shape[0] == scaling.shape[0]:
                nonsky_mask = ~sky_mask.to(device=scaling.device, dtype=torch.bool)
                if nonsky_mask.any():
                    scaling = scaling[nonsky_mask]
                else:
                    return scaling.new_zeros(())
        s_max = scaling.max(dim=1).values
        s_min = scaling.min(dim=1).values.clamp(min=1e-6)

        aniso_mode = self.config["Training"].get("aniso_mode", "hinge")
        weight = float(
            self.config["Training"].get(
                "gaussian_ratio_reg_weight_init" if init_phase else "gaussian_ratio_reg_weight",
                6.0 if init_phase else 4.0,
            )
        )

        if aniso_mode == "hinge":
            gamma = float(self.config["Training"].get("aniso_gamma", 10.0))
            ratio = s_max / s_min
            aniso_loss = F.relu(ratio - gamma).mean()

            s_sorted, _ = scaling.sort(dim=1, descending=True)
            s_mid = s_sorted[:, 1].clamp(min=1e-6)
            needle_ratio = s_sorted[:, 0] / s_mid
            needle_gamma = float(self.config["Training"].get("needle_gamma", 5.0))
            needle_weight = float(self.config["Training"].get("needle_reg_weight", 3.0))
            needle_loss = (F.relu(needle_ratio - needle_gamma) ** 2).mean()

            return weight * aniso_loss + needle_weight * needle_loss
        else:
            # Legacy log-ratio soft penalty
            log_ratio = torch.log((s_max / s_min).clamp(min=1.0))
            soft_ratio = float(
                self.config["Training"].get(
                    "gaussian_ratio_soft_init" if init_phase else "gaussian_ratio_soft",
                    3.0 if init_phase else 2.5,
                )
            )
            excess = torch.clamp(log_ratio - np.log(soft_ratio), min=0.0)
            return weight * F.smooth_l1_loss(
                excess, torch.zeros_like(excess), reduction="mean"
            )

    def _sky_scale_regularization(self, init_phase: bool = False):
        """Soft penalty on oversized sky Gaussians.

        Works for both init and mapping phases.  The weight/cap values are
        read from separate config keys so each phase can be tuned independently:
          init:    sky_scale_reg_weight_init  / sky_scale_soft_cap_init
          mapping: sky_scale_reg_weight_mapping / sky_scale_soft_cap_mapping
        Setting weight or cap to 0 (or omitting the key) disables the penalty
        for that phase.
        """
        scaling = self.gaussians.get_scaling
        if scaling.numel() == 0:
            return scaling.new_zeros(())

        weight_key = "sky_scale_reg_weight_init" if init_phase else "sky_scale_reg_weight_mapping"
        cap_key    = "sky_scale_soft_cap_init"   if init_phase else "sky_scale_soft_cap_mapping"
        weight   = float(self.config["Training"].get(weight_key, 0.0))
        soft_cap = float(self.config["Training"].get(cap_key,    0.0))
        if weight <= 0.0 or soft_cap <= 0.0:
            return scaling.new_zeros(())

        sky_mask = getattr(self.gaussians, "_is_sky", None)
        if sky_mask is None or sky_mask.shape[0] != scaling.shape[0]:
            return scaling.new_zeros(())

        sky_mask = sky_mask.to(device=scaling.device, dtype=torch.bool)
        if not sky_mask.any():
            return scaling.new_zeros(())

        sky_scale_max = scaling[sky_mask].max(dim=1).values
        excess = F.relu(sky_scale_max - soft_cap)
        return (weight * excess.mean()).to(dtype=scaling.dtype)

    def _far_layer_scale_regularization(self, init_phase: bool = False):
        """Soft scale upper-bound penalty for 'far' layer (layer_id=1) Gaussians.

        Prevents far-field Gaussians from growing into large floating artifacts.
        Controlled by config:
          far_scale_reg_weight   (float, default 0.0 = disabled)
          far_scale_soft_cap     (float, default 20.0 metres)
        Setting weight to 0 disables the penalty.
        """
        if not bool(self.config["Training"].get("enable_layered_map", False)):
            return self.gaussians.get_scaling.new_zeros(())

        weight  = float(self.config["Training"].get("far_scale_reg_weight", 0.0))
        soft_cap = float(self.config["Training"].get("far_scale_soft_cap", 20.0))
        if weight <= 0.0 or soft_cap <= 0.0:
            return self.gaussians.get_scaling.new_zeros(())

        scaling = self.gaussians.get_scaling
        if scaling.numel() == 0:
            return scaling.new_zeros(())

        far_mask = self.gaussians.layer_mask(1, device=scaling.device)
        if not far_mask.any():
            return scaling.new_zeros(())

        far_scale_max = scaling[far_mask].max(dim=1).values
        excess = F.relu(far_scale_max - soft_cap)
        return (weight * excess.mean()).to(dtype=scaling.dtype)

    def _near_layer_prune(self):
        """Extra opacity-based pruning pass targeting 'near' layer (layer_id=0).

        Called every ``near_layer_prune_interval`` keyframes when
        ``enable_layered_map=True``.  Removes near-layer Gaussians whose
        opacity (sigmoid) has decayed below ``near_layer_prune_tau``.

        Returns:
            Number of near-layer Gaussians removed (0 if skipped or none).
        """
        if not bool(self.config["Training"].get("enable_layered_map", False)):
            return 0
        tau = float(self.config["Training"].get("near_layer_prune_tau", 0.05))
        near_mask = self.gaussians.layer_mask(0, device="cuda")
        opacity_vals = self.gaussians.get_opacity.squeeze(1)          # (N,)
        to_prune = near_mask & (opacity_vals < tau)
        if to_prune.any():
            n = int(to_prune.sum().item())
            self.gaussians.prune_points(to_prune)
            return n
        return 0

    def _get_init_prune_screen_size(self, viewpoint):
        if not isinstance(viewpoint, PanoramaCamera):
            return None

        training_cfg = self.config["Training"]
        if not bool(training_cfg.get("init_enable_sky_screen_prune", False)):
            return None

        threshold = training_cfg.get(
            "init_size_threshold",
            training_cfg.get("init_max_screen_size", training_cfg.get("size_threshold")),
        )
        if threshold is None:
            return None

        threshold = float(threshold)
        return threshold if threshold > 0 else None

    def _clamp_gaussian_ratios(self):
        ratio_cap = float(self.config["Training"].get("gaussian_ratio_cap", 0.0))
        # ratio_cap <= 1.0 means disabled.
        # WARNING: small caps (e.g. 8.0) destroy flat ground/surface Gaussians:
        # a pancake with s=[2, 2, 0.001] gets clamped to s=[0.008, 0.008, 0.001]
        # (completely invisible).  Only enable with a high cap (>= 50) if needed.
        if ratio_cap <= 1.0:
            return
        self.gaussians.clamp_scaling_ratios(ratio_cap)

    # Insert new Gaussians into the Gaussian scene based on the new keyframe's viewpoint and geometry
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        """Add Gaussians from a new keyframe using full valid non-sky depth."""
        viewpoint.mdl_insert_mask = None
        viewpoint.mdl_overlap = float(self._current_overlap)

        submap_id = int(getattr(viewpoint, "submap_id", -1))
        if not self.enable_submap:
            submap_id = 0
        self.active_submap_id = submap_id
        if self.enable_submap:
            self._submap_manager.assign_frame(frame_idx)
        self.submaps.setdefault(submap_id, {"kf_ids": []})
        if frame_idx not in self.submaps[submap_id]["kf_ids"]:
            self.submaps[submap_id]["kf_ids"].append(frame_idx)

        use_global_world_points = bool(getattr(viewpoint, "global_world_points_required", False))
        fastgs_insert_mask = None
        fastgs_insert_stats = None
        if self.use_anchor_scaffold and depth_map is not None and not use_global_world_points:
            fastgs_insert_mask, fastgs_insert_stats = self._build_anchor_fastgs_insert_mask(
                viewpoint,
                init=init,
            )
        dia_insert_mask = None
        dia_insert_stats = None
        if self.use_anchor_scaffold and depth_map is not None and not use_global_world_points:
            dia_insert_mask, dia_insert_stats = self._build_anchor_dia_insert_mask(
                viewpoint,
                depth_map,
                init=init,
            )
        anchor_insert_mask = None
        if fastgs_insert_mask is not None and dia_insert_mask is not None:
            anchor_insert_mask = np.logical_or(fastgs_insert_mask, dia_insert_mask)
        elif dia_insert_mask is not None:
            anchor_insert_mask = dia_insert_mask
        else:
            anchor_insert_mask = fastgs_insert_mask
        if self.use_anchor_scaffold and use_global_world_points:
            world_mask = getattr(viewpoint, "global_world_points_valid_mask", None)
            if world_mask is not None:
                if isinstance(world_mask, torch.Tensor):
                    anchor_insert_mask = world_mask.detach().cpu().numpy().astype(bool)
                else:
                    anchor_insert_mask = np.asarray(world_mask, dtype=bool)
                if anchor_insert_mask.ndim == 3:
                    anchor_insert_mask = anchor_insert_mask[0]
        if (
            self.use_anchor_scaffold
            and anchor_insert_mask is None
            and not init
            and bool(self.config["Training"].get("use_kf_novelty_mask_for_anchor_insert", True))
        ):
            novelty = getattr(viewpoint, "kf_novelty_mask", None)
            if novelty is not None:
                novelty_np = novelty.detach().cpu().numpy() if isinstance(novelty, torch.Tensor) else np.asarray(novelty)
                if novelty_np.ndim == 3:
                    novelty_np = novelty_np[0]
                if depth_map is not None and tuple(novelty_np.shape) == tuple(np.asarray(depth_map).shape):
                    anchor_insert_mask = novelty_np.astype(bool)
        if self.use_anchor_scaffold and anchor_insert_mask is not None:
            self._save_anchor_insert_mask_visualization(
                frame_idx,
                viewpoint,
                anchor_insert_mask,
                dia_insert_stats or fastgs_insert_stats,
            )

        self._ensure_anchor_hash_ready()
        extend_kwargs = {
            "kf_id": frame_idx,
            "init": init,
            "scale": scale,
            "depthmap": depth_map,
            "anchor_submap": submap_id,
            "birth_frame": frame_idx,
        }
        if self.use_anchor_scaffold:
            extend_kwargs.update(
                {
                    "anchor_hash_index": self.anchor_hash_index,
                    "pixel_mask": anchor_insert_mask,
                }
            )
        growth_stats = self.gaussians.extend_from_pcd_seq(
            viewpoint,
            **extend_kwargs,
        )
        if self.use_anchor_scaffold:
            self._save_new_anchor_visualization(
                frame_idx,
                viewpoint,
                growth_stats,
                insert_mask=anchor_insert_mask,
            )
        if (
            self.use_anchor_scaffold
            and isinstance(growth_stats, dict)
            and hasattr(self.gaussians, "record_anchor_replacement_evidence")
            and (
                dia_insert_mask is not None
                or bool(self.config["Training"].get("enable_global_anchor_prune", False))
            )
        ):
            inconsistent_mask = dia_insert_mask
            consistency = getattr(viewpoint, "erp_consistency_mask", None)
            if consistency is not None:
                consistency_np = (
                    consistency.detach().cpu().numpy()
                    if isinstance(consistency, torch.Tensor)
                    else np.asarray(consistency)
                )
                if consistency_np.ndim == 3:
                    consistency_np = consistency_np[0]
                if depth_map is not None and tuple(consistency_np.shape) == tuple(np.asarray(depth_map).shape):
                    inconsistent_mask = ~consistency_np.astype(bool)
            region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
            valid_region = region_masks.get("valid", None)
            evidence_stats = self.gaussians.record_anchor_replacement_evidence(
                viewpoint,
                inconsistent_mask=inconsistent_mask,
                aligned_depth=depth_map,
                valid_mask=valid_region,
                sky_mask=getattr(viewpoint, "erp_sky_mask", None),
                old_anchor_count=int(growth_stats.get("n_anchor_before", 0)),
                new_anchor_start=int(growth_stats.get("new_structure_start", -1)),
                new_anchor_count=int(growth_stats.get("new_structure_count", 0)),
            )
            growth_stats.update(evidence_stats)
        self._rebuild_submap_gaussian_indices()
        self._append_anchor_fastgs_insert_debug(
            frame_idx=frame_idx,
            metric_stats=fastgs_insert_stats or dia_insert_stats,
            growth_stats=growth_stats if isinstance(growth_stats, dict) else None,
        )
        if isinstance(growth_stats, dict):
            viewpoint.anchor_growth_stats = dict(growth_stats)
            self._annotate_frontend_kf_render(frame_idx, growth_stats)
        
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self._stability_tracker.reset()
        self._current_overlap = 1.0
        self.submaps = {}
        self.active_submap_id = -1
        self._submap_manager = SubmapManager(
            interval=int(self.config["Training"].get("submap_kf_interval", 10)),
            overlap_kfs=int(self.config["Training"].get("submap_overlap_kfs", 3)),
        )
        if self.use_anchor_scaffold:
            self.anchor_hash_index = AnchorHashIndex()

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            unpack_queue_message(self.backend_queue.get())

    def _select_mapping_window(self, current_window):
        if len(current_window) == 0:
            return []
        current_window = normalize_window_order(current_window)
        current_window = self._filter_registered_window(
            current_window, log_context="mapping"
        )
        if len(current_window) == 0:
            return []
        if not self.enable_submap:
            return current_window
        frame_to_submap = {
            int(frame_id): int(getattr(vp, "submap_id", -1))
            for frame_id, vp in self.viewpoints.items()
        }
        filtered = self._submap_manager.filter_window(
            list(current_window), frame_to_submap, self.active_submap_id
        )
        result = filtered if filtered else list(current_window)
        return self._filter_registered_window(result, log_context="mapping/submap")

    def _filter_registered_window(self, window, log_context="window"):
        window = normalize_window_order(window)
        registered = [int(kf) for kf in window if int(kf) in self.viewpoints]
        missing = [int(kf) for kf in window if int(kf) not in self.viewpoints]
        if missing:
            Log(
                f"[Window] backend dropped unregistered keyframes {missing} "
                f"during {log_context}; available={sorted(self.viewpoints.keys())}",
                tag="BackEnd",
            )
        return normalize_window_order(registered)

    def _submap_keyframes(self, submap_ids):
        if not self.enable_submap:
            return sorted([int(kf_id) for kf_id in self.viewpoints.keys()], reverse=True)
        kf_ids = []
        for submap_id in submap_ids:
            record = self._submap_manager.submaps.get(submap_id)
            if record is None:
                continue
            for kf_id in record.kf_ids:
                if kf_id in self.viewpoints and kf_id not in kf_ids:
                    kf_ids.append(int(kf_id))
        return sorted(kf_ids, reverse=True)

    def _rebuild_submap_gaussian_indices(self):
        if not self.enable_submap:
            return
        anchor_submap = getattr(self.gaussians, "_anchor_submap", None)
        if anchor_submap is None:
            return
        self._submap_manager.rebuild_gaussian_indices(anchor_submap)

    def submap_local_optimize(self, submap_ids, iters=1, up_pose=True):
        if not self.enable_submap:
            return
        submap_ids = [int(v) for v in submap_ids if int(v) >= 0]
        if not submap_ids:
            return
        local_window = self._submap_keyframes(submap_ids)
        if not local_window:
            return
        Log(
            f"[SubmapLocalOptimize] submaps={submap_ids} "
            f"kfs={local_window} iters={iters} up_pose={up_pose}",
            tag="BackEnd",
        )
        self.map(local_window, iters=int(iters), up_pose=up_pose)
        self._submap_manager.mark_refined(submap_ids, self.iteration_count)

    def _apply_loop_closure_update(self, new_poses_c2w, affected_submaps):
        from backend.legacy_360gs.utils.loop_closure import correct_gaussian_map

        old_poses = {}
        for frame_id, vp in self.viewpoints.items():
            w2c = getWorld2View2(vp.R, vp.T).detach().cpu().numpy()
            old_poses[int(frame_id)] = np.linalg.inv(w2c)
        correct_gaussian_map(
            self.gaussians,
            old_poses,
            new_poses_c2w,
            self.viewpoints,
            self.config,
        )
        if self.use_anchor_scaffold:
            self._rebuild_anchor_hash_tables(
                resolve_collisions=True,
                snap_xyz_to_voxel=True,
            )
            self._rebuild_submap_gaussian_indices()
        for frame_id, T_c2w in new_poses_c2w.items():
            vp = self.viewpoints.get(int(frame_id))
            if vp is None:
                continue
            submap_id = int(getattr(vp, "submap_id", -1))
            if self.enable_submap and submap_id >= 0:
                self._submap_manager.update_anchor_pose(submap_id, T_c2w)

        refine_iters = int(self.config["Training"].get("submap_local_refine_iters", 0))
        if (not self.enable_submap) or refine_iters <= 0 or not affected_submaps:
            return
        self._rebuild_submap_gaussian_indices()
        freeze_pose = bool(self.config["Training"].get("freeze_pose", False))
        self.submap_local_optimize(
            affected_submaps, iters=refine_iters, up_pose=(not freeze_pose)
        )

    def _append_sky_debug(
        self,
        phase: str,
        frame_idx: int,
        viewpoint,
        render_img,
        opacity,
        iteration=None,
        sky_bg_only=None,
        sky_bg_alpha=None,
    ):
        if self.save_dir is None or viewpoint is None or render_img is None or opacity is None:
            return
        try:
            supervision = _get_panorama_supervision(
                viewpoint,
                self.config,
                device=render_img.device,
                dtype=render_img.dtype,
                depth_shape=opacity.shape,
            )
            sky_mask = supervision["sky_rgb_mask"].to(device=render_img.device, dtype=torch.bool)
            _, H, W = render_img.shape
            top_pole = erp_top_latitude_mask(H, W, render_img.device, self.config)
            coverage = opacity.clamp(0.0, 1.0) > 0.10
            sky_luma = render_img.mean(dim=0, keepdim=True)
            upper_lat = sky_mask & (~top_pole)
            sky_cov = (
                float((coverage & sky_mask).float().sum().item())
                / max(float(sky_mask.float().sum().item()), 1.0)
            )
            top_cov = 0.0
            if top_pole is not None:
                top_sky_mask = sky_mask & top_pole
                top_cov = (
                    float((coverage & top_sky_mask).float().sum().item())
                    / max(float(top_sky_mask.float().sum().item()), 1.0)
                )
            upper_dark = 0.0
            if upper_lat.any():
                upper_dark = (
                    float(((sky_luma < 0.12) & upper_lat).float().sum().item())
                    / max(float(upper_lat.float().sum().item()), 1.0)
                )
            region_tag = getattr(self.gaussians, "_region_tag", None)
            polar_count = 0
            horizon_count = 0
            if region_tag is not None:
                polar_count = int(
                    (region_tag == self.gaussians.REGION_TAG_POLAR_CAP_SKY).sum().item()
                )
                horizon_count = int(
                    (region_tag == self.gaussians.REGION_TAG_HORIZON_BG).sum().item()
                )
            row = {
                "phase": phase,
                "frame_idx": int(frame_idx),
                "iteration": None if iteration is None else int(iteration),
                "sky_coverage_ratio": float(sky_cov),
                "top_pole_sky_coverage_ratio": float(top_cov),
                "upper_sky_dark_ratio": float(upper_dark),
                "polar_cap_gaussian_count": polar_count,
                "horizon_bg_gaussian_count": horizon_count,
            }
            if sky_bg_only is not None and sky_mask.any():
                gt_image = supervision["gt_image"]
                diff = (sky_bg_only - gt_image)[:, sky_mask.squeeze(0)]
                mse = float((diff * diff).mean().item()) if diff.numel() > 0 else 0.0
                row["sky_bg_psnr"] = 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))
                row["sky_bg_smoothness"] = float(
                    (
                        torch.abs(sky_bg_only[:, :, 1:] - sky_bg_only[:, :, :-1]).mean()
                        + torch.abs(sky_bg_only[:, 1:, :] - sky_bg_only[:, :-1, :]).mean()
                    ).item()
                )
            if sky_bg_alpha is not None and sky_mask.any():
                row["sky_bg_mean_alpha"] = float(
                    sky_bg_alpha[sky_mask].mean().item()
                )
            with open(os.path.join(self.save_dir, "sky_debug.jsonl"), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            Log(f"[BackEnd] _append_sky_debug failed: {e}")

    def _append_sky_opacity_stats(self, frame_idx: int) -> None:
        """Per-KF dump of (anchor projection in sky_mask) opacity distribution,
        stratified by ERP latitude band. Used to pick a safe per-band opacity
        threshold for sky-direction anchor pruning. See
        ``utils/sky_opacity_stats.py`` for the math.
        """
        if self.save_dir is None:
            return
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        if not bool(tr_cfg.get("enable_sky_opacity_stats", False)):
            return
        try:
            from backend.legacy_360gs.utils.sky_opacity_stats import (
                collect_sky_opacity_stats,
                DEFAULT_BANDS,
                DEFAULT_OPACITY_THRESHOLDS,
            )

            window = self._filter_registered_window(
                self.current_window, log_context="sky_opacity_stats"
            )
            keyframes = [
                self.viewpoints[k] for k in window if k in self.viewpoints
            ]
            if not keyframes:
                return
            bands = tr_cfg.get("sky_opacity_stats_bands", None)
            if not bands:
                bands = DEFAULT_BANDS
            else:
                bands = tuple((float(a), float(b)) for a, b in bands)
            thresholds = tr_cfg.get("sky_opacity_stats_thresholds", None)
            if not thresholds:
                thresholds = DEFAULT_OPACITY_THRESHOLDS
            else:
                thresholds = tuple(float(t) for t in thresholds)

            collect_sky_opacity_stats(
                self.gaussians,
                keyframes,
                frame_idx=int(frame_idx),
                out_path=os.path.join(self.save_dir, "sky_opacity_stats.jsonl"),
                bands=bands,
                opacity_thresholds=thresholds,
                extra_meta={"window": [int(k) for k in window]},
            )
        except Exception as e:
            Log(f"[BackEnd] _append_sky_opacity_stats failed: {e}")

    def _save_init_sky_panel(self, init_vis_dir: str, iteration: int, viewpoint, render_img, opacity):
        if init_vis_dir is None:
            return
        try:
            import cv2

            supervision = _get_panorama_supervision(
                viewpoint,
                self.config,
                device=render_img.device,
                dtype=render_img.dtype,
                depth_shape=opacity.shape,
            )
            sky_mask = supervision["sky_rgb_mask"].to(device=render_img.device, dtype=torch.bool)
            _, Hm, Wm = render_img.shape
            top_pole = erp_top_latitude_mask(Hm, Wm, render_img.device, self.config)
            polar_mask = sky_mask & top_pole
            uncovered = polar_mask & (opacity.clamp(0.0, 1.0) <= 0.10)

            def _to_bgr(t):
                return (
                    t.detach().permute(1, 2, 0).cpu().numpy() * 255
                ).clip(0, 255).astype(np.uint8)[:, :, ::-1]

            gt_bgr = _to_bgr(viewpoint.original_image.cuda().clamp(0, 1))
            render_bgr = _to_bgr(render_img.clamp(0, 1))
            overlay = gt_bgr.copy()
            sky_np = sky_mask.squeeze(0).detach().cpu().numpy()
            polar_np = polar_mask.squeeze(0).detach().cpu().numpy()
            uncovered_np = uncovered.squeeze(0).detach().cpu().numpy()
            overlay[sky_np] = (0.65 * overlay[sky_np] + 0.35 * np.array([255, 180, 80])).astype(np.uint8)
            overlay[polar_np] = (0.55 * overlay[polar_np] + 0.45 * np.array([80, 220, 255])).astype(np.uint8)
            overlay[uncovered_np] = np.array([0, 0, 255], dtype=np.uint8)

            H, W = gt_bgr.shape[:2]
            gap = 4
            canvas = np.zeros((H + 24, W * 3 + gap * 2, 3), dtype=np.uint8)
            canvas[24:, :W] = gt_bgr
            canvas[24:, W + gap:2 * W + gap] = render_bgr
            canvas[24:, 2 * W + 2 * gap:] = overlay
            cv2.putText(
                canvas,
                f"init iter {iteration:04d}  GT | Render | Sky/TopPole/Uncovered",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if canvas.shape[1] > 2400:
                sc = 2400 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * sc), int(canvas.shape[0] * sc)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                os.path.join(init_vis_dir, f"init_iter_{iteration:04d}_sky_panel.png"),
                canvas,
            )
        except Exception as e:
            Log(f"[BackEnd] _save_init_sky_panel failed: {e}")
    def _save_init_vis(self, init_vis_dir: str, iteration: int, viewpoint, render_img, gs_only=None, sky_bg_only=None):
        """Save initialization render comparison panels under init_vis."""
        try:
            import cv2

            def _to_bgr(t):
                return (
                    t.detach().permute(1, 2, 0).cpu().numpy() * 255
                ).clip(0, 255).astype(np.uint8)[:, :, ::-1]

            gt_bgr     = _to_bgr(viewpoint.original_image.cuda().clamp(0, 1))
            final_bgr = _to_bgr(render_img.clamp(0, 1))
            gs_bgr = _to_bgr(gs_only.clamp(0, 1)) if gs_only is not None else final_bgr
            bg_bgr = _to_bgr(sky_bg_only.clamp(0, 1)) if sky_bg_only is not None else np.zeros_like(final_bgr)
            H, W = gt_bgr.shape[:2]
            gap = 4
            canvas = np.zeros((H + 24, W * 4 + gap * 3, 3), dtype=np.uint8)
            canvas[24:, :W] = gt_bgr
            canvas[24:, W + gap:2 * W + gap] = gs_bgr
            canvas[24:, 2 * W + 2 * gap:3 * W + 2 * gap] = bg_bgr
            canvas[24:, 3 * W + 3 * gap:] = final_bgr
            cv2.putText(
                canvas, f"init iter {iteration:04d}  GT | GS-only | BG-only | Final",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            if canvas.shape[1] > 1920:
                sc = 1920 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * sc), int(canvas.shape[0] * sc)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                os.path.join(init_vis_dir, f"init_iter_{iteration:04d}.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
        except Exception as e:
            Log(f"[BackEnd] _save_init_vis failed: {e}")

    # Initialize the SLAM map by optimizing Gaussians through multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        if isinstance(viewpoint, PanoramaCamera) and self.save_dir is not None:
            init_vis_dir = os.path.join(
                self.config["Results"]["save_dir"], "init_vis"
            )
            os.makedirs(init_vis_dir, exist_ok=True)
        else:
            init_vis_dir = None

        skip_sky_bg_in_init = bool(
            self.config.get("Training", {}).get("erp_sky_bg_skip_during_map_init", True)
        )

        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            if isinstance(viewpoint, PanoramaCamera):
                # Use zero deltas (detached) during map init 鈥?no pose opt yet
                _theta0 = torch.zeros(1, 3, device=self.device)
                _rho0   = torch.zeros(1, 3, device=self.device)
                render_pkg = render_panorama_for_config(
                    viewpoint, self.gaussians, self.pipeline_params, self.background,
                    config=self.config,
                    theta=_theta0, rho=_rho0,
                    skip_erp_sky_bg=skip_sky_bg_in_init,
                )
                image     = render_pkg["render"]
                depth     = render_pkg["depth"]
                opacity   = render_pkg["opacity"]
                n_touched = render_pkg["n_touched"]
                loss_init = get_loss_mapping(
                    self.config,
                    image,
                    viewpoint,
                    depth=depth,
                    initialization=True,
                ) + self._gaussian_ratio_regularization(
                    init_phase=True
                )
            else:
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_init = get_loss_mapping(
                    self.config,
                    image,
                    viewpoint,
                    depth=depth,
                    initialization=True,
                ) + self._gaussian_ratio_regularization(
                    init_phase=True
                )
            loss_init.backward()

            with torch.no_grad():
                skip_anchor_densify_stats = self._skip_anchor_explicit_densify_stats()
                if isinstance(viewpoint, PanoramaCamera):
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]
                    viewspace_point_tensor = render_pkg["viewspace_points"]
                    if not skip_anchor_densify_stats:
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter
                        )
                else:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(  
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussians.add_densification_stats(                 
                        viewspace_point_tensor, visibility_filter
                    )
                if mapping_iteration % self.init_gaussian_update == 0:  
                    init_prune_size_threshold = self._get_init_prune_screen_size(
                        viewpoint
                    )
                    n_before = self.gaussians.get_xyz.shape[0]
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        init_prune_size_threshold,
                        init_phase=True,
                    )
                    if self.use_anchor_scaffold and self.gaussians.get_xyz.shape[0] != n_before:
                        self._rebuild_anchor_hash_tables(
                            resolve_collisions=False,
                            snap_xyz_to_voxel=False,
                        )
                        self._rebuild_submap_gaussian_indices()

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    if not isinstance(viewpoint, PanoramaCamera):
                        self.gaussians.reset_opacity()
                    elif bool(
                        self.config["Training"].get(
                            "panorama_init_reset_sky_opacity", False
                        )
                    ):
                        self.gaussians.reset_opacity()
                    else:
                        # 鍏ㄦ櫙榛樿锛氫粎閲嶇疆闈炲ぉ绌洪珮鏂紝淇濇姢澶╃┖楂樻柉棰滆壊涓嶈鐮村潖
                        self.gaussians.reset_opacity_nonsky()

                # --- True sky LR amplification (post-step scaling) ---
                # Adam normalises every gradient by its running RMS, so
                # gradient-scaling is cancelled.  Instead we snapshot sky
                # params before the step, let Adam run normally, then
                # rescale the actual update by sky_lr_mult_init.
                _sky_lr_mult = float(
                    self.config["Training"].get("sky_lr_mult_init", 1.0)
                )
                _old_sky = (
                    self.gaussians.save_sky_params()
                    if _sky_lr_mult > 1.0
                    else {}
                )
                self._freeze_xyz_gradients(self._build_xyz_freeze_mask())
                self.gaussians.optimizer.step()
                if _sky_lr_mult > 1.0:
                    self.gaussians.amplify_sky_adam_update(_sky_lr_mult, _old_sky)
                self._clamp_gaussian_ratios()
                max_abs_s = float(self.config["Training"].get("max_abs_scaling", 0.0))
                if max_abs_s > 0:
                    self.gaussians.clamp_max_scaling(max_abs_s)
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if init_vis_dir is not None and isinstance(viewpoint, PanoramaCamera) and (
                    mapping_iteration % 100 == 0
                    or mapping_iteration == self.init_itr_num - 1
                ):
                    with torch.no_grad():
                        _t0 = torch.zeros(1, 3, device=self.device)
                        _r0 = torch.zeros(1, 3, device=self.device)
                        _vis_pkg = render_panorama_for_config(
                            viewpoint,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                            config=self.config,
                            theta=_t0,
                            rho=_r0,
                            skip_erp_sky_bg=skip_sky_bg_in_init,
                        )
                    self._save_init_vis(
                        init_vis_dir,
                        mapping_iteration,
                        viewpoint,
                        _vis_pkg["render"],
                        gs_only=_vis_pkg.get("gs_only"),
                        sky_bg_only=_vis_pkg.get("sky_bg_only"),
                    )
                    self._save_init_sky_panel(
                        init_vis_dir,
                        mapping_iteration,
                        viewpoint,
                        _vis_pkg["render"],
                        _vis_pkg["opacity"],
                    )
                    self._append_sky_debug(
                        phase="init",
                        frame_idx=cur_frame_idx,
                        viewpoint=viewpoint,
                        render_img=_vis_pkg["render"],
                        opacity=_vis_pkg["opacity"],
                        iteration=mapping_iteration,
                        sky_bg_only=_vis_pkg.get("sky_bg_only"),
                        sky_bg_alpha=_vis_pkg.get("sky_bg_alpha"),
                    )

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long().cpu()
        self._append_anchor_debug(
            frame_idx=cur_frame_idx,
            render_loss=float(loss_init.detach().item()) if torch.is_tensor(loss_init) else None,
        )
        Log("Initialized map")
        return render_pkg
    # Optimize keyframe poses and Gaussians scene
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return
        current_window = self._select_mapping_window(current_window)
        if len(current_window) == 0:
            return
        profile_detailed = bool(
            self.config["Training"].get("profile_backend_detailed", False)
        )

        def _sync():
            if profile_detailed and torch.cuda.is_available():
                torch.cuda.synchronize()

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        total_kf_count = len(self.viewpoints)
        protect_early_gaussians = total_kf_count < self.early_prune_kf_count
        frames_to_optimize = self.config["Training"]["pose_window"]

        # face_weights needed for both current-window and random-replay panoramic renders
        face_weights = self.config["Training"].get(
            "face_weights", [1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        )

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items():  # Add viewpoints outside the current window to the random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)        
            
        for _ in range(iters):
            map_iter_start = time.perf_counter()
            self.iteration_count += 1
            self.last_sent += 1
            current_window_render_ms = 0.0
            replay_render_ms = 0.0
            regularization_ms = 0.0
            backward_ms = 0.0
            occ_visibility_ms = 0.0
            densify_stats_ms = 0.0
            structure_update_ms = 0.0
            structure_phase_ms = {}
            occupancy_prune_ms = 0.0
            occupancy_pruned = 0
            occupancy_submap_rebuild_ms = 0.0
            opacity_reset_ms = 0.0
            stability_mask_ms = 0.0
            optimizer_step_ms = 0.0
            needle_cleanup_ms = 0.0
            zero_grad_ms = 0.0
            lr_update_ms = 0.0
            keyframe_optimizer_ms = 0.0
            pose_update_ms = 0.0
            local_anchor_stats = {
                "n_protected_static": 0,
                "n_local_active": 0,
                "n_frozen_grad_zeroed": 0,
            }

            loss_mapping = 0
            viewspace_point_tensor_acm = []                 
            visibility_filter_acm = []                      
            radii_acm = []                                  
            n_touched_acm = []                            
            n_touched_kf_ids = []

            keyframes_opt = []          

            _sync()
            stage_start = time.perf_counter()
            random_window_frame = bool(
                self.config["Training"].get("random_window_frame_per_iter", False)
            ) and not prune
            if random_window_frame and len(current_window) > 1:
                current_cam_indices = [random.randrange(len(current_window))]
            else:
                current_cam_indices = list(range(len(current_window)))
            for cam_idx in current_cam_indices:      # Render selected current-window frames and compute loss
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                if isinstance(viewpoint, PanoramaCamera):
                    # Direct ERP rendering 鈥?differentiable w.r.t. pose deltas.
                    # When pose is frozen (up_pose=False), pass explicit zero
                    # deltas to prevent gradient accumulation in cam_*_delta.
                    if up_pose:
                        pano_pkg = render_panorama_for_config(
                            viewpoint, self.gaussians, self.pipeline_params, self.background,
                            config=self.config,
                        )
                    else:
                        _t0 = torch.zeros(1, 3, device=self.device)
                        _r0 = torch.zeros(1, 3, device=self.device)
                        pano_pkg = render_panorama_for_config(
                            viewpoint, self.gaussians, self.pipeline_params, self.background,
                            config=self.config,
                            theta=_r0, rho=_t0,
                        )
                    erp_render = pano_pkg["render"]   # (C, H, W)
                    erp_depth  = pano_pkg["depth"]    # (1, H, W)
                    loss_mapping += get_loss_mapping(
                        self.config, erp_render, viewpoint, depth=erp_depth, monodepth=True
                    )
                    loss_mapping += self._neural_sky_alpha_loss(pano_pkg, viewpoint)
                    # Densification stats from ERP render
                    viewspace_point_tensor_acm.append(pano_pkg["viewspace_points"])
                    visibility_filter_acm.append(pano_pkg["visibility_filter"])
                    radii_acm.append(pano_pkg["radii"])
                    n_touched_acm.append(pano_pkg["n_touched"])
                    n_touched_kf_ids.append(current_window[cam_idx])
                else:
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                    loss_mapping += self._neural_sky_alpha_loss(render_pkg, viewpoint)
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                    n_touched_acm.append(n_touched)
                    n_touched_kf_ids.append(current_window[cam_idx])
            _sync()
            current_window_render_ms = (time.perf_counter() - stage_start) * 1000.0

            # In each iteration, randomly select some non-window keyframes
            # for optimization. Replay strength is configurable via
            # ``Training.replay_random_kfs`` (default 2 for legacy parity).
            _sync()
            stage_start = time.perf_counter()
            replay_n = int(
                self.config["Training"].get("replay_random_kfs", 2)
            )
            if replay_n < 0:
                replay_n = 0
            replay_n = min(replay_n, len(random_viewpoint_stack))
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:replay_n]:
                viewpoint = random_viewpoint_stack[cam_idx]

                if isinstance(viewpoint, PanoramaCamera):
                    # Random-replay keyframe: ERP rendering (no pose opt)
                    _theta0 = torch.zeros(1, 3, device=self.device)
                    _rho0   = torch.zeros(1, 3, device=self.device)
                    pano_pkg = render_panorama_for_config(
                        viewpoint, self.gaussians, self.pipeline_params, self.background,
                        config=self.config,
                        theta=_theta0, rho=_rho0,
                    )
                    erp_render = pano_pkg["render"]
                    erp_depth = pano_pkg["depth"]
                    loss_mapping += get_loss_mapping(
                        self.config, erp_render, viewpoint, depth=erp_depth, monodepth=True
                    )
                    loss_mapping += self._neural_sky_alpha_loss(pano_pkg, viewpoint)
                    viewspace_point_tensor_acm.append(pano_pkg["viewspace_points"])
                    visibility_filter_acm.append(pano_pkg["visibility_filter"])
                    radii_acm.append(pano_pkg["radii"])
                else:
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                    loss_mapping += self._neural_sky_alpha_loss(render_pkg, viewpoint)
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
            _sync()
            replay_render_ms = (time.perf_counter() - stage_start) * 1000.0

            _sync()
            stage_start = time.perf_counter()
            loss_mapping += self._gaussian_ratio_regularization(init_phase=False)
            loss_mapping += self._far_layer_scale_regularization(init_phase=False)
            _sync()
            regularization_ms = (time.perf_counter() - stage_start) * 1000.0
            _sync()
            stage_start = time.perf_counter()
            loss_mapping.backward()
            _sync()
            backward_ms = (time.perf_counter() - stage_start) * 1000.0
            gaussian_split = False
            
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                _sync()
                stage_start = time.perf_counter()
                self.occ_aware_visibility = {}
                _n_touched_sum = None
                for idx, kf_idx in enumerate(n_touched_kf_ids):
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long().cpu()
                    # Accumulate n_touched for EMA stability update
                    if _n_touched_sum is None:
                        _n_touched_sum = n_touched.clone().float()
                    else:
                        _n_touched_sum = _n_touched_sum + n_touched.float()
                # Phase 4: update EMA stability tracker
                if _n_touched_sum is not None:
                    self._stability_tracker.update(_n_touched_sum)
                _sync()
                occ_visibility_ms = (time.perf_counter() - stage_start) * 1000.0

                # Only prune on the last iteration and when we have full window
                if prune:
                    _sync()
                    stage_start = time.perf_counter()
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = self.config["Training"]["prune_num"]  # prune parameter
                        enable_occupancy_prune = bool(
                            self.config["Training"].get("enable_occupancy_prune", True)
                        )
                        if self._anchor_fastgs_active():
                            enable_occupancy_prune = False
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if enable_occupancy_prune:
                            if prune_mode == "odometry":
                                to_prune = self.gaussians.n_obs < 3
                                # make sure we don't split the gaussians, break here.
                            if prune_mode == "slam":
                                sorted_window = sorted(current_window, reverse=True)
                                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                                if not self.initialized:
                                    mask = self.gaussians.unique_kfIDs >= 0
                                to_prune = torch.logical_and(
                                    self.gaussians.n_obs <= prune_coviz, mask
                                )
                                # Protection 1: preserve high-opacity Gaussians
                                high_opacity = (self.gaussians.get_opacity.squeeze() > 0.8).cpu()
                                to_prune = to_prune & ~high_opacity
                                # Protection 2: minimum Gaussian count guard
                                min_gaussians = int(
                                    self.config["Training"].get("min_gaussians_after_prune", 5000)
                                )
                                n_remaining = int((~to_prune).sum().item())
                                if n_remaining < min_gaussians:
                                    n_can_prune = max(
                                        0, self.gaussians.get_xyz.shape[0] - min_gaussians
                                    )
                                    if n_can_prune > 0:
                                        prune_indices = torch.where(to_prune)[0]
                                        opacities = self.gaussians.get_opacity.squeeze().cpu()[
                                            prune_indices
                                        ]
                                        _, keep_order = opacities.sort(descending=True)
                                        n_keep = max(0, len(prune_indices) - n_can_prune)
                                        to_prune[prune_indices[keep_order[:n_keep]]] = False
                                    else:
                                        to_prune.fill_(False)
                                # Protection 3: protect sky Gaussians from occupancy pruning
                                # (controlled by protect_sky_occupancy_prune, default True).
                                if bool(
                                    self.config["Training"].get(
                                        "protect_sky_occupancy_prune", True
                                    )
                                ):
                                    if self.gaussians._is_sky.shape[0] == to_prune.shape[0]:
                                        sky_cpu = self.gaussians._is_sky.to(dtype=torch.bool)
                                        to_prune = to_prune & ~sky_cpu
                                region_tag = getattr(self.gaussians, "_region_tag", None)
                                if region_tag is not None and region_tag.shape[0] == to_prune.shape[0]:
                                    horizon_bg = region_tag == self.gaussians.REGION_TAG_HORIZON_BG
                                    upper_sky = region_tag == self.gaussians.REGION_TAG_UPPER_SKY
                                    polar_cap = region_tag == self.gaussians.REGION_TAG_POLAR_CAP_SKY
                                    bottom = region_tag == self.gaussians.REGION_TAG_BOTTOM_POLE_GROUND
                                    min_bottom_obs = int(
                                        self.config["Training"].get(
                                            "region_min_obs_before_prune_bottom_pole", 4
                                        )
                                    )
                                    immature_bottom = bottom & (
                                        self.gaussians.n_obs < min_bottom_obs
                                    )
                                    protected = (
                                        horizon_bg | upper_sky | polar_cap | immature_bottom
                                    )
                                    if protected.any():
                                        to_prune = to_prune & ~protected
                                        Log(
                                            f"[RegionTag] occupancy protected "
                                            f"upper={int(upper_sky.sum().item())} "
                                            f"polar={int(polar_cap.sum().item())} "
                                            f"horizon={int(horizon_bg.sum().item())} "
                                            f"bottom_immature={int(immature_bottom.sum().item())}",
                                            tag="BackEnd",
                                        )
                        n_total = self.gaussians.get_xyz.shape[0]
                        n_prune = int(to_prune.sum().item()) if to_prune is not None else 0
                        occupancy_pruned = n_prune
                        if enable_occupancy_prune and n_prune > 0:
                            Log(
                                f"Occupancy prune: {n_prune}/{n_total} Gaussians "
                                f"(mode={prune_mode}, coviz<={prune_coviz})"
                            )
                        if to_prune is not None and self.monocular:
                            _sync()
                            rebuild_start = time.perf_counter()
                            self.gaussians.prune_points(to_prune.cuda())
                            if self.use_anchor_scaffold:
                                self._rebuild_anchor_hash_tables(
                                    resolve_collisions=False,
                                    snap_xyz_to_voxel=False,
                                )
                            self._rebuild_submap_gaussian_indices()
                            _sync()
                            occupancy_submap_rebuild_ms = (
                                time.perf_counter() - rebuild_start
                            ) * 1000.0
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    _sync()
                    occupancy_prune_ms = (time.perf_counter() - stage_start) * 1000.0
                    self._append_jsonl(
                        "backend_perf.jsonl",
                        {
                            "event": "map_iter",
                            "iteration": int(self.iteration_count),
                            "window_size": int(len(current_window)),
                            "sampled_window_size": int(len(current_cam_indices)),
                            "sampled_window_kfs": [int(current_window[idx]) for idx in current_cam_indices],
                            "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                            "update_gaussian": False,
                            "prune_only": True,
                            "gaussian_split": False,
                            "structure_update_ms": 0.0,
                            "current_window_render_ms": float(current_window_render_ms),
                            "replay_render_ms": float(replay_render_ms),
                            "regularization_ms": float(regularization_ms),
                            "backward_ms": float(backward_ms),
                            "occ_visibility_ms": float(occ_visibility_ms),
                            "occupancy_prune_ms": float(occupancy_prune_ms),
                            "occupancy_pruned": int(occupancy_pruned),
                            "occupancy_submap_rebuild_ms": float(occupancy_submap_rebuild_ms),
                            "map_iter_ms": (time.perf_counter() - map_iter_start) * 1000.0,
                        },
                    )
                    return False

                _sync()
                stage_start = time.perf_counter()
                if not self._skip_anchor_explicit_densify_stats():
                    for idx in range(len(viewspace_point_tensor_acm)):
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                            radii_acm[idx][visibility_filter_acm[idx]],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                        )
                _sync()
                densify_stats_ms = (time.perf_counter() - stage_start) * 1000.0

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    structure_start = time.perf_counter()
                    prune_opacity_th = (
                        self.early_gaussian_th
                        if protect_early_gaussians
                        else self.gaussian_th
                    )
                    prune_size_threshold = (
                        None if protect_early_gaussians else self.size_threshold
                    )
                    mapping_grad_th = float(
                        self.config["Training"].get(
                            "mapping_densify_grad_threshold",
                            self.opt_params.densify_grad_threshold,
                        )
                    )
                    current_kf_id = (
                        max(self.current_window) if self.current_window else None
                    )
                    importance_score = None
                    pruning_score = None
                    fastgs_debug = None
                    enable_fastgs_vcp = False
                    should_compute_fastgs_score = False
                    if self.enable_fastgs_erp:
                        from backend.legacy_360gs.utils.fastgs_erp import compute_gaussian_score_fastgs_erp

                        enable_fastgs_vcp = (
                            not self.fastgs_vcd_only
                            and len(current_window) == self.window_size
                            and len(self.viewpoints) >= self.fastgs_vcp_warmup_kfs
                        )
                        should_compute_fastgs_score = (
                            (not self.use_anchor_scaffold) or bool(enable_fastgs_vcp)
                        )
                        if should_compute_fastgs_score:
                            importance_score, pruning_score, fastgs_debug = (
                                compute_gaussian_score_fastgs_erp(
                                    viewpoint_stack,
                                    self.gaussians,
                                    self.background,
                                    self.config,
                                    replay_viewpoints=random_viewpoint_stack,
                                    optimize_uids=set(int(kf_id) for kf_id in current_window),
                                )
                            )
                    n_before = self.gaussians.get_xyz.shape[0]
                    densify_stats = self.gaussians.densify_and_prune(
                        mapping_grad_th,
                        prune_opacity_th,
                        self.gaussian_extent,
                        prune_size_threshold,
                        current_kf_id=current_kf_id,
                        importance_score=importance_score,
                        pruning_score=pruning_score if enable_fastgs_vcp else None,
                        fastgs_enabled=self.enable_fastgs_erp,
                        fastgs_vcd_only=(not enable_fastgs_vcp),
                        current_window=current_window,
                    )
                    n_after = self.gaussians.get_xyz.shape[0]
                    structure_update_ms = (time.perf_counter() - structure_start) * 1000.0
                    structure_phase_ms = densify_stats.get("phase_ms", {})
                    if abs(n_after - n_before) > 0:
                        Log(f"Densify: {n_before} -> {n_after} ({n_after - n_before:+d}) "
                            f"[kf={current_kf_id}, grad_th={mapping_grad_th:.4f}]")
                    if n_after != n_before:
                        _sync()
                        rebuild_start = time.perf_counter()
                        if self.use_anchor_scaffold:
                            self._rebuild_anchor_hash_tables(
                                resolve_collisions=False,
                                snap_xyz_to_voxel=False,
                            )
                        self._rebuild_submap_gaussian_indices()
                        _sync()
                        structure_phase_ms["submap_rebuild_ms"] = (
                            time.perf_counter() - rebuild_start
                        ) * 1000.0
                    gaussian_split = True
                    if self.enable_fastgs_erp and self.fastgs_debug_log_scores and fastgs_debug is not None:
                        event = {
                            "event": "fastgs_score",
                            "iteration": int(self.iteration_count),
                            "score_mode": "anchor" if self.use_anchor_scaffold else "gaussian",
                            "current_kf_id": int(current_kf_id) if current_kf_id is not None else -1,
                            "window_kfs": [int(kf_id) for kf_id in current_window],
                            "window_view_uids": fastgs_debug.get("window_uids", []),
                            "replay_view_uids": fastgs_debug.get("replay_uids", []),
                            "num_views": int(fastgs_debug.get("num_views", 0)),
                            "enable_vcp": bool(enable_fastgs_vcp),
                            "importance_stats": fastgs_debug.get("importance_stats", {}),
                            "pruning_stats": fastgs_debug.get("pruning_stats", {}),
                            "view_stats": fastgs_debug.get("view_stats", []),
                            "n_before": int(n_before),
                            "n_after": int(n_after),
                            "densify_clone": int(densify_stats.get("n_clone", 0)),
                            "densify_split": int(densify_stats.get("n_split", 0)),
                            "fastgs_prune_candidates": int(
                                densify_stats.get("fastgs_prune_candidates", 0)
                            ),
                            "fastgs_pruned": int(densify_stats.get("fastgs_pruned", 0)),
                            "structure_update_ms": float(structure_update_ms),
                        }
                        self._append_jsonl("fastgs_debug.jsonl", event)
                        self._write_yaml("fastgs_score_stats.yml", event)

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    if not (protect_early_gaussians and self.early_disable_opacity_reset):
                        _sync()
                        stage_start = time.perf_counter()
                        Log("Resetting the opacity of non-visible Gaussians")
                        # Compute protected mask so that early-KF anchors
                        # (outside the current mapping window when Fix2 is on)
                        # are not hard-overwritten by reset.
                        reset_protected_mask = None
                        if self.use_anchor_scaffold and hasattr(
                            self.gaussians, "get_protected_anchor_mask"
                        ):
                            reset_kf_id = (
                                max(current_window) if current_window else None
                            )
                            reset_protected_mask = (
                                self.gaussians.get_protected_anchor_mask(
                                    current_window=current_window,
                                    current_kf_id=reset_kf_id,
                                    device=self.device,
                                )
                            )
                        self.gaussians.reset_opacity_nonvisible(
                            visibility_filter_acm,
                            protected_mask=reset_protected_mask,
                        )
                        _sync()
                        opacity_reset_ms = (time.perf_counter() - stage_start) * 1000.0
                        gaussian_split = True

                # Phase 4: mask gradients of stable Gaussians before step
                _sync()
                stage_start = time.perf_counter()
                if not self.use_anchor_scaffold:
                    self._stability_tracker.mask_stable_gradients(self.gaussians)
                current_kf_id_for_freeze = max(current_window) if current_window else None
                local_anchor_stats = self._mask_inactive_anchor_gradients(
                    current_kf_id_for_freeze,
                    current_window=current_window,
                )
                self._freeze_xyz_gradients(self._build_xyz_freeze_mask())
                _sync()
                stability_mask_ms = (time.perf_counter() - stage_start) * 1000.0
                _sync()
                stage_start = time.perf_counter()
                self.gaussians.optimizer.step()
                self._clamp_gaussian_ratios()
                max_abs_s = float(self.config["Training"].get("max_abs_scaling", 0.0))
                if max_abs_s > 0:
                    self.gaussians.clamp_max_scaling(max_abs_s)
                _sync()
                optimizer_step_ms = (time.perf_counter() - stage_start) * 1000.0
                needle_cleanup_every = int(
                    self.config["Training"].get("needle_cleanup_every", 500)
                )
                if needle_cleanup_every > 0 and self.iteration_count % needle_cleanup_every == 0:
                    _sync()
                    stage_start = time.perf_counter()
                    n_pruned = self.gaussians.prune_needles(
                        max_ratio=float(self.config["Training"].get("needle_prune_ratio", 50.0)),
                        protect_sky=True,
                    )
                    if n_pruned > 0:
                        self._rebuild_submap_gaussian_indices()
                        Log(f"Needle cleanup: pruned {n_pruned} elongated Gaussians "
                            f"(ratio > {self.config['Training'].get('needle_prune_ratio', 50.0)})")
                    _sync()
                    needle_cleanup_ms = (time.perf_counter() - stage_start) * 1000.0
                _sync()
                stage_start = time.perf_counter()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                _sync()
                zero_grad_ms = (time.perf_counter() - stage_start) * 1000.0
                _sync()
                stage_start = time.perf_counter()
                self.gaussians.update_learning_rate(self.iteration_count)
                _sync()
                lr_update_ms = (time.perf_counter() - stage_start) * 1000.0
                if self.keyframe_optimizers is not None:
                    _sync()
                    stage_start = time.perf_counter()
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    _sync()
                    keyframe_optimizer_ms = (time.perf_counter() - stage_start) * 1000.0
                # Pose update
                if up_pose:
                    _sync()
                    stage_start = time.perf_counter()
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
                    _sync()
                    pose_update_ms = (time.perf_counter() - stage_start) * 1000.0
                self._append_jsonl(
                    "backend_perf.jsonl",
                    {
                        "event": "map_iter",
                        "iteration": int(self.iteration_count),
                        "window_size": int(len(current_window)),
                        "sampled_window_size": int(len(current_cam_indices)),
                        "sampled_window_kfs": [int(current_window[idx]) for idx in current_cam_indices],
                        "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                        "update_gaussian": bool(update_gaussian),
                        "prune_only": False,
                        "gaussian_split": bool(gaussian_split),
                        "structure_update_ms": float(structure_update_ms),
                        "current_window_render_ms": float(current_window_render_ms),
                        "replay_render_ms": float(replay_render_ms),
                        "regularization_ms": float(regularization_ms),
                        "backward_ms": float(backward_ms),
                        "occ_visibility_ms": float(occ_visibility_ms),
                        "densify_stats_ms": float(densify_stats_ms),
                        "opacity_reset_ms": float(opacity_reset_ms),
                        "stability_mask_ms": float(stability_mask_ms),
                        "optimizer_step_ms": float(optimizer_step_ms),
                        "needle_cleanup_ms": float(needle_cleanup_ms),
                        "zero_grad_ms": float(zero_grad_ms),
                        "lr_update_ms": float(lr_update_ms),
                        "keyframe_optimizer_ms": float(keyframe_optimizer_ms),
                        "pose_update_ms": float(pose_update_ms),
                        "occupancy_prune_ms": 0.0,
                        "occupancy_pruned": 0,
                        "occupancy_submap_rebuild_ms": 0.0,
                        "structure_grads_ms": float(structure_phase_ms.get("grads_ms", 0.0)),
                        "structure_fastgs_mask_ms": float(structure_phase_ms.get("fastgs_mask_ms", 0.0)),
                        "structure_clone_ms": float(structure_phase_ms.get("clone_ms", 0.0)),
                        "structure_split_ms": float(structure_phase_ms.get("split_ms", 0.0)),
                        "structure_prune_mask_ms": float(structure_phase_ms.get("prune_mask_ms", 0.0)),
                        "structure_prune_apply_ms": float(structure_phase_ms.get("prune_apply_ms", 0.0)),
                        "structure_submap_rebuild_ms": float(structure_phase_ms.get("submap_rebuild_ms", 0.0)),
                        "structure_pruned": int(densify_stats.get("n_pruned", 0)) if update_gaussian else 0,
                        "densify_clone": int(densify_stats.get("n_clone", 0)) if update_gaussian else 0,
                        "densify_split": int(densify_stats.get("n_split", 0)) if update_gaussian else 0,
                        "dia_anchor_reset": int(densify_stats.get("dia_anchor_reset", 0)) if update_gaussian else 0,
                        "dia_anchor_pruned": int(densify_stats.get("dia_anchor_pruned", 0)) if update_gaussian else 0,
                        "dia_anchor_depth_candidates": int(
                            densify_stats.get("dia_anchor_depth_candidates", 0)
                        ) if update_gaussian else 0,
                        "dia_anchor_strong_candidates": int(
                            densify_stats.get("dia_anchor_strong_candidates", 0)
                        ) if update_gaussian else 0,
                        "dia_anchor_hard_prune_candidates": int(
                            densify_stats.get("dia_anchor_hard_prune_candidates", 0)
                        ) if update_gaussian else 0,
                        "dia_anchor_sky_floater_candidates": int(
                            densify_stats.get("dia_anchor_sky_floater_candidates", 0)
                        ) if update_gaussian else 0,
                        "protected_opacity_prune_blocked": int(
                            densify_stats.get("protected_opacity_prune_blocked", 0)
                        ) if update_gaussian else 0,
                        "protected_dia_reset_blocked": int(
                            densify_stats.get("protected_dia_reset_blocked", 0)
                        ) if update_gaussian else 0,
                        "protected_dia_prune_blocked": int(
                            densify_stats.get("protected_dia_prune_blocked", 0)
                        ) if update_gaussian else 0,
                        "n_protected_static": int(local_anchor_stats.get("n_protected_static", 0)),
                        "n_local_active": int(local_anchor_stats.get("n_local_active", 0)),
                        "n_frozen_grad_zeroed": int(local_anchor_stats.get("n_frozen_grad_zeroed", 0)),
                        "n_local_thaw": int(densify_stats.get("n_local_thaw", 0)) if update_gaussian else 0,
                        "n_local_new": int(densify_stats.get("n_local_new", 0)) if update_gaussian else 0,
                        "dia_anchor_inconsistent_candidates": int(
                            densify_stats.get("dia_anchor_inconsistent_candidates", 0)
                        ) if update_gaussian else 0,
                        "dia_anchor_replacement_candidates": int(
                            densify_stats.get("dia_anchor_replacement_candidates", 0)
                        ) if update_gaussian else 0,
                        "fastgs_prune_candidates": int(densify_stats.get("fastgs_prune_candidates", 0)) if update_gaussian else 0,
                        "fastgs_pruned": int(densify_stats.get("fastgs_pruned", 0)) if update_gaussian else 0,
                        "map_iter_ms": (time.perf_counter() - map_iter_start) * 1000.0,
                    },
                )
                self._append_anchor_debug(
                    frame_idx=current_window[0] if current_window else None,
                    render_loss=float(loss_mapping.detach().item())
                    if torch.is_tensor(loss_mapping)
                    else None,
                )
        return gaussian_split
                
    # Run color refinement as a post-processing step after SLAM
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = int(
            self.config["Training"].get("color_refinement_itr_num", 10000)
        )
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())      
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]

            if isinstance(viewpoint_cam, PanoramaCamera):
                _theta0 = torch.zeros(1, 3, device=self.device)
                _rho0   = torch.zeros(1, 3, device=self.device)
                pano_pkg = render_panorama_for_config(
                    viewpoint_cam, self.gaussians, self.pipeline_params, self.background,
                    config=self.config,
                    theta=_theta0, rho=_rho0,
                )
                image          = pano_pkg["render"]
                rendered_depth = pano_pkg["depth"]
                gt_image = viewpoint_cam.original_image.cuda()

                cr_lambda = float(self.config["Training"].get("color_refinement_lambda_dssim", 0.05))
                depth_w   = float(self.config["Training"].get("erp_mapping_depth_weight", 0.06))
                chb_eps   = float(self.config["Training"].get("erp_mapping_charbonnier_eps", 1e-3))

                Ll1  = l1_loss(image, gt_image)
                loss = (1.0 - cr_lambda) * Ll1 + cr_lambda * (1.0 - ssim(image, gt_image))
                loss += self._gaussian_ratio_regularization(init_phase=False)
                supervision = _get_panorama_supervision(
                    viewpoint_cam, self.config,
                    device=image.device, dtype=image.dtype,
                    depth_shape=rendered_depth.shape,
                )
                depth_valid = supervision["depth_valid"] & supervision["nonsky_mask"]
                if supervision["mono_depth"] is not None and depth_valid.any():
                    depth_loss, _, _ = robust_relative_depth_loss(
                        rendered_depth,
                        supervision["mono_depth"],
                        depth_valid,
                        torch.ones_like(rendered_depth),
                        self.config,
                        chb_eps,
                        loss_type=self.config["Training"].get("erp_mapping_depth_loss_type", "berhu"),
                    )
                    loss += depth_w * depth_loss

                loss.backward()
                with torch.no_grad():
                    vf = pano_pkg["visibility_filter"]
                    r  = pano_pkg["radii"]
                    if not self._skip_anchor_explicit_densify_stats():
                        self.gaussians.max_radii2D[vf] = torch.max(
                            self.gaussians.max_radii2D[vf], r[vf]
                        )
                        self.gaussians.add_densification_stats(
                            pano_pkg["viewspace_points"], vf
                        )
            else:
                render_pkg = render(
                    viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                )
                image, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - self.opt_params.lambda_dssim) * (
                    Ll1
                ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss += self._gaussian_ratio_regularization(init_phase=False)
                loss.backward()
                with torch.no_grad():
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self._clamp_gaussian_ratios()
                max_abs_s = float(self.config["Training"].get("max_abs_scaling", 0.0))
                if max_abs_s > 0:
                    self.gaussians.clamp_max_scaling(max_abs_s)
                if iteration <= 8000 and iteration % 500 == 0:
                    n_before = self.gaussians.get_xyz.shape[0]
                    self.gaussians.densify_and_prune(
                        float(self.config["Training"].get("mapping_densify_grad_threshold", 0.008)),
                        self.gaussian_th, self.gaussian_extent, self.size_threshold,
                    )
                    if self.use_anchor_scaffold and self.gaussians.get_xyz.shape[0] != n_before:
                        self._rebuild_anchor_hash_tables(
                            resolve_collisions=False,
                            snap_xyz_to_voxel=False,
                        )
                        self._rebuild_submap_gaussian_indices()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration % 2000 == 0:
                    n_p = self.gaussians.prune_needles(
                        max_ratio=float(self.config["Training"].get("needle_prune_ratio", 50.0)),
                        protect_sky=False,
                    )
                    if n_p > 0:
                        if self.use_anchor_scaffold:
                            self._rebuild_anchor_hash_tables(
                                resolve_collisions=False,
                                snap_xyz_to_voxel=False,
                            )
                            self._rebuild_submap_gaussian_indices()
                        Log(f"Color refinement needle cleanup: pruned {n_p}")
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration_total)
                if self.use_anchor_scaffold and (
                    iteration == 1 or iteration % 200 == 0 or iteration == iteration_total
                ):
                    self._append_anchor_debug(
                        frame_idx=viewpoint_cam_idx,
                        render_loss=float(loss.detach().item()) if torch.is_tensor(loss) else None,
                    )
        Log("Map refinement done")

    def _periodic_global_refinement(self, iters: int) -> None:
        """Run a mid-mapping global BA burst over all keyframes."""
        if iters <= 0 or self.gaussians is None:
            return
        if len(self.viewpoints) == 0:
            return
        cfg = self.config["Training"]
        snapshot_keys = (
            "enable_protected_anchor_freeze",
            "enable_local_anchor_freeze",
            "enable_structure_xyz_freeze",
        )
        snapshot = {k: cfg.get(k, None) for k in snapshot_keys}
        cfg["enable_protected_anchor_freeze"] = False
        cfg["enable_local_anchor_freeze"] = False
        # Keep enable_structure_xyz_freeze whatever it was; unfreezing xyz too
        # eagerly here would re-introduce the side-effects the unfreeze
        # experiment originally guarded against.

        cr_lambda = float(cfg.get("periodic_global_ba_lambda_dssim", 0.10))
        depth_w = float(cfg.get("erp_mapping_depth_weight", 0.0))
        chb_eps = float(cfg.get("erp_mapping_charbonnier_eps", 1e-3))
        vp_ids = list(self.viewpoints.keys())
        t_start = time.perf_counter()
        loss_start = None
        loss_end = None
        try:
            for i in range(int(iters)):
                vp_id = random.choice(vp_ids)
                viewpoint_cam = self.viewpoints[vp_id]
                if isinstance(viewpoint_cam, PanoramaCamera):
                    _theta0 = torch.zeros(1, 3, device=self.device)
                    _rho0 = torch.zeros(1, 3, device=self.device)
                    pano_pkg = render_panorama_for_config(
                        viewpoint_cam,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        config=self.config,
                        theta=_theta0,
                        rho=_rho0,
                    )
                    image = pano_pkg["render"]
                    rendered_depth = pano_pkg["depth"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - cr_lambda) * Ll1 + cr_lambda * (
                        1.0 - ssim(image, gt_image)
                    )
                    loss += self._gaussian_ratio_regularization(init_phase=False)
                    supervision = _get_panorama_supervision(
                        viewpoint_cam,
                        self.config,
                        device=image.device,
                        dtype=image.dtype,
                        depth_shape=rendered_depth.shape,
                    )
                    depth_valid = (
                        supervision["depth_valid"] & supervision["nonsky_mask"]
                    )
                    if (
                        depth_w > 0
                        and supervision["mono_depth"] is not None
                        and depth_valid.any()
                    ):
                        depth_loss, _, _ = robust_relative_depth_loss(
                            rendered_depth,
                            supervision["mono_depth"],
                            depth_valid,
                            torch.ones_like(rendered_depth),
                            self.config,
                            chb_eps,
                            loss_type=self.config["Training"].get(
                                "erp_mapping_depth_loss_type", "berhu"
                            ),
                        )
                        loss += depth_w * depth_loss
                else:
                    render_pkg = render(
                        viewpoint_cam,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                    )
                    image = render_pkg["render"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + (
                        self.opt_params.lambda_dssim
                        * (1.0 - ssim(image, gt_image))
                    )
                    loss += self._gaussian_ratio_regularization(init_phase=False)
                if i == 0:
                    loss_start = float(loss.detach().item())
                loss.backward()
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self._clamp_gaussian_ratios()
                    max_abs_s = float(cfg.get("max_abs_scaling", 0.0))
                    if max_abs_s > 0:
                        self.gaussians.clamp_max_scaling(max_abs_s)
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                if i == int(iters) - 1:
                    loss_end = float(loss.detach().item())
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self._append_jsonl(
                "backend_perf.jsonl",
                {
                    "event": "global_ba_burst",
                    "n_viewpoints": len(self.viewpoints),
                    "iters": int(iters),
                    "loss_start": loss_start,
                    "loss_end": loss_end,
                    "elapsed_ms": elapsed_ms,
                },
            )
            Log(
                f"[BackEnd] global BA burst: kf_count={len(self.viewpoints)} "
                f"iters={iters} loss {loss_start:.4f} -> {loss_end:.4f} "
                f"in {elapsed_ms:.1f} ms"
            )
        finally:
            for k, v in snapshot.items():
                if v is None:
                    cfg.pop(k, None)
                else:
                    cfg[k] = v

    def _save_kf_render_backend(self, frame_idx: int, viewpoint):
        """Save post-optimisation keyframe render (GT | Render) with PSNR."""
        if self.save_dir is None or viewpoint is None:
            return
        try:
            import cv2
            import os
            from backend.legacy_360gs.gaussian_splatting.utils.image_utils import psnr as compute_psnr
            from backend.legacy_360gs.utils.kf_render_io import save_kf_canvas

            kf_dir = os.path.join(self.save_dir, "kf_renders_opt")
            os.makedirs(kf_dir, exist_ok=True)

            with torch.no_grad():
                if isinstance(viewpoint, PanoramaCamera):
                    _theta0 = torch.zeros(1, 3, device=self.device)
                    _rho0   = torch.zeros(1, 3, device=self.device)
                    pkg = render_panorama_for_config(
                        viewpoint, self.gaussians, self.pipeline_params, self.background,
                        config=self.config,
                        theta=_theta0, rho=_rho0,
                    )
                    erp_render = pkg["render"]
                else:
                    from backend.legacy_360gs.gaussian_splatting.gaussian_renderer import render as _render
                    pkg = _render(viewpoint, self.gaussians,
                                  self.pipeline_params, self.background)
                    erp_render = pkg["render"]

            erp_render = erp_render.clamp(0, 1)
            gt = viewpoint.original_image.cuda().clamp(0, 1)
            gs_only = pkg.get("gs_only", erp_render).clamp(0, 1)
            sky_bg_only = pkg.get("sky_bg_only", None)
            if sky_bg_only is not None:
                sky_bg_only = sky_bg_only.clamp(0, 1)

            psnr_val = compute_psnr(erp_render.unsqueeze(0),
                                    gt.unsqueeze(0)).item()

            def _to_bgr(t):
                return (t.detach().permute(1, 2, 0).cpu().numpy() * 255
                        ).clip(0, 255).astype(np.uint8)[:, :, ::-1]

            gt_bgr     = _to_bgr(gt)
            gs_bgr = _to_bgr(gs_only)
            bg_bgr = _to_bgr(sky_bg_only) if sky_bg_only is not None else np.zeros_like(gt_bgr)
            render_bgr = _to_bgr(erp_render)
            H, W = gt_bgr.shape[:2]
            gap = 4
            canvas = np.zeros((H + 28, W * 4 + gap * 3, 3), dtype=np.uint8)
            canvas[28:, :W] = gt_bgr
            canvas[28:, W + gap:2 * W + gap] = gs_bgr
            canvas[28:, 2 * W + 2 * gap:3 * W + 2 * gap] = bg_bgr
            canvas[28:, 3 * W + 3 * gap:] = render_bgr
            label = (f"KF {frame_idx:04d} [post-opt]  "
                     f"PSNR: {psnr_val:.2f} dB")
            growth_stats = getattr(viewpoint, "anchor_growth_stats", None)
            if isinstance(growth_stats, dict):
                label += (
                    f" | new={int(growth_stats.get('n_structure_hash_new', 0)) + int(growth_stats.get('n_sky_hash_new', 0))}"
                    f" merged={int(growth_stats.get('n_structure_hash_merged', 0))}"
                )
            cv2.putText(canvas, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if canvas.shape[1] > 1920:
                sc = 1920 / canvas.shape[1]
                canvas = cv2.resize(
                    canvas,
                    (int(canvas.shape[1] * sc), int(canvas.shape[0] * sc)),
                    interpolation=cv2.INTER_AREA,
                )
            _results = self.config.get("Results", {})
            _out = save_kf_canvas(
                canvas,
                os.path.join(kf_dir, f"kf_{frame_idx:04d}"),
                _results,
            )
            depth_tensor = pkg.get("depth", None)
            if depth_tensor is not None:
                depth_dir = os.path.join(self.save_dir, "kf_depths_opt")
                os.makedirs(depth_dir, exist_ok=True)
                depth_np = depth_tensor.detach().float().squeeze().cpu().numpy()
                valid = np.isfinite(depth_np) & (depth_np > 0)
                depth_canvas = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
                if valid.any():
                    lo, hi = np.percentile(depth_np[valid], [2.0, 98.0])
                    if hi <= lo:
                        hi = lo + 1.0
                    norm = ((depth_np - lo) / (hi - lo)).clip(0.0, 1.0)
                    gray = (norm * 255.0).astype(np.uint8)
                    depth_canvas = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                    depth_canvas[~valid] = 0
                cv2.imwrite(os.path.join(depth_dir, f"kf_{frame_idx:04d}.png"), depth_canvas)
            self._append_sky_debug(
                phase="kf",
                frame_idx=frame_idx,
                viewpoint=viewpoint,
                render_img=erp_render,
                opacity=pkg["opacity"],
                sky_bg_only=pkg.get("sky_bg_only"),
                sky_bg_alpha=pkg.get("sky_bg_alpha"),
            )
            self._append_sky_opacity_stats(frame_idx=frame_idx)
            Log(f"[BackEnd] KF {frame_idx} post-opt PSNR: {psnr_val:.2f} dB 鈫?{_out}")
        except Exception as e:
            Log(f"[BackEnd] _save_kf_render_backend failed: {e}")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        self.current_window = self._filter_registered_window(
            self.current_window, log_context="push_to_frontend"
        )
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append(
                (
                    kf_idx,
                    clone_obj_to_device(kf.R, device="cpu"),
                    clone_obj_to_device(kf.T, device="cpu"),
                )
            )
        if tag is None:
            tag = "sync_backend"

        msg = [
            tag,
            clone_obj_to_device(
                self.gaussians,
                device="cpu",
                skip_attrs={"optimizer"},
                cpu_only_attrs=GAUSSIAN_CPU_ONLY_ATTRS,
            ),
            clone_obj_to_device(self.occ_aware_visibility, device="cpu"),
            keyframes,
        ]
        self.frontend_queue.put(pack_queue_message(msg))
    # Main execution loop: 
    # process backend messages, perform initialization, optimize keyframe map, color refinement,
    # synchronize data, and push updates to the frontend
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:       
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = unpack_queue_message(self.backend_queue.get())
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend("color_refinement")
                elif data[0] == "loop_closure":
                    new_poses = {
                        int(frame_id): np.asarray(T_c2w, dtype=np.float64)
                        for frame_id, T_c2w in data[1].items()
                    }
                    affected_submaps = [int(v) for v in data[2]]
                    self._apply_loop_closure_update(new_poses, affected_submaps)
                    self.push_to_frontend()

                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    move_obj_to_device_(viewpoint, device=self.device)
                    depth_map = move_obj_to_device_(depth_map, device=self.device)
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    if self.enable_submap:
                        self._submap_manager.assign_frame(cur_frame_idx, pose_c2w=T_np)
                    self.current_window = normalize_window_order([cur_frame_idx])
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "register":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    move_obj_to_device_(viewpoint, device=self.device)
                    depth_map = move_obj_to_device_(depth_map, device=self.device)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    self.viewpoints[cur_frame_idx] = viewpoint
                    if self.enable_submap:
                        self._submap_manager.assign_frame(cur_frame_idx, pose_c2w=T_np)
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = normalize_window_order(data[3])
                    depth_map = data[4]
                    move_obj_to_device_(viewpoint, device=self.device)
                    depth_map = move_obj_to_device_(depth_map, device=self.device)
                    self.theta = move_obj_to_device_(data[5], device=self.device)
                    print("current keyframe ",cur_frame_idx,'window is ',current_window)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    if self.enable_submap:
                        self._submap_manager.assign_frame(cur_frame_idx, pose_c2w=T_np)
                    self.current_window = self._filter_registered_window(
                        current_window, log_context=f"keyframe {cur_frame_idx}"
                    )
                    if cur_frame_idx not in self.current_window:
                        self.current_window = normalize_window_order(
                            [cur_frame_idx] + self.current_window
                        )
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    # Phase 4: dynamic budget scheduling
                    _motion_norm = getattr(viewpoint, "motion_norm_m", None)
                    if _motion_norm is None:
                        _motion_norm = float(
                            self.config["Training"].get("accel_motion_ref", 0.3)
                        )
                    _motion_norm = float(_motion_norm)
                    _motion_rot = float(getattr(viewpoint, "motion_rot_deg", 0.0))
                    _budget = schedule_budget(self._current_overlap, _motion_norm, self.config)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_nosingle = self.config["Training"]["mapping_itr_nosingle"]
                    # Use dynamic map_iters from Phase-4 budget when enable_accel=True,
                    # otherwise fall back to static config value.
                    iter_per_kf = (
                        _budget.map_iters
                        if bool(self.config["Training"].get("enable_accel", False))
                        else (self.mapping_itr_num if self.single_thread else iter_nosingle)
                    )
                    # Enable backend pose joint-optimisation by default (now
                    # that the ERP projection gradient bug is fixed).  Users
                    # who rely exclusively on the frontend RANSAC poses can
                    # disable this via  freeze_pose: true  in the config.
                    freeze_pose = self.config["Training"].get("freeze_pose", False)
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx, kf_idx in enumerate(self.current_window):
                        if kf_idx == 0:
                            continue
                        viewpoint = self.viewpoints[kf_idx]
                        # Only include pose params when pose optimisation is
                        # enabled (freeze_pose=False).
                        if not freeze_pose and cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                    self.keyframe_optimizers = (
                        torch.optim.Adam(opt_params) if opt_params else None
                    )

                    if bool(self.config["Training"].get("enable_accel", False)):
                        Log(
                            f"[Budget] kf={cur_frame_idx} overlap={self._current_overlap:.4f} "
                            f"motion={_motion_norm:.5f}m rot={_motion_rot:.3f}deg "
                            f"map_iters={iter_per_kf} (budget_map={_budget.map_iters})"
                        )
                    else:
                        Log(
                            f"[Budget] kf={cur_frame_idx} overlap={self._current_overlap:.4f} "
                            f"motion={_motion_norm:.5f}m rot={_motion_rot:.3f}deg "
                            f"map_iters={iter_per_kf} (accel off)"
                        )

                    self.map(self.current_window, iters=iter_per_kf,
                             up_pose=(not freeze_pose))
                    self.map(self.current_window, prune=True)

                    # Near-layer periodic prune (Phase 3 layered map).
                    _prune_interval = int(
                        self.config["Training"].get("near_layer_prune_interval", 10)
                    )
                    _n_vp = len(self.viewpoints)
                    if (_n_vp % _prune_interval == 0):
                        _n_pr = self._near_layer_prune()
                        Log(
                            f"[NearLayerPrune] kf={cur_frame_idx} n_viewpoints={_n_vp} "
                            f"interval={_prune_interval} pruned_near={_n_pr}"
                        )

                    # Fix3 鈥?periodic mid-mapping global BA burst.
                    if bool(self.config["Training"].get("enable_periodic_global_ba", False)):
                        _ba_interval = int(
                            self.config["Training"].get("periodic_global_ba_interval", 30)
                        )
                        _ba_warmup = int(
                            self.config["Training"].get("periodic_global_ba_warmup_kfs", 30)
                        )
                        if (
                            _ba_interval > 0
                            and _n_vp >= _ba_warmup
                            and (_n_vp % _ba_interval) == 0
                        ):
                            _ba_iters = int(
                                self.config["Training"].get("periodic_global_ba_iters", 800)
                            )
                            self._periodic_global_refinement(_ba_iters)

                    # Save post-optimisation keyframe render (shows actual map quality)
                    self._save_kf_render_backend(cur_frame_idx,
                                                 self.viewpoints[cur_frame_idx])
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            unpack_queue_message(self.backend_queue.get())
        while not self.frontend_queue.empty():
            unpack_queue_message(self.frontend_queue.get())
        return
