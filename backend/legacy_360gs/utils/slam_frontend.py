import time

import numpy as np
import torch
import torch.multiprocessing as mp
import os

from backend.legacy_360gs.gaussian_splatting.gaussian_renderer import render
from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from backend.legacy_360gs.utils.camera_utils import Camera, PanoramaCamera
from backend.legacy_360gs.utils.eval_utils import (
    eval_ate,
    eval_pose_dict,
    save_gaussians,
    save_pose_dict_artifact,
)
from backend.legacy_360gs.utils.logging_utils import Log
from backend.legacy_360gs.utils.multiprocessing_utils import (
    GAUSSIAN_CPU_ONLY_ATTRS,
    clone_obj,
    clone_obj_to_device,
    move_obj_to_device_,
    pack_queue_message,
    unpack_queue_message,
)
from backend.legacy_360gs.utils.pano_masking import build_erp_ignore_mask
from backend.legacy_360gs.utils.pano_consistency import (
    depth_render_consistency_mask,
    depth_render_novelty_components,
    latitude_weights_for_uv,
    sample_mask_at_uv,
)
from backend.legacy_360gs.utils.pose_utils import update_pose
from backend.legacy_360gs.utils.slam_utils import (
    align_mono_depth_to_render_np,
    get_loss_tracking,
    get_median_depth,
)
from backend.legacy_360gs.utils.depth_utils import process_depth
from backend.legacy_360gs.utils.submap_manager import SubmapManager, normalize_window_order
from backend.legacy_360gs.utils.frontend_360dvo import Frontend360DVO, Online360DVOFrontend


def _lazy_get_pose_depth_imports():
    from backend.legacy_360gs.utils.init_pose import get_depth, get_pose

    return get_pose, get_depth

class FrontEnd(mp.Process):
    def __init__(self, config, model, save_dir=None, dap_model=None,
                 sphereglue_matcher=None):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.save_dir = save_dir

        self.initialized = False            
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.pause = False

        self.model = model  # MASt3R Model
        self.theta = 0

        # DAP panoramic depth model (optional; when set, replaces MASt3R depth
        # in panoramic mode while leaving all other paths untouched)
        self.dap_model = dap_model

        # SphereGlue matcher (optional; when set together with dap_model,
        # enables spherical 3D-2D RANSAC pose initialisation in panoramic mode)
        self.sphereglue_matcher = sphereglue_matcher
        self.frontend_mode = str(
            config["Training"].get("frontend_mode", "spherical")
        ).lower()
        if self.frontend_mode == "original":
            self.frontend_mode = "spherical"
        if self.frontend_mode not in {"spherical", "360dvo", "hybrid"}:
            raise ValueError(
                "Training.frontend_mode must be one of "
                "'spherical', '360dvo', or 'hybrid'."
            )
        frontend_360dvo_cfg = config["Training"].get("frontend_360dvo", {}) or {}
        frontend_360dvo_mode = str(
            frontend_360dvo_cfg.get("mode", "offline_tum")
        ).lower()
        if frontend_360dvo_mode == "online":
            self.frontend_360dvo = Online360DVOFrontend(config, save_dir=save_dir)
        else:
            self.frontend_360dvo = Frontend360DVO(config, save_dir=save_dir)
        if self.frontend_mode in {"360dvo", "hybrid"}:
            if frontend_360dvo_mode == "online":
                Log(
                    "360DVO online frontend enabled: "
                    f"network={self.frontend_360dvo.network_path} "
                    f"pose_scale_policy={self.frontend_360dvo.pose_scale_policy} "
                    f"depth_scale_policy={self.frontend_360dvo.depth_scale_policy}",
                    tag="FrontEnd",
                )
            elif self.frontend_360dvo.available:
                Log(
                    "360DVO frontend prior loaded: "
                    f"{self.frontend_360dvo.trajectory_path}",
                    tag="FrontEnd",
                )
            else:
                Log(
                    "360DVO frontend mode enabled but no offline TUM prior is "
                    "available yet; spherical fallback may be used if enabled.",
                    tag="FrontEnd",
                )

        # Panoramic mode flags (populated from config in set_hyperparams)
        self.panoramic_mode = config["Dataset"].get("type", "") == "panorama"
        self.erp_as_perspective = bool(
            config["Training"].get("erp_as_perspective", False)
        )
        self.face_w = config["Training"].get("face_w", 256)
        self.face_weights = config["Training"].get(
            "face_weights", [1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        )

        # Per-frame pre-refinement poses (frame_idx 鈫?4脳4 w2c numpy float32)
        # Populated in tracking_panoramic(); saved at end of run()
        self._pre_refinement_poses: dict = {}
        self._post_frontend_refine_poses: dict = {}
        self._post_backend_local_ba_poses: dict = {}

        # Phase 5: loop closure detector (no-op when enable_loop_closure=False)
        from backend.legacy_360gs.utils.loop_closure import PanoLoopDetector
        self._loop_detector = PanoLoopDetector(config)

        # Previous consecutive frame cache for spherical RANSAC.
        # Unlike `last_kf` (which can be many frames old), these always refer
        # to the frame immediately before the current one, mirroring what
        # test_panoramic_odometry.py does.  Updated at the end of every
        # tracking_panoramic() call and initialised in initialize().
        self._prev_frame_img: torch.Tensor = None   # (C, H, W) float [0,1]
        self._prev_frame_depth: np.ndarray = None   # (H, W) float32
        self._prev_frame_depth_valid_mask = None
        self._prev_frame_depth_source = "mono_depth"
        self._prev_mono_depth_aligned: np.ndarray = None
        self._prev_mono_depth_valid_mask = None
        self._prev_mono_consistency_mask = None
        self._prev_mono_w2c: np.ndarray = None
        self._prev_kf_mono_depth_aligned: np.ndarray = None
        self._prev_kf_mono_depth_valid_mask = None
        self._prev_kf_mono_w2c: np.ndarray = None
        self._prev_kf_mono_frame_idx: int | None = None
        self._prev_frame_w2c: np.ndarray = None     # (4, 4) float32
        self._prev_frame_valid_mask = None
        self._prev_frame_consistency_mask = None
        self._prev_kf_novelty_mask = None
        self._last_rel_w2c: np.ndarray | None = None
        self._dap_depth_global_scale = 1.0
        self._last_dap_align_stats: dict = {}
        self.enable_submap = bool(config["Training"].get("enable_submap", True))
        self._submaps: dict = {}
        self._active_submap_id: int = -1
        self._submap_manager = SubmapManager(
            interval=int(config["Training"].get("submap_kf_interval", 10)),
            overlap_kfs=int(config["Training"].get("submap_overlap_kfs", 3)),
        )
        # Debug-output directories (created lazily in run())
        self._match_vis_dir: str = None   # per-frame SphereGlue match visualisations
        self._kf_render_dir: str = None   # per-keyframe GT-vs-render comparisons
        self._depth_vis_dir: str = None   # per-frame DAP depth visualisations
        self._depth_compare_vis_dir: str = None
        self._consistency_vis_dir: str = None
        self._ransac_sample_dir: str = None
        self._last_ransac_info: dict = {}
        self._warned_legacy_kf_depth_mode = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.pano_force_kf_interval = int(
            self.config["Training"].get("pano_force_kf_interval", 15)
        )

    def _save_depth_visualization(self, frame_idx: int, depth_map: np.ndarray):
        if self._depth_vis_dir is None or depth_map is None:
            return
        try:
            import cv2

            depth = np.nan_to_num(depth_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 100.0)
            depth_u8 = (depth / 100.0 * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            save_path = os.path.join(self._depth_vis_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, depth_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as exc:
            Log(f"[DepthVis] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    def _save_depth_compare_visualization(
        self,
        frame_idx: int,
        image_t,
        render_depth,
        aligned_dap_depth,
        opacity=None,
        valid_mask=None,
    ):
        if self._depth_compare_vis_dir is None or render_depth is None or aligned_dap_depth is None:
            return
        try:
            import cv2

            def _to_hw_np(value) -> np.ndarray:
                if isinstance(value, torch.Tensor):
                    arr = value.detach().squeeze().cpu().numpy()
                else:
                    arr = np.asarray(value).squeeze()
                return arr.astype(np.float32)

            def _as_valid(mask, shape):
                if mask is None:
                    return np.ones(shape, dtype=bool)
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.detach().squeeze().cpu().numpy()
                else:
                    mask_np = np.asarray(mask).squeeze()
                if mask_np.shape != shape:
                    return np.ones(shape, dtype=bool)
                return mask_np.astype(bool)

            def _colorize(values, valid, vmin, vmax, colormap=cv2.COLORMAP_TURBO):
                values = np.nan_to_num(values.astype(np.float32), nan=vmin, posinf=vmax, neginf=vmin)
                norm = ((np.clip(values, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6) * 255.0).astype(np.uint8)
                color = cv2.applyColorMap(norm, colormap)
                color[~valid] = 0
                return color

            rgb = (image_t.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            render_np = _to_hw_np(render_depth)
            dap_np = _to_hw_np(aligned_dap_depth)
            if render_np.shape != dap_np.shape or render_np.ndim != 2:
                return

            compare_valid = _as_valid(valid_mask, render_np.shape)
            valid = (
                np.isfinite(render_np)
                & np.isfinite(dap_np)
                & (render_np > 0.01)
                & (dap_np > 0.01)
            )
            valid &= compare_valid
            if opacity is not None:
                op_np = _to_hw_np(opacity)
                opacity_min = float(self.config["Training"].get("sca_refine_opacity_min", 0.15))
                if op_np.shape == render_np.shape:
                    valid &= np.isfinite(op_np) & (op_np > opacity_min)

            abs_res = np.abs(render_np - dap_np)
            rel_res = abs_res / np.maximum(dap_np, 1e-3)
            depth_max = float(
                self.config["Training"].get(
                    "debug_depth_compare_max_depth",
                    self.config["Training"].get("dap_depth_max_valid", 100.0),
                )
            )
            rel_max = float(self.config["Training"].get("debug_depth_compare_relerr_max", 1.0))
            abs_max = float(self.config["Training"].get("debug_depth_compare_abserr_max", 25.0))

            render_vis = _colorize(render_np, np.isfinite(render_np) & (render_np > 0.01) & compare_valid, 0.0, depth_max)
            dap_vis = _colorize(dap_np, np.isfinite(dap_np) & (dap_np > 0.01) & compare_valid, 0.0, depth_max)
            rel_vis = _colorize(rel_res, valid, 0.0, rel_max, cv2.COLORMAP_INFERNO)
            abs_vis = _colorize(abs_res, valid, 0.0, abs_max, cv2.COLORMAP_INFERNO)

            panels = [
                ("rgb", rgb_bgr),
                ("render_depth", render_vis),
                ("dap_aligned", dap_vis),
                ("rel_residual", rel_vis),
                ("abs_residual", abs_vis),
            ]
            for label, panel in panels:
                cv2.putText(
                    panel,
                    label,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if valid.any():
                stats_label = (
                    f"valid={valid.mean():.3f} "
                    f"rel_mean={float(np.nanmean(rel_res[valid])):.3f} "
                    f"rel_med={float(np.nanmedian(rel_res[valid])):.3f}"
                )
            else:
                stats_label = "valid=0.000"
            cv2.putText(
                panels[0][1],
                stats_label,
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            canvas = np.concatenate([panel for _, panel in panels], axis=1)
            if canvas.shape[1] > 2400:
                scale = 2400 / canvas.shape[1]
                canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))
            save_path = os.path.join(self._depth_compare_vis_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as exc:
            Log(f"[DepthCompareVis] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    def _save_consistency_mask_visualization(
        self,
        frame_idx: int,
        image_t,
        mask,
        stats=None,
        valid_mask=None,
    ):
        if self._consistency_vis_dir is None or frame_idx is None or mask is None:
            return
        try:
            import cv2

            rgb = (image_t.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = np.asarray(mask, dtype=bool)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            valid_np = np.ones_like(mask_np, dtype=bool)
            if valid_mask is not None:
                valid_np = np.asarray(valid_mask, dtype=bool)
                if valid_np.ndim == 3:
                    valid_np = valid_np[0]
                if valid_np.shape != mask_np.shape:
                    valid_np = np.ones_like(mask_np, dtype=bool)
            consistent = mask_np & valid_np
            inconsistent = (~mask_np) & valid_np
            invalid = ~valid_np
            overlay = rgb.copy()
            overlay[consistent] = (
                0.35 * overlay[consistent] + 0.65 * np.array([0, 255, 0])
            ).astype(np.uint8)
            overlay[inconsistent] = (
                0.65 * overlay[inconsistent] + 0.35 * np.array([255, 0, 0])
            ).astype(np.uint8)
            overlay[invalid] = (
                0.65 * overlay[invalid] + 0.35 * np.array([128, 128, 128])
            ).astype(np.uint8)
            canvas = np.ascontiguousarray(np.concatenate([rgb, overlay], axis=1)[:, :, ::-1])
            if stats:
                label = f"consistency={float(stats.get('coverage', 0.0)):.3f}"
                if "consistent_valid_ratio" in stats:
                    label += f" valid_cons={float(stats.get('consistent_valid_ratio', 0.0)):.3f}"
                if "temporal_source_kf" in stats:
                    label += f" ref_kf={int(stats.get('temporal_source_kf', -1))}"
                if "dap_scale_frame" in stats:
                    label += f" scale={float(stats.get('dap_scale_frame', 1.0)):.3f}"
                if "dap_align_pixels" in stats:
                    label += f" align_px={int(stats.get('dap_align_pixels', 0))}"
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
                canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))
            save_path = os.path.join(self._consistency_vis_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as exc:
            Log(f"[ConsistencyVis] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    def _to_hw_bool_mask(self, mask_like, shape=None):
        if mask_like is None:
            return None
        if isinstance(mask_like, torch.Tensor):
            mask_np = mask_like.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_like)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if shape is not None and tuple(mask_np.shape) != tuple(shape):
            return None
        return mask_np.astype(bool)

    def _compose_non_sky_valid_mask(self, viewpoint, shape, *, base_mask=None):
        valid = np.ones(tuple(shape), dtype=bool)
        base_np = self._to_hw_bool_mask(base_mask, shape)
        if base_np is not None:
            valid &= base_np
        sky_np = self._to_hw_bool_mask(getattr(viewpoint, "erp_sky_mask", None), shape)
        if sky_np is not None:
            valid &= ~sky_np
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        region_valid = self._to_hw_bool_mask(region_masks.get("valid", None), shape)
        if region_valid is not None:
            valid &= region_valid
        return valid

    def _dap_depth_max_valid(self) -> float:
        return float(
            self.config["Training"].get(
                "dap_depth_max_valid",
                self.config["Training"].get("ransac", {}).get("depth_max", 80.0),
            )
        )

    def _erp_sky_depth_threshold(self) -> float:
        return float(
            self.config["Training"].get(
                "erp_sky_depth_threshold", self._dap_depth_max_valid()
            )
        )

    def _compute_dap_masks(self, depth_map: np.ndarray):
        depth_map = depth_map.astype(np.float32)
        valid = (depth_map > 0.01) & (depth_map < self._dap_depth_max_valid())
        sky = depth_map >= self._erp_sky_depth_threshold()
        return valid, sky

    def _get_viewpoint_sky_mask(self, viewpoint, depth_map: np.ndarray) -> np.ndarray:
        sky_mask = getattr(viewpoint, "erp_sky_mask", None)
        if sky_mask is not None:
            return np.asarray(sky_mask, dtype=bool).copy()
        _, sky_mask = self._compute_dap_masks(depth_map.astype(np.float32))
        return sky_mask.copy()

    def _fill_sky_mask_above_dynamic_horizon(
        self,
        viewpoint,
        sky_mask: np.ndarray,
        depth_map: np.ndarray | None = None,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fill DAP sky holes above the first bottom-up sky-dominant run."""
        tr_cfg = self.config.get("Training", {})
        if not bool(tr_cfg.get("force_sky_above_horizon", False)):
            return sky_mask
        sky_np = np.asarray(sky_mask, dtype=bool)
        if sky_np.ndim == 3:
            sky_np = sky_np[0]
        if sky_np.ndim != 2:
            return sky_mask
        H, W = sky_np.shape

        valid = np.ones((H, W), dtype=bool)
        if depth_map is not None:
            depth_np = np.asarray(depth_map, dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            if depth_np.shape == (H, W):
                valid &= np.isfinite(depth_np) & (depth_np > 0.01)
        if bool(tr_cfg.get("force_sky_use_valid_mask", False)) and valid_mask is not None:
            valid_np = np.asarray(valid_mask, dtype=bool)
            if valid_np.ndim == 3:
                valid_np = valid_np[0]
            if valid_np.shape == (H, W):
                valid &= valid_np

        ignore_mask = build_erp_ignore_mask(H, W, self.config)
        ignore_np = np.asarray(ignore_mask, dtype=bool)
        if ignore_np.ndim == 3:
            ignore_np = ignore_np[0]
        if ignore_np.shape == (H, W):
            valid &= ~ignore_np

        row_valid = valid.sum(axis=1).astype(np.float32)
        min_valid_frac = float(tr_cfg.get("force_sky_min_valid_row_frac", 0.20))
        min_valid = max(1.0, float(W) * min_valid_frac)
        row_sky = (sky_np & valid).sum(axis=1).astype(np.float32)
        row_ratio = np.zeros((H,), dtype=np.float32)
        enough_valid = row_valid >= min_valid
        row_ratio[enough_valid] = row_sky[enough_valid] / np.maximum(row_valid[enough_valid], 1.0)

        ratio_thresh = float(tr_cfg.get("force_sky_row_ratio_thresh", 0.95))
        min_run = max(1, int(tr_cfg.get("force_sky_min_run_rows", 3)))
        good = enough_valid & (row_ratio >= ratio_thresh)

        bottommost_sky_row = None
        run_len = 0
        run_bottom = -1
        for row in range(H - 1, -1, -1):
            if good[row]:
                if run_len == 0:
                    run_bottom = row
                run_len += 1
                if run_len >= min_run:
                    bottommost_sky_row = int(run_bottom)
                    break
            else:
                run_len = 0
                run_bottom = -1
        if bottommost_sky_row is None:
            valid_rows = np.flatnonzero(enough_valid)
            best_row = int(valid_rows[np.argmax(row_ratio[valid_rows])]) if valid_rows.size > 0 else -1
            viewpoint.erp_sky_horizon_stats = {
                "mode": "bottom_up_consecutive_rows_miss",
                "horizon_row": -1,
                "fill_to_row": -1,
                "row_ratio_thresh": float(ratio_thresh),
                "min_run_rows": int(min_run),
                "best_row": int(best_row),
                "best_row_sky_ratio": float(row_ratio[best_row]) if best_row >= 0 else 0.0,
            }
            return sky_np

        fill_to = max(0, min(H - 1, bottommost_sky_row))
        sky_np = sky_np.copy()
        sky_np[: fill_to + 1, :] = True
        viewpoint.erp_sky_horizon_stats = {
            "mode": "bottom_up_consecutive_rows",
            "horizon_row": int(bottommost_sky_row),
            "fill_to_row": int(fill_to),
            "row_ratio_thresh": float(ratio_thresh),
            "min_run_rows": int(min_run),
            "horizon_row_sky_ratio": float(row_ratio[bottommost_sky_row]),
        }
        return sky_np

    def _apply_dap_global_scale(self, depth_map: np.ndarray) -> np.ndarray:
        if not bool(self.config["Training"].get("dap_use_global_scale", False)):
            return depth_map
        return (depth_map.astype(np.float32) * float(self._dap_depth_global_scale)).astype(np.float32)

    def _update_dap_global_scale(self, correction: float) -> None:
        if not bool(self.config["Training"].get("dap_use_global_scale", False)):
            return
        if not np.isfinite(correction) or correction <= 0:
            return
        alpha = float(self.config["Training"].get("dap_global_scale_ema_alpha", 0.25))
        new_scale = float(self._dap_depth_global_scale) * float(correction)
        scale_min = float(self.config["Training"].get("dap_global_scale_min", 0.2))
        scale_max = float(self.config["Training"].get("dap_global_scale_max", 5.0))
        new_scale = float(np.clip(new_scale, scale_min, scale_max))
        self._dap_depth_global_scale = float(
            (1.0 - alpha) * float(self._dap_depth_global_scale) + alpha * new_scale
        )

    def _set_dap_depth_raw(self, viewpoint, depth_map: np.ndarray) -> np.ndarray:
        """Store DAP's raw prediction separately from per-frame aligned depth."""
        depth_raw = np.asarray(depth_map, dtype=np.float32).copy()
        viewpoint.mono_depth_raw = depth_raw.copy()
        viewpoint.mono_depth = self._apply_dap_global_scale(depth_raw)
        viewpoint.mono_depth_dvo = None
        return viewpoint.mono_depth

    def _dvo_depth_scale_policy(self) -> str:
        cfg = self.config["Training"].get("frontend_360dvo", {}) or {}
        policy = str(cfg.get("depth_scale_policy", "none")).lower()
        if policy in {
            "align_depth_to_dvo",
            "dvo",
            "dvo_scale",
            "scale_depth_to_dvo",
        }:
            return "align_depth_to_dvo"
        if policy in {"bootstrap_init_only", "init_bootstrap", "bootstrap"}:
            return "bootstrap_init_only"
        return "none"

    def _dvo_continuous_dap_scale_enabled(self) -> bool:
        cfg = self.config["Training"].get("frontend_360dvo", {}) or {}
        update = str(cfg.get("dap_dvo_scale_update", "bootstrap_only")).lower()
        return update in {"ema_after_bootstrap", "ema", "online", "continuous", "always"}

    def _dvo_canonical_depth_enabled(self) -> bool:
        return (
            self.frontend_mode == "360dvo"
            and getattr(self.frontend_360dvo, "mode", "offline_tum") == "online"
            and self._dvo_depth_scale_policy() in {"align_depth_to_dvo", "bootstrap_init_only"}
        )

    def _dvo_mono_valid_mask(self, viewpoint, shape: tuple[int, int]):
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        return self._compose_non_sky_valid_mask(
            viewpoint,
            shape,
            base_mask=region_masks.get("valid", None),
        )

    def _dvo_calibrated_depth(self, viewpoint, depth_raw: np.ndarray | None = None):
        depth_dvo = getattr(viewpoint, "mono_depth_dvo", None)
        if depth_dvo is not None:
            depth_dvo = np.asarray(depth_dvo, dtype=np.float32)
            if depth_dvo.ndim == 3:
                depth_dvo = depth_dvo[0]
            if depth_raw is None or depth_dvo.shape == depth_raw.shape:
                return depth_dvo
        if depth_raw is None:
            depth_raw = getattr(viewpoint, "mono_depth_raw", None)
        if depth_raw is None:
            return None
        depth_raw = np.asarray(depth_raw, dtype=np.float32)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        scale_info = getattr(viewpoint, "dvo_depth_scale_info", None)
        if not hasattr(viewpoint, "dvo_depth_scale") and not isinstance(scale_info, dict):
            return None
        scale = float(getattr(viewpoint, "dvo_depth_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        return (depth_raw * scale).astype(np.float32)

    def _apply_360dvo_depth_scale(
        self,
        viewpoint,
        prior_info: dict | None,
        *,
        frame_idx: int,
        reason: str,
    ) -> bool:
        """Scale DAP depth into the online 360DVO coordinate scale.

        The DVO pose stays in its own online monocular scale.  When DPVO sparse
        patch depth is available, we use it only to rescale the monocular DAP
        depth that seeds/updates the Gaussian map.
        """
        policy = self._dvo_depth_scale_policy()
        if policy not in {"align_depth_to_dvo", "bootstrap_init_only"}:
            return False
        continuous_dap_dvo = self._dvo_continuous_dap_scale_enabled()
        if (
            policy == "bootstrap_init_only"
            and not continuous_dap_dvo
            and not str(reason).startswith("initialize")
        ):
            return False
        if not prior_info:
            return False
        depth_scale = prior_info.get("depth_scale", None)
        if depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0:
            return False
        depth_source = str(prior_info.get("depth_scale_source", "configured"))
        cfg = self.config["Training"].get("frontend_360dvo", {}) or {}
        allow_bootstrap = bool(cfg.get("depth_scale_apply_bootstrap", False))
        if (
            depth_source in {"configured", "dpvo_no_pose", "dpvo_warmup_random_depth"}
            and not allow_bootstrap
        ):
            return False
        if bool(cfg.get("depth_scale_apply_requires_stable", False)) and not bool(
            prior_info.get("depth_scale_stable", False)
        ):
            return False

        depth_raw = getattr(viewpoint, "mono_depth_raw", None)
        if depth_raw is None:
            depth_raw = getattr(viewpoint, "mono_depth", None)
        if depth_raw is None:
            return False
        depth_raw = np.asarray(depth_raw, dtype=np.float32)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        base_depth = (
            depth_raw
            if policy == "bootstrap_init_only"
            else self._apply_dap_global_scale(depth_raw)
        )
        scaled_depth = (base_depth * float(depth_scale)).astype(np.float32)
        viewpoint.mono_depth_dvo = scaled_depth.copy()
        viewpoint.mono_depth = scaled_depth
        viewpoint.dvo_depth_scale = float(depth_scale)
        if policy == "bootstrap_init_only" and bool(
            self.config["Training"].get("dap_use_global_scale", False)
        ):
            self._dap_depth_global_scale = float(depth_scale)
        viewpoint.dvo_depth_scale_info = {
            "frame_idx": int(frame_idx),
            "reason": str(reason),
            "depth_scale": float(depth_scale),
            "dap_to_dvo_scale": float(depth_scale),
            "depth_scale_source": depth_source,
            "dap_to_dvo_scale_source": str(
                prior_info.get("dap_to_dvo_scale_source", depth_source)
            ),
            "depth_scale_stable": bool(prior_info.get("depth_scale_stable", False)),
            "dap_to_dvo_scale_stable": bool(
                prior_info.get(
                    "dap_to_dvo_scale_stable",
                    prior_info.get("depth_scale_stable", False),
                )
            ),
            "sparse_valid": int(prior_info.get("sparse_valid", 0)),
            "dap_to_dvo_ratio_mad_norm": float(
                prior_info.get("dap_to_dvo_ratio_mad_norm", float("nan"))
            ),
            "dap_to_dvo_ratio_count": int(
                prior_info.get("dap_to_dvo_ratio_count", 0)
            ),
        }
        Log(
            f"[360DVO depth] frame={frame_idx} reason={reason} "
            f"depth_scale={float(depth_scale):.5f} "
            f"source={depth_source} "
            f"stable={bool(prior_info.get('depth_scale_stable', False))} "
            f"sparse={int(prior_info.get('sparse_valid', 0))} "
            f"mad={float(prior_info.get('dap_to_dvo_ratio_mad_norm', 0.0)):.3f}",
            tag="FrontEnd",
        )
        return True

    def _current_dap_depth_for_alignment(
        self,
        viewpoint,
        depth_raw: np.ndarray,
    ) -> np.ndarray:
        if self._dvo_canonical_depth_enabled():
            depth_dvo = self._dvo_calibrated_depth(viewpoint, depth_raw)
            if depth_dvo is not None:
                return depth_dvo
        if self._dvo_depth_scale_policy() == "align_depth_to_dvo":
            depth_scaled = getattr(viewpoint, "mono_depth", None)
            if depth_scaled is not None:
                depth_scaled = np.asarray(depth_scaled, dtype=np.float32)
                if depth_scaled.ndim == 3:
                    depth_scaled = depth_scaled[0]
                if depth_scaled.shape == depth_raw.shape:
                    return depth_scaled
        return self._apply_dap_global_scale(depth_raw)

    def _advance_online_360dvo_for_viewpoint(
        self,
        frame_idx: int,
        viewpoint,
        *,
        previous_w2c: np.ndarray | None,
        reason: str,
    ):
        if not (
            self.frontend_mode == "360dvo"
            and getattr(self.frontend_360dvo, "mode", "offline_tum") == "online"
        ):
            return None
        mono_for_dvo = getattr(viewpoint, "mono_depth_raw", None)
        if mono_for_dvo is None:
            mono_for_dvo = getattr(viewpoint, "mono_depth", None)
        mono_valid_mask = None
        if mono_for_dvo is not None:
            mono_np = np.asarray(mono_for_dvo, dtype=np.float32)
            if mono_np.ndim == 3:
                mono_np = mono_np[0]
            mono_valid_mask = self._dvo_mono_valid_mask(viewpoint, mono_np.shape)
        prior = self.frontend_360dvo.get_prior(
            frame_idx,
            mono_depth=mono_for_dvo,
            mono_valid_mask=mono_valid_mask,
            erp_image=viewpoint.original_image,
            previous_w2c=previous_w2c,
        )
        if prior is None:
            return None
        sparse_prior = None
        if hasattr(self.frontend_360dvo, "get_sparse_prior"):
            sparse_prior = self.frontend_360dvo.get_sparse_prior(
                frame_idx, mono_for_dvo
            )
        if sparse_prior is not None:
            viewpoint.dvo_sparse_prior = sparse_prior
        applied_depth_scale = self._apply_360dvo_depth_scale(
            viewpoint,
            prior.info,
            frame_idx=frame_idx,
            reason=reason,
        )
        if (
            not applied_depth_scale
            and self._dvo_depth_scale_policy() == "bootstrap_init_only"
            and isinstance(prior.info, dict)
        ):
            depth_scale = float(
                prior.info.get("depth_scale", self._dap_depth_global_scale)
            )
            if np.isfinite(depth_scale) and depth_scale > 0:
                viewpoint.dvo_depth_scale = float(depth_scale)
                viewpoint.dvo_depth_scale_info = {
                    "frame_idx": int(frame_idx),
                    "reason": str(reason),
                    "depth_scale": float(depth_scale),
                    "dap_to_dvo_scale": float(depth_scale),
                    "depth_scale_source": str(
                        prior.info.get("depth_scale_source", "dvo_bootstrap")
                    ),
                    "dap_to_dvo_scale_source": str(
                        prior.info.get(
                            "dap_to_dvo_scale_source",
                            prior.info.get("depth_scale_source", "dvo_bootstrap"),
                        )
                    ),
                    "depth_scale_stable": bool(
                        prior.info.get("depth_scale_stable", True)
                    ),
                    "dap_to_dvo_scale_stable": bool(
                        prior.info.get(
                            "dap_to_dvo_scale_stable",
                            prior.info.get("depth_scale_stable", True),
                        )
                    ),
                    "sparse_valid": int(prior.info.get("sparse_valid", 0)),
                    "dap_to_dvo_ratio_mad_norm": float(
                        prior.info.get("dap_to_dvo_ratio_mad_norm", float("nan"))
                    ),
                    "dap_to_dvo_ratio_count": int(
                        prior.info.get("dap_to_dvo_ratio_count", 0)
                    ),
                }
        return prior

    def _online_360dvo_enabled(self) -> bool:
        return (
            self.frontend_mode == "360dvo"
            and getattr(self.frontend_360dvo, "mode", "offline_tum") == "online"
        )

    def _bootstrap_online_360dvo_depth_scale_for_init(
        self,
        cur_frame_idx: int,
        viewpoint,
        *,
        initial_w2c: np.ndarray,
    ):
        if not self._online_360dvo_enabled():
            return None
        cfg = self.config["Training"].get("frontend_360dvo", {}) or {}
        n_frames = max(1, int(cfg.get("init_depth_bootstrap_frames", 1)))
        max_frame = min(len(self.dataset), int(cur_frame_idx) + n_frames)
        previous_w2c = np.asarray(initial_w2c, dtype=np.float32).copy()
        last_prior = self._advance_online_360dvo_for_viewpoint(
            cur_frame_idx,
            viewpoint,
            previous_w2c=previous_w2c,
            reason="initialize",
        )
        if last_prior is not None:
            previous_w2c = last_prior.w2c.astype(np.float32).copy()
        if n_frames <= 1 or self.dap_model is None:
            return last_prior

        Log(
            f"[360DVO init] bootstrap depth scale using frames "
            f"{cur_frame_idx}..{max_frame - 1}",
            tag="FrontEnd",
        )
        for boot_idx in range(int(cur_frame_idx) + 1, max_frame):
            try:
                boot_view = PanoramaCamera.init_from_panorama_dataset(
                    self.dataset,
                    boot_idx,
                    self.face_w,
                    face_zfar=float(
                        self.config["Training"].get("panorama_face_zfar", 500.0)
                    ),
                )
                boot_view.compute_grad_mask(self.config)
                depth_raw, _ = self.dap_model.infer(boot_view.original_image)
                _, sky_mask_raw = self._compute_dap_masks(depth_raw.astype(np.float32))
                boot_view.erp_sky_mask = sky_mask_raw.copy()
                self._set_dap_depth_raw(boot_view, depth_raw)
                self._compute_erp_region_masks(boot_view)
                prior = self._advance_online_360dvo_for_viewpoint(
                    boot_idx,
                    boot_view,
                    previous_w2c=previous_w2c,
                    reason="initialize_bootstrap",
                )
                if prior is not None:
                    last_prior = prior
                    previous_w2c = prior.w2c.astype(np.float32).copy()
            except Exception as exc:
                Log(
                    f"[360DVO init] bootstrap frame={boot_idx} failed: {exc}",
                    tag="FrontEnd",
                )
                break

        if hasattr(self.frontend_360dvo, "finalize_depth_scale_bootstrap"):
            self.frontend_360dvo.finalize_depth_scale_bootstrap(
                source=f"dvo_bootstrap_{n_frames}_frames"
            )
        latest_info = getattr(self.frontend_360dvo, "_last_scale_info", None)
        if isinstance(latest_info, dict):
            self._apply_360dvo_depth_scale(
                viewpoint,
                latest_info,
                frame_idx=cur_frame_idx,
                reason=f"initialize_bootstrap_{n_frames}",
            )
            should_freeze = bool(cfg.get("depth_scale_freeze_after_bootstrap", True))
            should_freeze = should_freeze and not self._dvo_continuous_dap_scale_enabled()
            if should_freeze and hasattr(
                self.frontend_360dvo, "freeze_depth_scale"
            ):
                self.frontend_360dvo.freeze_depth_scale(
                    source=f"dvo_bootstrap_{n_frames}_frames"
                )
            Log(
                f"[360DVO init] frame={cur_frame_idx} depth bootstrap "
                f"scale={float(latest_info.get('depth_scale', 1.0)):.5f} "
                f"source={latest_info.get('depth_scale_source', '-')} "
                f"stable={bool(latest_info.get('depth_scale_stable', False))} "
                f"sparse={int(latest_info.get('sparse_valid', 0))}",
                tag="FrontEnd",
            )
        return last_prior

    def _mono_depth_valid_mask(self, viewpoint, mono_depth_np: np.ndarray, region_valid=None) -> np.ndarray:
        mono_valid = (
            np.isfinite(mono_depth_np)
            & (mono_depth_np > 0.01)
            & (mono_depth_np < self._dap_depth_max_valid())
        )
        sky_mask = getattr(viewpoint, "erp_sky_mask", None)
        if sky_mask is not None:
            sky_np = np.asarray(sky_mask, dtype=bool)
            if sky_np.ndim == 3:
                sky_np = sky_np[0]
            if sky_np.shape == mono_valid.shape:
                mono_valid &= ~sky_np
        if region_valid is not None:
            region_np = np.asarray(region_valid, dtype=bool)
            if region_np.ndim == 3:
                region_np = region_np[0]
            if region_np.shape == mono_valid.shape:
                mono_valid &= region_np
        return mono_valid

    def _cache_aligned_mono_depth(
        self,
        viewpoint,
        w2c: np.ndarray,
        *,
        frame_idx: int,
        cache_keyframe: bool = False,
        region_valid=None,
    ) -> None:
        mono_depth_np = getattr(viewpoint, "mono_depth", None)
        if mono_depth_np is None or w2c is None:
            return
        mono_depth_np = np.asarray(mono_depth_np, dtype=np.float32)
        if mono_depth_np.ndim == 3:
            mono_depth_np = mono_depth_np[0]
        mono_valid = self._mono_depth_valid_mask(viewpoint, mono_depth_np, region_valid=region_valid)
        self._prev_mono_depth_aligned = mono_depth_np.copy()
        self._prev_mono_depth_valid_mask = mono_valid.copy()
        self._prev_mono_w2c = np.asarray(w2c, dtype=np.float32).copy()
        if cache_keyframe:
            self._prev_kf_mono_depth_aligned = mono_depth_np.copy()
            self._prev_kf_mono_depth_valid_mask = mono_valid.copy()
            self._prev_kf_mono_w2c = np.asarray(w2c, dtype=np.float32).copy()
            self._prev_kf_mono_frame_idx = int(frame_idx)

    def _align_viewpoint_dap_depth_to_render(
        self,
        frame_idx: int,
        viewpoint,
        render_depth,
        opacity=None,
        *,
        valid_rgb=None,
        reason: str = "tracking",
    ):
        if (
            not self.panoramic_mode
            or not isinstance(viewpoint, PanoramaCamera)
            or self.dap_model is None
        ):
            return getattr(viewpoint, "mono_depth", None), 1.0, None

        depth_raw = getattr(viewpoint, "mono_depth_raw", None)
        if depth_raw is None:
            depth_raw = getattr(viewpoint, "mono_depth", None)
        if depth_raw is None:
            return None, 1.0, None
        depth_raw = np.asarray(depth_raw, dtype=np.float32)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        depth_align_base = self._current_dap_depth_for_alignment(viewpoint, depth_raw)
        dvo_canonical_depth = (
            self._dvo_canonical_depth_enabled()
            and self._dvo_calibrated_depth(viewpoint, depth_raw) is not None
        )

        valid_dap, _ = self._compute_dap_masks(depth_raw)
        sky_mask = self._get_viewpoint_sky_mask(viewpoint, depth_raw)
        sky_mask = self._fill_sky_mask_above_dynamic_horizon(
            viewpoint,
            sky_mask,
            depth_raw,
            valid_mask=valid_dap,
        )
        viewpoint.erp_sky_mask = sky_mask.copy()
        valid_align = valid_dap & (~sky_mask)

        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        valid_region = region_masks.get("valid", None)
        if valid_region is not None:
            valid_region = np.asarray(valid_region, dtype=bool)
            if valid_region.ndim == 3:
                valid_region = valid_region[0]
            if valid_region.shape == depth_raw.shape:
                valid_align &= valid_region

        if valid_rgb is not None:
            valid_rgb_np = valid_rgb.detach().cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else np.asarray(valid_rgb)
            if valid_rgb_np.ndim == 3:
                valid_rgb_np = valid_rgb_np[0]
            if valid_rgb_np.shape == depth_raw.shape:
                valid_align &= valid_rgb_np.astype(bool)

        tr_cfg = self.config.get("Training", {})
        min_depth = float(tr_cfg.get("dap_align_min_depth", 0.05))
        max_depth = float(tr_cfg.get("dap_align_max_depth", tr_cfg.get("dap_depth_max_valid", 99.9)))
        min_pixels = int(tr_cfg.get("dap_align_min_pixels", 512))

        def _as_2d_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            elif value.ndim != 2:
                value = np.squeeze(value)
            if value.shape != depth_raw.shape:
                return None
            return value

        render_depth_np = _as_2d_numpy(render_depth)
        opacity_np = _as_2d_numpy(opacity)
        if render_depth_np is not None:
            aligned_depth, scale_frame, aligned_valid, align_fit_stats = align_mono_depth_to_render_np(
                render_depth_np,
                depth_align_base,
                valid_align,
                self.config,
                opacity=opacity_np,
                return_stats=True,
            )
            align_model = "render_" + str(align_fit_stats.get("align_model", tr_cfg.get("dap_align_model", "scale")))
            fit_pixels = int(align_fit_stats.get("fit_pixels", int(valid_align.sum())))
            align_shift = float(align_fit_stats.get("align_shift", 0.0))
            fit_rmse = float(align_fit_stats.get("fit_rmse", 0.0))
            fit_mask = valid_align & np.isfinite(render_depth_np)
            fit_mask &= (render_depth_np > min_depth) & (render_depth_np < max_depth)
            if opacity_np is not None:
                opacity_min = float(tr_cfg.get("dap_align_min_opacity", 0.2))
                fit_mask &= opacity_np > opacity_min
            ratios_np = render_depth_np[fit_mask] / np.clip(depth_align_base[fit_mask], min_depth, None)
            ratios_np = ratios_np[np.isfinite(ratios_np)]
            if fit_pixels < min_pixels:
                scale_frame = 1.0
                aligned_depth = depth_align_base.copy()
                aligned_valid = valid_align.copy()
                align_model = "render_fallback"
                align_shift = 0.0
                fit_rmse = 0.0
            reliable = int(fit_pixels) >= min_pixels and np.isfinite(float(scale_frame))
        else:
            scale_frame = 1.0
            aligned_depth = depth_align_base.copy()
            aligned_valid = valid_align.copy()
            align_model = "render_missing"
            align_shift = 0.0
            fit_pixels = 0
            fit_rmse = 0.0
            reliable = False
            ratios_np = np.asarray([], dtype=np.float32)

        render_debug_scale_frame = float(scale_frame)
        render_debug_shift = float(align_shift)
        render_debug_model = str(align_model)
        render_debug_reliable = bool(reliable)
        if dvo_canonical_depth:
            aligned_depth = np.asarray(depth_align_base, dtype=np.float32).copy()
            aligned_valid = valid_align.copy()
            scale_frame = 1.0
            align_shift = 0.0
            align_model = "dvo_scale_only"
            dvo_info = getattr(viewpoint, "dvo_depth_scale_info", None) or {}
            reliable = bool(dvo_info.get("dap_to_dvo_scale_stable", dvo_info.get("depth_scale_stable", False)))

        raw_median = float(np.median(ratios_np)) if ratios_np.size else 1.0
        raw_mad = float(np.median(np.abs(ratios_np - raw_median))) if ratios_np.size else 0.0
        valid_final = np.asarray(aligned_valid, dtype=bool).copy()
        valid_final &= valid_align
        valid_final &= np.isfinite(aligned_depth)
        valid_final &= (aligned_depth > min_depth) & (aligned_depth < max_depth)
        aligned_depth = np.asarray(aligned_depth, dtype=np.float32).copy()
        if bool(tr_cfg.get("dap_align_zero_invalid_depth", True)):
            aligned_depth[~valid_final] = 0.0

        stats = {
            "frame_idx": int(frame_idx),
            "reason": str(reason),
            "scale_frame": float(scale_frame),
            "align_shift": align_shift,
            "align_model": align_model,
            "align_fit_pixels": fit_pixels,
            "align_fit_rmse": fit_rmse,
            "raw_scale_median": raw_median,
            "raw_scale_mad": raw_mad,
            "align_pixels": int(ratios_np.size),
            "align_valid_ratio": float(valid_final.mean()),
            "valid_ratio": float(valid_dap.mean()),
            "sky_ratio": float(sky_mask.mean()),
            "temporal_warp_coverage": 0.0,
            "temporal_projected": 0,
            "temporal_min_pixels": int(min_pixels),
            "temporal_source_kf": -1,
            "reliable": bool(reliable),
            "global_scale": float(self._dap_depth_global_scale),
            "dvo_canonical_depth": bool(dvo_canonical_depth),
            "render_debug_align_model": render_debug_model,
            "render_debug_scale_frame": render_debug_scale_frame,
            "render_debug_shift": render_debug_shift,
            "render_debug_reliable": render_debug_reliable,
        }
        dvo_info = getattr(viewpoint, "dvo_depth_scale_info", None)
        if isinstance(dvo_info, dict):
            stats.update(
                {
                    "dap_to_dvo_scale": float(dvo_info.get("dap_to_dvo_scale", dvo_info.get("depth_scale", 1.0))),
                    "dap_to_dvo_scale_source": str(
                        dvo_info.get("dap_to_dvo_scale_source", dvo_info.get("depth_scale_source", "-"))
                    ),
                    "dap_to_dvo_scale_stable": bool(
                        dvo_info.get("dap_to_dvo_scale_stable", dvo_info.get("depth_scale_stable", False))
                    ),
                    "dap_to_dvo_ratio_mad_norm": float(dvo_info.get("dap_to_dvo_ratio_mad_norm", float("nan"))),
                    "dap_to_dvo_ratio_count": int(dvo_info.get("dap_to_dvo_ratio_count", 0)),
                }
            )
        horizon_stats = getattr(viewpoint, "erp_sky_horizon_stats", None)
        if isinstance(horizon_stats, dict):
            stats.update(
                {
                    "sky_horizon_mode": str(horizon_stats.get("mode", "")),
                    "sky_horizon_row": int(horizon_stats.get("horizon_row", -1)),
                    "sky_horizon_fill_to_row": int(horizon_stats.get("fill_to_row", -1)),
                    "sky_horizon_row_ratio": float(horizon_stats.get("horizon_row_sky_ratio", 0.0)),
                    "sky_horizon_best_row": int(horizon_stats.get("best_row", -1)),
                    "sky_horizon_best_row_ratio": float(horizon_stats.get("best_row_sky_ratio", 0.0)),
                }
            )
        viewpoint.mono_depth = aligned_depth
        if dvo_canonical_depth:
            viewpoint.mono_depth_dvo = aligned_depth.copy()
        viewpoint.dap_depth_align_stats = stats
        viewpoint.dap_align_reliable = bool(reliable)
        self._last_dap_align_stats[int(frame_idx)] = stats
        if not dvo_canonical_depth:
            self._update_dap_global_scale(float(scale_frame))
        horizon_label = ""
        if isinstance(horizon_stats, dict):
            row_ratio_value = float(
                horizon_stats.get(
                    "horizon_row_sky_ratio",
                    horizon_stats.get("best_row_sky_ratio", 0.0),
                )
            )
            horizon_label = (
                f" horizon={int(horizon_stats.get('fill_to_row', -1))}"
                f"/{int(horizon_stats.get('horizon_row', -1))}"
                f" ratio={row_ratio_value:.3f}"
                f" mode={str(horizon_stats.get('mode', ''))}"
            )
        Log(
            f"[DAP render align] frame={frame_idx} reason={reason} "
            f"model={align_model} scale_frame={float(scale_frame):.3f} "
            f"shift={align_shift:.3f} raw_median={raw_median:.3f} "
            f"mad={raw_mad:.3f} fit_pixels={fit_pixels} "
            f"valid={valid_final.mean():.3f} sky={sky_mask.mean():.3f} "
            f"reliable={bool(reliable)} global_scale={self._dap_depth_global_scale:.3f}"
            f" dvo_canonical={bool(dvo_canonical_depth)}"
            f"{horizon_label}",
            tag="FrontEnd",
        )
        return aligned_depth, float(scale_frame), valid_final

    def _ransac_depth_source(self) -> str:
        return str(
            self.config["Training"].get("ransac_depth_source", "render_depth")
        ).lower()

    def _render_pose_depth_for_viewpoint(self, viewpoint, *, reason: str = "pose_depth"):
        if (
            not self.panoramic_mode
            or not isinstance(viewpoint, PanoramaCamera)
            or self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) <= 0
        ):
            return None, None, "missing_render_context"
        try:
            from backend.legacy_360gs.utils.panoramic_renderer import render_panorama_for_config

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
                return None, None, "missing_render_depth"
            depth_np = (
                render_pkg["depth"].detach().squeeze(0).cpu().numpy().astype(np.float32)
            )
            valid = np.isfinite(depth_np) & (depth_np > 0.01) & (depth_np < self._dap_depth_max_valid())
            opacity = render_pkg.get("opacity", None)
            if opacity is not None:
                opacity_np = opacity.detach().squeeze(0).cpu().numpy().astype(np.float32)
                opacity_min = float(
                    self.config["Training"].get("ransac_render_opacity_min", 0.08)
                )
                if opacity_np.shape == depth_np.shape:
                    valid &= opacity_np > opacity_min
            region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
            valid_region = region_masks.get("valid", None)
            if valid_region is not None:
                valid_region = np.asarray(valid_region, dtype=bool)
                if valid_region.ndim == 3:
                    valid_region = valid_region[0]
                if valid_region.shape == depth_np.shape:
                    valid &= valid_region
            consistency = getattr(viewpoint, "erp_consistency_mask", None)
            if (
                bool(self.config["Training"].get("ransac_render_require_consistency", True))
                and consistency is not None
            ):
                consistency = np.asarray(consistency, dtype=bool)
                if consistency.ndim == 3:
                    consistency = consistency[0]
                if consistency.shape == depth_np.shape:
                    valid &= consistency
            viewpoint.pose_depth = depth_np
            viewpoint.pose_depth_valid_mask = valid
            viewpoint.pose_depth_source = "render_depth"
            Log(
                f"[RANSAC depth] frame={getattr(viewpoint, 'uid', -1)} "
                f"source=render_depth reason={reason} valid={float(valid.mean()):.3f}",
                tag="FrontEnd",
            )
            return depth_np, valid, "render_depth"
        except Exception as exc:
            Log(f"[RANSAC depth] render depth failed ({reason}): {exc}", tag="FrontEnd")
            return None, None, "render_failed"

    def _select_pose_depth_for_viewpoint(self, viewpoint, *, reason: str = "pose_depth"):
        source = self._ransac_depth_source()
        policy = str(
            self.config["Training"].get("pose_depth_policy", "configured")
        ).lower()
        if (
            policy in {"mono_until_map_ready", "mono_first", "conservative_mono_first"}
            and source in {"render", "render_depth", "internal", "internal_depth"}
        ):
            min_render_kfs = int(
                self.config["Training"].get("pose_depth_render_min_kfs", 4)
            )
            if len(self.kf_indices) < min_render_kfs:
                source = "mono_depth"
                Log(
                    f"[RANSAC depth] frame={getattr(viewpoint, 'uid', -1)} "
                    f"source=mono_depth policy={policy} "
                    f"kfs={len(self.kf_indices)}<{min_render_kfs}",
                    tag="FrontEnd",
                )
        if source in {"render", "render_depth", "internal", "internal_depth"}:
            allow_mono_fallback = bool(
                self.config["Training"].get("ransac_render_fallback_mono", False)
            )
            depth_np, valid_np, depth_source = self._render_pose_depth_for_viewpoint(
                viewpoint, reason=reason
            )
            if depth_np is not None:
                min_render_valid = float(
                    self.config["Training"].get("ransac_render_min_valid_ratio_for_pose", 0.08)
                )
                valid_ratio = float(valid_np.mean()) if isinstance(valid_np, np.ndarray) else 0.0
                if (
                    valid_ratio >= min_render_valid
                    or not bool(self.config["Training"].get("ransac_render_fallback_low_valid", True))
                ):
                    return depth_np, valid_np, depth_source
                if not allow_mono_fallback:
                    Log(
                        f"[RANSAC depth] frame={getattr(viewpoint, 'uid', -1)} "
                        f"render valid={valid_ratio:.3f}<{min_render_valid:.3f}; "
                        "mono fallback disabled",
                        tag="FrontEnd",
                    )
                    return None, None, "render_low_valid"
                Log(
                    f"[RANSAC depth] frame={getattr(viewpoint, 'uid', -1)} "
                    f"render valid={valid_ratio:.3f}<{min_render_valid:.3f}; "
                    "fallback to mono_depth",
                    tag="FrontEnd",
                )
            if not allow_mono_fallback:
                return None, None, depth_source
        depth_np = getattr(viewpoint, "mono_depth", None)
        if depth_np is None:
            return None, None, "missing_mono_depth"
        depth_np = np.asarray(depth_np, dtype=np.float32)
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        valid_np = self._mono_depth_valid_mask(
            viewpoint,
            depth_np[0] if depth_np.ndim == 3 else depth_np,
            region_valid=region_masks.get("valid", None),
        )
        viewpoint.pose_depth = depth_np.copy()
        viewpoint.pose_depth_valid_mask = valid_np
        viewpoint.pose_depth_source = "mono_depth"
        return viewpoint.pose_depth, valid_np, "mono_depth"

    def _update_prev_pose_depth_cache(self, viewpoint, *, reason: str = "tracking"):
        depth_np, valid_np, depth_source = self._select_pose_depth_for_viewpoint(
            viewpoint, reason=reason
        )
        self._prev_frame_depth = depth_np.copy() if depth_np is not None else None
        self._prev_frame_depth_valid_mask = (
            valid_np.copy() if isinstance(valid_np, np.ndarray) else None
        )
        self._prev_frame_depth_source = depth_source

    def _pose_delta_metrics(self, pose_a: np.ndarray, pose_b: np.ndarray):
        trans_delta = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
        rot_rel = pose_a[:3, :3] @ pose_b[:3, :3].T
        trace = np.clip((np.trace(rot_rel) - 1.0) * 0.5, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(trace)))
        return trans_delta, rot_deg

    def _camera_w2c_numpy(self, camera) -> np.ndarray:
        return (
            getWorld2View2(camera.R, camera.T)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _save_stage_pose_results(self):
        if not self.save_dir:
            return

        stage_root = os.path.join(self.save_dir, "pose_stage_eval")
        os.makedirs(stage_root, exist_ok=True)

        artifact_paths = {}
        if self._pre_refinement_poses:
            np.save(
                os.path.join(self.save_dir, "pre_refinement_poses.npy"),
                self._pre_refinement_poses,
            )
            artifact_paths["spherical_pnp_initial"] = save_pose_dict_artifact(
                self._pre_refinement_poses, stage_root, "spherical_pnp_initial_poses"
            )
        if self._post_frontend_refine_poses:
            np.save(
                os.path.join(self.save_dir, "post_frontend_refine_poses.npy"),
                self._post_frontend_refine_poses,
            )
            artifact_paths["frontend_erp_refine"] = save_pose_dict_artifact(
                self._post_frontend_refine_poses, stage_root, "frontend_erp_refine_poses"
            )
        if self._post_backend_local_ba_poses:
            np.save(
                os.path.join(self.save_dir, "post_backend_local_ba_poses.npy"),
                self._post_backend_local_ba_poses,
            )
            artifact_paths["backend_local_ba"] = save_pose_dict_artifact(
                self._post_backend_local_ba_poses, stage_root, "backend_local_ba_poses"
            )

        metrics_summary = {"artifacts": artifact_paths, "stages": {}, "comparisons": {}}

        if self._pre_refinement_poses:
            metrics_summary["stages"]["spherical_pnp_initial_all"] = eval_pose_dict(
                self.cameras,
                self._pre_refinement_poses,
                os.path.join(stage_root, "spherical_pnp_initial_all"),
                label="spherical_pnp_initial_all",
                monocular=self.monocular,
            )

        if self._post_frontend_refine_poses:
            metrics_summary["stages"]["frontend_erp_refine_all"] = eval_pose_dict(
                self.cameras,
                self._post_frontend_refine_poses,
                os.path.join(stage_root, "frontend_erp_refine_all"),
                label="frontend_erp_refine_all",
                monocular=self.monocular,
            )

        keyframe_ids = sorted(int(kf_id) for kf_id in self.kf_indices)
        if self._pre_refinement_poses:
            metrics_summary["stages"]["spherical_pnp_initial_kf"] = eval_pose_dict(
                self.cameras,
                self._pre_refinement_poses,
                os.path.join(stage_root, "spherical_pnp_initial_kf"),
                label="spherical_pnp_initial_kf",
                frame_ids=keyframe_ids,
                monocular=self.monocular,
            )
        if self._post_frontend_refine_poses:
            metrics_summary["stages"]["frontend_erp_refine_kf"] = eval_pose_dict(
                self.cameras,
                self._post_frontend_refine_poses,
                os.path.join(stage_root, "frontend_erp_refine_kf"),
                label="frontend_erp_refine_kf",
                frame_ids=keyframe_ids,
                monocular=self.monocular,
            )
        if self._post_backend_local_ba_poses:
            metrics_summary["stages"]["backend_local_ba_kf"] = eval_pose_dict(
                self.cameras,
                self._post_backend_local_ba_poses,
                os.path.join(stage_root, "backend_local_ba_kf"),
                label="backend_local_ba_kf",
                frame_ids=keyframe_ids,
                monocular=self.monocular,
            )

        comparable = [
            ("spherical_pnp_initial_kf", "frontend_erp_refine_kf"),
            ("frontend_erp_refine_kf", "backend_local_ba_kf"),
            ("spherical_pnp_initial_kf", "backend_local_ba_kf"),
        ]
        for prev_stage, next_stage in comparable:
            prev_metrics = metrics_summary["stages"].get(prev_stage)
            next_metrics = metrics_summary["stages"].get(next_stage)
            if not prev_metrics or not next_metrics:
                continue
            if prev_metrics.get("count", 0) < 2 or next_metrics.get("count", 0) < 2:
                continue
            metrics_summary["comparisons"][f"{next_stage}_minus_{prev_stage}"] = {
                "ate_rmse_delta": (
                    None
                    if np.isnan(prev_metrics.get("ate_rmse", np.nan))
                    or np.isnan(next_metrics.get("ate_rmse", np.nan))
                    else float(next_metrics["ate_rmse"] - prev_metrics["ate_rmse"])
                ),
                "raw_translation_rmse_delta_m": (
                    None
                    if not prev_metrics.get("raw_translation_error_m")
                    or not next_metrics.get("raw_translation_error_m")
                    else float(
                        next_metrics["raw_translation_error_m"]["rmse"]
                        - prev_metrics["raw_translation_error_m"]["rmse"]
                    )
                ),
                "raw_rotation_rmse_delta_deg": (
                    None
                    if not prev_metrics.get("raw_rotation_error_deg")
                    or not next_metrics.get("raw_rotation_error_deg")
                    else float(
                        next_metrics["raw_rotation_error_deg"]["rmse"]
                        - prev_metrics["raw_rotation_error_deg"]["rmse"]
                    )
                ),
            }

        summary_path = os.path.join(stage_root, "stage_pose_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            import json

            json.dump(metrics_summary, f, indent=2)
        Log(f"Saved stage-wise pose evaluation 鈫?{summary_path}", tag="FrontEnd")

    def _get_kf_depth_init_mode(self) -> str:
        mode = str(
            self.config["Training"].get("kf_depth_init_mode", "dap_only")
        ).lower()
        if mode == "fused_render_dap":
            if not self._warned_legacy_kf_depth_mode:
                Log(
                    "[KFDepth] kf_depth_init_mode='fused_render_dap' is deprecated; "
                    "using 'dap_only'.",
                    tag="FrontEnd",
                )
                self._warned_legacy_kf_depth_mode = True
            return "dap_only"
        if mode != "dap_only":
            raise ValueError(
                f"Unsupported kf_depth_init_mode='{mode}'. "
                "Expected one of: ['dap_only', 'fused_render_dap']."
            )
        return mode

    def _store_motion_metrics(
        self,
        viewpoint,
        prev_w2c: np.ndarray,
        cur_w2c: np.ndarray,
    ):
        if prev_w2c is None or cur_w2c is None:
            viewpoint.motion_norm_m = 0.0
            viewpoint.motion_rot_deg = 0.0
            return
        trans_delta, rot_delta = self._pose_delta_metrics(cur_w2c, prev_w2c)
        viewpoint.motion_norm_m = float(trans_delta)
        viewpoint.motion_rot_deg = float(rot_delta)

    def _assign_submap_id(self, frame_idx: int) -> int:
        if not self.enable_submap:
            submap_id = 0
            self._active_submap_id = submap_id
            self._submaps.setdefault(submap_id, {"kf_ids": []})
            if frame_idx not in self._submaps[submap_id]["kf_ids"]:
                self._submaps[submap_id]["kf_ids"].append(frame_idx)
            return submap_id
        submap_id = self._submap_manager.assign_frame(frame_idx)
        self._active_submap_id = submap_id
        self._submaps.setdefault(submap_id, {"kf_ids": []})
        if frame_idx not in self._submaps[submap_id]["kf_ids"]:
            self._submaps[submap_id]["kf_ids"].append(frame_idx)
        return submap_id

    def _select_frontend_window(self, window):
        if not window:
            return []
        window = normalize_window_order(window)
        if not self.enable_submap:
            return window
        frame_to_submap = {
            int(frame_id): int(getattr(cam, "submap_id", -1))
            for frame_id, cam in self.cameras.items()
        }
        filtered = self._submap_manager.filter_window(
            list(window), frame_to_submap, self._active_submap_id
        )
        if filtered != list(window):
            removed = [int(kf) for kf in window if kf not in filtered]
            if removed:
                Log(
                    f"[Window] dropped stale keyframes {removed} "
                    f"for active_submap={self._active_submap_id}",
                    tag="FrontEnd",
                )
        result = filtered if filtered else list(window)
        return normalize_window_order(result)

    def _compute_erp_region_masks(self, viewpoint):
        from backend.legacy_360gs.utils.pano_structure import build_pano_region_masks

        sky_mask_np = getattr(viewpoint, "erp_sky_mask", None)
        if sky_mask_np is None and getattr(viewpoint, "mono_depth", None) is not None:
            _, sky_mask_np = self._compute_dap_masks(viewpoint.mono_depth.astype(np.float32))
        if sky_mask_np is not None and getattr(viewpoint, "mono_depth", None) is not None:
            depth_np = np.asarray(viewpoint.mono_depth, dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            valid_np, _ = self._compute_dap_masks(depth_np)
            sky_mask_np = self._fill_sky_mask_above_dynamic_horizon(
                viewpoint,
                np.asarray(sky_mask_np, dtype=bool),
                depth_np,
                valid_mask=valid_np,
            )
            viewpoint.erp_sky_mask = np.asarray(sky_mask_np, dtype=bool).copy()
        if sky_mask_np is None:
            h, w = viewpoint.image_height, viewpoint.image_width
            sky_mask = torch.zeros((1, h, w), dtype=torch.bool)
        else:
            sky_mask = torch.from_numpy(np.asarray(sky_mask_np, dtype=bool))[None]
        region_masks = build_pano_region_masks(
            sky_mask,
            horizon_deg=float(self.config["Training"].get("erp_region_horizon_deg", 18.0)),
            top_pole_deg=float(self.config["Training"].get("erp_region_top_pole_deg", 65.0)),
            bottom_pole_deg=float(self.config["Training"].get("erp_region_bottom_pole_deg", 55.0)),
        )
        ignore_mask = build_erp_ignore_mask(
            int(viewpoint.image_height),
            int(viewpoint.image_width),
            self.config,
        )
        if isinstance(ignore_mask, np.ndarray):
            ignore_mask = torch.from_numpy(ignore_mask.astype(bool))[None]
        valid_mask = ~ignore_mask
        viewpoint.erp_region_masks = {
            key: value.detach().cpu() for key, value in region_masks.items()
        }
        viewpoint.erp_region_masks["ignore"] = ignore_mask.detach().cpu()
        viewpoint.erp_region_masks["valid"] = valid_mask.detach().cpu()
        viewpoint.config_training_overrides = {
            "erp_top_pole_struct_weight": float(
                self.config["Training"].get("erp_top_pole_struct_weight", 0.2)
            ),
            "erp_bottom_pole_struct_weight": float(
                self.config["Training"].get("erp_bottom_pole_struct_weight", 0.85)
            ),
        }
        return region_masks

    def _compose_pose_from_parts(
        self,
        base_pose: np.ndarray,
        rot_pose: np.ndarray | None,
        trans_pose: np.ndarray | None,
    ) -> np.ndarray:
        pose = base_pose.copy()
        if rot_pose is not None:
            pose[:3, :3] = rot_pose[:3, :3]
        if trans_pose is not None:
            pose[:3, 3] = trans_pose[:3, 3]
        return pose

    def _match_sphereglue_pair(self, prev_img: torch.Tensor, cur_img: torch.Tensor):
        def _to_uint8(t: torch.Tensor) -> np.ndarray:
            return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        result = self.sphereglue_matcher.match(_to_uint8(prev_img), _to_uint8(cur_img))
        mkpts0 = result["mkpts0"]
        mkpts1 = result["mkpts1"]
        mscores = result["mscores"]
        max_matches = int(self.config["Training"].get("sg_max_matches", 800))
        if len(mkpts0) > max_matches:
            top_idx = np.argsort(mscores)[::-1][:max_matches]
            mkpts0 = mkpts0[top_idx]
            mkpts1 = mkpts1[top_idx]
            mscores = mscores[top_idx]
        return {"mkpts0": mkpts0, "mkpts1": mkpts1, "mscores": mscores}

    def _spherical_pose_match_stats(
        self,
        prev_depth: np.ndarray,
        prev_w2c: np.ndarray,
        cur_w2c: np.ndarray,
        match_result: dict,
        prev_region_mask=None,
        cur_region_mask=None,
        prev_depth_valid_mask=None,
    ) -> dict | None:
        if prev_depth is None or match_result is None:
            return None
        try:
            from backend.legacy_360gs.utils.panoramic_pose_solver import (
                angular_errors,
                build_spherical_3d2d_correspondences,
            )

            result = self._filter_match_result_by_region(
                match_result, prev_region_mask, point_key="mkpts0"
            )
            result = self._filter_match_result_by_region(
                result, cur_region_mask, point_key="mkpts1"
            )
            mkpts0 = result["mkpts0"]
            mkpts1 = result["mkpts1"]
            if len(mkpts0) < 6:
                return None
            matches_uv = torch.from_numpy(
                np.concatenate([mkpts0, mkpts1], axis=1)
            ).double().to(self.device)
            depth_t = torch.from_numpy(prev_depth.astype(np.float64)).to(self.device)
            valid_t = (depth_t > 0.01) & (depth_t < self._dap_depth_max_valid())
            if prev_depth_valid_mask is not None:
                valid_np = np.asarray(prev_depth_valid_mask, dtype=bool)
                if valid_np.shape == prev_depth.shape:
                    valid_t &= torch.from_numpy(valid_np).to(self.device)
            ransac_cfg = self.config["Training"].get("ransac", {})
            X_ref, b_cur, _ = build_spherical_3d2d_correspondences(
                depth_ref=depth_t,
                valid_mask_ref=valid_t,
                matches_uv=matches_uv,
                depth_min=float(ransac_cfg.get("depth_min", 0.5)),
                depth_max=float(ransac_cfg.get("depth_max", 80.0)),
            )
            if X_ref is None or X_ref.shape[0] < 6:
                return None
            T_prev = torch.from_numpy(prev_w2c).double().to(self.device)
            T_cur = torch.from_numpy(cur_w2c).double().to(self.device)
            T_rel = T_cur @ torch.linalg.inv(T_prev)
            theta = angular_errors(T_rel, X_ref, b_cur)
            tau = np.deg2rad(float(ransac_cfg.get("tau_ang_deg", 1.5)))
            return {
                "count": int(theta.numel()),
                "mean_deg": float(torch.rad2deg(theta.mean()).item()),
                "median_deg": float(torch.rad2deg(theta.median()).item()),
                "inlier_ratio": float((theta < tau).float().mean().item()),
            }
        except Exception as exc:
            Log(f"[PoseCheck] spherical match stats failed: {exc}", tag="FrontEnd")
            return None

    def _source_points_to_world(
        self, X_src: torch.Tensor, src_w2c: np.ndarray
    ) -> torch.Tensor:
        """Transform row-vector camera points from a source camera to world."""
        T_src = torch.from_numpy(np.asarray(src_w2c, dtype=np.float64)).to(
            device=X_src.device, dtype=X_src.dtype
        )
        R = T_src[:3, :3]
        t = T_src[:3, 3]
        return (X_src - t.unsqueeze(0)) @ R

    def _build_multiref_pose_sources(
        self,
        *,
        cur_frame_idx: int,
        erp_cur: torch.Tensor,
        primary_matches: dict,
    ) -> list[dict]:
        sources = [
            {
                "label": "prev",
                "image": self._prev_frame_img,
                "depth": self._prev_frame_depth,
                "valid": self._prev_frame_depth_valid_mask,
                "w2c": self._prev_frame_w2c,
                "region": self._prev_frame_valid_mask,
                "consistency": self._prev_frame_consistency_mask,
                "depth_source": self._prev_frame_depth_source,
                "match_result": primary_matches,
            }
        ]
        max_sources = int(self.config["Training"].get("ransac_multiref_max_sources", 3))
        if max_sources <= 1:
            return sources

        recent_kfs = []
        for kf_id in reversed(self.kf_indices):
            kf_id = int(kf_id)
            if kf_id == cur_frame_idx or kf_id not in self.cameras:
                continue
            if recent_kfs and kf_id == recent_kfs[-1]:
                continue
            recent_kfs.append(kf_id)
            if len(recent_kfs) >= max_sources - 1:
                break

        for kf_id in recent_kfs:
            cam = self.cameras[kf_id]
            try:
                depth, valid, depth_source = self._select_pose_depth_for_viewpoint(
                    cam, reason=f"multiref_kf_{kf_id}_to_{cur_frame_idx}"
                )
            except Exception as exc:
                Log(f"[RANSAC multiref] skip kf={kf_id}: depth failed: {exc}", tag="FrontEnd")
                continue
            if depth is None:
                continue
            sources.append(
                {
                    "label": f"kf{kf_id}",
                    "image": cam.original_image,
                    "depth": depth,
                    "valid": valid,
                    "w2c": self._camera_w2c_numpy(cam),
                    "region": (getattr(cam, "erp_region_masks", None) or {}).get("valid", None),
                    "consistency": getattr(cam, "erp_consistency_mask", None),
                    "depth_source": depth_source,
                    "match_result": None,
                }
            )
        return sources[:max_sources]

    def _spherical_multiref_world_pose_init(
        self,
        erp_cur: torch.Tensor,
        sources: list[dict],
        *,
        cur_region_mask=None,
        frame_idx: int | None = None,
        T_init_np: np.ndarray | None = None,
    ):
        """Estimate absolute current w2c from multiple source views.

        Each source contributes source-depth 3-D points transformed to world
        coordinates, plus current-frame spherical bearings.  The generic PnP
        solver therefore returns the absolute current world-to-camera pose.
        """
        from backend.legacy_360gs.utils.panoramic_pose_solver import (
            angular_errors,
            build_spherical_3d2d_correspondences,
            solve_pose_spherical_3d2d_points_ransac,
        )

        ransac_cfg = self.config["Training"].get("ransac", {})
        sample_size = int(ransac_cfg.get("sample_size", 6))
        min_matches = int(self.config["Training"].get("ransac_multiref_min_matches", 24))
        use_conf_weights = bool(
            self.config["Training"].get("ransac_use_confidence_weights", True)
        )

        X_world_all = []
        b_cur_all = []
        weights_all = []
        source_stats = []
        total_matches = 0
        total_valid_pixels = []

        for source in sources:
            if (
                source.get("image") is None
                or source.get("depth") is None
                or source.get("w2c") is None
            ):
                continue
            result = source.get("match_result")
            if result is None:
                result = self._match_sphereglue_pair(source["image"], erp_cur)
            result = self._filter_match_result_by_region(
                result, source.get("region"), point_key="mkpts0"
            )
            result = self._filter_match_result_by_region(
                result, cur_region_mask, point_key="mkpts1"
            )
            mkpts0 = result["mkpts0"]
            mkpts1 = result["mkpts1"]
            mscores = result["mscores"]
            if len(mkpts0) < min_matches:
                source_stats.append(
                    {
                        "label": source.get("label", "src"),
                        "matches": int(len(mkpts0)),
                        "used": False,
                    }
                )
                continue

            weights_np = None
            sca_stats = {}
            consistency_keep = None
            if use_conf_weights:
                weights_np, consistency_keep, sca_stats = self._spherical_consistency_match_weights(
                    mkpts0,
                    mkpts1,
                    mscores,
                    source.get("consistency"),
                )
                min_keep = int(self.config["Training"].get("sca_pose_min_filtered_matches", 48))
                if consistency_keep is not None and int(consistency_keep.sum()) >= min_keep:
                    result = self._filter_match_result_by_pixel_keep(result, consistency_keep)
                    weights_np = weights_np[consistency_keep]
                    mkpts0 = result["mkpts0"]
                    mkpts1 = result["mkpts1"]
                    mscores = result["mscores"]

            matches_uv = torch.from_numpy(
                np.concatenate([mkpts0, mkpts1], axis=1)
            ).double().to(self.device)
            depth_np = np.asarray(source["depth"], dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            depth_t = torch.from_numpy(depth_np.astype(np.float64)).to(self.device)
            valid_t = (depth_t > 0.01) & (depth_t < self._dap_depth_max_valid())
            valid_np = source.get("valid")
            if valid_np is not None:
                valid_np = np.asarray(valid_np, dtype=bool)
                if valid_np.ndim == 3:
                    valid_np = valid_np[0]
                if valid_np.shape == depth_np.shape:
                    valid_t &= torch.from_numpy(valid_np).to(self.device)
            valid_ratio = float(valid_t.float().mean().item())
            X_src, b_cur, keep_idx = build_spherical_3d2d_correspondences(
                depth_ref=depth_t,
                valid_mask_ref=valid_t,
                matches_uv=matches_uv,
                depth_min=float(ransac_cfg.get("depth_min", 0.5)),
                depth_max=float(ransac_cfg.get("depth_max", 80.0)),
            )
            if X_src is None or X_src.shape[0] < sample_size:
                source_stats.append(
                    {
                        "label": source.get("label", "src"),
                        "matches": int(len(mkpts0)),
                        "valid_corr": 0 if X_src is None else int(X_src.shape[0]),
                        "valid_depth_ratio": valid_ratio,
                        "used": False,
                    }
                )
                continue

            X_world = self._source_points_to_world(X_src, source["w2c"])
            X_world_all.append(X_world)
            b_cur_all.append(b_cur)
            if weights_np is not None and keep_idx is not None:
                w_t = torch.from_numpy(weights_np.astype(np.float64)).to(self.device)
                if w_t.numel() >= int(keep_idx.max().item()) + 1:
                    weights_all.append(w_t[keep_idx])
                else:
                    weights_all.append(torch.ones((X_src.shape[0],), device=self.device, dtype=torch.float64))
            else:
                weights_all.append(torch.ones((X_src.shape[0],), device=self.device, dtype=torch.float64))
            total_matches += int(len(mkpts0))
            total_valid_pixels.append(valid_ratio)
            stat = {
                "label": source.get("label", "src"),
                "matches": int(len(mkpts0)),
                "valid_corr": int(X_src.shape[0]),
                "valid_depth_ratio": valid_ratio,
                "depth_source": str(source.get("depth_source", "unknown")),
                "used": True,
            }
            stat.update(sca_stats)
            source_stats.append(stat)

        if not X_world_all:
            return None, {
                "success": False,
                "reason": "no_multiref_correspondences",
                "sources": source_stats,
            }

        X_world = torch.cat(X_world_all, dim=0)
        b_cur = torch.cat(b_cur_all, dim=0)
        weights = torch.cat(weights_all, dim=0) if weights_all else None
        max_corr = int(self.config["Training"].get("ransac_multiref_max_corr", 1200))
        if weights is not None and X_world.shape[0] > max_corr > sample_size:
            top = torch.topk(weights, k=max_corr, largest=True).indices
            X_world = X_world[top]
            b_cur = b_cur[top]
            weights = weights[top]

        T_init = (
            torch.from_numpy(T_init_np.astype(np.float64)).to(self.device)
            if T_init_np is not None else None
        )
        result = solve_pose_spherical_3d2d_points_ransac(
            X_ref=X_world,
            b_cur=b_cur,
            sample_size=sample_size,
            max_ransac_iters=int(
                self.config["Training"].get(
                    "ransac_multiref_max_iters", ransac_cfg.get("max_iters", 80)
                )
            ),
            tau_ang_deg=float(ransac_cfg.get("tau_ang_deg", 1.5)),
            lm_iters_hypothesis=int(ransac_cfg.get("lm_iters_hyp", 5)),
            lm_iters_final=int(ransac_cfg.get("lm_iters_final", 15)),
            dtype=torch.float64,
            T_init=T_init,
            correspondence_weights=weights,
            debug_record_samples=False,
        )
        T_w2c, inliers = result[0], result[1]
        if T_w2c is None or inliers is None:
            return None, {
                "success": False,
                "reason": "multiref_solver_failed",
                "matches": int(total_matches),
                "valid_correspondences": int(X_world.shape[0]),
                "sources": source_stats,
            }

        theta = angular_errors(T_w2c, X_world, b_cur)
        n_in = int(inliers.sum().item())
        inlier_ratio = float(n_in / max(int(X_world.shape[0]), 1))
        mean_ang_deg = float(torch.rad2deg(theta.mean()).item())
        median_ang_deg = float(torch.rad2deg(theta.median()).item())
        pose_np = T_w2c.float().cpu().numpy()
        info = {
            "success": True,
            "source": "multiref_world",
            "matches": int(total_matches),
            "valid_correspondences": int(X_world.shape[0]),
            "inliers": n_in,
            "inlier_ratio": inlier_ratio,
            "valid_depth_ratio": float(np.mean(total_valid_pixels)) if total_valid_pixels else 0.0,
            "depth_source": "+".join(
                sorted({str(s.get("depth_source", "unknown")) for s in source_stats if s.get("used")})
            ),
            "t_norm": float(np.linalg.norm(pose_np[:3, 3])),
            "mean_ang_deg": mean_ang_deg,
            "median_ang_deg": median_ang_deg,
            "sources": source_stats,
        }
        Log(
            f"[RANSAC multiref] frame={frame_idx} sources="
            f"{sum(1 for s in source_stats if s.get('used'))}/{len(sources)} "
            f"corr={X_world.shape[0]} inliers={n_in} ratio={inlier_ratio:.3f} "
            f"mean_ang={mean_ang_deg:.3f}deg",
            tag="FrontEnd",
        )
        return pose_np, info

    def _filter_match_result_by_region(
        self, match_result: dict, region_mask, point_key: str = "mkpts1"
    ) -> dict:
        if match_result is None or region_mask is None:
            return match_result
        mask_np = np.asarray(region_mask, dtype=bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.ndim != 2 or len(match_result.get(point_key, [])) == 0:
            return match_result
        h, w = mask_np.shape
        xy = match_result[point_key]
        xs = np.clip(np.round(xy[:, 0]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(xy[:, 1]).astype(np.int32), 0, h - 1)
        keep = mask_np[ys, xs]
        return {
            "mkpts0": match_result["mkpts0"][keep],
            "mkpts1": match_result["mkpts1"][keep],
            "mscores": match_result["mscores"][keep],
        }

    def _filter_match_result_by_pixel_keep(self, match_result: dict, keep) -> dict:
        if match_result is None or keep is None:
            return match_result
        keep = np.asarray(keep, dtype=bool)
        if keep.shape[0] != len(match_result.get("mkpts0", [])):
            return match_result
        return {
            "mkpts0": match_result["mkpts0"][keep],
            "mkpts1": match_result["mkpts1"][keep],
            "mscores": match_result["mscores"][keep],
        }

    def _spherical_consistency_match_weights(
        self,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mscores: np.ndarray,
        prev_consistency_mask=None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        n = len(mkpts0)
        if n == 0:
            return np.zeros((0,), dtype=np.float32), None, {}
        training_cfg = self.config["Training"]
        h = int(self.config.get("Dataset", {}).get("Calibration", {}).get("height", 0))
        if h <= 0:
            h = int(max(np.max(mkpts0[:, 1]), np.max(mkpts1[:, 1])) + 1)

        scores = np.asarray(mscores, dtype=np.float32)
        if scores.shape[0] != n:
            scores = np.ones((n,), dtype=np.float32)
        scores = scores - np.nanmin(scores)
        if np.nanmax(scores) > 1e-6:
            scores = scores / np.nanmax(scores)
        weights = np.clip(scores, 0.05, 1.0)

        lat_floor = float(training_cfg.get("sca_pose_latitude_weight_floor", 0.25))
        lat_w = 0.5 * (
            latitude_weights_for_uv(mkpts0, h) + latitude_weights_for_uv(mkpts1, h)
        )
        weights *= np.clip(lat_w, lat_floor, 1.0)

        consistency_keep = None
        if prev_consistency_mask is not None:
            consistency_keep = sample_mask_at_uv(prev_consistency_mask, mkpts0, default=False)
            weak = float(training_cfg.get("sca_pose_inconsistent_weight", 0.20))
            weights *= np.where(consistency_keep, 1.0, weak).astype(np.float32)

        weights = np.nan_to_num(weights, nan=0.05, posinf=1.0, neginf=0.05)
        weights = np.clip(weights, 1e-4, 1.0).astype(np.float32)
        stats = {
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        }
        if consistency_keep is not None:
            stats["consistency_match_ratio"] = float(consistency_keep.mean())
        return weights, consistency_keep, stats

    def _maybe_update_erp_consistency_mask(self, viewpoint, render_pkg):
        if render_pkg is None or getattr(viewpoint, "mono_depth", None) is None:
            return None
        if not bool(self.config["Training"].get("enable_sca_refine_mask", False)):
            return None
        try:
            render_depth = render_pkg["depth"].detach().squeeze(0).cpu().numpy().astype(np.float32)
            opacity = render_pkg.get("opacity", None)
            opacity_np = (
                opacity.detach().squeeze(0).cpu().numpy().astype(np.float32)
                if opacity is not None else None
            )
            region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
            valid_mask = self._compose_non_sky_valid_mask(
                viewpoint,
                render_depth.shape,
                base_mask=region_masks.get("valid", None),
            )
            dvo_canonical_depth = self._dvo_canonical_depth_enabled()
            dvo_depth_for_mask = (
                self._dvo_calibrated_depth(viewpoint)
                if dvo_canonical_depth
                else None
            )
            if dvo_depth_for_mask is not None and dvo_depth_for_mask.shape != render_depth.shape:
                dvo_depth_for_mask = None
            depth_for_mask = (
                dvo_depth_for_mask.astype(np.float32)
                if dvo_depth_for_mask is not None
                else viewpoint.mono_depth.astype(np.float32)
            )
            compare_valid_mask = valid_mask.copy()
            if dvo_depth_for_mask is not None:
                compare_valid_mask &= (
                    np.isfinite(depth_for_mask)
                    & (depth_for_mask > float(self.config["Training"].get("dap_align_min_depth", 0.05)))
                    & (depth_for_mask < float(
                        self.config["Training"].get(
                            "dap_align_max_depth",
                            self.config["Training"].get("dap_depth_max_valid", 99.9),
                        )
                    ))
                )
            elif bool(self.config["Training"].get("consistency_align_to_render_depth", False)):
                depth_raw = getattr(viewpoint, "mono_depth_raw", None)
                if depth_raw is None:
                    depth_raw = viewpoint.mono_depth
                depth_raw = np.asarray(depth_raw, dtype=np.float32)
                if depth_raw.ndim == 3:
                    depth_raw = depth_raw[0]
                min_depth = float(self.config["Training"].get("dap_align_min_depth", 0.05))
                max_depth = float(
                    self.config["Training"].get(
                        "dap_align_max_depth",
                        self.config["Training"].get("dap_depth_max_valid", 99.9),
                    )
                )
                compare_valid_mask &= (
                    np.isfinite(depth_raw)
                    & (depth_raw > min_depth)
                    & (depth_raw < max_depth)
                )
                aligned_for_mask, render_scale, render_valid, render_align_stats = align_mono_depth_to_render_np(
                    render_depth,
                    depth_raw,
                    compare_valid_mask,
                    self.config,
                    opacity=opacity_np,
                    return_stats=True,
                )
                fit_pixels = int(render_align_stats.get("fit_pixels", int(compare_valid_mask.sum())))
                min_pixels = int(self.config["Training"].get("dap_align_min_pixels", 512))
                render_align_reliable = (
                    fit_pixels >= min_pixels
                    and np.isfinite(float(render_scale))
                    and aligned_for_mask.shape == render_depth.shape
                )
                if render_align_reliable:
                    depth_for_mask = aligned_for_mask.astype(np.float32)
                    compare_valid_mask = compare_valid_mask & np.asarray(render_valid, dtype=bool)
                else:
                    compare_valid_mask = valid_mask
            sca_opacity_min = float(self.config["Training"].get("sca_refine_opacity_min", 0.15))
            sca_rel_thresh = float(self.config["Training"].get("sca_refine_depth_rel_thresh", 0.12))
            novelty_opacity_min = float(
                self.config["Training"].get("kf_novelty_opacity_min", sca_opacity_min)
            )
            novelty_rel_thresh = float(
                self.config["Training"].get("kf_novelty_depth_rel_thresh", sca_rel_thresh)
            )
            novelty_abs_thresh = float(
                self.config["Training"].get("kf_novelty_depth_abs_thresh", 0.0)
            )
            novelty_rel_mode = str(self.config["Training"].get("kf_novelty_depth_rel_mode", "mono"))
            mask, stats = depth_render_consistency_mask(
                render_depth,
                depth_for_mask,
                valid_mask=compare_valid_mask,
                opacity=opacity_np,
                opacity_min=sca_opacity_min,
                rel_thresh=sca_rel_thresh,
                abs_thresh=float(self.config["Training"].get("sca_refine_depth_abs_thresh", 0.0)),
                rel_mode=str(self.config["Training"].get("sca_refine_depth_rel_mode", self.config["Training"].get("kf_novelty_depth_rel_mode", "mono"))),
            )
            novelty_mask, coverage_hole_mask, depth_conflict_mask, novelty_stats = depth_render_novelty_components(
                render_depth,
                depth_for_mask,
                valid_mask=compare_valid_mask,
                opacity=opacity_np,
                opacity_min=novelty_opacity_min,
                rel_thresh=novelty_rel_thresh,
                abs_thresh=novelty_abs_thresh,
                rel_mode=novelty_rel_mode,
                edge_guard=bool(self.config["Training"].get("kf_novelty_edge_guard", False)),
                edge_rel_thresh=float(self.config["Training"].get("kf_novelty_edge_rel_thresh", 0.08)),
            )
            abs_err = np.abs(render_depth - depth_for_mask)
            if novelty_rel_mode.lower() in {"symmetric", "sym", "balanced"}:
                rel_err = 2.0 * abs_err / np.maximum(render_depth + depth_for_mask, 1e-3)
            else:
                rel_err = abs_err / np.maximum(depth_for_mask, 1e-3)
            rel_norm = rel_err / max(float(novelty_rel_thresh), 1e-6)
            abs_norm = abs_err / max(float(novelty_abs_thresh), 1e-6)
            depth_conflict_score = np.zeros_like(render_depth, dtype=np.float32)
            score_valid = np.asarray(depth_conflict_mask, dtype=bool) & np.isfinite(rel_norm) & np.isfinite(abs_norm)
            depth_conflict_score[score_valid] = (
                rel_norm[score_valid] * abs_norm[score_valid]
            ).astype(np.float32)
            if score_valid.any():
                novelty_stats.update(
                    {
                        "depth_conflict_score_mean": float(depth_conflict_score[score_valid].mean()),
                        "depth_conflict_score_p95": float(
                            np.percentile(depth_conflict_score[score_valid], 95)
                        ),
                    }
                )
            if dvo_depth_for_mask is not None:
                dvo_info = getattr(viewpoint, "dvo_depth_scale_info", None) or {}
                stats.update(
                    {
                        "consistency_depth_source": "dvo_calibrated_dap",
                        "consistency_align_scale": 1.0,
                        "consistency_align_shift": 0.0,
                        "consistency_align_pixels": int(compare_valid_mask.sum()),
                        "consistency_align_reliable": bool(
                            dvo_info.get(
                                "dap_to_dvo_scale_stable",
                                dvo_info.get("depth_scale_stable", False),
                            )
                        ),
                        "dap_to_dvo_scale": float(
                            dvo_info.get("dap_to_dvo_scale", dvo_info.get("depth_scale", 1.0))
                        ),
                        "dap_to_dvo_scale_source": str(
                            dvo_info.get(
                                "dap_to_dvo_scale_source",
                                dvo_info.get("depth_scale_source", "-"),
                            )
                        ),
                    }
                )
                novelty_stats.update(
                    {
                        "consistency_depth_source": stats["consistency_depth_source"],
                        "consistency_align_scale": stats["consistency_align_scale"],
                        "consistency_align_shift": stats["consistency_align_shift"],
                        "consistency_align_pixels": stats["consistency_align_pixels"],
                        "consistency_align_reliable": stats["consistency_align_reliable"],
                        "dap_to_dvo_scale": stats["dap_to_dvo_scale"],
                        "dap_to_dvo_scale_source": stats["dap_to_dvo_scale_source"],
                    }
                )
            elif bool(self.config["Training"].get("consistency_align_to_render_depth", False)):
                consistency_align_reliable = (
                    bool(render_align_reliable) if "render_align_reliable" in locals() else False
                )
                consistency_depth_source = (
                    "render_aligned_dap"
                    if consistency_align_reliable
                    else "temporal_aligned_dap_fallback"
                )
                stats.update(
                    {
                        "consistency_depth_source": consistency_depth_source,
                        "consistency_align_scale": float(render_scale) if "render_scale" in locals() else 1.0,
                        "consistency_align_shift": float(render_align_stats.get("align_shift", 0.0)) if "render_align_stats" in locals() else 0.0,
                        "consistency_align_pixels": int(render_align_stats.get("fit_pixels", 0)) if "render_align_stats" in locals() else 0,
                        "consistency_align_reliable": consistency_align_reliable,
                    }
                )
                novelty_stats.update(
                    {
                        "consistency_depth_source": stats["consistency_depth_source"],
                        "consistency_align_scale": stats["consistency_align_scale"],
                        "consistency_align_shift": stats["consistency_align_shift"],
                        "consistency_align_pixels": stats["consistency_align_pixels"],
                        "consistency_align_reliable": stats["consistency_align_reliable"],
                    }
                )
            align_stats = getattr(viewpoint, "dap_depth_align_stats", None)
            if isinstance(align_stats, dict):
                stats.update(
                    {
                        "dap_scale_frame": float(align_stats.get("scale_frame", 1.0)),
                        "dap_align_pixels": int(align_stats.get("align_pixels", 0)),
                        "dap_align_reliable": bool(align_stats.get("reliable", False)),
                        "temporal_source_kf": int(align_stats.get("temporal_source_kf", -1)),
                    }
                )
                novelty_stats.update(
                    {
                        "dap_scale_frame": float(align_stats.get("scale_frame", 1.0)),
                        "dap_align_pixels": int(align_stats.get("align_pixels", 0)),
                        "dap_align_reliable": bool(align_stats.get("reliable", False)),
                        "temporal_source_kf": int(align_stats.get("temporal_source_kf", -1)),
                    }
                )
            viewpoint.erp_consistency_mask = mask
            viewpoint.erp_consistency_stats = stats
            viewpoint.kf_novelty_mask = novelty_mask
            viewpoint.kf_coverage_hole_mask = coverage_hole_mask
            viewpoint.kf_depth_conflict_mask = depth_conflict_mask
            viewpoint.kf_depth_conflict_score = depth_conflict_score
            viewpoint.kf_novelty_stats = novelty_stats
            viewpoint.kf_novelty_ratio = float(novelty_stats.get("novelty_ratio", 0.0))
            Log(
                f"[SCA mask] frame={int(getattr(viewpoint, 'uid', -1))} "
                f"consistency={float(stats.get('coverage', 0.0)):.3f} "
                f"valid={float(stats.get('valid_ratio', 0.0)):.3f} "
                f"valid_cons={float(stats.get('consistent_valid_ratio', 0.0)):.3f} "
                f"novelty={float(novelty_stats.get('novelty_ratio', 0.0)):.3f} "
                f"holes={float(novelty_stats.get('coverage_hole_ratio', 0.0)):.3f} "
                f"conflict={float(novelty_stats.get('depth_conflict_ratio', 0.0)):.3f} "
                f"novelty_pixels={int(novelty_stats.get('novelty_pixels', 0))}",
                tag="FrontEnd",
            )
            if bool(self.config["Training"].get("debug_visualize_consistency", False)):
                self._save_consistency_mask_visualization(
                    int(getattr(viewpoint, "uid", -1)),
                    viewpoint.original_image,
                    mask,
                    stats,
                    valid_mask=compare_valid_mask,
                )
            if bool(self.config["Training"].get("debug_visualize_depth_compare", False)):
                self._save_depth_compare_visualization(
                    int(getattr(viewpoint, "uid", -1)),
                    viewpoint.original_image,
                    render_depth,
                    depth_for_mask,
                    opacity=opacity_np,
                    valid_mask=compare_valid_mask,
                )
            return mask
        except Exception as exc:
            Log(f"[SCA] consistency mask update failed: {exc}", tag="FrontEnd")
            return None

    def _blend_pose_estimates(self, primary_w2c: np.ndarray, secondary_w2c: np.ndarray):
        rot_a = primary_w2c[:3, :3]
        rot_b = secondary_w2c[:3, :3]
        rot_mix = 0.7 * rot_a + 0.3 * rot_b
        u, _, vh = np.linalg.svd(rot_mix)
        rot = u @ vh
        trans = 0.7 * primary_w2c[:3, 3] + 0.3 * secondary_w2c[:3, 3]
        pose = primary_w2c.copy()
        pose[:3, :3] = rot.astype(np.float32)
        pose[:3, 3] = trans.astype(np.float32)
        return pose

    def _fuse_pano_depth(
        self,
        viewpoint,
        render_depth: np.ndarray,
        valid_rgb_np: np.ndarray,
    ):
        dap_depth = viewpoint.mono_depth.astype(np.float32)
        render_depth = render_depth.astype(np.float32)
        valid_dap, sky_mask = self._compute_dap_masks(dap_depth)
        valid_render = render_depth > 0.01

        consistency_relerr = float(
            self.config["Training"].get("dap_render_consistency_relerr", 0.25)
        )
        consistent = (
            valid_dap
            & valid_render
            & (np.abs(render_depth - dap_depth) / np.maximum(dap_depth, 1e-3) < consistency_relerr)
        )

        fused = np.zeros_like(dap_depth, dtype=np.float32)
        fused[consistent] = 0.6 * dap_depth[consistent] + 0.4 * render_depth[consistent]
        dap_only = valid_dap & (~valid_render)
        render_only = (~valid_dap) & valid_render
        inconsistent = valid_dap & valid_render & (~consistent)
        fused[dap_only] = dap_depth[dap_only]
        fused[render_only] = render_depth[render_only]
        fused[inconsistent] = render_depth[inconsistent]

        fused[sky_mask] = 0.0
        if valid_rgb_np.shape == fused.shape:
            fused[~valid_rgb_np] = 0.0

        viewpoint.erp_sky_mask = sky_mask.copy()

        stats = {
            "dap_valid_ratio": float(valid_dap.mean()),
            "render_valid_ratio": float(valid_render.mean()),
            "consistent_ratio": float(consistent.mean()),
            "sky_ratio": float(sky_mask.mean()),
        }
        return fused, stats

    def _render_coverage_ratio(self, render_pkg):
        if render_pkg is None:
            return 0.0
        opacity = render_pkg["opacity"]
        if opacity is None:
            return 0.0
        return float((opacity > 0.1).float().mean().item())
    # Add a new keyframe. Create valid pixel mask using RGB boundary threshold from config, then generate initial depth map
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        # Compute angular difference with the previous frame (not used)
        R_now = viewpoint.R
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        #print("angular difference is:",self.theta)
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]      # Check if sum of RGB channels exceeds threshold; add a new dimension to match expected shape
        if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
            region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
            valid_region = region_masks.get("valid", None)
            if valid_region is not None:
                if isinstance(valid_region, torch.Tensor):
                    valid_region = valid_region.to(device=valid_rgb.device, dtype=torch.bool)
                else:
                    valid_region = torch.from_numpy(np.asarray(valid_region, dtype=bool)).to(
                        device=valid_rgb.device
                    )
                if valid_region.ndim == 2:
                    valid_region = valid_region.unsqueeze(0)
                valid_rgb = valid_rgb & valid_region.view_as(valid_rgb)
        if self.monocular:
            if depth is None:
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)     # For the first frame, use MASt3R to estimate depth during map initialization
                print("Initial depth map stats for frame", cur_frame_idx, ":",
                    f"Max: {torch.max(initial_depth).item()}",
                    f"Min: {torch.min(initial_depth).item()}",
                    f"Mean: {torch.mean(initial_depth).item()}",
                    f"Median: {torch.median(initial_depth).item()}",
                    f"Std: {torch.std(initial_depth).item()}")
                if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
                    sky_mask = self._get_viewpoint_sky_mask(
                        viewpoint, viewpoint.mono_depth.astype(np.float32)
                    )
                    viewpoint.erp_sky_mask = sky_mask.copy()
                    initial_depth[:, sky_mask] = 0
                initial_depth[~valid_rgb.cpu()] = 0
                return initial_depth[0].numpy()
            else:                               # For non-initial keyframes, prefer DAP-only depth in panorama mode
                depth = depth.detach().clone()
                opacity = opacity.detach()

                initial_depth = depth

                if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera) \
                        and self.dap_model is not None \
                        and viewpoint.mono_depth is not None:
                    kf_depth_mode = self._get_kf_depth_init_mode()
                    dap_depth, dap_scale, aligned_valid = self._align_viewpoint_dap_depth_to_render(
                        cur_frame_idx,
                        viewpoint,
                        depth,
                        opacity,
                        valid_rgb=valid_rgb,
                        reason="keyframe",
                    )
                    align_stats = getattr(viewpoint, "dap_depth_align_stats", {}) or {}
                    Log(
                        f"[add_new_keyframe] frame {cur_frame_idx}: aligned DAP depth "
                        f"(mode={kf_depth_mode}, valid_ratio={align_stats.get('valid_ratio', 0.0):.3f}, "
                        f"insert_ratio={float(np.asarray(aligned_valid, dtype=bool).mean()) if aligned_valid is not None else 0.0:.3f}, "
                        f"sky_ratio={align_stats.get('sky_ratio', 0.0):.3f}, "
                        f"scale_frame={dap_scale:.3f}, reliable={align_stats.get('reliable', False)})",
                        tag="FrontEnd",
                    )
                    return dap_depth

                # Perspective or no-DAP path: use process_depth scale alignment
                if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
                    from backend.legacy_360gs.utils.panoramic_renderer import render_panoramic
                    render_pkg = render_panoramic(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    render_depth = (
                        render_pkg["depth"].squeeze(0).detach().cpu().numpy().astype(np.float32)
                    )  # (H_erp, W_erp) float32 numpy
                    mono_cur_proc = viewpoint.mono_depth          # (H_erp, W_erp)
                    mono_last_proc = viewpoint_last.mono_depth    # (H_erp, W_erp)
                else:
                    render_depth = initial_depth.cpu().numpy()[0]  # (H_face, W_face)
                    mono_cur = viewpoint.mono_depth
                    mono_last = viewpoint_last.mono_depth
                    if mono_cur.shape != render_depth.shape:
                        import cv2 as _cv2
                        h_r, w_r = render_depth.shape
                        mono_cur_proc = _cv2.resize(
                            mono_cur.astype(np.float32), (w_r, h_r),
                            interpolation=_cv2.INTER_LINEAR,
                        )
                        mono_last_proc = _cv2.resize(
                            mono_last.astype(np.float32), (w_r, h_r),
                            interpolation=_cv2.INTER_LINEAR,
                        )
                    else:
                        mono_cur_proc = mono_cur
                        mono_last_proc = mono_last

                initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth(
                    render_depth,
                    mono_cur_proc,
                    last_depth=mono_last_proc,
                    im1=viewpoint_last.original_image,
                    im2=viewpoint.original_image,
                    model=self.model,
                    patch_size=self.config["depth"]["patch_size"],
                    mean_threshold=self.config["depth"]["mean_threshold"],
                    std_threshold=self.config["depth"]["std_threshold"],
                    error_threshold=self.config["depth"]["error_threshold"],
                    final_error_threshold=self.config["depth"]["final_error_threshold"],
                    min_accurate_pixels_ratio=self.config["depth"]["min_accurate_pixels_ratio"],
                )

                # Correct MASt3R scale
                viewpoint.mono_depth = viewpoint.mono_depth * scale_factor

                pixel_num = viewpoint.image_height * viewpoint.image_width
                #print("Initialization info for frame", cur_frame_idx, ":", 
                #    f"Max: {np.max(initial_depth)}", f"Min: {np.min(initial_depth)}", f"Mean: {np.mean(initial_depth)}",
                #    f"Median: {np.median(initial_depth)}", f"Std: {np.std(initial_depth)}", f"Scale Factor: {scale_factor}", 
                #    f"Accurate Pixel Ratio: {num_accurate_pixels / pixel_num}", f"Accurate Pixel Ratio: {np.sum(error_mask) / pixel_num}")
                
                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0 
            return initial_depth
        # Keep ground truth depth usage
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)     
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # initial_depth is a 4D tensor (1, C, H, W); extract the first channel as (C, H, W)
    
    # Initialize the SLAM system: clear backend queue, reset state, set current frame to ground-truth pose, 
    # add a new keyframe, and push related info into the backend queue
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self._pre_refinement_poses = {}
        self._post_frontend_refine_poses = {}
        self._post_backend_local_ba_poses = {}
        self._last_rel_w2c = None
        # remove everything from the queues
        while not self.backend_queue.empty():
            unpack_queue_message(self.backend_queue.get())

        # Pose initialisation for frame 0.
        # In DAP+SphereGlue mode we do NOT rely on GT: use identity (world ==
        # camera-0 frame).  GT is still available in viewpoint.R_gt / T_gt for
        # evaluation, but is not used to seed the map.
        # In all other modes we fall back to the original GT-init behaviour.
        _use_no_gt_frontend = (
            self.dap_model is not None
            and (
                self.sphereglue_matcher is not None
                or self.frontend_mode in {"360dvo", "hybrid"}
            )
        )
        if self.panoramic_mode and _use_no_gt_frontend:
            identity_R = torch.eye(3, dtype=torch.float32, device=self.device)
            identity_T = torch.zeros(3, dtype=torch.float32, device=self.device)
            viewpoint.update_RT(identity_R, identity_T)
            Log(
                f"Frame 0: identity pose (frontend_mode={self.frontend_mode}, no GT required)",
                tag="FrontEnd",
            )
        else:
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        # get mono_depth 鈥?DAP (panoramic) or MASt3R (perspective)
        if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
            if self.dap_model is not None:
                # DAP: direct ERP metric depth, no cubemap decomposition needed
                depth_m_raw, _ = self.dap_model.infer(viewpoint.original_image)
                _, sky_mask_raw = self._compute_dap_masks(depth_m_raw.astype(np.float32))
                viewpoint.erp_sky_mask = sky_mask_raw.copy()
                depth_m = self._set_dap_depth_raw(viewpoint, depth_m_raw)
                viewpoint.mono_depth = depth_m
                self._save_depth_visualization(cur_frame_idx, depth_m)
            else:
                raise RuntimeError(
                    "Panoramic mode now requires DAP depth; "
                    "the legacy panoramic_pose fallback has been removed."
                )
        else:
            img = viewpoint.original_image
            _, get_depth = _lazy_get_pose_depth_imports()
            viewpoint.mono_depth = get_depth(img, img, self.model)

        if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
            viewpoint.submap_id = self._assign_submap_id(cur_frame_idx)
            self._compute_erp_region_masks(viewpoint)
            if self._online_360dvo_enabled():
                init_w2c = getWorld2View2(
                    viewpoint.R, viewpoint.T
                ).cpu().numpy().astype(np.float32)
                init_prior = self._bootstrap_online_360dvo_depth_scale_for_init(
                    cur_frame_idx,
                    viewpoint,
                    initial_w2c=init_w2c,
                )
                if init_prior is not None:
                    Log(
                        f"[360DVO] frame={cur_frame_idx} seeded online DPVO/bootstrap "
                        f"init={init_prior.info.get('initialized', False)} "
                        f"pose_scale={init_prior.info.get('pose_scale', init_prior.info.get('scale', 1.0)):.4f} "
                        f"depth_scale={init_prior.info.get('depth_scale', 1.0):.5f} "
                        f"depth_source={init_prior.info.get('depth_scale_source', '-')}",
                        tag="FrontEnd",
                    )
        
        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)      # Request initialization and push related info into the backend queue
        self.reset = False

        # Initialise the previous-frame cache so that frame 1's RANSAC can
        # match against frame 0's image and depth.
        if self.panoramic_mode and (
            self.sphereglue_matcher is not None
            or self.frontend_mode in {"360dvo", "hybrid"}
        ):
            self._prev_frame_img = viewpoint.original_image.clone()
            if self._ransac_depth_source() in {"render", "render_depth", "internal", "internal_depth"}:
                self._prev_frame_depth = None
                self._prev_frame_depth_valid_mask = None
                self._prev_frame_depth_source = "pending_render_depth"
            else:
                self._update_prev_pose_depth_cache(viewpoint, reason="init_mono_cache")
            w2c_0 = np.eye(4, dtype=np.float32)
            w2c_0[:3, :3] = viewpoint.R.float().cpu().numpy()
            w2c_0[:3, 3] = viewpoint.T.float().cpu().numpy()
            self._prev_frame_w2c = w2c_0
            region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
            self._prev_frame_valid_mask = region_masks.get("valid", None)
            self._prev_frame_consistency_mask = self._prev_frame_valid_mask
            self._cache_aligned_mono_depth(
                viewpoint,
                w2c_0,
                frame_idx=cur_frame_idx,
                cache_keyframe=True,
                region_valid=self._prev_frame_valid_mask,
            )
            self._store_motion_metrics(viewpoint, None, w2c_0)
            self._post_frontend_refine_poses[cur_frame_idx] = w2c_0.copy()
   
    # ------------------------------------------------------------------
    # Spherical RANSAC pose initialisation (used in panoramic tracking)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Debug visualization helpers
    # ------------------------------------------------------------------

    def _save_match_visualization(
        self,
        frame_idx: int,
        prev_img: torch.Tensor,
        cur_img: torch.Tensor,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inlier_mask=None,
    ):
        """Save side-by-side SphereGlue match image (prev | cur with lines)."""
        if self._match_vis_dir is None or frame_idx is None:
            return
        try:
            import cv2
            def _to_bgr(t: torch.Tensor) -> np.ndarray:
                rgb = (t.detach().permute(1, 2, 0).cpu().numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                return rgb[:, :, ::-1].copy()

            prev_bgr = _to_bgr(prev_img)
            cur_bgr  = _to_bgr(cur_img)
            H, W = prev_bgr.shape[:2]
            gap = 4
            canvas = np.zeros((H, W * 2 + gap, 3), dtype=np.uint8)
            canvas[:, :W]       = prev_bgr
            canvas[:, W + gap:] = cur_bgr

            # Draw at most 200 matches for legibility
            n = len(mkpts0)
            step = max(1, n // 200)
            for i in range(0, n, step):
                x0, y0 = int(mkpts0[i, 0]), int(mkpts0[i, 1])
                x1, y1 = int(mkpts1[i, 0]) + W + gap, int(mkpts1[i, 1])
                is_in = bool(inlier_mask[i]) if inlier_mask is not None else True
                color = (0, 255, 0) if is_in else (0, 0, 200)
                cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
                cv2.circle(canvas, (x0, y0), 2, color, -1)
                cv2.circle(canvas, (x1, y1), 2, color, -1)

            # Downscale to max 1920 px wide for disk-space savings
            if canvas.shape[1] > 1920:
                scale = 1920 / canvas.shape[1]
                canvas = cv2.resize(canvas,
                                    (int(canvas.shape[1] * scale),
                                     int(canvas.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)

            save_path = os.path.join(self._match_vis_dir,
                                     f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as exc:
            Log(f"[MatchVis] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    def _save_ransac_sample_visualizations(
        self,
        frame_idx: int,
        prev_img: torch.Tensor,
        cur_img: torch.Tensor,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        debug_info: dict | None,
    ):
        if self._ransac_sample_dir is None or frame_idx is None or not debug_info:
            return
        samples = debug_info.get("samples", [])
        if not samples:
            return
        try:
            import json
            import cv2

            frame_dir = os.path.join(self._ransac_sample_dir, f"frame_{frame_idx:04d}")
            os.makedirs(frame_dir, exist_ok=True)
            with open(os.path.join(frame_dir, "samples.json"), "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)

            max_vis = int(self.config["Training"].get("debug_ransac_sample_vis_max", 12))
            if max_vis <= 0:
                return

            def _to_bgr(t: torch.Tensor) -> np.ndarray:
                rgb = (t.detach().permute(1, 2, 0).cpu().numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                return rgb[:, :, ::-1].copy()

            prev_bgr = _to_bgr(prev_img)
            cur_bgr = _to_bgr(cur_img)
            H, W = prev_bgr.shape[:2]
            gap = 4
            chosen = samples[:max_vis]
            for sample in chosen:
                canvas = np.zeros((H, W * 2 + gap, 3), dtype=np.uint8)
                canvas[:, :W] = prev_bgr
                canvas[:, W + gap:] = cur_bgr
                color = (0, 255, 255) if sample.get("became_best") else (255, 180, 0)
                for idx in sample.get("match_indices", []):
                    idx = int(idx)
                    if idx < 0 or idx >= len(mkpts0):
                        continue
                    x0, y0 = int(mkpts0[idx, 0]), int(mkpts0[idx, 1])
                    x1, y1 = int(mkpts1[idx, 0]) + W + gap, int(mkpts1[idx, 1])
                    cv2.line(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (x0, y0), 4, color, -1)
                    cv2.circle(canvas, (x1, y1), 4, color, -1)
                cv2.putText(
                    canvas,
                    f"iter={sample.get('iter')} best={bool(sample.get('became_best', False))}",
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
                    os.path.join(frame_dir, f"iter_{int(sample.get('iter', 0)):04d}.jpg"),
                    canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )
        except Exception as exc:
            Log(f"[RANSACSampleVis] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    def _save_kf_render(self, frame_idx: int, viewpoint, depth_map=None):
        """Render current Gaussian map from viewpoint and save GT/render/depth panels."""
        _ = depth_map  # callers still pass depth; MDL mask panel not used here
        if self._kf_render_dir is None:
            return
        try:
            import cv2
            from backend.legacy_360gs.utils.panoramic_renderer import render_panorama_for_config

            with torch.no_grad():
                _theta0 = torch.zeros(1, 3, device=self.device)
                _rho0   = torch.zeros(1, 3, device=self.device)
                render_pkg = render_panorama_for_config(
                    viewpoint, self.gaussians, self.pipeline_params, self.background,
                    config=self.config,
                    theta=_theta0, rho=_rho0,
                )
                erp_render = render_pkg["render"]
                erp_depth = render_pkg.get("depth", None)

            def _to_bgr(t: torch.Tensor) -> np.ndarray:
                rgb = (t.detach().permute(1, 2, 0).cpu().numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                return rgb[:, :, ::-1].copy()

            gt_bgr     = _to_bgr(viewpoint.original_image)
            render_bgr = _to_bgr(erp_render)

            H, W = gt_bgr.shape[:2]
            if erp_depth is not None:
                depth_np = erp_depth.detach().squeeze(0).cpu().numpy().astype(np.float32)
                if depth_np.ndim == 3:
                    depth_np = depth_np[0]
                depth_valid = np.isfinite(depth_np) & (depth_np > 0.01)
                depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
                if depth_valid.any():
                    lo, hi = np.percentile(depth_np[depth_valid], [2.0, 98.0])
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo = float(depth_np[depth_valid].min())
                        hi = float(depth_np[depth_valid].max())
                    denom = max(hi - lo, 1e-6)
                    depth_vis = np.clip((depth_np - lo) / denom * 255.0, 0, 255).astype(np.uint8)
                    depth_vis[~depth_valid] = 0
                depth_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                if depth_bgr.shape[:2] != (H, W):
                    depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                depth_bgr = np.zeros_like(gt_bgr)

            gap = 4
            labels = ["GT", "Render", "Render Depth"]
            panels = [gt_bgr, render_bgr, depth_bgr]

            header_h = 26
            canvas_w = W * len(panels) + gap * (len(panels) - 1)
            canvas = np.zeros((H + header_h, canvas_w, 3), dtype=np.uint8)
            x = 0
            for label, panel in zip(labels, panels):
                canvas[header_h:, x:x + W] = panel
                cv2.putText(
                    canvas,
                    label,
                    (x + 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                x += W + gap

            meta = []
            if self.current_window:
                meta.append(f"ref_kf={int(self.current_window[0])}")
            novelty = getattr(viewpoint, "kf_novelty_ratio", None)
            if novelty is not None:
                meta.append(f"novelty={float(novelty):.3f}")
            if meta:
                cv2.putText(
                    canvas,
                    " | ".join(meta),
                    (max(6, W + gap + 6), 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if canvas.shape[1] > 1920:
                scale = 1920 / canvas.shape[1]
                canvas = cv2.resize(canvas,
                                    (int(canvas.shape[1] * scale),
                                     int(canvas.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)

            from backend.legacy_360gs.utils.kf_render_io import save_kf_canvas
            _results = self.config.get("Results", {})
            save_path = save_kf_canvas(
                canvas,
                os.path.join(self._kf_render_dir, f"kf_{frame_idx:04d}"),
                _results,
                )
            Log(f"[KFRender] frame {frame_idx} saved {save_path}", tag="FrontEnd")
        except Exception as exc:
            Log(f"[KFRender] frame {frame_idx} save failed: {exc}", tag="FrontEnd")

    # ------------------------------------------------------------------

    def _spherical_ransac_pose_init(
        self,
        prev_img: torch.Tensor,
        prev_depth: np.ndarray,
        prev_w2c: np.ndarray,
        erp_cur: torch.Tensor,
        match_result: dict | None = None,
        prev_region_mask=None,
        cur_region_mask=None,
        prev_consistency_mask=None,
        prev_depth_valid_mask=None,
        prev_depth_source: str = "depth",
        frame_idx: int = None,
    ):
        """Estimate w2c pose for the current frame via:
          1. SphereGlue ERP matching against the PREVIOUS CONSECUTIVE frame
             (NOT the last keyframe 鈥?consecutive frames have higher overlap).
          2. Spherical 3D-2D RANSAC using the selected metric depth of the previous frame.
          3. Compose T_rel (prev鈫抍ur) with prev_w2c to get absolute w2c.

        Args:
            prev_img:   (C, H, W) float [0,1] ERP image of the previous frame.
            prev_depth: (H, W) float32 metric depth of the previous frame.
            prev_w2c:   (4, 4) float32 w2c pose of the previous frame.
            erp_cur:    (C, H, W) float [0,1] ERP image of the current frame.

        Returns: (4, 4) float32 numpy w2c matrix.  Falls back to prev_w2c (no
                 motion assumed) on any failure.
        """
        from backend.legacy_360gs.utils.panoramic_pose_solver import solve_pose_spherical_3d2d_ransac

        fallback = prev_w2c  # assume no motion

        if prev_depth is None or prev_img is None:
            Log("[RANSAC] missing prev depth/image 鈫?no-motion fallback", tag="FrontEnd")
            self._last_ransac_info = {"success": False, "reason": "missing_prev"}
            return fallback, dict(self._last_ransac_info)

        # Convert ERP tensors (C,H,W) float [0,1] 鈫?uint8 HWC for SphereGlue
        result = match_result if match_result is not None else self._match_sphereglue_pair(prev_img, erp_cur)
        result = self._filter_match_result_by_region(result, prev_region_mask, point_key="mkpts0")
        result = self._filter_match_result_by_region(result, cur_region_mask, point_key="mkpts1")
        sca_pose_enabled = bool(self.config["Training"].get("enable_sca_pose", False))
        sca_weight_stats = {}
        pre_sca_matches = len(result["mkpts0"])
        if sca_pose_enabled and prev_consistency_mask is not None and pre_sca_matches > 0:
            _, consistency_keep, sca_weight_stats = self._spherical_consistency_match_weights(
                result["mkpts0"], result["mkpts1"], result["mscores"], prev_consistency_mask
            )
            min_keep = int(self.config["Training"].get("sca_pose_min_filtered_matches", 48))
            if consistency_keep is not None and int(consistency_keep.sum()) >= min_keep:
                result = self._filter_match_result_by_pixel_keep(result, consistency_keep)
                Log(
                    f"[SCA pose] consistency-filtered matches "
                    f"{pre_sca_matches}->{len(result['mkpts0'])}",
                    tag="FrontEnd",
                )
            elif consistency_keep is not None:
                Log(
                    f"[SCA pose] keep unfiltered matches; only "
                    f"{int(consistency_keep.sum())}/{pre_sca_matches} pass consistency",
                    tag="FrontEnd",
                )
        mkpts0 = result["mkpts0"]   # (N,2) pixel [x,y] in prev frame
        mkpts1 = result["mkpts1"]   # (N,2) pixel [x,y] in cur frame
        mscores = result["mscores"]
        n_matches = len(mkpts0)
        Log(f"[RANSAC] SphereGlue matches: {n_matches}", tag="FrontEnd")

        if n_matches < 6:
            Log("[RANSAC] too few matches 鈫?no-motion fallback", tag="FrontEnd")
            self._last_ransac_info = {
                "success": False,
                "reason": "few_matches",
                "matches": int(n_matches),
            }
            return fallback, dict(self._last_ransac_info)

        matches_uv = torch.from_numpy(
            np.concatenate([mkpts0, mkpts1], axis=1)  # (N,4): prev_xy, cur_xy
        ).double().to(self.device)
        correspondence_weights = None
        if sca_pose_enabled:
            weights_np, _, sca_weight_stats = self._spherical_consistency_match_weights(
                mkpts0, mkpts1, mscores, prev_consistency_mask
            )
            correspondence_weights = torch.from_numpy(weights_np).double().to(self.device)

        depth_t = torch.from_numpy(prev_depth.astype(np.float64)).to(self.device)
        valid_t = (depth_t > 0.01) & (depth_t < self._dap_depth_max_valid())
        if prev_depth_valid_mask is not None:
            valid_np = np.asarray(prev_depth_valid_mask, dtype=bool)
            if valid_np.shape == prev_depth.shape:
                valid_t &= torch.from_numpy(valid_np).to(self.device)
        valid_ratio = float(valid_t.float().mean().item())

        ransac_cfg = self.config["Training"].get("ransac", {})
        T_rel_init = None
        if getattr(self, "_last_rel_w2c", None) is not None:
            T_rel_init = torch.from_numpy(self._last_rel_w2c.astype(np.float64)).to(self.device)

        result_ransac = solve_pose_spherical_3d2d_ransac(
            depth_ref=depth_t,
            valid_mask_ref=valid_t,
            matches_uv=matches_uv,
            depth_min=float(ransac_cfg.get("depth_min", 0.5)),
            depth_max=float(ransac_cfg.get("depth_max", 80.0)),
            sample_size=int(ransac_cfg.get("sample_size", 6)),
            max_ransac_iters=int(ransac_cfg.get("max_iters", 80)),
            tau_ang_deg=float(ransac_cfg.get("tau_ang_deg", 1.5)),
            lm_iters_hypothesis=int(ransac_cfg.get("lm_iters_hyp", 5)),
            lm_iters_final=int(ransac_cfg.get("lm_iters_final", 15)),
            dtype=torch.float64,
            T_init=T_rel_init,
            correspondence_weights=correspondence_weights,
            debug_record_samples=bool(
                self.config["Training"].get("debug_ransac_samples", False)
            ),
        )
        T_rel = result_ransac[0]
        inliers = result_ransac[1]
        ransac_debug_info = result_ransac[5] if len(result_ransac) > 5 else None

        if T_rel is None:
            Log("[RANSAC] solver failed 鈫?no-motion fallback", tag="FrontEnd")
            self._last_ransac_info = {
                "success": False,
                "reason": "solver_failed",
                "matches": int(len(mkpts0)),
                "valid_depth_ratio": valid_ratio,
            }
            return fallback, dict(self._last_ransac_info)

        n_in = int(inliers.sum().item()) if inliers is not None else 0
        t_norm = float(T_rel[:3, 3].norm().item())
        inlier_ratio = float(n_in / max(len(mkpts0), 1))
        theta_valid = None
        try:
            from backend.legacy_360gs.utils.panoramic_pose_solver import angular_errors
            theta_valid = angular_errors(T_rel, result_ransac[2], result_ransac[3])
        except Exception:
            theta_valid = None
        mean_ang_deg = (
            float(torch.rad2deg(theta_valid.mean()).item())
            if theta_valid is not None and theta_valid.numel() > 0 else float("nan")
        )
        median_ang_deg = (
            float(torch.rad2deg(theta_valid.median()).item())
            if theta_valid is not None and theta_valid.numel() > 0 else float("nan")
        )
        Log(
            f"[RANSAC] source={prev_depth_source} inliers={n_in}/{len(mkpts0)}, "
            f"ratio={inlier_ratio:.3f}, "
            f"valid_depth={valid_ratio:.3f}, |t|={t_norm:.3f}m, "
            f"mean_ang={mean_ang_deg:.3f}deg",
            tag="FrontEnd",
        )
        self._last_ransac_info = {
            "success": True,
            "matches": int(len(mkpts0)),
            "matches_before_sca": int(pre_sca_matches),
            "inliers": n_in,
            "inlier_ratio": inlier_ratio,
            "valid_depth_ratio": valid_ratio,
            "depth_source": str(prev_depth_source),
            "t_norm": t_norm,
            "mean_ang_deg": mean_ang_deg,
            "median_ang_deg": median_ang_deg,
        }
        self._last_ransac_info.update(sca_weight_stats)

        # Save match visualisation (inlier mask may be shorter than mkpts0
        # if RANSAC subsampled; align lengths for safety)
        if self._match_vis_dir is not None and frame_idx is not None:
            try:
                in_mask = None
                if inliers is not None:
                    in_mask = inliers.cpu().numpy().astype(bool)
                    if len(in_mask) != len(mkpts0):
                        in_mask = None  # length mismatch 鈥?skip colouring
                self._save_match_visualization(
                    frame_idx, prev_img, erp_cur, mkpts0, mkpts1, in_mask)
                self._save_ransac_sample_visualizations(
                    frame_idx, prev_img, erp_cur, mkpts0, mkpts1, ransac_debug_info
                )
            except Exception:
                pass

        # Compose: T_cur_w2c = T_rel (prev鈫抍ur) @ T_prev_w2c
        T_prev_w2c = torch.from_numpy(prev_w2c).double().to(self.device)
        T_cur_w2c = T_rel @ T_prev_w2c
        return T_cur_w2c.float().cpu().numpy(), dict(self._last_ransac_info)

    # ------------------------------------------------------------------

    def tracking_panoramic(self, cur_frame_idx, viewpoint: PanoramaCamera):
        """
        Panoramic tracking:
          1. DAP metric depth for the current frame.
          2. SphereGlue + Spherical 3D-2D RANSAC pose estimation against the
             PREVIOUS CONSECUTIVE frame.
          3. Apply RANSAC pose as the initial camera pose.
          4. (Optional) ERP photometric refinement via GaussianRasterizerERP,
             using cam_rot_delta / cam_trans_delta to refine the pose with
             differentiable rendering.
          5. One no-grad forward ERP render to obtain visibility / depth
             tensors needed by the keyframe decision logic in run().
        """
        from backend.legacy_360gs.utils.panoramic_renderer import render_panorama_for_config

        erp_cur = viewpoint.original_image

        # 鈹€鈹€ Step 1: DAP depth for the current frame 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if self.dap_model is not None:
            depth_m_raw, _ = self.dap_model.infer(erp_cur)
            _, sky_mask_raw = self._compute_dap_masks(depth_m_raw.astype(np.float32))
            viewpoint.erp_sky_mask = sky_mask_raw.copy()
            depth_m = self._set_dap_depth_raw(viewpoint, depth_m_raw)
            viewpoint.mono_depth = depth_m
        else:
            raise RuntimeError(
                "Panoramic mode requires DAP depth. Pass --use_dap."
            )
        viewpoint.submap_id = self._assign_submap_id(cur_frame_idx)
        self._compute_erp_region_masks(viewpoint)
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        pure_online_360dvo = (
            self.frontend_mode == "360dvo"
            and getattr(self.frontend_360dvo, "mode", "offline_tum") == "online"
        )

        if (
            not pure_online_360dvo
            and self._ransac_depth_source() in {"render", "render_depth", "internal", "internal_depth"}
        ):
            prev_frame_idx = cur_frame_idx - self.use_every_n_frames
            refresh_depth = bool(
                self.config["Training"].get("ransac_render_refresh_before_pose", True)
            )
            if (
                self._prev_frame_img is not None
                and prev_frame_idx in self.cameras
                and (refresh_depth or self._prev_frame_depth is None)
            ):
                self._update_prev_pose_depth_cache(
                    self.cameras[prev_frame_idx],
                    reason=f"pre_ransac_frame_{cur_frame_idx}",
                )

        # 鈹€鈹€ Step 2: RANSAC pose estimation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        primary_matches = None
        spherical_pose = None
        spherical_info = {}
        dvo_prior = None
        pose_init_source = "spherical_pnp"
        fallback_to_spherical = bool(
            self.config["Training"].get("frontend_360dvo_fallback_to_spherical", True)
        ) and not pure_online_360dvo
        hybrid_strategy = str(
            self.config["Training"].get("frontend_hybrid_strategy", "prefer_360dvo")
        ).lower()
        run_hybrid_spherical_diagnostics = bool(
            self.config["Training"].get("frontend_hybrid_run_spherical_diagnostics", False)
        )
        prev_anchor_w2c = (
            self._prev_frame_w2c.copy()
            if self._prev_frame_w2c is not None
            else np.eye(4, dtype=np.float32)
        )

        def motion_fallback_pose(reason: str):
            if not bool(
                self.config["Training"].get(
                    "frontend_motion_fallback_on_failure", True
                )
            ):
                return None, {}
            if self._prev_frame_w2c is None:
                return None, {}
            if self._last_rel_w2c is not None:
                pose = (
                    self._last_rel_w2c.astype(np.float64)
                    @ self._prev_frame_w2c.astype(np.float64)
                ).astype(np.float32)
                source = "constant_velocity_fallback"
            else:
                pose = self._prev_frame_w2c.copy()
                source = "last_pose_fallback"
            dt, dr = self._pose_delta_metrics(pose, self._prev_frame_w2c)
            max_dt = float(
                self.config["Training"].get(
                    "frontend_motion_fallback_max_dt_m", 2.0
                )
            )
            max_dr = float(
                self.config["Training"].get(
                    "frontend_motion_fallback_max_dr_deg", 45.0
                )
            )
            if dt > max_dt or dr > max_dr:
                Log(
                    f"[MotionFallback] rejected reason={reason} "
                    f"dt={dt:.3f}m dr={dr:.3f}deg",
                    tag="FrontEnd",
                )
                return None, {}
            info = {
                "success": True,
                "source": source,
                "fallback_reason": reason,
                "t_norm": float(dt),
                "rel_rot_deg": float(dr),
                "inlier_ratio": 0.0,
                "valid_depth_ratio": 0.0,
                "matches": 0,
                "inliers": 0,
                "mean_ang_deg": 999.0,
                "median_ang_deg": 999.0,
            }
            Log(
                f"[MotionFallback] using {source} reason={reason} "
                f"dt={dt:.3f}m dr={dr:.3f}deg",
                tag="FrontEnd",
            )
            return pose, info

        if self.frontend_mode in {"360dvo", "hybrid"}:
            if pure_online_360dvo:
                dvo_prior = self._advance_online_360dvo_for_viewpoint(
                    cur_frame_idx,
                    viewpoint,
                    previous_w2c=self._prev_frame_w2c,
                    reason="tracking",
                )
            else:
                dvo_prior = self.frontend_360dvo.get_prior(
                    cur_frame_idx,
                    mono_depth=viewpoint.mono_depth,
                    erp_image=erp_cur,
                    previous_w2c=self._prev_frame_w2c,
                )
            if dvo_prior is not None:
                Log(
                    f"[360DVO] frame={cur_frame_idx} prior "
                    f"pose_scale={dvo_prior.info.get('pose_scale', dvo_prior.info.get('scale', 1.0)):.4f} "
                    f"source={dvo_prior.info.get('scale_source', 'configured')} "
                    f"depth_scale={dvo_prior.info.get('depth_scale', 1.0):.5f} "
                    f"depth_source={dvo_prior.info.get('depth_scale_source', '-')} "
                    f"init={dvo_prior.info.get('initialized', True)} "
                    f"conf={dvo_prior.info.get('confidence', 1.0):.3f} "
                    f"|t|={dvo_prior.info.get('t_norm', 0.0):.3f}m "
                    f"dr={dvo_prior.info.get('rel_rot_deg', 0.0):.3f}deg "
                    f"clamp={dvo_prior.info.get('translation_clamped', False)} "
                    f"clamp_reason={dvo_prior.info.get('translation_clamp_reason', '-')}",
                    tag="FrontEnd",
                )
        prefer_available_dvo = (
            dvo_prior is not None
            and (
                self.frontend_mode == "360dvo"
                or (
                    self.frontend_mode == "hybrid"
                    and hybrid_strategy != "prefer_spherical"
                    and not run_hybrid_spherical_diagnostics
                )
            )
        )
        allow_spherical = (
            self.frontend_mode in {"spherical", "hybrid"} or fallback_to_spherical
        ) and not prefer_available_dvo
        use_ransac = (
            allow_spherical
            and self.sphereglue_matcher is not None
            and self.dap_model is not None
            and self._prev_frame_img is not None
            and self._prev_frame_depth is not None
            and self._prev_frame_w2c is not None
        )
        if use_ransac:
            prev_anchor_w2c = self._prev_frame_w2c.copy()
            primary_matches = self._match_sphereglue_pair(self._prev_frame_img, erp_cur)
            full_pose, full_info = self._spherical_ransac_pose_init(
                self._prev_frame_img,
                self._prev_frame_depth,
                self._prev_frame_w2c,
                erp_cur,
                match_result=primary_matches,
                prev_region_mask=self._prev_frame_valid_mask,
                cur_region_mask=region_masks.get("valid", None),
                prev_consistency_mask=self._prev_frame_consistency_mask,
                prev_depth_valid_mask=self._prev_frame_depth_valid_mask,
                prev_depth_source=self._prev_frame_depth_source,
                frame_idx=cur_frame_idx,
            )
            spherical_pose = full_pose
            spherical_info = dict(full_info or self._last_ransac_info or {})
            if bool(self.config["Training"].get("enable_kf_anchor_pose_check", True)):
                last_kf_idx = self.current_window[0] if self.current_window else None
                min_gap = int(
                    self.config["Training"].get(
                        "kf_anchor_pose_min_gap", self.pano_force_kf_interval
                    )
                )
                if (
                    last_kf_idx is not None
                    and last_kf_idx in self.cameras
                    and (cur_frame_idx - int(last_kf_idx)) >= min_gap
                ):
                    anchor_cam = self.cameras[int(last_kf_idx)]
                    anchor_depth, anchor_valid, anchor_depth_source = self._select_pose_depth_for_viewpoint(
                        anchor_cam,
                        reason=f"kf_anchor_{int(last_kf_idx)}",
                    )
                    anchor_w2c = getWorld2View2(
                        anchor_cam.R, anchor_cam.T
                    ).cpu().numpy().astype(np.float32)
                    anchor_pose = None
                    anchor_info = {}
                    try:
                        anchor_matches = self._match_sphereglue_pair(
                            anchor_cam.original_image, erp_cur
                        )
                        anchor_pose, anchor_info = self._spherical_ransac_pose_init(
                            anchor_cam.original_image,
                            anchor_depth,
                            anchor_w2c,
                            erp_cur,
                            match_result=anchor_matches,
                            prev_region_mask=getattr(anchor_cam, "erp_region_masks", {}).get("valid", None),
                            cur_region_mask=region_masks.get("valid", None),
                            prev_consistency_mask=getattr(anchor_cam, "erp_consistency_mask", None),
                            prev_depth_valid_mask=anchor_valid,
                            prev_depth_source=anchor_depth_source,
                            frame_idx=None,
                        )
                    except Exception as exc:
                        Log(f"[RANSAC] KF-anchor check failed: {exc}", tag="FrontEnd")
                    finally:
                        self._last_ransac_info = dict(spherical_info)
                    min_anchor_inlier = float(
                        self.config["Training"].get("kf_anchor_pose_min_inlier_ratio", 0.35)
                    )
                    max_anchor_mean = float(
                        self.config["Training"].get("kf_anchor_pose_max_mean_ang_deg", 2.0)
                    )
                    if (
                        anchor_info.get("success", False)
                        and anchor_info.get("inlier_ratio", 0.0) >= min_anchor_inlier
                        and anchor_info.get("mean_ang_deg", 999.0) <= max_anchor_mean
                    ):
                        spherical_pose = anchor_pose
                        spherical_info = dict(anchor_info)
                        spherical_info["source"] = "kf_anchor"
                        Log(
                            f"[RANSAC] frame={cur_frame_idx} using KF-anchor pose "
                            f"kf={last_kf_idx} inlier={anchor_info.get('inlier_ratio', 0):.3f} "
                            f"mean_ang={anchor_info.get('mean_ang_deg', 0):.3f}deg",
                            tag="FrontEnd",
                        )
            if bool(self.config["Training"].get("enable_multiref_world_pose", False)):
                try:
                    sources = self._build_multiref_pose_sources(
                        cur_frame_idx=cur_frame_idx,
                        erp_cur=erp_cur,
                        primary_matches=primary_matches,
                    )
                    multiref_pose, multiref_info = self._spherical_multiref_world_pose_init(
                        erp_cur,
                        sources,
                        cur_region_mask=region_masks.get("valid", None),
                        frame_idx=cur_frame_idx,
                        T_init_np=spherical_pose,
                    )
                    min_multiref_inlier = float(
                        self.config["Training"].get("multiref_pose_min_inlier_ratio", 0.35)
                    )
                    max_multiref_mean = float(
                        self.config["Training"].get("multiref_pose_max_mean_ang_deg", 2.0)
                    )
                    if (
                        multiref_info.get("success", False)
                        and multiref_pose is not None
                        and multiref_info.get("inlier_ratio", 0.0) >= min_multiref_inlier
                        and multiref_info.get("mean_ang_deg", 999.0) <= max_multiref_mean
                    ):
                        multiref_dt, multiref_dr = self._pose_delta_metrics(
                            multiref_pose, prev_anchor_w2c
                        )
                        multiref_info["t_norm"] = float(multiref_dt)
                        multiref_info["rel_rot_deg"] = float(multiref_dr)
                        spherical_pose = multiref_pose
                        spherical_info = dict(multiref_info)
                        Log(
                            f"[RANSAC] frame={cur_frame_idx} using multi-ref world pose "
                            f"inlier={multiref_info.get('inlier_ratio', 0):.3f} "
                            f"mean_ang={multiref_info.get('mean_ang_deg', 0):.3f}deg "
                            f"dt={multiref_dt:.3f}m dr={multiref_dr:.3f}deg",
                            tag="FrontEnd",
                        )
                    elif multiref_info is not None:
                        Log(
                            f"[RANSAC] frame={cur_frame_idx} keep single-ref pose; "
                            f"multi-ref reason={multiref_info.get('reason', 'rejected')} "
                            f"inlier={multiref_info.get('inlier_ratio', 0):.3f} "
                            f"mean_ang={multiref_info.get('mean_ang_deg', 999):.3f}deg",
                            tag="FrontEnd",
                        )
                except Exception as exc:
                    Log(f"[RANSAC] multi-ref world pose failed: {exc}", tag="FrontEnd")
        if self.frontend_mode == "spherical":
            if spherical_pose is None:
                raise RuntimeError(
                    "Panoramic spherical frontend requires SphereGlue + DAP + "
                    "spherical RANSAC. Pass --use_dap --use_sphereglue."
                )
            pose_init = spherical_pose
            selected_pose_info = dict(spherical_info)
            pose_init_source = selected_pose_info.get("source", "spherical_pnp")
        elif self.frontend_mode == "360dvo":
            if dvo_prior is not None:
                pose_init = dvo_prior.w2c
                selected_pose_info = dict(dvo_prior.info)
                pose_init_source = "360dvo_prior"
            elif spherical_pose is not None and fallback_to_spherical:
                pose_init = spherical_pose
                selected_pose_info = dict(spherical_info)
                pose_init_source = selected_pose_info.get("source", "spherical_pnp")
                Log("[360DVO] missing prior; using spherical fallback", tag="FrontEnd")
            else:
                pose_init, selected_pose_info = motion_fallback_pose(
                    "missing_360dvo_prior"
                )
                if pose_init is None:
                    if pure_online_360dvo:
                        raise RuntimeError(
                            "Training.frontend_mode=360dvo with "
                            "Training.frontend_360dvo.mode=online did not return "
                            "an online 360DVO prior for this frame."
                        )
                    raise RuntimeError(
                        "Training.frontend_mode=360dvo but no 360DVO TUM prior is "
                        "available for this frame. Configure "
                        "Training.frontend_360dvo.tum_trajectory or enable fallback."
                    )
                pose_init_source = selected_pose_info.get(
                    "source", "motion_fallback"
                )
        else:
            if hybrid_strategy == "prefer_spherical" and spherical_pose is not None:
                pose_init = spherical_pose
                selected_pose_info = dict(spherical_info)
                pose_init_source = selected_pose_info.get("source", "spherical_pnp")
            elif dvo_prior is not None:
                pose_init = dvo_prior.w2c
                selected_pose_info = dict(dvo_prior.info)
                pose_init_source = "360dvo_prior"
                if spherical_pose is not None:
                    dt, dr = self._pose_delta_metrics(dvo_prior.w2c, spherical_pose)
                    selected_pose_info["spherical_disagree_dt_m"] = float(dt)
                    selected_pose_info["spherical_disagree_dr_deg"] = float(dr)
            elif spherical_pose is not None:
                pose_init = spherical_pose
                selected_pose_info = dict(spherical_info)
                pose_init_source = selected_pose_info.get("source", "spherical_pnp")
            else:
                reason = (
                    "missing_360dvo_prior_and_spherical_fallback"
                    if dvo_prior is None
                    else "spherical_fallback_unavailable"
                )
                pose_init, selected_pose_info = motion_fallback_pose(reason)
                if pose_init is None:
                    raise RuntimeError(
                        "Hybrid frontend has neither a valid 360DVO prior nor a "
                        "valid spherical RANSAC fallback for this frame."
                    )
                pose_init_source = selected_pose_info.get(
                    "source", "motion_fallback"
                )

        primary_ransac_info = dict(selected_pose_info or self._last_ransac_info or {})
        primary_ransac_info["frontend_mode"] = self.frontend_mode
        primary_ransac_info["pose_init_source"] = pose_init_source
        self._last_ransac_info = primary_ransac_info

        # Record pre-refinement pose for analysis
        self._pre_refinement_poses[cur_frame_idx] = pose_init.copy()
        viewpoint.pose_pre_refinement = pose_init.copy()

        # Apply RANSAC pose as the initial camera pose
        pose_init_t = torch.from_numpy(pose_init).to(self.device).float()
        viewpoint.update_RT(pose_init_t[:3, :3], pose_init_t[:3, 3])

        # 鈹€鈹€ Step 3: ERP photometric refinement 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # Only run when the map has enough Gaussians to generate a meaningful
        # photometric signal.  Skip on the very first frame or when the
        # Gaussian count is too low.
        render_pkg = None
        accepted_refine = False
        n_gaussians = (
            self.gaussians.get_xyz.shape[0]
            if self.gaussians is not None else 0
        )
        min_gaussians_for_refine = self.config["Training"].get(
            "erp_refine_min_gaussians", 500
        )
        erp_refine_itr = int(self.config["Training"].get(
            "erp_tracking_itr_num", 30
        ))
        erp_refine_warmup_kfs = int(
            self.config["Training"].get("erp_refine_warmup_kfs", 4)
        )
        min_inlier_ratio = float(
            self.config["Training"].get("erp_refine_min_inlier_ratio", 0.55)
        )
        min_valid_depth_ratio = float(
            self.config["Training"].get("erp_refine_min_valid_depth_ratio", 0.10)
        )
        min_t_m = float(self.config["Training"].get("erp_refine_min_t_m", 0.30))
        max_t_m = float(self.config["Training"].get("erp_refine_max_t_m", 0.70))
        lr_rot = float(
            self.config["Training"].get(
                "erp_refine_lr_rot", self.config["Training"]["lr"]["cam_rot_delta"]
            )
        )
        lr_trans = float(
            self.config["Training"].get(
                "erp_refine_lr_trans", self.config["Training"]["lr"]["cam_trans_delta"]
            )
        )
        step_clip_trans = float(
            self.config["Training"].get("erp_refine_step_clip_trans", 0.03)
        )
        step_clip_rot_deg = float(
            self.config["Training"].get("erp_refine_step_clip_rot_deg", 0.2)
        )
        accept_max_trans = float(
            self.config["Training"].get("erp_refine_max_trans_m", 0.15)
        )
        accept_max_rot_deg = float(
            self.config["Training"].get("erp_refine_max_rot_deg", 1.0)
        )
        accept_min_coverage = float(
            self.config["Training"].get("erp_refine_min_coverage", 0.60)
        )
        accept_min_loss_drop = float(
            self.config["Training"].get("erp_refine_min_loss_drop", 0.0)
        )
        force_accept_refine = bool(
            self.config["Training"].get("erp_refine_force_accept", False)
        )

        pose_init_w2c = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
        pose_init_R = viewpoint.R.detach().clone()
        pose_init_T = viewpoint.T.detach().clone()
        applied_refine_trans = 0.0
        applied_refine_rot = 0.0
        refine_source = pose_init_source

        ransac_info = self._last_ransac_info or {}
        # In panoramic mode, allow refine once enough Gaussians and KFs exist,
        # without waiting for the full window to be built (self.initialized
        # in monocular mode stays False until window_size KFs are inserted,
        # which is too late for early pose correction).
        _cond_initialized   = self.initialized or (
            n_gaussians >= min_gaussians_for_refine
            and len(self.kf_indices) >= erp_refine_warmup_kfs
        )
        _cond_gaussians     = n_gaussians >= min_gaussians_for_refine
        _cond_itr           = erp_refine_itr > 0
        _cond_warmup        = len(self.kf_indices) >= erp_refine_warmup_kfs
        _cond_ransac_ok     = ransac_info.get("success", False)
        _cond_inlier        = ransac_info.get("inlier_ratio", 0.0) >= min_inlier_ratio
        _cond_depth         = ransac_info.get("valid_depth_ratio", 0.0) >= min_valid_depth_ratio
        _t_norm             = ransac_info.get("t_norm", 0.0)
        _cond_t_norm        = min_t_m <= _t_norm <= max_t_m
        allow_refine = (
            _cond_initialized and _cond_gaussians and _cond_itr and _cond_warmup
            and _cond_ransac_ok and _cond_inlier and _cond_depth and _cond_t_norm
        )
        if not allow_refine:
            _reasons = []
            if not _cond_initialized:   _reasons.append("not_initialized")
            if not _cond_gaussians:     _reasons.append(f"n_gaussians={n_gaussians}<{min_gaussians_for_refine}")
            if not _cond_itr:           _reasons.append("erp_refine_itr=0")
            if not _cond_warmup:        _reasons.append(f"warmup_kfs={len(self.kf_indices)}<{erp_refine_warmup_kfs}")
            if not _cond_ransac_ok:     _reasons.append("ransac_failed")
            if not _cond_inlier:        _reasons.append(f"inlier_ratio={ransac_info.get('inlier_ratio',0.0):.3f}<{min_inlier_ratio}")
            if not _cond_depth:         _reasons.append(f"valid_depth_ratio={ransac_info.get('valid_depth_ratio',0.0):.3f}<{min_valid_depth_ratio}")
            if not _cond_t_norm:        _reasons.append(f"t_norm={_t_norm:.3f}鈭塠{min_t_m},{max_t_m}]")
            Log(f"[ERP refine] frame={cur_frame_idx} skipped: {', '.join(_reasons)}", tag="FrontEnd")

        if allow_refine:
            opt_params = [
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": lr_rot,
                    "name": "rot_{}".format(viewpoint.uid),
                },
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": lr_trans,
                    "name": "trans_{}".format(viewpoint.uid),
                },
            ]
            pose_optimizer = torch.optim.Adam(opt_params)

            from backend.legacy_360gs.utils.slam_utils import get_loss_tracking
            from backend.legacy_360gs.utils.pose_utils import update_pose

            with torch.no_grad():
                _zero = torch.zeros(1, 3, device=self.device)
                render_before = render_panorama_for_config(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    config=self.config,
                    theta=_zero,
                    rho=_zero,
                )
                self._align_viewpoint_dap_depth_to_render(
                    cur_frame_idx,
                    viewpoint,
                    render_before["depth"],
                    render_before.get("opacity", None),
                    reason="refine_before",
                )
                self._maybe_update_erp_consistency_mask(viewpoint, render_before)
                loss_before, details_before = get_loss_tracking(
                    self.config,
                    render_before["render"],
                    render_before["depth"],
                    render_before["opacity"],
                    viewpoint,
                    return_details=True,
                )
                best_state = {
                    "loss": float(loss_before.item()),
                    "R": viewpoint.R.detach().clone(),
                    "T": viewpoint.T.detach().clone(),
                    "coverage": self._render_coverage_ratio(render_before),
                    "details": {k: float(v.item()) for k, v in details_before.items()},
                }
                coverage_before = best_state["coverage"]

            for _itr in range(erp_refine_itr):
                render_pkg = render_panorama_for_config(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    config=self.config,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                pose_optimizer.zero_grad()
                loss_tracking, loss_details = get_loss_tracking(
                    self.config, image, depth, opacity, viewpoint, return_details=True
                )
                loss_tracking.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    trans_norm = viewpoint.cam_trans_delta.data.norm()
                    if trans_norm > step_clip_trans > 0:
                        viewpoint.cam_trans_delta.data.mul_(step_clip_trans / trans_norm)
                    rot_clip_rad = np.deg2rad(step_clip_rot_deg)
                    rot_norm = viewpoint.cam_rot_delta.data.norm()
                    if rot_norm > rot_clip_rad > 0:
                        viewpoint.cam_rot_delta.data.mul_(rot_clip_rad / rot_norm)
                    converged = update_pose(viewpoint)
                    cur_pose_w2c = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
                    delta_trans_norm, delta_rot_deg = self._pose_delta_metrics(
                        cur_pose_w2c, pose_init_w2c
                    )
                    coverage_ratio = self._render_coverage_ratio(render_pkg)
                    loss_item = float(loss_tracking.item())
                    if loss_item < best_state["loss"]:
                        best_state = {
                            "loss": loss_item,
                            "R": viewpoint.R.detach().clone(),
                            "T": viewpoint.T.detach().clone(),
                            "coverage": coverage_ratio,
                            "details": {k: float(v.item()) for k, v in loss_details.items()},
                        }
                if converged:
                    break
            viewpoint.update_RT(best_state["R"], best_state["T"])

            best_pose_w2c = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
            best_delta_trans, best_delta_rot = self._pose_delta_metrics(
                best_pose_w2c, pose_init_w2c
            )
            loss_drop = (
                (float(loss_before.item()) - best_state["loss"])
                / max(float(loss_before.item()), 1e-6)
            )
            accepted_refine = (
                best_state["loss"] <= float(loss_before.item())
                and loss_drop >= accept_min_loss_drop
                and best_delta_trans <= accept_max_trans
                and best_delta_rot <= accept_max_rot_deg
                and best_state["coverage"] >= accept_min_coverage
            )
            if force_accept_refine:
                accepted_refine = True
            if bool(self.config["Training"].get("erp_refine_geometric_gate", True)):
                init_stats = self._spherical_pose_match_stats(
                    self._prev_frame_depth,
                    prev_anchor_w2c,
                    pose_init_w2c,
                    primary_matches,
                    prev_region_mask=self._prev_frame_valid_mask,
                    cur_region_mask=region_masks.get("valid", None),
                    prev_depth_valid_mask=self._prev_frame_depth_valid_mask,
                )
                best_stats = self._spherical_pose_match_stats(
                    self._prev_frame_depth,
                    prev_anchor_w2c,
                    best_pose_w2c,
                    primary_matches,
                    prev_region_mask=self._prev_frame_valid_mask,
                    cur_region_mask=region_masks.get("valid", None),
                    prev_depth_valid_mask=self._prev_frame_depth_valid_mask,
                )
                max_mean_increase = float(
                    self.config["Training"].get("erp_refine_max_geom_mean_increase_deg", 0.20)
                )
                min_inlier_keep = float(
                    self.config["Training"].get("erp_refine_min_geom_inlier_keep", 0.95)
                )
                geom_ok = True
                if init_stats is not None and best_stats is not None:
                    geom_ok = (
                        best_stats["mean_deg"] <= init_stats["mean_deg"] + max_mean_increase
                        and best_stats["inlier_ratio"] >= init_stats["inlier_ratio"] * min_inlier_keep
                    )
                if not geom_ok:
                    accepted_refine = False
                    Log(
                        f"[ERP refine] geometric gate rejected frame={cur_frame_idx} "
                        f"mean {init_stats['mean_deg']:.3f}->{best_stats['mean_deg']:.3f}deg "
                        f"inlier {init_stats['inlier_ratio']:.3f}->{best_stats['inlier_ratio']:.3f}",
                        tag="FrontEnd",
                    )
            Log(
                f"[ERP refine] frame={cur_frame_idx} accepted={accepted_refine} "
                f"loss_before={float(loss_before.item()):.6f} "
                f"loss_best={best_state['loss']:.6f} "
                f"loss_drop={loss_drop:.4f} "
                f"coverage_before={coverage_before:.3f} "
                f"coverage_best={best_state['coverage']:.3f} "
                f"dt={best_delta_trans:.3f}m dr={best_delta_rot:.3f}deg "
                f"force_accept={force_accept_refine}",
                tag="FrontEnd",
            )
            if accepted_refine:
                applied_refine_trans = float(best_delta_trans)
                applied_refine_rot = float(best_delta_rot)
                refine_source = "erp_refine"
            if not accepted_refine:
                viewpoint.update_RT(pose_init_R, pose_init_T)

        # 鈹€鈹€ Step 4: Final no-grad ERP render 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # Used to obtain n_touched / depth / opacity for keyframe decisions.
        with torch.no_grad():
            render_pkg = render_panorama_for_config(
                viewpoint, self.gaussians, self.pipeline_params, self.background,
                config=self.config,
                theta=torch.zeros(1, 3, device=self.device),
                rho=torch.zeros(1, 3, device=self.device),
            )
            if render_pkg is not None:
                self.median_depth = get_median_depth(
                    render_pkg["depth"], render_pkg["opacity"]
                )
                self._align_viewpoint_dap_depth_to_render(
                    cur_frame_idx,
                    viewpoint,
                    render_pkg["depth"],
                    render_pkg.get("opacity", None),
                    reason="tracking",
                )
                self._save_depth_visualization(cur_frame_idx, viewpoint.mono_depth)
                if bool(self.config["Training"].get("debug_visualize_depth_compare", False)):
                    depth_vis_np = np.asarray(viewpoint.mono_depth, dtype=np.float32)
                    if depth_vis_np.ndim == 3:
                        depth_vis_np = depth_vis_np[0]
                    region_valid = (getattr(viewpoint, "erp_region_masks", {}) or {}).get(
                        "valid", None
                    )
                    compare_valid = self._compose_non_sky_valid_mask(
                        viewpoint,
                        depth_vis_np.shape,
                        base_mask=region_valid,
                    )
                    self._save_depth_compare_visualization(
                        cur_frame_idx,
                        viewpoint.original_image,
                        render_pkg["depth"],
                        viewpoint.mono_depth,
                        opacity=render_pkg.get("opacity", None),
                        valid_mask=compare_valid,
                    )
            else:
                self.median_depth = torch.tensor(1.0, device=self.device)
            cur_consistency_mask = self._maybe_update_erp_consistency_mask(
                viewpoint, render_pkg
            )
        # Update GUI
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image,
                gtdepth=np.zeros((viewpoint.image_height, viewpoint.image_width)),
            )
        )

        # 鈹€鈹€ Update previous-frame cache 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self._prev_frame_img = erp_cur.detach().clone()
        if pure_online_360dvo:
            self._prev_frame_depth = None
            self._prev_frame_depth_valid_mask = None
            self._prev_frame_depth_source = "online_360dvo"
        else:
            self._update_prev_pose_depth_cache(viewpoint, reason="tracking_cache")
        self._prev_frame_valid_mask = region_masks.get("valid", None)
        self._prev_frame_consistency_mask = (
            cur_consistency_mask
            if cur_consistency_mask is not None
            else self._prev_frame_valid_mask
        )
        # Only accept the refined pose as the next anchor when diagnostics agree.
        w2c_cur = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
        adopted_w2c = w2c_cur if accepted_refine else pose_init_w2c
        self._cache_aligned_mono_depth(
            viewpoint,
            adopted_w2c,
            frame_idx=cur_frame_idx,
            cache_keyframe=False,
            region_valid=self._prev_frame_valid_mask,
        )
        applied_refine_trans, applied_refine_rot = self._pose_delta_metrics(
            adopted_w2c, pose_init_w2c
        )
        viewpoint.erp_refine_applied_dt_m = float(applied_refine_trans)
        viewpoint.erp_refine_applied_dr_deg = float(applied_refine_rot)
        viewpoint.erp_refine_source = refine_source
        Log(
            f"[ERP refine pose] frame={cur_frame_idx} source={refine_source} "
            f"applied_dt={applied_refine_trans:.3f}m "
            f"applied_dr={applied_refine_rot:.3f}deg",
            tag="FrontEnd",
        )
        self._store_motion_metrics(viewpoint, prev_anchor_w2c, adopted_w2c)
        try:
            self._last_rel_w2c = (
                adopted_w2c.astype(np.float64) @ np.linalg.inv(prev_anchor_w2c.astype(np.float64))
            ).astype(np.float32)
        except Exception:
            self._last_rel_w2c = None
        self._prev_frame_w2c = adopted_w2c
        self._post_frontend_refine_poses[cur_frame_idx] = adopted_w2c.copy()

        return render_pkg

    def tracking(self, cur_frame_idx, viewpoint):
        # Dispatch to panoramic tracking if enabled
        if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
            return self.tracking_panoramic(cur_frame_idx, viewpoint)

        ##=====================Pointmap Anchored Pose Estimation(PAPE)=====================
        # The previous frame
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        pose_prev = getWorld2View2(prev.R, prev.T)
        
        # adjacent keyframe
        last_keyframe_idx = self.current_window[0]
        last_kf = self.cameras[last_keyframe_idx]
        pose_last_kf = getWorld2View2(last_kf.R, last_kf.T)
        img1 = last_kf.original_image
        
        # Estimate the relative pose between the current frame and its adjacent keyframe
        img2 = viewpoint.original_image
        get_pose, get_depth = _lazy_get_pose_depth_imports()
        rel_pose, render_depth = get_pose(img1=img1, img2=img2, model=self.model, dist_coeffs=self.dataset.dist_coeffs, 
                            viewpoint=last_kf, gaussians=self.gaussians, pipeline_params=self.pipeline_params, background=self.background)
        
        # get mono_depth from MASt3R
        viewpoint.mono_depth = get_depth(img2, img2, self.model)
        
        # Compute current frame's pose estimation
        identity_matrix = torch.eye(4, device=self.device)
        rel_pose = torch.from_numpy(rel_pose).to(self.device).float()
        # If the relative pose is identity (no motion), treat as a failure and use the previous pose
        if torch.allclose(rel_pose, identity_matrix, atol=1e-6):  
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(prev.R, prev.T)
        else:
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(pose_init[:3, :3], pose_init[:3, 3])

        # Use previous frame pose (for ablation)
        #viewpoint.update_RT(prev.R, prev.T)
        
        ## ===================================Pose Optimization=================================
        opt_params = [
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            },
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            },
        ]

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 

            if tracking_itr % 10 == 0:              
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break
        
        self.median_depth = get_median_depth(depth, opacity)        # Median of rendered depth, used to determine whether the frame is a keyframe
        prev_pose_w2c = pose_prev.detach().cpu().numpy().astype(np.float32)
        cur_pose_w2c = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
        self._store_motion_metrics(viewpoint, prev_pose_w2c, cur_pose_w2c)
        return render_pkg
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)  
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)          
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])        # Get transformation matrix from current frame to previous keyframe; extract translation and compute distance
        kf_depth_cap = float(self.config["Training"].get("kf_depth_cap", float("inf")))
        eff_depth = min(self.median_depth, kf_depth_cap)
        dist_check = dist > kf_translation * eff_depth
        dist_check2 = dist > kf_min_translation * eff_depth

        if last_keyframe_idx not in occ_aware_visibility:
            Log(
                f"[KF] missing visibility for keyframe {last_keyframe_idx}; "
                "fallback to translation-only decision",
                tag="FrontEnd",
            )
            return dist_check

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check     # Small co-visibility or large camera motion

    def _consistency_keyframe_trigger(self, cur_frame_idx: int, last_keyframe_idx: int, viewpoint) -> tuple[bool, dict]:
        if not bool(self.config["Training"].get("enable_consistency_kf_decision", False)):
            return False, {}
        min_gap = int(
            self.config["Training"].get(
                "kf_consistency_min_gap", self.config["Training"].get("kf_interval", 1)
            )
        )
        gap = int(cur_frame_idx) - int(last_keyframe_idx)
        novelty_ratio = float(getattr(viewpoint, "kf_novelty_ratio", 0.0))
        threshold = float(
            self.config["Training"].get("kf_consistency_novelty_thresh", 0.08)
        )
        valid_pixels = 0
        stats = getattr(viewpoint, "kf_novelty_stats", None)
        if isinstance(stats, dict):
            valid_pixels = int(stats.get("valid_pixels", 0))
        min_valid = int(self.config["Training"].get("kf_consistency_min_valid_pixels", 100))
        trigger = gap >= min_gap and valid_pixels >= min_valid and novelty_ratio > threshold
        info = {
            "gap": gap,
            "min_gap": min_gap,
            "novelty_ratio": novelty_ratio,
            "threshold": threshold,
            "valid_pixels": valid_pixels,
            "trigger": bool(trigger),
        }
        return bool(trigger), info
    
    # Add current frame to the window and remove the least important keyframe based on overlap ratio to keep window size within limit
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        window = normalize_window_order([cur_frame_idx] + list(window))
        window_size = int(self.config["Training"]["window_size"])
        drop_policy = str(self.config["Training"].get("window_drop_policy", "heuristic"))
        if drop_policy == "oldest":
            removed_frame = None
            if len(window) > window_size:
                removed_frame = int(window[-1])
                window = window[:window_size]
            return normalize_window_order(window), removed_frame

        N_dont_touch = 2
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            kf_visibility = occ_aware_visibility.get(kf_idx)
            if kf_visibility is None:
                Log(
                    f"[Window] missing visibility for keyframe {kf_idx}; "
                    "marking it removable",
                    tag="FrontEnd",
                )
                to_remove.append(kf_idx)
                continue
            if kf_visibility.shape != cur_frame_visibility_filter.shape:
                Log(
                    f"[Window] visibility shape mismatch for keyframe {kf_idx}: "
                    f"{tuple(kf_visibility.shape)} vs "
                    f"{tuple(cur_frame_visibility_filter.shape)}; "
                    "marking it removable",
                    tag="FrontEnd",
                )
                to_remove.append(kf_idx)
                continue
            # szymkiewicz鈥搒impson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, kf_visibility
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                kf_visibility.count_nonzero(),
            )
            if denom.item() == 0:
                to_remove.append(kf_idx)
                continue
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if (point_ratio_2 <= cut_off) and (len(window) > window_size):        
            #if (point_ratio_2 <= cut_off):
                to_remove.append(kf_idx)
        # Remove the earliest keyframe among those with overlap below the threshold
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # If the window is still too large, remove the farthest keyframe
        if len(window) > window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        return normalize_window_order(window), removed_frame
    
    # Request to add a new keyframe and push related info into the backend queue
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        current_window = normalize_window_order(current_window)
        msg = [
            "keyframe",
            cur_frame_idx,
            self._request_viewpoint_payload(viewpoint),
            current_window,
            clone_obj_to_device(depthmap, device="cpu"),
            clone_obj_to_device(self.theta, device="cpu"),
        ]
        self.backend_queue.put(pack_queue_message(msg))
        self.requested_keyframe += 1
    
    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, self._request_viewpoint_payload(viewpoint)]
        self.backend_queue.put(pack_queue_message(msg))

    def _run_loop_closure(self, cur_frame_idx, viewpoint, depth_map):
        """Phase 5: run loop closure detection and optionally trigger PGO.

        This is called after every panoramic keyframe is added.  It is a
        no-op when ``enable_loop_closure=False`` in config.
        """
        if not bool(self.config.get("Training", {}).get("enable_loop_closure", False)):
            return

        # Build c2w from current viewpoint R, T (w2c)
        import numpy as np
        from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getWorld2View2
        from backend.legacy_360gs.utils.loop_closure import pose_graph_optimize, correct_gaussian_map

        R_np = viewpoint.R.float().cpu().numpy()
        T_np = viewpoint.T.float().cpu().numpy()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_np
        w2c[:3,  3] = T_np
        c2w = np.linalg.inv(w2c)

        img = viewpoint.original_image.clamp(0, 1)
        self._loop_detector.register_keyframe(cur_frame_idx, img, c2w)

        candidates = self._loop_detector.query(cur_frame_idx)
        for cand_idx in candidates:
            dm = depth_map if depth_map is not None else None
            matches_uv = None
            # Attempt SphereGlue match between current frame and candidate
            if self.sphereglue_matcher is not None and cand_idx in self.cameras:
                try:
                    cand_img = self.cameras[cand_idx].original_image
                    match_result = self.sphereglue_matcher.match(
                        (cand_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
                        (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
                    )
                    kps0 = match_result.get("mkpts0")
                    kps1 = match_result.get("mkpts1")
                    if kps0 is not None and kps0.shape[0] >= 8:
                        matches_uv = np.concatenate([kps0, kps1], axis=1)
                except Exception:
                    matches_uv = None
            rel_pose, ok = self._loop_detector.verify(
                cur_frame_idx, cand_idx, dm, matches_uv
            )
            if ok:
                self._loop_detector.add_loop_edge(cur_frame_idx, cand_idx, rel_pose)
                Log(f"[LoopClosure] Confirmed loop {cur_frame_idx} 鈫?{cand_idx}")

        if self._loop_detector.should_optimize():
            Log("[LoopClosure] Running pose graph optimisation ...")
            nodes, poses_c2w, loop_edges = self._loop_detector.get_graph()
            old_poses = self._loop_detector.get_old_poses()
            new_poses = pose_graph_optimize(nodes, poses_c2w, loop_edges, self.config)
            changed_frames = []
            for frame_id, T_new in new_poses.items():
                T_old = old_poses.get(frame_id)
                if T_old is None:
                    continue
                if np.linalg.norm(T_new[:3, 3] - T_old[:3, 3]) > float(
                    self.config.get("Training", {}).get("loop_map_correct_threshold", 0.05)
                ):
                    changed_frames.append(int(frame_id))
            if self.gaussians is not None:
                correct_gaussian_map(
                    self.gaussians, old_poses, new_poses, self.cameras, self.config
                )
            affected_submaps = self._submap_manager.affected_submaps_from_frames(
                changed_frames,
                {frame_id: int(getattr(cam, "submap_id", -1)) for frame_id, cam in self.cameras.items()},
            )
            backend_msg = [
                "loop_closure",
                {int(k): v.astype(np.float32) for k, v in new_poses.items()},
                affected_submaps,
            ]
            self.backend_queue.put(pack_queue_message(backend_msg))
            self._loop_detector.update_poses(new_poses)
            Log("[LoopClosure] Pose graph optimisation done.")

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = [
            "init",
            cur_frame_idx,
            self._request_viewpoint_payload(viewpoint),
            clone_obj_to_device(depth_map, device="cpu"),
        ]
        self.backend_queue.put(pack_queue_message(msg))
        self.requested_init = True

    def _request_viewpoint_payload(self, viewpoint):
        return clone_obj_to_device(viewpoint, device="cpu")

    def _recv_backend_message(self):
        try:
            return unpack_queue_message(self.frontend_queue.get())
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Backend process likely died before frontend queue transfer completed. "
                "Check the backend exitcode and the latest SLAM log for the first failure."
            ) from exc

    # Synchronize data from backend, including Gaussian scene, occlusion-aware visibility, and keyframe info; update keyframe
    def sync_backend(self, data):
        tag = data[0]
        self.gaussians = data[1]
        move_obj_to_device_(
            self.gaussians,
            device=self.device,
            skip_attrs={"optimizer"},
            cpu_only_attrs=GAUSSIAN_CPU_ONLY_ATTRS,
        )
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
            if tag in {"init", "keyframe"}:
                self._post_backend_local_ba_poses[int(kf_id)] = self._camera_w2c_numpy(
                    self.cameras[kf_id]
                )
        if (
            tag in {"init", "keyframe"}
            and self.panoramic_mode
            and self._ransac_depth_source() in {"render", "render_depth", "internal", "internal_depth"}
        ):
            latest_kf_id = None
            if self.current_window:
                latest_kf_id = int(self.current_window[0])
            elif keyframes:
                latest_kf_id = int(keyframes[0][0])
            if latest_kf_id is not None and latest_kf_id in self.cameras:
                latest_cam = self.cameras[latest_kf_id]
                self._prev_frame_img = latest_cam.original_image.detach().clone()
                self._update_prev_pose_depth_cache(
                    latest_cam,
                    reason=f"backend_{tag}_cache",
                )
                self._prev_frame_valid_mask = (
                    getattr(latest_cam, "erp_region_masks", {}) or {}
                ).get("valid", None)
                self._prev_frame_consistency_mask = getattr(
                    latest_cam, "erp_consistency_mask", self._prev_frame_valid_mask
                )
                self._prev_frame_w2c = self._camera_w2c_numpy(latest_cam)
    # Clear current frame's camera data; clear CUDA cache every 10 frames
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    # Main execution loop: process messages in frontend and backend queues, perform tracking, keyframe management,
    # synchronize data, clean up resources, and save results
    def run(self):
        cur_frame_idx = 0

        # Set up debug-output directories for panoramic mode
        if self.panoramic_mode and self.save_dir:
            self._match_vis_dir = os.path.join(self.save_dir, "match_vis")
            self._kf_render_dir = os.path.join(self.save_dir, "kf_renders")
            self._depth_vis_dir = os.path.join(self.save_dir, "depth_vis")
            self._depth_compare_vis_dir = os.path.join(self.save_dir, "depth_compare_vis")
            self._consistency_vis_dir = os.path.join(self.save_dir, "consistency_mask_vis")
            self._ransac_sample_dir = os.path.join(self.save_dir, "ransac_sample_vis")
            os.makedirs(self._match_vis_dir, exist_ok=True)
            os.makedirs(self._kf_render_dir, exist_ok=True)
            os.makedirs(self._depth_vis_dir, exist_ok=True)
            os.makedirs(self._depth_compare_vis_dir, exist_ok=True)
            os.makedirs(self._consistency_vis_dir, exist_ok=True)
            os.makedirs(self._ransac_sample_dir, exist_ok=True)
            Log(f"Match vis  鈫?{self._match_vis_dir}", tag="FrontEnd")
            Log(f"KF renders 鈫?{self._kf_render_dir}", tag="FrontEnd")
            Log(f"Depth vis  鈫?{self._depth_vis_dir}", tag="FrontEnd")
            Log(f"Depth compare vis 鈫?{self._depth_compare_vis_dir}", tag="FrontEnd")
            Log(f"Consistency mask vis 鈫?{self._consistency_vis_dir}", tag="FrontEnd")
            Log(f"RANSAC sample vis 鈫?{self._ransac_sample_dir}", tag="FrontEnd")

        # For panoramic mode use face intrinsics; monocular uses dataset calibration
        if self.panoramic_mode:
            fw = self.face_w
            projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=fw / 2.0,
                fy=fw / 2.0,
                cx=fw / 2.0 - 0.5,
                cy=fw / 2.0 - 0.5,
                W=fw,
                H=fw,
            ).transpose(0, 1)
        else:
            projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.dataset.fx,
                fy=self.dataset.fy,
                cx=self.dataset.cx,
                cy=self.dataset.cy,
                W=self.dataset.width,
                H=self.dataset.height,
            ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)      
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():      
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(pack_queue_message(["pause"]))
                    continue
                else:
                    self.backend_queue.put(pack_queue_message(["unpause"]))

            if self.frontend_queue.empty():    
                tic.record()
                if cur_frame_idx >= len(self.dataset):  # If current frame index exceeds dataset length, evaluate results, save, and exit the loop
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                        self._save_stage_pose_results()
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                # Do not advance to the next frame while a keyframe is still
                # being digested by the backend; otherwise the frontend would
                # skip keyframe decisions on the intervening frames.
                if self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
               
                if self.panoramic_mode and not self.erp_as_perspective:
                    face_zfar = float(
                        self.config["Training"].get("panorama_face_zfar", 500.0)
                    )
                    viewpoint = PanoramaCamera.init_from_panorama_dataset(
                        self.dataset, cur_frame_idx, self.face_w,
                        face_zfar=face_zfar,
                    )
                    # Compute grad_mask on the ERP body image
                    viewpoint.compute_grad_mask(self.config)
                else:
                    viewpoint = Camera.init_from_dataset(
                        self.dataset, cur_frame_idx, projection_matrix
                    )
                    viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.current_window = self._select_frontend_window(self.current_window)
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                last_keyframe_idx = self.current_window[0]
                last_kf_visibility = self.occ_aware_visibility.get(last_keyframe_idx)
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval    # Frame interval is used as a criterion for keyframe selection
                curr_visibility = (render_pkg["n_touched"] > 0).long().cpu()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,         
                )
                if bool(self.config["Training"].get("kf_enforce_interval", True)):
                    create_kf = check_time and create_kf
                consistency_kf, consistency_kf_info = self._consistency_keyframe_trigger(
                    cur_frame_idx, last_keyframe_idx, viewpoint
                )
                if consistency_kf:
                    Log(
                        "[KFConsistency] triggering keyframe "
                        f"frame={cur_frame_idx} last_kf={last_keyframe_idx} "
                        f"novelty={consistency_kf_info.get('novelty_ratio', 0.0):.3f} "
                        f"thr={consistency_kf_info.get('threshold', 0.0):.3f}",
                        tag="FrontEnd",
                    )
                    if str(
                        self.config["Training"].get("kf_consistency_decision_mode", "or")
                    ).lower() == "replace":
                        create_kf = True
                    else:
                        create_kf = create_kf or consistency_kf
                if len(self.current_window) < self.window_size and last_kf_visibility is not None:
                    union = torch.logical_or(
                        curr_visibility, last_kf_visibility
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, last_kf_visibility
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                    if consistency_kf:
                        create_kf = create_kf or consistency_kf
                if self.single_thread:      
                    create_kf = check_time and create_kf
                # In panoramic mode, force a keyframe every pano_force_kf_interval
                # frames to prevent long stretches without map updates
                if self.panoramic_mode and not create_kf:
                    create_kf = (
                        (cur_frame_idx - last_keyframe_idx) >= self.pano_force_kf_interval
                    )
                if create_kf:
                    self.current_window = self._select_frontend_window(
                        self.current_window
                    )
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    kf_w2c = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy().astype(np.float32)
                    self._cache_aligned_mono_depth(
                        viewpoint,
                        kf_w2c,
                        frame_idx=cur_frame_idx,
                        cache_keyframe=True,
                        region_valid=getattr(viewpoint, "erp_region_masks", {}).get("valid", None),
                    )
                    if self.panoramic_mode and not self.erp_as_perspective:
                        self._save_kf_render(cur_frame_idx, viewpoint, depth_map=depth_map)
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )

                    # Phase 5: loop closure detection for panoramic keyframes
                    if self.panoramic_mode and isinstance(viewpoint, PanoramaCamera):
                        self._run_loop_closure(cur_frame_idx, viewpoint, depth_map)
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1          

                if (        # Evaluate camera pose if the conditions are satisfied
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()       
                if create_kf:
                    # throttle at 3fps when keyframe is added   
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:      
                data = self._recv_backend_message()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
