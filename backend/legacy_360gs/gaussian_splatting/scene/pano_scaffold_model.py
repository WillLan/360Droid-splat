from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import torch

from backend.legacy_360gs.gaussian_splatting.scene.gaussian_model import GaussianModel
from backend.legacy_360gs.gaussian_splatting.utils.general_utils import inverse_sigmoid
from backend.legacy_360gs.gaussian_splatting.utils.sh_utils import RGB2SH
from backend.legacy_360gs.utils.erp_geometry import erp_dense_pixel_center_bearings


@dataclass
class ActiveAnchorSelection:
    indices: torch.Tensor
    selection_mask: torch.Tensor
    sky_mask: torch.Tensor


@dataclass
class VoxelCandidateBatch:
    xyz: np.ndarray
    rgb: np.ndarray
    confidence: np.ndarray
    region_tag: np.ndarray
    level: np.ndarray
    voxel_size: np.ndarray
    grid_coord: np.ndarray
    insert_enabled: np.ndarray

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def select(self, indices) -> "VoxelCandidateBatch":
        idx = np.asarray(indices, dtype=np.int64)
        return VoxelCandidateBatch(
            xyz=self.xyz[idx].copy(),
            rgb=self.rgb[idx].copy(),
            confidence=self.confidence[idx].copy(),
            region_tag=self.region_tag[idx].copy(),
            level=self.level[idx].copy(),
            voxel_size=self.voxel_size[idx].copy(),
            grid_coord=self.grid_coord[idx].copy(),
            insert_enabled=self.insert_enabled[idx].copy(),
        )


class PanoScaffoldModel(GaussianModel):
    """Minimal panorama scaffold model.

    This keeps the legacy explicit-Gaussian optimizer/state layout for
    compatibility, but changes map growth to voxelized anchor insertion and
    adds scaffold-oriented metadata plus active-anchor selection.
    """

    def __init__(self, sh_degree: int, config=None):
        super().__init__(sh_degree=sh_degree, config=config)
        self.map_mode = "anchor_scaffold_panorama"
        self.active_sh_degree = min(int(sh_degree), 2)

        hierarchical = (config or {}).get("Hierarchical", {})
        skybox = (config or {}).get("SkyBox", {})
        self.depth_edges = [float(x) for x in hierarchical.get("distance_lis", [40.0, 80.0])]
        self.voxel_size_lis = [float(x) for x in hierarchical.get("voxel_size_lis", [0.12, 0.45, 1.80])]
        self.max_active_anchors_per_frame = int(
            hierarchical.get("max_active_anchors_per_frame", 30000)
        )
        self.force_all_visible = bool(hierarchical.get("force_all_visible", False))
        self.min_active_opacity = float(hierarchical.get("min_opacity", 0.005))
        self.sky_enabled = bool(skybox.get("enabled", True))
        self.sky_radius = float(skybox.get("radius", 220.0))
        self.sky_n_anchors_init = int(skybox.get("n_anchors_init", 2048))
        self.sky_init_scale = float(skybox.get("init_scale", 10.0))
        self.sky_opacity_init = float(skybox.get("opacity_init", 0.06))
        self.sky_feat_dim = int(skybox.get("feat_dim", 16))
        self.sky_freeze_xyz = bool(skybox.get("freeze_xyz", True))
        self.sky_active_budget = int(skybox.get("active_budget", min(1024, self.sky_n_anchors_init)))
        training = (config or {}).get("Training", {})
        self.anchor_tolerant_match = bool(training.get("anchor_tolerant_match", True))
        self.anchor_match_grid_radius = max(0, int(training.get("anchor_match_grid_radius", 1)))
        self.anchor_match_level_radius = max(0, int(training.get("anchor_match_level_radius", 1)))
        self.anchor_match_dist_factor = float(training.get("anchor_match_dist_factor", 1.25))

        self._anchor_level = torch.zeros(0, dtype=torch.int8)
        self._anchor_voxel_size = torch.zeros(0, dtype=torch.float32)
        self._is_sky_anchor = torch.zeros(0, dtype=torch.bool)
        self._anchor_grid_coord = torch.zeros((0, 3), dtype=torch.int32)
        self._anchor_obs_count = torch.zeros(0, dtype=torch.int32)
        self._anchor_last_seen_kf = torch.zeros(0, dtype=torch.int32)
        self._anchor_conf_accum = torch.zeros(0, dtype=torch.float32)
        self._anchor_replacement_obs = torch.zeros(0, dtype=torch.int32)
        self._anchor_inconsistent_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_depth_inconsistent_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_strong_prune_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_reset_evidence_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_sky_floater_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_replacement_hits = torch.zeros(0, dtype=torch.int32)
        self._anchor_replacement_supported = torch.zeros(0, dtype=torch.bool)
        self._local_thaw_anchor_mask = torch.zeros(0, dtype=torch.bool)
        self._local_new_anchor_mask = torch.zeros(0, dtype=torch.bool)
        self._last_local_active_stats: dict = {
            "n_local_thaw": 0,
            "n_local_new": 0,
            "n_protected_static": 0,
            "n_frozen_static": 0,
        }
        self._anchor_level_pending: Optional[torch.Tensor] = None
        self._anchor_voxel_pending: Optional[torch.Tensor] = None
        self._is_sky_anchor_pending: Optional[torch.Tensor] = None
        self._anchor_grid_coord_pending: Optional[torch.Tensor] = None
        self._anchor_obs_count_pending: Optional[torch.Tensor] = None
        self._anchor_last_seen_kf_pending: Optional[torch.Tensor] = None
        self._anchor_conf_accum_pending: Optional[torch.Tensor] = None
        self._sky_box_initialized = False
        self._last_render_stats: dict = {}
        self._last_growth_stats: dict = {"n_new_anchors_added": 0, "n_anchors_pruned": 0}
        self._last_hash_stats: dict = {
            "n_structure_hash_hits": 0,
            "n_structure_hash_near_hits": 0,
            "n_structure_hash_misses": 0,
            "n_structure_hash_new": 0,
            "n_structure_hash_merged": 0,
            "n_structure_vcd_suppressed": 0,
            "n_hash_rebuild_collisions": 0,
            "n_duplicate_pruned": 0,
            "n_structure_obs_updates": 0,
        }

    def _ensure_anchor_replacement_state(self) -> None:
        n = int(self.get_xyz.shape[0])

        def _resize_1d(t: torch.Tensor, fill_value=0, dtype=torch.int32) -> torch.Tensor:
            if t.shape[0] == n:
                return t.to(dtype=dtype, device="cpu")
            out = torch.full((n,), fill_value, dtype=dtype)
            keep = min(int(t.shape[0]), n)
            if keep > 0:
                out[:keep] = t[:keep].detach().cpu().to(dtype=dtype)
            return out

        self._anchor_replacement_obs = _resize_1d(
            self._anchor_replacement_obs, dtype=torch.int32
        )
        self._anchor_inconsistent_hits = _resize_1d(
            self._anchor_inconsistent_hits, dtype=torch.int32
        )
        self._anchor_depth_inconsistent_hits = _resize_1d(
            self._anchor_depth_inconsistent_hits, dtype=torch.int32
        )
        self._anchor_strong_prune_hits = _resize_1d(
            self._anchor_strong_prune_hits, dtype=torch.int32
        )
        self._anchor_reset_evidence_hits = _resize_1d(
            self._anchor_reset_evidence_hits, dtype=torch.int32
        )
        self._anchor_sky_floater_hits = _resize_1d(
            self._anchor_sky_floater_hits, dtype=torch.int32
        )
        self._anchor_replacement_hits = _resize_1d(
            self._anchor_replacement_hits, dtype=torch.int32
        )
        self._anchor_replacement_supported = _resize_1d(
            self._anchor_replacement_supported, fill_value=False, dtype=torch.bool
        )

    def _ensure_local_anchor_masks(self) -> None:
        n = int(self.get_xyz.shape[0])

        def _resize_bool(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] == n:
                return t.to(dtype=torch.bool, device="cpu")
            out = torch.zeros((n,), dtype=torch.bool)
            keep = min(int(t.shape[0]), n)
            if keep > 0:
                out[:keep] = t[:keep].detach().cpu().to(dtype=torch.bool)
            return out

        self._local_thaw_anchor_mask = _resize_bool(self._local_thaw_anchor_mask)
        self._local_new_anchor_mask = _resize_bool(self._local_new_anchor_mask)

    def set_local_anchor_active_sets(
        self,
        *,
        thaw_rows: list[int] | np.ndarray | torch.Tensor | None = None,
        new_start: int = -1,
        new_count: int = 0,
        current_kf_id: int | None = None,
    ) -> dict:
        self._ensure_local_anchor_masks()
        n = int(self.get_xyz.shape[0])
        thaw_mask = torch.zeros((n,), dtype=torch.bool)
        new_mask = torch.zeros((n,), dtype=torch.bool)
        if thaw_rows is not None:
            rows = torch.as_tensor(thaw_rows, dtype=torch.long)
            if rows.numel() > 0:
                rows = rows[(rows >= 0) & (rows < n)]
                if rows.numel() > 0:
                    thaw_mask[rows] = True
        if int(new_start) >= 0 and int(new_count) > 0:
            start = max(0, int(new_start))
            end = min(n, start + int(new_count))
            if end > start:
                new_mask[start:end] = True
        self._local_thaw_anchor_mask = thaw_mask
        self._local_new_anchor_mask = new_mask
        protected = self.get_protected_anchor_mask(current_kf_id=current_kf_id, device="cpu")
        frozen = protected & ~(thaw_mask | new_mask)
        self._last_local_active_stats = {
            "n_local_thaw": int(thaw_mask.sum().item()),
            "n_local_new": int(new_mask.sum().item()),
            "n_protected_static": int(protected.sum().item()),
            "n_frozen_static": int(frozen.sum().item()),
        }
        return dict(self._last_local_active_stats)

    def get_local_active_anchor_mask(self, *, device="cuda") -> torch.Tensor:
        self._ensure_local_anchor_masks()
        mask = self._local_thaw_anchor_mask | self._local_new_anchor_mask
        return mask.to(device=device)

    def get_protected_anchor_mask(
        self,
        *,
        current_window=None,
        current_kf_id: int | None = None,
        device="cuda",
    ) -> torch.Tensor:
        """Return the mask of anchors that should be treated as protected.

        Protection semantics (Fix2):

        * If ``current_window`` is provided (list/tuple/iterable of KF ids),
          an anchor is protected iff its ``birth_frame`` is **not in** the
          current mapping window. This is the more permissive ("window-based")
          rule introduced by the plan ``early-kf-blur-fix``: every KF still in
          the active sliding window is allowed to freely refine the anchors it
          created, regardless of whether ``birth < current_kf_id``.
        * Otherwise (legacy callers without window info), fall back to the
          ``birth < current_kf_id`` rule.

        Sky anchors stay protected as before (controlled by
        ``protect_sky_anchor_freeze``).
        """
        n = int(self.get_xyz.shape[0])
        if n <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)
        training_cfg = self.config.get("Training", {}) if self.config else {}
        if not bool(training_cfg.get("enable_protected_anchor_freeze", True)):
            return torch.zeros((n,), dtype=torch.bool, device=device)
        if self._birth_frame.shape[0] == n:
            birth = self._birth_frame.to(dtype=torch.int32)
        elif self._anchor_kf.shape[0] == n:
            birth = self._anchor_kf.to(dtype=torch.int32)
        else:
            return torch.zeros((n,), dtype=torch.bool, device=device)

        protected = None
        if current_window is not None:
            window_list = [int(x) for x in current_window if int(x) >= 0]
            if len(window_list) > 0:
                window_ids = torch.as_tensor(
                    sorted(set(window_list)),
                    dtype=torch.int32,
                )
                in_window = torch.isin(birth, window_ids)
                protected = (birth >= 0) & (~in_window)

        if protected is None:
            if current_kf_id is None:
                current_kf_id = getattr(self, "_current_kf_id", None)
            if current_kf_id is None or int(current_kf_id) <= 0:
                return torch.zeros((n,), dtype=torch.bool, device=device)
            protected = (birth >= 0) & (birth < int(current_kf_id))

        protect_kf_max = training_cfg.get("protected_anchor_kf_max", None)
        if protect_kf_max is not None:
            protected = protected & (birth <= int(protect_kf_max))
        if bool(training_cfg.get("protect_sky_anchor_freeze", True)) and self._is_sky_anchor.shape[0] == n:
            protected = protected | self._is_sky_anchor.to(dtype=torch.bool)
        return protected.to(device=device)

    @staticmethod
    def _mask_to_bool_np(mask) -> np.ndarray | None:
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.ndim != 2:
            return None
        return mask_np.astype(bool)

    def _project_anchor_uv_numpy(self, cam, n: int, h: int, w: int):
        if n <= 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=bool)
        xyz = self.get_xyz[:n].detach().cpu().numpy().astype(np.float64, copy=False)
        R_np = cam.R.float().cpu().numpy().astype(np.float64)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)
        pts_cam = (R_np @ xyz.T).T + T_np
        radius = np.linalg.norm(pts_cam, axis=1)
        valid = np.isfinite(radius) & (radius > 1e-6)
        radius_safe = np.maximum(radius, 1e-12)
        x = pts_cam[:, 0] / radius_safe
        y = pts_cam[:, 1] / radius_safe
        z = pts_cam[:, 2] / radius_safe
        lam = np.arctan2(x, z)
        phi = np.arcsin(np.clip(y, -1.0, 1.0))
        u = w * (lam / (2.0 * math.pi) + 0.5) - 0.5
        v = h * (phi / math.pi + 0.5) - 0.5
        valid &= np.isfinite(u) & np.isfinite(v) & (v >= 0.0) & (v <= h - 1.0)
        return u.astype(np.float32), v.astype(np.float32), valid

    @staticmethod
    def _sample_mask_nearest(mask_np: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        h, w = mask_np.shape
        xs = np.mod(np.round(u).astype(np.int64), w)
        ys = np.clip(np.round(v).astype(np.int64), 0, h - 1)
        return mask_np[ys, xs].astype(bool)

    @staticmethod
    def _sample_float_nearest(image_np: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        h, w = image_np.shape
        xs = np.mod(np.round(u).astype(np.int64), w)
        ys = np.clip(np.round(v).astype(np.int64), 0, h - 1)
        return image_np[ys, xs].astype(np.float32, copy=False)

    def _new_anchor_neighbor_support(
        self,
        old_indices: np.ndarray,
        *,
        old_count: int,
        new_anchor_start: int,
        new_anchor_count: int,
        grid_radius: int,
    ) -> np.ndarray:
        support = np.zeros((old_indices.shape[0],), dtype=bool)
        total = int(self.get_xyz.shape[0])
        if (
            old_indices.size == 0
            or new_anchor_count <= 0
            or self._anchor_grid_coord.shape[0] != total
            or self._anchor_level.shape[0] != total
        ):
            return support
        new_start = max(0, int(new_anchor_start))
        new_end = min(total, new_start + int(new_anchor_count))
        if new_start >= new_end or old_count <= 0:
            return support

        grid_radius = max(0, int(grid_radius))
        new_levels = self._anchor_level[new_start:new_end].cpu().numpy().astype(np.int32)
        new_grids = self._anchor_grid_coord[new_start:new_end].cpu().numpy().astype(np.int32)
        new_by_level: dict[int, set[tuple[int, int, int]]] = {}
        for level, grid in zip(new_levels, new_grids):
            new_by_level.setdefault(int(level), set()).add(
                (int(grid[0]), int(grid[1]), int(grid[2]))
            )
        if not new_by_level:
            return support

        old_levels = self._anchor_level[:old_count].cpu().numpy().astype(np.int32)
        old_grids = self._anchor_grid_coord[:old_count].cpu().numpy().astype(np.int32)
        offsets = self._neighbor_offsets(grid_radius)
        for out_i, row_idx in enumerate(old_indices):
            row = int(row_idx)
            level = int(old_levels[row])
            level_keys = new_by_level.get(level)
            if not level_keys:
                continue
            base = old_grids[row]
            for off in offsets:
                key = (
                    int(base[0] + off[0]),
                    int(base[1] + off[1]),
                    int(base[2] + off[2]),
                )
                if key in level_keys:
                    support[out_i] = True
                    break
        return support

    def record_anchor_replacement_evidence(
        self,
        cam,
        *,
        inconsistent_mask,
        aligned_depth=None,
        valid_mask=None,
        sky_mask=None,
        old_anchor_count: int,
        new_anchor_start: int,
        new_anchor_count: int,
    ) -> dict:
        """Accumulate multi-evidence reset/prune signals for existing anchors."""
        mask_np = self._mask_to_bool_np(inconsistent_mask)
        if mask_np is None:
            return {}
        self._ensure_anchor_replacement_state()
        total = int(self.get_xyz.shape[0])
        old_count = min(max(0, int(old_anchor_count)), total)
        if old_count <= 0:
            return {}
        h, w = mask_np.shape
        u, v, valid = self._project_anchor_uv_numpy(cam, old_count, h, w)
        if not np.any(valid):
            return {
                "dia_anchor_evidence_obs": 0,
                "dia_anchor_inconsistent_hits": 0,
                "dia_anchor_replacement_hits": 0,
                "dia_anchor_supported": 0,
            }
        xyz = self.get_xyz[:old_count].detach().cpu().numpy().astype(np.float64, copy=False)
        R_np = cam.R.float().cpu().numpy().astype(np.float64)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)
        pts_cam = (R_np @ xyz.T).T + T_np
        old_depth = np.linalg.norm(pts_cam, axis=1).astype(np.float32)
        valid &= np.isfinite(old_depth) & (old_depth > 1e-6)

        valid_np = self._mask_to_bool_np(valid_mask)
        if valid_np is not None and valid_np.shape == mask_np.shape:
            valid &= self._sample_mask_nearest(valid_np, u, v)

        sky_np = self._mask_to_bool_np(sky_mask)
        if sky_np is None:
            sky_np = self._mask_to_bool_np(getattr(cam, "erp_sky_mask", None))
        projected_to_sky = np.zeros((old_count,), dtype=bool)
        if sky_np is not None and sky_np.shape == mask_np.shape:
            projected_to_sky = valid & self._sample_mask_nearest(sky_np, u, v)

        is_sky_anchor = np.zeros((old_count,), dtype=bool)
        if self._is_sky_anchor.shape[0] >= old_count:
            is_sky_anchor = self._is_sky_anchor[:old_count].cpu().numpy().astype(bool, copy=False)
        elif self._is_sky.shape[0] >= old_count:
            is_sky_anchor = self._is_sky[:old_count].cpu().numpy().astype(bool, copy=False)
        non_sky_anchor = ~is_sky_anchor

        sampled = self._sample_mask_nearest(mask_np, u, v)
        valid_non_sky = valid & non_sky_anchor & (~projected_to_sky)
        inconsistent = valid_non_sky & sampled

        depth_np = None
        if aligned_depth is not None:
            depth_np = np.asarray(aligned_depth, dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            if depth_np.shape != mask_np.shape:
                depth_np = None
        rel_err = np.zeros((old_count,), dtype=np.float32)
        depth_inconsistent = np.zeros((old_count,), dtype=bool)
        depth_valid = np.zeros((old_count,), dtype=bool)
        training_cfg = self.config.get("Training", {}) if self.config else {}
        rel_thresh = float(training_cfg.get("global_prune_depth_rel_thresh", 0.15))
        far_prune_depth_start = training_cfg.get("global_prune_far_depth_start", None)
        far_prune_rel_mult = max(1.0, float(training_cfg.get("global_prune_far_depth_rel_thresh_mult", 1.0)))
        far_depth_valid = np.zeros((old_count,), dtype=bool)
        if depth_np is not None:
            dap_depth = self._sample_float_nearest(depth_np, u, v)
            depth_valid = valid_non_sky & np.isfinite(dap_depth) & (dap_depth > 0.01)
            rel_err[depth_valid] = np.abs(old_depth[depth_valid] - dap_depth[depth_valid]) / np.maximum(
                dap_depth[depth_valid], 1e-3
            )
            rel_thresh_per_anchor = np.full((old_count,), rel_thresh, dtype=np.float32)
            if (
                far_prune_depth_start is not None
                and np.isfinite(float(far_prune_depth_start))
            ):
                far_depth_valid = depth_valid & (dap_depth >= float(far_prune_depth_start))
                rel_thresh_per_anchor[far_depth_valid] = rel_thresh * far_prune_rel_mult
            depth_inconsistent = depth_valid & (rel_err > rel_thresh_per_anchor)

        candidate_rows = np.flatnonzero(valid_non_sky & (inconsistent | depth_inconsistent))
        protected_blocked_count = 0
        if (
            bool(training_cfg.get("dia_anchor_block_protected_evidence", False))
            and bool(training_cfg.get("enable_protected_anchor_freeze", True))
            and candidate_rows.size > 0
        ):
            current_kf_id = int(getattr(cam, "uid", -1))
            protected = self.get_protected_anchor_mask(
                current_kf_id=current_kf_id, device="cpu"
            ).numpy()
            self._ensure_local_anchor_masks()
            local_thaw = self._local_thaw_anchor_mask.numpy()
            blocked = protected & (~local_thaw)
            keep_candidate = ~blocked[candidate_rows]
            protected_blocked_count = int((~keep_candidate).sum())
            candidate_rows = candidate_rows[keep_candidate]
        grid_radius = int(
            training_cfg.get(
                "global_prune_replacement_neighbor_grid_radius",
                training_cfg.get("dia_anchor_replacement_neighbor_grid_radius", 2),
            )
        )
        supported = self._new_anchor_neighbor_support(
            candidate_rows,
            old_count=old_count,
            new_anchor_start=int(new_anchor_start),
            new_anchor_count=int(new_anchor_count),
            grid_radius=grid_radius,
        )
        supported_rows = candidate_rows[supported]
        reset_rows = supported_rows[
            inconsistent[supported_rows] | depth_inconsistent[supported_rows]
        ]

        scale_max = self.get_scaling[:old_count].max(dim=1).values.detach().cpu().numpy()
        radii_np = self.max_radii2D[:old_count].detach().cpu().numpy()
        scale_thr = float(training_cfg.get("global_prune_sky_floater_scale_thresh", 4.0))
        radii_thr = float(training_cfg.get("global_prune_sky_floater_radii_thresh", 80.0))
        oversized = (scale_max > scale_thr) | (radii_np > radii_thr)
        require_oversized = bool(
            training_cfg.get("global_prune_sky_floater_require_oversized", True)
        )
        sky_floater = valid & non_sky_anchor & projected_to_sky
        if require_oversized:
            sky_floater &= oversized
        strong_prune = inconsistent & depth_inconsistent

        valid_t = torch.from_numpy(np.flatnonzero(valid_non_sky | sky_floater).astype(np.int64))
        inconsistent_t = torch.from_numpy(candidate_rows[inconsistent[candidate_rows]].astype(np.int64))
        depth_t = torch.from_numpy(candidate_rows[depth_inconsistent[candidate_rows]].astype(np.int64))
        strong_t = torch.from_numpy(candidate_rows[strong_prune[candidate_rows]].astype(np.int64))
        replacement_t = torch.from_numpy(supported_rows.astype(np.int64))
        reset_t = torch.from_numpy(reset_rows.astype(np.int64))
        sky_t = torch.from_numpy(np.flatnonzero(sky_floater).astype(np.int64))
        if valid_t.numel() > 0:
            self._anchor_replacement_obs[valid_t] += 1
        if inconsistent_t.numel() > 0:
            self._anchor_inconsistent_hits[inconsistent_t] += 1
        if depth_t.numel() > 0:
            self._anchor_depth_inconsistent_hits[depth_t] += 1
        if strong_t.numel() > 0:
            self._anchor_strong_prune_hits[strong_t] += 1
        if replacement_t.numel() > 0:
            self._anchor_replacement_hits[replacement_t] += 1
            self._anchor_replacement_supported[replacement_t] = True
        if reset_t.numel() > 0:
            self._anchor_reset_evidence_hits[reset_t] += 1
        if sky_t.numel() > 0:
            self._anchor_sky_floater_hits[sky_t] += 1

        rel_valid = rel_err[depth_valid]
        rel_p50 = float(np.percentile(rel_valid, 50)) if rel_valid.size > 0 else 0.0
        rel_p90 = float(np.percentile(rel_valid, 90)) if rel_valid.size > 0 else 0.0
        rel_p95 = float(np.percentile(rel_valid, 95)) if rel_valid.size > 0 else 0.0
        far_rel_valid = rel_err[far_depth_valid]
        far_rel_p90 = float(np.percentile(far_rel_valid, 90)) if far_rel_valid.size > 0 else 0.0

        return {
            "dia_anchor_evidence_obs": int(valid_t.numel()),
            "dia_anchor_inconsistent_hits": int(inconsistent_t.numel()),
            "dia_anchor_depth_inconsistent_hits": int(depth_t.numel()),
            "dia_anchor_far_depth_valid": int(far_depth_valid.sum()),
            "dia_anchor_far_depth_inconsistent_hits": int(
                (depth_inconsistent & far_depth_valid).sum()
            ),
            "dia_anchor_strong_prune_hits": int(strong_t.numel()),
            "dia_anchor_replacement_hits": int(replacement_t.numel()),
            "dia_anchor_reset_evidence_hits": int(reset_t.numel()),
            "dia_anchor_sky_floater_hits": int(sky_t.numel()),
            "dia_anchor_supported": int(self._anchor_replacement_supported.sum().item()),
            "dia_anchor_protected_evidence_blocked": protected_blocked_count,
            "dia_anchor_rel_err_p50": rel_p50,
            "dia_anchor_rel_err_p90": rel_p90,
            "dia_anchor_rel_err_p95": rel_p95,
            "dia_anchor_far_rel_err_p90": far_rel_p90,
        }

    def _depth_to_level(self, depth_values: np.ndarray) -> np.ndarray:
        edges = self.depth_edges
        if len(edges) < 2:
            return np.zeros_like(depth_values, dtype=np.int8)
        level = np.zeros_like(depth_values, dtype=np.int8)
        level[depth_values >= edges[0]] = 1
        level[depth_values >= edges[1]] = 2
        return level

    def _voxelize_level(
        self,
        pts_world: np.ndarray,
        rgb_valid: np.ndarray,
        region_valid: np.ndarray,
        level_valid: np.ndarray,
    ):
        pts_chunks = []
        rgb_chunks = []
        region_chunks = []
        level_chunks = []
        voxel_chunks = []
        for level_id, voxel_size in enumerate(self.voxel_size_lis):
            mask = level_valid == level_id
            if not np.any(mask):
                continue
            pts_l = pts_world[mask]
            rgb_l = rgb_valid[mask]
            region_l = region_valid[mask]
            vox = np.floor(pts_l / max(voxel_size, 1e-6)).astype(np.int64)
            _, keep_idx = np.unique(vox, axis=0, return_index=True)
            keep_idx = np.sort(keep_idx)
            pts_chunks.append(pts_l[keep_idx])
            rgb_chunks.append(rgb_l[keep_idx])
            region_chunks.append(region_l[keep_idx])
            level_chunks.append(np.full((keep_idx.shape[0],), level_id, dtype=np.int8))
            voxel_chunks.append(np.full((keep_idx.shape[0],), voxel_size, dtype=np.float32))
        if not pts_chunks:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int8),
                np.zeros((0,), dtype=np.int8),
                np.zeros((0,), dtype=np.float32),
            )
        return (
            np.concatenate(pts_chunks, axis=0).astype(np.float32),
            np.concatenate(rgb_chunks, axis=0).astype(np.float32),
            np.concatenate(region_chunks, axis=0).astype(np.int8),
            np.concatenate(level_chunks, axis=0).astype(np.int8),
            np.concatenate(voxel_chunks, axis=0).astype(np.float32),
        )

    def _build_single_hemisphere_skybox(self, cam):
        mono_depth = getattr(cam, "mono_depth", None)
        if mono_depth is None:
            sky_rgb = self._FALLBACK_SKY_RGB.copy()
        else:
            sky_rgb = self._estimate_sky_band_rgb_erp(
                cam,
                np.asarray(mono_depth, dtype=np.float32),
                band="upper_all",
            )
        pts_world, rgb = self._create_fibonacci_sky_band(
            cam=cam,
            radius=self.sky_radius,
            n_samples=self.sky_n_anchors_init,
            elev_min_deg=0.0,
            elev_max_deg=90.0,
            sky_rgb=sky_rgb,
        )
        n = pts_world.shape[0]
        region = np.full((n,), self.REGION_TAG_UPPER_SKY, dtype=np.int8)
        level = np.full((n,), len(self.voxel_size_lis) - 1, dtype=np.int8)
        voxel = np.full((n,), self.sky_init_scale, dtype=np.float32)
        return pts_world.astype(np.float32), rgb.astype(np.float32), region, level, voxel

    def _empty_voxel_candidates(self) -> VoxelCandidateBatch:
        return VoxelCandidateBatch(
            xyz=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            region_tag=np.zeros((0,), dtype=np.int8),
            level=np.zeros((0,), dtype=np.int8),
            voxel_size=np.zeros((0,), dtype=np.float32),
            grid_coord=np.zeros((0, 3), dtype=np.int32),
            insert_enabled=np.zeros((0,), dtype=bool),
        )

    @staticmethod
    def _neighbor_offsets(grid_radius: int) -> list[tuple[int, int, int]]:
        radius = max(0, int(grid_radius))
        return [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        ]

    def _lookup_existing_anchor_tolerant(
        self,
        anchor_hash_index,
        *,
        candidate_xyz: np.ndarray,
        candidate_voxel_size: float,
        candidate_level: int,
        anchor_xyz_cpu: np.ndarray,
        anchor_voxel_cpu: np.ndarray,
        neighbor_offsets: list[tuple[int, int, int]],
    ) -> int | None:
        if anchor_hash_index is None or not self.anchor_tolerant_match:
            return None
        if anchor_xyz_cpu.shape[0] == 0:
            return None

        n_levels = len(anchor_hash_index.structure_hash)
        if n_levels <= 0:
            return None

        level_min = max(0, int(candidate_level) - self.anchor_match_level_radius)
        level_max = min(n_levels - 1, int(candidate_level) + self.anchor_match_level_radius)
        best_row = None
        best_score = None
        seen_rows: set[int] = set()

        for level_id in range(level_min, level_max + 1):
            table = anchor_hash_index.structure_hash[level_id]
            query_voxel = float(self.voxel_size_lis[level_id])
            if query_voxel <= 1e-6 or not np.isfinite(query_voxel):
                query_voxel = max(float(candidate_voxel_size), 1e-6)
            base_key = np.floor(candidate_xyz / query_voxel).astype(np.int32)

            for dx, dy, dz in neighbor_offsets:
                row_idx = table.get(
                    (int(base_key[0] + dx), int(base_key[1] + dy), int(base_key[2] + dz))
                )
                if row_idx is None or row_idx in seen_rows or row_idx >= anchor_xyz_cpu.shape[0]:
                    continue
                seen_rows.add(int(row_idx))

                anchor_voxel = float(anchor_voxel_cpu[row_idx])
                if anchor_voxel <= 1e-6 or not np.isfinite(anchor_voxel):
                    anchor_voxel = query_voxel
                dist_thresh = self.anchor_match_dist_factor * max(
                    float(candidate_voxel_size), anchor_voxel
                )
                if dist_thresh <= 1e-6:
                    continue

                dist = float(np.linalg.norm(anchor_xyz_cpu[row_idx] - candidate_xyz))
                if dist > dist_thresh:
                    continue

                score = dist / dist_thresh
                if best_score is None or score < best_score:
                    best_score = score
                    best_row = int(row_idx)

        return best_row

    def build_voxel_candidates_from_erp_depth(
        self,
        cam,
        depthmap: np.ndarray,
        pixel_mask: Optional[np.ndarray] = None,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ) -> VoxelCandidateBatch:
        del init, anchor_submap, birth_frame
        H, W = depthmap.shape
        dx, dy, dz = erp_dense_pixel_center_bearings(H, W)
        training_cfg = self.config.get("Training", {}) if self.config else {}
        depth_valid_max = float(
            training_cfg.get(
                "dap_depth_max_valid",
                training_cfg.get("ransac", {}).get("depth_max", 120.0),
            )
        )
        sky_mask = getattr(cam, "erp_sky_mask", None)
        if sky_mask is not None and tuple(sky_mask.shape) != tuple(depthmap.shape):
            sky_mask = None

        depth_valid = np.isfinite(depthmap) & (depthmap > 0.01) & (depthmap < depth_valid_max)
        if sky_mask is not None:
            depth_valid = depth_valid & (~np.asarray(sky_mask, dtype=bool))
        if not np.any(depth_valid):
            return self._empty_voxel_candidates()

        pixel_insert_valid = None
        if pixel_mask is not None:
            pixel_mask_np = np.asarray(pixel_mask, dtype=bool)
            if tuple(pixel_mask_np.shape) == tuple(depthmap.shape):
                pixel_insert_valid = pixel_mask_np.reshape(-1)

        pts_cam = np.stack([dx * depthmap, dy * depthmap, dz * depthmap], axis=-1)
        flat_valid = depth_valid.reshape(-1)
        pts_cam_valid = pts_cam.reshape(-1, 3)[flat_valid]

        R_np = cam.R.float().cpu().numpy().astype(np.float64)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)
        pts_world = (R_np.T @ pts_cam_valid.T).T - (R_np.T @ T_np)

        image_ab = cam.original_image.clamp(0.0, 1.0)
        rgb_np = image_ab.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        rgb_valid = rgb_np.reshape(-1, 3)[flat_valid]
        region_valid = np.full((pts_world.shape[0],), self.REGION_TAG_DEFAULT, dtype=np.int8)
        confidence_valid = np.ones((pts_world.shape[0],), dtype=np.float32)
        if pixel_insert_valid is None:
            insert_valid = np.ones((pts_world.shape[0],), dtype=bool)
        else:
            insert_valid = pixel_insert_valid[flat_valid].astype(bool, copy=False)
        depth_valid_values = depthmap.reshape(-1)[flat_valid]
        level_valid = self._depth_to_level(depth_valid_values)

        xyz_chunks = []
        rgb_chunks = []
        conf_chunks = []
        region_chunks = []
        level_chunks = []
        voxel_chunks = []
        grid_chunks = []
        insert_chunks = []
        for level_id, voxel_size in enumerate(self.voxel_size_lis):
            mask = level_valid == level_id
            if not np.any(mask):
                continue
            pts_l = pts_world[mask]
            rgb_l = rgb_valid[mask]
            conf_l = confidence_valid[mask]
            region_l = region_valid[mask]
            insert_l = insert_valid[mask]
            grid_l = np.floor(pts_l / max(voxel_size, 1e-6)).astype(np.int32)
            _, keep_idx, inverse = np.unique(
                grid_l, axis=0, return_index=True, return_inverse=True
            )
            order = np.argsort(keep_idx)
            keep_idx = keep_idx[order]
            insert_unique = np.zeros((inverse.max() + 1,), dtype=np.uint8)
            np.maximum.at(insert_unique, inverse, insert_l.astype(np.uint8))
            insert_unique = insert_unique.astype(bool)[order]
            grid_keep = grid_l[keep_idx]
            xyz_chunks.append(((grid_keep.astype(np.float32) + 0.5) * voxel_size).astype(np.float32))
            rgb_chunks.append(rgb_l[keep_idx].astype(np.float32))
            conf_chunks.append(conf_l[keep_idx].astype(np.float32))
            region_chunks.append(region_l[keep_idx].astype(np.int8))
            level_chunks.append(np.full((keep_idx.shape[0],), level_id, dtype=np.int8))
            voxel_chunks.append(np.full((keep_idx.shape[0],), voxel_size, dtype=np.float32))
            grid_chunks.append(grid_keep.astype(np.int32))
            insert_chunks.append(insert_unique)

        if not xyz_chunks:
            return self._empty_voxel_candidates()

        return VoxelCandidateBatch(
            xyz=np.concatenate(xyz_chunks, axis=0).astype(np.float32),
            rgb=np.concatenate(rgb_chunks, axis=0).astype(np.float32),
            confidence=np.concatenate(conf_chunks, axis=0).astype(np.float32),
            region_tag=np.concatenate(region_chunks, axis=0).astype(np.int8),
            level=np.concatenate(level_chunks, axis=0).astype(np.int8),
            voxel_size=np.concatenate(voxel_chunks, axis=0).astype(np.float32),
            grid_coord=np.concatenate(grid_chunks, axis=0).astype(np.int32),
            insert_enabled=np.concatenate(insert_chunks, axis=0).astype(bool),
        )

    def build_voxel_candidates_from_global_world_points(
        self,
        cam,
        depthmap: Optional[np.ndarray] = None,
        pixel_mask: Optional[np.ndarray] = None,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ) -> VoxelCandidateBatch:
        del init, anchor_submap, birth_frame
        expected_shape = None if depthmap is None else depthmap.shape
        payload = self._world_points_payload_from_viewpoint(cam, expected_shape=expected_shape)
        if payload is None:
            raise ValueError("global_world_points are required for world-points anchor insertion.")
        flat_valid = payload["valid"].reshape(-1)
        if not np.any(flat_valid):
            raise ValueError(f"Keyframe {cam.uid} has no valid global world points.")

        pixel_insert_valid = None
        if pixel_mask is not None:
            pixel_mask_np = np.asarray(pixel_mask, dtype=bool)
            if tuple(pixel_mask_np.shape) == tuple(payload["valid"].shape):
                pixel_insert_valid = pixel_mask_np.reshape(-1)

        pts_world = payload["points"].reshape(-1, 3)[flat_valid]
        rgb_valid = payload["rgb"].reshape(-1, 3)[flat_valid]
        confidence_valid = payload["confidence"].reshape(-1)[flat_valid]
        region_valid = payload["region_tag"].reshape(-1)[flat_valid]
        depth_valid_values = payload["distance"].reshape(-1)[flat_valid]
        level_valid = self._depth_to_level(depth_valid_values)
        if pixel_insert_valid is None:
            insert_valid = np.ones((pts_world.shape[0],), dtype=bool)
        else:
            insert_valid = pixel_insert_valid[flat_valid].astype(bool, copy=False)

        xyz_chunks = []
        rgb_chunks = []
        conf_chunks = []
        region_chunks = []
        level_chunks = []
        voxel_chunks = []
        grid_chunks = []
        insert_chunks = []
        for level_id, voxel_size in enumerate(self.voxel_size_lis):
            mask = level_valid == level_id
            if not np.any(mask):
                continue
            pts_l = pts_world[mask]
            rgb_l = rgb_valid[mask]
            conf_l = confidence_valid[mask]
            region_l = region_valid[mask]
            insert_l = insert_valid[mask]
            grid_l = np.floor(pts_l / max(voxel_size, 1e-6)).astype(np.int32)
            _, keep_idx, inverse = np.unique(
                grid_l, axis=0, return_index=True, return_inverse=True
            )
            order = np.argsort(keep_idx)
            keep_idx = keep_idx[order]
            insert_unique = np.zeros((inverse.max() + 1,), dtype=np.uint8)
            np.maximum.at(insert_unique, inverse, insert_l.astype(np.uint8))
            insert_unique = insert_unique.astype(bool)[order]
            xyz_chunks.append(pts_l[keep_idx].astype(np.float32))
            rgb_chunks.append(rgb_l[keep_idx].astype(np.float32))
            conf_chunks.append(conf_l[keep_idx].astype(np.float32))
            region_chunks.append(region_l[keep_idx].astype(np.int8))
            level_chunks.append(np.full((keep_idx.shape[0],), level_id, dtype=np.int8))
            voxel_chunks.append(np.full((keep_idx.shape[0],), voxel_size, dtype=np.float32))
            grid_chunks.append(grid_l[keep_idx].astype(np.int32))
            insert_chunks.append(insert_unique)

        if not xyz_chunks:
            return self._empty_voxel_candidates()

        return VoxelCandidateBatch(
            xyz=np.concatenate(xyz_chunks, axis=0).astype(np.float32),
            rgb=np.concatenate(rgb_chunks, axis=0).astype(np.float32),
            confidence=np.concatenate(conf_chunks, axis=0).astype(np.float32),
            region_tag=np.concatenate(region_chunks, axis=0).astype(np.int8),
            level=np.concatenate(level_chunks, axis=0).astype(np.int8),
            voxel_size=np.concatenate(voxel_chunks, axis=0).astype(np.float32),
            grid_coord=np.concatenate(grid_chunks, axis=0).astype(np.int32),
            insert_enabled=np.concatenate(insert_chunks, axis=0).astype(bool),
        )

    def _prepare_anchor_tensors(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        voxel_size: np.ndarray,
        is_sky_anchor: np.ndarray,
    ):
        if xyz.shape[0] == 0:
            return None
        fused_point_cloud = torch.from_numpy(xyz).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(rgb).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device="cuda",
        )
        features[:, :3, 0] = fused_color
        if self.max_sh_degree > 0:
            features[:, :, 1:] = 0.0

        scale_values = np.repeat((voxel_size * 0.8)[:, None], 3, axis=1).astype(np.float32)
        if is_sky_anchor.any():
            scale_values[is_sky_anchor] = float(self.sky_init_scale)
        scales = torch.log(
            torch.from_numpy(scale_values).float().cuda().clamp(min=1e-4)
        )
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0
        opacity_init = np.full((fused_point_cloud.shape[0], 1), 0.08, dtype=np.float32)
        opacity_init[is_sky_anchor, 0] = self.sky_opacity_init
        opacities = inverse_sigmoid(torch.from_numpy(opacity_init).float().cuda())
        return fused_point_cloud, features, scales, rots, opacities

    def _append_anchor_batch(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        level: np.ndarray,
        voxel_size: np.ndarray,
        grid_coord: np.ndarray,
        *,
        frame_idx: int,
        anchor_submap: int,
        birth_frame: int,
        region_tag: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        is_sky_anchor: Optional[np.ndarray] = None,
        obs_count: Optional[np.ndarray] = None,
        last_seen_kf: Optional[np.ndarray] = None,
        conf_accum: Optional[np.ndarray] = None,
    ) -> tuple[int, int]:
        n_total = int(xyz.shape[0])
        if n_total <= 0:
            return int(self.get_xyz.shape[0]), 0

        if region_tag is None:
            region_tag = np.full((n_total,), self.REGION_TAG_DEFAULT, dtype=np.int8)
        if confidence is None:
            confidence = np.ones((n_total,), dtype=np.float32)
        if is_sky_anchor is None:
            is_sky_anchor = np.zeros((n_total,), dtype=bool)
        if obs_count is None:
            obs_count = np.ones((n_total,), dtype=np.int32)
        if last_seen_kf is None:
            last_seen_kf = np.full((n_total,), int(frame_idx), dtype=np.int32)
        if conf_accum is None:
            conf_accum = confidence.astype(np.float32, copy=True)

        tensors = self._prepare_anchor_tensors(
            xyz=xyz,
            rgb=rgb,
            voxel_size=voxel_size,
            is_sky_anchor=is_sky_anchor,
        )
        if tensors is None:
            return int(self.get_xyz.shape[0]), 0

        self._n_sky_pending = int(is_sky_anchor.sum())
        self._layer_pending = torch.from_numpy(level.copy())
        self._confidence_pending = torch.from_numpy(confidence.copy())
        self._region_tag_pending = torch.from_numpy(region_tag.copy())
        self._anchor_kf_pending = torch.full((n_total,), int(frame_idx), dtype=torch.int32)
        self._anchor_submap_pending = torch.full((n_total,), int(anchor_submap), dtype=torch.int32)
        self._birth_frame_pending = torch.full((n_total,), int(birth_frame), dtype=torch.int32)
        self._anchor_level_pending = torch.from_numpy(level.copy())
        self._anchor_voxel_pending = torch.from_numpy(voxel_size.copy())
        self._is_sky_anchor_pending = torch.from_numpy(is_sky_anchor.copy())
        self._anchor_grid_coord_pending = torch.from_numpy(grid_coord.copy())
        self._anchor_obs_count_pending = torch.from_numpy(obs_count.copy())
        self._anchor_last_seen_kf_pending = torch.from_numpy(last_seen_kf.copy())
        self._anchor_conf_accum_pending = torch.from_numpy(conf_accum.copy())

        before_n = int(self.get_xyz.shape[0])
        self.extend_from_pcd(*tensors, kf_id=int(frame_idx))
        return before_n, int(self.get_xyz.shape[0] - before_n)

    def extend_or_merge_from_voxel_candidates(
        self,
        candidates: VoxelCandidateBatch,
        anchor_hash_index,
        *,
        cam,
        frame_idx: int,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ) -> dict:
        if birth_frame is None:
            birth_frame = int(frame_idx)
        n_anchor_before = int(self.get_xyz.shape[0])
        if anchor_hash_index is not None:
            anchor_hash_index.ensure_num_levels(len(self.voxel_size_lis))

        exact_hit_rows: list[int] = []
        exact_hit_candidate_idx: list[int] = []
        provisional_miss_indices: list[int] = []
        for idx in range(len(candidates)):
            key = tuple(int(v) for v in candidates.grid_coord[idx].tolist())
            row_idx = None if anchor_hash_index is None else anchor_hash_index.lookup(
                int(candidates.level[idx]),
                key,
            )
            if row_idx is None:
                provisional_miss_indices.append(idx)
            else:
                exact_hit_candidate_idx.append(idx)
                exact_hit_rows.append(int(row_idx))

        near_hit_rows: list[int] = []
        near_hit_candidate_idx: list[int] = []
        miss_indices: list[int] = []
        if (
            provisional_miss_indices
            and anchor_hash_index is not None
            and self.anchor_tolerant_match
            and int(self.get_xyz.shape[0]) > 0
        ):
            anchor_xyz_cpu = self.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
            if self._anchor_voxel_size.shape[0] == self.get_xyz.shape[0]:
                anchor_voxel_cpu = (
                    self._anchor_voxel_size.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            else:
                anchor_voxel_cpu = np.full(
                    (self.get_xyz.shape[0],),
                    float(self.voxel_size_lis[0]),
                    dtype=np.float32,
                )
            neighbor_offsets = self._neighbor_offsets(self.anchor_match_grid_radius)

            for idx in provisional_miss_indices:
                row_idx = self._lookup_existing_anchor_tolerant(
                    anchor_hash_index,
                    candidate_xyz=candidates.xyz[idx],
                    candidate_voxel_size=float(candidates.voxel_size[idx]),
                    candidate_level=int(candidates.level[idx]),
                    anchor_xyz_cpu=anchor_xyz_cpu,
                    anchor_voxel_cpu=anchor_voxel_cpu,
                    neighbor_offsets=neighbor_offsets,
                )
                if row_idx is None:
                    miss_indices.append(idx)
                else:
                    near_hit_candidate_idx.append(idx)
                    near_hit_rows.append(int(row_idx))
        else:
            miss_indices = provisional_miss_indices

        hit_rows = exact_hit_rows + near_hit_rows
        hit_candidate_idx = exact_hit_candidate_idx + near_hit_candidate_idx
        thaw_rows = [
            int(row)
            for row, cand_idx in zip(hit_rows, hit_candidate_idx)
            if bool(candidates.insert_enabled[int(cand_idx)])
        ]

        if hit_rows:
            hit_rows_t = torch.as_tensor(hit_rows, dtype=torch.long)
            if self._anchor_obs_count.shape[0] == self.get_xyz.shape[0]:
                self._anchor_obs_count[hit_rows_t] += 1
            if self._anchor_last_seen_kf.shape[0] == self.get_xyz.shape[0]:
                self._anchor_last_seen_kf[hit_rows_t] = int(frame_idx)
            if self._anchor_conf_accum.shape[0] == self.get_xyz.shape[0]:
                conf_delta = torch.from_numpy(
                    candidates.confidence[np.asarray(hit_candidate_idx, dtype=np.int64)]
                ).to(dtype=torch.float32)
                self._anchor_conf_accum[hit_rows_t] += conf_delta

        n_new_structure = 0
        new_structure_start = -1
        suppressed_miss_count = 0
        create_indices = [
            idx for idx in miss_indices if bool(candidates.insert_enabled[idx])
        ]
        suppressed_miss_count = int(len(miss_indices) - len(create_indices))
        if create_indices:
            new_candidates = candidates.select(create_indices)
            before_n, n_new_structure = self._append_anchor_batch(
                xyz=new_candidates.xyz,
                rgb=new_candidates.rgb,
                level=new_candidates.level,
                voxel_size=new_candidates.voxel_size,
                grid_coord=new_candidates.grid_coord,
                frame_idx=int(frame_idx),
                anchor_submap=int(anchor_submap),
                birth_frame=int(birth_frame),
                region_tag=new_candidates.region_tag,
                confidence=new_candidates.confidence,
                is_sky_anchor=np.zeros((len(new_candidates),), dtype=bool),
                obs_count=np.ones((len(new_candidates),), dtype=np.int32),
                last_seen_kf=np.full((len(new_candidates),), int(frame_idx), dtype=np.int32),
                conf_accum=new_candidates.confidence.astype(np.float32, copy=True),
            )
            new_structure_start = int(before_n)
            if anchor_hash_index is not None and n_new_structure > 0:
                for offset, idx in enumerate(create_indices):
                    key = tuple(int(v) for v in candidates.grid_coord[idx].tolist())
                    anchor_hash_index.insert(int(candidates.level[idx]), key, before_n + offset)

        n_new_sky = 0
        if self.sky_enabled and init and not self._sky_box_initialized:
            sky_pts, sky_rgb, sky_region, sky_level, sky_voxel = self._build_single_hemisphere_skybox(cam)
            sky_grid = np.zeros((sky_pts.shape[0], 3), dtype=np.int32)
            _, n_new_sky = self._append_anchor_batch(
                xyz=sky_pts,
                rgb=sky_rgb,
                level=sky_level,
                voxel_size=sky_voxel,
                grid_coord=sky_grid,
                frame_idx=int(frame_idx),
                anchor_submap=int(anchor_submap),
                birth_frame=int(birth_frame),
                region_tag=sky_region,
                confidence=np.ones((sky_pts.shape[0],), dtype=np.float32),
                is_sky_anchor=np.ones((sky_pts.shape[0],), dtype=bool),
                obs_count=np.ones((sky_pts.shape[0],), dtype=np.int32),
                last_seen_kf=np.full((sky_pts.shape[0],), int(frame_idx), dtype=np.int32),
                conf_accum=np.ones((sky_pts.shape[0],), dtype=np.float32),
            )
            self._sky_box_initialized = True

        local_stats = self.set_local_anchor_active_sets(
            thaw_rows=thaw_rows,
            new_start=new_structure_start,
            new_count=n_new_structure,
            current_kf_id=int(frame_idx),
        )

        self._last_growth_stats = {
            "n_new_anchors_added": int(n_new_structure + n_new_sky),
            "n_anchors_pruned": 0,
        }
        self._last_hash_stats.update(
            {
                "n_structure_hash_hits": int(len(exact_hit_rows)),
                "n_structure_hash_near_hits": int(len(near_hit_rows)),
                "n_structure_hash_misses": int(len(miss_indices)),
                "n_structure_hash_new": int(n_new_structure),
                "n_structure_hash_merged": int(len(hit_rows)),
                "n_hash_hit_thaw": int(len(thaw_rows)),
                "n_hash_miss_new": int(n_new_structure),
                "n_structure_vcd_suppressed": int(suppressed_miss_count),
                "n_hash_rebuild_collisions": 0,
                "n_duplicate_pruned": 0,
                "n_structure_obs_updates": int(len(hit_rows)),
                **local_stats,
            }
        )
        return {
            "n_anchor_before": int(n_anchor_before),
            "n_voxel_candidates": int(len(candidates)),
            "n_insert_enabled_candidates": int(candidates.insert_enabled.sum()),
            "n_structure_hash_hits": int(len(exact_hit_rows)),
            "n_structure_hash_near_hits": int(len(near_hit_rows)),
            "n_structure_hash_misses": int(len(miss_indices)),
            "n_structure_hash_new": int(n_new_structure),
            "new_structure_start": int(new_structure_start),
            "new_structure_count": int(n_new_structure),
            "local_thaw_rows": [int(v) for v in thaw_rows],
            "n_hash_hit_thaw": int(len(thaw_rows)),
            "n_hash_miss_new": int(n_new_structure),
            "n_structure_vcd_suppressed": int(suppressed_miss_count),
            "n_sky_hash_new": int(n_new_sky),
            **local_stats,
        }

    def _create_pcd_from_erp_depth(
        self,
        cam,
        depthmap: np.ndarray,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ):
        if self._world_points_payload_from_viewpoint(cam, expected_shape=depthmap.shape) is not None:
            return self._create_pcd_from_global_world_points(
                cam,
                depthmap=depthmap,
                init=init,
                anchor_submap=anchor_submap,
                birth_frame=birth_frame,
            )
        H, W = depthmap.shape
        dx, dy, dz = erp_dense_pixel_center_bearings(H, W)
        training_cfg = self.config.get("Training", {}) if self.config else {}
        depth_valid_max = float(
            training_cfg.get(
                "dap_depth_max_valid",
                training_cfg.get("ransac", {}).get("depth_max", 120.0),
            )
        )
        sky_mask = getattr(cam, "erp_sky_mask", None)
        if sky_mask is not None and tuple(sky_mask.shape) != tuple(depthmap.shape):
            sky_mask = None

        depth_valid = np.isfinite(depthmap) & (depthmap > 0.01) & (depthmap < depth_valid_max)
        if sky_mask is not None:
            depth_valid = depth_valid & (~np.asarray(sky_mask, dtype=bool))

        pts_cam = np.stack([dx * depthmap, dy * depthmap, dz * depthmap], axis=-1)
        flat_valid = depth_valid.reshape(-1)
        pts_cam_valid = pts_cam.reshape(-1, 3)[flat_valid]

        R_np = cam.R.float().cpu().numpy().astype(np.float64)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)
        pts_world = (R_np.T @ pts_cam_valid.T).T - (R_np.T @ T_np)

        image_ab = cam.original_image.clamp(0.0, 1.0)
        rgb_np = image_ab.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        rgb_valid = rgb_np.reshape(-1, 3)[flat_valid]

        region_valid = np.full((pts_world.shape[0],), self.REGION_TAG_DEFAULT, dtype=np.int8)
        depth_valid_values = depthmap.reshape(-1)[flat_valid]
        level_valid = self._depth_to_level(depth_valid_values)

        pts_world, rgb_valid, region_valid, level_valid, voxel_valid = self._voxelize_level(
            pts_world, rgb_valid, region_valid, level_valid
        )

        if pts_world.shape[0] == 0:
            pts_world = np.zeros((1, 3), dtype=np.float32)
            rgb_valid = np.zeros((1, 3), dtype=np.float32)
            region_valid = np.zeros((1,), dtype=np.int8)
            level_valid = np.zeros((1,), dtype=np.int8)
            voxel_valid = np.full((1,), self.voxel_size_lis[0], dtype=np.float32)

        is_sky_anchor = np.zeros((pts_world.shape[0],), dtype=bool)

        if self.sky_enabled and init and not self._sky_box_initialized:
            sky_pts, sky_rgb, sky_region, sky_level, sky_voxel = self._build_single_hemisphere_skybox(cam)
            pts_world = np.concatenate([pts_world, sky_pts], axis=0)
            rgb_valid = np.concatenate([rgb_valid, sky_rgb], axis=0)
            region_valid = np.concatenate([region_valid, sky_region], axis=0)
            level_valid = np.concatenate([level_valid, sky_level], axis=0)
            voxel_valid = np.concatenate([voxel_valid, sky_voxel], axis=0)
            is_sky_anchor = np.concatenate(
                [is_sky_anchor, np.ones((sky_pts.shape[0],), dtype=bool)], axis=0
            )
            self._sky_box_initialized = True

        fused_point_cloud = torch.from_numpy(pts_world).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(rgb_valid).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device="cuda",
        )
        features[:, :3, 0] = fused_color
        if self.max_sh_degree > 0:
            features[:, :, 1:] = 0.0

        scales = torch.log(
            torch.from_numpy(np.repeat((voxel_valid * 0.8)[:, None], 3, axis=1)).float().cuda().clamp(min=1e-4)
        )
        if is_sky_anchor.any():
            sky_mask_t = torch.from_numpy(is_sky_anchor).to(device=scales.device, dtype=torch.bool)
            scales[sky_mask_t] = np.log(max(self.sky_init_scale, 1e-4))

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0
        opacity_init = np.full((fused_point_cloud.shape[0], 1), 0.08, dtype=np.float32)
        opacity_init[is_sky_anchor, 0] = self.sky_opacity_init
        opacities = inverse_sigmoid(torch.from_numpy(opacity_init).float().cuda())

        self._n_sky_pending = int(is_sky_anchor.sum())
        self._layer_pending = torch.from_numpy(level_valid.copy())
        self._region_tag_pending = torch.from_numpy(region_valid.copy())
        if birth_frame is None:
            birth_frame = int(cam.uid)
        n_total = pts_world.shape[0]
        self._anchor_kf_pending = torch.full((n_total,), int(cam.uid), dtype=torch.int32)
        self._anchor_submap_pending = torch.full((n_total,), int(anchor_submap), dtype=torch.int32)
        self._birth_frame_pending = torch.full((n_total,), int(birth_frame), dtype=torch.int32)
        self._anchor_level_pending = torch.from_numpy(level_valid.copy())
        self._anchor_voxel_pending = torch.from_numpy(voxel_valid.copy())
        self._is_sky_anchor_pending = torch.from_numpy(is_sky_anchor.copy())
        grid_coord = np.floor(pts_world / np.maximum(voxel_valid[:, None], 1e-6)).astype(np.int32)
        self._anchor_grid_coord_pending = torch.from_numpy(grid_coord)
        self._anchor_obs_count_pending = torch.ones((n_total,), dtype=torch.int32)
        self._anchor_last_seen_kf_pending = torch.full((n_total,), int(cam.uid), dtype=torch.int32)
        self._anchor_conf_accum_pending = torch.ones((n_total,), dtype=torch.float32)
        self._last_growth_stats = {"n_new_anchors_added": int(n_total), "n_anchors_pruned": 0}
        return fused_point_cloud, features, scales, rots, opacities

    def _create_pcd_from_global_world_points(
        self,
        cam,
        depthmap=None,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ):
        candidates = self.build_voxel_candidates_from_global_world_points(
            cam,
            depthmap=depthmap,
            pixel_mask=None,
            init=init,
            anchor_submap=anchor_submap,
            birth_frame=birth_frame,
        )
        if len(candidates) <= 0:
            raise ValueError(f"Keyframe {cam.uid} has no valid global world point candidates.")

        is_sky_anchor = np.zeros((len(candidates),), dtype=bool)
        tensors = self._prepare_anchor_tensors(
            xyz=candidates.xyz,
            rgb=candidates.rgb,
            voxel_size=candidates.voxel_size,
            is_sky_anchor=is_sky_anchor,
        )
        if tensors is None:
            raise ValueError(f"Keyframe {cam.uid} produced empty global world point tensors.")

        if birth_frame is None:
            birth_frame = int(cam.uid)
        n_total = len(candidates)
        self._n_sky_pending = 0
        self._layer_pending = torch.from_numpy(candidates.level.copy())
        self._confidence_pending = torch.from_numpy(candidates.confidence.copy())
        self._region_tag_pending = torch.from_numpy(candidates.region_tag.copy())
        self._anchor_kf_pending = torch.full((n_total,), int(cam.uid), dtype=torch.int32)
        self._anchor_submap_pending = torch.full((n_total,), int(anchor_submap), dtype=torch.int32)
        self._birth_frame_pending = torch.full((n_total,), int(birth_frame), dtype=torch.int32)
        self._anchor_level_pending = torch.from_numpy(candidates.level.copy())
        self._anchor_voxel_pending = torch.from_numpy(candidates.voxel_size.copy())
        self._is_sky_anchor_pending = torch.from_numpy(is_sky_anchor.copy())
        self._anchor_grid_coord_pending = torch.from_numpy(candidates.grid_coord.copy())
        self._anchor_obs_count_pending = torch.ones((n_total,), dtype=torch.int32)
        self._anchor_last_seen_kf_pending = torch.full((n_total,), int(cam.uid), dtype=torch.int32)
        self._anchor_conf_accum_pending = torch.from_numpy(candidates.confidence.copy())
        self._last_growth_stats = {"n_new_anchors_added": int(n_total), "n_anchors_pruned": 0}
        return tensors

    def extend_from_pcd_seq(
        self,
        cam_info,
        kf_id=-1,
        init=False,
        scale=2.0,
        depthmap=None,
        pixel_mask=None,
        anchor_submap=-1,
        birth_frame=None,
        anchor_hash_index=None,
    ):
        if self._world_points_payload_from_viewpoint(
            cam_info,
            expected_shape=None if depthmap is None else depthmap.shape,
        ) is not None:
            candidates = self.build_voxel_candidates_from_global_world_points(
                cam_info,
                depthmap=depthmap,
                pixel_mask=pixel_mask,
                init=init,
                anchor_submap=anchor_submap,
                birth_frame=birth_frame,
            )
            return self.extend_or_merge_from_voxel_candidates(
                candidates,
                anchor_hash_index,
                cam=cam_info,
                frame_idx=int(kf_id if kf_id is not None else cam_info.uid),
                init=init,
                anchor_submap=anchor_submap,
                birth_frame=birth_frame,
            )
        if depthmap is None:
            return super().extend_from_pcd_seq(
                cam_info,
                kf_id=kf_id,
                init=init,
                scale=scale,
                depthmap=depthmap,
                anchor_submap=anchor_submap,
                birth_frame=birth_frame,
            )

        candidates = self.build_voxel_candidates_from_erp_depth(
            cam_info,
            depthmap,
            pixel_mask=pixel_mask,
            init=init,
            anchor_submap=anchor_submap,
            birth_frame=birth_frame,
        )
        return self.extend_or_merge_from_voxel_candidates(
            candidates,
            anchor_hash_index,
            cam=cam_info,
            frame_idx=int(kf_id if kf_id is not None else cam_info.uid),
            init=init,
            anchor_submap=anchor_submap,
            birth_frame=birth_frame,
        )

    def extend_from_pcd(
        self,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        kf_id,
        anchor_kf=None,
        anchor_submap=None,
        birth_frame=None,
    ):
        before_n = int(self.get_xyz.shape[0])
        super().extend_from_pcd(
            fused_point_cloud,
            features,
            scales,
            rots,
            opacities,
            kf_id,
            anchor_kf=anchor_kf,
            anchor_submap=anchor_submap,
            birth_frame=birth_frame,
        )
        n_new = int(self.get_xyz.shape[0] - before_n)
        if n_new <= 0:
            return

        level_pending = self._anchor_level_pending
        self._anchor_level_pending = None
        if level_pending is None:
            level_pending = torch.zeros((n_new,), dtype=torch.int8)
        voxel_pending = self._anchor_voxel_pending
        self._anchor_voxel_pending = None
        if voxel_pending is None:
            voxel_pending = torch.full((n_new,), self.voxel_size_lis[0], dtype=torch.float32)
        sky_pending = self._is_sky_anchor_pending
        self._is_sky_anchor_pending = None
        if sky_pending is None:
            sky_pending = torch.zeros((n_new,), dtype=torch.bool)
        grid_pending = self._anchor_grid_coord_pending
        self._anchor_grid_coord_pending = None
        if grid_pending is None:
            grid_pending = torch.zeros((n_new, 3), dtype=torch.int32)
        obs_count_pending = self._anchor_obs_count_pending
        self._anchor_obs_count_pending = None
        if obs_count_pending is None:
            obs_count_pending = torch.ones((n_new,), dtype=torch.int32)
        last_seen_pending = self._anchor_last_seen_kf_pending
        self._anchor_last_seen_kf_pending = None
        if last_seen_pending is None:
            last_seen_pending = torch.full((n_new,), -1, dtype=torch.int32)
        conf_accum_pending = self._anchor_conf_accum_pending
        self._anchor_conf_accum_pending = None
        if conf_accum_pending is None:
            conf_accum_pending = torch.zeros((n_new,), dtype=torch.float32)

        self._anchor_level = torch.cat([self._anchor_level, level_pending.to(dtype=torch.int8)])
        self._anchor_voxel_size = torch.cat(
            [self._anchor_voxel_size, voxel_pending.to(dtype=torch.float32)]
        )
        self._is_sky_anchor = torch.cat(
            [self._is_sky_anchor, sky_pending.to(dtype=torch.bool)]
        )
        self._anchor_grid_coord = torch.cat(
            [self._anchor_grid_coord, grid_pending.to(dtype=torch.int32)]
        )
        self._anchor_obs_count = torch.cat(
            [self._anchor_obs_count, obs_count_pending.to(dtype=torch.int32)]
        )
        self._anchor_last_seen_kf = torch.cat(
            [self._anchor_last_seen_kf, last_seen_pending.to(dtype=torch.int32)]
        )
        self._anchor_conf_accum = torch.cat(
            [self._anchor_conf_accum, conf_accum_pending.to(dtype=torch.float32)]
        )
        self._anchor_replacement_obs = torch.cat(
            [self._anchor_replacement_obs, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_inconsistent_hits = torch.cat(
            [self._anchor_inconsistent_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_depth_inconsistent_hits = torch.cat(
            [self._anchor_depth_inconsistent_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_strong_prune_hits = torch.cat(
            [self._anchor_strong_prune_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_reset_evidence_hits = torch.cat(
            [self._anchor_reset_evidence_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_sky_floater_hits = torch.cat(
            [self._anchor_sky_floater_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_replacement_hits = torch.cat(
            [self._anchor_replacement_hits, torch.zeros((n_new,), dtype=torch.int32)]
        )
        self._anchor_replacement_supported = torch.cat(
            [self._anchor_replacement_supported, torch.zeros((n_new,), dtype=torch.bool)]
        )

    def prune_points(self, mask):
        valid_points_mask = ~mask
        super().prune_points(mask)
        keep_mask_cpu = valid_points_mask.detach().cpu()
        if self._anchor_level.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_level = self._anchor_level[keep_mask_cpu]
        if self._anchor_voxel_size.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_voxel_size = self._anchor_voxel_size[keep_mask_cpu]
        if self._is_sky_anchor.shape[0] == keep_mask_cpu.shape[0]:
            self._is_sky_anchor = self._is_sky_anchor[keep_mask_cpu]
        if self._anchor_grid_coord.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_grid_coord = self._anchor_grid_coord[keep_mask_cpu]
        if self._anchor_obs_count.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_obs_count = self._anchor_obs_count[keep_mask_cpu]
        if self._anchor_last_seen_kf.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_last_seen_kf = self._anchor_last_seen_kf[keep_mask_cpu]
        if self._anchor_conf_accum.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_conf_accum = self._anchor_conf_accum[keep_mask_cpu]
        if self._anchor_replacement_obs.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_replacement_obs = self._anchor_replacement_obs[keep_mask_cpu]
        if self._anchor_inconsistent_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_inconsistent_hits = self._anchor_inconsistent_hits[keep_mask_cpu]
        if self._anchor_depth_inconsistent_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_depth_inconsistent_hits = self._anchor_depth_inconsistent_hits[keep_mask_cpu]
        if self._anchor_strong_prune_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_strong_prune_hits = self._anchor_strong_prune_hits[keep_mask_cpu]
        if self._anchor_reset_evidence_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_reset_evidence_hits = self._anchor_reset_evidence_hits[keep_mask_cpu]
        if self._anchor_sky_floater_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_sky_floater_hits = self._anchor_sky_floater_hits[keep_mask_cpu]
        if self._anchor_replacement_hits.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_replacement_hits = self._anchor_replacement_hits[keep_mask_cpu]
        if self._anchor_replacement_supported.shape[0] == keep_mask_cpu.shape[0]:
            self._anchor_replacement_supported = self._anchor_replacement_supported[keep_mask_cpu]
        if self._local_thaw_anchor_mask.shape[0] == keep_mask_cpu.shape[0]:
            self._local_thaw_anchor_mask = self._local_thaw_anchor_mask[keep_mask_cpu]
        if self._local_new_anchor_mask.shape[0] == keep_mask_cpu.shape[0]:
            self._local_new_anchor_mask = self._local_new_anchor_mask[keep_mask_cpu]

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        current_kf_id=None,
        importance_score=None,
        pruning_score=None,
        fastgs_enabled=False,
        fastgs_vcd_only=True,
        init_phase: bool = False,
        current_window=None,
    ):
        del max_grad, extent, max_screen_size, importance_score, init_phase
        if self.get_xyz.shape[0] == 0:
            return {"n_clone": 0, "n_split": 0, "n_pruned": 0, "fastgs_prune_candidates": 0, "fastgs_pruned": 0, "phase_ms": {}}
        opacity = self.get_opacity.squeeze(-1)
        device = opacity.device
        base_prune_mask = opacity < float(min_opacity)
        current_kf = int(current_kf_id) if current_kf_id is not None else None
        self._current_kf_id = current_kf
        protected_mask = self.get_protected_anchor_mask(
            current_window=current_window,
            current_kf_id=current_kf,
            device=device,
        )
        local_active_mask = self.get_local_active_anchor_mask(device=device)
        frozen_protected_mask = protected_mask & (~local_active_mask)
        protected_opacity_prune_blocked = int((base_prune_mask & frozen_protected_mask).sum().item())
        base_prune_mask = base_prune_mask & ~frozen_protected_mask
        dia_reset_count = 0
        dia_prune_mask = torch.zeros_like(base_prune_mask)
        dia_reset_mask = torch.zeros_like(base_prune_mask)
        dia_inconsistent_candidates = 0
        dia_replacement_candidates = 0
        dia_depth_candidates = 0
        dia_strong_candidates = 0
        dia_hard_prune_candidates = 0
        dia_sky_floater_candidates = 0
        protected_dia_reset_blocked = 0
        protected_dia_prune_blocked = 0
        training_cfg = self.config.get("Training", {}) if self.config else {}
        global_prune_enabled = bool(training_cfg.get("enable_global_anchor_prune", False))
        if bool(training_cfg.get("enable_depth_inlier_densify", False)) or global_prune_enabled:
            self._ensure_anchor_replacement_state()
            if global_prune_enabled:
                hit_thresh = max(1, int(training_cfg.get("global_prune_min_hits", 2)))
                reset_hit_thresh = max(1, int(training_cfg.get("global_reset_min_hits", hit_thresh)))
                non_sky_anchor = torch.ones_like(base_prune_mask, dtype=torch.bool)
                if self._is_sky_anchor.shape[0] == base_prune_mask.shape[0]:
                    non_sky_anchor &= ~self._is_sky_anchor.to(device=device, dtype=torch.bool)
                elif self._is_sky.shape[0] == base_prune_mask.shape[0]:
                    non_sky_anchor &= ~self._is_sky.to(device=device, dtype=torch.bool)

                inconsistent_high = self._anchor_inconsistent_hits.to(device=device) >= hit_thresh
                depth_high = self._anchor_depth_inconsistent_hits.to(device=device) >= hit_thresh
                strong_high = self._anchor_strong_prune_hits.to(device=device) >= hit_thresh
                reset_high = self._anchor_reset_evidence_hits.to(device=device) >= reset_hit_thresh
                sky_hits = self._anchor_sky_floater_hits.to(device=device)
                sky_reset_high = sky_hits >= 1
                supported = self._anchor_replacement_supported.to(device=device)

                dia_inconsistent_candidates = int(inconsistent_high.sum().item())
                dia_depth_candidates = int(depth_high.sum().item())
                dia_strong_candidates = int(strong_high.sum().item())
                dia_replacement_candidates = int((reset_high & supported).sum().item())
                dia_sky_floater_candidates = int(sky_reset_high.sum().item())

                hard_prune_high = inconsistent_high & depth_high & strong_high & supported
                dia_hard_prune_candidates = int(hard_prune_high.sum().item())
                dia_prune_mask = hard_prune_high & non_sky_anchor & (~local_active_mask)
                if not bool(training_cfg.get("global_anchor_prune_enabled", True)):
                    dia_prune_mask = torch.zeros_like(dia_prune_mask)
                dia_reset_mask = (
                    (reset_high | sky_reset_high)
                    & non_sky_anchor
                    & (~dia_prune_mask)
                    & (~local_active_mask)
                )
                if not bool(training_cfg.get("global_anchor_reset_enabled", True)):
                    dia_reset_mask = torch.zeros_like(dia_reset_mask)

                protect_min_obs = int(training_cfg.get("global_prune_protect_min_obs", 0))
                if protect_min_obs > 0 and self._anchor_obs_count.shape[0] == base_prune_mask.shape[0]:
                    high_obs = self._anchor_obs_count.to(device=device) >= protect_min_obs
                    dia_prune_mask = dia_prune_mask & ~high_obs
                    dia_reset_mask = dia_reset_mask & ~high_obs

                respect_protected = bool(training_cfg.get("global_prune_respect_protected", True))
                if respect_protected:
                    protected_dia_reset_blocked = int((dia_reset_mask & frozen_protected_mask).sum().item())
                    protected_dia_prune_blocked = int((dia_prune_mask & frozen_protected_mask).sum().item())
                    dia_reset_mask = dia_reset_mask & ~frozen_protected_mask
                    dia_prune_mask = dia_prune_mask & ~frozen_protected_mask
            else:
                obs = self._anchor_replacement_obs.to(device=device, dtype=torch.float32)
                obs_safe = obs.clamp_min(1.0)
                inconsistent_ratio = (
                    self._anchor_inconsistent_hits.to(device=device, dtype=torch.float32)
                    / obs_safe
                )
                replacement_ratio = (
                    self._anchor_replacement_hits.to(device=device, dtype=torch.float32)
                    / obs_safe
                )
                supported = self._anchor_replacement_supported.to(device=device)
                reset_thresh = float(
                    training_cfg.get("dia_anchor_inconsistent_reset_ratio", 0.5)
                )
                replacement_thresh = float(
                    training_cfg.get("dia_anchor_replacement_prune_ratio", 0.5)
                )
                inconsistent_high = inconsistent_ratio >= reset_thresh
                replacement_high = replacement_ratio >= replacement_thresh
                dia_inconsistent_candidates = int(inconsistent_high.sum().item())
                dia_replacement_candidates = int((replacement_high & supported).sum().item())

                dia_reset_mask = inconsistent_high & (~replacement_high)
                if bool(training_cfg.get("dia_anchor_prune_enabled", False)):
                    dia_prune_mask = replacement_high & supported
                protected_dia_reset_blocked = int((dia_reset_mask & frozen_protected_mask).sum().item())
                protected_dia_prune_blocked = int((dia_prune_mask & frozen_protected_mask).sum().item())
                dia_reset_mask = dia_reset_mask & ~frozen_protected_mask
                dia_prune_mask = dia_prune_mask & ~frozen_protected_mask
            reset_enabled = bool(
                training_cfg.get(
                    "global_anchor_reset_enabled",
                    training_cfg.get("dia_anchor_reset_enabled", False),
                )
            )
            if reset_enabled and dia_reset_mask.any():
                reset_to = float(
                    training_cfg.get(
                        "global_anchor_reset_opacity",
                        training_cfg.get("dia_anchor_reset_opacity", min(float(min_opacity), 0.01)),
                    )
                )
                self._opacity.data[dia_reset_mask, 0] = inverse_sigmoid(
                    torch.full(
                        (int(dia_reset_mask.sum().item()),),
                        reset_to,
                        dtype=self._opacity.dtype,
                        device=self._opacity.device,
                    )
                )
                dia_reset_count = int(dia_reset_mask.sum().item())
        # Anchor scaffold pruning is intentionally limited to opacity pruning
        # plus the PFGS-style replacement evidence rules above.
        fastgs_prune_mask = torch.zeros_like(base_prune_mask)
        fastgs_prune_candidates = int(fastgs_prune_mask.sum().item())
        fastgs_pruned = int((fastgs_prune_mask & ~base_prune_mask).sum().item())
        prune_mask = base_prune_mask | fastgs_prune_mask | dia_prune_mask
        n_pruned = int(prune_mask.sum().item())
        if n_pruned > 0:
            self.prune_points(prune_mask)
        self._last_growth_stats = {"n_new_anchors_added": 0, "n_anchors_pruned": n_pruned}
        return {
            "n_clone": 0,
            "n_split": 0,
            "n_pruned": n_pruned,
            "fastgs_prune_candidates": fastgs_prune_candidates,
            "fastgs_pruned": fastgs_pruned,
            "dia_anchor_reset": dia_reset_count,
            "dia_anchor_pruned": int(dia_prune_mask.sum().item()),
            "dia_anchor_reset_candidates": int(dia_reset_mask.sum().item()),
            "dia_anchor_inconsistent_candidates": int(dia_inconsistent_candidates),
            "dia_anchor_depth_candidates": int(dia_depth_candidates),
            "dia_anchor_strong_candidates": int(dia_strong_candidates),
            "dia_anchor_hard_prune_candidates": int(dia_hard_prune_candidates),
            "dia_anchor_replacement_candidates": int(dia_replacement_candidates),
            "dia_anchor_sky_floater_candidates": int(dia_sky_floater_candidates),
            "protected_opacity_prune_blocked": int(protected_opacity_prune_blocked),
            "protected_dia_reset_blocked": int(protected_dia_reset_blocked),
            "protected_dia_prune_blocked": int(protected_dia_prune_blocked),
            **self._last_local_active_stats,
            "phase_ms": {},
        }

    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self.optimizer is None:
            return
        cfg_train = (self.config or {}).get("Training", {})
        if bool(cfg_train.get("enable_structure_xyz_freeze", False)):
            # Legacy behaviour: hard-freeze xyz lr for every anchor.
            for group in self.optimizer.param_groups:
                if group.get("name") == "xyz":
                    group["lr"] = 0.0
            self.lr_init = 0.0
            self.lr_final = 0.0
        # Otherwise leave xyz lr / scheduler as configured by base class so
        # structure anchors can refine their position. Sky anchors are still
        # protected via `_freeze_xyz_gradients` in the backend (mask-only).

    def update_learning_rate(self, iteration):
        cfg_train = (self.config or {}).get("Training", {})
        if not bool(cfg_train.get("enable_structure_xyz_freeze", False)):
            return super().update_learning_rate(iteration)
        if self.optimizer is None:
            return 0.0
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "xyz":
                param_group["lr"] = 0.0
                return 0.0
        return 0.0

    def build_active_anchor_selection(self, body_cam) -> ActiveAnchorSelection:
        total = int(self.get_xyz.shape[0])
        device = self.get_xyz.device
        if total == 0:
            empty_long = torch.zeros((0,), device=device, dtype=torch.long)
            empty_bool = torch.zeros((0,), device=device, dtype=torch.bool)
            return ActiveAnchorSelection(indices=empty_long, selection_mask=empty_bool, sky_mask=empty_bool)

        xyz = self.get_xyz.detach()
        cam_center = body_cam.camera_center.to(device=xyz.device, dtype=xyz.dtype)
        dists = torch.linalg.norm(xyz - cam_center[None, :], dim=1)
        opacity = self.get_opacity.detach().squeeze(-1)
        base_mask = opacity >= float(self.min_active_opacity)
        level_cpu = self._anchor_level
        if level_cpu.shape[0] != total:
            level_cpu = torch.zeros((total,), dtype=torch.int8)
        level = level_cpu.to(device=device)
        sky_cpu = self._is_sky_anchor if self._is_sky_anchor.shape[0] == total else self._is_sky
        if sky_cpu.shape[0] != total:
            sky_cpu = torch.zeros((total,), dtype=torch.bool)
        sky_mask = sky_cpu.to(device=device)

        if self.force_all_visible:
            selection_mask = torch.ones((total,), device=device, dtype=torch.bool)
            selected = torch.arange(total, device=device, dtype=torch.long)
            return ActiveAnchorSelection(
                indices=selected,
                selection_mask=selection_mask,
                sky_mask=sky_mask,
            )

        selection_mask = torch.zeros((total,), device=device, dtype=torch.bool)
        score = opacity / dists.clamp_min(1.0)

        sky_idx = torch.where(base_mask & sky_mask)[0]
        if sky_idx.numel() > 0:
            sky_budget = min(int(self.sky_active_budget), int(sky_idx.numel()))
            if sky_idx.numel() > sky_budget:
                sky_keep = torch.topk(score[sky_idx], k=sky_budget, largest=True).indices
                sky_idx = sky_idx[sky_keep]
            selection_mask[sky_idx] = True

        struct_budget = max(
            0, int(self.max_active_anchors_per_frame) - int(selection_mask.sum().item())
        )
        budget_weights = torch.tensor([0.5, 0.3, 0.2], device=device, dtype=torch.float32)
        level_budgets = torch.floor(struct_budget * budget_weights).to(dtype=torch.long)
        level_budgets[0] += struct_budget - int(level_budgets.sum().item())
        for level_id in range(3):
            idx = torch.where(base_mask & (~sky_mask) & (level == level_id))[0]
            if idx.numel() == 0:
                continue
            budget = min(int(level_budgets[level_id].item()), int(idx.numel()))
            if budget <= 0:
                continue
            if idx.numel() > budget:
                keep = torch.topk(score[idx], k=budget, largest=True).indices
                idx = idx[keep]
            selection_mask[idx] = True

        selected = torch.where(selection_mask)[0]
        if selected.numel() == 0 and total > 0:
            best = torch.argmax(score).view(1)
            selection_mask[best] = True
            selected = best
        return ActiveAnchorSelection(indices=selected, selection_mask=selection_mask, sky_mask=sky_mask)

    def record_render_stats(
        self,
        selection: ActiveAnchorSelection,
        radii: torch.Tensor,
        render_loss: Optional[float] = None,
        sky_loss: Optional[float] = None,
    ) -> None:
        sky_active = int((selection.selection_mask & selection.sky_mask).sum().item())
        self._last_render_stats = {
            "n_structure_anchors_total": int((~self._is_sky_anchor).sum().item())
            if self._is_sky_anchor.shape[0] == self.get_xyz.shape[0]
            else int(self.get_xyz.shape[0]),
            "n_structure_anchors_active": int(
                (selection.selection_mask & (~selection.sky_mask)).sum().item()
            ),
            "n_sky_anchors_active": sky_active,
            "n_decoded_gaussians": int(selection.indices.numel()),
            "max_scale": float(self.get_scaling.max().item()) if self.get_scaling.numel() > 0 else 0.0,
            "mean_radii": float(radii.float().mean().item()) if radii.numel() > 0 else 0.0,
            "render_loss": float(render_loss) if render_loss is not None else None,
            "sky_loss": float(sky_loss) if sky_loss is not None else None,
        }

    def get_anchor_debug_stats(self) -> dict:
        stats = {
            "n_structure_anchors_total": int((~self._is_sky_anchor).sum().item())
            if self._is_sky_anchor.shape[0] == self.get_xyz.shape[0]
            else int(self.get_xyz.shape[0]),
            "n_structure_anchors_active": 0,
            "n_sky_anchors_active": 0,
            "n_decoded_gaussians": 0,
            "n_new_anchors_added": int(self._last_growth_stats.get("n_new_anchors_added", 0)),
            "n_anchors_pruned": int(self._last_growth_stats.get("n_anchors_pruned", 0)),
            "render_loss": None,
            "sky_loss": None,
            "max_scale": float(self.get_scaling.max().item()) if self.get_scaling.numel() > 0 else 0.0,
            "nan_count": int(torch.isnan(self.get_xyz).sum().item()) if self.get_xyz.numel() > 0 else 0,
        }
        stats.update(self._last_hash_stats)
        stats.update(self._last_render_stats)
        stats.update(self._last_local_active_stats)
        return stats
