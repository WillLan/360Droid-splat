"""Minimal panoramic Gaussian map and mapper.

The map is intentionally compact: it exposes the attributes expected by the
PFGS360 adapter while keeping anchor-scaffold metadata local to this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import PFGS360Renderer, PanoRenderCamera
from backend.pano_gs.losses import BackendLossWeights, backend_render_loss
from backend.pano_gs.pose_param import PoseDelta
from frontend.pano_droid.interfaces import FrontendOutput
from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing, pixel_grid
from geometry.pose import invert_c2w
from geometry.sim3 import apply_sim3, sim3_components, sim3_inverse
from mapping.gaussian_initializer import GaussianSeedBatch
from models.per_pixel_gaussian_observation import (
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    real_sh_basis,
)


@dataclass
class MapperStats:
    n_keyframes: int = 0
    n_anchors: int = 0
    last_loss: float | None = None
    last_phase: str | None = None
    last_pose_delta_norm: float | None = None
    optimization_steps: int = 0
    last_backend: str = "pfgs360_gsplat"
    fallback_renderer: bool = False
    last_inserted_anchors: int = 0
    last_skipped_voxel: int = 0
    last_skipped_budget: int = 0
    last_window_size: int = 0
    last_window_keyframes: list[int] = field(default_factory=list)
    last_active_keyframes: list[int] = field(default_factory=list)
    last_window_observations: list[int] = field(default_factory=list)
    last_feedforward_current_frames: list[int] = field(default_factory=list)
    last_feedforward_history_frames: list[int] = field(default_factory=list)
    last_sampled_keyframes: list[int] = field(default_factory=list)
    last_trainable_pose_count: int = 0
    last_feedforward_opacity_resets: int = 0
    last_feedforward_pruned: int = 0
    last_replace_deleted: int = 0
    last_replace_fused: int = 0
    last_replace_compacted: int = 0
    last_sky_pruned: int = 0
    last_sky_compacted: int = 0
    last_hash_hits: int = 0
    last_hash_near_hits: int = 0
    last_suppressed_insert: int = 0
    last_outlier_resets: int = 0
    last_outlier_pruned: int = 0
    last_render_missing_pixels: int = 0
    last_render_depth_mismatch_pixels: int = 0
    last_render_bad_pixels: int = 0
    last_missing_seed_candidates: int = 0
    last_depth_mismatch_seed_candidates: int = 0
    last_skipped_missing_budget: int = 0
    last_skipped_depth_mismatch_budget: int = 0
    last_dense_seed_candidates: int = 0
    last_insert_mask_seed_candidates: int = 0
    last_voxel_seed_candidates: int = 0
    last_replace_fused_existing: int = 0
    last_replace_fused_new_duplicate: int = 0
    last_replace_newly_inserted: int = 0
    last_pred_depth_generated_seeds: int = 0
    last_pred_depth_invalid_pixels: int = 0
    last_insert_mask_pixels: int = 0
    last_anchor_count_before_insert: int = 0
    last_anchor_count_after_insert: int = 0
    last_neural_insert_total_sec: float = 0.0
    last_neural_insert_accept_sec: float = 0.0
    last_neural_insert_append_sec: float = 0.0
    last_neural_insert_compact_sec: float = 0.0
    last_resplat_fused: int = 0
    last_resplat_inserted: int = 0
    last_resplat_skipped: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class MapperKeyframe:
    frame_id: int
    image: torch.Tensor
    gaussian_start: int
    gaussian_end: int
    sky_mask: torch.Tensor | None = None
    target_depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None


@dataclass
class MapperObservation:
    frame_id: int
    image: torch.Tensor
    pose_c2w: torch.Tensor
    is_keyframe: bool = False
    sky_mask: torch.Tensor | None = None
    target_depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None
    target_depth_local: torch.Tensor | None = None
    target_depth_scale: float = 1.0
    owner_window_id: int | None = None


@dataclass
class KeyframeRenderDiagnostic:
    frame_id: int
    target: torch.Tensor
    render: torch.Tensor
    depth: torch.Tensor | None
    target_depth: torch.Tensor | None
    loss: float
    psnr: float
    anchor_count: int
    phase: str | None


@dataclass
class DepthInsertionDiagnostic:
    frame_id: int
    render_depth: torch.Tensor | None
    predicted_depth: torch.Tensor | None
    rel_depth_error: torch.Tensor | None
    missing_mask: torch.Tensor | None
    depth_mismatch_mask: torch.Tensor | None
    render_bad_mask: torch.Tensor | None
    depth_scale: float = 1.0
    depth_shift: float = 0.0


class PanoGaussianMap(nn.Module):
    """Anchor-scaffold panorama map with gsplat360-compatible accessors."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        sh_degree: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.map_mode = "anchor_scaffold_panorama"
        backend_cfg = self.config.get("BackendOptimization", {}) if isinstance(self.config, dict) else {}
        configured_sh_degree = int(backend_cfg.get("sh_degree", sh_degree)) if isinstance(backend_cfg, dict) else int(sh_degree)
        global_selfi_cfg = self.config.get("SphericalSelfiGlobalBackend", {}) if isinstance(self.config, dict) else {}
        if isinstance(global_selfi_cfg, dict) and bool(global_selfi_cfg.get("enabled", False)):
            configured_sh_degree = max(configured_sh_degree, int(global_selfi_cfg.get("rgb_sh_degree", 2)))
        self.max_sh_degree = max(0, min(configured_sh_degree, 2))
        self.active_sh_degree = self.max_sh_degree
        self.device_hint = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._reset_parameters()
        self._reset_skybox()
        self._anchor_level = torch.zeros(0, dtype=torch.int8)
        self._anchor_voxel_size = torch.zeros(0, dtype=torch.float32)
        self._anchor_grid_coord = torch.zeros(0, 3, dtype=torch.int32)
        self._anchor_obs_count = torch.zeros(0, dtype=torch.int32)
        self._anchor_conf_accum = torch.zeros(0, dtype=torch.float32)
        self._anchor_birth_frame = torch.zeros(0, dtype=torch.int32)
        self._anchor_last_seen_kf = torch.zeros(0, dtype=torch.int32)
        self._anchor_last_update_kf_ord = torch.zeros(0, dtype=torch.int32)
        self._anchor_source_window_id = torch.zeros(0, dtype=torch.int32)
        self._anchor_source_frame_start = torch.zeros(0, dtype=torch.int32)
        self._anchor_source_frame_end = torch.zeros(0, dtype=torch.int32)
        self._anchor_inlier_obs = torch.zeros(0, dtype=torch.int32)
        self._anchor_outlier_obs = torch.zeros(0, dtype=torch.int32)
        self._anchor_owner_window_id = torch.full((0,), -1, dtype=torch.int32)
        self._anchor_quality = torch.zeros(0, dtype=torch.float32)
        self._anchor_visibility_count = torch.zeros(0, dtype=torch.int32)
        self._anchor_render_error_ema = torch.zeros(0, dtype=torch.float32)
        self._anchor_depth_selected_levels = False
        self._lazy_owner_transforms_enabled = False
        self._lazy_owner_reference_transforms: dict[int, torch.Tensor] = {}
        self._lazy_owner_current_transforms: dict[int, torch.Tensor] = {}
        self._lazy_sh_rotation_cache: dict[tuple[int, str, str, int], torch.Tensor] = {}

    def _reset_parameters(self) -> None:
        device = self.device_hint
        sh_rest_dim = max(0, (int(self.max_sh_degree) + 1) ** 2 - 1)
        self.xyz = nn.Parameter(torch.zeros(0, 3, device=device))
        self.rotation = nn.Parameter(torch.zeros(0, 4, device=device))
        self.scaling = nn.Parameter(torch.zeros(0, 3, device=device))
        self.opacity_logit = nn.Parameter(torch.zeros(0, 1, device=device))
        self.features = nn.Parameter(torch.zeros(0, 3, device=device))
        self.sh_rest = nn.Parameter(torch.zeros(0, sh_rest_dim, 3, device=device))

    def _reset_skybox(self) -> None:
        cfg = self.config.get("SkyBox", {}) if isinstance(self.config, dict) else {}
        self.skybox_enabled = bool(cfg.get("enabled", False))
        self.skybox_optimize = bool(cfg.get("optimize", True))
        self.skybox_optimization_mask_enable = bool(cfg.get("optimization_mask_enable", True))
        self.skybox_force_sky_render = bool(cfg.get("force_sky_render", False))
        self.skybox_init_fallback_to_full_image = bool(cfg.get("init_fallback_to_full_image", False))
        self.skybox_resolution = max(4, int(cfg.get("resolution", 512)))
        self.skybox_lr = float(cfg.get("lr", 1.0e-2))
        self._skybox_initialized = False
        self.skybox_logits: nn.Parameter | None = None
        if self.skybox_enabled:
            init = torch.full(
                (6, 3, self.skybox_resolution, self.skybox_resolution),
                0.5,
                device=self.device_hint,
                dtype=torch.float32,
            )
            self.skybox_logits = nn.Parameter(self._inv_sigmoid(init))

    @property
    def get_xyz(self) -> torch.Tensor:
        if not self._lazy_owner_transforms_enabled or self.xyz.numel() == 0:
            return self.xyz
        output = self.xyz.clone()
        for owner, mask in self._lazy_owner_masks(device=output.device):
            delta = self._lazy_owner_delta(owner, device=output.device, dtype=output.dtype)
            output[mask] = apply_sim3(delta, self.xyz[mask])
        return output

    @property
    def get_rotation(self) -> torch.Tensor:
        base = self._base_rotation()
        if not self._lazy_owner_transforms_enabled or base.numel() == 0:
            return base
        output = base.clone()
        for owner, mask in self._lazy_owner_masks(device=base.device):
            delta = self._lazy_owner_delta(owner, device=base.device, dtype=base.dtype)
            _, rotation, _ = sim3_components(delta)
            quaternion = matrix_to_quaternion(rotation).view(1, 4)
            output[mask] = normalize_quaternion(
                quaternion_multiply(quaternion, base[mask])
            )
        return output

    @property
    def get_scaling(self) -> torch.Tensor:
        base = self._base_scaling()
        if not self._lazy_owner_transforms_enabled or base.numel() == 0:
            return base
        output = base.clone()
        for owner, mask in self._lazy_owner_masks(device=base.device):
            delta = self._lazy_owner_delta(owner, device=base.device, dtype=base.dtype)
            scale, _, _ = sim3_components(delta)
            output[mask] = scale * base[mask]
        return output

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.sigmoid(self.features)

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        base = self._base_sh_coefficients()
        if not self._lazy_owner_transforms_enabled or base.numel() == 0:
            return base
        output = base.clone()
        for owner, mask in self._lazy_owner_masks(device=base.device):
            matrix = self._lazy_sh_rotation_matrix(
                owner,
                device=base.device,
                dtype=base.dtype,
            )
            output[mask] = torch.einsum("ij,njc->nic", matrix, base[mask])
        return output

    def _base_sh_coefficients(self) -> torch.Tensor:
        from backend.pano_gs.adapter import SH_C0

        dc = ((self.get_features - 0.5) / SH_C0).unsqueeze(1)
        if int(self.sh_rest.shape[1]) <= 0:
            return dc
        return torch.cat([dc, self.sh_rest.to(device=dc.device, dtype=dc.dtype)], dim=1)

    def _base_rotation(self) -> torch.Tensor:
        if self.rotation.numel() == 0:
            return self.rotation
        return torch.nn.functional.normalize(self.rotation, dim=-1, eps=1e-12)

    def _base_scaling(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.scaling) + 1e-5

    def configure_lazy_owner_transforms(self, enabled: bool) -> None:
        self._lazy_owner_transforms_enabled = bool(enabled)
        self._lazy_sh_rotation_cache.clear()

    def set_lazy_owner_transform(
        self,
        owner_window_id: int,
        transform: torch.Tensor,
        *,
        set_reference: bool = False,
    ) -> None:
        owner = int(owner_window_id)
        value = transform.detach().cpu().float().clone()
        if tuple(value.shape) != (4, 4) or not bool(torch.isfinite(value).all()):
            raise ValueError("Lazy owner transform must be a finite 4x4 Sim(3)")
        scale, rotation, _ = sim3_components(value)
        if float(scale) <= 0.0 or float(torch.linalg.det(rotation)) <= 0.0:
            raise ValueError("Lazy owner transform must have positive scale and rotation determinant")
        if set_reference or owner not in self._lazy_owner_reference_transforms:
            self._lazy_owner_reference_transforms[owner] = value.clone()
        self._lazy_owner_current_transforms[owner] = value
        for key in list(self._lazy_sh_rotation_cache):
            if key[0] == owner:
                del self._lazy_sh_rotation_cache[key]

    def lazy_owner_transform_state(self) -> dict[str, object]:
        return {
            "enabled": bool(self._lazy_owner_transforms_enabled),
            "reference": {
                int(owner): value.clone()
                for owner, value in self._lazy_owner_reference_transforms.items()
            },
            "current": {
                int(owner): value.clone()
                for owner, value in self._lazy_owner_current_transforms.items()
            },
        }

    def materialized_anchor_voxel_size(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return per-anchor voxel sizes after lazy owner Sim(3) corrections."""

        target_device = self.xyz.device if device is None else torch.device(device)
        target_dtype = self.xyz.dtype if dtype is None else dtype
        output = self._anchor_voxel_size.to(
            device=target_device,
            dtype=target_dtype,
        ).clone()
        if not self._lazy_owner_transforms_enabled or output.numel() == 0:
            return output
        for owner, mask in self._lazy_owner_masks(device=target_device):
            delta = self._lazy_owner_delta(
                owner,
                device=target_device,
                dtype=target_dtype,
            )
            scale, _, _ = sim3_components(delta)
            output[mask] *= scale
        return output

    def materialized_anchor_geometry_rows(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize world xyz and voxel size for selected anchor rows only."""

        rows_cpu = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if int(rows_cpu.numel()) == 0:
            return (
                self.xyz.new_zeros((0, 3)),
                self.xyz.new_zeros((0,)),
            )
        count = self.anchor_count()
        if int(rows_cpu.min()) < 0 or int(rows_cpu.max()) >= count:
            raise IndexError("Materialized anchor rows are outside the current map")

        rows = rows_cpu.to(device=self.xyz.device)
        xyz = self.xyz.detach().index_select(0, rows).clone()
        voxel_size = (
            self._anchor_voxel_size.detach()
            .index_select(0, rows_cpu)
            .to(
                device=self.xyz.device,
                dtype=self.xyz.dtype,
            )
        )
        if (
            not self._lazy_owner_transforms_enabled
            or int(self._anchor_owner_window_id.numel()) != count
        ):
            return xyz, voxel_size

        owners = self._anchor_owner_window_id.index_select(0, rows_cpu).to(
            device=self.xyz.device,
            dtype=torch.long,
        )
        for owner in torch.unique(owners).detach().cpu().tolist():
            owner_id = int(owner)
            if (
                owner_id < 0
                or owner_id not in self._lazy_owner_reference_transforms
                or owner_id not in self._lazy_owner_current_transforms
            ):
                continue
            mask = owners == owner_id
            delta = self._lazy_owner_delta(
                owner_id,
                device=self.xyz.device,
                dtype=self.xyz.dtype,
            )
            scale, _, _ = sim3_components(delta)
            xyz[mask] = apply_sim3(delta, xyz[mask])
            voxel_size[mask] *= scale
        return xyz, voxel_size

    def _lazy_owner_masks(
        self,
        *,
        device: torch.device,
    ) -> list[tuple[int, torch.Tensor]]:
        if int(self._anchor_owner_window_id.numel()) != self.anchor_count():
            return []
        owners = self._anchor_owner_window_id.to(device=device, dtype=torch.long)
        output: list[tuple[int, torch.Tensor]] = []
        for owner in torch.unique(owners).detach().cpu().tolist():
            owner_id = int(owner)
            if (
                owner_id < 0
                or owner_id not in self._lazy_owner_reference_transforms
                or owner_id not in self._lazy_owner_current_transforms
            ):
                continue
            output.append((owner_id, owners == owner_id))
        return output

    def _lazy_owner_delta(
        self,
        owner: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        reference = self._lazy_owner_reference_transforms[int(owner)].to(
            device=device, dtype=dtype
        )
        current = self._lazy_owner_current_transforms[int(owner)].to(
            device=device, dtype=dtype
        )
        return current @ sim3_inverse(reference)

    def _lazy_sh_rotation_matrix(
        self,
        owner: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(owner), str(device), str(dtype), int(self.active_sh_degree))
        cached = self._lazy_sh_rotation_cache.get(key)
        if cached is not None:
            return cached
        delta = self._lazy_owner_delta(owner, device=device, dtype=dtype)
        _, rotation, _ = sim3_components(delta)
        coefficient_count = (int(self.active_sh_degree) + 1) ** 2
        count = max(32, coefficient_count * 4)
        index = torch.arange(count, device=device, dtype=dtype)
        y = 1.0 - 2.0 * (index + 0.5) / float(count)
        radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
        angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
        target_direction = torch.stack(
            [radius * torch.cos(angle), y, radius * torch.sin(angle)], dim=-1
        )
        target_basis = real_sh_basis(self.active_sh_degree, target_direction)
        local_basis = real_sh_basis(self.active_sh_degree, target_direction @ rotation)
        matrix = torch.linalg.pinv(target_basis) @ local_basis
        self._lazy_sh_rotation_cache[key] = matrix
        return matrix

    @staticmethod
    def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-5, 1.0 - 1e-5)
        return torch.log(x / (1.0 - x))

    @staticmethod
    def _inverse_softplus_scale(scale: torch.Tensor) -> torch.Tensor:
        """Convert renderer-space scale to this map's unconstrained parameter."""

        target = (scale.clamp_min(2.0e-5) - 1.0e-5).clamp_min(1.0e-8)
        return torch.log(torch.expm1(target).clamp_min(1.0e-8))

    @staticmethod
    def _normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        quat = torch.nan_to_num(quaternion, nan=0.0, posinf=0.0, neginf=0.0)
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        return torch.where(norm > float(eps), quat / norm.clamp_min(float(eps)), identity)

    @property
    def has_skybox(self) -> bool:
        return bool(self.skybox_enabled and self.skybox_logits is not None)

    @property
    def get_skybox_faces(self) -> torch.Tensor | None:
        if self.skybox_logits is None:
            return None
        return torch.sigmoid(self.skybox_logits)

    def gaussian_parameters(self) -> list[nn.Parameter]:
        params = [self.xyz, self.rotation, self.scaling, self.opacity_logit, self.features]
        if int(self.sh_rest.shape[1]) > 0:
            params.append(self.sh_rest)
        return params

    def skybox_parameters(self) -> list[nn.Parameter]:
        if self.skybox_logits is None or not self.skybox_optimize:
            return []
        return [self.skybox_logits]

    def anchor_count(self) -> int:
        return int(self.xyz.shape[0])

    def add_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        voxel_size: float | None = None,
        last_update_kf_ord: int | None = None,
    ) -> int:
        if len(seeds) == 0:
            return 0
        device = self.xyz.device
        dtype = self.xyz.dtype
        xyz = seeds.xyz.to(device=device, dtype=dtype)
        rgb = seeds.rgb.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        conf = seeds.confidence.to(device=device, dtype=dtype).view(-1, 1).clamp(1e-4, 1.0)
        scale = seeds.scale.to(device=device, dtype=dtype).view(-1, 1).expand(-1, 3)
        quat = torch.zeros(xyz.shape[0], 4, device=device, dtype=dtype)
        quat[:, 0] = 1.0

        new_xyz = torch.cat([self.xyz.detach(), xyz], dim=0)
        new_rot = torch.cat([self.rotation.detach(), quat], dim=0)
        new_scaling = torch.cat([self.scaling.detach(), torch.log(torch.expm1(scale.clamp_min(1e-5)))], dim=0)
        new_opacity = torch.cat([self.opacity_logit.detach(), self._inv_sigmoid(conf)], dim=0)
        new_features = torch.cat([self.features.detach(), self._inv_sigmoid(rgb)], dim=0)
        sh_rest = torch.zeros(xyz.shape[0], int(self.sh_rest.shape[1]), 3, device=device, dtype=dtype)
        new_sh_rest = torch.cat([self.sh_rest.detach(), sh_rest], dim=0)

        self.xyz = nn.Parameter(new_xyz)
        self.rotation = nn.Parameter(new_rot)
        self.scaling = nn.Parameter(new_scaling)
        self.opacity_logit = nn.Parameter(new_opacity)
        self.features = nn.Parameter(new_features)
        self.sh_rest = nn.Parameter(new_sh_rest)

        self._anchor_level = torch.cat([self._anchor_level, seeds.level.detach().cpu().to(torch.int8)], dim=0)
        self._anchor_voxel_size = torch.cat(
            [self._anchor_voxel_size, seeds.scale.detach().cpu().to(torch.float32)], dim=0
        )
        if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == int(len(seeds)):
            grid = seeds.grid_coord.detach().cpu().to(torch.int32)
        elif voxel_size is not None and float(voxel_size) > 0.0:
            grid = torch.floor(seeds.xyz.detach().cpu() / float(voxel_size)).to(torch.int32)
        else:
            grid = torch.floor(seeds.xyz.detach().cpu() / seeds.scale.detach().cpu().view(-1, 1).clamp_min(1e-6)).to(torch.int32)
        self._anchor_grid_coord = torch.cat([self._anchor_grid_coord, grid.to(torch.int32)], dim=0)
        self._anchor_obs_count = torch.cat(
            [self._anchor_obs_count, torch.ones(len(seeds), dtype=torch.int32)], dim=0
        )
        self._anchor_conf_accum = torch.cat(
            [self._anchor_conf_accum, seeds.confidence.detach().cpu().to(torch.float32)], dim=0
        )
        frame_ids = torch.full((len(seeds),), int(seeds.frame_id), dtype=torch.int32)
        self._anchor_birth_frame = torch.cat([self._anchor_birth_frame, frame_ids], dim=0)
        self._anchor_last_seen_kf = torch.cat([self._anchor_last_seen_kf, frame_ids], dim=0)
        update_ord = int(seeds.frame_id) if last_update_kf_ord is None else int(last_update_kf_ord)
        self._anchor_last_update_kf_ord = torch.cat(
            [self._anchor_last_update_kf_ord, torch.full((len(seeds),), update_ord, dtype=torch.int32)],
            dim=0,
        )
        self._anchor_source_window_id = torch.cat(
            [self._anchor_source_window_id, torch.full((len(seeds),), -1, dtype=torch.int32)],
            dim=0,
        )
        self._anchor_source_frame_start = torch.cat([self._anchor_source_frame_start, frame_ids], dim=0)
        self._anchor_source_frame_end = torch.cat([self._anchor_source_frame_end, frame_ids], dim=0)
        self._anchor_inlier_obs = torch.cat([self._anchor_inlier_obs, torch.zeros(len(seeds), dtype=torch.int32)], dim=0)
        self._anchor_outlier_obs = torch.cat([self._anchor_outlier_obs, torch.zeros(len(seeds), dtype=torch.int32)], dim=0)
        self._anchor_owner_window_id = torch.cat(
            [self._anchor_owner_window_id, torch.full((len(seeds),), -1, dtype=torch.int32)], dim=0
        )
        self._anchor_quality = torch.cat(
            [self._anchor_quality, seeds.confidence.detach().cpu().float()], dim=0
        )
        self._anchor_visibility_count = torch.cat(
            [self._anchor_visibility_count, torch.zeros(len(seeds), dtype=torch.int32)], dim=0
        )
        self._anchor_render_error_ema = torch.cat(
            [self._anchor_render_error_ema, torch.zeros(len(seeds), dtype=torch.float32)], dim=0
        )
        return int(xyz.shape[0])

    def add_or_fuse_resplat_gaussians(
        self,
        state,
        *,
        batch_index: int = 0,
        frame_ids: list[int] | tuple[int, ...] | None = None,
        window_id: int | None = None,
        config: dict | None = None,
    ) -> dict[str, int | float]:
        """Insert or merge a refined local ``PanoGaussianState`` into the global map.

        ``state`` is intentionally duck-typed to avoid making the backend import
        the frontend module at import time. Expected tensor shapes follow
        ``PanoGaussianState``: ``means/log_scales/...`` are ``B x N x ...`` and
        ``sh_coeffs`` is ``B x N x 3 x SH``.
        """

        cfg = config if isinstance(config, dict) else {}
        idx = int(batch_index)
        if idx < 0 or idx >= int(state.means.shape[0]):
            raise IndexError(f"batch_index={idx} out of range for state batch {int(state.means.shape[0])}")

        device = self.xyz.device
        dtype = self.xyz.dtype
        from backend.pano_gs.adapter import SH_C0

        means = state.means[idx].detach().to(device=device, dtype=dtype)
        log_scales = state.log_scales[idx].detach().to(device=device, dtype=dtype)
        rotations = state.rotations_unnorm[idx].detach().to(device=device, dtype=dtype)
        opacity_logits = state.opacity_logits[idx].detach().to(device=device, dtype=dtype)
        sh_coeffs = state.sh_coeffs[idx].detach().to(device=device, dtype=dtype)
        valid_mask = state.valid_mask[idx].detach().to(device=device, dtype=torch.bool).view(-1)
        confidence_t = getattr(state, "confidence", None)
        if confidence_t is None:
            confidence = torch.ones(means.shape[0], device=device, dtype=dtype)
        else:
            confidence = confidence_t[idx].detach().to(device=device, dtype=dtype).view(-1).clamp(0.0, 1.0)

        target_sh_dim = int(self.sh_rest.shape[1]) + 1
        target_sh_dim = max(1, target_sh_dim)
        incoming_sh_dim = int(sh_coeffs.shape[-1])
        if incoming_sh_dim != target_sh_dim:
            padded = torch.zeros(
                int(sh_coeffs.shape[0]),
                3,
                target_sh_dim,
                device=device,
                dtype=dtype,
            )
            copy_dim = min(incoming_sh_dim, target_sh_dim)
            padded[..., :copy_dim] = sh_coeffs[..., :copy_dim]
            sh_coeffs = padded

        scale = torch.exp(torch.nan_to_num(log_scales, nan=-8.0, posinf=2.0, neginf=-8.0))
        scale = scale.clamp(
            min=float(cfg.get("min_scale", 1.0e-5)),
            max=float(cfg.get("max_scale", 1.0)),
        )
        opacity_logits = torch.nan_to_num(opacity_logits, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-20.0, 20.0)
        opacity_prob = torch.sigmoid(opacity_logits).view(-1)
        finite = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(scale).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacity_logits).all(dim=-1)
            & torch.isfinite(sh_coeffs).all(dim=(-1, -2))
            & torch.isfinite(confidence)
        )
        keep = (
            valid_mask
            & finite
            & (confidence >= float(cfg.get("min_confidence", 0.0)))
            & (opacity_prob >= float(cfg.get("min_opacity", 0.0)))
        )
        requested = int(means.shape[0])
        if not bool(keep.any()):
            return {
                "requested": requested,
                "valid": 0,
                "fused": 0,
                "inserted": 0,
                "skipped": requested,
                "anchors_before": self.anchor_count(),
                "anchors_after": self.anchor_count(),
            }

        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
        weight = (confidence * opacity_prob).clamp_min(1.0e-6)
        order = keep_idx[torch.argsort(weight[keep_idx].detach(), descending=True)]

        voxel_size = float(cfg.get("voxel_size", cfg.get("merge_radius", 0.12)))
        voxel_size = max(voxel_size, 1.0e-8)
        merge_radius = float(cfg.get("merge_radius", voxel_size))
        max_scale_ratio = max(1.0, float(cfg.get("max_scale_ratio", 8.0)))
        neighbor_radius = max(0, int(cfg.get("neighbor_radius", 1)))
        max_new = int(cfg.get("max_new_gaussians_per_window", requested))
        max_total = int(cfg.get("max_total_gaussians", 0))
        remaining_total = requested if max_total <= 0 else max(0, max_total - self.anchor_count())
        insert_budget = max(0, min(max_new, remaining_total))
        last_frame = int(frame_ids[-1]) if frame_ids else int(window_id or 0)
        first_frame = int(frame_ids[0]) if frame_ids else last_frame
        update_ord = int(window_id) if window_id is not None else last_frame

        old_count = self.anchor_count()
        xyz = self.xyz.detach().clone()
        rot = self.get_rotation.detach().clone()
        scale_actual = self.get_scaling.detach().clone()
        opacity = self.get_opacity.detach().clone()
        old_sh = self.get_sh_coefficients.detach().permute(0, 2, 1).contiguous()
        if int(old_sh.shape[-1]) != target_sh_dim:
            resized = torch.zeros(old_count, 3, target_sh_dim, device=device, dtype=dtype)
            copy_dim = min(int(old_sh.shape[-1]), target_sh_dim)
            if copy_dim > 0:
                resized[..., :copy_dim] = old_sh[..., :copy_dim]
            old_sh = resized

        meta_level = self._anchor_level.detach().clone()
        meta_voxel = self._anchor_voxel_size.detach().clone()
        meta_grid = self._anchor_grid_coord.detach().clone()
        meta_obs = self._anchor_obs_count.detach().clone()
        meta_conf = self._anchor_conf_accum.detach().clone()
        meta_birth = self._anchor_birth_frame.detach().clone()
        meta_seen = self._anchor_last_seen_kf.detach().clone()
        meta_update = self._anchor_last_update_kf_ord.detach().clone()
        meta_window = self._anchor_source_window_id.detach().clone()
        meta_frame_start = self._anchor_source_frame_start.detach().clone()
        meta_frame_end = self._anchor_source_frame_end.detach().clone()
        meta_inlier = self._anchor_inlier_obs.detach().clone()
        meta_outlier = self._anchor_outlier_obs.detach().clone()
        meta_owner = self._anchor_owner_window_id.detach().clone()
        meta_quality = self._anchor_quality.detach().clone()
        meta_visibility = self._anchor_visibility_count.detach().clone()
        meta_render_error = self._anchor_render_error_ema.detach().clone()

        def key_for_point(point: torch.Tensor) -> tuple[int, int, int]:
            grid = torch.floor(point.detach().cpu().float() / voxel_size).to(torch.int64)
            return (int(grid[0].item()), int(grid[1].item()), int(grid[2].item()))

        occupied: dict[tuple[int, int, int], list[int]] = {}
        for row in range(old_count):
            key = key_for_point(xyz[row])
            occupied.setdefault(key, []).append(row)

        insert_xyz: list[torch.Tensor] = []
        insert_rot: list[torch.Tensor] = []
        insert_scale: list[torch.Tensor] = []
        insert_opacity: list[torch.Tensor] = []
        insert_sh: list[torch.Tensor] = []
        insert_grid: list[tuple[int, int, int]] = []
        insert_conf: list[float] = []
        insert_weight: list[float] = []
        insert_obs: list[int] = []
        fused = 0
        inserted = 0
        skipped_budget = 0

        def params_for_index(global_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if global_idx < old_count:
                return xyz[global_idx], scale_actual[global_idx], opacity[global_idx], old_sh[global_idx]
            rel = int(global_idx) - old_count
            return insert_xyz[rel], insert_scale[rel], insert_opacity[rel], insert_sh[rel]

        def assign_existing(global_idx: int, new_xyz: torch.Tensor, new_scale: torch.Tensor, new_rot: torch.Tensor, new_opacity: torch.Tensor, new_sh: torch.Tensor, alpha: float) -> None:
            a = float(alpha)
            if global_idx < old_count:
                old_r = rot[global_idx]
                if torch.sum(old_r * new_rot) < 0:
                    new_rot = -new_rot
                xyz[global_idx] = (1.0 - a) * xyz[global_idx] + a * new_xyz
                log_s = (1.0 - a) * scale_actual[global_idx].clamp_min(1.0e-8).log() + a * new_scale.clamp_min(1.0e-8).log()
                scale_actual[global_idx] = torch.exp(log_s).clamp_min(1.0e-5)
                rot[global_idx] = self._normalize_quaternion((1.0 - a) * old_r + a * new_rot)
                opacity[global_idx] = ((1.0 - a) * opacity[global_idx] + a * new_opacity).clamp(1.0e-5, 1.0 - 1.0e-5)
                old_sh[global_idx] = (1.0 - a) * old_sh[global_idx] + a * new_sh
            else:
                rel = int(global_idx) - old_count
                old_r = insert_rot[rel]
                if torch.sum(old_r * new_rot) < 0:
                    new_rot = -new_rot
                insert_xyz[rel] = (1.0 - a) * insert_xyz[rel] + a * new_xyz
                log_s = (1.0 - a) * insert_scale[rel].clamp_min(1.0e-8).log() + a * new_scale.clamp_min(1.0e-8).log()
                insert_scale[rel] = torch.exp(log_s).clamp_min(1.0e-5)
                insert_rot[rel] = self._normalize_quaternion((1.0 - a) * old_r + a * new_rot)
                insert_opacity[rel] = ((1.0 - a) * insert_opacity[rel] + a * new_opacity).clamp(1.0e-5, 1.0 - 1.0e-5)
                insert_sh[rel] = (1.0 - a) * insert_sh[rel] + a * new_sh

        for src_idx in order.tolist():
            mean_i = torch.nan_to_num(means[src_idx], nan=0.0, posinf=0.0, neginf=0.0)
            scale_i = scale[src_idx]
            rot_i = self._normalize_quaternion(rotations[src_idx])
            opacity_i = opacity_prob[src_idx].view(1).clamp(1.0e-5, 1.0 - 1.0e-5)
            sh_i = torch.nan_to_num(sh_coeffs[src_idx], nan=0.0, posinf=0.0, neginf=0.0)
            key = key_for_point(mean_i)
            candidates: list[int] = []
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                for dy in range(-neighbor_radius, neighbor_radius + 1):
                    for dx in range(-neighbor_radius, neighbor_radius + 1):
                        candidates.extend(occupied.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
            best_idx: int | None = None
            best_dist = float("inf")
            for cand in candidates:
                cand_xyz, cand_scale, _, _ = params_for_index(cand)
                dist = float(torch.linalg.norm((cand_xyz - mean_i).detach()).cpu())
                if dist > merge_radius or dist >= best_dist:
                    continue
                ratio = torch.maximum(
                    cand_scale.clamp_min(1.0e-8) / scale_i.clamp_min(1.0e-8),
                    scale_i.clamp_min(1.0e-8) / cand_scale.clamp_min(1.0e-8),
                )
                if float(ratio.max().detach().cpu()) > max_scale_ratio:
                    continue
                best_idx = int(cand)
                best_dist = dist
            conf_i = float(confidence[src_idx].detach().cpu())
            weight_i = float(weight[src_idx].detach().cpu())
            if best_idx is not None:
                if best_idx < old_count:
                    old_weight = float(max(1.0e-6, meta_conf[best_idx].item()))
                    meta_conf[best_idx] = float(meta_conf[best_idx].item()) + weight_i
                    meta_obs[best_idx] = int(meta_obs[best_idx].item()) + 1
                    meta_seen[best_idx] = last_frame
                    meta_update[best_idx] = update_ord
                    meta_window[best_idx] = update_ord
                    meta_frame_start[best_idx] = first_frame
                    meta_frame_end[best_idx] = last_frame
                else:
                    rel = best_idx - old_count
                    old_weight = max(1.0e-6, insert_weight[rel])
                    insert_weight[rel] += weight_i
                    insert_conf[rel] += conf_i
                    insert_obs[rel] += 1
                alpha = weight_i / max(old_weight + weight_i, 1.0e-6)
                assign_existing(best_idx, mean_i, scale_i, rot_i, opacity_i, sh_i, alpha)
                fused += 1
                continue
            if inserted >= insert_budget:
                skipped_budget += 1
                continue
            global_idx = old_count + len(insert_xyz)
            insert_xyz.append(mean_i)
            insert_scale.append(scale_i)
            insert_rot.append(rot_i)
            insert_opacity.append(opacity_i)
            insert_sh.append(sh_i)
            insert_grid.append(key)
            insert_conf.append(conf_i)
            insert_weight.append(weight_i)
            insert_obs.append(1)
            occupied.setdefault(key, []).append(global_idx)
            inserted += 1

        if insert_xyz:
            xyz = torch.cat([xyz, torch.stack(insert_xyz, dim=0)], dim=0)
            rot = torch.cat([rot, torch.stack(insert_rot, dim=0)], dim=0)
            scale_actual = torch.cat([scale_actual, torch.stack(insert_scale, dim=0)], dim=0)
            opacity = torch.cat([opacity, torch.stack(insert_opacity, dim=0)], dim=0)
            old_sh = torch.cat([old_sh, torch.stack(insert_sh, dim=0)], dim=0)
            n_new = len(insert_xyz)
            meta_level = torch.cat([meta_level, torch.zeros(n_new, dtype=torch.int8)], dim=0)
            meta_voxel = torch.cat([meta_voxel, torch.full((n_new,), voxel_size, dtype=torch.float32)], dim=0)
            meta_grid = torch.cat([meta_grid, torch.tensor(insert_grid, dtype=torch.int32)], dim=0)
            meta_obs = torch.cat([meta_obs, torch.tensor(insert_obs, dtype=torch.int32)], dim=0)
            meta_conf = torch.cat([meta_conf, torch.tensor(insert_weight, dtype=torch.float32)], dim=0)
            meta_birth = torch.cat([meta_birth, torch.full((n_new,), last_frame, dtype=torch.int32)], dim=0)
            meta_seen = torch.cat([meta_seen, torch.full((n_new,), last_frame, dtype=torch.int32)], dim=0)
            meta_update = torch.cat([meta_update, torch.full((n_new,), update_ord, dtype=torch.int32)], dim=0)
            meta_window = torch.cat([meta_window, torch.full((n_new,), update_ord, dtype=torch.int32)], dim=0)
            meta_frame_start = torch.cat([meta_frame_start, torch.full((n_new,), first_frame, dtype=torch.int32)], dim=0)
            meta_frame_end = torch.cat([meta_frame_end, torch.full((n_new,), last_frame, dtype=torch.int32)], dim=0)
            meta_inlier = torch.cat([meta_inlier, torch.zeros(n_new, dtype=torch.int32)], dim=0)
            meta_outlier = torch.cat([meta_outlier, torch.zeros(n_new, dtype=torch.int32)], dim=0)
            meta_owner = torch.cat([meta_owner, torch.full((n_new,), update_ord, dtype=torch.int32)], dim=0)
            meta_quality = torch.cat([meta_quality, torch.tensor(insert_weight, dtype=torch.float32)], dim=0)
            meta_visibility = torch.cat([meta_visibility, torch.zeros(n_new, dtype=torch.int32)], dim=0)
            meta_render_error = torch.cat([meta_render_error, torch.zeros(n_new, dtype=torch.float32)], dim=0)

        self.xyz = nn.Parameter(xyz)
        self.rotation = nn.Parameter(rot)
        self.scaling = nn.Parameter(self._inverse_softplus_scale(scale_actual))
        self.opacity_logit = nn.Parameter(self._inv_sigmoid(opacity))
        dc_rgb = (old_sh[..., 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
        self.features = nn.Parameter(self._inv_sigmoid(dc_rgb))
        rest_dim = int(self.sh_rest.shape[1])
        if rest_dim > 0:
            self.sh_rest = nn.Parameter(old_sh[..., 1 : 1 + rest_dim].permute(0, 2, 1).contiguous())
        else:
            self.sh_rest = nn.Parameter(torch.zeros(int(xyz.shape[0]), 0, 3, device=device, dtype=dtype))

        self._anchor_level = meta_level
        self._anchor_voxel_size = meta_voxel
        self._anchor_grid_coord = meta_grid
        self._anchor_obs_count = meta_obs
        self._anchor_conf_accum = meta_conf
        self._anchor_birth_frame = meta_birth
        self._anchor_last_seen_kf = meta_seen
        self._anchor_last_update_kf_ord = meta_update
        self._anchor_source_window_id = meta_window
        self._anchor_source_frame_start = meta_frame_start
        self._anchor_source_frame_end = meta_frame_end
        self._anchor_inlier_obs = meta_inlier
        self._anchor_outlier_obs = meta_outlier
        self._anchor_owner_window_id = meta_owner
        self._anchor_quality = meta_quality
        self._anchor_visibility_count = meta_visibility
        self._anchor_render_error_ema = meta_render_error

        valid_count = int(keep.sum().detach().cpu())
        return {
            "requested": requested,
            "valid": valid_count,
            "fused": int(fused),
            "inserted": int(inserted),
            "skipped": int(requested - valid_count + skipped_budget),
            "skipped_budget": int(skipped_budget),
            "anchors_before": int(old_count),
            "anchors_after": self.anchor_count(),
        }

    def prune_anchors(self, prune_mask: torch.Tensor) -> int:
        if self.anchor_count() == 0:
            return 0
        mask = prune_mask.detach().to(device=self.xyz.device, dtype=torch.bool).view(-1)
        if int(mask.shape[0]) != self.anchor_count():
            raise ValueError(f"Prune mask length {int(mask.shape[0])} does not match anchor count {self.anchor_count()}")
        keep = ~mask
        n_pruned = int(mask.sum().item())
        if n_pruned <= 0:
            return 0
        self.xyz = nn.Parameter(self.xyz.detach()[keep])
        self.rotation = nn.Parameter(self.rotation.detach()[keep])
        self.scaling = nn.Parameter(self.scaling.detach()[keep])
        self.opacity_logit = nn.Parameter(self.opacity_logit.detach()[keep])
        self.features = nn.Parameter(self.features.detach()[keep])
        self.sh_rest = nn.Parameter(self.sh_rest.detach()[keep])
        keep_cpu = keep.detach().cpu()
        self._anchor_level = self._anchor_level[keep_cpu]
        self._anchor_voxel_size = self._anchor_voxel_size[keep_cpu]
        self._anchor_grid_coord = self._anchor_grid_coord[keep_cpu]
        self._anchor_obs_count = self._anchor_obs_count[keep_cpu]
        self._anchor_conf_accum = self._anchor_conf_accum[keep_cpu]
        self._anchor_birth_frame = self._anchor_birth_frame[keep_cpu]
        self._anchor_last_seen_kf = self._anchor_last_seen_kf[keep_cpu]
        self._anchor_last_update_kf_ord = self._anchor_last_update_kf_ord[keep_cpu]
        self._anchor_source_window_id = self._anchor_source_window_id[keep_cpu]
        self._anchor_source_frame_start = self._anchor_source_frame_start[keep_cpu]
        self._anchor_source_frame_end = self._anchor_source_frame_end[keep_cpu]
        self._anchor_inlier_obs = self._anchor_inlier_obs[keep_cpu]
        self._anchor_outlier_obs = self._anchor_outlier_obs[keep_cpu]
        self._anchor_owner_window_id = self._anchor_owner_window_id[keep_cpu]
        self._anchor_quality = self._anchor_quality[keep_cpu]
        self._anchor_visibility_count = self._anchor_visibility_count[keep_cpu]
        self._anchor_render_error_ema = self._anchor_render_error_ema[keep_cpu]
        return n_pruned

    def make_optimizer(self, *, lr: float = 2e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        param_groups = [{"params": self.gaussian_parameters(), "lr": float(lr), "name": "gaussians"}]
        sky_params = self.skybox_parameters()
        if sky_params:
            param_groups.append({"params": sky_params, "lr": self.skybox_lr, "name": "skybox"})
        return torch.optim.AdamW(param_groups, weight_decay=float(weight_decay))

    def initialize_skybox_from_image(
        self,
        image: torch.Tensor,
        c2w: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None = None,
        force: bool = False,
    ) -> bool:
        if not self.has_skybox:
            return False
        if self._skybox_initialized and not force:
            return False
        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] != 3:
            return False
        device = self.skybox_logits.device
        img = img.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        _, H, W = img.shape
        if sky_mask is None:
            sky = self._sky_mask_from_image(img)
        else:
            sky = sky_mask.detach().bool().to(device=device)
            if sky.ndim == 3:
                sky = sky[0]
            if tuple(sky.shape[-2:]) != (H, W):
                sky = F.interpolate(sky.float().view(1, 1, *sky.shape[-2:]), size=(H, W), mode="nearest")[0, 0] > 0.5
        if not bool(sky.any()):
            if not self.skybox_init_fallback_to_full_image:
                return False
            sky = torch.ones(H, W, device=device, dtype=torch.bool)
        dirs = self._world_dirs_for_erp(H, W, c2w.to(device=device, dtype=torch.float32))
        face, uv = self._directions_to_cubemap(dirs.reshape(-1, 3))
        sky_flat = sky.reshape(-1)
        img_flat = img.permute(1, 2, 0).reshape(-1, 3)
        selected = torch.nonzero(sky_flat, as_tuple=False).flatten()
        if selected.numel() == 0:
            return False
        face_sel = face[selected]
        uv_sel = uv[selected]
        s = int(self.skybox_resolution)
        ix = ((uv_sel[:, 0] + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        iy = ((uv_sel[:, 1] + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        linear = face_sel.long() * (s * s) + iy * s + ix
        accum = torch.zeros(6 * s * s, 3, device=device, dtype=torch.float32)
        counts = torch.zeros(6 * s * s, 1, device=device, dtype=torch.float32)
        accum.index_add_(0, linear, img_flat[selected])
        counts.index_add_(0, linear, torch.ones(selected.numel(), 1, device=device, dtype=torch.float32))
        mean_sky = img_flat[selected].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
        flat = mean_sky.expand(6 * s * s, 3).clone()
        filled = counts[:, 0] > 0
        flat[filled] = accum[filled] / counts[filled].clamp_min(1.0)
        faces = flat.view(6, s, s, 3).permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        with torch.no_grad():
            self.skybox_logits.copy_(self._inv_sigmoid(faces))
        self._skybox_initialized = True
        return True

    def sample_skybox(self, world_dirs: torch.Tensor) -> torch.Tensor:
        faces = self.get_skybox_faces
        if faces is None:
            raise RuntimeError("SkyBox is disabled.")
        original_shape = world_dirs.shape[:-1]
        dirs = torch.nn.functional.normalize(world_dirs.reshape(-1, 3).to(faces), dim=-1, eps=1e-8)
        face, uv = self._directions_to_cubemap(dirs)
        out = torch.zeros(dirs.shape[0], 3, device=faces.device, dtype=faces.dtype)
        for face_idx in range(6):
            mask = face == face_idx
            if not bool(mask.any()):
                continue
            grid = uv[mask].view(1, -1, 1, 2).to(device=faces.device, dtype=faces.dtype)
            sampled = F.grid_sample(
                faces[face_idx : face_idx + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            out[mask] = sampled[0, :, :, 0].T
        return out.view(*original_shape, 3)

    def skybox_erp_preview(self, *, height: int = 256, width: int = 512, c2w: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.has_skybox:
            return None
        pose = torch.eye(4, device=self.skybox_logits.device, dtype=torch.float32) if c2w is None else c2w
        dirs = self._world_dirs_for_erp(int(height), int(width), pose.to(device=self.skybox_logits.device, dtype=torch.float32))
        rgb = self.sample_skybox(dirs).permute(2, 0, 1).contiguous()
        return rgb.detach().cpu().clamp(0.0, 1.0)

    def _world_dirs_for_erp(self, H: int, W: int, c2w: torch.Tensor) -> torch.Tensor:
        grid = pixel_grid(H, W, device=c2w.device, dtype=c2w.dtype).view(-1, 2)
        dirs_cam = erp_pixel_to_bearing(grid, H, W).to(device=c2w.device, dtype=c2w.dtype)
        rot = c2w[:3, :3]
        dirs_world = (rot @ dirs_cam.T).T
        return dirs_world.view(H, W, 3)

    @staticmethod
    def _directions_to_cubemap(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dirs = torch.nn.functional.normalize(directions.float(), dim=-1, eps=1e-8)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        ax, ay, az = x.abs(), y.abs(), z.abs()
        face = torch.zeros(dirs.shape[0], device=dirs.device, dtype=torch.long)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        is_x = (ax >= ay) & (ax >= az)
        is_y = (ay > ax) & (ay >= az)
        is_z = ~(is_x | is_y)
        pos = is_x & (x >= 0)
        neg = is_x & (x < 0)
        face[pos] = 0
        u[pos] = -z[pos] / ax[pos].clamp_min(1e-8)
        v[pos] = -y[pos] / ax[pos].clamp_min(1e-8)
        face[neg] = 1
        u[neg] = z[neg] / ax[neg].clamp_min(1e-8)
        v[neg] = -y[neg] / ax[neg].clamp_min(1e-8)
        pos = is_y & (y >= 0)
        neg = is_y & (y < 0)
        face[pos] = 2
        u[pos] = x[pos] / ay[pos].clamp_min(1e-8)
        v[pos] = z[pos] / ay[pos].clamp_min(1e-8)
        face[neg] = 3
        u[neg] = x[neg] / ay[neg].clamp_min(1e-8)
        v[neg] = -z[neg] / ay[neg].clamp_min(1e-8)
        pos = is_z & (z >= 0)
        neg = is_z & (z < 0)
        face[pos] = 4
        u[pos] = x[pos] / az[pos].clamp_min(1e-8)
        v[pos] = -y[pos] / az[pos].clamp_min(1e-8)
        face[neg] = 5
        u[neg] = -x[neg] / az[neg].clamp_min(1e-8)
        v[neg] = -y[neg] / az[neg].clamp_min(1e-8)
        return face, torch.stack([u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)], dim=-1)

    def _sky_mask_from_image(self, image: torch.Tensor) -> torch.Tensor:
        cfg = self.config.get("SkyBox", {}) if isinstance(self.config, dict) else {}
        top_ratio = float(cfg.get("sky_mask_top_ratio", self.config.get("Mapping", {}).get("sky_mask_top_ratio", 0.58)))
        min_blue = float(cfg.get("sky_mask_min_blue", self.config.get("Mapping", {}).get("sky_mask_min_blue", 0.35)))
        blue_margin = float(cfg.get("sky_mask_blue_margin", self.config.get("Mapping", {}).get("sky_mask_blue_margin", 0.05)))
        cloud_brightness = float(
            cfg.get("sky_mask_cloud_brightness", self.config.get("Mapping", {}).get("sky_mask_cloud_brightness", 0.72))
        )
        cloud_saturation = float(
            cfg.get("sky_mask_cloud_saturation", self.config.get("Mapping", {}).get("sky_mask_cloud_saturation", 0.22))
        )
        texture_threshold = float(
            cfg.get("sky_mask_texture_threshold", self.config.get("Mapping", {}).get("sky_mask_texture_threshold", 0.08))
        )
        img = image.detach().float().clamp(0.0, 1.0)
        _, H, _ = img.shape
        rows = torch.arange(H, device=img.device, dtype=img.dtype).view(H, 1)
        upper = rows < float(H) * top_ratio
        r, g, b = img[0], img[1], img[2]
        max_rgb = img.max(dim=0).values
        min_rgb = img.min(dim=0).values
        saturation = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-6)
        blue_sky = (b >= min_blue) & (b >= r + blue_margin) & (b >= g + 0.5 * blue_margin)
        gray = img.mean(dim=0)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
        dy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
        low_texture = (dx + dy) <= texture_threshold
        cloud_sky = (max_rgb >= cloud_brightness) & (saturation <= cloud_saturation) & low_texture
        return upper & (blue_sky | cloud_sky)

    def save_ply(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self.get_xyz.detach().cpu().float().numpy()
        n = int(xyz.shape[0])
        normals = np.zeros_like(xyz, dtype=np.float32)
        rgb = self.get_features.detach().cpu().float().clamp(0.0, 1.0).numpy()
        f_dc = (rgb - 0.5) / 0.28209479177387814
        f_rest = np.zeros((n, 24), dtype=np.float32)
        opacity = self.opacity_logit.detach().cpu().float().numpy()
        scale = self.get_scaling.detach().cpu().float().clamp_min(1.0e-8).log().numpy()
        rot = self.get_rotation.detach().cpu().float().numpy()
        attributes = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            *(f"f_rest_{idx}" for idx in range(24)),
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        dtype = np.dtype([(name, "<f4") for name in attributes])
        elements = np.empty(n, dtype=dtype)
        values = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rot), axis=1).astype(np.float32, copy=False)
        for idx, name in enumerate(attributes):
            elements[name] = values[:, idx]
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header.extend(f"property float {name}" for name in attributes)
        header.append("end_header")
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            elements.tofile(f)
        return str(path)

    def save_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "anchor_level": self._anchor_level,
            "anchor_voxel_size": self._anchor_voxel_size,
            "anchor_grid_coord": self._anchor_grid_coord,
            "anchor_obs_count": self._anchor_obs_count,
            "anchor_conf_accum": self._anchor_conf_accum,
            "anchor_birth_frame": self._anchor_birth_frame,
            "anchor_last_seen_kf": self._anchor_last_seen_kf,
            "anchor_last_update_kf_ord": self._anchor_last_update_kf_ord,
            "anchor_inlier_obs": self._anchor_inlier_obs,
            "anchor_outlier_obs": self._anchor_outlier_obs,
            "anchor_owner_window_id": self._anchor_owner_window_id,
            "anchor_quality": self._anchor_quality,
            "anchor_visibility_count": self._anchor_visibility_count,
            "anchor_render_error_ema": self._anchor_render_error_ema,
            "anchor_depth_selected_levels": bool(
                self._anchor_depth_selected_levels
            ),
            "config": self.config,
        }
        if self._lazy_owner_transforms_enabled:
            payload["lazy_owner_transforms"] = self.lazy_owner_transform_state()
        torch.save(payload, path)
        return str(path)


class PanoGaussianMapper:
    """Keyframe-driven map insertion and optional render refinement."""

    def __init__(
        self,
        gaussian_map: PanoGaussianMap,
        *,
        renderer: PFGS360Renderer | None = None,
        lr: float = 2e-3,
        loss_weights: BackendLossWeights | None = None,
    ) -> None:
        self.map = gaussian_map
        self.renderer = renderer or PFGS360Renderer(config=gaussian_map.config)
        self.optimizer = gaussian_map.make_optimizer(lr=lr)
        if loss_weights is None:
            sky_cfg = gaussian_map.config.get("SkyBox", {}) if isinstance(gaussian_map.config, dict) else {}
            self.loss_weights = BackendLossWeights(
                sky_alpha=float(sky_cfg.get("sky_alpha_loss_weight", 0.0)),
            )
        else:
            self.loss_weights = loss_weights
        self.stats = MapperStats()
        self.optim_cfg = gaussian_map.config.get("BackendOptimization", {}) if isinstance(gaussian_map.config, dict) else {}
        mapping_cfg = gaussian_map.config.get("Mapping", {}) if isinstance(gaussian_map.config, dict) else {}
        novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
        pano_cfg = gaussian_map.config.get("PanoVGGT", {}) if isinstance(gaussian_map.config, dict) else {}
        m3_enabled = bool((pano_cfg.get("M3Sphere", {}) or {}).get("enabled", False)) if isinstance(pano_cfg, dict) else False
        self.novel_insertion_enabled = bool(novel_cfg.get("enabled", m3_enabled))
        self.novel_insertion_strategy = str(novel_cfg.get("strategy", "legacy") or "legacy").lower()
        self.pfgs360_replace_fuse_enabled = bool(
            self.novel_insertion_enabled and self.novel_insertion_strategy == "pfgs360_replace_fuse"
        )
        self.neural_anchor_mode = str(getattr(gaussian_map, "map_mode", "")).lower() == "neural_anchor_scaffold_panorama"
        if (
            self.neural_anchor_mode
            and self.pfgs360_replace_fuse_enabled
        ):
            raise ValueError("neural_anchor_scaffold_panorama does not support pfgs360_replace_fuse in this backend.")
        self.pfgs360_insertion_enabled = bool(
            self.novel_insertion_enabled and self.novel_insertion_strategy in {"pfgs360", "pfgs360_replace_fuse"}
        )
        self.sky_mask_source = str(mapping_cfg.get("sky_mask_source", "heuristic") or "heuristic").lower()
        self.pfgs360_voxel_size = max(float(novel_cfg.get("voxel_size", 0.12)), 1.0e-6)
        self.replace_fuse_delete_rel_min = float(novel_cfg.get("replace_delete_rel_min", 0.20))
        self.replace_fuse_delete_rel_max = float(novel_cfg.get("replace_delete_rel_max", 0.30))
        self.replace_fuse_insert_rel_min = float(novel_cfg.get("replace_insert_rel_min", self.replace_fuse_delete_rel_min))
        self.replace_fuse_front_depth_abs_tol = float(novel_cfg.get("replace_front_depth_abs_tol", 0.03))
        self.replace_fuse_front_depth_rel_tol = float(novel_cfg.get("replace_front_depth_rel_tol", 0.02))
        self.replace_fuse_max_delete_per_keyframe = max(0, int(novel_cfg.get("max_replace_delete_per_keyframe", 30000)))
        self.replace_fuse_compact_voxels = bool(novel_cfg.get("compact_voxels", True))
        self.replace_fuse_sky_prune_enabled = bool(novel_cfg.get("sky_prune_enabled", True))
        self.replace_fuse_insert_occupancy_radius_voxels = max(
            0.0,
            float(novel_cfg.get("insert_occupancy_radius_voxels", 2.0)),
        )
        self.pfgs360_render_alpha_min = float(novel_cfg.get("render_alpha_min", 0.20))
        self.pfgs360_missing_alpha_min = float(novel_cfg.get("missing_alpha_min", self.pfgs360_render_alpha_min))
        self.pfgs360_render_depth_rel_threshold = float(novel_cfg.get("render_depth_rel_threshold", 0.10))
        self.pfgs360_foreground_rel_threshold = float(novel_cfg.get("foreground_rel_threshold", 0.10))
        self.pfgs360_photometric_error_threshold = float(novel_cfg.get("photometric_error_threshold", 0.08))
        self.pfgs360_near_grid_radius = max(0, int(novel_cfg.get("near_grid_radius", 1)))
        self.pfgs360_near_distance_factor = max(0.0, float(novel_cfg.get("near_distance_factor", 1.0)))
        self.pfgs360_gaussian_scale_mode = str(novel_cfg.get("gaussian_scale_mode", "voxel") or "voxel").lower()
        self.pfgs360_gaussian_scale_factor = float(novel_cfg.get("gaussian_scale_factor", 1.25))
        self.pfgs360_gaussian_scale_min = max(float(novel_cfg.get("gaussian_scale_min", 0.008)), 1.0e-8)
        self.pfgs360_gaussian_scale_max = max(
            self.pfgs360_gaussian_scale_min,
            float(novel_cfg.get("gaussian_scale_max", 0.08)),
        )
        self.pfgs360_gaussian_scale_lat_cos_min = min(
            1.0,
            max(0.0, float(novel_cfg.get("gaussian_scale_lat_cos_min", 0.25))),
        )
        default_reset = 0 if self.pfgs360_replace_fuse_enabled else 3
        default_prune = 0 if self.pfgs360_replace_fuse_enabled else 6
        self.pfgs360_reset_after_outliers = max(0, int(novel_cfg.get("reset_after_outlier_observations", default_reset)))
        self.pfgs360_prune_after_outliers = max(0, int(novel_cfg.get("prune_after_outlier_observations", default_prune)))
        self.pfgs360_protect_recent_keyframes = max(0, int(novel_cfg.get("protect_recent_keyframes", 8)))
        self.pfgs360_max_prune_per_keyframe = max(0, int(novel_cfg.get("max_prune_per_keyframe", 500)))
        self.first_keyframe_max_seeds = int(novel_cfg.get("first_keyframe_max_seeds", 80000))
        self.keyframe_max_seeds = int(novel_cfg.get("keyframe_max_seeds", 30000))
        self.max_missing_seeds_per_keyframe = int(novel_cfg.get("max_missing_seeds_per_keyframe", 0))
        self.max_depth_mismatch_seeds_per_keyframe = int(
            novel_cfg.get("max_depth_mismatch_seeds_per_keyframe", 0)
        )
        self.prioritize_depth_mismatch = bool(novel_cfg.get("prioritize_depth_mismatch", True))
        self.global_anchor_budget = int(novel_cfg.get("global_anchor_budget", 1500000))
        self.first_keyframe_voxel_neighbor_radius = max(
            0,
            int(novel_cfg.get("first_keyframe_voxel_neighbor_radius", novel_cfg.get("voxel_neighbor_radius", 0))),
        )
        self.voxel_neighbor_radius = max(0, int(novel_cfg.get("voxel_neighbor_radius", 0)))
        self.keyframes: list[MapperKeyframe] = []
        self.observations: dict[int, MapperObservation] = {}
        self.pose_deltas: dict[int, PoseDelta] = {}
        self._spherical_selfi_rollback_state: tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]] | None = None
        self.last_inserted_range: tuple[int, int] = (0, 0)
        self.last_requested_source_flat_idx: torch.Tensor | None = None
        self.last_inserted_source_flat_idx: torch.Tensor | None = None
        self.last_source_hw: tuple[int, int] | None = None
        self.last_depth_insertion_diagnostic: DepthInsertionDiagnostic | None = None
        self.frontend_graph_window_ids: tuple[int, ...] = ()
        self._neural_first_chunk_optimized = False
        if loss_weights is None and self.pfgs360_replace_fuse_enabled:
            sky_cfg = gaussian_map.config.get("SkyBox", {}) if isinstance(gaussian_map.config, dict) else {}
            self.loss_weights = BackendLossWeights(
                depth=float(self.optim_cfg.get("depth_loss_weight", 0.03)),
                opacity=float(self.optim_cfg.get("opacity_loss_weight", 0.0)),
                sky_alpha=float(sky_cfg.get("sky_alpha_loss_weight", 0.0)),
                photometric_mode=str(self.optim_cfg.get("photometric_loss_mode", "l1_dssim")),
                rgb_l1_weight=float(self.optim_cfg.get("rgb_l1_weight", 0.8)),
                dssim_weight=float(self.optim_cfg.get("dssim_weight", 0.2)),
                depth_loss_mode=str(self.optim_cfg.get("depth_loss_mode", "relative_clamped")),
                depth_residual_clamp=float(self.optim_cfg.get("depth_residual_clamp", 0.20)),
            )

    @property
    def feedforward_window_enabled(self) -> bool:
        cfg = self._feedforward_window_cfg()
        return bool(cfg.get("enabled", False)) or bool(self.optim_cfg.get("optimize_after_every_chunk", False))

    def _feedforward_window_cfg(self) -> dict:
        cfg = self.optim_cfg.get("FeedForwardWindow", {}) if isinstance(self.optim_cfg, dict) else {}
        return cfg if isinstance(cfg, dict) else {}

    @property
    def uses_joint_optimization(self) -> bool:
        cfg = self.optim_cfg
        return bool(cfg.get("enabled", False)) or bool(cfg.get("pose_refine_enable", False))

    def insert_keyframe(
        self,
        seeds: GaussianSeedBatch,
        frontend_output: FrontendOutput,
        image: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
        insert_occupancy_radius_voxels_override: float | None = None,
        compact_after_insert: bool = False,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> int:
        requested = len(seeds)
        current_kf_ord = int(self.stats.n_keyframes)
        self.last_requested_source_flat_idx = (
            None if seeds.source_flat_idx is None else seeds.source_flat_idx.detach().cpu().long()
        )
        self.last_source_hw = seeds.source_hw
        self.last_depth_insertion_diagnostic = None
        sky_mask = self._resolve_input_sky_mask(
            image,
            sky_mask,
            context=f"frame {int(frontend_output.frame_id)} keyframe insertion",
        )
        if image is not None and self.map.has_skybox:
            self.map.initialize_skybox_from_image(image, frontend_output.pose_c2w, sky_mask=sky_mask)
        start = self.map.anchor_count()
        if self.neural_anchor_mode and hasattr(self.map, "insert_from_frontend_output") and image is not None:
            n = self.map.insert_from_frontend_output(
                frontend_output,
                image,
                sky_mask=sky_mask,
                last_update_kf_ord=current_kf_ord,
            )
            self.last_inserted_source_flat_idx = getattr(self.map, "last_inserted_source_flat_idx", None)
            self.last_source_hw = getattr(self.map, "last_source_hw", None) or self.last_source_hw
            filter_stats = {
                "skipped_voxel": 0,
                "skipped_budget": 0,
                "dense_seed_candidates": int(getattr(self.map, "last_candidate_count", requested)),
                "newly_inserted": int(n),
                "compacted": int(getattr(self.map, "last_compacted_anchors", 0)),
                "neural_insert_total_sec": float(getattr(self.map, "last_insert_total_sec", 0.0)),
                "neural_insert_accept_sec": float(getattr(self.map, "last_insert_accept_sec", 0.0)),
                "neural_insert_append_sec": float(getattr(self.map, "last_insert_append_sec", 0.0)),
                "neural_insert_compact_sec": float(getattr(self.map, "last_insert_compact_sec", 0.0)),
            }
        else:
            seeds, filter_stats = self._filter_novel_seeds(
                seeds,
                frontend_output=frontend_output,
                image=image,
                sky_mask=sky_mask,
                insert_occupancy_radius_voxels_override=insert_occupancy_radius_voxels_override,
                replace_delete_keyframe_ids=replace_delete_keyframe_ids,
            )
            self.last_inserted_source_flat_idx = (
                None if seeds.source_flat_idx is None else seeds.source_flat_idx.detach().cpu().long()
            )
            if self.last_source_hw is None:
                self.last_source_hw = seeds.source_hw
            n = self.map.add_seeds(
                seeds,
                voxel_size=self.pfgs360_voxel_size if self.pfgs360_replace_fuse_enabled else None,
                last_update_kf_ord=current_kf_ord if self.pfgs360_replace_fuse_enabled else None,
            )
        end = self.map.anchor_count()
        self.last_inserted_range = (start, end)
        filter_stats["newly_inserted"] = int(n)
        if self.pfgs360_replace_fuse_enabled:
            compacted = self._refresh_pfgs360_voxel_cache(compact=bool(compact_after_insert))
            filter_stats["compacted"] = int(filter_stats.get("compacted", 0)) + int(compacted)
        filter_stats["anchors_before"] = int(start)
        filter_stats["anchors_after"] = int(self.map.anchor_count())
        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        self.stats.n_keyframes += 1
        self.stats.n_anchors = self.map.anchor_count()
        self.stats.last_inserted_anchors = int(n)
        self.stats.last_skipped_voxel = int(filter_stats.get("skipped_voxel", 0))
        self.stats.last_skipped_budget = int(filter_stats.get("skipped_budget", 0))
        self.stats.last_hash_hits = int(filter_stats.get("hash_hits", 0))
        self.stats.last_hash_near_hits = int(filter_stats.get("hash_near_hits", 0))
        self.stats.last_suppressed_insert = int(filter_stats.get("suppressed_insert", 0))
        self.stats.last_outlier_resets = int(filter_stats.get("outlier_resets", 0))
        self.stats.last_outlier_pruned = int(filter_stats.get("outlier_pruned", 0))
        self.stats.last_render_missing_pixels = int(filter_stats.get("missing_pixels", 0))
        self.stats.last_render_depth_mismatch_pixels = int(filter_stats.get("depth_mismatch_pixels", 0))
        self.stats.last_render_bad_pixels = int(filter_stats.get("render_bad_pixels", 0))
        self.stats.last_missing_seed_candidates = int(filter_stats.get("missing_seed_candidates", 0))
        self.stats.last_depth_mismatch_seed_candidates = int(
            filter_stats.get("depth_mismatch_seed_candidates", 0)
        )
        self.stats.last_skipped_missing_budget = int(filter_stats.get("skipped_missing_budget", 0))
        self.stats.last_skipped_depth_mismatch_budget = int(
            filter_stats.get("skipped_depth_mismatch_budget", 0)
        )
        self.stats.last_replace_deleted = int(filter_stats.get("replace_deleted", 0))
        self.stats.last_replace_fused = int(filter_stats.get("fused", 0))
        self.stats.last_replace_compacted = int(filter_stats.get("compacted", 0))
        self.stats.last_dense_seed_candidates = int(filter_stats.get("dense_seed_candidates", requested))
        self.stats.last_insert_mask_seed_candidates = int(filter_stats.get("insert_mask_seed_candidates", 0))
        self.stats.last_voxel_seed_candidates = int(filter_stats.get("voxel_seed_candidates", 0))
        self.stats.last_replace_fused_existing = int(filter_stats.get("fused_existing", 0))
        self.stats.last_replace_fused_new_duplicate = int(filter_stats.get("fused_new_duplicate", 0))
        self.stats.last_replace_newly_inserted = int(filter_stats.get("newly_inserted", n))
        self.stats.last_pred_depth_generated_seeds = int(filter_stats.get("pred_depth_generated_seeds", 0))
        self.stats.last_pred_depth_invalid_pixels = int(filter_stats.get("pred_depth_invalid_pixels", 0))
        self.stats.last_insert_mask_pixels = int(filter_stats.get("insert_mask_pixels", 0))
        self.stats.last_anchor_count_before_insert = int(filter_stats.get("anchors_before", start))
        self.stats.last_anchor_count_after_insert = int(filter_stats.get("anchors_after", self.map.anchor_count()))
        self.stats.last_neural_insert_total_sec = float(filter_stats.get("neural_insert_total_sec", 0.0))
        self.stats.last_neural_insert_accept_sec = float(filter_stats.get("neural_insert_accept_sec", 0.0))
        self.stats.last_neural_insert_append_sec = float(filter_stats.get("neural_insert_append_sec", 0.0))
        self.stats.last_neural_insert_compact_sec = float(filter_stats.get("neural_insert_compact_sec", 0.0))
        if image is not None:
            register_start, register_end = self.last_inserted_range
            self._register_keyframe(frontend_output, image, start=register_start, end=register_end, sky_mask=sky_mask)
            self.register_observation(frontend_output, image, is_keyframe=True, sky_mask=sky_mask)
        if n == 0:
            self.stats.notes.append(f"frame {frontend_output.frame_id}: no seeds inserted")
        if self.novel_insertion_enabled and requested != n:
            self.stats.notes.append(
                (
                    f"frame {frontend_output.frame_id}: novel insertion kept {n}/{requested} "
                    f"seeds, skipped_voxel={self.stats.last_skipped_voxel}, "
                    f"skipped_budget={self.stats.last_skipped_budget}"
                )
            )
        if self.pfgs360_insertion_enabled and (
            self.stats.last_suppressed_insert or self.stats.last_outlier_resets or self.stats.last_outlier_pruned
        ):
            self.stats.notes.append(
                (
                    f"frame {frontend_output.frame_id}: pfgs360 insertion "
                    f"hits={self.stats.last_hash_hits}, near_hits={self.stats.last_hash_near_hits}, "
                    f"suppressed={self.stats.last_suppressed_insert}, "
                    f"resets={self.stats.last_outlier_resets}, pruned={self.stats.last_outlier_pruned}"
                )
            )
        return n

    def fuse_resplat_state(
        self,
        state,
        *,
        frame_ids: list[int] | tuple[int, ...],
        config: dict | None = None,
        window_id: int | None = None,
    ) -> dict[str, int | float]:
        """Fuse a local ReSplat ``PanoGaussianState`` directly into the global map."""

        ids = [int(fid) for fid in frame_ids]
        start = self.map.anchor_count()
        stats = self.map.add_or_fuse_resplat_gaussians(
            state,
            frame_ids=ids,
            window_id=window_id,
            config=config,
        )
        end = self.map.anchor_count()
        self.last_inserted_range = (start, end)
        base_lr = float(self.optimizer.param_groups[0].get("lr", 2e-3)) if self.optimizer.param_groups else 2e-3
        self.optimizer = self.map.make_optimizer(lr=base_lr)

        for fid in ids:
            obs = self.observations.get(int(fid))
            if obs is None:
                continue
            device = self.map.get_xyz.device
            dtype = self.map.get_xyz.dtype
            self.pose_deltas[int(fid)] = PoseDelta(obs.pose_c2w.to(device=device, dtype=dtype)).to(device=device)
            obs.is_keyframe = True
            record = MapperKeyframe(
                frame_id=int(fid),
                image=obs.image.detach().cpu().float(),
                gaussian_start=start,
                gaussian_end=end,
                sky_mask=None if obs.sky_mask is None else obs.sky_mask.detach().cpu().bool(),
                target_depth=None if obs.target_depth is None else obs.target_depth.detach().cpu().float(),
                depth_confidence=None if obs.depth_confidence is None else obs.depth_confidence.detach().cpu().float(),
            )
            self.keyframes = [kf for kf in self.keyframes if int(kf.frame_id) != int(fid)]
            self.keyframes.append(record)
        self.keyframes.sort(key=lambda kf: int(kf.frame_id))
        self.stats.n_keyframes = int(len(self.keyframes))
        self.stats.n_anchors = self.map.anchor_count()
        self.stats.last_inserted_anchors = int(stats.get("inserted", 0))
        self.stats.last_replace_fused = int(stats.get("fused", 0))
        self.stats.last_replace_newly_inserted = int(stats.get("inserted", 0))
        self.stats.last_anchor_count_before_insert = int(stats.get("anchors_before", start))
        self.stats.last_anchor_count_after_insert = int(stats.get("anchors_after", end))
        self.stats.last_resplat_fused = int(stats.get("fused", 0))
        self.stats.last_resplat_inserted = int(stats.get("inserted", 0))
        self.stats.last_resplat_skipped = int(stats.get("skipped", 0))
        return dict(stats)

    def optimize_resplat_global_window(
        self,
        *,
        frame_ids: list[int] | tuple[int, ...],
        iters: int = 20,
        settings: dict | None = None,
    ) -> dict[str, float]:
        """Run Gaussian-only global refinement after one ReSplat local window."""

        steps = max(0, int(iters))
        ids = [int(fid) for fid in frame_ids]
        if steps <= 0 or not ids:
            return {"loss": 0.0, "steps": 0.0, "resplat_global_steps": 0.0}
        resplat_cfg = self.optim_cfg.get("ReSplatGlobal", {}) if isinstance(self.optim_cfg, dict) else {}
        if not isinstance(resplat_cfg, dict):
            resplat_cfg = {}
        effective_cfg = dict(resplat_cfg)
        effective_cfg.update(dict(settings or {}))
        old_enabled = self.optim_cfg.get("enabled", None)
        old_pose = self.optim_cfg.get("pose_refine_enable", None)
        old_ff = self.optim_cfg.get("FeedForwardWindow", None)
        old_early = self.optim_cfg.get("early_stop", None)
        old_opt_after_chunk = self.optim_cfg.get("optimize_after_every_chunk", None)
        overridden_root = {
            key: self.optim_cfg.get(key, None)
            for key in (
                "pose_lr",
                "pose_prior_weight",
                "pose_grad_clip",
                "gaussian_lr",
                "separate_gaussian_lrs",
                "xyz_lr",
                "feature_lr",
                "sh_rest_lr",
                "opacity_lr",
                "scaling_lr",
                "rotation_lr",
                "scale_gaussian_parameter_updates",
                "fixed_pose_frame_ids",
            )
        }
        ff_cfg = dict(old_ff) if isinstance(old_ff, dict) else {}
        ff_cfg["enabled"] = True
        ff_cfg["steps"] = steps
        ff_cfg["gaussian_scope"] = str(effective_cfg.get("gaussian_scope", "all"))
        if "active_owner_window_id" in effective_cfg:
            ff_cfg["active_owner_window_id"] = int(effective_cfg["active_owner_window_id"])
        ff_cfg["visible_neighbor_lr_scale"] = float(effective_cfg.get("visible_neighbor_lr_scale", 0.1))
        ff_cfg["optimize_non_keyframe_observations"] = True
        ff_cfg["random_observation_per_iter"] = bool(effective_cfg.get("random_observation_per_iter", False))
        ff_cfg["sample_observations_per_step"] = int(effective_cfg.get("sample_observations_per_step", len(ids)))
        ff_cfg["sampler"] = str(effective_cfg.get("sampler", "random"))
        ff_cfg["sampler_seed"] = int(effective_cfg.get("sampler_seed", 123))
        ff_cfg["skip_prune"] = bool(effective_cfg.get("skip_prune", False))
        self.optim_cfg["enabled"] = True
        self.optim_cfg["pose_refine_enable"] = bool(effective_cfg.get("pose_refine_enable", False))
        for key in overridden_root:
            if key in effective_cfg:
                self.optim_cfg[key] = effective_cfg[key]
        self.optim_cfg["optimize_after_every_chunk"] = False
        self.optim_cfg["FeedForwardWindow"] = ff_cfg
        self.optim_cfg["early_stop"] = {"enabled": False}
        try:
            metrics = self.optimize_feedforward_window(
                current_frame_ids=ids,
                history_frame_ids=[],
                chunk_index=None,
                active_keyframe_ids=ids,
            )
        finally:
            if old_enabled is None:
                self.optim_cfg.pop("enabled", None)
            else:
                self.optim_cfg["enabled"] = old_enabled
            if old_pose is None:
                self.optim_cfg.pop("pose_refine_enable", None)
            else:
                self.optim_cfg["pose_refine_enable"] = old_pose
            if old_ff is None:
                self.optim_cfg.pop("FeedForwardWindow", None)
            else:
                self.optim_cfg["FeedForwardWindow"] = old_ff
            if old_early is None:
                self.optim_cfg.pop("early_stop", None)
            else:
                self.optim_cfg["early_stop"] = old_early
            if old_opt_after_chunk is None:
                self.optim_cfg.pop("optimize_after_every_chunk", None)
            else:
                self.optim_cfg["optimize_after_every_chunk"] = old_opt_after_chunk
            for key, value in overridden_root.items():
                if value is None:
                    self.optim_cfg.pop(key, None)
                else:
                    self.optim_cfg[key] = value
        out = dict(metrics)
        out["resplat_global_steps"] = float(out.get("steps", 0.0))
        out["resplat_global_configured_steps"] = float(steps)
        return out

    def optimize_spherical_selfi_window(
        self,
        *,
        window_id: int,
        frame_ids: list[int] | tuple[int, ...],
        iters: int = 20,
        settings: dict | None = None,
        extra_loss_fn=None,
    ) -> dict[str, float]:
        """Jointly refine one spherical-Selfi owner window and its SE(3) poses."""

        cfg = dict(settings or {})
        cfg.update(
            {
                "gaussian_scope": "owner_window_visible",
                "active_owner_window_id": int(window_id),
                "pose_refine_enable": bool(cfg.get("pose_refine_enable", True)),
                "random_observation_per_iter": True,
                "sample_observations_per_step": 1,
                "sampler": "shuffled_cycle",
                "skip_prune": True,
            }
        )
        parameter_snapshot = {
            name: value.detach().clone()
            for name, value in self.map.named_parameters()
        }
        pose_snapshot = {
            int(frame_id): self.pose_deltas[int(frame_id)].delta.detach().clone()
            for frame_id in frame_ids
            if int(frame_id) in self.pose_deltas
        }
        self._spherical_selfi_rollback_state = (parameter_snapshot, pose_snapshot)
        previous_extra_loss_fn = getattr(self, "_spherical_selfi_extra_loss_fn", None)
        self._spherical_selfi_extra_loss_fn = extra_loss_fn
        try:
            metrics = self.optimize_resplat_global_window(
                frame_ids=frame_ids,
                iters=iters,
                settings=cfg,
            )
        finally:
            self._spherical_selfi_extra_loss_fn = previous_extra_loss_fn
        if float(metrics.get("non_finite_window", 0.0)) > 0.0:
            self.rollback_spherical_selfi_window()
            metrics["steps"] = 0.0
            metrics["window_rollback"] = 1.0
        metrics["spherical_selfi_window_id"] = float(window_id)
        return metrics

    def commit_spherical_selfi_window(self) -> None:
        self._spherical_selfi_rollback_state = None

    def rollback_spherical_selfi_window(self) -> bool:
        state = self._spherical_selfi_rollback_state
        self._spherical_selfi_rollback_state = None
        if state is None:
            return False
        parameter_snapshot, pose_snapshot = state
        with torch.no_grad():
            for name, value in self.map.named_parameters():
                saved = parameter_snapshot.get(name)
                if saved is not None and tuple(saved.shape) == tuple(value.shape):
                    value.copy_(saved.to(value))
            for frame_id, saved in pose_snapshot.items():
                if frame_id in self.pose_deltas:
                    self.pose_deltas[frame_id].delta.copy_(saved.to(self.pose_deltas[frame_id].delta))
        lr = float(self.optimizer.param_groups[0].get("lr", 2.0e-3)) if self.optimizer.param_groups else 2.0e-3
        self.optimizer = self.map.make_optimizer(lr=lr)
        return True

    def prepare_spherical_selfi_window(self, frame_ids: list[int] | tuple[int, ...]) -> int:
        """Promote registered RGB/depth observations without inserting legacy seeds."""

        prepared = 0
        device, dtype = self.map.get_xyz.device, self.map.get_xyz.dtype
        for frame_id in (int(value) for value in frame_ids):
            observation = self.observations.get(frame_id)
            if observation is None:
                continue
            if self.map.has_skybox and not bool(getattr(self.map, "_skybox_initialized", False)):
                self.map.initialize_skybox_from_image(
                    observation.image,
                    observation.pose_c2w,
                    sky_mask=observation.sky_mask,
                )
            # The overlap frame must reuse its unique PoseDelta rather than
            # being reinitialized when the next window arrives.
            if frame_id not in self.pose_deltas:
                self.pose_deltas[frame_id] = PoseDelta(
                    observation.pose_c2w.to(device=device, dtype=dtype)
                ).to(device=device)
            observation.is_keyframe = True
            record = MapperKeyframe(
                frame_id=frame_id,
                image=observation.image.detach().cpu().float(),
                gaussian_start=0,
                gaussian_end=self.map.anchor_count(),
                sky_mask=None if observation.sky_mask is None else observation.sky_mask.detach().cpu().bool(),
                target_depth=None if observation.target_depth is None else observation.target_depth.detach().cpu().float(),
                depth_confidence=None if observation.depth_confidence is None else observation.depth_confidence.detach().cpu().float(),
            )
            self.keyframes = [value for value in self.keyframes if int(value.frame_id) != frame_id]
            self.keyframes.append(record)
            prepared += 1
        self.keyframes.sort(key=lambda value: int(value.frame_id))
        self.stats.n_keyframes = len(self.keyframes)
        return prepared

    def set_frontend_graph_window_ids(self, frame_ids) -> None:
        ids: list[int] = []
        if frame_ids is None:
            self.frontend_graph_window_ids = ()
            return
        for frame_id in frame_ids:
            try:
                value = int(frame_id)
            except (TypeError, ValueError):
                continue
            if value not in ids:
                ids.append(value)
        self.frontend_graph_window_ids = tuple(ids)

    def register_observation(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        is_keyframe: bool | None = None,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        self.register_observation_values(
            frame_id=int(frontend_output.frame_id),
            image=image,
            c2w=frontend_output.pose_c2w,
            inverse_depth=frontend_output.inverse_depth,
            depth_confidence=frontend_output.depth_confidence,
            world_points=frontend_output.world_points,
            world_points_confidence=frontend_output.world_points_confidence,
            is_keyframe=bool(frontend_output.is_keyframe) if is_keyframe is None else bool(is_keyframe),
            sky_mask=sky_mask,
        )

    def register_observation_values(
        self,
        *,
        frame_id: int,
        image: torch.Tensor,
        c2w: torch.Tensor,
        inverse_depth: torch.Tensor | None = None,
        depth_confidence: torch.Tensor | None = None,
        world_points: torch.Tensor | None = None,
        world_points_confidence: torch.Tensor | None = None,
        is_keyframe: bool = False,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        img = image.detach().cpu().float()
        if img.ndim == 4 and int(img.shape[0]) == 1:
            img = img[0]
        if img.ndim != 3:
            return
        frame_id = int(frame_id)
        sky = self._resolve_input_sky_mask(
            img,
            sky_mask,
            context=f"frame {int(frame_id)} observation registration",
        )
        depth, conf = self._target_depth_from_tensors(
            inverse_depth=inverse_depth,
            world_points=world_points,
            pose_c2w=c2w,
            confidence=depth_confidence if depth_confidence is not None else world_points_confidence,
            size=(int(img.shape[-2]), int(img.shape[-1])),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if depth is not None and sky is not None:
            if conf is None:
                conf = torch.ones_like(depth)
            sky_depth_mask = self._normalize_skybox_mask(
                sky,
                height=int(img.shape[-2]),
                width=int(img.shape[-1]),
                device=torch.device("cpu"),
            )
            conf = conf.masked_fill(sky_depth_mask.bool(), 0.0)
        self.observations[frame_id] = MapperObservation(
            frame_id=frame_id,
            image=img,
            pose_c2w=c2w.detach().cpu().float(),
            is_keyframe=bool(is_keyframe),
            sky_mask=sky.detach().cpu().bool() if torch.is_tensor(sky) else None,
            target_depth=None if depth is None else depth.detach().cpu().float(),
            depth_confidence=None if conf is None else conf.detach().cpu().float(),
        )

    def _filter_novel_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None = None,
        image: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
        insert_occupancy_radius_voxels_override: float | None = None,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        if not self.novel_insertion_enabled:
            return seeds, {"skipped_voxel": 0, "skipped_budget": 0}
        if len(seeds) == 0 and not self.pfgs360_replace_fuse_enabled:
            return seeds, {"skipped_voxel": 0, "skipped_budget": 0}
        if self.pfgs360_insertion_enabled:
            return self._filter_pfgs360_seeds(
                seeds,
                frontend_output=frontend_output,
                image=image,
                sky_mask=sky_mask,
                insert_occupancy_radius_voxels_override=insert_occupancy_radius_voxels_override,
                replace_delete_keyframe_ids=replace_delete_keyframe_ids,
            )
        per_keyframe_budget = self.first_keyframe_max_seeds if self.stats.n_keyframes == 0 else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))

        xyz_cpu = seeds.xyz.detach().cpu().float()
        scale_cpu = seeds.scale.detach().cpu().float().clamp_min(1.0e-6)
        level_cpu = seeds.level.detach().cpu()
        conf_cpu = seeds.confidence.detach().cpu().float()
        order = torch.argsort(conf_cpu, descending=True)
        occupied = self._build_voxel_index()
        neighbor_radius = (
            self.first_keyframe_voxel_neighbor_radius if self.stats.n_keyframes == 0 else self.voxel_neighbor_radius
        )
        kept: list[int] = []
        skipped_voxel = 0
        skipped_budget = 0
        for seed_idx in order.tolist():
            key = self._seed_voxel_key_from_cpu(xyz_cpu, scale_cpu, level_cpu, int(seed_idx))
            hit = self._find_voxel_hit(occupied, key, radius=neighbor_radius)
            if hit is not None:
                self._accumulate_existing_observation(hit, float(conf_cpu[seed_idx]))
                skipped_voxel += 1
                continue
            if len(kept) >= budget:
                skipped_budget += 1
                continue
            kept.append(int(seed_idx))
            occupied[key] = -1
        if not kept:
            return self._empty_seed_like(seeds), {"skipped_voxel": skipped_voxel, "skipped_budget": skipped_budget}
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), {"skipped_voxel": skipped_voxel, "skipped_budget": skipped_budget}

    def _filter_pfgs360_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None,
        image: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
        insert_occupancy_radius_voxels_override: float | None = None,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        if self.pfgs360_replace_fuse_enabled:
            return self._filter_replace_fuse_seeds(
                seeds,
                frontend_output=frontend_output,
                image=image,
                sky_mask=sky_mask,
                insert_occupancy_radius_voxels_override=insert_occupancy_radius_voxels_override,
                replace_delete_keyframe_ids=replace_delete_keyframe_ids,
            )
        seeds = self._with_pfgs360_seed_metadata(seeds)
        per_keyframe_budget = self.first_keyframe_max_seeds if self.stats.n_keyframes == 0 else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))

        stats = {
            "skipped_voxel": 0,
            "skipped_budget": 0,
            "hash_hits": 0,
            "hash_near_hits": 0,
            "suppressed_insert": 0,
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
            "missing_seed_candidates": 0,
            "depth_mismatch_seed_candidates": 0,
            "skipped_missing_budget": 0,
            "skipped_depth_mismatch_budget": 0,
        }
        insert_enabled = (
            torch.ones(len(seeds), dtype=torch.bool)
            if seeds.insert_enabled is None
            else seeds.insert_enabled.detach().cpu().bool()
        )
        depth_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        missing_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        first_keyframe = self.stats.n_keyframes == 0 or self.map.anchor_count() == 0
        if first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            self.last_depth_insertion_diagnostic = self._prediction_depth_insertion_diagnostic(
                frontend_output,
                image_hw,
            )
        if not first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            temporal_ok = self._seed_pixel_mask(seeds, insert_enabled, image_hw)
            render_masks, evidence_stats = self._pfgs360_render_bad_mask(
                frontend_output,
                image,
                sky_mask=sky_mask,
                temporal_ok=temporal_ok,
            )
            stats.update(evidence_stats)
            if render_masks is not None:
                depth_seed_mask = (
                    self._sample_seed_mask(render_masks["depth_mismatch"], seeds).detach().cpu().bool()
                    & insert_enabled
                )
                missing_seed_mask = (
                    self._sample_seed_mask(render_masks["missing"], seeds).detach().cpu().bool()
                    & insert_enabled
                )
                if self.prioritize_depth_mismatch:
                    missing_seed_mask &= ~depth_seed_mask
                render_bad_seed_mask = depth_seed_mask | missing_seed_mask
                stats["depth_mismatch_seed_candidates"] = int(depth_seed_mask.sum().item())
                stats["missing_seed_candidates"] = int(missing_seed_mask.sum().item())
                insert_enabled &= render_bad_seed_mask

        score = seeds.insert_score if seeds.insert_score is not None else seeds.confidence
        score_cpu = score.detach().cpu().float()
        conf_cpu = seeds.confidence.detach().cpu().float()
        xyz_cpu = seeds.xyz.detach().cpu().float()
        grid_cpu = (
            seeds.grid_coord.detach().cpu().to(torch.int32)
            if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == len(seeds)
            else torch.floor(xyz_cpu / float(self.pfgs360_voxel_size)).to(torch.int32)
        )
        order = torch.argsort(score_cpu, descending=True)
        occupied = self._build_voxel_index()
        anchor_xyz_cpu = self.map.get_xyz.detach().cpu().float() if self.map.anchor_count() > 0 else torch.zeros(0, 3)
        kept: list[int] = []
        kept_missing = 0
        kept_depth_mismatch = 0
        missing_budget = int(self.max_missing_seeds_per_keyframe)
        depth_budget = int(self.max_depth_mismatch_seeds_per_keyframe)
        for seed_idx in order.tolist():
            key = (0, int(grid_cpu[seed_idx, 0]), int(grid_cpu[seed_idx, 1]), int(grid_cpu[seed_idx, 2]))
            hit, near_hit = self._find_pfgs360_hash_hit(
                occupied,
                key,
                xyz_cpu[seed_idx],
                anchor_xyz_cpu,
            )
            if hit is not None:
                if int(hit) >= 0:
                    self._accumulate_existing_observation(
                        int(hit),
                        float(conf_cpu[seed_idx]),
                        frame_id=int(seeds.frame_id),
                    )
                    if near_hit:
                        stats["hash_near_hits"] += 1
                    else:
                        stats["hash_hits"] += 1
                stats["skipped_voxel"] += 1
                continue
            if not bool(insert_enabled[seed_idx]):
                stats["suppressed_insert"] += 1
                continue
            is_depth_mismatch_seed = bool(depth_seed_mask[seed_idx])
            is_missing_seed = bool(missing_seed_mask[seed_idx]) and not is_depth_mismatch_seed
            if is_depth_mismatch_seed and depth_budget > 0 and kept_depth_mismatch >= depth_budget:
                stats["skipped_depth_mismatch_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if is_missing_seed and missing_budget > 0 and kept_missing >= missing_budget:
                stats["skipped_missing_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if len(kept) >= budget:
                stats["skipped_budget"] += 1
                continue
            kept.append(int(seed_idx))
            if is_depth_mismatch_seed:
                kept_depth_mismatch += 1
            elif is_missing_seed:
                kept_missing += 1
            occupied[key] = -1
        if not kept:
            return self._empty_seed_like(seeds), stats
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), stats

    def _filter_replace_fuse_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None,
        image: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
        insert_occupancy_radius_voxels_override: float | None = None,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        seeds = self._with_pfgs360_seed_metadata(seeds)

        stats = {
            "skipped_voxel": 0,
            "skipped_budget": 0,
            "hash_hits": 0,
            "hash_near_hits": 0,
            "suppressed_insert": 0,
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "replace_deleted": 0,
            "fused": 0,
            "fused_existing": 0,
            "fused_new_duplicate": 0,
            "compacted": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
            "missing_seed_candidates": 0,
            "depth_mismatch_seed_candidates": 0,
            "dense_seed_candidates": int(len(seeds)),
            "insert_mask_seed_candidates": 0,
            "voxel_seed_candidates": 0,
            "newly_inserted": 0,
            "pred_depth_generated_seeds": 0,
            "pred_depth_invalid_pixels": 0,
            "insert_mask_pixels": 0,
            "skipped_missing_budget": 0,
            "skipped_depth_mismatch_budget": 0,
            "anchors_before": int(self.map.anchor_count()),
            "anchors_after": int(self.map.anchor_count()),
        }
        insert_enabled = (
            torch.ones(len(seeds), dtype=torch.bool)
            if seeds.insert_enabled is None
            else seeds.insert_enabled.detach().cpu().bool()
        )
        first_keyframe = self.stats.n_keyframes == 0 or self.map.anchor_count() == 0
        depth_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        if first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            self.last_depth_insertion_diagnostic = self._prediction_depth_insertion_diagnostic(
                frontend_output,
                image_hw,
            )
        elif frontend_output is not None and image is not None:
            render_masks, evidence_stats = self._pfgs360_replace_fuse_masks_and_delete(
                frontend_output,
                image,
                sky_mask=sky_mask,
                replace_delete_keyframe_ids=replace_delete_keyframe_ids,
            )
            stats.update({key: int(stats.get(key, 0)) + int(value) for key, value in evidence_stats.items()})
            if render_masks is not None:
                seeds, depth_seed_mask, pred_stats = self._replace_fuse_seeds_from_pred_depth(
                    frontend_output,
                    image,
                    render_masks,
                    frame_id=int(seeds.frame_id),
                )
                stats.update({key: int(stats.get(key, 0)) + int(value) for key, value in pred_stats.items()})
                insert_enabled = torch.ones(len(seeds), dtype=torch.bool)
                stats["dense_seed_candidates"] = int(len(seeds))
            else:
                seeds = self._empty_seed_like(seeds)
                insert_enabled = torch.zeros(0, dtype=torch.bool)
                depth_seed_mask = torch.zeros(0, dtype=torch.bool)
        per_keyframe_budget = self.first_keyframe_max_seeds if first_keyframe else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))
        stats["insert_mask_seed_candidates"] = int(insert_enabled.sum().item())
        stats["suppressed_insert"] = int((~insert_enabled).sum().item())
        if not bool(insert_enabled.any()):
            return self._empty_seed_like(seeds), stats

        xyz_cpu = seeds.xyz.detach().cpu().float()
        active_rows = torch.nonzero(insert_enabled, as_tuple=False).flatten()
        if active_rows.numel() == 0:
            return self._empty_seed_like(seeds), stats
        stats["voxel_seed_candidates"] = int(active_rows.numel())
        active_depth_seed = depth_seed_mask.index_select(0, active_rows).bool()
        stats["depth_mismatch_seed_candidates"] = int(active_depth_seed.sum().item())
        stats["missing_seed_candidates"] = int((~active_depth_seed).sum().item())

        anchor_xyz_cpu = self.map.get_xyz.detach().cpu().float() if self.map.anchor_count() > 0 else torch.zeros(0, 3)
        anchor_levels_cpu = (
            self.map._anchor_level[: self.map.anchor_count()].detach().cpu().to(torch.int8)
            if self.map.anchor_count() > 0
            else torch.zeros(0, dtype=torch.int8)
        )
        old_index = self._build_replace_fuse_radius_index(anchor_xyz_cpu, anchor_levels_cpu)
        new_index: dict[tuple[int, int, int, int], list[int]] = {}
        accepted_xyz: list[torch.Tensor] = []
        kept: list[int] = []
        kept_depth_mismatch = 0
        kept_insert_only = 0
        depth_budget = int(self.max_depth_mismatch_seeds_per_keyframe)
        insert_only_budget = 0 if first_keyframe else int(self.max_missing_seeds_per_keyframe)
        if budget <= 0:
            stats["skipped_budget"] = int(active_rows.numel())
            return self._empty_seed_like(seeds), stats
        generator = torch.Generator()
        generator.manual_seed(int(seeds.frame_id) & 0x7FFFFFFF)
        order = active_rows.index_select(0, torch.randperm(int(active_rows.numel()), generator=generator))
        radius_voxels = (
            float(self.replace_fuse_insert_occupancy_radius_voxels)
            if insert_occupancy_radius_voxels_override is None
            else float(insert_occupancy_radius_voxels_override)
        )
        radius = max(0.0, radius_voxels) * float(self.pfgs360_voxel_size)
        radius_cells = max(0, int(math.ceil(max(0.0, radius_voxels))))
        seed_levels_cpu = (
            seeds.level.detach().cpu().to(torch.int32)
            if seeds.level is not None and int(seeds.level.numel()) == len(seeds)
            else torch.zeros(len(seeds), dtype=torch.int32)
        )
        for seed_idx in order.tolist():
            seed_level = int(seed_levels_cpu[int(seed_idx)])
            candidate_xyz = xyz_cpu[int(seed_idx)]
            if self._replace_fuse_radius_hit(
                old_index,
                anchor_xyz_cpu,
                candidate_xyz,
                level=seed_level,
                radius=radius,
                radius_cells=radius_cells,
            ) is not None:
                stats["hash_hits"] += 1
                stats["skipped_voxel"] += 1
                continue
            if self._replace_fuse_radius_hit(
                new_index,
                accepted_xyz,
                candidate_xyz,
                level=seed_level,
                radius=radius,
                radius_cells=radius_cells,
            ) is not None:
                stats["skipped_voxel"] += 1
                stats["fused"] += 1
                stats["fused_new_duplicate"] += 1
                continue
            is_delete_band_seed = bool(depth_seed_mask[int(seed_idx)])
            if is_delete_band_seed and depth_budget > 0 and kept_depth_mismatch >= depth_budget:
                stats["skipped_depth_mismatch_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if (not is_delete_band_seed) and insert_only_budget > 0 and kept_insert_only >= insert_only_budget:
                stats["skipped_missing_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if len(kept) >= budget:
                stats["skipped_budget"] += 1
                continue
            seed_idx = int(seed_idx)
            kept.append(seed_idx)
            if is_delete_band_seed:
                kept_depth_mismatch += 1
            else:
                kept_insert_only += 1
            key = self._replace_fuse_spatial_key(candidate_xyz, level=seed_level)
            new_index.setdefault(key, []).append(len(accepted_xyz))
            accepted_xyz.append(candidate_xyz)
        if not kept:
            return self._empty_seed_like(seeds), stats
        stats["newly_inserted"] = int(len(kept))
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), stats

    def _with_pfgs360_seed_metadata(self, seeds: GaussianSeedBatch) -> GaussianSeedBatch:
        n = len(seeds)
        device = seeds.xyz.device
        dtype = seeds.xyz.dtype
        grid = (
            seeds.grid_coord.to(device=device, dtype=torch.int32)
            if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == n
            else torch.floor(seeds.xyz.detach() / float(self.pfgs360_voxel_size)).to(device=device, dtype=torch.int32)
        )
        return GaussianSeedBatch(
            xyz=seeds.xyz,
            rgb=seeds.rgb,
            confidence=seeds.confidence,
            scale=(
                seeds.scale.to(device=device, dtype=dtype)
                if self.pfgs360_replace_fuse_enabled
                else torch.full((n,), float(self.pfgs360_voxel_size), device=device, dtype=dtype)
            ),
            level=torch.zeros(n, dtype=torch.int8, device=device),
            frame_id=int(seeds.frame_id),
            source_flat_idx=seeds.source_flat_idx,
            source_hw=seeds.source_hw,
            insert_enabled=(
                torch.ones(n, dtype=torch.bool, device=device)
                if seeds.insert_enabled is None
                else seeds.insert_enabled.to(device=device, dtype=torch.bool)
            ),
            insert_score=(
                seeds.confidence.to(device=device, dtype=dtype).clamp(0.0, 1.0)
                if seeds.insert_score is None
                else seeds.insert_score.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            ),
            grid_coord=grid,
        )

    def _replace_fuse_seeds_from_pred_depth(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        masks: dict[str, torch.Tensor],
        *,
        frame_id: int,
    ) -> tuple[GaussianSeedBatch, torch.Tensor, dict[str, int]]:
        target = image.detach()
        if target.ndim == 4:
            target = target[0]
        if target.ndim != 3:
            raise ValueError(f"Expected keyframe image as 3xHxW, got {tuple(target.shape)}")
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        target = target.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        stats = {
            "pred_depth_generated_seeds": 0,
            "pred_depth_invalid_pixels": 0,
            "insert_mask_pixels": 0,
        }

        insert_mask_t = masks.get("insert", masks.get("render_bad"))
        if not torch.is_tensor(insert_mask_t):
            return self._empty_pred_depth_seed_batch(frame_id, H, W, device, dtype), torch.zeros(0, dtype=torch.bool), stats
        insert_mask = insert_mask_t.detach().bool()
        if insert_mask.ndim == 2:
            insert_mask = insert_mask.unsqueeze(0)
        if tuple(insert_mask.shape[-2:]) != (H, W):
            insert_mask = F.interpolate(
                insert_mask.float().view(1, 1, *insert_mask.shape[-2:]),
                size=(H, W),
                mode="nearest",
            )[0] > 0.5
        insert_mask = insert_mask.to(device=device, dtype=torch.bool)
        stats["insert_mask_pixels"] = int(insert_mask.sum().detach().cpu())
        if stats["insert_mask_pixels"] <= 0:
            return self._empty_pred_depth_seed_batch(frame_id, H, W, device, dtype), torch.zeros(0, dtype=torch.bool), stats

        depth_t = masks.get("predicted_depth")
        if torch.is_tensor(depth_t):
            pred_depth = depth_t.detach().float()
            if pred_depth.ndim == 2:
                pred_depth = pred_depth.unsqueeze(0)
            if tuple(pred_depth.shape[-2:]) != (H, W):
                pred_depth = F.interpolate(pred_depth.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            pred_depth = pred_depth.to(device=device, dtype=dtype)
        else:
            pred_depth, _ = self._target_depth_from_output(frontend_output, (H, W), device, dtype)
            if pred_depth is None:
                stats["pred_depth_invalid_pixels"] = stats["insert_mask_pixels"]
                return self._empty_pred_depth_seed_batch(frame_id, H, W, device, dtype), torch.zeros(0, dtype=torch.bool), stats

        numeric_valid = torch.isfinite(pred_depth) & (pred_depth > 1.0e-6)
        raw_valid = self._raw_predicted_depth_valid_mask(frontend_output, (H, W), device)
        if raw_valid is not None:
            numeric_valid = numeric_valid & raw_valid
        seed_pixel_mask = insert_mask & numeric_valid
        stats["pred_depth_invalid_pixels"] = int((insert_mask & ~numeric_valid).sum().detach().cpu())
        flat_idx = torch.nonzero(seed_pixel_mask.reshape(-1), as_tuple=False).flatten()
        if flat_idx.numel() == 0:
            return self._empty_pred_depth_seed_batch(frame_id, H, W, device, dtype), torch.zeros(0, dtype=torch.bool), stats

        grid = pixel_grid(H, W, device=device, dtype=dtype).view(-1, 2).index_select(0, flat_idx)
        bearing = erp_pixel_to_bearing(grid, H, W).to(device=device, dtype=dtype)
        depth = pred_depth.reshape(-1).index_select(0, flat_idx).clamp_min(1.0e-6)
        pts_cam = bearing * depth.unsqueeze(-1)
        c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        pts_h = torch.cat([pts_cam, torch.ones(int(flat_idx.numel()), 1, device=device, dtype=dtype)], dim=-1)
        xyz = (c2w @ pts_h.T).T[:, :3]

        rgb = target.permute(1, 2, 0).reshape(-1, 3).index_select(0, flat_idx).contiguous()
        confidence = torch.ones(int(flat_idx.numel()), device=device, dtype=dtype)
        scale = self._replace_fuse_seed_scale_from_depth(
            xyz,
            flat_idx,
            H,
            W,
            c2w,
            device=device,
            dtype=dtype,
        )
        delete_mask = masks.get("delete", masks.get("depth_mismatch"))
        if torch.is_tensor(delete_mask):
            delete_t = delete_mask.detach().bool()
            if delete_t.ndim == 2:
                delete_t = delete_t.unsqueeze(0)
            if tuple(delete_t.shape[-2:]) != (H, W):
                delete_t = F.interpolate(
                    delete_t.float().view(1, 1, *delete_t.shape[-2:]),
                    size=(H, W),
                    mode="nearest",
                )[0] > 0.5
            depth_seed_mask = delete_t.reshape(-1).to(device=device, dtype=torch.bool).index_select(0, flat_idx)
        else:
            depth_seed_mask = torch.zeros(int(flat_idx.numel()), dtype=torch.bool, device=device)

        seeds = GaussianSeedBatch(
            xyz=xyz,
            rgb=rgb,
            confidence=confidence,
            scale=scale,
            level=torch.zeros(int(flat_idx.numel()), dtype=torch.int8, device=device),
            frame_id=int(frame_id),
            source_flat_idx=flat_idx.to(device=device, dtype=torch.long),
            source_hw=(H, W),
            insert_enabled=torch.ones(int(flat_idx.numel()), dtype=torch.bool, device=device),
            insert_score=torch.ones(int(flat_idx.numel()), dtype=dtype, device=device),
            grid_coord=torch.floor(xyz.detach() / float(self.pfgs360_voxel_size)).to(device=device, dtype=torch.int32),
        )
        stats["pred_depth_generated_seeds"] = int(len(seeds))
        return seeds, depth_seed_mask.detach().cpu().bool(), stats

    @staticmethod
    def _empty_pred_depth_seed_batch(
        frame_id: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=torch.zeros(0, 3, device=device, dtype=dtype),
            rgb=torch.zeros(0, 3, device=device, dtype=dtype),
            confidence=torch.zeros(0, device=device, dtype=dtype),
            scale=torch.zeros(0, device=device, dtype=dtype),
            level=torch.zeros(0, dtype=torch.int8, device=device),
            frame_id=int(frame_id),
            source_flat_idx=torch.zeros(0, dtype=torch.long, device=device),
            source_hw=(int(H), int(W)),
            insert_enabled=torch.zeros(0, dtype=torch.bool, device=device),
            insert_score=torch.zeros(0, device=device, dtype=dtype),
            grid_coord=torch.zeros(0, 3, dtype=torch.int32, device=device),
        )

    def _replace_fuse_seed_scale_from_depth(
        self,
        xyz: torch.Tensor,
        source_flat_idx: torch.Tensor,
        H: int,
        W: int,
        c2w: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if str(self.pfgs360_gaussian_scale_mode).lower() not in {"erp_depth_latitude", "depth_latitude"}:
            return torch.full((int(xyz.shape[0]),), float(self.pfgs360_voxel_size), device=device, dtype=dtype)
        center = c2w[:3, 3] if tuple(c2w.shape) == (4, 4) else torch.zeros(3, device=device, dtype=dtype)
        depth = torch.linalg.norm(xyz.to(device=device, dtype=dtype) - center.view(1, 3), dim=-1).clamp_min(1.0e-6)
        rows = torch.div(source_flat_idx.to(device=device, dtype=torch.long), int(W), rounding_mode="floor").to(dtype=dtype)
        lat = math.pi * 0.5 - (rows + 0.5) * math.pi / float(H)
        cos_lat = torch.cos(lat).clamp(float(self.pfgs360_gaussian_scale_lat_cos_min), 1.0)
        angular = math.sqrt((2.0 * math.pi / float(W)) * (math.pi / float(H))) * torch.sqrt(cos_lat)
        scale = float(self.pfgs360_gaussian_scale_factor) * depth * angular
        return scale.clamp(float(self.pfgs360_gaussian_scale_min), float(self.pfgs360_gaussian_scale_max))

    def _raw_predicted_depth_valid_mask(
        self,
        frontend_output: FrontendOutput,
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        H, W = int(size[0]), int(size[1])
        if frontend_output.inverse_depth is not None:
            inv = frontend_output.inverse_depth.detach().float()
            if inv.ndim == 2:
                inv = inv.unsqueeze(0)
            valid = torch.isfinite(inv) & (inv > 1.0e-6)
        elif frontend_output.world_points is not None:
            pts = frontend_output.world_points.detach().float()
            if pts.ndim == 4 and int(pts.shape[0]) == 1:
                pts = pts[0]
            if pts.ndim != 3 or int(pts.shape[-1]) != 3:
                return None
            valid = torch.isfinite(pts).all(dim=-1, keepdim=False).unsqueeze(0)
            valid_mask = frontend_output.valid_world_points_mask
            if valid_mask is not None:
                extra = valid_mask.detach().bool()
                if extra.ndim == 2:
                    extra = extra.unsqueeze(0)
                valid = valid & extra
        else:
            return None
        if tuple(valid.shape[-2:]) != (H, W):
            valid = F.interpolate(valid.float().unsqueeze(0), size=(H, W), mode="nearest")[0] > 0.5
        return valid.to(device=device, dtype=torch.bool)

    def _replace_fuse_spatial_key(
        self,
        xyz_cpu: torch.Tensor,
        *,
        level: int = 0,
    ) -> tuple[int, int, int, int]:
        coord = torch.floor(xyz_cpu.detach().cpu().float() / float(self.pfgs360_voxel_size)).to(torch.int32)
        return (int(level), int(coord[0]), int(coord[1]), int(coord[2]))

    def _build_replace_fuse_radius_index(
        self,
        xyz_cpu: torch.Tensor,
        levels_cpu: torch.Tensor | None = None,
    ) -> dict[tuple[int, int, int, int], list[int]]:
        index: dict[tuple[int, int, int, int], list[int]] = {}
        if xyz_cpu.numel() == 0:
            return index
        xyz_cpu = xyz_cpu.detach().cpu().float()
        levels = (
            levels_cpu.detach().cpu().to(torch.int32).view(-1)
            if torch.is_tensor(levels_cpu) and int(levels_cpu.numel()) == int(xyz_cpu.shape[0])
            else torch.zeros(int(xyz_cpu.shape[0]), dtype=torch.int32)
        )
        coords = torch.floor(xyz_cpu / float(self.pfgs360_voxel_size)).to(torch.int32)
        for idx in range(int(xyz_cpu.shape[0])):
            key = (int(levels[idx]), int(coords[idx, 0]), int(coords[idx, 1]), int(coords[idx, 2]))
            index.setdefault(key, []).append(int(idx))
        return index

    def _replace_fuse_radius_hit(
        self,
        index: dict[tuple[int, int, int, int], list[int]],
        points,
        candidate_xyz: torch.Tensor,
        *,
        level: int,
        radius: float,
        radius_cells: int,
    ) -> int | None:
        if radius <= 0.0 or not index:
            return None
        candidate = candidate_xyz.detach().cpu().float()
        _, cx, cy, cz = self._replace_fuse_spatial_key(candidate, level=int(level))
        radius_sq = float(radius) * float(radius)
        best_idx: int | None = None
        best_dist = float("inf")
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    rows = index.get((int(level), cx + dx, cy + dy, cz + dz), [])
                    for row in rows:
                        point = points[int(row)] if isinstance(points, list) else points[int(row)]
                        diff = point.detach().cpu().float() - candidate
                        dist_sq = float(torch.dot(diff, diff).item())
                        if dist_sq <= radius_sq + 1.0e-12 and dist_sq < best_dist:
                            best_dist = dist_sq
                            best_idx = int(row)
        return best_idx

    def _find_pfgs360_hash_hit(
        self,
        occupied: dict[tuple[int, int, int, int], int],
        key: tuple[int, int, int, int],
        candidate_xyz: torch.Tensor,
        anchor_xyz_cpu: torch.Tensor,
    ) -> tuple[int | None, bool]:
        hit = occupied.get(key)
        if hit is not None:
            return int(hit), False
        radius = int(self.pfgs360_near_grid_radius)
        if radius <= 0 or anchor_xyz_cpu.numel() == 0 or self.pfgs360_near_distance_factor <= 0.0:
            return None, False
        level, x, y, z = key
        best_row: int | None = None
        best_dist = float("inf")
        thresh = float(self.pfgs360_near_distance_factor) * float(self.pfgs360_voxel_size)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    row = occupied.get((level, x + dx, y + dy, z + dz))
                    if row is None or int(row) < 0 or int(row) >= int(anchor_xyz_cpu.shape[0]):
                        continue
                    dist = float(torch.linalg.norm(anchor_xyz_cpu[int(row)] - candidate_xyz).item())
                    if dist <= thresh and dist < best_dist:
                        best_dist = dist
                        best_row = int(row)
        return best_row, best_row is not None

    def _pfgs360_replace_fuse_masks_and_delete(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
        stats = {
            "replace_deleted": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
        }
        if self.map.anchor_count() == 0:
            return None, stats
        target = image.detach().to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        if target.ndim == 4:
            target = target[0]
        H, W = int(target.shape[-2]), int(target.shape[-1])
        target_depth, depth_conf = self._target_depth_from_output(frontend_output, (H, W), target.device, target.dtype)
        if target_depth is None:
            return None, stats
        with torch.no_grad():
            camera = PanoRenderCamera(
                image_height=H,
                image_width=W,
                c2w=frontend_output.pose_c2w.detach().to(device=target.device, dtype=target.dtype),
            )
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, sky_mask))
            render_depth = pkg.get("depth")
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not torch.is_tensor(render_depth):
                return None, stats
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(render_depth)
            render_depth = render_depth.to(device=target.device, dtype=target.dtype)
            alpha = alpha.to(device=target.device, dtype=target.dtype)
            valid = torch.isfinite(target_depth) & torch.isfinite(render_depth) & (target_depth > 1.0e-6) & (render_depth > 1.0e-6)
            if depth_conf is not None:
                valid = valid & (depth_conf.to(device=target.device, dtype=target.dtype) > 1.0e-6)
            non_sky = None
            if sky_mask is not None:
                non_sky = ~self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)
            scale_valid = valid & (alpha >= self.pfgs360_render_alpha_min)
            if non_sky is not None:
                scale_valid = scale_valid & non_sky
            scale, shift = self._robust_depth_scale_shift(target_depth, render_depth, scale_valid)
            aligned_target = (target_depth * scale + shift).clamp_min(1.0e-6)
            valid_aligned = torch.isfinite(aligned_target) & torch.isfinite(render_depth) & (render_depth > 1.0e-6)
            rel = (aligned_target - render_depth).abs() / torch.maximum(aligned_target, render_depth).clamp_min(1.0e-6)
            missing = ~valid_aligned
            delete_band = (
                valid_aligned
                & (rel >= float(self.replace_fuse_delete_rel_min))
                & (rel <= float(self.replace_fuse_delete_rel_max))
            )
            delete_foreground_large = (
                valid_aligned
                & (rel > float(self.replace_fuse_delete_rel_max))
                & (render_depth < aligned_target)
            )
            delete_mask = delete_band | delete_foreground_large
            insert_depth_mask = valid_aligned & (rel >= float(self.replace_fuse_delete_rel_min))
            if non_sky is not None:
                missing = missing & non_sky
                delete_mask = delete_mask & non_sky
                insert_depth_mask = insert_depth_mask & non_sky
            insert_mask = missing | insert_depth_mask
            stats["missing_pixels"] = int(missing.sum().detach().cpu())
            stats["depth_mismatch_pixels"] = int(delete_mask.sum().detach().cpu())
            stats["render_bad_pixels"] = int(insert_mask.sum().detach().cpu())
            stats["replace_deleted"] = self._delete_responsible_replace_fuse_anchors(
                delete_mask.detach(),
                aligned_target.detach(),
                pkg,
                frontend_output,
                H,
                W,
                replace_delete_keyframe_ids=replace_delete_keyframe_ids,
            )
            masks = {
                "insert": insert_mask.detach().cpu().bool(),
                "delete": delete_mask.detach().cpu().bool(),
                "missing": missing.detach().cpu().bool(),
                "depth_mismatch": delete_mask.detach().cpu().bool(),
                "render_bad": insert_mask.detach().cpu().bool(),
                "predicted_depth": aligned_target.detach().cpu().float(),
            }
            self.last_depth_insertion_diagnostic = DepthInsertionDiagnostic(
                frame_id=int(frontend_output.frame_id),
                render_depth=render_depth.detach().cpu().float(),
                predicted_depth=aligned_target.detach().cpu().float(),
                rel_depth_error=rel.detach().cpu().float(),
                missing_mask=masks["missing"],
                depth_mismatch_mask=masks["depth_mismatch"],
                render_bad_mask=masks["render_bad"],
                depth_scale=float(scale.detach().cpu()),
                depth_shift=float(shift.detach().cpu()),
            )
            return masks, stats

    def _delete_responsible_replace_fuse_anchors(
        self,
        delete_mask: torch.Tensor,
        target_depth: torch.Tensor,
        render_pkg: dict,
        frontend_output: FrontendOutput,
        H: int,
        W: int,
        replace_delete_keyframe_ids: list[int] | tuple[int, ...] | None = None,
    ) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            return 0
        if not torch.is_tensor(target_depth):
            return 0
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        allowed_mask = None
        if replace_delete_keyframe_ids is not None:
            allowed_mask = self._active_anchor_mask_for_keyframe_update_ids(
                [int(fid) for fid in replace_delete_keyframe_ids]
            ).to(device=device, dtype=torch.bool)
            if int(allowed_mask.numel()) != n or not bool(allowed_mask.any().detach().cpu()):
                return 0
        xyz = self.map.get_xyz.detach()
        c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        w2c = invert_c2w(c2w)
        xyz_h = torch.cat([xyz, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        visibility = render_pkg.get("visibility_filter")
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            valid = valid & visibility.to(device=device, dtype=torch.bool).view(-1)
        if allowed_mask is not None:
            valid = valid & allowed_mask
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() == 0:
            return 0
        cam_rows = cam.index_select(0, rows)
        pixels = bearing_to_erp_pixel(cam_rows, int(H), int(W))
        ui = pixels[:, 0].round().long().remainder(int(W))
        vi = pixels[:, 1].round().long().clamp(0, int(H) - 1)
        delete_at_pixel = delete_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
        td = target_depth.to(device=device, dtype=dtype)[0, vi, ui]
        valid_target_depth = torch.isfinite(td) & (td > 1.0e-6)
        anchor_depth = dist.index_select(0, rows)
        tol = torch.maximum(
            td.new_full(td.shape, float(self.replace_fuse_front_depth_abs_tol)),
            td.clamp_min(1.0e-6) * float(self.replace_fuse_front_depth_rel_tol),
        )
        remove_depth = anchor_depth <= (td + tol)
        prune_rows = rows[delete_at_pixel & valid_target_depth & remove_depth]
        if prune_rows.numel() == 0:
            return 0
        max_delete = int(self.replace_fuse_max_delete_per_keyframe)
        if max_delete > 0 and prune_rows.numel() > max_delete:
            prune_rows = prune_rows[:max_delete]
        prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
        prune_mask[prune_rows] = True
        deleted = self.map.prune_anchors(prune_mask)
        if deleted > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            self._refresh_pfgs360_voxel_cache(compact=False)
        return int(deleted)

    def _build_replace_fuse_voxel_index(self) -> dict[tuple[int, int, int, int], list[int]]:
        occupied: dict[tuple[int, int, int, int], list[int]] = {}
        if self.map.anchor_count() <= 0:
            return occupied
        self._refresh_pfgs360_voxel_cache(compact=False)
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            key = (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            occupied.setdefault(key, []).append(int(idx))
        return occupied

    def _refresh_pfgs360_voxel_cache(self, *, compact: bool) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            self.map._anchor_grid_coord = torch.zeros(0, 3, dtype=torch.int32)
            return 0
        grid = torch.floor(self.map.get_xyz.detach().cpu().float() / float(self.pfgs360_voxel_size)).to(torch.int32)
        self.map._anchor_grid_coord = grid
        if not compact:
            return 0
        return self._compact_replace_fuse_voxels()

    def _compact_replace_fuse_voxels(self) -> int:
        n = self.map.anchor_count()
        if n <= 1:
            return 0
        index = self._build_replace_fuse_voxel_index_no_refresh()
        duplicate_rows: list[int] = []
        opacity = self.map.get_opacity.detach().cpu().view(-1)
        obs = self.map._anchor_obs_count[:n].detach().cpu().float().clamp_min(1.0)
        score = opacity * obs
        for rows in index.values():
            if len(rows) <= 1:
                continue
            rows_t = torch.tensor(rows, dtype=torch.long)
            keeper = int(rows_t[torch.argmax(score.index_select(0, rows_t))])
            self._merge_anchor_group_into_anchor(keeper, [int(row) for row in rows])
            for row in rows:
                if int(row) == keeper:
                    continue
                duplicate_rows.append(int(row))
        if not duplicate_rows:
            return 0
        prune_mask = torch.zeros(n, dtype=torch.bool, device=self.map.get_xyz.device)
        prune_mask[torch.tensor(duplicate_rows, dtype=torch.long, device=prune_mask.device)] = True
        pruned = self.map.prune_anchors(prune_mask)
        if pruned > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            self._refresh_pfgs360_voxel_cache(compact=False)
        return int(pruned)

    def _build_replace_fuse_voxel_index_no_refresh(self) -> dict[tuple[int, int, int, int], list[int]]:
        occupied: dict[tuple[int, int, int, int], list[int]] = {}
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            key = (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            occupied.setdefault(key, []).append(int(idx))
        return occupied

    def _merge_anchor_group_into_anchor(self, keeper: int, rows: list[int]) -> None:
        n = self.map.anchor_count()
        unique_rows = sorted({int(row) for row in rows if 0 <= int(row) < n})
        if int(keeper) not in unique_rows:
            return
        if len(unique_rows) <= 1:
            return
        device = self.map.get_xyz.device
        rows_t = torch.tensor(unique_rows, dtype=torch.long, device=device)
        rows_cpu = torch.tensor(unique_rows, dtype=torch.long)
        with torch.no_grad():
            self.map.xyz.data[keeper] = self.map.xyz.data.index_select(0, rows_t).mean(dim=0)
            rgb = self.map.get_features.detach().index_select(0, rows_t).mean(dim=0).clamp(0.0, 1.0)
            self.map.features.data[keeper] = self.map._inv_sigmoid(rgb.view(1, 3)).view(3)
            opacity = self.map.get_opacity.detach().index_select(0, rows_t).mean(dim=0).clamp(0.0, 1.0)
            self.map.opacity_logit.data[keeper] = self.map._inv_sigmoid(opacity.view(1, 1)).view(1)
            scale = self.map.get_scaling.detach().index_select(0, rows_t).mean(dim=0)
            self.map.scaling.data[keeper] = torch.log(torch.expm1(scale.clamp_min(1.0e-5)))
            rotations = self.map.get_rotation.detach().index_select(0, rows_t)
            ref_pos = unique_rows.index(int(keeper))
            ref = rotations[ref_pos]
            if torch.linalg.norm(ref) <= 1.0e-8:
                ref = torch.zeros(4, device=device, dtype=rotations.dtype)
                ref[0] = 1.0
            dots = (rotations * ref.view(1, 4)).sum(dim=-1, keepdim=True)
            aligned = torch.where(dots < 0.0, -rotations, rotations)
            quat = aligned.sum(dim=0)
            quat_norm = torch.linalg.norm(quat)
            if quat_norm <= 1.0e-8:
                quat = ref
            else:
                quat = quat / quat_norm
            self.map.rotation.data[keeper] = quat
            if int(self.map.sh_rest.shape[1]) > 0:
                self.map.sh_rest.data[keeper] = self.map.sh_rest.data.index_select(0, rows_t).mean(dim=0)
        self.map._anchor_obs_count[keeper] = self.map._anchor_obs_count.index_select(0, rows_cpu).sum()
        self.map._anchor_conf_accum[keeper] = self.map._anchor_conf_accum.index_select(0, rows_cpu).sum()
        self.map._anchor_last_seen_kf[keeper] = self.map._anchor_last_seen_kf.index_select(0, rows_cpu).max()
        self.map._anchor_last_update_kf_ord[keeper] = self.map._anchor_last_update_kf_ord.index_select(0, rows_cpu).max()
        self.map._anchor_birth_frame[keeper] = self.map._anchor_birth_frame.index_select(0, rows_cpu).min()
        self.map._anchor_inlier_obs[keeper] = self.map._anchor_inlier_obs.index_select(0, rows_cpu).sum()
        self.map._anchor_outlier_obs[keeper] = self.map._anchor_outlier_obs.index_select(0, rows_cpu).sum()
        self.map._anchor_voxel_size[keeper] = self.map._anchor_voxel_size.index_select(0, rows_cpu).mean()

    def _pfgs360_render_bad_mask(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
        temporal_ok: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
        stats = {
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
        }
        if self.map.anchor_count() == 0:
            return None, stats
        target = image.detach().to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        if target.ndim == 4:
            target = target[0]
        H, W = int(target.shape[-2]), int(target.shape[-1])
        target_depth, depth_conf = self._target_depth_from_output(frontend_output, (H, W), target.device, target.dtype)
        if target_depth is None:
            return None, stats
        with torch.no_grad():
            camera = PanoRenderCamera(
                image_height=H,
                image_width=W,
                c2w=frontend_output.pose_c2w.detach().to(device=target.device, dtype=target.dtype),
            )
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, sky_mask))
            render_depth = pkg.get("depth")
            render_rgb = pkg.get("render")
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not (torch.is_tensor(render_depth) and torch.is_tensor(render_rgb)):
                return None, stats
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(render_depth)
            render_depth = render_depth.to(device=target.device, dtype=target.dtype)
            alpha = alpha.to(device=target.device, dtype=target.dtype)
            valid = torch.isfinite(target_depth) & torch.isfinite(render_depth) & (target_depth > 1.0e-6) & (render_depth > 1.0e-6)
            if depth_conf is not None:
                valid = valid & (depth_conf.to(device=target.device, dtype=target.dtype) > 1.0e-6)
            scale, shift = self._robust_depth_scale_shift(target_depth, render_depth, valid & (alpha >= self.pfgs360_render_alpha_min))
            aligned_target = (target_depth * scale + shift).clamp_min(1.0e-6)
            valid_aligned = torch.isfinite(aligned_target) & torch.isfinite(render_depth) & (render_depth > 1.0e-6)
            missing_alpha_min = max(0.0, min(1.0, float(self.pfgs360_missing_alpha_min)))
            missing = (alpha < missing_alpha_min) | (~valid_aligned)
            rel = (aligned_target - render_depth).abs() / torch.maximum(aligned_target, render_depth).clamp_min(1.0e-6)
            depth_mismatch = valid_aligned & (rel > self.pfgs360_render_depth_rel_threshold)
            render_bad = missing | depth_mismatch
            if sky_mask is not None:
                non_sky = ~self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)
                missing = missing & non_sky
                depth_mismatch = depth_mismatch & non_sky
                render_bad = render_bad & non_sky
            stats["missing_pixels"] = int(missing.sum().detach().cpu())
            stats["depth_mismatch_pixels"] = int(depth_mismatch.sum().detach().cpu())
            stats["render_bad_pixels"] = int(render_bad.sum().detach().cpu())
            evidence_stats = self._update_pfgs360_outlier_evidence(
                render_bad.detach(),
                pkg,
                frontend_output,
                H,
                W,
                temporal_ok=temporal_ok,
            )
            stats.update(evidence_stats)
            masks = {
                "render_bad": render_bad.detach().cpu().bool(),
                "missing": missing.detach().cpu().bool(),
                "depth_mismatch": depth_mismatch.detach().cpu().bool(),
            }
            self.last_depth_insertion_diagnostic = DepthInsertionDiagnostic(
                frame_id=int(frontend_output.frame_id),
                render_depth=render_depth.detach().cpu().float(),
                predicted_depth=aligned_target.detach().cpu().float(),
                rel_depth_error=rel.detach().cpu().float(),
                missing_mask=masks["missing"],
                depth_mismatch_mask=masks["depth_mismatch"],
                render_bad_mask=masks["render_bad"],
                depth_scale=float(scale.detach().cpu()),
                depth_shift=float(shift.detach().cpu()),
            )
            return masks, stats

    def _prediction_depth_insertion_diagnostic(
        self,
        frontend_output: FrontendOutput,
        image_hw: tuple[int, int],
    ) -> DepthInsertionDiagnostic | None:
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        target_depth, _ = self._target_depth_from_output(frontend_output, image_hw, device, dtype)
        if target_depth is None:
            return None
        return DepthInsertionDiagnostic(
            frame_id=int(frontend_output.frame_id),
            render_depth=None,
            predicted_depth=target_depth.detach().cpu().float(),
            rel_depth_error=None,
            missing_mask=None,
            depth_mismatch_mask=None,
            render_bad_mask=None,
            depth_scale=1.0,
            depth_shift=0.0,
        )

    def _target_depth_from_output(
        self,
        frontend_output: FrontendOutput,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._target_depth_from_tensors(
            inverse_depth=frontend_output.inverse_depth,
            world_points=frontend_output.world_points,
            pose_c2w=frontend_output.pose_c2w,
            confidence=(
                frontend_output.depth_confidence
                if frontend_output.depth_confidence is not None
                else frontend_output.world_points_confidence
            ),
            size=size,
            device=device,
            dtype=dtype,
        )

    def _target_depth_from_tensors(
        self,
        *,
        inverse_depth: torch.Tensor | None,
        world_points: torch.Tensor | None,
        pose_c2w: torch.Tensor,
        confidence: torch.Tensor | None,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        H, W = int(size[0]), int(size[1])
        target_depth: torch.Tensor | None = None
        if inverse_depth is not None:
            inv = inverse_depth.detach().float()
            if inv.ndim == 2:
                inv = inv.unsqueeze(0)
            target_depth = inv.clamp_min(1.0e-6).reciprocal()
        elif world_points is not None:
            pts = world_points.detach().float()
            if pts.ndim == 4 and int(pts.shape[0]) == 1:
                pts = pts[0]
            if pts.ndim == 3 and int(pts.shape[-1]) == 3:
                c2w = pose_c2w.detach().float()
                center = c2w[:3, 3].view(1, 1, 3)
                target_depth = torch.linalg.norm(pts - center, dim=-1, keepdim=False).unsqueeze(0)
        if target_depth is None:
            return None, None
        if tuple(target_depth.shape[-2:]) != (H, W):
            target_depth = F.interpolate(target_depth.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
        target_depth = target_depth.to(device=device, dtype=dtype)
        conf_t = None
        if confidence is not None:
            conf_t = confidence.detach().float()
            if conf_t.ndim == 2:
                conf_t = conf_t.unsqueeze(0)
            if tuple(conf_t.shape[-2:]) != (H, W):
                conf_t = F.interpolate(conf_t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            conf_t = conf_t.to(device=device, dtype=dtype)
        return target_depth, conf_t

    @staticmethod
    def _robust_depth_scale_shift(
        target_depth: torch.Tensor,
        render_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        if values.numel() < 16:
            one = render_depth.new_tensor(1.0)
            zero = render_depth.new_tensor(0.0)
            return one, zero
        td = target_depth.reshape(-1).index_select(0, values).clamp_min(1.0e-6)
        rd = render_depth.reshape(-1).index_select(0, values).clamp_min(1.0e-6)
        scale = torch.median(rd / td).clamp(0.05, 20.0)
        shift = torch.median(rd - scale * td)
        return scale, shift

    @staticmethod
    def _sample_seed_mask(mask: torch.Tensor, seeds: GaussianSeedBatch) -> torch.Tensor:
        if seeds.source_flat_idx is None or seeds.source_hw is None:
            return torch.ones(len(seeds), dtype=torch.bool)
        flat = mask.detach().bool().reshape(-1)
        idx = seeds.source_flat_idx.detach().cpu().long()
        valid = (idx >= 0) & (idx < int(flat.shape[0]))
        out = torch.ones(len(seeds), dtype=torch.bool)
        if bool(valid.any()):
            out[valid] = flat.index_select(0, idx[valid])
        return out

    @staticmethod
    def _seed_pixel_mask(
        seeds: GaussianSeedBatch,
        flags: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if seeds.source_flat_idx is None or seeds.source_hw is None:
            return None
        H, W = int(image_hw[0]), int(image_hw[1])
        if tuple(seeds.source_hw) != (H, W):
            return None
        idx = seeds.source_flat_idx.detach().cpu().long()
        valid = (idx >= 0) & (idx < H * W)
        if not bool(valid.any()):
            return None
        out = torch.zeros(1, H, W, dtype=torch.bool)
        out.view(-1)[idx[valid]] = flags.detach().cpu().bool()[valid]
        return out

    def _update_pfgs360_outlier_evidence(
        self,
        render_bad: torch.Tensor,
        render_pkg: dict,
        frontend_output: FrontendOutput,
        H: int,
        W: int,
        *,
        temporal_ok: torch.Tensor | None = None,
    ) -> dict[str, int]:
        stats = {"outlier_resets": 0, "outlier_pruned": 0}
        n = self.map.anchor_count()
        if n <= 0 or self.map._anchor_outlier_obs.shape[0] != n:
            return stats
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        xyz = self.map.get_xyz.detach()
        c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        w2c = invert_c2w(c2w)
        xyz_h = torch.cat([xyz, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        visibility = render_pkg.get("visibility_filter")
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            valid = valid & visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() == 0:
            return stats
        pixels = bearing_to_erp_pixel(cam.index_select(0, rows), H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        bad = render_bad.to(device=device, dtype=torch.bool)[0, vi, ui]
        if temporal_ok is not None:
            temporal = temporal_ok.to(device=device, dtype=torch.bool)[0, vi, ui]
            rows = rows.index_select(0, torch.nonzero(temporal, as_tuple=False).flatten())
            bad = bad[temporal]
            if rows.numel() == 0:
                return stats
        rows_cpu = rows.detach().cpu()
        bad_cpu = bad.detach().cpu()
        if bool((~bad_cpu).any()):
            self.map._anchor_inlier_obs[rows_cpu[~bad_cpu]] += 1
        if bool(bad_cpu.any()):
            outlier_rows = rows_cpu[bad_cpu]
            self.map._anchor_outlier_obs[outlier_rows] += 1
            self.map._anchor_last_seen_kf[outlier_rows] = int(frontend_output.frame_id)
        reset_rows = torch.zeros(0, dtype=torch.long)
        if self.pfgs360_reset_after_outliers > 0:
            reset_rows = torch.nonzero(
                self.map._anchor_outlier_obs >= int(self.pfgs360_reset_after_outliers),
                as_tuple=False,
            ).flatten()
            if reset_rows.numel() > 0:
                rows_dev = reset_rows.to(device=device)
                low_opacity = torch.full((int(rows_dev.numel()), 1), 0.01, device=device, dtype=dtype)
                with torch.no_grad():
                    self.map.opacity_logit.data[rows_dev] = torch.minimum(
                        self.map.opacity_logit.data[rows_dev],
                        self.map._inv_sigmoid(low_opacity),
                    )
                stats["outlier_resets"] = int(reset_rows.numel())
        if self.pfgs360_prune_after_outliers > 0 and self.pfgs360_max_prune_per_keyframe > 0:
            old_enough = self.map._anchor_birth_frame <= int(frontend_output.frame_id) - int(self.pfgs360_protect_recent_keyframes)
            prune_rows = torch.nonzero(
                (self.map._anchor_outlier_obs >= int(self.pfgs360_prune_after_outliers)) & old_enough,
                as_tuple=False,
            ).flatten()
            if prune_rows.numel() > self.pfgs360_max_prune_per_keyframe:
                prune_rows = prune_rows[: self.pfgs360_max_prune_per_keyframe]
            if prune_rows.numel() > 0:
                prune_mask = torch.zeros(n, dtype=torch.bool)
                prune_mask[prune_rows] = True
                stats["outlier_pruned"] = self.map.prune_anchors(prune_mask.to(device=device))
        return stats

    def _build_voxel_index(self) -> dict[tuple[int, int, int, int], int]:
        occupied: dict[tuple[int, int, int, int], int] = {}
        if self.map._anchor_grid_coord.numel() == 0:
            return occupied
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            occupied.setdefault((int(level), int(coord[0]), int(coord[1]), int(coord[2])), int(idx))
        return occupied

    @staticmethod
    def _seed_voxel_key_from_cpu(
        xyz_cpu: torch.Tensor,
        scale_cpu: torch.Tensor,
        level_cpu: torch.Tensor,
        seed_idx: int,
    ) -> tuple[int, int, int, int]:
        level = int(level_cpu[seed_idx])
        scale = float(scale_cpu[seed_idx])
        coord = torch.floor(xyz_cpu[seed_idx] / scale).to(torch.int32)
        return (level, int(coord[0]), int(coord[1]), int(coord[2]))

    def _find_voxel_hit(
        self,
        occupied: dict[tuple[int, int, int, int], int],
        key: tuple[int, int, int, int],
        *,
        radius: int | None = None,
    ) -> int | None:
        level, x, y, z = key
        radius = int(self.voxel_neighbor_radius if radius is None else radius)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    hit = occupied.get((level, x + dx, y + dy, z + dz))
                    if hit is not None:
                        return int(hit)
        return None

    def _accumulate_existing_observation(self, anchor_idx: int, confidence: float, *, frame_id: int | None = None) -> None:
        if int(anchor_idx) < 0:
            return
        if int(anchor_idx) >= int(self.map._anchor_obs_count.shape[0]):
            return
        self.map._anchor_obs_count[int(anchor_idx)] += 1
        self.map._anchor_conf_accum[int(anchor_idx)] += float(confidence)
        if frame_id is not None and int(anchor_idx) < int(self.map._anchor_last_seen_kf.shape[0]):
            self.map._anchor_last_seen_kf[int(anchor_idx)] = int(frame_id)

    @staticmethod
    def _subset_seeds(seeds: GaussianSeedBatch, keep_idx: torch.Tensor) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=seeds.xyz.index_select(0, keep_idx.to(device=seeds.xyz.device)),
            rgb=seeds.rgb.index_select(0, keep_idx.to(device=seeds.rgb.device)),
            confidence=seeds.confidence.index_select(0, keep_idx.to(device=seeds.confidence.device)),
            scale=seeds.scale.index_select(0, keep_idx.to(device=seeds.scale.device)),
            level=seeds.level.index_select(0, keep_idx.to(device=seeds.level.device)),
            frame_id=int(seeds.frame_id),
            source_flat_idx=(
                None
                if seeds.source_flat_idx is None
                else seeds.source_flat_idx.index_select(0, keep_idx.to(device=seeds.source_flat_idx.device))
            ),
            source_hw=seeds.source_hw,
            insert_enabled=(
                None
                if seeds.insert_enabled is None
                else seeds.insert_enabled.index_select(0, keep_idx.to(device=seeds.insert_enabled.device))
            ),
            insert_score=(
                None
                if seeds.insert_score is None
                else seeds.insert_score.index_select(0, keep_idx.to(device=seeds.insert_score.device))
            ),
            grid_coord=(
                None
                if seeds.grid_coord is None
                else seeds.grid_coord.index_select(0, keep_idx.to(device=seeds.grid_coord.device))
            ),
        )

    @staticmethod
    def _empty_seed_like(seeds: GaussianSeedBatch) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=seeds.xyz[:0],
            rgb=seeds.rgb[:0],
            confidence=seeds.confidence[:0],
            scale=seeds.scale[:0],
            level=seeds.level[:0],
            frame_id=int(seeds.frame_id),
            source_flat_idx=None if seeds.source_flat_idx is None else seeds.source_flat_idx[:0],
            source_hw=seeds.source_hw,
            insert_enabled=None if seeds.insert_enabled is None else seeds.insert_enabled[:0],
            insert_score=None if seeds.insert_score is None else seeds.insert_score[:0],
            grid_coord=None if seeds.grid_coord is None else seeds.grid_coord[:0],
        )

    def _register_keyframe(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        start: int,
        end: int,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        frame_id = int(frontend_output.frame_id)
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        base_c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        self.pose_deltas[frame_id] = PoseDelta(base_c2w).to(device=device)
        record = MapperKeyframe(
            frame_id=frame_id,
            image=image.detach().cpu().float(),
            gaussian_start=int(start),
            gaussian_end=int(end),
            sky_mask=sky_mask.detach().cpu().bool() if torch.is_tensor(sky_mask) else None,
            target_depth=self._keyframe_target_depth(frontend_output, image, sky_mask=sky_mask),
            depth_confidence=self._keyframe_depth_confidence(frontend_output, image, sky_mask=sky_mask),
        )
        self.keyframes = [kf for kf in self.keyframes if kf.frame_id != frame_id]
        self.keyframes.append(record)

    def _keyframe_target_depth(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.pfgs360_insertion_enabled:
            return None
        img = image.detach()
        if img.ndim == 4:
            img = img[0]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        depth, _ = self._target_depth_from_output(frontend_output, (H, W), torch.device("cpu"), torch.float32)
        if depth is None:
            return None
        if sky_mask is not None:
            mask = self._normalize_skybox_mask(sky_mask, height=H, width=W, device=torch.device("cpu"))
            depth = depth.masked_fill(mask.bool(), 0.0)
        return depth.detach().cpu().float()

    def _keyframe_depth_confidence(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.pfgs360_insertion_enabled:
            return None
        img = image.detach()
        if img.ndim == 4:
            img = img[0]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        _, conf = self._target_depth_from_output(frontend_output, (H, W), torch.device("cpu"), torch.float32)
        if conf is None:
            conf = torch.ones(1, H, W, dtype=torch.float32)
        if sky_mask is not None:
            mask = self._normalize_skybox_mask(sky_mask, height=H, width=W, device=torch.device("cpu"))
            conf = conf.masked_fill(mask.bool(), 0.0)
        return conf.detach().cpu().float()

    def refined_pose_c2w(self, frame_id: int) -> torch.Tensor | None:
        pose_delta = self.pose_deltas.get(int(frame_id))
        if pose_delta is None:
            return None
        return pose_delta().detach().cpu()

    def refined_keyframe_poses(self) -> list[tuple[int, torch.Tensor]]:
        out = []
        for keyframe in self.keyframes:
            pose = self.refined_pose_c2w(keyframe.frame_id)
            if pose is not None:
                out.append((int(keyframe.frame_id), pose))
        return out

    def apply_frontend_pose_updates(self, updates: dict[int, torch.Tensor]) -> int:
        """Replace registered keyframe pose bases with frontend graph updates."""

        if not updates:
            return 0
        device = self.map.get_xyz.device
        registered = {int(keyframe.frame_id) for keyframe in self.keyframes}
        applied = 0
        for frame_id, pose in updates.items():
            fid = int(frame_id)
            if fid not in registered:
                continue
            pose_t = pose.detach().to(device=device, dtype=self.map.get_xyz.dtype)
            if tuple(pose_t.shape) != (4, 4) or not torch.isfinite(pose_t).all():
                continue
            self.pose_deltas[fid] = PoseDelta(pose_t).to(device=device)
            applied += 1
        if applied > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        return applied

    def set_spherical_selfi_observation_geometry(
        self,
        frame_id: int,
        *,
        target_depth_local: torch.Tensor,
        depth_scale: float,
        owner_window_id: int,
        depth_confidence: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
    ) -> bool:
        """Register immutable local depth and materialize its current global scale."""

        observation = self.observations.get(int(frame_id))
        if observation is None:
            return False
        local = target_depth_local.detach().cpu().float()
        scale = float(depth_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid spherical-Selfi depth scale {scale!r} for frame {frame_id}")
        observation.target_depth_local = local
        observation.target_depth_scale = scale
        observation.owner_window_id = int(owner_window_id)
        observation.target_depth = local * scale
        if depth_confidence is not None:
            observation.depth_confidence = depth_confidence.detach().cpu().float()
        if sky_mask is not None:
            observation.sky_mask = sky_mask.detach().cpu().bool()
        return True

    def apply_frontend_geometry_updates(self, updates: dict[int, object]) -> int:
        """Apply global c2w and graph depth scale without rescaling depth twice."""

        if not updates:
            return 0
        pose_updates = {
            int(frame_id): torch.as_tensor(getattr(update, "pose_c2w"))
            for frame_id, update in updates.items()
        }
        applied = self.apply_frontend_pose_updates(pose_updates)
        for frame_id, update in updates.items():
            observation = self.observations.get(int(frame_id))
            if observation is None:
                continue
            depth_owner = int(
                observation.owner_window_id
                if observation.owner_window_id is not None
                else getattr(update, "depth_owner_window_id", getattr(update, "owner_window_id"))
            )
            scales_by_window = dict(getattr(update, "depth_scales_by_window", {}) or {})
            scale = float(scales_by_window.get(depth_owner, getattr(update, "depth_scale")))
            if not math.isfinite(scale) or scale <= 0.0:
                continue
            if observation.target_depth_local is None and observation.target_depth is not None:
                observation.target_depth_local = observation.target_depth.detach().cpu().float().clone()
            observation.target_depth_scale = scale
            if observation.owner_window_id is None:
                observation.owner_window_id = depth_owner
            if observation.target_depth_local is not None:
                observation.target_depth = observation.target_depth_local * scale
                for keyframe in self.keyframes:
                    if int(keyframe.frame_id) == int(frame_id):
                        keyframe.target_depth = observation.target_depth.detach().cpu().float()
        return applied

    def render_view(
        self,
        *,
        image: torch.Tensor,
        c2w: torch.Tensor,
        sky_mask: torch.Tensor | None = None,
    ) -> dict | None:
        if self.map.anchor_count() == 0 and not self.map.has_skybox:
            return None
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        with torch.no_grad():
            pkg = self.renderer.render(camera, self.map)
            return self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, sky_mask))

    def _skybox_optimization_mask_enabled(self) -> bool:
        return bool(self._skybox_mask_enabled() and getattr(self.map, "skybox_optimize", False))

    def _skybox_mask_enabled(self) -> bool:
        return bool(
            self.map.has_skybox
            and getattr(self.map, "skybox_optimization_mask_enable", True)
        )

    def _requires_frontend_sky_mask(self) -> bool:
        return self.sky_mask_source in {"panovggt", "panovggt_head", "pano_vggt", "m3", "m3_head"}

    def _resolve_input_sky_mask(
        self,
        image: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
        *,
        context: str,
    ) -> torch.Tensor | None:
        if sky_mask is not None:
            mask = sky_mask.detach().bool()
            if image is not None:
                img = image.detach()
                if img.ndim == 4:
                    img = img[0]
                if img.ndim == 3:
                    mask = self._normalize_skybox_mask(
                        mask,
                        height=int(img.shape[-2]),
                        width=int(img.shape[-1]),
                        device=torch.device("cpu"),
                    )
            return mask.detach().cpu().bool()
        if self._requires_frontend_sky_mask():
            raise ValueError(f"{context}: Mapping.sky_mask_source=panovggt_head requires explicit sky_mask.")
        return self._skybox_mask_from_image(image)

    def _skybox_mask_from_image(self, image: torch.Tensor | None) -> torch.Tensor | None:
        if self._requires_frontend_sky_mask():
            return None
        if image is None or not self._skybox_mask_enabled():
            return None
        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.ndim != 3 or int(img.shape[0]) != 3:
            return None
        mask = self.map._sky_mask_from_image(img.clamp(0.0, 1.0))
        return mask.detach().bool().view(1, int(img.shape[-2]), int(img.shape[-1]))

    @staticmethod
    def _normalize_skybox_mask(
        sky_mask: torch.Tensor,
        *,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = sky_mask.detach().bool()
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected sky mask as HxW, 1xHxW, or Bx1xHxW, got {tuple(sky_mask.shape)}")
        if tuple(mask.shape[-2:]) != (int(height), int(width)):
            mask = (
                F.interpolate(
                    mask.float().view(1, 1, *mask.shape[-2:]),
                    size=(int(height), int(width)),
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
        return mask.to(device=device).view(1, int(height), int(width))

    def _skybox_mask_for_target(
        self,
        target: torch.Tensor,
        sky_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._skybox_mask_enabled():
            return None
        H, W = int(target.shape[-2]), int(target.shape[-1])
        if sky_mask is None:
            sky_mask = self._skybox_mask_from_image(target)
        if sky_mask is None:
            return None
        return self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)

    def _apply_skybox_optimization_mask(
        self,
        render_pkg: dict,
        sky_mask: torch.Tensor | None,
    ) -> dict:
        if sky_mask is None or not self._skybox_mask_enabled():
            return render_pkg
        gs_rgb = render_pkg.get("gs_only")
        sky_rgb = render_pkg.get("sky_bg_only")
        trans = render_pkg.get("sky_bg_alpha")
        if not (torch.is_tensor(gs_rgb) and torch.is_tensor(sky_rgb) and torch.is_tensor(trans)):
            return render_pkg
        if sky_rgb.ndim != 3 or trans.ndim != 3:
            return render_pkg
        mask = self._normalize_skybox_mask(
            sky_mask,
            height=int(sky_rgb.shape[-2]),
            width=int(sky_rgb.shape[-1]),
            device=sky_rgb.device,
        ).to(dtype=sky_rgb.dtype)
        sky_rgb_masked = sky_rgb * mask
        out = dict(render_pkg)
        if bool(getattr(self.map, "skybox_force_sky_render", False)):
            out["render"] = (gs_rgb * (1.0 - mask) + sky_rgb_masked).clamp(0.0, 1.0)
            out["skybox_force_sky_render"] = True
        else:
            out["render"] = (gs_rgb + trans.to(sky_rgb) * sky_rgb_masked).clamp(0.0, 1.0)
            out["skybox_force_sky_render"] = False
        out["skybox_optimization_mask"] = mask
        return out

    def render_keyframe_diagnostic(self, frame_id: int) -> KeyframeRenderDiagnostic | None:
        """Render an optimized keyframe for post-optimization diagnostics."""

        if self.map.anchor_count() == 0 and not self.map.has_skybox:
            return None
        frame_id = int(frame_id)
        keyframe = next((kf for kf in self.keyframes if int(kf.frame_id) == frame_id), None)
        pose_delta = self.pose_deltas.get(frame_id)
        if keyframe is None or pose_delta is None:
            return None
        target = keyframe.image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        with torch.no_grad():
            camera = PanoRenderCamera(image_height=H, image_width=W, c2w=pose_delta().detach())
            pkg = self.renderer.render(camera, self.map)
            sky_mask = self._skybox_mask_for_target(target, keyframe.sky_mask)
            pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
            non_sky_mask = None if sky_mask is None else ~sky_mask.to(device=target.device, dtype=torch.bool)
            loss, _ = backend_render_loss(
                pkg,
                target,
                depth_mask=non_sky_mask,
                weights=self.loss_weights,
            )
            render = pkg["render"].detach()
            mse = torch.mean((render - target).square()).clamp_min(1e-12)
            psnr = -10.0 * torch.log10(mse)
            depth = pkg.get("depth")
            return KeyframeRenderDiagnostic(
                frame_id=frame_id,
                target=target.detach().cpu(),
                render=render.cpu(),
                depth=depth.detach().cpu() if torch.is_tensor(depth) else None,
                target_depth=(
                    keyframe.target_depth.detach().cpu()
                    if torch.is_tensor(keyframe.target_depth)
                    else None
                ),
                loss=float(loss.detach().cpu()),
                psnr=float(psnr.detach().cpu()),
                anchor_count=self.map.anchor_count(),
                phase=self.stats.last_phase,
            )

    def refine_on_keyframe(
        self,
        *,
        image: torch.Tensor,
        c2w: torch.Tensor,
        steps: int = 1,
        sky_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if (self.map.anchor_count() == 0 and not self.map.has_skybox) or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        sky_mask = self._skybox_mask_for_target(target, sky_mask)
        non_sky_mask = None if sky_mask is None else ~sky_mask.to(device=target.device, dtype=torch.bool)
        last = {"loss": 0.0}
        for _ in range(int(steps)):
            self.optimizer.zero_grad(set_to_none=True)
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
            loss, metrics = backend_render_loss(
                pkg,
                target,
                depth_mask=non_sky_mask,
                weights=self.loss_weights,
            )
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = "legacy_keyframe"
        self.stats.optimization_steps += int(steps)
        return last

    def optimize_feedforward_window(
        self,
        *,
        current_frame_ids,
        history_frame_ids=None,
        chunk_index: int | None = None,
        active_keyframe_ids=None,
    ) -> dict[str, float]:
        if not self.uses_joint_optimization or not self.feedforward_window_enabled:
            return {}
        cfg = self._feedforward_window_cfg()
        neural_first_chunk = self._neural_should_train_mlp_for_chunk(chunk_index)
        if neural_first_chunk:
            steps = int(self.optim_cfg.get("first_chunk_steps", self.optim_cfg.get("steps_per_chunk", cfg.get("steps", 200))))
        elif self.pfgs360_replace_fuse_enabled and chunk_index is not None and int(chunk_index) == 0:
            steps = int(self.optim_cfg.get("first_chunk_steps", self.optim_cfg.get("steps_per_chunk", cfg.get("steps", 200))))
        elif self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            steps = int(self.optim_cfg.get("steps_per_chunk", cfg.get("steps", 200)))
        else:
            steps = int(cfg.get("steps", self.optim_cfg.get("sliding_window_steps", 0)))
        window_ids = self._feedforward_window_ids(current_frame_ids, history_frame_ids)
        observations = self._selected_observations_for_ids(window_ids)
        if not observations:
            return {"loss": 0.0, "steps": 0.0, "window_size": 0.0}
        if not bool(cfg.get("optimize_non_keyframe_observations", True)):
            observations = [obs for obs in observations if bool(obs.is_keyframe)]
        if not observations:
            return {"loss": 0.0, "steps": 0.0, "window_size": 0.0}
        total_start = time.perf_counter()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        selected_keyframe_ids = self._feedforward_keyframe_ids(window_ids)
        current_keyframe_ids = self._feedforward_keyframe_ids(current_frame_ids)
        if active_keyframe_ids is None:
            active_keyframe_ids_for_update = list(selected_keyframe_ids)
        else:
            active_keyframe_ids_for_update = []
            for fid in active_keyframe_ids:
                value = int(fid)
                if value not in active_keyframe_ids_for_update:
                    active_keyframe_ids_for_update.append(value)
        gaussian_scales = self._feedforward_gaussian_scales(active_keyframe_ids_for_update)
        gaussian_enabled = (
            gaussian_scales is not None
            and self.map.anchor_count() > 0
            and bool((gaussian_scales > 0).any().detach().cpu())
        )
        pose_enabled = bool(self.optim_cfg.get("pose_refine_enable", False))
        trainable_pose_ids = self._feedforward_trainable_pose_ids(current_keyframe_ids, pose_enabled=pose_enabled)
        pose_params = [
            self.pose_deltas[fid].delta
            for fid in trainable_pose_ids
            if fid in self.pose_deltas
        ]
        param_groups = self._map_param_groups(gaussian_enabled=gaussian_enabled, phase="feedforward_window")
        if pose_params:
            param_groups.append({"params": pose_params, "lr": float(self.optim_cfg.get("pose_lr", 1e-3))})
        if not param_groups:
            sky_pruned = (
                self._prune_sky_observations(observations)
                if self.pfgs360_replace_fuse_enabled and self.replace_fuse_sky_prune_enabled
                else 0
            )
            sky_compacted = (
                self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
                if self.pfgs360_replace_fuse_enabled
                else 0
            )
            self.stats.last_sky_pruned = int(sky_pruned)
            self.stats.last_sky_compacted = int(sky_compacted)
            self.stats.last_window_size = int(len(observations))
            self.stats.last_window_observations = [int(obs.frame_id) for obs in observations]
            self.stats.last_window_keyframes = list(selected_keyframe_ids)
            self.stats.last_active_keyframes = list(active_keyframe_ids_for_update)
            self.stats.last_feedforward_current_frames = [int(fid) for fid in current_frame_ids or []]
            hist_limit = max(0, int(cfg.get("history_keyframes", 2)))
            self.stats.last_feedforward_history_frames = [int(fid) for fid in (history_frame_ids or [])][-hist_limit:]
            self.stats.last_sampled_keyframes = []
            self.stats.last_trainable_pose_count = int(len(trainable_pose_ids))
            active_gaussian_count = (
                int((gaussian_scales > 0).sum().detach().cpu())
                if gaussian_scales is not None
                else 0
            )
            return {
                "loss": 0.0,
                "steps": 0.0,
                "window_size": float(len(observations)),
                "feedforward_window_size": float(len(observations)),
                "feedforward_keyframe_count": float(len(selected_keyframe_ids)),
                "active_keyframe_count": float(len(active_keyframe_ids_for_update)),
                "active_gaussian_count": float(active_gaussian_count),
                "sky_pruned": float(sky_pruned),
                "sky_compacted": float(sky_compacted),
                "profile_backend_feedforward_window_sky_pruned": float(sky_pruned),
                "profile_backend_feedforward_window_compacted": float(sky_compacted),
                "profile_backend_feedforward_window_sec": float(time.perf_counter() - total_start),
            }
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        pose_prior_weight = float(self.optim_cfg.get("pose_prior_weight", 1e-3))
        min_delta, patience = self._early_stop_options()
        best = float("inf")
        stale = 0
        actual_steps = 0
        last: dict[str, float] = {"loss": 0.0}
        last_sampled_ids: list[int] = []
        sample_sec = 0.0
        render_loss_sec = 0.0
        loss_eval_sec = 0.0
        backward_step_sec = 0.0
        renderer_profile_totals: dict[str, float] = {}
        renderer_profile_calls = 0
        sky_pruned_total = 0
        chunk_compacted = 0
        sampling_schedule: list[MapperObservation] | None = None
        sampled_frame_counts: dict[int, int] = {}
        if str(cfg.get("sampler", "")).lower() == "shuffled_cycle":
            rng = random.Random(int(cfg.get("sampler_seed", 123)) + sum(int(obs.frame_id) for obs in observations))
            sampling_schedule = []
            while len(sampling_schedule) < max(0, steps):
                cycle = list(observations)
                rng.shuffle(cycle)
                sampling_schedule.extend(cycle)
            sampling_schedule = sampling_schedule[: max(0, steps)]
        non_finite_window = False
        for step_idx in range(max(0, steps)):
            optimizer.zero_grad(set_to_none=True)
            render_losses = []
            metric_accum: dict[str, list[torch.Tensor]] = {}
            section_start = time.perf_counter()
            sampled = (
                [sampling_schedule[step_idx]]
                if sampling_schedule is not None
                else self._sample_observations_for_step(observations)
            )
            sample_sec += time.perf_counter() - section_start
            last_sampled_ids = [int(obs.frame_id) for obs in sampled]
            for frame_id in last_sampled_ids:
                sampled_frame_counts[frame_id] = sampled_frame_counts.get(frame_id, 0) + 1
            section_start = time.perf_counter()
            for obs in sampled:
                target = obs.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                c2w = self._observation_pose(obs, trainable_pose_ids=trainable_pose_ids).to(device=device, dtype=dtype)
                camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
                pkg = self.renderer.render(camera, self.map)
                renderer_profile_calls += 1
                for key, value in pkg.items():
                    if not str(key).startswith("profile_renderer_"):
                        continue
                    try:
                        scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                    except (TypeError, ValueError):
                        continue
                    renderer_profile_totals[str(key)] = renderer_profile_totals.get(str(key), 0.0) + scalar
                sky_mask = self._skybox_mask_for_target(target, obs.sky_mask)
                pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
                non_sky_mask = None if sky_mask is None else ~sky_mask.to(device=device, dtype=torch.bool)
                target_depth = None if obs.target_depth is None else obs.target_depth.to(device=device, dtype=dtype)
                depth_confidence = None if obs.depth_confidence is None else obs.depth_confidence.to(device=device, dtype=dtype)
                loss_start = time.perf_counter()
                loss_i, metrics_i = backend_render_loss(
                    pkg,
                    target,
                    target_depth=target_depth,
                    depth_confidence=depth_confidence,
                    depth_mask=non_sky_mask,
                    weights=self.loss_weights,
                )
                loss_eval_sec += time.perf_counter() - loss_start
                if sky_mask is not None:
                    metrics_i = dict(metrics_i)
                    metrics_i["skybox_mask_ratio"] = sky_mask.to(device=device, dtype=dtype).mean().detach()
                render_losses.append(loss_i)
                for key, value in metrics_i.items():
                    metric_accum.setdefault(key, []).append(value.detach())
            if not render_losses:
                break
            render_loss_sec += time.perf_counter() - section_start
            loss = torch.stack(render_losses).mean()
            if pose_params and pose_prior_weight > 0.0:
                prior = torch.stack([param.square().mean() for param in pose_params]).mean()
                loss = loss + pose_prior_weight * prior
            extra_loss_fn = getattr(self, "_spherical_selfi_extra_loss_fn", None)
            if callable(extra_loss_fn):
                extra_loss = extra_loss_fn(trainable_pose_ids)
                if torch.is_tensor(extra_loss):
                    loss = loss + extra_loss.to(loss)
                    metric_accum.setdefault("graph_factor_loss", []).append(extra_loss.detach())
            if not bool(torch.isfinite(loss).detach().cpu()):
                non_finite_window = True
                break
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                true_update_scaling = bool(
                    self.optim_cfg.get("scale_gaussian_parameter_updates", False)
                )
                if (
                    gaussian_enabled
                    and gaussian_scales is not None
                    and not true_update_scaling
                ):
                    self._apply_gaussian_grad_scales(gaussian_scales)
                pose_grad_clip = float(self.optim_cfg.get("pose_grad_clip", 0.0))
                if pose_params and pose_grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(pose_params, max_norm=pose_grad_clip)
                gradients_finite = all(
                    value.grad is None or bool(torch.isfinite(value.grad).all().detach().cpu())
                    for group in param_groups
                    for value in group["params"]
                )
                if not gradients_finite:
                    non_finite_window = True
                    break
                optimizer.step()
                if gaussian_enabled and gaussian_scales is not None and true_update_scaling:
                    self._apply_gaussian_adamw_update_scales(optimizer, gaussian_scales)
                with torch.no_grad():
                    if hasattr(self.map, "rotation") and torch.is_tensor(self.map.rotation):
                        self.map.rotation.copy_(F.normalize(self.map.rotation, dim=-1, eps=1.0e-8))
                    if hasattr(self.map, "scaling") and torch.is_tensor(self.map.scaling):
                        self.map.scaling.clamp_(-20.0, 20.0)
                    if hasattr(self.map, "opacity_logit") and torch.is_tensor(self.map.opacity_logit):
                        self.map.opacity_logit.clamp_(-12.0, 12.0)
                parameters_finite = all(
                    bool(torch.isfinite(value).all().detach().cpu())
                    for group in param_groups
                    for value in group["params"]
                )
                if not parameters_finite:
                    non_finite_window = True
                    break
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            actual_steps = step_idx + 1
            last = {
                key: float(torch.stack(values).mean().detach().cpu())
                for key, values in metric_accum.items()
                if values
            }
            last["loss"] = float(loss.detach().cpu())
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(actual_steps)
                        break
        active_mask = None
        if self.map.anchor_count() > 0:
            active_mask = self._active_anchor_mask_for_keyframes(active_keyframe_ids_for_update)
        prune_stats = (
            {"opacity_resets": 0, "pruned": 0}
            if self.pfgs360_replace_fuse_enabled or bool(cfg.get("skip_prune", False))
            else self._maybe_prune_feedforward_window(
                observations,
                active_mask=active_mask,
                selected_keyframe_ids=selected_keyframe_ids,
            )
        )
        if self.pfgs360_replace_fuse_enabled:
            if self.replace_fuse_sky_prune_enabled:
                sky_pruned_total = self._prune_sky_observations(observations)
            chunk_compacted = self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
        pose_norm = self._pose_delta_norm(trainable_pose_ids)
        total_sec = float(time.perf_counter() - total_start)
        last["steps"] = float(actual_steps)
        last["non_finite_window"] = float(non_finite_window)
        for frame_id, count in sampled_frame_counts.items():
            last[f"sample_count_frame_{int(frame_id)}"] = float(count)
        last["pose_delta_norm"] = pose_norm
        last["window_size"] = float(len(observations))
        last["feedforward_window_size"] = float(len(observations))
        last["feedforward_keyframe_count"] = float(len(selected_keyframe_ids))
        last["active_keyframe_count"] = float(len(active_keyframe_ids_for_update))
        last["active_gaussian_count"] = (
            float(int((gaussian_scales > 0).sum().detach().cpu()))
            if gaussian_scales is not None
            else 0.0
        )
        last["sampled_window_size"] = float(len(last_sampled_ids))
        last["last_sampled_keyframe"] = float(last_sampled_ids[0]) if last_sampled_ids else -1.0
        last["trainable_pose_count"] = float(len(trainable_pose_ids))
        last["frontend_graph_window_hint_count"] = float(len(self.frontend_graph_window_ids))
        last["feedforward_opacity_resets"] = float(prune_stats.get("opacity_resets", 0))
        last["feedforward_pruned"] = float(prune_stats.get("pruned", 0))
        last["sky_pruned"] = float(sky_pruned_total)
        last["sky_compacted"] = float(chunk_compacted)
        last["profile_backend_feedforward_window_sky_pruned"] = float(sky_pruned_total)
        last["profile_backend_feedforward_window_compacted"] = float(chunk_compacted)
        last["profile_backend_feedforward_window_sec"] = total_sec
        last["profile_backend_feedforward_window_step_avg_sec"] = total_sec / max(1, actual_steps)
        last["profile_backend_feedforward_window_sample_sec"] = float(sample_sec)
        last["profile_backend_feedforward_window_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_feedforward_window_loss_eval_sec"] = float(loss_eval_sec)
        last["profile_backend_feedforward_window_backward_step_sec"] = float(backward_step_sec)
        last["profile_backend_feedforward_window_renderer_calls"] = float(renderer_profile_calls)
        for key, total in renderer_profile_totals.items():
            suffix = str(key).removeprefix("profile_renderer_")
            out_key = f"profile_backend_feedforward_window_renderer_{suffix}"
            last[out_key] = float(total)
            if renderer_profile_calls > 0:
                if suffix.endswith("_sec"):
                    avg_key = f"profile_backend_feedforward_window_renderer_{suffix[:-4]}_avg_sec"
                else:
                    avg_key = f"profile_backend_feedforward_window_renderer_{suffix}_avg"
                last[avg_key] = float(total) / float(renderer_profile_calls)
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = "feedforward_window"
        self.stats.last_pose_delta_norm = pose_norm
        self.stats.last_window_size = int(len(observations))
        self.stats.last_window_observations = [int(obs.frame_id) for obs in observations]
        self.stats.last_window_keyframes = list(selected_keyframe_ids)
        self.stats.last_active_keyframes = list(active_keyframe_ids_for_update)
        self.stats.last_feedforward_current_frames = [int(fid) for fid in current_frame_ids or []]
        hist_limit = max(0, int(cfg.get("history_keyframes", 2)))
        self.stats.last_feedforward_history_frames = [int(fid) for fid in (history_frame_ids or [])][-hist_limit:]
        self.stats.last_sampled_keyframes = list(last_sampled_ids)
        self.stats.last_trainable_pose_count = int(len(trainable_pose_ids))
        self.stats.last_feedforward_opacity_resets = int(prune_stats.get("opacity_resets", 0))
        self.stats.last_feedforward_pruned = int(prune_stats.get("pruned", 0))
        self.stats.last_sky_pruned = int(sky_pruned_total)
        self.stats.last_sky_compacted = int(chunk_compacted)
        self.stats.optimization_steps += int(actual_steps)
        if neural_first_chunk and actual_steps > 0 and hasattr(self.map, "freeze_mlp"):
            self.map.freeze_mlp()
            self._neural_first_chunk_optimized = True
            last["neural_mlp_frozen"] = 1.0
        else:
            last["neural_mlp_frozen"] = float(int(bool(getattr(self.map, "mlp_frozen", False))))
        return last

    def _neural_should_train_mlp_for_chunk(self, chunk_index: int | None) -> bool:
        if not self.neural_anchor_mode:
            return False
        neural_cfg = self.map.config.get("NeuralScaffold", {}) if isinstance(self.map.config, dict) else {}
        if not bool(neural_cfg.get("freeze_mlp_after_first_chunk", True)):
            return False
        if self._neural_first_chunk_optimized or bool(getattr(self.map, "mlp_frozen", False)):
            return False
        return True

    def _feedforward_window_ids(self, current_frame_ids, history_frame_ids=None) -> list[int]:
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            current_per_chunk = max(
                1,
                int(self.optim_cfg.get("current_chunk_observation_frames", cfg.get("current_chunk_observation_frames", 4))),
            )
            recent_chunks = max(
                1,
                int(self.optim_cfg.get("recent_chunk_observation_chunks", cfg.get("recent_chunk_observation_chunks", 1))),
            )
            current_limit = max(1, current_per_chunk * recent_chunks)
            recent_keyframes = max(
                0,
                int(self.optim_cfg.get("recent_keyframe_observation_frames", cfg.get("recent_keyframe_observation_frames", 2))),
            )
            ids: list[int] = []
            current = [] if current_frame_ids is None else [int(fid) for fid in current_frame_ids]
            for fid in current[-current_limit:]:
                if fid not in ids:
                    ids.append(fid)
            if recent_keyframes > 0:
                for keyframe in self.keyframes[-recent_keyframes:]:
                    fid = int(keyframe.frame_id)
                    if fid not in ids:
                        ids.append(fid)
            return ids
        hist_limit = max(0, int(cfg.get("history_keyframes", 2)))
        ids: list[int] = []
        history = [] if history_frame_ids is None else [int(fid) for fid in history_frame_ids]
        current = [] if current_frame_ids is None else [int(fid) for fid in current_frame_ids]
        for fid in history[-hist_limit:] if hist_limit > 0 else []:
            if fid not in ids:
                ids.append(fid)
        for fid in current:
            if fid not in ids:
                ids.append(fid)
        return ids

    def _selected_observations_for_ids(self, frame_ids: list[int]) -> list[MapperObservation]:
        selected: list[MapperObservation] = []
        for fid in frame_ids:
            obs = self.observations.get(int(fid))
            if obs is not None:
                selected.append(obs)
        return selected

    def _feedforward_keyframe_ids(self, frame_ids) -> list[int]:
        registered = {int(kf.frame_id) for kf in self.keyframes}
        ids: list[int] = []
        for fid in frame_ids or []:
            value = int(fid)
            if value in registered and value not in ids:
                ids.append(value)
        return ids

    def _feedforward_trainable_pose_ids(self, current_keyframe_ids: list[int], *, pose_enabled: bool) -> set[int]:
        if not pose_enabled:
            return set()
        fixed = {int(fid) for fid in self.optim_cfg.get("fixed_pose_frame_ids", [])}
        return {
            int(fid)
            for fid in current_keyframe_ids
            if int(fid) in self.pose_deltas and int(fid) not in fixed
        }

    def _active_anchor_mask_for_keyframes(self, keyframe_ids: list[int]) -> torch.Tensor:
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        if n <= 0:
            return torch.zeros(n, dtype=torch.bool, device=device)
        if self.pfgs360_replace_fuse_enabled:
            return self._active_anchor_mask_for_keyframe_update_ids(keyframe_ids)
        if not keyframe_ids:
            return torch.zeros(n, dtype=torch.bool, device=device)
        birth = self.map._anchor_birth_frame.to(device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for fid in keyframe_ids:
            mask |= birth == int(fid)
        return mask

    def _active_anchor_mask_for_recent_updates(self) -> torch.Tensor:
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        if n <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        recent_kfs = max(1, int(self.optim_cfg.get("recent_insert_keyframes", 2)))
        latest_ord = max(0, int(self.stats.n_keyframes) - 1)
        min_ord = max(0, latest_ord - recent_kfs + 1)
        if int(self.map._anchor_last_update_kf_ord.shape[0]) != n:
            return torch.zeros(n, dtype=torch.bool, device=device)
        return self.map._anchor_last_update_kf_ord.to(device=device) >= int(min_ord)

    def _active_anchor_mask_for_keyframe_update_ids(self, keyframe_ids: list[int]) -> torch.Tensor:
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        if n <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if not keyframe_ids or int(self.map._anchor_last_update_kf_ord.shape[0]) != n:
            return torch.zeros(n, dtype=torch.bool, device=device)
        keyframe_set = {int(fid) for fid in keyframe_ids}
        ords = [
            int(ord_idx)
            for ord_idx, keyframe in enumerate(self.keyframes)
            if int(keyframe.frame_id) in keyframe_set
        ]
        if not ords:
            return torch.zeros(n, dtype=torch.bool, device=device)
        ord_tensor = torch.tensor(ords, dtype=torch.int32, device=device)
        updates = self.map._anchor_last_update_kf_ord.to(device=device, dtype=torch.int32)
        return (updates.view(-1, 1) == ord_tensor.view(1, -1)).any(dim=1)

    def _feedforward_gaussian_scales(self, selected_keyframe_ids: list[int]) -> torch.Tensor | None:
        if not bool(self.optim_cfg.get("gaussian_refine_enable", True)):
            return None
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        scales = torch.zeros(n, device=device, dtype=dtype)
        if n <= 0:
            return scales
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled:
            active = self._active_anchor_mask_for_keyframe_update_ids(selected_keyframe_ids)
            scales[active] = float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("new_gaussian_lr_scale", 1.0)))
            return scales
        scope = str(cfg.get("gaussian_scope", "selected_birth_keyframes")).lower()
        if scope == "owner_window_visible":
            owner_window_id = cfg.get("active_owner_window_id")
            owner = getattr(self.map, "_anchor_owner_window_id", None)
            if owner_window_id is None or not torch.is_tensor(owner) or int(owner.numel()) != n:
                return scales
            scales.fill_(float(cfg.get("visible_neighbor_lr_scale", 0.1)))
            owner_mask = owner.to(device=device, dtype=torch.long) == int(owner_window_id)
            scales[owner_mask] = float(cfg.get("gaussian_lr_scale", 1.0))
            return scales
        if scope == "all":
            scales.fill_(float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("existing_gaussian_lr_scale", 0.1))))
            return scales
        active = self._active_anchor_mask_for_keyframes(selected_keyframe_ids)
        scales[active] = float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("new_gaussian_lr_scale", 1.0)))
        return scales

    def _sample_observations_for_step(self, observations: list[MapperObservation]) -> list[MapperObservation]:
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            sample_n = max(
                1,
                int(self.optim_cfg.get("sample_frames_per_step", cfg.get("sample_observations_per_step", 1))),
            )
            return random.sample(observations, min(sample_n, len(observations))) if len(observations) > 1 else list(observations)
        random_window = bool(cfg.get("random_observation_per_iter", False))
        if random_window and len(observations) > 1:
            sample_n = max(1, int(cfg.get("sample_observations_per_step", self.optim_cfg.get("sample_keyframes_per_step", 1))))
            return random.sample(observations, min(sample_n, len(observations)))
        return list(observations)

    def _observation_pose(self, obs: MapperObservation, *, trainable_pose_ids: set[int]) -> torch.Tensor:
        pose_delta = self.pose_deltas.get(int(obs.frame_id))
        if pose_delta is None:
            return obs.pose_c2w.detach()
        if int(obs.frame_id) in trainable_pose_ids:
            return pose_delta()
        return pose_delta().detach()

    def _sky_prune_mask_from_render_pkg(
        self,
        obs: MapperObservation,
        render_pkg: dict,
        sky_mask: torch.Tensor | None,
        c2w: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor | None:
        n = self.map.anchor_count()
        if n <= 0:
            return None
        if sky_mask is None and obs.sky_mask is not None:
            sky_mask = self._normalize_skybox_mask(
                obs.sky_mask,
                height=int(H),
                width=int(W),
                device=self.map.get_xyz.device,
            )
        if sky_mask is None:
            return None
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        visibility = render_pkg.get("visibility_filter")
        visible = torch.ones(n, dtype=torch.bool, device=device)
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            visible = visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(visible, as_tuple=False).flatten()
        if rows.numel() == 0:
            return None
        xyz = self.map.get_xyz.detach()
        w2c = invert_c2w(c2w.detach().to(device=device, dtype=dtype))
        xyz_h = torch.cat([xyz.index_select(0, rows), torch.ones(rows.numel(), 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        if not bool(valid.any()):
            return None
        rows = rows.index_select(0, torch.nonzero(valid, as_tuple=False).flatten())
        cam = cam[valid]
        pixels = bearing_to_erp_pixel(cam, int(H), int(W))
        ui = pixels[:, 0].round().long().remainder(int(W))
        vi = pixels[:, 1].round().long().clamp(0, int(H) - 1)
        sky = sky_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
        if not bool(sky.any()):
            return None
        prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
        prune_mask[rows[sky]] = True
        return prune_mask

    def _apply_sky_prune_mask(self, prune_mask: torch.Tensor) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            return 0
        mask = prune_mask.detach().to(device=self.map.get_xyz.device, dtype=torch.bool).view(-1)
        if int(mask.numel()) != n or not bool(mask.any().detach().cpu()):
            return 0
        pruned = self.map.prune_anchors(mask)
        if pruned > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            if self.pfgs360_replace_fuse_enabled:
                self._refresh_pfgs360_voxel_cache(compact=False)
        return int(pruned)

    def _prune_sky_observations(self, observations: list[MapperObservation]) -> int:
        if not observations or self.map.anchor_count() <= 0:
            return 0
        total = 0
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        with torch.no_grad():
            for obs in observations:
                if self.map.anchor_count() <= 0:
                    break
                target = obs.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                c2w = self._observation_pose(obs, trainable_pose_ids=set()).to(device=device, dtype=dtype)
                sky_mask = (
                    self._normalize_skybox_mask(obs.sky_mask, height=H, width=W, device=device)
                    if obs.sky_mask is not None
                    else self._skybox_mask_for_target(target)
                )
                if sky_mask is None:
                    continue
                pkg = self.renderer.render(PanoRenderCamera(image_height=H, image_width=W, c2w=c2w), self.map)
                mask = self._sky_prune_mask_from_render_pkg(obs, pkg, sky_mask, c2w, H, W)
                if mask is not None:
                    total += self._apply_sky_prune_mask(mask)
        return int(total)

    def _clamp_replace_fuse_scaling(self) -> None:
        if self.map.anchor_count() <= 0:
            return
        min_scale = max(1.0e-5, float(self.optim_cfg.get("gaussian_scale_min", 0.008)))
        max_scale = max(min_scale, float(self.optim_cfg.get("gaussian_scale_max", 0.08)))
        with torch.no_grad():
            scale = self.map.get_scaling.detach().clamp(min=min_scale, max=max_scale)
            self.map.scaling.data.copy_(torch.log(torch.expm1(scale.clamp_min(1.0e-5))))

    def _maybe_prune_feedforward_window(
        self,
        observations: list[MapperObservation],
        *,
        active_mask: torch.Tensor | None,
        selected_keyframe_ids: list[int],
    ) -> dict[str, int]:
        cfg = self._feedforward_window_cfg()
        prune_cfg = cfg.get("prune", {}) if isinstance(cfg, dict) else {}
        prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
        stats = {"opacity_resets": 0, "pruned": 0}
        if not bool(prune_cfg.get("enabled", False)):
            return stats
        n = self.map.anchor_count()
        if n <= 0 or active_mask is None or int(active_mask.numel()) != n or not bool(active_mask.any()):
            return stats
        with torch.no_grad():
            for obs in observations:
                self._accumulate_feedforward_prune_evidence(obs, active_mask=active_mask)
            device = self.map.get_xyz.device
            active_cpu = active_mask.detach().cpu().bool()
            outlier = self.map._anchor_outlier_obs[:n].detach().cpu().float()
            inlier = self.map._anchor_inlier_obs[:n].detach().cpu().float()
            seen = outlier + inlier
            bad_ratio = outlier / seen.clamp_min(1.0)
            reset_after = max(0, int(prune_cfg.get("reset_after_bad", 3)))
            prune_after = max(0, int(prune_cfg.get("prune_after_bad", 6)))
            min_seen = max(1, int(prune_cfg.get("min_seen", 2)))
            min_bad_ratio = float(prune_cfg.get("min_bad_ratio", 0.7))
            max_inlier_count = max(0, int(prune_cfg.get("max_inlier_count", 1)))
            opacity_after_reset = max(1.0e-5, min(1.0 - 1.0e-5, float(prune_cfg.get("opacity_after_reset", 0.01))))
            current_opacity = self.map.get_opacity.detach().cpu().view(-1)
            reset_rows = torch.zeros(0, dtype=torch.long)
            if reset_after > 0:
                reset_rows = torch.nonzero(
                    active_cpu
                    & (outlier >= float(reset_after))
                    & (bad_ratio >= min_bad_ratio)
                    & (current_opacity > opacity_after_reset * 1.5),
                    as_tuple=False,
                ).flatten()
                if reset_rows.numel() > 0:
                    rows_dev = reset_rows.to(device=device)
                    low_opacity = torch.full(
                        (int(rows_dev.numel()), 1),
                        opacity_after_reset,
                        device=device,
                        dtype=self.map.get_xyz.dtype,
                    )
                    self.map.opacity_logit.data[rows_dev] = torch.minimum(
                        self.map.opacity_logit.data[rows_dev],
                        self.map._inv_sigmoid(low_opacity),
                    )
                    stats["opacity_resets"] = int(reset_rows.numel())
            if prune_after > 0:
                reset_set = set(int(row) for row in reset_rows.tolist())
                prune_rows = torch.nonzero(
                    active_cpu
                    & (outlier >= float(prune_after))
                    & (seen >= float(min_seen))
                    & (bad_ratio >= min_bad_ratio)
                    & (inlier <= float(max_inlier_count)),
                    as_tuple=False,
                ).flatten()
                if reset_set:
                    keep_rows = [int(row) for row in prune_rows.tolist() if int(row) not in reset_set]
                    prune_rows = torch.tensor(keep_rows, dtype=torch.long)
                max_prune = max(0, int(prune_cfg.get("max_prune_per_window", cfg.get("max_prune_per_window", 500))))
                if max_prune > 0 and prune_rows.numel() > max_prune:
                    order = torch.argsort(outlier[prune_rows], descending=True)
                    prune_rows = prune_rows.index_select(0, order[:max_prune])
                if prune_rows.numel() > 0:
                    prune_mask = torch.zeros(n, dtype=torch.bool)
                    prune_mask[prune_rows] = True
                    stats["pruned"] = self.map.prune_anchors(prune_mask.to(device=device))
                    if stats["pruned"] > 0:
                        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
                        self.stats.n_anchors = self.map.anchor_count()
                        self._rebuild_keyframe_ranges_from_birth_frames()
                        self.last_inserted_range = self._range_for_latest_keyframe()
        if stats["opacity_resets"] or stats["pruned"]:
            self.stats.notes.append(
                (
                    "feedforward prune: "
                    f"window_keyframes={list(int(fid) for fid in selected_keyframe_ids)}, "
                    f"opacity_resets={stats['opacity_resets']}, pruned={stats['pruned']}"
                )
            )
        return stats

    def _accumulate_feedforward_prune_evidence(
        self,
        obs: MapperObservation,
        *,
        active_mask: torch.Tensor,
    ) -> None:
        n = self.map.anchor_count()
        if n <= 0 or int(active_mask.numel()) != n:
            return
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        target = obs.image.to(device=device, dtype=dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        c2w = self._observation_pose(obs, trainable_pose_ids=set()).to(device=device, dtype=dtype)
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
        pkg = self.renderer.render(camera, self.map)
        sky_mask = self._skybox_mask_for_target(target, obs.sky_mask)
        pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
        visibility = pkg.get("visibility_filter")
        visible = active_mask.detach().to(device=device, dtype=torch.bool)
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            visible = visible & visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(visible, as_tuple=False).flatten()
        if rows.numel() == 0:
            return
        xyz = self.map.get_xyz.detach()
        w2c = invert_c2w(c2w)
        xyz_h = torch.cat([xyz.index_select(0, rows), torch.ones(rows.numel(), 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        if not bool(valid.any()):
            return
        rows = rows.index_select(0, torch.nonzero(valid, as_tuple=False).flatten())
        cam = cam[valid]
        pixels = bearing_to_erp_pixel(cam, H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        sky_bad = torch.zeros(rows.shape[0], device=device, dtype=torch.bool)
        non_sky = torch.ones(rows.shape[0], device=device, dtype=torch.bool)
        if sky_mask is not None:
            sky = sky_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
            sky_bad = sky
            non_sky = ~sky
        depth_bad = torch.zeros_like(sky_bad)
        render_depth = pkg.get("depth")
        if torch.is_tensor(render_depth) and obs.target_depth is not None:
            prune_cfg = self._feedforward_window_cfg().get("prune", {})
            prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
            target_depth = obs.target_depth.to(device=device, dtype=dtype)
            if tuple(target_depth.shape[-2:]) != (H, W):
                target_depth = F.interpolate(target_depth.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            rd = render_depth.to(device=device, dtype=dtype)
            td = target_depth.to(device=device, dtype=dtype)
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(rd)
            valid_depth = torch.isfinite(td) & torch.isfinite(rd) & (td > 1.0e-6) & (rd > 1.0e-6)
            scale, shift = self._robust_depth_scale_shift(
                td,
                rd,
                valid_depth & (alpha.to(device=device, dtype=dtype) >= self.pfgs360_render_alpha_min),
            )
            aligned = (td * scale + shift).clamp_min(1.0e-6)
            rel = (aligned - rd).abs() / torch.maximum(aligned, rd).clamp_min(1.0e-6)
            depth_threshold = float(prune_cfg.get("depth_rel_threshold", self.pfgs360_render_depth_rel_threshold))
            depth_bad = (rel[0, vi, ui] > depth_threshold) & non_sky
        render = pkg.get("render")
        photo_bad = torch.zeros_like(sky_bad)
        if torch.is_tensor(render):
            prune_cfg = self._feedforward_window_cfg().get("prune", {})
            prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
            photo_error = (render.to(device=device, dtype=dtype) - target).abs().mean(dim=0)
            photo_threshold = float(prune_cfg.get("photo_error_threshold", self.pfgs360_photometric_error_threshold))
            photo_bad = (photo_error[vi, ui] > photo_threshold) & non_sky
        bad = sky_bad | depth_bad | photo_bad
        inlier = (~bad) & non_sky
        rows_cpu = rows.detach().cpu()
        bad_cpu = bad.detach().cpu()
        inlier_cpu = inlier.detach().cpu()
        if bool(bad_cpu.any()):
            self.map._anchor_outlier_obs[rows_cpu[bad_cpu]] += 1
            self.map._anchor_last_seen_kf[rows_cpu[bad_cpu]] = int(obs.frame_id)
        if bool(inlier_cpu.any()):
            self.map._anchor_inlier_obs[rows_cpu[inlier_cpu]] += 1

    def _rebuild_keyframe_ranges_from_birth_frames(self) -> None:
        birth = self.map._anchor_birth_frame.detach().cpu().long()
        for keyframe in self.keyframes:
            rows = torch.nonzero(birth == int(keyframe.frame_id), as_tuple=False).flatten()
            if rows.numel() == 0:
                keyframe.gaussian_start = 0
                keyframe.gaussian_end = 0
            else:
                keyframe.gaussian_start = int(rows.min().item())
                keyframe.gaussian_end = int(rows.max().item()) + 1

    def _range_for_latest_keyframe(self) -> tuple[int, int]:
        if not self.keyframes:
            return (0, 0)
        latest = self.keyframes[-1]
        return (int(latest.gaussian_start), int(latest.gaussian_end))

    def _range_for_birth_frame(self, frame_id: int) -> tuple[int, int]:
        birth = self.map._anchor_birth_frame.detach().cpu().long()
        rows = torch.nonzero(birth == int(frame_id), as_tuple=False).flatten()
        if rows.numel() == 0:
            return (0, 0)
        return (int(rows.min().item()), int(rows.max().item()) + 1)

    def optimize_after_keyframe(self) -> dict[str, float]:
        """Run local and sliding-window joint Gaussian/pose optimization."""
        if self.feedforward_window_enabled:
            return {}
        if not self.uses_joint_optimization or not self.keyframes:
            return {}
        total_start = time.perf_counter()
        metrics: dict[str, float] = {}
        keyframe_steps = int(self.optim_cfg.get("keyframe_steps", 0))
        if keyframe_steps > 0:
            optimize_pose = bool(self.optim_cfg.get("keyframe_optimize_pose", self.optim_cfg.get("pose_refine_enable", False)))
            section_start = time.perf_counter()
            keyframe_metrics = self._optimize_keyframe_set(
                [self.keyframes[-1]],
                steps=keyframe_steps,
                phase="keyframe",
                gaussian_scales=self._gaussian_scales_for_phase("keyframe", [self.keyframes[-1]]),
                pose_enabled=optimize_pose,
            )
            metrics["profile_backend_keyframe_sec"] = float(time.perf_counter() - section_start)
            metrics.update(keyframe_metrics)
            if "loss" in keyframe_metrics:
                metrics["keyframe_loss"] = keyframe_metrics["loss"]
        local_steps = int(self.optim_cfg.get("local_submap_steps", 0))
        if local_steps > 0:
            local_window = int(self.optim_cfg.get("local_window_keyframes", 2))
            selected = self._select_backend_keyframe_window(max(1, local_window))
            section_start = time.perf_counter()
            local_metrics = self._optimize_keyframe_set(
                selected,
                steps=local_steps,
                phase="local_submap",
                gaussian_scales=self._gaussian_scales_for_phase("local_submap", selected),
            )
            metrics["profile_backend_local_submap_sec"] = float(time.perf_counter() - section_start)
            metrics.update(local_metrics)
            if "loss" in local_metrics:
                metrics["local_loss"] = local_metrics["loss"]

        sliding_steps = int(self.optim_cfg.get("sliding_window_steps", 0))
        if sliding_steps > 0:
            window = int(self.optim_cfg.get("window_keyframes", 8))
            selected = self._select_backend_keyframe_window(max(1, window))
            section_start = time.perf_counter()
            sliding_metrics = self._optimize_keyframe_set(
                selected,
                steps=sliding_steps,
                phase="sliding_window",
                gaussian_scales=self._gaussian_scales_for_phase("sliding_window", selected),
            )
            metrics["profile_backend_sliding_window_sec"] = float(time.perf_counter() - section_start)
            metrics.update(sliding_metrics)
            if "loss" in sliding_metrics:
                metrics["sliding_loss"] = sliding_metrics["loss"]
        metrics["profile_backend_optimize_after_keyframe_sec"] = float(time.perf_counter() - total_start)
        return metrics

    def _select_backend_keyframe_window(self, latest_count: int) -> list[MapperKeyframe]:
        latest_count = max(1, int(latest_count))
        if not bool(self.optim_cfg.get("use_frontend_graph_window", False)):
            return self.keyframes[-latest_count:]
        by_id = {int(kf.frame_id): kf for kf in self.keyframes}
        latest_ids = [int(kf.frame_id) for kf in self.keyframes[-latest_count:]]
        hint_ids = [int(fid) for fid in self.frontend_graph_window_ids if int(fid) in by_id]
        selected_ids: set[int] = set(hint_ids) | set(latest_ids)
        cap = max(0, int(self.optim_cfg.get("frontend_graph_window_max_keyframes", 0)))
        if cap > 0 and len(selected_ids) > cap:
            ordered: list[int] = []
            for fid in reversed(hint_ids):
                if fid not in ordered:
                    ordered.append(fid)
                if len(ordered) >= cap:
                    break
            if len(ordered) < cap:
                for kf in reversed(self.keyframes):
                    fid = int(kf.frame_id)
                    if fid not in ordered:
                        ordered.append(fid)
                    if len(ordered) >= cap:
                        break
            selected_ids = set(ordered)
        return [kf for kf in self.keyframes if int(kf.frame_id) in selected_ids]

    def bootstrap_latest_keyframe(self, *, steps: int, diagnostic_callback=None, diagnostic_every: int = 0) -> dict[str, float]:
        if not self.keyframes or int(steps) <= 0:
            return {}
        selected = [self.keyframes[-1]]
        total_start = time.perf_counter()
        metrics = self._optimize_keyframe_set(
            selected,
            steps=int(steps),
            phase="bootstrap",
            gaussian_scales=self._gaussian_scales_for_phase("bootstrap", selected),
            pose_enabled=False,
            diagnostic_callback=diagnostic_callback,
            diagnostic_every=diagnostic_every,
        )
        if self.pfgs360_replace_fuse_enabled:
            compacted = self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
            metrics["bootstrap_compacted"] = float(compacted)
            metrics["profile_backend_bootstrap_compacted"] = float(compacted)
            self.stats.last_replace_compacted = int(compacted)
        metrics["profile_backend_bootstrap_sec"] = float(time.perf_counter() - total_start)
        return metrics

    def optimize_frame_observation(
        self,
        *,
        image: torch.Tensor,
        c2w: torch.Tensor,
        steps: int,
        phase: str = "non_keyframe",
    ) -> dict[str, float]:
        if self.feedforward_window_enabled:
            return {"loss": 0.0, "steps": 0.0}
        if (self.map.anchor_count() == 0 and not self.map.has_skybox) or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        total_start = time.perf_counter()
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        sky_mask = self._skybox_mask_for_target(target)
        gaussian_scales = self._gaussian_scales_for_phase(phase, [])
        param_groups = self._map_param_groups(
            gaussian_enabled=gaussian_scales is not None and self.map.anchor_count() > 0,
            phase=phase,
        )
        if not param_groups:
            return {"loss": 0.0, "steps": 0.0, "profile_backend_non_keyframe_sec": float(time.perf_counter() - total_start)}
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)))
        last: dict[str, float] = {"loss": 0.0}
        best = float("inf")
        stale = 0
        min_delta, patience = self._early_stop_options()
        render_loss_sec = 0.0
        backward_step_sec = 0.0
        for step_idx in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            section_start = time.perf_counter()
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
            non_sky_mask = None if sky_mask is None else ~sky_mask.to(device=target.device, dtype=torch.bool)
            loss, metrics = backend_render_loss(
                pkg,
                target,
                depth_mask=non_sky_mask,
                weights=self.loss_weights,
            )
            render_loss_sec += time.perf_counter() - section_start
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                if gaussian_scales is not None:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
            last["loss"] = float(loss.detach().cpu())
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(step_idx + 1)
                        break
        last["steps"] = float(steps if "early_stop_step" not in last else int(last["early_stop_step"]))
        last["pose_delta_norm"] = 0.0
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = phase
        self.stats.last_pose_delta_norm = 0.0
        self.stats.optimization_steps += int(last["steps"])
        total_sec = float(time.perf_counter() - total_start)
        last["profile_backend_non_keyframe_sec"] = total_sec
        last["profile_backend_non_keyframe_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_non_keyframe_backward_step_sec"] = float(backward_step_sec)
        last["profile_backend_non_keyframe_step_avg_sec"] = total_sec / max(1.0, float(last["steps"]))
        return last

    def finalize_optimization(self) -> dict[str, float]:
        """Run low-frequency global polish after the sequence/block is complete."""
        if self.feedforward_window_enabled and not bool(self._feedforward_window_cfg().get("allow_final_global", False)):
            return {}
        if not self.uses_joint_optimization or not self.keyframes:
            return {}
        steps = int(self.optim_cfg.get("final_global_steps", 0))
        if steps <= 0:
            return {}
        max_kfs = int(self.optim_cfg.get("final_global_max_keyframes", 0))
        selected = self.keyframes if max_kfs <= 0 else self.keyframes[-max(1, max_kfs) :]
        metrics = self._optimize_keyframe_set(
            selected,
            steps=steps,
            phase="final_global",
            gaussian_scales=self._gaussian_scales_for_phase("final_global", selected),
        )
        if self.pfgs360_replace_fuse_enabled:
            compacted = self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
            metrics["final_global_compacted"] = float(compacted)
            metrics["profile_backend_final_global_compacted"] = float(compacted)
            self.stats.last_replace_compacted = int(compacted)
        return metrics

    def _gaussian_scales_for_phase(self, phase: str, selected: list[MapperKeyframe]) -> torch.Tensor | None:
        if not bool(self.optim_cfg.get("gaussian_refine_enable", True)):
            return None
        n = self.map.anchor_count()
        if n <= 0:
            return torch.zeros(0, device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        scales = torch.zeros(n, device=device, dtype=dtype)
        if phase == "final_global":
            scales.fill_(float(self.optim_cfg.get("global_gaussian_lr_scale", 1.0)))
            return scales
        if phase == "bootstrap":
            scales.fill_(1.0)
            return scales
        if phase == "keyframe":
            scales.fill_(float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.15)))
            new_start, new_end = self.last_inserted_range
            if new_end > new_start:
                scales[new_start:new_end] = float(self.optim_cfg.get("new_gaussian_lr_scale", 1.0))
            return scales
        if phase == "non_keyframe":
            scales.fill_(float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.15)))
            return scales

        new_start, new_end = self.last_inserted_range
        mode = str(self.optim_cfg.get("optimize_existing_gaussians", "visible_recent")).lower()
        existing_scale = float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.1))
        if phase == "sliding_window" and mode == "all":
            scales.fill_(existing_scale)
        elif phase == "sliding_window" and mode in {"visible_recent", "window", "recent"}:
            for kf in selected:
                if kf.gaussian_end > kf.gaussian_start:
                    scales[kf.gaussian_start : kf.gaussian_end] = existing_scale
        elif phase == "sliding_window" and mode in {"none", "frozen"}:
            pass

        if new_end > new_start:
            scales[new_start:new_end] = 1.0
        return scales

    def _map_param_groups(self, *, gaussian_enabled: bool, phase: str) -> list[dict]:
        groups: list[dict] = []
        if gaussian_enabled:
            if hasattr(self.map, "get_optimizer_param_groups"):
                groups.extend(self.map.get_optimizer_param_groups())
            elif self.pfgs360_replace_fuse_enabled or bool(
                self.optim_cfg.get("separate_gaussian_lrs", False)
            ):
                feature_params = [self.map.features]
                sh_rest = getattr(self.map, "sh_rest", None)
                combine_sh_rest = self.pfgs360_replace_fuse_enabled and not bool(
                    self.optim_cfg.get("separate_gaussian_lrs", False)
                )
                if (
                    combine_sh_rest
                    and torch.is_tensor(sh_rest)
                    and sh_rest.ndim == 3
                    and int(sh_rest.shape[1]) > 0
                ):
                    feature_params.append(sh_rest)
                gaussian_groups = [
                        {
                            "params": [self.map.xyz],
                            "lr": float(self.optim_cfg.get("xyz_lr", 5.0e-4)),
                            "name": "xyz",
                        },
                        {
                            "params": feature_params,
                            "lr": float(self.optim_cfg.get("feature_lr", 2.0e-3)),
                            "name": "features",
                        },
                        {
                            "params": [self.map.opacity_logit],
                            "lr": float(self.optim_cfg.get("opacity_lr", 1.0e-3)),
                            "name": "opacity",
                        },
                        {
                            "params": [self.map.scaling],
                            "lr": float(self.optim_cfg.get("scaling_lr", 1.0e-4)),
                            "name": "scaling",
                        },
                        {
                            "params": [self.map.rotation],
                            "lr": float(self.optim_cfg.get("rotation_lr", 1.0e-4)),
                            "name": "rotation",
                        },
                    ]
                if (
                    not combine_sh_rest
                    and torch.is_tensor(sh_rest)
                    and sh_rest.ndim == 3
                    and int(sh_rest.shape[1]) > 0
                ):
                    gaussian_groups.insert(
                        2,
                        {
                            "params": [sh_rest],
                            "lr": float(self.optim_cfg.get("sh_rest_lr", 1.0e-4)),
                            "name": "sh_rest",
                        },
                    )
                groups.extend(gaussian_groups)
            else:
                groups.append(
                    {
                        "params": self.map.gaussian_parameters(),
                        "lr": float(self.optim_cfg.get("gaussian_lr", self.optimizer.param_groups[0]["lr"])),
                        "name": "gaussians",
                    }
                )
        if bool(self.optim_cfg.get("optimize_skybox", True)):
            sky_params = self.map.skybox_parameters()
            if sky_params:
                groups.append(
                    {
                        "params": sky_params,
                        "lr": float(self.optim_cfg.get("skybox_lr", getattr(self.map, "skybox_lr", 1.0e-2))),
                        "name": "skybox",
                    }
                )
        return groups

    def _early_stop_options(self) -> tuple[float, int]:
        cfg = self.optim_cfg.get("early_stop", {}) if isinstance(self.optim_cfg, dict) else {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return 0.0, 0
        return float(cfg.get("min_delta", 0.0)), max(0, int(cfg.get("patience", 0)))

    def _pose_trainable_keyframes(
        self,
        keyframes: list[MapperKeyframe],
        *,
        fixed: int,
        pose_enabled: bool,
    ) -> list[MapperKeyframe]:
        if not pose_enabled:
            return []
        candidates = keyframes[max(0, int(fixed)) :]
        pose_window = int(
            self.optim_cfg.get(
                "pose_window_keyframes",
                self.optim_cfg.get("pose_window", 0),
            )
        )
        if pose_window > 0:
            candidates = candidates[-pose_window:]
        return candidates

    def _sample_keyframes_for_step(
        self,
        keyframes: list[MapperKeyframe],
        *,
        selected_ids: set[int],
    ) -> tuple[list[MapperKeyframe], set[int]]:
        random_window = bool(self.optim_cfg.get("random_window_frame_per_iter", False))
        if random_window and len(keyframes) > 1:
            sample_n = max(1, int(self.optim_cfg.get("sample_keyframes_per_step", 1)))
            sample_n = min(sample_n, len(keyframes))
            sampled = random.sample(keyframes, sample_n)
        else:
            sampled = list(keyframes)

        replay_ids: set[int] = set()
        replay_n = max(0, int(self.optim_cfg.get("replay_random_keyframes", 0)))
        if replay_n > 0:
            outside = [kf for kf in self.keyframes if int(kf.frame_id) not in selected_ids]
            if outside:
                replay = random.sample(outside, min(replay_n, len(outside)))
                sampled.extend(replay)
                replay_ids = {int(kf.frame_id) for kf in replay}
        return sampled, replay_ids

    def _optimize_keyframe_set(
        self,
        keyframes: list[MapperKeyframe],
        *,
        steps: int,
        phase: str,
        gaussian_scales: torch.Tensor | None,
        pose_enabled: bool | None = None,
        diagnostic_callback=None,
        diagnostic_every: int = 0,
    ) -> dict[str, float]:
        if not keyframes or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        total_start = time.perf_counter()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        gaussian_enabled = gaussian_scales is not None and self.map.anchor_count() > 0
        pose_enabled = bool(self.optim_cfg.get("pose_refine_enable", False)) if pose_enabled is None else bool(pose_enabled)
        fixed = max(0, int(self.optim_cfg.get("fixed_window_frames", 1)))
        if phase == "final_global":
            fixed = max(1, fixed)
        trainable_pose_ids = {
            int(kf.frame_id)
            for kf in self._pose_trainable_keyframes(
                keyframes,
                fixed=fixed,
                pose_enabled=pose_enabled,
            )
        }
        pose_params = [
            self.pose_deltas[fid].delta
            for fid in trainable_pose_ids
            if fid in self.pose_deltas
        ]
        param_groups = self._map_param_groups(gaussian_enabled=gaussian_enabled, phase=phase)
        if pose_params:
            param_groups.append(
                {
                    "params": pose_params,
                    "lr": float(self.optim_cfg.get("pose_lr", 1e-3)),
                }
            )
        if not param_groups:
            return {"loss": 0.0, "steps": 0.0, f"profile_backend_{str(phase)}_optimize_set_sec": float(time.perf_counter() - total_start)}

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        pose_prior_weight = float(self.optim_cfg.get("pose_prior_weight", 1e-3))
        last: dict[str, float] = {"loss": 0.0}
        best = float("inf")
        stale = 0
        min_delta, patience = self._early_stop_options()
        actual_steps = 0
        selected_ids = {int(kf.frame_id) for kf in keyframes}
        last_sampled_ids: list[int] = []
        sample_sec = 0.0
        render_loss_sec = 0.0
        backward_step_sec = 0.0
        diagnostic_sec = 0.0
        for step_idx in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            render_losses = []
            metric_accum: dict[str, list[torch.Tensor]] = {}
            section_start = time.perf_counter()
            sampled_keyframes, replay_ids = self._sample_keyframes_for_step(
                keyframes,
                selected_ids=selected_ids,
            )
            sample_sec += time.perf_counter() - section_start
            last_sampled_ids = [int(kf.frame_id) for kf in sampled_keyframes]
            section_start = time.perf_counter()
            for kf in sampled_keyframes:
                target = kf.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                pose_delta = self.pose_deltas.get(kf.frame_id)
                if pose_delta is None:
                    continue
                frame_id = int(kf.frame_id)
                if frame_id in trainable_pose_ids and frame_id not in replay_ids:
                    c2w = pose_delta()
                else:
                    c2w = pose_delta().detach()
                camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
                pkg = self.renderer.render(camera, self.map)
                sky_mask = self._skybox_mask_for_target(target, kf.sky_mask)
                pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
                non_sky_mask = None if sky_mask is None else ~sky_mask.to(device=device, dtype=torch.bool)
                target_depth = None if kf.target_depth is None else kf.target_depth.to(device=device, dtype=dtype)
                depth_confidence = None if kf.depth_confidence is None else kf.depth_confidence.to(device=device, dtype=dtype)
                loss_i, metrics_i = backend_render_loss(
                    pkg,
                    target,
                    target_depth=target_depth,
                    depth_confidence=depth_confidence,
                    depth_mask=non_sky_mask,
                    weights=self.loss_weights,
                )
                if sky_mask is not None:
                    metrics_i = dict(metrics_i)
                    metrics_i["skybox_mask_ratio"] = sky_mask.to(device=device, dtype=dtype).mean().detach()
                render_losses.append(loss_i)
                for key, value in metrics_i.items():
                    metric_accum.setdefault(key, []).append(value.detach())
            if not render_losses:
                elapsed = float(time.perf_counter() - total_start)
                return {"loss": 0.0, "steps": 0.0, f"profile_backend_{str(phase)}_optimize_set_sec": elapsed}
            render_loss_sec += time.perf_counter() - section_start
            loss = torch.stack(render_losses).mean()
            if pose_params and pose_prior_weight > 0.0:
                prior = torch.stack([param.square().mean() for param in pose_params]).mean()
                loss = loss + pose_prior_weight * prior
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                if gaussian_enabled:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            actual_steps = step_idx + 1
            last = {
                key: float(torch.stack(values).mean().detach().cpu())
                for key, values in metric_accum.items()
                if values
            }
            last["loss"] = float(loss.detach().cpu())
            if diagnostic_callback is not None and len(keyframes) == 1:
                section_start = time.perf_counter()
                every = max(1, int(diagnostic_every))
                if actual_steps == 1 or actual_steps % every == 0 or actual_steps == int(steps):
                    diagnostic = self.render_keyframe_diagnostic(int(keyframes[0].frame_id))
                    if diagnostic is not None:
                        diagnostic_callback(actual_steps, diagnostic)
                diagnostic_sec += time.perf_counter() - section_start
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(actual_steps)
                        break
        pose_norm = self._pose_delta_norm(trainable_pose_ids)
        last["steps"] = float(actual_steps)
        last["pose_delta_norm"] = pose_norm
        last["window_size"] = float(len(keyframes))
        last["sampled_window_size"] = float(len(last_sampled_ids))
        last["last_sampled_keyframe"] = float(last_sampled_ids[0]) if last_sampled_ids else -1.0
        last["trainable_pose_count"] = float(len(trainable_pose_ids))
        last["frontend_graph_window_hint_count"] = float(len(self.frontend_graph_window_ids))
        phase_key = str(phase).replace("/", "_")
        total_sec = float(time.perf_counter() - total_start)
        last["profile_backend_optimize_set_sec"] = total_sec
        last["profile_backend_step_avg_sec"] = total_sec / max(1, actual_steps)
        last["profile_backend_sample_sec"] = float(sample_sec)
        last["profile_backend_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_backward_step_sec"] = float(backward_step_sec)
        last["profile_backend_diagnostic_sec"] = float(diagnostic_sec)
        last[f"profile_backend_{phase_key}_optimize_set_sec"] = total_sec
        last[f"profile_backend_{phase_key}_step_avg_sec"] = total_sec / max(1, actual_steps)
        last[f"profile_backend_{phase_key}_sample_sec"] = float(sample_sec)
        last[f"profile_backend_{phase_key}_render_loss_sec"] = float(render_loss_sec)
        last[f"profile_backend_{phase_key}_backward_step_sec"] = float(backward_step_sec)
        last[f"profile_backend_{phase_key}_diagnostic_sec"] = float(diagnostic_sec)
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = phase
        self.stats.last_pose_delta_norm = pose_norm
        self.stats.last_window_size = int(len(keyframes))
        self.stats.last_window_keyframes = [int(kf.frame_id) for kf in keyframes]
        self.stats.last_sampled_keyframes = list(last_sampled_ids)
        self.stats.last_trainable_pose_count = int(len(trainable_pose_ids))
        self.stats.optimization_steps += int(actual_steps)
        return last

    def _apply_gaussian_grad_scales(self, scales: torch.Tensor) -> None:
        if scales.numel() != self.map.anchor_count():
            return
        for param in self.map.gaussian_parameters():
            if param.grad is None or param.grad.shape[0] != scales.shape[0]:
                continue
            view_shape = (scales.shape[0],) + (1,) * (param.grad.ndim - 1)
            param.grad.mul_(scales.view(view_shape).to(device=param.grad.device, dtype=param.grad.dtype))

    @torch.no_grad()
    def _apply_gaussian_adamw_update_scales(
        self,
        optimizer: torch.optim.AdamW,
        scales: torch.Tensor,
    ) -> None:
        """Scale the realized AdamW step per Gaussian row.

        Scaling an Adam gradient by a constant is largely cancelled by the
        adaptive denominator.  This correction is applied after ``step()`` to
        the actual bias-corrected Adam update, making a neighbor scale of 0.1
        produce a true one-tenth parameter step without copying the full map.
        """

        if int(scales.numel()) != self.map.anchor_count():
            return
        gaussian_ids = {id(value) for value in self.map.gaussian_parameters()}
        for group in optimizer.param_groups:
            weight_decay = float(group.get("weight_decay", 0.0))
            gaussian_params = [value for value in group["params"] if id(value) in gaussian_ids]
            if gaussian_params and weight_decay != 0.0:
                raise ValueError(
                    "scale_gaussian_parameter_updates requires weight_decay=0 for exact AdamW scaling"
                )
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            epsilon = float(group.get("eps", 1.0e-8))
            learning_rate = float(group["lr"])
            amsgrad = bool(group.get("amsgrad", False))
            for parameter in gaussian_params:
                if parameter.grad is None:
                    continue
                if parameter.ndim == 0 or int(parameter.shape[0]) != int(scales.numel()):
                    continue
                state = optimizer.state.get(parameter, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_value = state.get("step")
                if exp_avg is None or exp_avg_sq is None or step_value is None:
                    continue
                step = float(step_value.detach().cpu()) if torch.is_tensor(step_value) else float(step_value)
                if step <= 0.0:
                    continue
                denominator_source = state.get("max_exp_avg_sq", exp_avg_sq) if amsgrad else exp_avg_sq
                bias_correction1 = 1.0 - float(beta1) ** step
                bias_correction2_sqrt = math.sqrt(1.0 - float(beta2) ** step)
                denominator = denominator_source.sqrt() / bias_correction2_sqrt
                denominator = denominator.add(epsilon)
                adam_delta = exp_avg / denominator
                adam_delta.mul_(-learning_rate / bias_correction1)
                view_shape = (int(scales.numel()),) + (1,) * (parameter.ndim - 1)
                correction = scales.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).view(view_shape) - 1.0
                parameter.add_(adam_delta.to(parameter) * correction)

    def _pose_delta_norm(self, frame_ids: set[int] | None = None) -> float:
        deltas = []
        ids = frame_ids if frame_ids is not None else set(self.pose_deltas)
        for fid in ids:
            pose_delta = self.pose_deltas.get(int(fid))
            if pose_delta is not None:
                deltas.append(pose_delta.delta.detach().norm())
        if not deltas:
            return 0.0
        return float(torch.stack(deltas).mean().cpu())
