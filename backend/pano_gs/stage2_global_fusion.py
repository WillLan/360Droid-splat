"""Vectorized Stage-2 Gaussian conversion, voxel fusion, and loop correction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn

from frontend.spherical_selfi.window_packet import LocalGaussianWindowPacket
from geometry.sim3 import apply_sim3, sim3_components, sim3_inverse
from models.per_pixel_gaussian_observation import (
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    real_sh_basis,
)

from .mapper import PanoGaussianMap


@dataclass
class GlobalExplicitGaussianBatch:
    xyz: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    opacity_parameter: torch.Tensor
    sh_coefficients: torch.Tensor
    quality: torch.Tensor
    owner_window_id: torch.Tensor
    level: torch.Tensor
    voxel_size: torch.Tensor
    grid_coord: torch.Tensor
    observation_count: torch.Tensor
    confidence_accum: torch.Tensor
    birth_frame: torch.Tensor
    last_seen_frame: torch.Tensor
    visibility_count: torch.Tensor
    render_error_ema: torch.Tensor
    replacement_hits: torch.Tensor
    inconsistency_hits: torch.Tensor

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def index(self, selected: torch.Tensor) -> "GlobalExplicitGaussianBatch":
        return GlobalExplicitGaussianBatch(
            **{name: getattr(self, name)[selected] for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class PreparedPacketFusion:
    batch: GlobalExplicitGaussianBatch
    source_anchor_indices: torch.Tensor | None
    requested: int
    depth_selected: bool

    def index(self, selected: torch.Tensor) -> "PreparedPacketFusion":
        source = self.source_anchor_indices
        return PreparedPacketFusion(
            batch=self.batch.index(selected),
            source_anchor_indices=(
                None if source is None else source.index_select(0, selected.to(source.device))
            ),
            requested=int(self.requested),
            depth_selected=bool(self.depth_selected),
        )


@dataclass(frozen=True)
class ExistingAnchorEvidenceUpdate:
    indices: torch.Tensor
    observation_count_delta: torch.Tensor
    confidence_accum_delta: torch.Tensor
    last_seen_frame: torch.Tensor


def _fibonacci_directions(count: int, *, device, dtype) -> torch.Tensor:
    index = torch.arange(max(16, int(count)), device=device, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / float(index.numel())
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack([radius * torch.cos(angle), z, radius * torch.sin(angle)], dim=-1)


def sh_rotation_matrix(rotation_local_to_target: torch.Tensor, degree: int) -> torch.Tensor:
    """Return the real-SH coefficient transform from local to target frame."""

    coefficient_count = (int(degree) + 1) ** 2
    directions_target = _fibonacci_directions(
        max(32, coefficient_count * 4),
        device=rotation_local_to_target.device,
        dtype=rotation_local_to_target.dtype,
    )
    target_basis = real_sh_basis(degree, directions_target)
    directions_local = directions_target @ rotation_local_to_target
    local_basis = real_sh_basis(degree, directions_local)
    return torch.linalg.pinv(target_basis) @ local_basis


def rotate_sh_coefficients(
    coefficients: torch.Tensor,
    rotation_local_to_target: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    matrix = sh_rotation_matrix(rotation_local_to_target.to(coefficients), degree)
    return torch.einsum("ij,...jc->...ic", matrix, coefficients)


class Stage2GlobalMapFusion:
    def __init__(
        self,
        gaussian_map: PanoGaussianMap,
        *,
        voxel_sizes: Iterable[float] = (0.04, 0.08, 0.16, 0.32),
        min_confidence: float = 0.05,
        min_opacity: float = 0.02,
        max_total_gaussians: int = 0,
        coverage_aware_budget: bool = False,
        coverage_coarse_cell_size: float = 0.64,
        lazy_owner_transforms: bool = False,
    ) -> None:
        self.map = gaussian_map
        values = sorted({float(value) for value in voxel_sizes if float(value) > 0.0})
        if not values:
            raise ValueError("voxel_sizes must contain at least one positive value")
        self.voxel_sizes = tuple(values)
        self.min_confidence = float(min_confidence)
        self.min_opacity = float(min_opacity)
        self.max_total_gaussians = max(0, int(max_total_gaussians))
        self.coverage_aware_budget = bool(coverage_aware_budget)
        self.coverage_coarse_cell_size = float(coverage_coarse_cell_size)
        if self.coverage_aware_budget and (
            not math.isfinite(self.coverage_coarse_cell_size)
            or self.coverage_coarse_cell_size <= 0.0
        ):
            raise ValueError("coverage_coarse_cell_size must be positive")
        self.lazy_owner_transforms = bool(lazy_owner_transforms)
        self.map.configure_lazy_owner_transforms(self.lazy_owner_transforms)
        self.last_pre_cap_count = 0
        self.last_saturated = False
        # Set permanently once a VoxelAnchorRefiner packet is fused.  The
        # legacy path remains scale-selected when the feature is disabled.
        self._depth_selected_mode = bool(
            getattr(self.map, "_anchor_depth_selected_levels", False)
        )

    def _cap_to_budget(
        self,
        batch: GlobalExplicitGaussianBatch,
    ) -> GlobalExplicitGaussianBatch:
        if self.max_total_gaussians <= 0 or len(batch) <= self.max_total_gaussians:
            return batch
        if not self.coverage_aware_budget:
            selected = torch.topk(
                batch.quality,
                k=self.max_total_gaussians,
                largest=True,
            ).indices
            return batch.index(selected)

        coarse = torch.floor(
            batch.xyz / float(self.coverage_coarse_cell_size)
        ).to(torch.int64)
        key = torch.cat([batch.level.long()[:, None], coarse], dim=-1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        max_quality = batch.quality.new_full((int(unique.shape[0]),), -torch.inf)
        max_quality.scatter_reduce_(
            0, inverse, batch.quality, reduce="amax", include_self=True
        )
        rows = torch.arange(len(batch), device=batch.xyz.device, dtype=torch.long)
        candidates = torch.where(
            batch.quality >= max_quality[inverse] - 1.0e-12,
            rows,
            torch.full_like(rows, len(batch)),
        )
        coverage = torch.full(
            (int(unique.shape[0]),),
            len(batch),
            device=batch.xyz.device,
            dtype=torch.long,
        )
        coverage.scatter_reduce_(
            0, inverse, candidates, reduce="amin", include_self=True
        )
        coverage = coverage[coverage < len(batch)]
        if int(coverage.numel()) >= self.max_total_gaussians:
            chosen = coverage.index_select(
                0,
                torch.topk(
                    batch.quality.index_select(0, coverage),
                    k=self.max_total_gaussians,
                    largest=True,
                ).indices,
            )
            return batch.index(chosen)
        selected_mask = torch.zeros(
            len(batch), device=batch.xyz.device, dtype=torch.bool
        )
        selected_mask[coverage] = True
        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        fill_count = self.max_total_gaussians - int(coverage.numel())
        fill = remaining.index_select(
            0,
            torch.topk(
                batch.quality.index_select(0, remaining),
                k=fill_count,
                largest=True,
            ).indices,
        )
        return batch.index(torch.cat([coverage, fill], dim=0))

    @staticmethod
    def _coverage_first_rows(
        batch: GlobalExplicitGaussianBatch,
        *,
        limit: int,
        coarse_cell_size: float,
    ) -> torch.Tensor:
        """Select incoming rows without merging anchors across depth levels."""

        budget = max(0, int(limit))
        if budget <= 0 or len(batch) <= budget:
            return torch.arange(
                len(batch), device=batch.xyz.device, dtype=torch.long
            )
        if not math.isfinite(float(coarse_cell_size)) or coarse_cell_size <= 0.0:
            raise ValueError("Incoming coverage cell size must be positive")
        coarse = torch.floor(batch.xyz / float(coarse_cell_size)).to(torch.int64)
        # Level is part of the key: coarse coverage never collapses anchors
        # from different voxel/depth levels into one representative.
        key = torch.cat([batch.level.long()[:, None], coarse], dim=-1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        max_quality = batch.quality.new_full((int(unique.shape[0]),), -torch.inf)
        max_quality.scatter_reduce_(
            0, inverse, batch.quality, reduce="amax", include_self=True
        )
        rows = torch.arange(len(batch), device=batch.xyz.device, dtype=torch.long)
        candidates = torch.where(
            batch.quality >= max_quality[inverse] - 1.0e-12,
            rows,
            torch.full_like(rows, len(batch)),
        )
        coverage = torch.full(
            (int(unique.shape[0]),),
            len(batch),
            device=batch.xyz.device,
            dtype=torch.long,
        )
        coverage.scatter_reduce_(
            0, inverse, candidates, reduce="amin", include_self=True
        )
        coverage = coverage[coverage < len(batch)]
        if int(coverage.numel()) >= budget:
            order = torch.argsort(
                batch.quality.index_select(0, coverage),
                descending=True,
                stable=True,
            )[:budget]
            return coverage.index_select(0, order)
        selected_mask = torch.zeros(
            len(batch), device=batch.xyz.device, dtype=torch.bool
        )
        selected_mask[coverage] = True
        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        fill_count = budget - int(coverage.numel())
        fill_order = torch.argsort(
            batch.quality.index_select(0, remaining),
            descending=True,
            stable=True,
        )[:fill_count]
        return torch.cat(
            [coverage, remaining.index_select(0, fill_order)], dim=0
        )

    def limit_prepared_incoming_by_coverage(
        self,
        prepared: PreparedPacketFusion,
        *,
        max_new_gaussians: int,
        coarse_cell_size: float,
    ) -> tuple[PreparedPacketFusion, dict[str, int | float]]:
        """Apply an optional post-Hash per-chunk incoming safety budget."""

        before = len(prepared.batch)
        limit = max(0, int(max_new_gaussians))
        selected = self._coverage_first_rows(
            prepared.batch,
            limit=limit,
            coarse_cell_size=coarse_cell_size,
        )
        limited = prepared.index(selected)
        stats: dict[str, int | float] = {
            "incoming_budget_enabled": int(limit > 0),
            "incoming_budget_limit": limit,
            "incoming_budget_before": before,
            "incoming_budget_after": len(limited.batch),
            "incoming_budget_dropped": before - len(limited.batch),
            "incoming_budget_coarse_cell_size": float(coarse_cell_size),
            "incoming_budget_same_level_only": 1,
        }
        for level in range(len(self.voxel_sizes)):
            stats[f"incoming_budget_level_{level}_before"] = int(
                (prepared.batch.level == level).sum().detach().cpu()
            )
            stats[f"incoming_budget_level_{level}_after"] = int(
                (limited.batch.level == level).sum().detach().cpu()
            )
        return limited, stats

    def _levels_and_grid(self, xyz: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = xyz.new_tensor(self.voxel_sizes)
        characteristic = scale.clamp_min(1.0e-8).prod(dim=-1).pow(1.0 / 3.0)
        distance = (characteristic[:, None].log() - sizes[None].log()).abs()
        level = distance.argmin(dim=-1)
        selected_size = sizes[level]
        grid = torch.floor(xyz / selected_size[:, None]).to(torch.int64)
        return level.to(torch.int64), grid

    @staticmethod
    def _empty(device: torch.device, dtype: torch.dtype, sh_count: int) -> GlobalExplicitGaussianBatch:
        return GlobalExplicitGaussianBatch(
            xyz=torch.zeros(0, 3, device=device, dtype=dtype),
            scale=torch.zeros(0, 3, device=device, dtype=dtype),
            rotation=torch.zeros(0, 4, device=device, dtype=dtype),
            opacity=torch.zeros(0, 1, device=device, dtype=dtype),
            opacity_parameter=torch.zeros(0, 1, device=device, dtype=dtype),
            sh_coefficients=torch.zeros(0, sh_count, 3, device=device, dtype=dtype),
            quality=torch.zeros(0, device=device, dtype=dtype),
            owner_window_id=torch.zeros(0, device=device, dtype=torch.long),
            level=torch.zeros(0, device=device, dtype=torch.long),
            voxel_size=torch.zeros(0, 1, device=device, dtype=dtype),
            grid_coord=torch.zeros(0, 3, device=device, dtype=torch.long),
            observation_count=torch.zeros(0, device=device, dtype=torch.long),
            confidence_accum=torch.zeros(0, device=device, dtype=dtype),
            birth_frame=torch.zeros(0, device=device, dtype=torch.long),
            last_seen_frame=torch.zeros(0, device=device, dtype=torch.long),
            visibility_count=torch.zeros(0, device=device, dtype=torch.long),
            render_error_ema=torch.zeros(0, device=device, dtype=dtype),
            replacement_hits=torch.zeros(0, device=device, dtype=torch.long),
            inconsistency_hits=torch.zeros(0, device=device, dtype=torch.long),
        )

    def packet_to_global_batch(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
    ) -> GlobalExplicitGaussianBatch:
        observation = packet.observation
        device = self.map.xyz.device
        dtype = self.map.xyz.dtype
        transform = anchor_to_global.to(device=device, dtype=dtype)
        global_scale, global_rotation, _ = sim3_components(transform)
        target_sh_count = int(self.map.sh_rest.shape[1]) + 1
        parts: list[GlobalExplicitGaussianBatch] = []
        height, width = observation.image_size

        centers_camera = observation.centers_camera()[0].to(device=device, dtype=dtype)
        scale_camera = observation.scales()[0].permute(0, 2, 3, 1).to(device=device, dtype=dtype)
        local_quaternion = observation.local_quaternion[0].permute(0, 2, 3, 1).to(device=device, dtype=dtype)
        rgb_sh = observation.rgb_sh[0].permute(0, 3, 4, 1, 2).to(device=device, dtype=dtype)
        # Density SH is view-dependent in Stage 2.  The first global backend
        # materializes it once on the immutable source ray, matching the
        # Stage-2 observation contract while keeping the global renderer
        # compatible with scalar opacity.
        opacity = observation.source_view_confidence()[0].permute(0, 2, 3, 1).to(
            device=device, dtype=dtype
        )
        geometry_confidence = observation.confidence[0, :, 0].to(device=device, dtype=dtype)
        valid = packet.finite_gaussian_mask[0, :, 0].to(device=device)
        non_sky_probability = (1.0 - packet.sky_prob[0, :, 0].to(device=device, dtype=dtype)).clamp(0.0, 1.0)
        consistency = packet.geometry_consistency[0, :, 0].to(device=device, dtype=dtype)

        for view in range(observation.num_source_views):
            pose = packet.local_poses_c2w[view].to(device=device, dtype=dtype)
            center_anchor = torch.einsum("ij,hwj->hwi", pose[:3, :3], centers_camera[view]) + pose[:3, 3]
            center_global = apply_sim3(transform, center_anchor)
            pose_quaternion = matrix_to_quaternion(pose[:3, :3]).view(1, 1, 4)
            anchor_quaternion = normalize_quaternion(quaternion_multiply(pose_quaternion, local_quaternion[view]))
            global_quaternion = normalize_quaternion(
                quaternion_multiply(matrix_to_quaternion(global_rotation).view(1, 1, 4), anchor_quaternion)
            )
            scale_global = global_scale * scale_camera[view]
            combined_rotation = global_rotation @ pose[:3, :3]
            coefficients = rotate_sh_coefficients(
                rgb_sh[view], combined_rotation, observation.rgb_sh_degree
            )
            if int(coefficients.shape[-2]) != target_sh_count:
                resized = coefficients.new_zeros(height, width, target_sh_count, 3)
                copy_count = min(target_sh_count, int(coefficients.shape[-2]))
                resized[..., :copy_count, :] = coefficients[..., :copy_count, :]
                coefficients = resized
            confidence = geometry_confidence[view]
            # Stage-2 Gaussians are dense ERP predictions, but voxel quality is
            # an observation-quality score rather than an integration measure.
            # Fibonacci handles source-sphere area for graph sampling, so a
            # second cos(latitude) term here would suppress polar geometry twice.
            quality = (
                confidence
                * opacity[view, ..., 0]
                * non_sky_probability[view]
                * consistency[view]
            )
            keep = (
                valid[view]
                & torch.isfinite(center_global).all(dim=-1)
                & torch.isfinite(scale_global).all(dim=-1)
                & torch.isfinite(global_quaternion).all(dim=-1)
                & torch.isfinite(coefficients).all(dim=(-1, -2))
                & (confidence >= self.min_confidence)
                & (opacity[view, ..., 0] >= self.min_opacity)
            )
            selected = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            xyz = center_global.reshape(-1, 3)[selected]
            selected_scale = scale_global.reshape(-1, 3)[selected]
            level, grid = self._levels_and_grid(xyz, selected_scale)
            count = int(selected.numel())
            parts.append(
                GlobalExplicitGaussianBatch(
                    xyz=xyz,
                    scale=selected_scale,
                    rotation=global_quaternion.reshape(-1, 4)[selected],
                    opacity=opacity[view].reshape(-1, 1)[selected],
                    opacity_parameter=self.map._inv_sigmoid(
                        opacity[view].reshape(-1, 1)[selected]
                    ),
                    sh_coefficients=coefficients.reshape(-1, target_sh_count, 3)[selected],
                    quality=quality.reshape(-1)[selected],
                    owner_window_id=torch.full((count,), int(packet.window_id), device=device, dtype=torch.long),
                    level=level,
                    voxel_size=xyz.new_tensor(self.voxel_sizes)[level, None],
                    grid_coord=grid,
                    observation_count=torch.ones(count, device=device, dtype=torch.long),
                    confidence_accum=confidence.reshape(-1)[selected],
                    birth_frame=torch.full((count,), int(packet.frame_ids[view]), device=device, dtype=torch.long),
                    last_seen_frame=torch.full((count,), int(packet.frame_ids[-1]), device=device, dtype=torch.long),
                    visibility_count=torch.ones(count, device=device, dtype=torch.long),
                    render_error_ema=torch.zeros(count, device=device, dtype=dtype),
                    replacement_hits=torch.zeros(count, device=device, dtype=torch.long),
                    inconsistency_hits=torch.zeros(count, device=device, dtype=torch.long),
                )
            )
        if not parts:
            return self._empty(device, dtype, target_sh_count)
        return self._concatenate(parts)

    @staticmethod
    def _concatenate(parts: list[GlobalExplicitGaussianBatch]) -> GlobalExplicitGaussianBatch:
        return GlobalExplicitGaussianBatch(
            **{
                name: torch.cat([getattr(part, name) for part in parts], dim=0)
                for name in GlobalExplicitGaussianBatch.__dataclass_fields__
            }
        )

    @staticmethod
    def _segment_sum(value: torch.Tensor, inverse: torch.Tensor, count: int) -> torch.Tensor:
        output = value.new_zeros((count,) + tuple(value.shape[1:]))
        output.index_add_(0, inverse, value)
        return output

    def compact_within_window(self, batch: GlobalExplicitGaussianBatch) -> GlobalExplicitGaussianBatch:
        if len(batch) == 0:
            return batch
        key = torch.cat(
            [batch.owner_window_id[:, None], batch.level[:, None], batch.grid_coord], dim=-1
        )
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        count = int(unique.shape[0])
        weight = batch.quality.clamp_min(1.0e-8)
        weight_sum = self._segment_sum(weight[:, None], inverse, count).clamp_min(1.0e-8)

        def average(value: torch.Tensor) -> torch.Tensor:
            shaped_weight = weight.view(-1, *([1] * (value.ndim - 1)))
            return self._segment_sum(shaped_weight * value, inverse, count) / weight_sum.view(
                count, *([1] * (value.ndim - 1))
            )

        quaternion = batch.rotation.clone()
        quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
        observation_count = self._segment_sum(
            batch.observation_count[:, None], inverse, count
        )[:, 0]
        confidence_accum = self._segment_sum(
            batch.confidence_accum[:, None], inverse, count
        )[:, 0]
        birth = batch.birth_frame.new_full((count,), torch.iinfo(batch.birth_frame.dtype).max)
        birth.scatter_reduce_(0, inverse, batch.birth_frame, reduce="amin", include_self=True)
        last_seen = batch.last_seen_frame.new_zeros(count)
        last_seen.scatter_reduce_(0, inverse, batch.last_seen_frame, reduce="amax", include_self=True)
        visibility = self._segment_sum(batch.visibility_count[:, None], inverse, count)[:, 0]
        replacement_hits = self._segment_sum(
            batch.replacement_hits[:, None], inverse, count
        )[:, 0]
        inconsistency_hits = self._segment_sum(
            batch.inconsistency_hits[:, None], inverse, count
        )[:, 0]
        return GlobalExplicitGaussianBatch(
            xyz=average(batch.xyz),
            scale=torch.exp(average(batch.scale.clamp_min(1.0e-8).log())),
            rotation=normalize_quaternion(average(quaternion)),
            opacity=average(batch.opacity).clamp(1.0e-5, 1.0 - 1.0e-5),
            opacity_parameter=average(batch.opacity_parameter),
            sh_coefficients=average(batch.sh_coefficients),
            quality=(weight_sum[:, 0] / observation_count.clamp_min(1).to(weight_sum)).clamp_min(1.0e-8),
            owner_window_id=unique[:, 0].long(),
            level=unique[:, 1].long(),
            voxel_size=average(batch.voxel_size),
            grid_coord=unique[:, 2:].long(),
            observation_count=observation_count,
            confidence_accum=confidence_accum,
            birth_frame=birth,
            last_seen_frame=last_seen,
            visibility_count=visibility,
            render_error_ema=average(batch.render_error_ema[:, None])[:, 0],
            replacement_hits=replacement_hits,
            inconsistency_hits=inconsistency_hits,
        )

    def _winner_take_global_voxel(
        self,
        batch: GlobalExplicitGaussianBatch,
        *,
        preserve_levels: bool | None = None,
    ) -> GlobalExplicitGaussianBatch:
        if len(batch) == 0:
            self.last_pre_cap_count = 0
            self.last_saturated = False
            return batch
        preserve = self._depth_selected_mode if preserve_levels is None else bool(preserve_levels)
        if preserve:
            level = batch.level.long()
            selected_size = batch.voxel_size[:, 0].to(batch.xyz).clamp_min(1.0e-8)
            grid = torch.floor(batch.xyz / selected_size[:, None]).to(torch.int64)
        else:
            level, grid = self._levels_and_grid(batch.xyz, batch.scale)
            batch.voxel_size = batch.xyz.new_tensor(self.voxel_sizes)[level, None]
        batch.level, batch.grid_coord = level, grid
        key = torch.cat([level[:, None], grid], dim=-1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        max_quality = batch.quality.new_full((int(unique.shape[0]),), -torch.inf)
        max_quality.scatter_reduce_(0, inverse, batch.quality, reduce="amax", include_self=True)
        indices = torch.arange(len(batch), device=batch.xyz.device, dtype=torch.long)
        candidate = torch.where(
            batch.quality >= max_quality[inverse] - 1.0e-12,
            indices,
            torch.full_like(indices, len(batch)),
        )
        winner = torch.full(
            (int(unique.shape[0]),), len(batch), device=batch.xyz.device, dtype=torch.long
        )
        winner.scatter_reduce_(0, inverse, candidate, reduce="amin", include_self=True)
        winner = winner[winner < len(batch)]
        result = batch.index(winner)
        result.replacement_hits = self._segment_sum(
            batch.replacement_hits[:, None], inverse, int(unique.shape[0])
        )[:, 0]
        result.inconsistency_hits = self._segment_sum(
            batch.inconsistency_hits[:, None], inverse, int(unique.shape[0])
        )[:, 0]
        self.last_pre_cap_count = len(result)
        self.last_saturated = self.max_total_gaussians > 0 and len(result) > self.max_total_gaussians
        return self._cap_to_budget(result)

    def _winner_take_owner_voxel(
        self,
        batch: GlobalExplicitGaussianBatch,
        *,
        preserve_levels: bool | None = None,
    ) -> GlobalExplicitGaussianBatch:
        """Deduplicate only within an owner's immutable reference frame."""

        if len(batch) == 0:
            self.last_pre_cap_count = 0
            self.last_saturated = False
            return batch
        preserve = self._depth_selected_mode if preserve_levels is None else bool(preserve_levels)
        if preserve:
            level = batch.level.long()
            selected_size = batch.voxel_size[:, 0].to(batch.xyz).clamp_min(1.0e-8)
            grid = torch.floor(batch.xyz / selected_size[:, None]).to(torch.int64)
        else:
            level, grid = self._levels_and_grid(batch.xyz, batch.scale)
            batch.voxel_size = batch.xyz.new_tensor(self.voxel_sizes)[level, None]
        batch.level, batch.grid_coord = level, grid
        key = torch.cat([batch.owner_window_id[:, None], level[:, None], grid], dim=-1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        max_quality = batch.quality.new_full((int(unique.shape[0]),), -torch.inf)
        max_quality.scatter_reduce_(0, inverse, batch.quality, reduce="amax", include_self=True)
        indices = torch.arange(len(batch), device=batch.xyz.device, dtype=torch.long)
        candidate = torch.where(
            batch.quality >= max_quality[inverse] - 1.0e-12,
            indices,
            torch.full_like(indices, len(batch)),
        )
        winner = torch.full(
            (int(unique.shape[0]),), len(batch), device=batch.xyz.device, dtype=torch.long
        )
        winner.scatter_reduce_(0, inverse, candidate, reduce="amin", include_self=True)
        winner = winner[winner < len(batch)]
        result = batch.index(winner)
        result.replacement_hits = self._segment_sum(
            batch.replacement_hits[:, None], inverse, int(unique.shape[0])
        )[:, 0]
        result.inconsistency_hits = self._segment_sum(
            batch.inconsistency_hits[:, None], inverse, int(unique.shape[0])
        )[:, 0]
        self.last_pre_cap_count = len(result)
        self.last_saturated = self.max_total_gaussians > 0 and len(result) > self.max_total_gaussians
        return self._cap_to_budget(result)

    @staticmethod
    def _distribution(prefix: str, value: torch.Tensor) -> dict[str, float]:
        finite = value.detach().float().reshape(-1)
        if finite.numel() > 65536:
            stride = max(1, int(math.ceil(finite.numel() / 65536)))
            finite = finite[::stride][:65536]
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return {f"{prefix}_{name}": 0.0 for name in ("min", "mean", "median", "max")}
        return {
            f"{prefix}_min": float(finite.min().cpu()),
            f"{prefix}_mean": float(finite.mean().cpu()),
            f"{prefix}_median": float(finite.median().cpu()),
            f"{prefix}_max": float(finite.max().cpu()),
        }

    def _quality_diagnostics(
        self, packet: LocalGaussianWindowPacket, incoming: GlobalExplicitGaussianBatch
    ) -> dict[str, float]:
        observation = packet.observation
        mask = packet.finite_gaussian_mask[0, :, 0].bool()
        opacity = observation.source_view_confidence()[0, :, 0]
        components = {
            "quality_geom_conf": observation.confidence[0, :, 0][mask],
            "quality_opacity": opacity[mask],
            "quality_non_sky": (1.0 - packet.sky_prob[0, :, 0]).clamp(0.0, 1.0)[mask],
            "quality_consistency": packet.geometry_consistency[0, :, 0].float()[mask],
            "quality_product": incoming.quality,
        }
        diagnostics: dict[str, float] = {}
        for prefix, value in components.items():
            diagnostics.update(self._distribution(prefix, value))
        return diagnostics

    def _batch_from_map(
        self,
        *,
        preserve_levels: bool | None = None,
    ) -> GlobalExplicitGaussianBatch:
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        count = self.map.anchor_count()
        sh_count = int(self.map.sh_rest.shape[1]) + 1
        if count == 0:
            return self._empty(device, dtype, sh_count)

        def metadata(name: str, default: torch.Tensor) -> torch.Tensor:
            value = getattr(self.map, name, None)
            if torch.is_tensor(value) and int(value.numel()) == count:
                return value.to(device=device)
            return default

        preserve = self._depth_selected_mode if preserve_levels is None else bool(preserve_levels)
        map_xyz = self.map.xyz.detach() if self.lazy_owner_transforms else self.map.get_xyz.detach()
        map_scale = (
            self.map._base_scaling().detach()
            if self.lazy_owner_transforms
            else self.map.get_scaling.detach()
        )
        map_rotation = (
            self.map.rotation.detach()
            if self.map.gaussian_parameterization == "traditional_3dgs"
            else (
                self.map._base_rotation().detach()
                if self.lazy_owner_transforms
                else self.map.get_rotation.detach()
            )
        )
        map_sh = (
            self.map._base_sh_coefficients().detach()
            if self.lazy_owner_transforms
            else self.map.get_sh_coefficients.detach()
        )
        computed_level, computed_grid = self._levels_and_grid(map_xyz, map_scale)
        if preserve:
            level = metadata("_anchor_level", computed_level.cpu()).to(device=device).long()
            voxel_size = metadata(
                "_anchor_voxel_size",
                self.map.get_xyz.detach().new_tensor(self.voxel_sizes)[level.to(device=device)],
            ).to(device=device, dtype=dtype).reshape(-1, 1)
            grid = torch.floor(
                map_xyz / voxel_size.clamp_min(1.0e-8)
            ).long()
        else:
            level, grid = computed_level, computed_grid
            voxel_size = self.map.get_xyz.detach().new_tensor(self.voxel_sizes)[level, None]
        return GlobalExplicitGaussianBatch(
            xyz=map_xyz.clone(),
            scale=map_scale.clone(),
            rotation=map_rotation.clone(),
            opacity=self.map.get_opacity.detach().clone(),
            opacity_parameter=self.map.opacity_logit.detach().clone(),
            sh_coefficients=map_sh.clone(),
            quality=metadata("_anchor_quality", torch.ones(count)).to(dtype=dtype),
            owner_window_id=metadata("_anchor_owner_window_id", torch.full((count,), -1)).long(),
            level=level,
            voxel_size=voxel_size,
            grid_coord=grid,
            observation_count=metadata("_anchor_obs_count", torch.ones(count)).long(),
            confidence_accum=metadata("_anchor_conf_accum", torch.ones(count)).to(dtype=dtype),
            birth_frame=metadata("_anchor_birth_frame", torch.zeros(count)).long(),
            last_seen_frame=metadata("_anchor_last_seen_kf", torch.zeros(count)).long(),
            visibility_count=metadata("_anchor_visibility_count", torch.zeros(count)).long(),
            render_error_ema=metadata("_anchor_render_error_ema", torch.zeros(count)).to(dtype=dtype),
            replacement_hits=metadata("_anchor_inlier_obs", torch.zeros(count)).long(),
            inconsistency_hits=metadata("_anchor_outlier_obs", torch.zeros(count)).long(),
        )

    def _write_map(self, batch: GlobalExplicitGaussianBatch) -> None:
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        xyz = batch.xyz.to(device=device, dtype=dtype)
        scale = batch.scale.to(device=device, dtype=dtype)
        opacity = batch.opacity.to(device=device, dtype=dtype)
        sh = batch.sh_coefficients.to(device=device, dtype=dtype)
        self.map.xyz = nn.Parameter(xyz)
        self.map.rotation = nn.Parameter(
            batch.rotation.to(device=device, dtype=dtype)
            if self.map.gaussian_parameterization == "traditional_3dgs"
            else normalize_quaternion(batch.rotation.to(device=device, dtype=dtype))
        )
        self.map.scaling = nn.Parameter(self.map._scale_parameter_from_actual(scale))
        self.map.opacity_logit = nn.Parameter(
            batch.opacity_parameter.to(device=device, dtype=dtype)
            if self.map.gaussian_parameterization == "traditional_3dgs"
            else self.map._inv_sigmoid(opacity)
        )
        self.map.features = nn.Parameter(
            self.map._feature_parameter_from_sh_dc(sh[:, 0])
        )
        rest_count = int(self.map.sh_rest.shape[1])
        self.map.sh_rest = nn.Parameter(
            sh[:, 1 : 1 + rest_count].contiguous()
            if rest_count > 0
            else torch.zeros(len(batch), 0, 3, device=device, dtype=dtype)
        )
        self.map._anchor_level = batch.level.detach().cpu().to(torch.int8)
        self.map._anchor_voxel_size = batch.voxel_size[:, 0].detach().cpu().float()
        self.map._anchor_grid_coord = batch.grid_coord.detach().cpu().to(torch.int32)
        self.map._anchor_obs_count = batch.observation_count.detach().cpu().to(torch.int32)
        self.map._anchor_conf_accum = batch.confidence_accum.detach().cpu().float()
        self.map._anchor_birth_frame = batch.birth_frame.detach().cpu().to(torch.int32)
        self.map._anchor_last_seen_kf = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_last_update_kf_ord = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_source_window_id = batch.owner_window_id.detach().cpu().to(torch.int32)
        self.map._anchor_source_frame_start = batch.birth_frame.detach().cpu().to(torch.int32)
        self.map._anchor_source_frame_end = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_inlier_obs = batch.replacement_hits.detach().cpu().to(torch.int32)
        self.map._anchor_outlier_obs = batch.inconsistency_hits.detach().cpu().to(torch.int32)
        self.map._anchor_owner_window_id = batch.owner_window_id.detach().cpu().to(torch.int32)
        self.map._anchor_quality = batch.quality.detach().cpu().float()
        self.map._anchor_visibility_count = batch.visibility_count.detach().cpu().to(torch.int32)
        self.map._anchor_render_error_ema = batch.render_error_ema.detach().cpu().float()
        self.map._anchor_depth_selected_levels = bool(self._depth_selected_mode)

    def _anchor_packet_to_global_batch_with_indices(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
        *,
        apply_semantic_gates: bool = True,
    ) -> tuple[GlobalExplicitGaussianBatch, torch.Tensor]:
        """Transform refined voxel anchors without scale-based re-leveling."""

        anchor = packet.anchor_observation
        if anchor is None:
            raise ValueError("anchor_packet_to_global_batch requires anchor_observation")
        if anchor.batch_size != 1:
            raise ValueError("Global voxel-anchor fusion currently requires B=1 packets")
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        transform = anchor_to_global.to(device=device, dtype=dtype)
        global_scale, global_rotation, _ = sim3_components(transform)
        xyz = apply_sim3(transform, anchor.xyz.to(device=device, dtype=dtype))
        scale = global_scale * anchor.scaling.to(device=device, dtype=dtype)
        rotation = normalize_quaternion(
            quaternion_multiply(
                matrix_to_quaternion(global_rotation).view(1, 4),
                anchor.rotation.to(device=device, dtype=dtype),
            )
        )
        coefficients = rotate_sh_coefficients(
            anchor.sh_coefficients.to(device=device, dtype=dtype),
            global_rotation,
            degree=2,
        )
        target_sh_count = int(self.map.sh_rest.shape[1]) + 1
        if int(coefficients.shape[1]) != target_sh_count:
            resized = coefficients.new_zeros(int(coefficients.shape[0]), target_sh_count, 3)
            copied = min(target_sh_count, int(coefficients.shape[1]))
            resized[:, :copied] = coefficients[:, :copied]
            coefficients = resized
        opacity = anchor.opacity.to(device=device, dtype=dtype)
        quality = anchor.quality[:, 0].to(device=device, dtype=dtype)
        level = anchor.level.to(device=device, dtype=torch.long)
        voxel_size = global_scale * anchor.voxel_size.to(device=device, dtype=dtype)
        count = anchor.member_count[:, 0].round().clamp_min(1).to(device=device, dtype=torch.long)
        keep = (
            torch.isfinite(xyz).all(dim=-1)
            & torch.isfinite(scale).all(dim=-1)
            & (scale > 0.0).all(dim=-1)
            & torch.isfinite(rotation).all(dim=-1)
            & torch.isfinite(coefficients).all(dim=(-1, -2))
            & torch.isfinite(opacity[:, 0])
            & torch.isfinite(quality)
            & (voxel_size[:, 0] > 0.0)
            & (level >= 0)
            & (level < len(self.voxel_sizes))
        )
        if apply_semantic_gates:
            keep &= (quality >= self.min_confidence) & (
                opacity[:, 0] >= self.min_opacity
            )
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        if int(selected.numel()) == 0:
            return self._empty(device, dtype, target_sh_count), selected
        xyz = xyz.index_select(0, selected)
        voxel_size = voxel_size.index_select(0, selected)
        level = level.index_select(0, selected)
        quality = quality.index_select(0, selected)
        count = count.index_select(0, selected)
        grid = torch.floor(xyz / voxel_size.clamp_min(1.0e-8)).long()
        size = int(selected.numel())
        return (
            GlobalExplicitGaussianBatch(
                xyz=xyz,
                scale=scale.index_select(0, selected),
                rotation=rotation.index_select(0, selected),
                opacity=opacity.index_select(0, selected),
                opacity_parameter=self.map._inv_sigmoid(
                    opacity.index_select(0, selected)
                ),
                sh_coefficients=coefficients.index_select(0, selected),
                quality=quality,
                owner_window_id=torch.full((size,), int(packet.window_id), device=device, dtype=torch.long),
                level=level,
                voxel_size=voxel_size,
                grid_coord=grid,
                observation_count=count,
                confidence_accum=quality * count.to(dtype),
                birth_frame=torch.full((size,), int(packet.frame_ids[0]), device=device, dtype=torch.long),
                last_seen_frame=torch.full((size,), int(packet.frame_ids[-1]), device=device, dtype=torch.long),
                visibility_count=torch.ones(size, device=device, dtype=torch.long),
                render_error_ema=torch.zeros(size, device=device, dtype=dtype),
                replacement_hits=torch.zeros(size, device=device, dtype=torch.long),
                inconsistency_hits=torch.zeros(size, device=device, dtype=torch.long),
            ),
            selected,
        )

    def anchor_packet_to_global_batch(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
    ) -> GlobalExplicitGaussianBatch:
        batch, _ = self._anchor_packet_to_global_batch_with_indices(
            packet,
            anchor_to_global,
        )
        return batch

    def prepare_packet_batch(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
        *,
        apply_semantic_gates: bool = True,
    ) -> PreparedPacketFusion:
        depth_selected = packet.anchor_observation is not None
        if self._depth_selected_mode and not depth_selected and self.map.anchor_count() > 0:
            raise ValueError(
                "Cannot mix legacy scale-selected packets into a depth-selected voxel-anchor map"
            )
        if depth_selected:
            batch, source_indices = self._anchor_packet_to_global_batch_with_indices(
                packet,
                anchor_to_global,
                apply_semantic_gates=apply_semantic_gates,
            )
        else:
            batch = self.packet_to_global_batch(packet, anchor_to_global)
            source_indices = None
        return PreparedPacketFusion(
            batch=batch,
            source_anchor_indices=source_indices,
            requested=len(batch),
            depth_selected=depth_selected,
        )

    @staticmethod
    def _spatial_hash_keys(grid: torch.Tensor) -> torch.Tensor:
        """Hash integer xyz cells; exact cell equality resolves rare collisions."""

        value = grid.to(dtype=torch.int64)
        return (
            (value[..., 0] * 73_856_093)
            ^ (value[..., 1] * 19_349_663)
            ^ (value[..., 2] * 83_492_791)
        )

    @classmethod
    def _match_visible_level_vectorized(
        cls,
        *,
        incoming_xyz: torch.Tensor,
        incoming_voxel: torch.Tensor,
        incoming_rows: torch.Tensor,
        existing_xyz: torch.Tensor,
        existing_voxel: torch.Tensor,
        existing_quality: torch.Tensor,
        existing_rows: torch.Tensor,
        radius_scale: float,
        radius_cells: int,
        query_chunk_size: int = 16_384,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact incoming-to-existing matches without per-anchor Python work."""

        if int(incoming_rows.numel()) == 0 or int(existing_rows.numel()) == 0:
            return incoming_rows.new_empty(0), existing_rows.new_empty(0)
        device = incoming_xyz.device
        cell_size = torch.maximum(
            incoming_voxel.max(), existing_voxel.max()
        ).clamp_min(1.0e-8)
        old_grid = torch.floor(existing_xyz / cell_size).to(torch.int64)
        old_keys, old_order = torch.sort(cls._spatial_hash_keys(old_grid))
        axis = torch.arange(
            -int(radius_cells),
            int(radius_cells) + 1,
            device=device,
            dtype=torch.int64,
        )
        offsets = torch.cartesian_prod(axis, axis, axis).reshape(-1, 3)
        offset_count = int(offsets.shape[0])
        matched_new_parts: list[torch.Tensor] = []
        matched_old_parts: list[torch.Tensor] = []

        for start in range(0, int(incoming_rows.numel()), int(query_chunk_size)):
            end = min(int(incoming_rows.numel()), start + int(query_chunk_size))
            chunk_xyz = incoming_xyz[start:end]
            chunk_voxel = incoming_voxel[start:end]
            new_grid = torch.floor(chunk_xyz / cell_size).to(torch.int64)
            query_grid = new_grid[:, None, :] + offsets[None, :, :]
            flat_query_grid = query_grid.reshape(-1, 3)
            query_keys = cls._spatial_hash_keys(flat_query_grid).contiguous()
            left = torch.searchsorted(old_keys, query_keys, right=False)
            right = torch.searchsorted(old_keys, query_keys, right=True)
            counts = right - left
            active_queries = torch.nonzero(counts > 0, as_tuple=False).flatten()
            if int(active_queries.numel()) == 0:
                continue
            active_counts = counts.index_select(0, active_queries)
            total_candidates = int(active_counts.sum().item())
            if total_candidates == 0:
                continue
            query_ids = torch.repeat_interleave(active_queries, active_counts)
            active_left = left.index_select(0, active_queries)
            output_starts = torch.cumsum(active_counts, dim=0) - active_counts
            candidate_offsets = torch.arange(
                total_candidates, device=device, dtype=torch.long
            ) - torch.repeat_interleave(output_starts, active_counts)
            sorted_positions = (
                torch.repeat_interleave(active_left, active_counts)
                + candidate_offsets
            )
            old_local = old_order.index_select(0, sorted_positions)
            new_local = torch.div(
                query_ids, offset_count, rounding_mode="floor"
            )

            same_cell = (
                old_grid.index_select(0, old_local)
                == flat_query_grid.index_select(0, query_ids)
            ).all(dim=-1)
            candidate_xyz = existing_xyz.index_select(0, old_local)
            candidate_new_xyz = chunk_xyz.index_select(0, new_local)
            distances = torch.linalg.norm(candidate_xyz - candidate_new_xyz, dim=-1)
            radii = float(radius_scale) * 0.5 * (
                chunk_voxel.index_select(0, new_local)
                + existing_voxel.index_select(0, old_local)
            )
            eligible = same_cell & (distances <= radii + 1.0e-8)
            if not bool(eligible.any()):
                continue

            chunk_count = end - start
            minimum_distance = distances.new_full((chunk_count,), torch.inf)
            minimum_distance.scatter_reduce_(
                0,
                new_local,
                torch.where(eligible, distances, torch.inf),
                reduce="amin",
                include_self=True,
            )
            nearest = eligible & (
                distances <= minimum_distance.index_select(0, new_local) + 1.0e-8
            )
            candidate_quality = existing_quality.index_select(0, old_local)
            best_quality = candidate_quality.new_full((chunk_count,), -torch.inf)
            best_quality.scatter_reduce_(
                0,
                new_local,
                torch.where(nearest, candidate_quality, -torch.inf),
                reduce="amax",
                include_self=True,
            )
            quality_winner = nearest & (
                candidate_quality
                >= best_quality.index_select(0, new_local) - 1.0e-12
            )
            sentinel = torch.iinfo(torch.long).max
            candidate_global_rows = existing_rows.index_select(0, old_local)
            best_existing = torch.full(
                (chunk_count,), sentinel, device=device, dtype=torch.long
            )
            best_existing.scatter_reduce_(
                0,
                new_local,
                torch.where(
                    quality_winner,
                    candidate_global_rows,
                    torch.full_like(candidate_global_rows, sentinel),
                ),
                reduce="amin",
                include_self=True,
            )
            matched = best_existing != sentinel
            if bool(matched.any()):
                matched_local = torch.nonzero(
                    matched, as_tuple=False
                ).flatten()
                matched_new_parts.append(
                    incoming_rows[start:end].index_select(0, matched_local)
                )
                matched_old_parts.append(
                    best_existing.index_select(0, matched_local)
                )

        if not matched_new_parts:
            return incoming_rows.new_empty(0), existing_rows.new_empty(0)
        return torch.cat(matched_new_parts), torch.cat(matched_old_parts)

    def _filter_against_visible_map_vectorized(
        self,
        prepared: PreparedPacketFusion,
        *,
        incoming_visible: torch.Tensor,
        existing_visibility: torch.Tensor,
        radius_scale: float,
        radius_cells: int,
        update_existing_statistics: bool,
        stats: dict[str, int | float],
    ) -> tuple[
        PreparedPacketFusion,
        dict[str, int | float],
        ExistingAnchorEvidenceUpdate | None,
    ]:
        """Visible-only exact spatial hash that stays on the map device."""

        device = prepared.batch.xyz.device
        incoming_rows = torch.nonzero(
            incoming_visible.to(device=device), as_tuple=False
        ).flatten()
        existing_rows = torch.nonzero(
            existing_visibility.to(device=self.map.xyz.device), as_tuple=False
        ).flatten()
        stats["hash_vectorized"] = 1
        stats["hash_materialized_existing"] = int(existing_rows.numel())
        if int(incoming_rows.numel()) == 0 or int(existing_rows.numel()) == 0:
            return prepared, stats, None

        existing_rows_cpu = existing_rows.detach().cpu().long()
        existing_xyz, existing_voxel = self.map.materialized_anchor_geometry_rows(
            existing_rows
        )
        existing_level = self.map._anchor_level.index_select(
            0, existing_rows_cpu
        ).to(device=device, dtype=torch.long)
        existing_quality = self.map._anchor_quality.index_select(
            0, existing_rows_cpu
        ).to(device=device, dtype=prepared.batch.quality.dtype)
        incoming_level = prepared.batch.level.index_select(0, incoming_rows)
        matched_new_parts: list[torch.Tensor] = []
        matched_old_parts: list[torch.Tensor] = []

        for level in range(len(self.voxel_sizes)):
            new_selection = torch.nonzero(
                incoming_level == level, as_tuple=False
            ).flatten()
            old_selection = torch.nonzero(
                existing_level == level, as_tuple=False
            ).flatten()
            stats[f"hash_level_{level}_visible"] = int(new_selection.numel())
            if (
                int(new_selection.numel()) == 0
                or int(old_selection.numel()) == 0
                or radius_scale <= 0.0
            ):
                continue
            level_new_rows = incoming_rows.index_select(0, new_selection)
            level_old_rows = existing_rows.index_select(0, old_selection)
            matched_new, matched_old = self._match_visible_level_vectorized(
                incoming_xyz=prepared.batch.xyz.detach().index_select(
                    0, level_new_rows
                ),
                incoming_voxel=prepared.batch.voxel_size[
                    level_new_rows, 0
                ].detach().to(device=device),
                incoming_rows=level_new_rows,
                existing_xyz=existing_xyz.index_select(0, old_selection),
                existing_voxel=existing_voxel.index_select(0, old_selection),
                existing_quality=existing_quality.index_select(0, old_selection),
                existing_rows=level_old_rows,
                radius_scale=radius_scale,
                radius_cells=radius_cells,
            )
            level_hits = int(matched_new.numel())
            stats[f"hash_level_{level}_hits"] = level_hits
            stats[f"hash_level_{level}_kept"] = (
                int(stats[f"hash_level_{level}_incoming"]) - level_hits
            )
            if level_hits > 0:
                matched_new_parts.append(matched_new)
                matched_old_parts.append(matched_old)

        keep = torch.ones(len(prepared.batch), device=device, dtype=torch.bool)
        if matched_new_parts:
            matched_new = torch.cat(matched_new_parts)
            matched_old = torch.cat(matched_old_parts)
            keep[matched_new] = False
        else:
            matched_new = incoming_rows.new_empty(0)
            matched_old = existing_rows.new_empty(0)
        kept = torch.nonzero(keep, as_tuple=False).flatten()
        filtered = prepared.index(kept)
        stats["hash_hits"] = int(matched_new.numel())
        stats["hash_kept"] = len(filtered.batch)
        if not update_existing_statistics or int(matched_new.numel()) == 0:
            return filtered, stats, None

        evidence_rows, inverse = torch.unique(
            matched_old, sorted=True, return_inverse=True
        )
        observation_delta = torch.zeros(
            int(evidence_rows.numel()), device=device, dtype=torch.long
        )
        observation_delta.index_add_(
            0,
            inverse,
            prepared.batch.observation_count.index_select(0, matched_new).long(),
        )
        confidence_delta = torch.zeros(
            int(evidence_rows.numel()),
            device=device,
            dtype=prepared.batch.confidence_accum.dtype,
        )
        confidence_delta.index_add_(
            0,
            inverse,
            prepared.batch.confidence_accum.index_select(0, matched_new),
        )
        last_seen = torch.zeros(
            int(evidence_rows.numel()), device=device, dtype=torch.long
        )
        last_seen.scatter_reduce_(
            0,
            inverse,
            prepared.batch.last_seen_frame.index_select(0, matched_new).long(),
            reduce="amax",
            include_self=True,
        )
        evidence_update = ExistingAnchorEvidenceUpdate(
            indices=evidence_rows.detach().cpu(),
            observation_count_delta=observation_delta.detach().cpu(),
            confidence_accum_delta=confidence_delta.detach().cpu(),
            last_seen_frame=last_seen.detach().cpu(),
        )
        return filtered, stats, evidence_update

    def filter_against_visible_map(
        self,
        prepared: PreparedPacketFusion,
        *,
        incoming_anchor_visibility: torch.Tensor,
        existing_anchor_visibility: torch.Tensor,
        radius_voxels: float = 1.0,
        update_existing_statistics: bool = True,
    ) -> tuple[
        PreparedPacketFusion,
        dict[str, int | float],
        ExistingAnchorEvidenceUpdate | None,
    ]:
        """Drop visible same-level incoming anchors near visible existing anchors."""

        if not prepared.depth_selected or prepared.source_anchor_indices is None:
            raise ValueError("Visible-map hash filtering requires a refined anchor packet")
        incoming_visibility = incoming_anchor_visibility.detach().bool().reshape(-1)
        existing_visibility = existing_anchor_visibility.detach().bool().reshape(-1)
        source_indices = prepared.source_anchor_indices
        if int(source_indices.numel()) != len(prepared.batch):
            raise ValueError("Prepared source-anchor indices must match the incoming batch")
        if int(source_indices.numel()) > 0 and int(source_indices.max()) >= int(incoming_visibility.numel()):
            raise ValueError("Incoming anchor visibility does not cover all prepared anchors")
        if int(existing_visibility.numel()) != self.map.anchor_count():
            raise ValueError("Existing anchor visibility must match the current global map")

        stats: dict[str, int | float] = {
            "hash_requested": int(prepared.requested),
            "hash_candidates": len(prepared.batch),
            "hash_visible_incoming": 0,
            "hash_visible_existing": int(existing_visibility.sum().item()),
            "hash_hits": 0,
            "hash_kept": len(prepared.batch),
            "hash_radius_voxels": float(radius_voxels),
            "hash_vectorized": 0,
            "hash_materialized_existing": 0,
        }
        for level in range(len(self.voxel_sizes)):
            level_incoming = int((prepared.batch.level == level).sum().item())
            stats[f"hash_level_{level}_incoming"] = level_incoming
            stats[f"hash_level_{level}_visible"] = 0
            stats[f"hash_level_{level}_hits"] = 0
            stats[f"hash_level_{level}_kept"] = level_incoming
        if len(prepared.batch) == 0 or self.map.anchor_count() == 0:
            return prepared, stats, None

        incoming_visible = incoming_visibility.index_select(
            0,
            source_indices.to(incoming_visibility.device),
        )
        stats["hash_visible_incoming"] = int(incoming_visible.sum().item())
        radius_scale = max(0.0, float(radius_voxels))
        radius_cells = max(0, int(math.ceil(radius_scale)))
        if (
            prepared.batch.xyz.is_cuda
            and self.map.xyz.is_cuda
            and prepared.batch.xyz.device == self.map.xyz.device
        ):
            return self._filter_against_visible_map_vectorized(
                prepared,
                incoming_visible=incoming_visible,
                existing_visibility=existing_visibility,
                radius_scale=radius_scale,
                radius_cells=radius_cells,
                update_existing_statistics=update_existing_statistics,
                stats=stats,
            )
        stats["hash_materialized_existing"] = self.map.anchor_count()
        keep = torch.ones(len(prepared.batch), dtype=torch.bool)
        incoming_xyz = prepared.batch.xyz.detach().cpu().float()
        incoming_level = prepared.batch.level.detach().cpu().long()
        incoming_voxel = prepared.batch.voxel_size[:, 0].detach().cpu().float()
        incoming_observation_count = prepared.batch.observation_count.detach().cpu().long()
        incoming_confidence = prepared.batch.confidence_accum.detach().cpu().float()
        incoming_last_seen = prepared.batch.last_seen_frame.detach().cpu().long()
        incoming_visible_cpu = incoming_visible.detach().cpu().bool()

        existing_xyz = self.map.get_xyz.detach().cpu().float()
        existing_level = self.map._anchor_level.detach().cpu().long()
        existing_quality = self.map._anchor_quality.detach().cpu().float()
        existing_voxel = (
            self.map.materialized_anchor_voxel_size()
            .detach()
            .cpu()
            .float()
            .reshape(-1)
        )
        existing_visible_cpu = existing_visibility.detach().cpu().bool()
        evidence: dict[int, tuple[int, float, int]] = {}

        for level in range(len(self.voxel_sizes)):
            level_new = torch.nonzero(
                incoming_visible_cpu & (incoming_level == level),
                as_tuple=False,
            ).flatten()
            level_old = torch.nonzero(
                existing_visible_cpu & (existing_level == level),
                as_tuple=False,
            ).flatten()
            level_hits = 0
            stats[f"hash_level_{level}_visible"] = int(level_new.numel())
            if int(level_new.numel()) > 0 and int(level_old.numel()) > 0 and radius_scale > 0.0:
                cell_size = float(
                    torch.maximum(
                        incoming_voxel.index_select(0, level_new).max(),
                        existing_voxel.index_select(0, level_old).max(),
                    ).clamp_min(1.0e-8)
                )
                old_grid = torch.floor(
                    existing_xyz.index_select(0, level_old) / cell_size
                ).to(torch.int64)
                occupied: dict[tuple[int, int, int], list[int]] = {}
                for local_row, coord in enumerate(old_grid.tolist()):
                    occupied.setdefault(
                        (int(coord[0]), int(coord[1]), int(coord[2])),
                        [],
                    ).append(int(level_old[local_row]))

                for new_row in level_new.tolist():
                    candidate_xyz = incoming_xyz[int(new_row)]
                    coord = torch.floor(candidate_xyz / cell_size).to(torch.int64)
                    candidates: list[int] = []
                    for dx in range(-radius_cells, radius_cells + 1):
                        for dy in range(-radius_cells, radius_cells + 1):
                            for dz in range(-radius_cells, radius_cells + 1):
                                candidates.extend(
                                    occupied.get(
                                        (
                                            int(coord[0]) + dx,
                                            int(coord[1]) + dy,
                                            int(coord[2]) + dz,
                                        ),
                                        (),
                                    )
                                )
                    if not candidates:
                        continue
                    candidate_rows = torch.tensor(candidates, dtype=torch.long)
                    distances = torch.linalg.norm(
                        existing_xyz.index_select(0, candidate_rows) - candidate_xyz,
                        dim=-1,
                    )
                    # Owner Sim(3) corrections can make nominally equal-level
                    # anchors carry different world voxel sizes.  A symmetric
                    # radius avoids order-dependent suppression.
                    candidate_voxel = existing_voxel.index_select(
                        0, candidate_rows
                    )
                    radii = radius_scale * 0.5 * (
                        float(incoming_voxel[int(new_row)]) + candidate_voxel
                    )
                    within = torch.nonzero(
                        distances <= radii + 1.0e-8,
                        as_tuple=False,
                    ).flatten()
                    if int(within.numel()) == 0:
                        continue
                    eligible_rows = candidate_rows.index_select(0, within)
                    eligible_distances = distances.index_select(0, within)
                    min_distance = eligible_distances.min()
                    nearest = torch.nonzero(
                        eligible_distances <= min_distance + 1.0e-8,
                        as_tuple=False,
                    ).flatten()
                    nearest_rows = eligible_rows.index_select(0, nearest)
                    nearest_quality = existing_quality.index_select(0, nearest_rows)
                    best_quality = nearest_quality.max()
                    quality_winners = nearest_rows[
                        nearest_quality >= best_quality - 1.0e-12
                    ]
                    hit = int(quality_winners.min())
                    keep[int(new_row)] = False
                    level_hits += 1
                    if update_existing_statistics:
                        previous = evidence.get(hit, (0, 0.0, 0))
                        evidence[hit] = (
                            previous[0] + int(incoming_observation_count[int(new_row)]),
                            previous[1] + float(incoming_confidence[int(new_row)]),
                            max(previous[2], int(incoming_last_seen[int(new_row)])),
                        )
            stats[f"hash_level_{level}_hits"] = int(level_hits)
            stats[f"hash_level_{level}_kept"] = (
                int(stats[f"hash_level_{level}_incoming"]) - int(level_hits)
            )

        kept = torch.nonzero(keep, as_tuple=False).flatten()
        filtered = prepared.index(kept.to(prepared.batch.xyz.device))
        hit_count = int((~keep).sum().item())
        stats["hash_hits"] = hit_count
        stats["hash_kept"] = len(filtered.batch)
        if not evidence:
            return filtered, stats, None
        evidence_rows = sorted(evidence)
        evidence_update = ExistingAnchorEvidenceUpdate(
            indices=torch.tensor(evidence_rows, dtype=torch.long),
            observation_count_delta=torch.tensor(
                [evidence[index][0] for index in evidence_rows],
                dtype=torch.long,
            ),
            confidence_accum_delta=torch.tensor(
                [evidence[index][1] for index in evidence_rows],
                dtype=torch.float32,
            ),
            last_seen_frame=torch.tensor(
                [evidence[index][2] for index in evidence_rows],
                dtype=torch.long,
            ),
        )
        return filtered, stats, evidence_update

    def commit_prepared_packet(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
        prepared: PreparedPacketFusion,
        *,
        evidence_update: ExistingAnchorEvidenceUpdate | None = None,
        extra_stats: dict[str, int | float] | None = None,
    ) -> dict[str, int | float]:
        before = self.map.anchor_count()
        depth_selected = bool(prepared.depth_selected)
        if self._depth_selected_mode and not depth_selected and before > 0:
            raise ValueError(
                "Cannot mix legacy scale-selected packets into a depth-selected voxel-anchor map"
            )
        if depth_selected:
            incoming = prepared.batch
        else:
            incoming = self.compact_within_window(prepared.batch)
        existing = self._batch_from_map(preserve_levels=depth_selected)
        if evidence_update is not None:
            rows = evidence_update.indices.to(existing.observation_count.device)
            if int(rows.numel()) > 0 and int(rows.max()) >= len(existing):
                raise ValueError("Existing-anchor evidence update is outside the current map")
            existing.observation_count[rows] += evidence_update.observation_count_delta.to(
                existing.observation_count
            )
            existing.confidence_accum[rows] += evidence_update.confidence_accum_delta.to(
                existing.confidence_accum
            )
            existing.last_seen_frame[rows] = torch.maximum(
                existing.last_seen_frame[rows],
                evidence_update.last_seen_frame.to(existing.last_seen_frame),
            )
        combined = incoming if len(existing) == 0 else self._concatenate([existing, incoming])
        compacted = (
            self._winner_take_owner_voxel(combined, preserve_levels=depth_selected)
            if self.lazy_owner_transforms
            else self._winner_take_global_voxel(combined, preserve_levels=depth_selected)
        )
        pre_cap_count = int(self.last_pre_cap_count)
        saturated = bool(self.last_saturated)
        if self.lazy_owner_transforms:
            self.map.set_lazy_owner_transform(
                int(packet.window_id),
                anchor_to_global,
                set_reference=int(packet.window_id)
                not in self.map._lazy_owner_reference_transforms,
            )
        if depth_selected:
            self._depth_selected_mode = True
        self._write_map(compacted)
        after = len(compacted)
        inserted_or_replaced = max(0, after - before)
        stats: dict[str, int | float] = {
            "requested": int(prepared.requested),
            "window_compacted": len(incoming),
            "inserted": inserted_or_replaced,
            "deduplicated": max(0, len(combined) - after),
            "anchors_before": before,
            "anchors_after": after,
            "anchors_before_safety_cap": pre_cap_count,
            "map_saturated": int(saturated),
        }
        stats.update(self._quality_diagnostics(packet, prepared.batch))
        if extra_stats:
            stats.update(extra_stats)
        for level in range(len(self.voxel_sizes)):
            stats[f"incoming_level_{level}"] = int((incoming.level == level).sum().detach().cpu())
            stats[f"global_level_{level}"] = int((compacted.level == level).sum().detach().cpu())
        return stats

    def fuse_packet(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
    ) -> dict[str, int | float]:
        prepared = self.prepare_packet_batch(packet, anchor_to_global)
        return self.commit_prepared_packet(
            packet,
            anchor_to_global,
            prepared,
        )

    def apply_owner_corrections(
        self,
        old_transforms: dict[int, torch.Tensor],
        new_transforms: dict[int, torch.Tensor],
    ) -> dict[str, int]:
        if self.lazy_owner_transforms:
            moved = 0
            owners = self.map._anchor_owner_window_id
            for owner in sorted(set(old_transforms) & set(new_transforms)):
                count = int((owners == int(owner)).sum().item())
                if count <= 0:
                    continue
                self.map.set_lazy_owner_transform(int(owner), new_transforms[owner])
                moved += count
            return {"moved": moved, "deduplicated": 0, "lazy": 1}
        batch = self._batch_from_map()
        if len(batch) == 0:
            return {"moved": 0, "deduplicated": 0}
        moved = 0
        for owner in sorted(set(old_transforms) & set(new_transforms)):
            mask = batch.owner_window_id == int(owner)
            if not bool(mask.any()):
                continue
            old = old_transforms[owner].to(batch.xyz)
            new = new_transforms[owner].to(batch.xyz)
            delta = new @ sim3_inverse(old)
            delta_scale, delta_rotation, _ = sim3_components(delta)
            batch.xyz[mask] = apply_sim3(delta, batch.xyz[mask])
            delta_quaternion = matrix_to_quaternion(delta_rotation).view(1, 4)
            batch.rotation[mask] = normalize_quaternion(
                quaternion_multiply(delta_quaternion, batch.rotation[mask])
            )
            batch.scale[mask] = delta_scale * batch.scale[mask]
            batch.voxel_size[mask] = delta_scale * batch.voxel_size[mask]
            batch.sh_coefficients[mask] = rotate_sh_coefficients(
                batch.sh_coefficients[mask], delta_rotation, self.map.active_sh_degree
            )
            moved += int(mask.sum())
        before = len(batch)
        compacted = self._winner_take_global_voxel(batch)
        self._write_map(compacted)
        return {"moved": moved, "deduplicated": before - len(compacted)}

    def deduplicate_owner_neighborhood(
        self,
        owner_window_ids: set[int] | list[int] | tuple[int, ...],
    ) -> int:
        """Prune cross-owner voxel duplicates only in a committed loop neighborhood."""

        if not self.lazy_owner_transforms or self.map.anchor_count() <= 1:
            return 0
        owners = {int(value) for value in owner_window_ids}
        if len(owners) < 2:
            return 0
        owner_ids = self.map._anchor_owner_window_id
        selected_cpu = torch.zeros_like(owner_ids, dtype=torch.bool)
        for owner in owners:
            selected_cpu |= owner_ids == owner
        selected_rows_cpu = torch.nonzero(selected_cpu, as_tuple=False).flatten()
        if int(selected_rows_cpu.numel()) <= 1:
            return 0

        device = self.map.get_xyz.device
        selected_rows = selected_rows_cpu.to(device=device, dtype=torch.long)
        xyz = self.map.get_xyz.detach().index_select(0, selected_rows)
        scale = self.map.get_scaling.detach().index_select(0, selected_rows)
        level, grid = self._levels_and_grid(xyz, scale)
        key = torch.cat([level[:, None], grid], dim=-1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=True)
        if int(unique.shape[0]) == int(selected_rows.shape[0]):
            return 0
        quality = self.map._anchor_quality.to(device=device).index_select(
            0, selected_rows
        )
        opacity = self.map.get_opacity.detach().reshape(-1).index_select(
            0, selected_rows
        )
        score = quality.clamp_min(0.0) * opacity.clamp_min(0.0)
        maximum = score.new_full((int(unique.shape[0]),), -torch.inf)
        maximum.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
        local_index = torch.arange(
            int(selected_rows.shape[0]), device=device, dtype=torch.long
        )
        candidate = torch.where(
            score >= maximum[inverse] - 1.0e-12,
            local_index,
            torch.full_like(local_index, int(selected_rows.shape[0])),
        )
        winner = torch.full(
            (int(unique.shape[0]),),
            int(selected_rows.shape[0]),
            device=device,
            dtype=torch.long,
        )
        winner.scatter_reduce_(0, inverse, candidate, reduce="amin", include_self=True)
        keep_local = torch.zeros(
            int(selected_rows.shape[0]), device=device, dtype=torch.bool
        )
        keep_local[winner[winner < int(selected_rows.shape[0])]] = True
        duplicate_rows = selected_rows[~keep_local]
        if int(duplicate_rows.numel()) == 0:
            return 0
        prune = torch.zeros(self.map.anchor_count(), device=device, dtype=torch.bool)
        prune[duplicate_rows] = True
        return int(self.map.prune_anchors(prune))
