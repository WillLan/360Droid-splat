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

from .adapter import SH_C0
from .mapper import PanoGaussianMap


@dataclass
class GlobalExplicitGaussianBatch:
    xyz: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    sh_coefficients: torch.Tensor
    quality: torch.Tensor
    owner_window_id: torch.Tensor
    level: torch.Tensor
    grid_coord: torch.Tensor
    observation_count: torch.Tensor
    confidence_accum: torch.Tensor
    birth_frame: torch.Tensor
    last_seen_frame: torch.Tensor
    visibility_count: torch.Tensor
    render_error_ema: torch.Tensor

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def index(self, selected: torch.Tensor) -> "GlobalExplicitGaussianBatch":
        return GlobalExplicitGaussianBatch(
            **{name: getattr(self, name)[selected] for name in self.__dataclass_fields__}
        )


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
    ) -> None:
        self.map = gaussian_map
        values = sorted({float(value) for value in voxel_sizes if float(value) > 0.0})
        if not values:
            raise ValueError("voxel_sizes must contain at least one positive value")
        self.voxel_sizes = tuple(values)
        self.min_confidence = float(min_confidence)
        self.min_opacity = float(min_opacity)
        self.max_total_gaussians = max(0, int(max_total_gaussians))
        self.last_pre_cap_count = 0
        self.last_saturated = False

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
            sh_coefficients=torch.zeros(0, sh_count, 3, device=device, dtype=dtype),
            quality=torch.zeros(0, device=device, dtype=dtype),
            owner_window_id=torch.zeros(0, device=device, dtype=torch.long),
            level=torch.zeros(0, device=device, dtype=torch.long),
            grid_coord=torch.zeros(0, 3, device=device, dtype=torch.long),
            observation_count=torch.zeros(0, device=device, dtype=torch.long),
            confidence_accum=torch.zeros(0, device=device, dtype=dtype),
            birth_frame=torch.zeros(0, device=device, dtype=torch.long),
            last_seen_frame=torch.zeros(0, device=device, dtype=torch.long),
            visibility_count=torch.zeros(0, device=device, dtype=torch.long),
            render_error_ema=torch.zeros(0, device=device, dtype=dtype),
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
                    sh_coefficients=coefficients.reshape(-1, target_sh_count, 3)[selected],
                    quality=quality.reshape(-1)[selected],
                    owner_window_id=torch.full((count,), int(packet.window_id), device=device, dtype=torch.long),
                    level=level,
                    grid_coord=grid,
                    observation_count=torch.ones(count, device=device, dtype=torch.long),
                    confidence_accum=confidence.reshape(-1)[selected],
                    birth_frame=torch.full((count,), int(packet.frame_ids[view]), device=device, dtype=torch.long),
                    last_seen_frame=torch.full((count,), int(packet.frame_ids[-1]), device=device, dtype=torch.long),
                    visibility_count=torch.ones(count, device=device, dtype=torch.long),
                    render_error_ema=torch.zeros(count, device=device, dtype=dtype),
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
        return GlobalExplicitGaussianBatch(
            xyz=average(batch.xyz),
            scale=torch.exp(average(batch.scale.clamp_min(1.0e-8).log())),
            rotation=normalize_quaternion(average(quaternion)),
            opacity=average(batch.opacity).clamp(1.0e-5, 1.0 - 1.0e-5),
            sh_coefficients=average(batch.sh_coefficients),
            quality=(weight_sum[:, 0] / observation_count.clamp_min(1).to(weight_sum)).clamp_min(1.0e-8),
            owner_window_id=unique[:, 0].long(),
            level=unique[:, 1].long(),
            grid_coord=unique[:, 2:].long(),
            observation_count=observation_count,
            confidence_accum=confidence_accum,
            birth_frame=birth,
            last_seen_frame=last_seen,
            visibility_count=visibility,
            render_error_ema=average(batch.render_error_ema[:, None])[:, 0],
        )

    def _winner_take_global_voxel(self, batch: GlobalExplicitGaussianBatch) -> GlobalExplicitGaussianBatch:
        if len(batch) == 0:
            self.last_pre_cap_count = 0
            self.last_saturated = False
            return batch
        level, grid = self._levels_and_grid(batch.xyz, batch.scale)
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
        self.last_pre_cap_count = len(result)
        self.last_saturated = self.max_total_gaussians > 0 and len(result) > self.max_total_gaussians
        if self.max_total_gaussians > 0 and len(result) > self.max_total_gaussians:
            selected = torch.topk(result.quality, k=self.max_total_gaussians, largest=True).indices
            result = result.index(selected)
        return result

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

    def _batch_from_map(self) -> GlobalExplicitGaussianBatch:
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

        level, grid = self._levels_and_grid(self.map.get_xyz.detach(), self.map.get_scaling.detach())
        return GlobalExplicitGaussianBatch(
            xyz=self.map.get_xyz.detach().clone(),
            scale=self.map.get_scaling.detach().clone(),
            rotation=self.map.get_rotation.detach().clone(),
            opacity=self.map.get_opacity.detach().clone(),
            sh_coefficients=self.map.get_sh_coefficients.detach().clone(),
            quality=metadata("_anchor_quality", torch.ones(count)).to(dtype=dtype),
            owner_window_id=metadata("_anchor_owner_window_id", torch.full((count,), -1)).long(),
            level=level,
            grid_coord=grid,
            observation_count=metadata("_anchor_obs_count", torch.ones(count)).long(),
            confidence_accum=metadata("_anchor_conf_accum", torch.ones(count)).to(dtype=dtype),
            birth_frame=metadata("_anchor_birth_frame", torch.zeros(count)).long(),
            last_seen_frame=metadata("_anchor_last_seen_kf", torch.zeros(count)).long(),
            visibility_count=metadata("_anchor_visibility_count", torch.zeros(count)).long(),
            render_error_ema=metadata("_anchor_render_error_ema", torch.zeros(count)).to(dtype=dtype),
        )

    def _write_map(self, batch: GlobalExplicitGaussianBatch) -> None:
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        xyz = batch.xyz.to(device=device, dtype=dtype)
        scale = batch.scale.to(device=device, dtype=dtype)
        opacity = batch.opacity.to(device=device, dtype=dtype)
        sh = batch.sh_coefficients.to(device=device, dtype=dtype)
        self.map.xyz = nn.Parameter(xyz)
        self.map.rotation = nn.Parameter(normalize_quaternion(batch.rotation.to(device=device, dtype=dtype)))
        self.map.scaling = nn.Parameter(self.map._inverse_softplus_scale(scale))
        self.map.opacity_logit = nn.Parameter(self.map._inv_sigmoid(opacity))
        dc_rgb = (0.5 + SH_C0 * sh[:, 0]).clamp(0.0, 1.0)
        self.map.features = nn.Parameter(self.map._inv_sigmoid(dc_rgb))
        rest_count = int(self.map.sh_rest.shape[1])
        self.map.sh_rest = nn.Parameter(
            sh[:, 1 : 1 + rest_count].contiguous()
            if rest_count > 0
            else torch.zeros(len(batch), 0, 3, device=device, dtype=dtype)
        )
        self.map._anchor_level = batch.level.detach().cpu().to(torch.int8)
        sizes = torch.tensor(self.voxel_sizes, dtype=torch.float32)
        self.map._anchor_voxel_size = sizes[batch.level.detach().cpu().long()]
        self.map._anchor_grid_coord = batch.grid_coord.detach().cpu().to(torch.int32)
        self.map._anchor_obs_count = batch.observation_count.detach().cpu().to(torch.int32)
        self.map._anchor_conf_accum = batch.confidence_accum.detach().cpu().float()
        self.map._anchor_birth_frame = batch.birth_frame.detach().cpu().to(torch.int32)
        self.map._anchor_last_seen_kf = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_last_update_kf_ord = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_source_window_id = batch.owner_window_id.detach().cpu().to(torch.int32)
        self.map._anchor_source_frame_start = batch.birth_frame.detach().cpu().to(torch.int32)
        self.map._anchor_source_frame_end = batch.last_seen_frame.detach().cpu().to(torch.int32)
        self.map._anchor_inlier_obs = torch.zeros(len(batch), dtype=torch.int32)
        self.map._anchor_outlier_obs = torch.zeros(len(batch), dtype=torch.int32)
        self.map._anchor_owner_window_id = batch.owner_window_id.detach().cpu().to(torch.int32)
        self.map._anchor_quality = batch.quality.detach().cpu().float()
        self.map._anchor_visibility_count = batch.visibility_count.detach().cpu().to(torch.int32)
        self.map._anchor_render_error_ema = batch.render_error_ema.detach().cpu().float()

    def fuse_packet(
        self,
        packet: LocalGaussianWindowPacket,
        anchor_to_global: torch.Tensor,
    ) -> dict[str, int | float]:
        before = self.map.anchor_count()
        incoming_raw = self.packet_to_global_batch(packet, anchor_to_global)
        incoming = self.compact_within_window(incoming_raw)
        existing = self._batch_from_map()
        combined = incoming if len(existing) == 0 else self._concatenate([existing, incoming])
        compacted = self._winner_take_global_voxel(combined)
        pre_cap_count = int(self.last_pre_cap_count)
        saturated = bool(self.last_saturated)
        self._write_map(compacted)
        after = len(compacted)
        inserted_or_replaced = max(0, after - before)
        stats: dict[str, int | float] = {
            "requested": len(incoming_raw),
            "window_compacted": len(incoming),
            "inserted": inserted_or_replaced,
            "deduplicated": max(0, len(combined) - after),
            "anchors_before": before,
            "anchors_after": after,
            "anchors_before_safety_cap": pre_cap_count,
            "map_saturated": int(saturated),
        }
        stats.update(self._quality_diagnostics(packet, incoming_raw))
        for level in range(len(self.voxel_sizes)):
            stats[f"incoming_level_{level}"] = int((incoming.level == level).sum().detach().cpu())
            stats[f"global_level_{level}"] = int((compacted.level == level).sum().detach().cpu())
        return stats

    def apply_owner_corrections(
        self,
        old_transforms: dict[int, torch.Tensor],
        new_transforms: dict[int, torch.Tensor],
    ) -> dict[str, int]:
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
            batch.sh_coefficients[mask] = rotate_sh_coefficients(
                batch.sh_coefficients[mask], delta_rotation, self.map.active_sh_degree
            )
            moved += int(mask.sum())
        before = len(batch)
        compacted = self._winner_take_global_voxel(batch)
        self._write_map(compacted)
        return {"moved": moved, "deduplicated": before - len(compacted)}

    def prune_lifecycle(
        self,
        *,
        current_frame: int,
        max_stale_frames: int = 0,
        max_render_error: float = float("inf"),
    ) -> int:
        batch = self._batch_from_map()
        if len(batch) == 0:
            return 0
        mean_confidence = batch.confidence_accum / batch.observation_count.clamp_min(1).to(
            batch.confidence_accum
        )
        prune = (batch.opacity[:, 0] < self.min_opacity) | (mean_confidence < self.min_confidence)
        if int(max_stale_frames) > 0:
            prune |= (
                (int(current_frame) - batch.last_seen_frame) > int(max_stale_frames)
            ) & (batch.visibility_count <= 0)
        if math.isfinite(float(max_render_error)):
            prune |= batch.render_error_ema > float(max_render_error)
        removed = int(prune.sum())
        if removed:
            self._write_map(batch.index(~prune))
        return removed

    def update_lifecycle_observations(
        self,
        visible_indices: torch.Tensor,
        *,
        current_frame: int,
        render_error: torch.Tensor | None = None,
        ema_decay: float = 0.9,
    ) -> None:
        """Update visibility/error metadata from an external render pass."""

        batch = self._batch_from_map()
        indices = visible_indices.detach().to(device=batch.xyz.device, dtype=torch.long).view(-1)
        if len(batch) == 0 or indices.numel() == 0:
            return
        indices = indices[(indices >= 0) & (indices < len(batch))].unique()
        if indices.numel() == 0:
            return
        batch.visibility_count[indices] += 1
        batch.last_seen_frame[indices] = int(current_frame)
        if render_error is not None:
            error = render_error.detach().to(batch.render_error_ema).view(-1)
            if int(error.numel()) != int(indices.numel()):
                raise ValueError("render_error must contain one value per visible Gaussian")
            decay = min(max(float(ema_decay), 0.0), 1.0)
            batch.render_error_ema[indices] = (
                decay * batch.render_error_ema[indices]
                + (1.0 - decay) * error.clamp_min(0.0)
            )
        self._write_map(batch)
