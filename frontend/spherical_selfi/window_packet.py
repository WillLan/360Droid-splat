"""Private window packet shared by Stage 2, local BA, and the global backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch
import torch.nn.functional as F

from geometry.sim3 import apply_sim3_to_pose
from geometry.spherical_erp import erp_pixel_to_unit_ray
from models.per_pixel_gaussian_observation import (
    PerPixelGaussianObservation,
    normalize_quaternion,
)
from models.spherical_selfi_gaussian_head import erp_bilinear_resize


def _normalize_mask(
    mask: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    default: bool,
) -> torch.Tensor:
    if mask is None:
        return torch.full_like(reference, default, dtype=torch.bool)
    value = mask.to(device=reference.device)
    if value.ndim == 4:
        value = value.unsqueeze(2)
    if value.shape[:3] != reference.shape[:3]:
        raise ValueError("Window masks must share B/S/channel dimensions with observation validity")
    if tuple(value.shape[-2:]) != tuple(reference.shape[-2:]):
        batch, views = int(value.shape[0]), int(value.shape[1])
        value = F.interpolate(
            value.float().reshape(batch * views, 1, *value.shape[-2:]),
            size=tuple(reference.shape[-2:]),
            mode="nearest",
        ).reshape_as(reference)
    return value.bool()


def build_panorama_retrieval_descriptor(
    features: torch.Tensor,
    *,
    latitude_bands: int = 8,
) -> torch.Tensor:
    """Build a yaw-invariant, spherical-area-weighted descriptor.

    ``features`` may be ``SxCxHxW`` or ``BxSxCxHxW``. Longitude is pooled out,
    so cyclic ERP shifts do not change the descriptor.
    """

    value = features
    squeeze_batch = False
    if value.ndim == 4:
        value = value.unsqueeze(0)
        squeeze_batch = True
    if value.ndim != 5:
        raise ValueError("features must have shape SxCxHxW or BxSxCxHxW")
    batch, views, channels, height, width = (int(v) for v in value.shape)
    rows = torch.arange(height, device=value.device, dtype=value.dtype) + 0.5
    area = torch.cos(math.pi * (rows / float(height) - 0.5)).clamp_min(1.0e-6)
    area = area.view(1, 1, 1, height, 1)
    global_mean = (value * area).sum(dim=(-2, -1)) / (area.sum() * float(width)).clamp_min(1.0e-8)
    parts = [global_mean]
    band_count = max(1, min(int(latitude_bands), height))
    boundaries = torch.linspace(0, height, band_count + 1, device=value.device).round().long()
    for band in range(band_count):
        start, stop = int(boundaries[band]), int(boundaries[band + 1])
        stop = max(start + 1, stop)
        band_area = area[..., start:stop, :]
        band_value = value[..., start:stop, :]
        pooled = (band_value * band_area).sum(dim=(-2, -1)) / (
            band_area.sum() * float(width)
        ).clamp_min(1.0e-8)
        parts.append(pooled)
    descriptor = torch.cat(parts, dim=-1)
    descriptor = F.normalize(descriptor.float(), dim=-1, eps=1.0e-8)
    return descriptor[0] if squeeze_batch else descriptor


def _verification_features(features: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tuple(features.shape[-2:]) == tuple(int(v) for v in size):
        return F.normalize(features.float(), dim=2, eps=1.0e-8)
    batch, views, channels = (int(features.shape[i]) for i in range(3))
    resized = erp_bilinear_resize(
        features.reshape(batch * views, channels, *features.shape[-2:]),
        tuple(int(v) for v in size),
    ).reshape(batch, views, channels, *tuple(int(v) for v in size))
    return F.normalize(resized.float(), dim=2, eps=1.0e-8)


@dataclass
class LocalGaussianWindowPacket:
    window_id: int
    anchor_frame_id: int
    frame_ids: tuple[int, ...]
    local_poses_c2w: torch.Tensor  # Sx4x4, first pose identity
    observation: PerPixelGaussianObservation
    adapter_features: torch.Tensor  # 1xSxCxHxW
    retrieval_descriptors: torch.Tensor  # SxD
    verification_features: torch.Tensor  # 1xSxCxHvxWv
    valid_mask: torch.Tensor  # 1xSx1xHxW
    sky_mask: torch.Tensor
    static_mask: torch.Tensor
    geometry_consistency: torch.Tensor
    match_quality: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        views = len(self.frame_ids)
        if views < 1:
            raise ValueError("LocalGaussianWindowPacket requires at least one frame")
        if int(self.anchor_frame_id) != int(self.frame_ids[0]):
            raise ValueError("anchor_frame_id must equal the first frame id")
        if tuple(self.local_poses_c2w.shape) != (views, 4, 4):
            raise ValueError("local_poses_c2w must have shape Sx4x4")
        if self.observation.batch_size != 1 or self.observation.num_source_views != views:
            raise ValueError("Window packet currently requires a B=1 observation matching frame_ids")
        identity = torch.eye(4, device=self.local_poses_c2w.device, dtype=self.local_poses_c2w.dtype)
        if not torch.allclose(self.local_poses_c2w[0], identity, atol=1.0e-4, rtol=1.0e-4):
            raise ValueError("The first local pose must be identity")
        expected = tuple(self.observation.valid_mask.shape)
        for name in ("valid_mask", "sky_mask", "static_mask", "geometry_consistency"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")

    @classmethod
    def from_observation(
        cls,
        *,
        window_id: int,
        observation: PerPixelGaussianObservation,
        adapter_features: torch.Tensor,
        frame_ids: list[int] | tuple[int, ...] | None = None,
        verification_size: tuple[int, int] | None = (32, 64),
        latitude_bands: int = 8,
        sky_mask: torch.Tensor | None = None,
        static_mask: torch.Tensor | None = None,
        geometry_consistency: torch.Tensor | None = None,
        match_quality: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LocalGaussianWindowPacket":
        if observation.batch_size != 1:
            raise ValueError("Runtime window packets require batch_size=1")
        if adapter_features.ndim != 5 or int(adapter_features.shape[0]) != 1:
            raise ValueError("adapter_features must have shape 1xSxCxHxW")
        views = observation.num_source_views
        if int(adapter_features.shape[1]) != views:
            raise ValueError("adapter_features and observation must have the same view count")
        ids = tuple(
            int(value)
            for value in (
                frame_ids
                if frame_ids is not None
                else observation.frame_ids[0].detach().cpu().tolist()
            )
        )
        if len(ids) != views:
            raise ValueError("frame_ids must match the observation view count")

        poses = observation.poses_c2w[0].float()
        anchor_inverse = torch.linalg.inv(poses[0])
        local_poses = anchor_inverse.view(1, 4, 4) @ poses
        local_poses[0] = torch.eye(4, device=local_poses.device, dtype=local_poses.dtype)
        local_observation = observation.with_geometry(
            poses_c2w=local_poses.unsqueeze(0).to(observation.poses_c2w)
        )
        valid = local_observation.valid_mask.bool()
        sky = _normalize_mask(sky_mask, valid, default=False)
        static = _normalize_mask(static_mask, valid, default=True)
        consistent = _normalize_mask(geometry_consistency, valid, default=True)
        valid = valid & ~sky & static & consistent
        normalized_features = F.normalize(adapter_features.float(), dim=2, eps=1.0e-8)
        retrieval = build_panorama_retrieval_descriptor(
            normalized_features[0], latitude_bands=latitude_bands
        )
        verification = _verification_features(normalized_features, verification_size)
        return cls(
            window_id=int(window_id),
            anchor_frame_id=int(ids[0]),
            frame_ids=ids,
            local_poses_c2w=local_poses,
            observation=local_observation,
            adapter_features=normalized_features,
            retrieval_descriptors=retrieval,
            verification_features=verification,
            valid_mask=valid,
            sky_mask=sky,
            static_mask=static,
            geometry_consistency=consistent,
            match_quality=dict(match_quality or {}),
            metadata=dict(metadata or {}),
        )

    def frame_index(self, frame_id: int) -> int:
        try:
            return self.frame_ids.index(int(frame_id))
        except ValueError as exc:
            raise KeyError(f"Frame {frame_id} is not part of window {self.window_id}") from exc

    def local_points(self, frame_index: int) -> torch.Tensor:
        index = int(frame_index)
        camera = self.observation.centers_camera()[0, index]
        pose = self.local_poses_c2w[index].to(camera)
        return torch.einsum("ij,hwj->hwi", pose[:3, :3], camera) + pose[:3, 3]

    def global_poses(self, anchor_to_global: torch.Tensor) -> torch.Tensor:
        transform = anchor_to_global.to(self.local_poses_c2w)
        return apply_sim3_to_pose(
            transform.view(1, 4, 4).expand(len(self.frame_ids), -1, -1),
            self.local_poses_c2w,
        )

    def compact_for_memory(self) -> "LocalGaussianWindowPacket":
        """Return a detached CPU record at loop-verification resolution.

        The live full-resolution packet is needed only for fusion and the next
        overlapping window.  Keeping it for every historical keyframe would
        make memory grow with ``H*W*SH`` per window, so loop memory stores this
        reduced representation instead.
        """

        target_hw = tuple(int(value) for value in self.verification_features.shape[-2:])
        observation = self.observation
        batch, views = observation.batch_size, observation.num_source_views

        def resize_channels(value: torch.Tensor, channels: int, *, nearest: bool = False) -> torch.Tensor:
            flat = value.detach().float().reshape(batch * views, channels, *value.shape[-2:])
            if tuple(flat.shape[-2:]) != target_hw:
                if nearest:
                    flat = F.interpolate(flat, size=target_hw, mode="nearest")
                else:
                    flat = erp_bilinear_resize(flat, target_hw)
            return flat.reshape(batch, views, channels, *target_hw).cpu()

        rgb_count = (int(observation.rgb_sh_degree) + 1) ** 2
        density_count = (int(observation.density_sh_degree) + 1) ** 2
        rgb = resize_channels(
            observation.rgb_sh.reshape(batch, views, rgb_count * 3, *observation.image_size),
            rgb_count * 3,
        ).reshape(batch, views, rgb_count, 3, *target_hw)
        density = resize_channels(observation.density_sh, density_count)
        quaternion = normalize_quaternion(
            resize_channels(observation.local_quaternion, 4).permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3)
        height, width = target_hw
        row, column = torch.meshgrid(
            torch.arange(height, dtype=torch.float32) + 0.5,
            torch.arange(width, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        source_uv = torch.stack([column, row], dim=-1)
        source_ray = erp_pixel_to_unit_ray(source_uv, height, width).float()
        compact_observation = replace(
            observation,
            initial_depth=resize_channels(observation.initial_depth, 1),
            depth_residual=resize_channels(observation.depth_residual, 1),
            refined_depth=resize_channels(observation.refined_depth, 1),
            poses_c2w=observation.poses_c2w.detach().cpu().float(),
            local_quaternion=quaternion,
            log_scale_multiplier=resize_channels(observation.log_scale_multiplier, 3),
            rgb_sh=rgb,
            density_sh=density,
            confidence=resize_channels(observation.confidence, 1),
            valid_mask=resize_channels(observation.valid_mask.float(), 1, nearest=True).bool(),
            source_uv=source_uv,
            source_ray=source_ray,
            frame_ids=observation.frame_ids.detach().cpu(),
        )

        def compact_mask(value: torch.Tensor) -> torch.Tensor:
            return resize_channels(value.float(), 1, nearest=True).bool()

        verification = self.verification_features.detach().cpu().float()
        return LocalGaussianWindowPacket(
            window_id=int(self.window_id),
            anchor_frame_id=int(self.anchor_frame_id),
            frame_ids=self.frame_ids,
            local_poses_c2w=self.local_poses_c2w.detach().cpu().float(),
            observation=compact_observation,
            adapter_features=verification,
            retrieval_descriptors=self.retrieval_descriptors.detach().cpu().float(),
            verification_features=verification,
            valid_mask=compact_mask(self.valid_mask),
            sky_mask=compact_mask(self.sky_mask),
            static_mask=compact_mask(self.static_mask),
            geometry_consistency=compact_mask(self.geometry_consistency),
            match_quality={key: value.detach().cpu() for key, value in self.match_quality.items()},
            metadata=dict(self.metadata),
        )


class LocalGaussianWindowQueue:
    """Mixin-style queue implementing the private frontend/backend side channel."""

    def __init__(self) -> None:
        self._local_gaussian_windows: list[LocalGaussianWindowPacket] = []

    def enqueue_local_gaussian_window(self, packet: LocalGaussianWindowPacket) -> None:
        self._local_gaussian_windows.append(packet)

    def consume_local_gaussian_windows(self) -> list[LocalGaussianWindowPacket]:
        packets = list(self._local_gaussian_windows)
        self._local_gaussian_windows.clear()
        return packets
