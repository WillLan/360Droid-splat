"""Private window packet shared by Stage 2, local BA, and the global backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from geometry.pose import relative_c2w
from geometry.sim3 import apply_sim3_to_c2w
from geometry.spherical_erp import erp_pixel_to_unit_ray
from geometry.spherical_spectral_descriptor import (
    SO3_SH_GRAM_DESCRIPTOR_VERSION,
    build_so3_sh_gram_descriptor,
)
from models.per_pixel_gaussian_observation import (
    PerPixelGaussianObservation,
    normalize_quaternion,
)

if TYPE_CHECKING:
    from models.spherical_voxel_anchor_refiner import VoxelAnchorObservation
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


def _normalize_probability(
    probability: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    default: float,
) -> torch.Tensor:
    if probability is None:
        return torch.full_like(reference, float(default), dtype=torch.float32)
    value = probability.to(device=reference.device, dtype=torch.float32)
    if value.ndim == 4:
        value = value.unsqueeze(2)
    if value.shape[:3] != reference.shape[:3]:
        raise ValueError("Window probability maps must share B/S/channel dimensions with observation validity")
    if tuple(value.shape[-2:]) != tuple(reference.shape[-2:]):
        batch, views = int(value.shape[0]), int(value.shape[1])
        value = erp_bilinear_resize(
            value.reshape(batch * views, 1, *value.shape[-2:]),
            tuple(reference.shape[-2:]),
        ).reshape_as(reference.float())
    return value.clamp(0.0, 1.0)


def build_panorama_retrieval_descriptor(
    features: torch.Tensor,
    *,
    latitude_bands: int = 8,
    spatial_weight: torch.Tensor | None = None,
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
    area = area.view(1, 1, 1, height, 1).expand(1, 1, 1, height, width)
    if spatial_weight is not None:
        weight = spatial_weight.to(device=value.device, dtype=value.dtype)
        if weight.ndim == 4:
            weight = weight.unsqueeze(0)
        if weight.ndim != 5 or tuple(weight.shape[:2]) != (batch, views):
            raise ValueError("spatial_weight must have shape Sx1xHxW or BxSx1xHxW")
        if tuple(weight.shape[-2:]) != (height, width):
            weight = erp_bilinear_resize(
                weight.reshape(batch * views, 1, *weight.shape[-2:]), (height, width)
            ).reshape(batch, views, 1, height, width)
        area = area * weight.clamp(0.0, 1.0)
    global_mean = (value * area).sum(dim=(-2, -1)) / area.sum(dim=(-2, -1)).clamp_min(1.0e-8)
    parts = [global_mean]
    band_count = max(1, min(int(latitude_bands), height))
    boundaries = torch.linspace(0, height, band_count + 1, device=value.device).round().long()
    for band in range(band_count):
        start, stop = int(boundaries[band]), int(boundaries[band + 1])
        stop = max(start + 1, stop)
        band_area = area[..., start:stop, :]
        band_value = value[..., start:stop, :]
        pooled = (band_value * band_area).sum(dim=(-2, -1)) / band_area.sum(
            dim=(-2, -1)
        ).clamp_min(1.0e-8)
        parts.append(pooled)
    descriptor = torch.cat(parts, dim=-1)
    descriptor = F.normalize(descriptor.float(), dim=-1, eps=1.0e-8)
    return descriptor[0] if squeeze_batch else descriptor


def build_configured_panorama_retrieval_descriptor(
    features: torch.Tensor,
    *,
    mode: str = "latitude_bands",
    latitude_bands: int = 8,
    max_degree: int = 6,
    num_samples: int = 2048,
    spatial_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Build the configured loop descriptor and return its stable version tag."""

    descriptor_mode = str(mode).strip().lower()
    if descriptor_mode in {"latitude_bands", "yaw", "yaw_latitude_bands", "legacy"}:
        return (
            build_panorama_retrieval_descriptor(
                features,
                latitude_bands=latitude_bands,
                spatial_weight=spatial_weight,
            ),
            "yaw_latitude_bands_v1",
        )
    if descriptor_mode in {"so3_sh_gram", "so3", "spherical_harmonic_gram"}:
        return (
            build_so3_sh_gram_descriptor(
                features,
                max_degree=max_degree,
                num_samples=num_samples,
                spatial_weight=spatial_weight,
            ),
            SO3_SH_GRAM_DESCRIPTOR_VERSION,
        )
    raise ValueError(
        "retrieval descriptor mode must be 'latitude_bands' or 'so3_sh_gram'; "
        f"got {mode!r}"
    )


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
class BoundaryMatchBlock:
    """Compact first-to-last adapter correspondences for one local window.

    All correspondences use the canonical first-frame -> last-frame direction.
    Reverse adapter queries are swapped into that direction before entering the
    packet.  Network scores remain diagnostics/hard-gate inputs; they are never
    used as continuous global-BA weights.
    """

    source_uv: torch.Tensor  # Nx2, first-frame ERP pixels
    target_uv: torch.Tensor  # Nx2, last-frame ERP pixels
    source_bearing: torch.Tensor  # Nx3
    target_bearing: torch.Tensor  # Nx3
    top1_cosine: torch.Tensor  # N
    top2_margin: torch.Tensor  # N
    normalized_entropy: torch.Tensor  # N, normalized to [0, 1]

    def __post_init__(self) -> None:
        count = int(self.source_uv.shape[0])
        if tuple(self.source_uv.shape) != (count, 2) or tuple(self.target_uv.shape) != (count, 2):
            raise ValueError("Boundary match UV arrays must have shape Nx2")
        if tuple(self.source_bearing.shape) != (count, 3) or tuple(self.target_bearing.shape) != (count, 3):
            raise ValueError("Boundary match bearings must have shape Nx3")
        for name in ("top1_cosine", "top2_margin", "normalized_entropy"):
            if tuple(getattr(self, name).shape) != (count,):
                raise ValueError(f"Boundary match {name} must have shape N")

    @property
    def count(self) -> int:
        return int(self.source_uv.shape[0])

    def detached_clone(self, *, device: torch.device | str | None = None) -> "BoundaryMatchBlock":
        def clone(value: torch.Tensor) -> torch.Tensor:
            result = value.detach().clone()
            return result if device is None else result.to(device)

        return BoundaryMatchBlock(
            source_uv=clone(self.source_uv),
            target_uv=clone(self.target_uv),
            source_bearing=clone(self.source_bearing),
            target_bearing=clone(self.target_bearing),
            top1_cosine=clone(self.top1_cosine),
            top2_margin=clone(self.top2_margin),
            normalized_entropy=clone(self.normalized_entropy),
        )


@dataclass
class ChunkStrideMatchBlock:
    """Canonical matches between consecutive chunk-anchor frames."""

    source_index: int
    target_index: int
    source_uv: torch.Tensor
    target_uv: torch.Tensor
    source_bearing: torch.Tensor
    target_bearing: torch.Tensor
    top1_cosine: torch.Tensor
    top2_margin: torch.Tensor
    normalized_entropy: torch.Tensor
    query_direction: torch.Tensor  # N, 0=source->target, 1=target->source

    def __post_init__(self) -> None:
        if int(self.source_index) < 0 or int(self.target_index) <= int(
            self.source_index
        ):
            raise ValueError(
                "Chunk stride match indices must satisfy 0 <= source < target"
            )
        count = int(self.source_uv.shape[0])
        if tuple(self.source_uv.shape) != (count, 2) or tuple(
            self.target_uv.shape
        ) != (count, 2):
            raise ValueError("Chunk stride match UV arrays must have shape Nx2")
        if tuple(self.source_bearing.shape) != (count, 3) or tuple(
            self.target_bearing.shape
        ) != (count, 3):
            raise ValueError(
                "Chunk stride match bearings must have shape Nx3"
            )
        for name in (
            "top1_cosine",
            "top2_margin",
            "normalized_entropy",
            "query_direction",
        ):
            if tuple(getattr(self, name).shape) != (count,):
                raise ValueError(f"Chunk stride match {name} must have shape N")
        if count > 0 and not bool(
            ((self.query_direction == 0) | (self.query_direction == 1)).all()
        ):
            raise ValueError("Chunk stride query_direction must contain only 0/1")

    @property
    def count(self) -> int:
        return int(self.source_uv.shape[0])

    def detached_clone(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> "ChunkStrideMatchBlock":
        def clone(value: torch.Tensor) -> torch.Tensor:
            result = value.detach().clone()
            return result if device is None else result.to(device)

        return ChunkStrideMatchBlock(
            source_index=int(self.source_index),
            target_index=int(self.target_index),
            source_uv=clone(self.source_uv),
            target_uv=clone(self.target_uv),
            source_bearing=clone(self.source_bearing),
            target_bearing=clone(self.target_bearing),
            top1_cosine=clone(self.top1_cosine),
            top2_margin=clone(self.top2_margin),
            normalized_entropy=clone(self.normalized_entropy),
            query_direction=clone(self.query_direction).long(),
        )


def chunk_stride_matches_from_cache(
    cache: Any,
    image_size: tuple[int, int],
    *,
    stride: int,
) -> ChunkStrideMatchBlock | None:
    """Canonicalize cached bidirectional ``0 <-> stride`` correspondences."""

    target = int(stride)
    if (
        cache is None
        or int(cache.batch_size) != 1
        or target <= 0
        or target >= int(cache.num_views)
    ):
        return None
    height, width = (int(value) for value in image_size)
    entropy_scale = max(math.log(max(2, height * width)), 1.0e-8)
    values: dict[str, list[torch.Tensor]] = {
        "source_uv": [],
        "target_uv": [],
        "source_bearing": [],
        "target_bearing": [],
        "top1_cosine": [],
        "top2_margin": [],
        "normalized_entropy": [],
        "query_direction": [],
    }
    for edge_index, pair in enumerate(cache.edges.detach().cpu().tolist()):
        source_index, target_index = int(pair[0]), int(pair[1])
        if (source_index, target_index) not in {(0, target), (target, 0)}:
            continue
        keep = cache.valid_mask[0, edge_index].bool()
        if not bool(keep.any()):
            continue
        if source_index == 0:
            source_uv = cache.source_uv[0, 0, keep]
            target_uv = cache.target_uv[0, edge_index, keep]
            source_bearing = cache.source_ray[0, 0, keep]
            target_bearing = cache.target_ray[0, edge_index, keep]
            direction = 0
        else:
            source_uv = cache.target_uv[0, edge_index, keep]
            target_uv = cache.source_uv[0, target, keep]
            source_bearing = cache.target_ray[0, edge_index, keep]
            target_bearing = cache.source_ray[0, target, keep]
            direction = 1
        count = int(source_uv.shape[0])
        values["source_uv"].append(source_uv)
        values["target_uv"].append(target_uv)
        values["source_bearing"].append(source_bearing)
        values["target_bearing"].append(target_bearing)
        values["top1_cosine"].append(cache.top1_cosine[0, edge_index, keep])
        values["top2_margin"].append(cache.top2_margin[0, edge_index, keep])
        values["normalized_entropy"].append(
            (cache.entropy[0, edge_index, keep] / entropy_scale).clamp(0.0, 1.0)
        )
        values["query_direction"].append(
            torch.full(
                (count,),
                direction,
                device=source_uv.device,
                dtype=torch.long,
            )
        )
    if not values["source_uv"]:
        return None
    with torch.inference_mode(False):
        return ChunkStrideMatchBlock(
            source_index=0,
            target_index=target,
            **{
                name: torch.cat(parts, dim=0).detach().clone()
                for name, parts in values.items()
            },
        )


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
    valid_mask: torch.Tensor  # compatibility alias for finite_gaussian_mask
    finite_gaussian_mask: torch.Tensor  # 1xSx1xHxW
    sky_prob: torch.Tensor
    sky_mask: torch.Tensor
    static_mask: torch.Tensor
    geometry_consistency: torch.Tensor
    pre_depth_shift_depth: torch.Tensor | None = None
    anchor_observation: VoxelAnchorObservation | None = None
    boundary_matches: BoundaryMatchBlock | None = None
    chunk_stride_matches: ChunkStrideMatchBlock | None = None
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
        for name in ("valid_mask", "finite_gaussian_mask", "sky_prob", "sky_mask", "static_mask", "geometry_consistency"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not torch.equal(self.valid_mask.bool(), self.finite_gaussian_mask.bool()):
            raise ValueError("valid_mask must equal finite_gaussian_mask for backend compatibility")
        if self.pre_depth_shift_depth is not None and tuple(
            self.pre_depth_shift_depth.shape
        ) != tuple(self.observation.refined_depth.shape):
            raise ValueError(
                "pre_depth_shift_depth must match observation.refined_depth shape"
            )
        if self.anchor_observation is not None:
            anchors = self.anchor_observation
            if anchors.batch_size != 1 or anchors.num_views != views:
                raise ValueError("anchor_observation must be B=1 and match packet frame_ids")
            if not torch.equal(anchors.frame_ids[0].to(self.observation.frame_ids), self.observation.frame_ids[0]):
                raise ValueError("anchor_observation frame_ids must match the dense observation")

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
        retrieval_descriptor_mode: str = "latitude_bands",
        retrieval_descriptor_max_degree: int = 6,
        retrieval_descriptor_num_samples: int = 2048,
        retrieval_descriptor_store_fp16: bool = False,
        sky_prob: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
        sky_threshold: float = 0.5,
        static_mask: torch.Tensor | None = None,
        geometry_consistency: torch.Tensor | None = None,
        pre_depth_shift_depth: torch.Tensor | None = None,
        anchor_observation: VoxelAnchorObservation | None = None,
        boundary_matches: BoundaryMatchBlock | None = None,
        chunk_stride_matches: ChunkStrideMatchBlock | None = None,
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
        local_poses = relative_c2w(poses, poses[0].view(1, 4, 4))
        local_poses[0] = torch.eye(4, device=local_poses.device, dtype=local_poses.dtype)
        local_observation = observation.with_geometry(
            poses_c2w=local_poses.unsqueeze(0).to(observation.poses_c2w)
        )
        valid = local_observation.valid_mask.bool()
        probability = _normalize_probability(sky_prob, valid, default=0.0)
        sky = (
            _normalize_mask(sky_mask, valid, default=False)
            if sky_mask is not None
            else probability >= float(sky_threshold)
        )
        static = _normalize_mask(static_mask, valid, default=True)
        consistent = _normalize_mask(geometry_consistency, valid, default=True)
        valid = valid & ~sky & static & consistent
        normalized_features = F.normalize(adapter_features.float(), dim=2, eps=1.0e-8)
        descriptor_weight = valid.float() * (1.0 - probability).clamp(0.0, 1.0)
        retrieval, descriptor_version = build_configured_panorama_retrieval_descriptor(
            normalized_features[0],
            mode=retrieval_descriptor_mode,
            latitude_bands=latitude_bands,
            max_degree=retrieval_descriptor_max_degree,
            num_samples=retrieval_descriptor_num_samples,
            spatial_weight=descriptor_weight[0],
        )
        if retrieval_descriptor_store_fp16:
            retrieval = retrieval.to(dtype=torch.float16)
        verification = _verification_features(normalized_features, verification_size)
        packet_metadata = dict(metadata or {})
        packet_metadata.update(
            {
                "retrieval_descriptor_mode": str(retrieval_descriptor_mode).lower(),
                "retrieval_descriptor_version": descriptor_version,
                "retrieval_descriptor_dim": int(retrieval.shape[-1]),
                "retrieval_descriptor_max_degree": int(retrieval_descriptor_max_degree),
                "retrieval_descriptor_num_samples": int(retrieval_descriptor_num_samples),
                "retrieval_descriptor_storage": (
                    "float16" if retrieval_descriptor_store_fp16 else "float32"
                ),
            }
        )
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
            finite_gaussian_mask=valid,
            sky_prob=probability,
            sky_mask=sky,
            static_mask=static,
            geometry_consistency=consistent,
            pre_depth_shift_depth=(
                None
                if pre_depth_shift_depth is None
                else pre_depth_shift_depth.detach().clone().to(local_observation.refined_depth)
            ),
            anchor_observation=anchor_observation,
            boundary_matches=boundary_matches,
            chunk_stride_matches=chunk_stride_matches,
            match_quality=dict(match_quality or {}),
            metadata=packet_metadata,
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
        return apply_sim3_to_c2w(
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

        def compact_probability(value: torch.Tensor) -> torch.Tensor:
            return resize_channels(value.float(), 1).clamp(0.0, 1.0)

        verification = self.verification_features.detach().cpu().float()
        compact_metadata = dict(self.metadata)
        # This per-anchor sidecar is consumed by insertion filtering and has
        # no loop-closure use after the anchors have been fused.
        compact_metadata.pop("voxel_anchor_source_view_mask", None)
        return LocalGaussianWindowPacket(
            window_id=int(self.window_id),
            anchor_frame_id=int(self.anchor_frame_id),
            frame_ids=self.frame_ids,
            local_poses_c2w=self.local_poses_c2w.detach().cpu().float(),
            observation=compact_observation,
            adapter_features=verification,
            retrieval_descriptors=self.retrieval_descriptors.detach().cpu().clone(),
            verification_features=verification,
            valid_mask=compact_mask(self.valid_mask),
            finite_gaussian_mask=compact_mask(self.finite_gaussian_mask),
            sky_prob=compact_probability(self.sky_prob),
            sky_mask=compact_mask(self.sky_mask),
            static_mask=compact_mask(self.static_mask),
            geometry_consistency=compact_mask(self.geometry_consistency),
            pre_depth_shift_depth=None,
            # Historical packets retain only the low-resolution data needed
            # for loop verification.  The explicit anchors have already been
            # fused into the backend map before this compact copy is stored.
            anchor_observation=None,
            boundary_matches=(
                None
                if self.boundary_matches is None
                else self.boundary_matches.detached_clone(device="cpu")
            ),
            chunk_stride_matches=(
                None
                if self.chunk_stride_matches is None
                else self.chunk_stride_matches.detached_clone(device="cpu")
            ),
            match_quality={key: value.detach().cpu() for key, value in self.match_quality.items()},
            metadata=compact_metadata,
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
