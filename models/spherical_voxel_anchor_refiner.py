"""Depth-selected voxel anchors and the simplified Stage 3 anchor refiner.

The legacy Stage 3 path keeps one Gaussian per valid ERP pixel.  This module
implements the config-gated alternative requested for the spherical-Selfi
pipeline: BA-refined per-pixel Gaussians are compacted into one explicit
Gaussian per depth-selected voxel, rendering error is pooled across target
views with binary validity weights, and a small shared GRU updates the anchor
parameters without changing camera poses or dense depth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import PanoRenderCamera
from geometry.pose import relative_c2w
from geometry.spherical_erp import sample_erp_with_wrap, unit_ray_to_erp_pixel
from models.per_pixel_gaussian_observation import (
    PerPixelGaussianObservation,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    real_sh_basis,
)
from models.spherical_recurrent_gaussian_refiner import (
    EncodedTargetReference,
    ReSplatErrorEncoder,
    quaternion_exp_map,
    quaternion_inverse,
    quaternion_log_map,
)


def _inv_sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(value / (1.0 - value))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``wxyz`` quaternions to rotation matrices."""

    q = normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    return torch.stack(
        [
            1.0 - two * (y.square() + z.square()),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x.square() + z.square()),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x.square() + y.square()),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _fibonacci_directions(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(max(16, int(count)), device=device, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / float(index.numel())
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack([radius * torch.cos(angle), z, radius * torch.sin(angle)], dim=-1)


def rotate_sh_coefficients(
    coefficients: torch.Tensor,
    rotation_local_to_target: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Rotate real RGB-SH coefficients into the target coordinate frame."""

    coefficient_count = (int(degree) + 1) ** 2
    work_dtype = (
        torch.float32
        if coefficients.dtype in {torch.float16, torch.bfloat16}
        else coefficients.dtype
    )
    directions_target = _fibonacci_directions(
        max(32, coefficient_count * 4),
        device=coefficients.device,
        dtype=work_dtype,
    )
    target_basis = real_sh_basis(degree, directions_target)
    directions_local = directions_target @ rotation_local_to_target.to(
        device=coefficients.device,
        dtype=work_dtype,
    )
    local_basis = real_sh_basis(degree, directions_local)
    transform = torch.linalg.pinv(target_basis) @ local_basis
    rotated = torch.einsum(
        "ij,...jc->...ic",
        transform,
        coefficients.to(dtype=work_dtype),
    )
    return rotated.to(dtype=coefficients.dtype)


@dataclass(frozen=True)
class VoxelAnchorConfig:
    depth_boundaries: tuple[float, float, float] = (5.0, 20.0, 40.0)
    voxel_sizes: tuple[float, float, float, float] = (0.04, 0.08, 0.16, 0.32)
    alpha_threshold: float = 0.05
    depth_abs_threshold: float = 0.10
    depth_rel_threshold: float = 0.05
    tangent_scale_floor_ratio: float = 1.0 / 3.0
    normal_scale_floor_ratio: float = 0.05
    hidden_dim: int = 32
    adapter_dim: int = 24
    iterations: int = 3
    use_resnet_error: bool = True
    pretrained_resnet: bool = True

    def __post_init__(self) -> None:
        if len(self.depth_boundaries) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.depth_boundaries
        ):
            raise ValueError("depth_boundaries must contain three positive finite values.")
        if not all(left < right for left, right in zip(self.depth_boundaries, self.depth_boundaries[1:])):
            raise ValueError("depth_boundaries must be strictly increasing.")
        if len(self.voxel_sizes) != 4 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.voxel_sizes
        ):
            raise ValueError("voxel_sizes must contain four positive finite values.")
        if int(self.iterations) != 3:
            raise ValueError("The simplified Stage 3 anchor refiner requires exactly three iterations.")
        if int(self.adapter_dim) <= 0 or int(self.hidden_dim) <= 0:
            raise ValueError("adapter_dim and hidden_dim must be positive.")
        if not 0.0 <= float(self.alpha_threshold) <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1].")
        if float(self.depth_abs_threshold) < 0.0 or float(self.depth_rel_threshold) < 0.0:
            raise ValueError("Depth thresholds must be non-negative.")
        if float(self.tangent_scale_floor_ratio) <= 0.0 or float(self.normal_scale_floor_ratio) <= 0.0:
            raise ValueError("Scale floor ratios must be positive.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "VoxelAnchorConfig":
        raw = dict(value or {})
        return cls(
            depth_boundaries=tuple(float(x) for x in raw.get("depth_boundaries", (5.0, 20.0, 40.0))),
            voxel_sizes=tuple(float(x) for x in raw.get("voxel_sizes", (0.04, 0.08, 0.16, 0.32))),
            alpha_threshold=float(raw.get("alpha_threshold", 0.05)),
            depth_abs_threshold=float(raw.get("depth_abs_threshold", 0.10)),
            depth_rel_threshold=float(raw.get("depth_rel_threshold", 0.05)),
            tangent_scale_floor_ratio=float(raw.get("tangent_scale_floor_ratio", 1.0 / 3.0)),
            normal_scale_floor_ratio=float(raw.get("normal_scale_floor_ratio", 0.05)),
            hidden_dim=int(raw.get("hidden_dim", 32)),
            adapter_dim=int(raw.get("adapter_dim", 24)),
            iterations=int(raw.get("iterations", 3)),
            use_resnet_error=bool(raw.get("use_resnet_error", True)),
            pretrained_resnet=bool(raw.get("pretrained_resnet", True)),
        )


def depth_to_voxel_level(reference_depth: torch.Tensor, config: VoxelAnchorConfig) -> torch.Tensor:
    """Return levels 0..3 with boundaries assigned to the coarser level."""

    boundaries = reference_depth.new_tensor(config.depth_boundaries)
    return torch.bucketize(reference_depth, boundaries, right=True).long()


def depth_to_voxel_size(reference_depth: torch.Tensor, config: VoxelAnchorConfig) -> torch.Tensor:
    levels = depth_to_voxel_level(reference_depth, config)
    sizes = reference_depth.new_tensor(config.voxel_sizes)
    return sizes[levels]


@dataclass
class AnchorMembership:
    anchor_index: torch.Tensor
    batch_index: torch.Tensor
    source_view_index: torch.Tensor
    source_pixel_uv: torch.Tensor
    weight: torch.Tensor
    reference_depth: torch.Tensor

    def detached(self, *, device: torch.device | str | None = None) -> "AnchorMembership":
        def move(value: torch.Tensor) -> torch.Tensor:
            result = value.detach()
            return result if device is None else result.to(device)

        return AnchorMembership(
            anchor_index=move(self.anchor_index),
            batch_index=move(self.batch_index),
            source_view_index=move(self.source_view_index),
            source_pixel_uv=move(self.source_pixel_uv),
            weight=move(self.weight),
            reference_depth=move(self.reference_depth),
        )


@dataclass
class ExplicitVoxelAnchorSet:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    anchor_indices: torch.Tensor
    config: dict[str, Any] | None = None
    active_sh_degree: int = 0
    max_sh_degree: int = 0

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    @property
    def get_features(self) -> torch.Tensor:
        return self.features


@dataclass
class VoxelAnchorObservation:
    """Variable-length batched explicit anchor state in first-frame coordinates."""

    base_xyz: torch.Tensor
    xyz: torch.Tensor
    voxel_center: torch.Tensor
    position_latent: torch.Tensor
    base_rotation: torch.Tensor
    rotation: torch.Tensor
    base_log_scales: torch.Tensor
    log_scales: torch.Tensor
    min_scales: torch.Tensor
    max_scales: torch.Tensor
    base_opacity_logit: torch.Tensor
    opacity_logit: torch.Tensor
    base_sh_coefficients: torch.Tensor
    sh_coefficients: torch.Tensor
    static_input: torch.Tensor
    quality: torch.Tensor
    level: torch.Tensor
    voxel_size: torch.Tensor
    grid_coord: torch.Tensor
    member_count: torch.Tensor
    batch_index: torch.Tensor
    local_poses_c2w: torch.Tensor
    frame_ids: torch.Tensor
    image_size: tuple[int, int]
    membership: AnchorMembership
    config: VoxelAnchorConfig
    renderer_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        count = int(self.xyz.shape[0])
        expected = {
            "base_xyz": (count, 3),
            "voxel_center": (count, 3),
            "position_latent": (count, 3),
            "base_rotation": (count, 4),
            "rotation": (count, 4),
            "base_log_scales": (count, 3),
            "log_scales": (count, 3),
            "min_scales": (count, 3),
            "max_scales": (count, 3),
            "base_opacity_logit": (count, 1),
            "opacity_logit": (count, 1),
            "base_sh_coefficients": (count, 9, 3),
            "sh_coefficients": (count, 9, 3),
            "static_input": (count, int(self.config.adapter_dim) + 3),
            "quality": (count, 1),
            "level": (count,),
            "voxel_size": (count, 1),
            "grid_coord": (count, 3),
            "member_count": (count, 1),
            "batch_index": (count,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(getattr(self, name).shape)}.")
        if self.local_poses_c2w.ndim != 4 or tuple(self.local_poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError("local_poses_c2w must have shape BxSx4x4.")
        if tuple(self.frame_ids.shape) != tuple(self.local_poses_c2w.shape[:2]):
            raise ValueError("frame_ids must match local_poses_c2w B/S dimensions.")

    @property
    def batch_size(self) -> int:
        return int(self.local_poses_c2w.shape[0])

    @property
    def num_views(self) -> int:
        return int(self.local_poses_c2w.shape[1])

    @property
    def num_anchors(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scales).clamp_min(1.0e-8)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    def indices_for_batch(self, batch_index: int) -> torch.Tensor:
        return torch.nonzero(self.batch_index == int(batch_index), as_tuple=False).flatten()

    def materialize_batch(
        self,
        cameras: Iterable[PanoRenderCamera],
        *,
        batch_index: int,
    ) -> ExplicitVoxelAnchorSet:
        camera_list = list(cameras)
        indices = self.indices_for_batch(batch_index)
        xyz = self.xyz.index_select(0, indices)
        scaling = self.scaling.index_select(0, indices)
        rotation = self.rotation.index_select(0, indices)
        opacity = self.opacity.index_select(0, indices)
        coefficients = self.sh_coefficients.index_select(0, indices)
        if int(indices.numel()) == 0:
            features = xyz.new_zeros(len(camera_list), 0, 3)
        else:
            target_centers = torch.stack(
                [camera.c2w[:3, 3].to(xyz) for camera in camera_list], dim=0
            )
            direction = F.normalize(
                xyz.unsqueeze(0) - target_centers[:, None, :], dim=-1, eps=1.0e-8
            )
            basis = real_sh_basis(2, direction)
            features = (0.5 + torch.einsum("cnk,nkd->cnd", basis, coefficients)).clamp(0.0, 1.0)
        return ExplicitVoxelAnchorSet(
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            features=features,
            anchor_indices=indices,
            config=self.renderer_config,
        )

    def with_updates(self, **values: torch.Tensor) -> "VoxelAnchorObservation":
        return replace(self, **values)

    def detach_parameters(self, *, device: torch.device | str | None = None) -> "VoxelAnchorObservation":
        def move(value: torch.Tensor) -> torch.Tensor:
            result = value.detach()
            return result if device is None else result.to(device)

        tensor_fields = {
            name: move(getattr(self, name))
            for name in (
                "base_xyz", "xyz", "voxel_center", "position_latent",
                "base_rotation", "rotation", "base_log_scales", "log_scales",
                "min_scales", "max_scales", "base_opacity_logit", "opacity_logit",
                "base_sh_coefficients", "sh_coefficients", "static_input", "quality",
                "level", "voxel_size", "grid_coord", "member_count", "batch_index",
                "local_poses_c2w", "frame_ids",
            )
        }
        return replace(
            self,
            **tensor_fields,
            membership=self.membership.detached(device=device),
        )

    def detach_for_backend(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> "VoxelAnchorObservation":
        """Drop per-member provenance after the final refinement iteration."""

        result = self.detach_parameters(device=device)
        tensor_device = result.xyz.device
        tensor_dtype = result.xyz.dtype
        empty_long = torch.zeros(0, device=tensor_device, dtype=torch.long)
        empty = torch.zeros(0, device=tensor_device, dtype=tensor_dtype)
        return replace(
            result,
            membership=AnchorMembership(
                anchor_index=empty_long,
                batch_index=empty_long.clone(),
                source_view_index=empty_long.clone(),
                source_pixel_uv=torch.zeros(0, 2, device=tensor_device, dtype=tensor_dtype),
                weight=empty,
                reference_depth=empty.clone(),
            ),
        )

    def rescale_geometry(self, scale: float) -> "VoxelAnchorObservation":
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Anchor geometry scale must be positive and finite.")
        floor = torch.cat(
            [
                float(self.config.normal_scale_floor_ratio) * self.voxel_size,
                float(self.config.tangent_scale_floor_ratio) * self.voxel_size,
                float(self.config.tangent_scale_floor_ratio) * self.voxel_size,
            ],
            dim=-1,
        ).clamp_min(1.0e-6)
        base_scales = torch.maximum(torch.exp(self.base_log_scales) * value, floor)
        current_scales = torch.maximum(self.scaling * value, floor)
        poses = self.local_poses_c2w.clone()
        poses[:, :, :3, 3] *= value
        return replace(
            self,
            base_xyz=self.base_xyz * value,
            xyz=self.xyz * value,
            voxel_center=self.voxel_center * value,
            base_log_scales=base_scales.log(),
            log_scales=current_scales.log(),
            min_scales=floor,
            max_scales=torch.maximum(self.max_scales * value, current_scales),
            local_poses_c2w=poses,
            membership=replace(
                self.membership,
                reference_depth=self.membership.reference_depth * value,
            ),
        )


def _segment_sum(value: torch.Tensor, inverse: torch.Tensor, count: int) -> torch.Tensor:
    output = value.new_zeros((int(count),) + tuple(value.shape[1:]))
    output.index_add_(0, inverse, value)
    return output


def _chunked_eigh_3x3(
    covariance: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Avoid cuSOLVER's large-batch limit for full-resolution anchor windows."""

    if covariance.ndim != 3 or tuple(covariance.shape[-2:]) != (3, 3):
        raise ValueError("covariance must have shape Nx3x3")
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("Anchor covariance contains non-finite values after member filtering")
    size = max(1, int(chunk_size))
    eigenvalue_parts = []
    eigenvector_parts = []
    for start in range(0, int(covariance.shape[0]), size):
        values, vectors = torch.linalg.eigh(covariance[start : start + size].float())
        eigenvalue_parts.append(values)
        eigenvector_parts.append(vectors)
    if not eigenvalue_parts:
        return (
            covariance.new_zeros(0, 3, dtype=torch.float32),
            covariance.new_zeros(0, 3, 3, dtype=torch.float32),
        )
    return torch.cat(eigenvalue_parts, dim=0), torch.cat(eigenvector_parts, dim=0)


def voxelize_per_pixel_gaussians(
    observation: PerPixelGaussianObservation,
    adapter_features: torch.Tensor,
    images: torch.Tensor,
    config: VoxelAnchorConfig,
    *,
    valid_mask: torch.Tensor | None = None,
) -> VoxelAnchorObservation:
    """Fuse BA-refined per-pixel Gaussians into depth-selected local anchors."""

    batch, views = observation.batch_size, observation.num_source_views
    height, width = observation.image_size
    if tuple(adapter_features.shape) != (batch, views, config.adapter_dim, height, width):
        raise ValueError("adapter_features must match observation B/S/H/W and configured adapter_dim.")
    if tuple(images.shape) != (batch, views, 3, height, width):
        raise ValueError("images must have shape BxSx3xHxW matching the observation.")
    selection = observation.valid_mask.bool()
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(selection.shape):
            raise ValueError("valid_mask must match observation.valid_mask.")
        selection = selection & valid_mask.bool().to(selection.device)

    poses = observation.poses_c2w.to(observation.refined_depth)
    local_poses = relative_c2w(poses, poses[:, :1]).to(observation.refined_depth)
    local_poses[:, 0] = torch.eye(4, device=poses.device, dtype=poses.dtype)
    centers_camera = observation.centers_camera()
    dense_scales = observation.scales()
    dense_opacity = observation.source_view_confidence()
    local_rotation = local_poses[:, :, :3, :3]
    local_translation = local_poses[:, :, :3, 3]
    centers_local = torch.einsum("bsij,bshwj->bshwi", local_rotation, centers_camera)
    centers_local = centers_local + local_translation[:, :, None, None, :]
    reference_depth = torch.linalg.norm(centers_local, dim=-1)
    finite = torch.isfinite(centers_local).all(dim=-1) & torch.isfinite(reference_depth) & (reference_depth > 0.0)
    finite = finite & torch.isfinite(dense_scales).all(dim=2)
    finite = finite & torch.isfinite(observation.local_quaternion).all(dim=2)
    finite = finite & torch.isfinite(observation.confidence[:, :, 0])
    finite = finite & torch.isfinite(dense_opacity[:, :, 0])
    finite = finite & torch.isfinite(adapter_features).all(dim=2)
    finite = finite & torch.isfinite(images).all(dim=2)
    finite = finite & torch.isfinite(observation.rgb_sh).all(dim=(2, 3))
    selection_4d = selection[:, :, 0] & finite

    flat_selected = torch.nonzero(selection_4d.reshape(-1), as_tuple=False).flatten()
    device, dtype = centers_local.device, centers_local.dtype
    if int(flat_selected.numel()) == 0:
        empty = torch.zeros(0, device=device, dtype=dtype)
        empty_long = torch.zeros(0, device=device, dtype=torch.long)
        empty3 = torch.zeros(0, 3, device=device, dtype=dtype)
        empty4 = torch.zeros(0, 4, device=device, dtype=dtype)
        empty_sh = torch.zeros(0, 9, 3, device=device, dtype=dtype)
        return VoxelAnchorObservation(
            base_xyz=empty3, xyz=empty3.clone(), voxel_center=empty3.clone(), position_latent=empty3.clone(),
            base_rotation=empty4, rotation=empty4.clone(), base_log_scales=empty3.clone(), log_scales=empty3.clone(),
            min_scales=empty3.clone(), max_scales=empty3.clone(), base_opacity_logit=empty[:, None],
            opacity_logit=empty[:, None].clone(), base_sh_coefficients=empty_sh, sh_coefficients=empty_sh.clone(),
            static_input=torch.zeros(0, config.adapter_dim + 3, device=device, dtype=dtype), quality=empty[:, None],
            level=empty_long, voxel_size=empty[:, None], grid_coord=torch.zeros(0, 3, device=device, dtype=torch.long),
            member_count=empty[:, None], batch_index=empty_long, local_poses_c2w=local_poses,
            frame_ids=observation.frame_ids, image_size=(height, width),
            membership=AnchorMembership(empty_long, empty_long, empty_long, torch.zeros(0, 2, device=device, dtype=dtype), empty, empty),
            config=config, renderer_config=observation.config,
        )

    total_per_batch = views * height * width
    linear = flat_selected
    member_batch = torch.div(linear, total_per_batch, rounding_mode="floor")
    within_batch = linear.remainder(total_per_batch)
    member_view = torch.div(within_batch, height * width, rounding_mode="floor")
    member_pixel = within_batch.remainder(height * width)

    centers = centers_local.reshape(-1, 3).index_select(0, flat_selected)
    depths = reference_depth.reshape(-1).index_select(0, flat_selected)
    levels = depth_to_voxel_level(depths, config)
    voxel_sizes = depths.new_tensor(config.voxel_sizes)[levels]
    grids = torch.floor(centers / voxel_sizes[:, None]).long()
    keys = torch.cat([member_batch[:, None], levels[:, None], grids], dim=-1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True, sorted=True)
    anchor_count = int(unique.shape[0])

    confidence = observation.confidence[:, :, 0].reshape(-1).index_select(0, flat_selected).to(dtype)
    opacity_member = dense_opacity[:, :, 0].reshape(-1).index_select(0, flat_selected).to(dtype)
    # The Stage-2 confidence is the existing observation-quality score.  Keep
    # it as the sole moment-matching weight instead of applying opacity twice.
    weight = confidence.clamp_min(1.0e-8)
    weight_sum = _segment_sum(weight[:, None], inverse, anchor_count).clamp_min(1.0e-8)

    def average(value: torch.Tensor) -> torch.Tensor:
        shaped = weight.view(-1, *([1] * (value.ndim - 1)))
        return _segment_sum(shaped * value, inverse, anchor_count) / weight_sum.view(
            anchor_count, *([1] * (value.ndim - 1))
        )

    anchor_xyz = average(centers)
    anchor_levels = unique[:, 1].long()
    anchor_grids = unique[:, 2:].long()
    anchor_voxel = centers.new_tensor(config.voxel_sizes)[anchor_levels]
    voxel_center = (anchor_grids.to(dtype) + 0.5) * anchor_voxel[:, None]

    member_scale = dense_scales.permute(0, 1, 3, 4, 2).reshape(-1, 3).index_select(0, flat_selected).to(dtype)
    local_quaternion = observation.local_quaternion.permute(0, 1, 3, 4, 2).reshape(-1, 4).index_select(0, flat_selected).to(dtype)
    member_pose_rotation = local_rotation[member_batch, member_view]
    member_rotation_matrix = member_pose_rotation @ quaternion_to_matrix(local_quaternion)
    covariance = member_rotation_matrix @ torch.diag_embed(member_scale.square()) @ member_rotation_matrix.transpose(-1, -2)
    offset = centers - anchor_xyz[inverse]
    covariance = covariance + offset[:, :, None] * offset[:, None, :]
    anchor_covariance = average(covariance)
    anchor_covariance = 0.5 * (anchor_covariance + anchor_covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = _chunked_eigh_3x3(anchor_covariance)
    determinant = torch.linalg.det(eigenvectors)
    eigenvectors = eigenvectors.clone()
    eigenvectors[determinant < 0.0, :, 2] *= -1.0
    eigenvalues = eigenvalues.to(dtype=dtype).clamp_min(1.0e-12)
    eigenvectors = eigenvectors.to(dtype=dtype)
    anchor_scale = torch.sqrt(eigenvalues)
    floor = torch.stack(
        [
            float(config.normal_scale_floor_ratio) * anchor_voxel,
            float(config.tangent_scale_floor_ratio) * anchor_voxel,
            float(config.tangent_scale_floor_ratio) * anchor_voxel,
        ],
        dim=-1,
    )
    min_scales = floor.clamp_min(1.0e-6)
    anchor_scale = torch.maximum(anchor_scale, min_scales)
    # The floor prevents holes, but moment matching is never clipped.  This
    # upper bound only constrains the subsequent three log-scale updates.
    max_scales = anchor_scale * math.exp(sum(SimplifiedVoxelAnchorRefiner.scale_limits))
    anchor_rotation = matrix_to_quaternion(eigenvectors)

    rgb_sh = observation.rgb_sh.permute(0, 1, 4, 5, 2, 3).reshape(-1, 9, 3)
    rgb_sh = rgb_sh.index_select(0, flat_selected).to(dtype)
    rotated_sh = torch.empty_like(rgb_sh)
    for batch_index in range(batch):
        for source_index in range(views):
            member_mask = (member_batch == batch_index) & (member_view == source_index)
            if bool(member_mask.any()):
                rotated_sh[member_mask] = rotate_sh_coefficients(
                    rgb_sh[member_mask],
                    local_rotation[batch_index, source_index],
                    degree=2,
                )
    anchor_sh = average(rotated_sh)
    anchor_opacity = average(opacity_member[:, None]).clamp(1.0e-5, 1.0 - 1.0e-5)

    features = adapter_features.permute(0, 1, 3, 4, 2).reshape(-1, config.adapter_dim).index_select(0, flat_selected).to(dtype)
    rgb = images.permute(0, 1, 3, 4, 2).reshape(-1, 3).index_select(0, flat_selected).to(dtype)
    static_input = torch.cat([average(features), average(rgb)], dim=-1)
    quality = (weight_sum / _segment_sum(torch.ones_like(weight)[:, None], inverse, anchor_count).clamp_min(1.0)).clamp(0.0, 1.0)
    member_count = _segment_sum(torch.ones_like(weight)[:, None], inverse, anchor_count)

    normalized = (2.0 * (anchor_xyz - voxel_center) / anchor_voxel[:, None]).clamp(-0.999, 0.999)
    position_latent = torch.atanh(normalized)
    parameterized_xyz = voxel_center + 0.5 * anchor_voxel[:, None] * torch.tanh(position_latent)
    source_uv = observation.source_uv.reshape(-1, 2).index_select(0, member_pixel)
    membership = AnchorMembership(
        anchor_index=inverse,
        batch_index=member_batch,
        source_view_index=member_view,
        source_pixel_uv=source_uv,
        weight=weight,
        reference_depth=depths,
    )
    return VoxelAnchorObservation(
        base_xyz=parameterized_xyz,
        xyz=parameterized_xyz.clone(),
        voxel_center=voxel_center,
        position_latent=position_latent,
        base_rotation=anchor_rotation,
        rotation=anchor_rotation.clone(),
        base_log_scales=anchor_scale.log(),
        log_scales=anchor_scale.log().clone(),
        min_scales=min_scales,
        max_scales=max_scales,
        base_opacity_logit=_inv_sigmoid(anchor_opacity),
        opacity_logit=_inv_sigmoid(anchor_opacity).clone(),
        base_sh_coefficients=anchor_sh,
        sh_coefficients=anchor_sh.clone(),
        static_input=static_input,
        quality=quality,
        level=anchor_levels,
        voxel_size=anchor_voxel[:, None],
        grid_coord=anchor_grids,
        member_count=member_count,
        batch_index=unique[:, 0].long(),
        local_poses_c2w=local_poses,
        frame_ids=observation.frame_ids,
        image_size=(height, width),
        membership=membership,
        config=config,
        renderer_config=observation.config,
    )


@dataclass
class VoxelAnchorRenderGroup:
    rendered: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    anchor_visibility: torch.Tensor
    profiles: dict[str, float]


def render_voxel_anchor_group(renderer: Any, observation: VoxelAnchorObservation) -> VoxelAnchorRenderGroup:
    """Render each batch's shared anchors into all local target cameras."""

    batch, views = observation.batch_size, observation.num_views
    height, width = observation.image_size
    rendered, depth, alpha = [], [], []
    visibility = torch.zeros(
        batch,
        views,
        observation.num_anchors,
        device=observation.xyz.device,
        dtype=torch.bool,
    )
    profile_sums: dict[str, float] = {}
    for batch_idx in range(batch):
        cameras = [
            PanoRenderCamera(height, width, observation.local_poses_c2w[batch_idx, target].float())
            for target in range(views)
        ]
        explicit = observation.materialize_batch(cameras, batch_index=batch_idx)
        package = renderer.render_cameras(cameras, explicit)
        rendered.append(package["render"])
        depth.append(package["depth"])
        alpha.append(package["alpha"])
        visible = package["visibility_filter"].to(device=visibility.device).bool()
        if tuple(visible.shape) != (views, int(explicit.anchor_indices.numel())):
            raise ValueError("Batched anchor visibility must have shape SxN_batch.")
        visibility[batch_idx, :, explicit.anchor_indices] = visible
        for key, value in package.items():
            if str(key).startswith("profile_renderer_") and isinstance(value, (float, int)):
                profile_sums[str(key)] = profile_sums.get(str(key), 0.0) + float(value)
    return VoxelAnchorRenderGroup(
        rendered=torch.stack(rendered, dim=0),
        depth=torch.stack(depth, dim=0),
        alpha=torch.stack(alpha, dim=0),
        anchor_visibility=visibility,
        profiles={
            "materialized_gaussians": float(observation.num_anchors) / max(1, batch),
            **{key: value / max(1, batch) for key, value in profile_sums.items()},
        },
    )


@dataclass
class AnchorErrorPoolOutput:
    feature: torch.Tensor
    raw_statistics: torch.Tensor
    has_feedback: torch.Tensor
    coverage: torch.Tensor


class BinaryAnchorErrorPooler(nn.Module):
    """Pool exactly signed mean, absolute mean, coverage, and an 8D global token."""

    def __init__(self, error_dim: int = 32, global_dim: int = 8) -> None:
        super().__init__()
        self.error_dim = int(error_dim)
        self.global_dim = int(global_dim)
        self.global_projection = nn.Linear(self.error_dim, self.global_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(2 * self.error_dim + 1 + self.global_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.error_dim),
        )

    def forward(
        self,
        observation: VoxelAnchorObservation,
        target_error_maps: torch.Tensor,
        render_group: VoxelAnchorRenderGroup,
        target_valid: torch.Tensor,
    ) -> AnchorErrorPoolOutput:
        batch, views, channels, low_h, low_w = target_error_maps.shape
        if (batch, views, channels) != (observation.batch_size, observation.num_views, self.error_dim):
            raise ValueError("target_error_maps must have shape BxSx32xH4xW4.")
        height, width = observation.image_size
        if tuple(render_group.depth.shape) != (batch, views, 1, height, width):
            raise ValueError("Rendered depth must have shape BxSx1xHxW.")
        if tuple(render_group.alpha.shape) != tuple(render_group.depth.shape):
            raise ValueError("Rendered alpha must match rendered depth.")
        if tuple(target_valid.shape) != tuple(render_group.depth.shape):
            raise ValueError("target_valid must have shape BxSx1xHxW.")
        if tuple(render_group.anchor_visibility.shape) != (batch, views, observation.num_anchors):
            raise ValueError("anchor_visibility must have shape BxSxN.")

        rows = torch.arange(low_h, device=target_error_maps.device, dtype=target_error_maps.dtype) + 0.5
        area = torch.cos(math.pi * (rows / float(low_h) - 0.5)).clamp_min(0.0).view(1, 1, 1, low_h, 1)
        global_mean = (target_error_maps * area).sum(dim=(-2, -1)) / (
            area.sum().clamp_min(1.0e-8) * float(low_w)
        )
        global_token = self.global_projection(global_mean.float()).to(target_error_maps.dtype).mean(dim=1)

        signed_sum = observation.xyz.new_zeros(observation.num_anchors, self.error_dim)
        absolute_sum = torch.zeros_like(signed_sum)
        count = observation.xyz.new_zeros(observation.num_anchors, 1)
        for batch_idx in range(batch):
            indices = observation.indices_for_batch(batch_idx)
            if int(indices.numel()) == 0:
                continue
            xyz = observation.xyz.index_select(0, indices)
            for target in range(views):
                pose = observation.local_poses_c2w[batch_idx, target].to(xyz)
                point = torch.einsum("ij,nj->ni", pose[:3, :3].transpose(0, 1), xyz - pose[:3, 3])
                anchor_depth = torch.linalg.norm(point, dim=-1)
                ray = F.normalize(point, dim=-1, eps=1.0e-8)
                uv = unit_ray_to_erp_pixel(ray, height, width)
                low_uv = uv.clone()
                low_uv[:, 0] *= float(low_w) / float(width)
                low_uv[:, 1] *= float(low_h) / float(height)
                sampled_error = sample_erp_with_wrap(target_error_maps[batch_idx, target], low_uv)
                sampled_depth = sample_erp_with_wrap(render_group.depth[batch_idx, target], uv)[:, 0]
                sampled_alpha = sample_erp_with_wrap(render_group.alpha[batch_idx, target], uv)[:, 0]
                sampled_target_valid = sample_erp_with_wrap(target_valid[batch_idx, target].float(), uv)[:, 0] > 0.5
                depth_error = (sampled_depth - anchor_depth).abs()
                threshold = float(observation.config.depth_abs_threshold) + float(
                    observation.config.depth_rel_threshold
                ) * anchor_depth
                valid = render_group.anchor_visibility[batch_idx, target].index_select(0, indices)
                valid = valid & sampled_target_valid
                valid = valid & torch.isfinite(anchor_depth) & torch.isfinite(sampled_depth) & torch.isfinite(sampled_alpha)
                valid = valid & (sampled_alpha > float(observation.config.alpha_threshold))
                valid = valid & (depth_error <= threshold)
                weight = valid.to(sampled_error.dtype).unsqueeze(-1)
                signed_sum.index_add_(0, indices, sampled_error * weight)
                absolute_sum.index_add_(0, indices, sampled_error.abs() * weight)
                count.index_add_(0, indices, weight[:, :1].to(count.dtype))

        denom = count.clamp_min(1.0)
        signed_mean = signed_sum / denom
        absolute_mean = absolute_sum / denom
        coverage = count / float(max(1, views))
        broadcast_global = global_token.index_select(0, observation.batch_index)
        raw = torch.cat([signed_mean, absolute_mean, coverage, broadcast_global], dim=-1)
        if int(raw.shape[-1]) != 73:
            raise AssertionError(f"Simplified anchor error statistics must be 73D, got {raw.shape[-1]}.")
        feature = self.output_projection(raw)
        return AnchorErrorPoolOutput(
            feature=feature,
            raw_statistics=raw,
            has_feedback=count[:, 0] > 0.0,
            coverage=coverage,
        )


@dataclass
class VoxelAnchorRefinerOutput:
    observation: VoxelAnchorObservation
    hidden: torch.Tensor
    raw_geometry: torch.Tensor
    raw_appearance: torch.Tensor
    normalized_update_energy: torch.Tensor


class SimplifiedVoxelAnchorRefiner(nn.Module):
    position_limits = (0.25, 0.15, 0.10)
    rotation_limits_deg = (5.0, 3.0, 2.0)
    scale_limits = (0.15, 0.10, 0.05)
    rgb_dc_limits = (0.25, 0.15, 0.10)
    rgb_ac_limits = (0.10, 0.075, 0.05)
    opacity_limits = (1.0, 0.75, 0.50)

    def __init__(self, config: VoxelAnchorConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(int(config.adapter_dim) + 3, 32),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(38, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.fusion = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        self.hidden_initializer = nn.Linear(64, hidden_dim)
        self.gru = nn.GRUCell(64, hidden_dim)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.GELU(),
            nn.Linear(64, 9),
        )
        self.appearance_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.GELU(),
            nn.Linear(64, 28),
        )
        self._zero_output_heads()

    def _zero_output_heads(self) -> None:
        for head in (self.geometry_head, self.appearance_head):
            output = head[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _state_tensor(observation: VoxelAnchorObservation) -> torch.Tensor:
        position = (observation.xyz - observation.base_xyz) / observation.voxel_size.clamp_min(1.0e-8)
        relative_q = quaternion_multiply(observation.rotation, quaternion_inverse(observation.base_rotation))
        rotation = quaternion_log_map(relative_q)
        log_scale = observation.log_scales - observation.base_log_scales
        sh = (observation.sh_coefficients - observation.base_sh_coefficients).reshape(-1, 27)
        opacity = observation.opacity_logit - observation.base_opacity_logit
        return torch.cat([position, rotation, log_scale, sh, opacity, observation.quality], dim=-1)

    def forward(
        self,
        observation: VoxelAnchorObservation,
        error: AnchorErrorPoolOutput,
        *,
        iteration_index: int,
        hidden: torch.Tensor | None = None,
    ) -> VoxelAnchorRefinerOutput:
        index = int(iteration_index)
        if index < 0 or index >= 3:
            raise ValueError("iteration_index must be 0, 1, or 2.")
        if tuple(error.feature.shape) != (observation.num_anchors, 32):
            raise ValueError("Anchor error feature must have shape Nx32.")
        static = self.static_encoder(observation.static_input)
        state = self.state_encoder(self._state_tensor(observation))
        fused = self.fusion(torch.cat([static, state, error.feature], dim=-1))
        initialized_hidden = torch.tanh(self.hidden_initializer(torch.cat([static, state], dim=-1)))
        if hidden is None:
            hidden = initialized_hidden
        else:
            hidden = hidden + initialized_hidden * 0.0
        next_hidden = self.gru(fused, hidden)
        geometry = self.geometry_head(torch.cat([next_hidden, state, error.feature], dim=-1))
        appearance = self.appearance_head(torch.cat([next_hidden, static, error.feature], dim=-1))
        update_mask = error.has_feedback.to(geometry.dtype).unsqueeze(-1)

        position_delta = (
            float(self.position_limits[index])
            * observation.voxel_size
            * torch.tanh(geometry[:, :3])
            * update_mask
        )
        proposed_xyz = observation.xyz + position_delta
        normalized = (2.0 * (proposed_xyz - observation.voxel_center) / observation.voxel_size).clamp(-0.999, 0.999)
        proposed_latent = torch.atanh(normalized)
        proposed_xyz = observation.voxel_center + 0.5 * observation.voxel_size * torch.tanh(proposed_latent)
        active = error.has_feedback.unsqueeze(-1)
        position_latent = torch.where(active, proposed_latent, observation.position_latent)
        xyz = torch.where(active, proposed_xyz, observation.xyz)

        raw_rotation = torch.tanh(geometry[:, 3:6])
        max_angle = math.radians(float(self.rotation_limits_deg[index]))
        rotation_norm = raw_rotation.norm(dim=-1, keepdim=True)
        rotation_vector = raw_rotation * torch.clamp(max_angle / rotation_norm.clamp_min(1.0e-8), max=1.0)
        rotation_vector = rotation_vector * update_mask
        proposed_rotation = normalize_quaternion(
            quaternion_multiply(quaternion_exp_map(rotation_vector), observation.rotation)
        )
        rotation = torch.where(active, proposed_rotation, observation.rotation)

        scale_delta = float(self.scale_limits[index]) * torch.tanh(geometry[:, 6:9]) * update_mask
        proposed_log_scales = (observation.log_scales + scale_delta).clamp(
            observation.min_scales.log(), observation.max_scales.log()
        )
        log_scales = torch.where(active, proposed_log_scales, observation.log_scales)

        rgb_raw = appearance[:, :27].reshape(-1, 9, 3)
        rgb_limits = rgb_raw.new_full((1, 9, 1), float(self.rgb_ac_limits[index]))
        rgb_limits[:, 0] = float(self.rgb_dc_limits[index])
        rgb_delta = rgb_limits * torch.tanh(rgb_raw) * update_mask[:, None]
        proposed_sh = observation.sh_coefficients + rgb_delta
        sh = torch.where(active[:, None], proposed_sh, observation.sh_coefficients)
        opacity_delta = float(self.opacity_limits[index]) * torch.tanh(appearance[:, 27:28]) * update_mask
        proposed_opacity_logit = observation.opacity_logit + opacity_delta
        opacity_logit = torch.where(active, proposed_opacity_logit, observation.opacity_logit)

        updated = observation.with_updates(
            xyz=xyz,
            position_latent=position_latent,
            rotation=rotation,
            log_scales=log_scales,
            sh_coefficients=sh,
            opacity_logit=opacity_logit,
        )
        if observation.num_anchors == 0:
            energy = observation.xyz.sum() * 0.0
        else:
            energy = torch.stack(
                [
                    (position_delta / observation.voxel_size.clamp_min(1.0e-8)).square().mean(),
                    rotation_vector.square().mean(),
                    scale_delta.square().mean(),
                    rgb_delta.square().mean(),
                    opacity_delta.square().mean(),
                ]
            ).mean()
        return VoxelAnchorRefinerOutput(
            observation=updated,
            hidden=next_hidden,
            raw_geometry=geometry,
            raw_appearance=appearance,
            normalized_update_energy=energy,
        )


class VoxelAnchorStage3Model(nn.Module):
    """ReSplat error encoder + binary pooler + simplified anchor GRU."""

    def __init__(self, config: VoxelAnchorConfig) -> None:
        super().__init__()
        self.config = config
        self.error_encoder = ReSplatErrorEncoder(
            use_resnet=bool(config.use_resnet_error),
            pretrained_resnet=bool(config.pretrained_resnet),
        )
        self.error_pooler = BinaryAnchorErrorPooler()
        self.refiner = SimplifiedVoxelAnchorRefiner(config)

    @torch.no_grad()
    def encode_references(self, images: torch.Tensor) -> EncodedTargetReference:
        batch, views, channels, height, width = images.shape
        return self.error_encoder.encode_reference(images.reshape(batch * views, channels, height, width))

    def forward_step(
        self,
        observation: VoxelAnchorObservation,
        render_group: VoxelAnchorRenderGroup,
        reference: EncodedTargetReference,
        target_valid: torch.Tensor,
        *,
        iteration_index: int,
        hidden: torch.Tensor | None,
    ) -> VoxelAnchorRefinerOutput:
        batch, views = observation.batch_size, observation.num_views
        error_maps = self.error_encoder(
            render_group.rendered.reshape(batch * views, 3, *observation.image_size),
            reference,
        )
        error_maps = error_maps.reshape(batch, views, 32, *error_maps.shape[-2:])
        pooled = self.error_pooler(observation, error_maps, render_group, target_valid)
        return self.refiner(
            observation,
            pooled,
            iteration_index=iteration_index,
            hidden=hidden,
        )


def load_voxel_anchor_checkpoint(
    path: str,
    *,
    model: VoxelAnchorStage3Model,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "spherical_voxel_anchor_refiner_v1":
        raise ValueError(f"Unsupported voxel-anchor refiner checkpoint: {path}.")
    model.load_state_dict(payload["model"], strict=True)
    return payload
