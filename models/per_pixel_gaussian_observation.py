"""Dense per-pixel Gaussian observations for the spherical Selfi Stage 2 head.

The canonical representation remains dense and pixel aligned.  Flattening and
opacity pruning happen only while materializing a particular render camera.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize ``wxyz`` quaternions with an identity fallback."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"Quaternion must end in four values, got {tuple(quaternion.shape)}.")
    norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return torch.where(norm > eps, quaternion / norm.clamp_min(eps), identity)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose two broadcastable ``wxyz`` quaternions."""

    left, right = torch.broadcast_tensors(left, right)
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def matrix_to_quaternion(matrix: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Convert rotation matrices to normalized ``wxyz`` quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrix must end in 3x3, got {tuple(matrix.shape)}.")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0.0).sqrt()
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()], dim=-1),
        ],
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., :, None].clamp_min(eps))
    choice = q_abs.argmax(dim=-1)
    gather = choice[..., None, None].expand(*choice.shape, 1, 4)
    quaternion = candidates.gather(dim=-2, index=gather).squeeze(-2)
    return normalize_quaternion(quaternion, eps=eps)


def real_sh_basis(degree: int, direction: torch.Tensor) -> torch.Tensor:
    """Evaluate the real 3DGS spherical-harmonic basis through degree two."""

    value = int(degree)
    if value < 0 or value > 2:
        raise ValueError(f"Only SH degrees 0, 1, and 2 are supported, got {degree}.")
    if direction.shape[-1] != 3:
        raise ValueError(f"Direction must end in three values, got {tuple(direction.shape)}.")
    direction = F.normalize(direction, dim=-1, eps=1.0e-8)
    x, y, z = direction.unbind(dim=-1)
    basis = [torch.ones_like(x) * SH_C0]
    if value >= 1:
        basis.extend([-SH_C1 * y, SH_C1 * z, -SH_C1 * x])
    if value >= 2:
        basis.extend(
            [
                SH_C2[0] * x * y,
                SH_C2[1] * y * z,
                SH_C2[2] * (2.0 * z.square() - x.square() - y.square()),
                SH_C2[3] * x * z,
                SH_C2[4] * (x.square() - y.square()),
            ]
        )
    return torch.stack(basis, dim=-1)


@dataclass
class ExplicitPerPixelGaussianSet:
    """A differentiable renderer-compatible explicit Gaussian collection."""

    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    confidence: torch.Tensor
    source_frame_index: torch.Tensor
    source_pixel_uv: torch.Tensor
    source_ray: torch.Tensor
    source_depth: torch.Tensor
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
class PerPixelGaussianObservation:
    """Canonical dense Gaussian predictions with immutable pixel provenance."""

    initial_depth: torch.Tensor
    depth_residual: torch.Tensor
    refined_depth: torch.Tensor
    poses_c2w: torch.Tensor
    local_quaternion: torch.Tensor
    log_scale_multiplier: torch.Tensor
    rgb_sh: torch.Tensor
    density_sh: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor
    source_uv: torch.Tensor
    source_ray: torch.Tensor
    frame_ids: torch.Tensor
    rgb_sh_degree: int = 2
    density_sh_degree: int = 1
    min_scale: float = 1.0e-5
    max_scale_ratio: float = 0.25
    latitude_cos_min: float = 1.0e-3
    log_scale_clamp: float = 5.0
    render_prune_fraction: float = 0.30
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.initial_depth.ndim != 5 or int(self.initial_depth.shape[2]) != 1:
            raise ValueError("initial_depth must have shape BxSx1xHxW.")
        b, s, _, h, w = (int(value) for value in self.initial_depth.shape)
        expected_single = (b, s, 1, h, w)
        for name in ("depth_residual", "refined_depth", "confidence", "valid_mask"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != expected_single:
                raise ValueError(f"{name} must have shape {expected_single}, got {tuple(tensor.shape)}.")
        if tuple(self.poses_c2w.shape) != (b, s, 4, 4):
            raise ValueError(f"poses_c2w must have shape {(b, s, 4, 4)}, got {tuple(self.poses_c2w.shape)}.")
        if tuple(self.local_quaternion.shape) != (b, s, 4, h, w):
            raise ValueError("local_quaternion must have shape BxSx4xHxW.")
        if tuple(self.log_scale_multiplier.shape) != (b, s, 3, h, w):
            raise ValueError("log_scale_multiplier must have shape BxSx3xHxW.")
        rgb_count = (int(self.rgb_sh_degree) + 1) ** 2
        density_count = (int(self.density_sh_degree) + 1) ** 2
        if tuple(self.rgb_sh.shape) != (b, s, rgb_count, 3, h, w):
            raise ValueError(
                f"rgb_sh must have shape {(b, s, rgb_count, 3, h, w)}, got {tuple(self.rgb_sh.shape)}."
            )
        if tuple(self.density_sh.shape) != (b, s, density_count, h, w):
            raise ValueError(
                f"density_sh must have shape {(b, s, density_count, h, w)}, got {tuple(self.density_sh.shape)}."
            )
        if tuple(self.source_uv.shape) != (h, w, 2) or tuple(self.source_ray.shape) != (h, w, 3):
            raise ValueError("source_uv/source_ray must have shapes HxWx2 and HxWx3.")
        if tuple(self.frame_ids.shape) != (b, s):
            raise ValueError(f"frame_ids must have shape {(b, s)}, got {tuple(self.frame_ids.shape)}.")
        if not 0.0 <= float(self.render_prune_fraction) < 1.0:
            raise ValueError("render_prune_fraction must be in [0, 1).")

    @property
    def batch_size(self) -> int:
        return int(self.initial_depth.shape[0])

    @property
    def num_source_views(self) -> int:
        return int(self.initial_depth.shape[1])

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.initial_depth.shape[-2]), int(self.initial_depth.shape[-1])

    @property
    def canonical_count(self) -> int:
        return int(self.valid_mask.sum().detach().cpu())

    def centers_camera(self, depth: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``B x S x H x W x 3`` camera-local centers."""

        selected_depth = self.refined_depth if depth is None else depth
        if tuple(selected_depth.shape) != tuple(self.refined_depth.shape):
            raise ValueError("Replacement depth must match refined_depth shape.")
        ray = self.source_ray.to(device=selected_depth.device, dtype=selected_depth.dtype)
        return selected_depth.squeeze(2).unsqueeze(-1) * ray.view(1, 1, *ray.shape)

    def centers_world(
        self,
        depth: torch.Tensor | None = None,
        poses_c2w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``B x S x H x W x 3`` world centers."""

        poses = self.poses_c2w if poses_c2w is None else poses_c2w
        if tuple(poses.shape) != tuple(self.poses_c2w.shape):
            raise ValueError("Replacement poses must match poses_c2w shape.")
        camera = self.centers_camera(depth)
        rotation = poses[..., :3, :3].to(device=camera.device, dtype=camera.dtype)
        translation = poses[..., :3, 3].to(device=camera.device, dtype=camera.dtype)
        return torch.einsum("bsij,bshwj->bshwi", rotation, camera) + translation[:, :, None, None, :]

    def scales(self, depth: torch.Tensor | None = None) -> torch.Tensor:
        """Return positive world scales as ``B x S x 3 x H x W``."""

        selected_depth = self.refined_depth if depth is None else depth
        if tuple(selected_depth.shape) != tuple(self.refined_depth.shape):
            raise ValueError("Replacement depth must match refined_depth shape.")
        height, width = self.image_size
        rows = torch.arange(height, device=selected_depth.device, dtype=selected_depth.dtype) + 0.5
        latitude = math.pi * (rows / float(height) - 0.5)
        cos_latitude = torch.cos(latitude).clamp_min(float(self.latitude_cos_min))
        angular_area = (2.0 * math.pi / float(width)) * (math.pi / float(height))
        footprint = torch.sqrt(cos_latitude * angular_area).view(1, 1, 1, height, 1)
        base = selected_depth * footprint
        multiplier = torch.exp(
            self.log_scale_multiplier.clamp(-float(self.log_scale_clamp), float(self.log_scale_clamp))
        )
        scale = base * multiplier
        max_scale = selected_depth * float(self.max_scale_ratio)
        return torch.minimum(scale.clamp_min(float(self.min_scale)), max_scale.clamp_min(float(self.min_scale)))

    def with_geometry(
        self,
        *,
        poses_c2w: torch.Tensor | None = None,
        refined_depth: torch.Tensor | None = None,
    ) -> "PerPixelGaussianObservation":
        """Return a geometry-updated view for the future spherical BA stage."""

        poses = self.poses_c2w if poses_c2w is None else poses_c2w
        depth = self.refined_depth if refined_depth is None else refined_depth
        if tuple(poses.shape) != tuple(self.poses_c2w.shape):
            raise ValueError("Updated poses must match poses_c2w shape.")
        if tuple(depth.shape) != tuple(self.refined_depth.shape):
            raise ValueError("Updated depth must match refined_depth shape.")
        return replace(
            self,
            poses_c2w=poses,
            refined_depth=depth,
            depth_residual=depth - self.initial_depth.to(device=depth.device, dtype=depth.dtype),
        )

    def source_view_confidence(self, density_sh: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate density SH in each Gaussian's immutable source-ray direction."""

        coefficients = self.density_sh if density_sh is None else density_sh
        if tuple(coefficients.shape) != tuple(self.density_sh.shape):
            raise ValueError("Replacement density_sh must match density_sh shape.")
        ray = self.source_ray.to(device=coefficients.device, dtype=coefficients.dtype)
        basis = real_sh_basis(self.density_sh_degree, ray).permute(2, 0, 1)
        logits = (coefficients * basis.view(1, 1, *basis.shape)).sum(dim=2, keepdim=True)
        return torch.sigmoid(logits)

    def with_updates(
        self,
        *,
        poses_c2w: torch.Tensor | None = None,
        refined_depth: torch.Tensor | None = None,
        local_quaternion: torch.Tensor | None = None,
        log_scale_multiplier: torch.Tensor | None = None,
        rgb_sh: torch.Tensor | None = None,
        density_sh: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> "PerPixelGaussianObservation":
        """Return a parameter-updated observation while preserving provenance.

        Stage 3 uses this functional update instead of mutating canonical
        observations.  Pixel coordinates, rays, frame ids, and initial depth
        remain immutable so later BA passes can always rebuild geometry.
        """

        poses = self.poses_c2w if poses_c2w is None else poses_c2w
        depth = self.refined_depth if refined_depth is None else refined_depth
        quaternion = self.local_quaternion if local_quaternion is None else local_quaternion
        log_scale = self.log_scale_multiplier if log_scale_multiplier is None else log_scale_multiplier
        rgb = self.rgb_sh if rgb_sh is None else rgb_sh
        density = self.density_sh if density_sh is None else density_sh
        mask = self.valid_mask if valid_mask is None else valid_mask
        confidence = self.confidence if density_sh is None else self.source_view_confidence(density)
        return replace(
            self,
            poses_c2w=poses,
            refined_depth=depth,
            depth_residual=depth - self.initial_depth.to(device=depth.device, dtype=depth.dtype),
            local_quaternion=normalize_quaternion(quaternion.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3),
            log_scale_multiplier=log_scale,
            rgb_sh=rgb,
            density_sh=density,
            confidence=confidence,
            valid_mask=mask,
        )

    def detach_parameters(self) -> "PerPixelGaussianObservation":
        """Detach the mutable Stage 3 state while retaining canonical metadata."""

        return replace(
            self,
            depth_residual=self.depth_residual.detach(),
            refined_depth=self.refined_depth.detach(),
            poses_c2w=self.poses_c2w.detach(),
            local_quaternion=self.local_quaternion.detach(),
            log_scale_multiplier=self.log_scale_multiplier.detach(),
            rgb_sh=self.rgb_sh.detach(),
            density_sh=self.density_sh.detach(),
            confidence=self.confidence.detach(),
            valid_mask=self.valid_mask.detach(),
        )

    def materialize(self, camera: Any) -> ExplicitPerPixelGaussianSet:
        """Renderer hook for the common ``B=1`` training/inference case."""

        if self.batch_size != 1:
            raise ValueError("materialize(camera) requires batch_size=1; use materialize_batch otherwise.")
        return self.materialize_batch(camera, batch_index=0)

    def materialize_batch(
        self,
        camera: Any,
        *,
        batch_index: int,
        source_indices: Iterable[int] | torch.Tensor | None = None,
        prune_fraction: float | None = None,
    ) -> ExplicitPerPixelGaussianSet:
        """Materialize selected source views for one target render camera."""

        batch = int(batch_index)
        if batch < 0 or batch >= self.batch_size:
            raise IndexError(f"batch_index {batch} is outside [0, {self.batch_size}).")
        device = self.refined_depth.device
        dtype = (
            torch.float32
            if self.refined_depth.dtype in {torch.float16, torch.bfloat16}
            else self.refined_depth.dtype
        )
        if source_indices is None:
            source_index = torch.arange(self.num_source_views, device=device, dtype=torch.long)
        else:
            source_index = torch.as_tensor(list(source_indices) if not torch.is_tensor(source_indices) else source_indices, device=device, dtype=torch.long).view(-1)
        if source_index.numel() == 0:
            return self._empty_materialized(device=device, dtype=dtype)
        if int(source_index.min()) < 0 or int(source_index.max()) >= self.num_source_views:
            raise IndexError("source_indices contain values outside the source-view range.")

        centers = self.centers_world()[batch].index_select(0, source_index).to(dtype=dtype)
        scale = self.scales()[batch].index_select(0, source_index).permute(0, 2, 3, 1).to(dtype=dtype)
        local_quaternion = (
            self.local_quaternion[batch].index_select(0, source_index).permute(0, 2, 3, 1).to(dtype=dtype)
        )
        poses = self.poses_c2w[batch].index_select(0, source_index).to(device=device, dtype=dtype)
        pose_quaternion = matrix_to_quaternion(poses[:, :3, :3]).view(-1, 1, 1, 4)
        world_quaternion = normalize_quaternion(quaternion_multiply(pose_quaternion, local_quaternion))

        target_pose = camera.c2w.to(device=device, dtype=dtype)
        target_center = target_pose[:3, 3]
        direction_world = F.normalize(centers - target_center.view(1, 1, 1, 3), dim=-1, eps=1.0e-8)
        rotation_c2w = poses[:, :3, :3]
        direction_local = torch.einsum("sij,shwi->shwj", rotation_c2w, direction_world)

        rgb_coefficients = (
            self.rgb_sh[batch].index_select(0, source_index).permute(0, 3, 4, 1, 2).to(dtype=dtype)
        )
        density_coefficients = (
            self.density_sh[batch].index_select(0, source_index).permute(0, 2, 3, 1).to(dtype=dtype)
        )
        rgb_basis = real_sh_basis(self.rgb_sh_degree, direction_local)
        density_basis = real_sh_basis(self.density_sh_degree, direction_local)
        rgb = (0.5 + (rgb_basis.unsqueeze(-1) * rgb_coefficients).sum(dim=-2)).clamp(0.0, 1.0)
        opacity = torch.sigmoid((density_basis * density_coefficients).sum(dim=-1, keepdim=True))

        valid = self.valid_mask[batch].index_select(0, source_index)[:, 0].bool()
        confidence = self.confidence[batch].index_select(0, source_index)[:, 0].to(dtype=dtype)
        source_depth = self.refined_depth[batch].index_select(0, source_index)[:, 0].to(dtype=dtype)
        frame_index = self.frame_ids[batch].index_select(0, source_index)
        height, width = self.image_size
        uv = self.source_uv.to(device=device, dtype=dtype).view(1, height, width, 2).expand(source_index.numel(), -1, -1, -1)
        ray = self.source_ray.to(device=device, dtype=dtype).view(1, height, width, 3).expand(source_index.numel(), -1, -1, -1)
        frame_map = frame_index.view(-1, 1, 1).expand(-1, height, width)

        mask = valid.reshape(-1)
        xyz = centers.reshape(-1, 3)[mask]
        scaling = scale.reshape(-1, 3)[mask]
        rotation = world_quaternion.reshape(-1, 4)[mask]
        opacity_flat = opacity.reshape(-1, 1)[mask]
        rgb_flat = rgb.reshape(-1, 3)[mask]
        confidence_flat = confidence.reshape(-1, 1)[mask]
        source_frame_flat = frame_map.reshape(-1)[mask]
        uv_flat = uv.reshape(-1, 2)[mask]
        ray_flat = ray.reshape(-1, 3)[mask]
        depth_flat = source_depth.reshape(-1, 1)[mask]

        fraction = self.render_prune_fraction if prune_fraction is None else float(prune_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("prune_fraction must be in [0, 1).")
        if fraction > 0.0 and int(opacity_flat.shape[0]) > 1:
            keep_count = max(1, int(math.ceil(float(opacity_flat.shape[0]) * (1.0 - fraction))))
            keep = torch.topk(opacity_flat[:, 0], k=keep_count, largest=True, sorted=False).indices.sort().values
            xyz = xyz.index_select(0, keep)
            scaling = scaling.index_select(0, keep)
            rotation = rotation.index_select(0, keep)
            opacity_flat = opacity_flat.index_select(0, keep)
            rgb_flat = rgb_flat.index_select(0, keep)
            confidence_flat = confidence_flat.index_select(0, keep)
            source_frame_flat = source_frame_flat.index_select(0, keep)
            uv_flat = uv_flat.index_select(0, keep)
            ray_flat = ray_flat.index_select(0, keep)
            depth_flat = depth_flat.index_select(0, keep)

        return ExplicitPerPixelGaussianSet(
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity_flat,
            features=rgb_flat,
            confidence=confidence_flat,
            source_frame_index=source_frame_flat,
            source_pixel_uv=uv_flat,
            source_ray=ray_flat,
            source_depth=depth_flat,
            config=self.config,
        )

    def _empty_materialized(self, *, device: torch.device, dtype: torch.dtype) -> ExplicitPerPixelGaussianSet:
        return ExplicitPerPixelGaussianSet(
            xyz=torch.zeros(0, 3, device=device, dtype=dtype),
            scaling=torch.zeros(0, 3, device=device, dtype=dtype),
            rotation=torch.zeros(0, 4, device=device, dtype=dtype),
            opacity=torch.zeros(0, 1, device=device, dtype=dtype),
            features=torch.zeros(0, 3, device=device, dtype=dtype),
            confidence=torch.zeros(0, 1, device=device, dtype=dtype),
            source_frame_index=torch.zeros(0, device=device, dtype=torch.long),
            source_pixel_uv=torch.zeros(0, 2, device=device, dtype=dtype),
            source_ray=torch.zeros(0, 3, device=device, dtype=dtype),
            source_depth=torch.zeros(0, 1, device=device, dtype=dtype),
            config=self.config,
        )
