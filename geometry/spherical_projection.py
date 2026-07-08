"""Projection utilities for Stage 1A spherical pseudo correspondence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .spherical_erp import (
    DEFAULT_ERP_HEIGHT,
    DEFAULT_ERP_WIDTH,
    erp_pixel_to_unit_ray,
    unit_ray_to_erp_pixel,
)


@dataclass
class SphericalProjectionResult:
    """Projection result from a source ERP sample into a target ERP camera."""

    target_uv: torch.Tensor
    target_ray: torch.Tensor
    target_range: torch.Tensor
    source_ray: torch.Tensor
    world_points: torch.Tensor
    target_camera_points: torch.Tensor


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    value = depth.float() if not depth.is_floating_point() else depth
    if value.shape[-1:] == (1,):
        value = value.squeeze(-1)
    return value


def _project_single_or_batched(
    points_cam: torch.Tensor,
    src_c2w: torch.Tensor,
    tgt_c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src_c2w.shape[-2:] != (4, 4) or tgt_c2w.shape[-2:] != (4, 4):
        raise ValueError("src_c2w and tgt_c2w must end with shape 4x4.")
    src_pose = src_c2w.to(device=points_cam.device, dtype=points_cam.dtype)
    tgt_pose = tgt_c2w.to(device=points_cam.device, dtype=points_cam.dtype)
    if src_pose.ndim == 2:
        world = points_cam @ src_pose[:3, :3].transpose(0, 1) + src_pose[:3, 3]
        target = (world - tgt_pose[:3, 3]) @ tgt_pose[:3, :3]
        return world, target
    if points_cam.ndim != 3 or src_pose.ndim != 3 or tgt_pose.ndim != 3:
        raise ValueError(
            "Batched projection expects points BxSx3 and poses Bx4x4; "
            f"got points={tuple(points_cam.shape)}, src={tuple(src_pose.shape)}, tgt={tuple(tgt_pose.shape)}."
        )
    world = torch.einsum("bij,bsj->bsi", src_pose[:, :3, :3], points_cam) + src_pose[:, :3, 3].unsqueeze(1)
    target = torch.einsum(
        "bij,bsj->bsi",
        tgt_pose[:, :3, :3].transpose(-1, -2),
        world - tgt_pose[:, :3, 3].unsqueeze(1),
    )
    return world, target


def project_source_to_target_erp(
    src_uv: torch.Tensor,
    src_depth: torch.Tensor,
    src_c2w: torch.Tensor,
    tgt_c2w: torch.Tensor,
    *,
    height: int = DEFAULT_ERP_HEIGHT,
    width: int = DEFAULT_ERP_WIDTH,
    eps: float = 1.0e-12,
) -> SphericalProjectionResult:
    """Project source ERP pixels with ray depth into a target ERP camera.

    ``src_depth`` is interpreted as Euclidean range along the source unit ray.
    This function does not infer or convert z-depth.
    """

    if src_uv.shape[-1] != 2:
        raise ValueError(f"src_uv must end with dimension 2, got {tuple(src_uv.shape)}.")
    depth = _normalize_depth(src_depth).to(device=src_uv.device)
    src_ray = erp_pixel_to_unit_ray(src_uv, height, width).to(device=src_uv.device, dtype=src_uv.dtype)
    depth = depth.to(dtype=src_ray.dtype)
    if depth.shape != src_ray.shape[:-1]:
        try:
            depth = torch.broadcast_to(depth, src_ray.shape[:-1])
        except RuntimeError as exc:
            raise ValueError(
                f"src_depth shape {tuple(src_depth.shape)} cannot broadcast to src_uv shape {tuple(src_uv.shape[:-1])}."
            ) from exc
    points_cam = src_ray * depth.clamp_min(eps).unsqueeze(-1)
    world_points, target_camera_points = _project_single_or_batched(points_cam, src_c2w, tgt_c2w)
    target_range = torch.linalg.norm(target_camera_points, dim=-1)
    target_ray = F.normalize(target_camera_points, dim=-1, eps=eps)
    target_uv = unit_ray_to_erp_pixel(target_ray, height, width)
    return SphericalProjectionResult(
        target_uv=target_uv,
        target_ray=target_ray,
        target_range=target_range,
        source_ray=src_ray,
        world_points=world_points,
        target_camera_points=target_camera_points,
    )
