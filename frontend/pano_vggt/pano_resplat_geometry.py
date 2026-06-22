"""ERP geometry helpers for Pano-ReSplat feed-forward Gaussian prediction."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    pixel_grid,
    wrap_horizontal,
)


class ERPProjection(NamedTuple):
    uv: torch.Tensor
    grid: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    bearing: torch.Tensor
    camera_points: torch.Tensor


def erp_pixel_grid(
    image_hw: tuple[int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ERP pixel-center coordinates with shape ``H x W x 2``."""

    height, width = int(image_hw[0]), int(image_hw[1])
    return pixel_grid(height, width, device=device, dtype=dtype)


def erp_uv_to_bearing(uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Convert ERP pixel coordinates to unit camera bearings."""

    height, width = int(image_hw[0]), int(image_hw[1])
    return erp_pixel_to_bearing(uv, height, width).to(device=uv.device, dtype=uv.dtype)


def world_to_camera(world_points: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Transform world points into camera coordinates with a c2w pose."""

    if world_points.shape[-1] != 3:
        raise ValueError(f"world_points must have last dim 3, got {tuple(world_points.shape)}")
    if tuple(c2w.shape[-2:]) != (4, 4):
        raise ValueError(f"c2w must end with 4x4, got {tuple(c2w.shape)}")
    dtype = world_points.dtype
    device = world_points.device
    pose = c2w.to(device=device, dtype=dtype)
    rot = pose[..., :3, :3]
    trans = pose[..., :3, 3]
    while rot.ndim < world_points.ndim + 1:
        rot = rot.unsqueeze(-3)
        trans = trans.unsqueeze(-2)
    return torch.matmul(rot.transpose(-1, -2), (world_points - trans).unsqueeze(-1)).squeeze(-1)


def camera_to_erp_uv(
    camera_points: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    min_depth: float = 1.0e-6,
    require_forward: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-space points to ERP pixel coordinates.

    Returns ``(uv, radial_depth, valid_mask, bearing)``.  Horizontal coordinates
    are wrapped into ``[0, W)``.  When ``require_forward`` is true, points with
    non-positive forward ``z`` are marked invalid.
    """

    if camera_points.shape[-1] != 3:
        raise ValueError(f"camera_points must have last dim 3, got {tuple(camera_points.shape)}")
    height, width = int(image_hw[0]), int(image_hw[1])
    finite = torch.isfinite(camera_points).all(dim=-1)
    depth = torch.linalg.norm(camera_points, dim=-1).clamp_min(float(min_depth))
    bearing = F.normalize(torch.nan_to_num(camera_points, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=float(min_depth))
    uv = bearing_to_erp_pixel(bearing, height, width, wrap=True).to(device=camera_points.device, dtype=camera_points.dtype)
    uv = torch.stack([wrap_horizontal(uv[..., 0], width), uv[..., 1].clamp(0.0, max(float(height - 1), 0.0))], dim=-1)
    valid = finite & (depth > float(min_depth))
    if require_forward:
        valid = valid & (camera_points[..., 2] > float(min_depth))
    return uv, depth, valid, bearing


def project_world_to_erp_grid(
    world_points: torch.Tensor,
    c2w: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    min_depth: float = 1.0e-6,
    require_forward: bool = True,
    align_corners: bool = True,
) -> ERPProjection:
    """Project world points to ERP ``grid_sample`` coordinates and validity."""

    height, width = int(image_hw[0]), int(image_hw[1])
    cam = world_to_camera(world_points, c2w)
    uv, depth, mask, bearing = camera_to_erp_uv(
        cam,
        (height, width),
        min_depth=min_depth,
        require_forward=require_forward,
    )
    if align_corners:
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
    else:
        denom_x = max(width, 1)
        denom_y = max(height, 1)
    grid_x = 2.0 * uv[..., 0] / float(denom_x) - 1.0
    grid_y = 2.0 * uv[..., 1] / float(denom_y) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).clamp(-1.0, 1.0)
    return ERPProjection(uv=uv, grid=grid, depth=depth, mask=mask, bearing=bearing, camera_points=cam)
