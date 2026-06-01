"""Shared ERP projective geometry for PanoDROID graph optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    pixel_grid,
    seam_aware_delta,
    spherical_log_residual,
)


def grid_sample_from_pixel_centers(
    image: torch.Tensor,
    pixels: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample ``image`` at ERP pixel-center coordinates."""
    norm_x = 2.0 * (pixels[..., 0] - 0.5) / max(width - 1, 1) - 1.0
    norm_y = 2.0 * (pixels[..., 1] - 0.5) / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def project_edges(
    poses_c2w: torch.Tensor,
    inverse_depth: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    *,
    height: int,
    width: int,
    pixels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project source-frame ERP pixels from edge ``ii`` into target frame ``jj``."""
    B = poses_c2w.shape[0]
    E = int(ii.numel())
    full_grid = pixels is None
    if pixels is None:
        pixels = pixel_grid(height, width, device=poses_c2w.device, dtype=poses_c2w.dtype)
    else:
        pixels = pixels.to(device=poses_c2w.device, dtype=poses_c2w.dtype)
    Hs, Ws = int(pixels.shape[0]), int(pixels.shape[1])

    if full_grid and Hs == height and Ws == width:
        inv_src = inverse_depth[:, ii, 0]
    else:
        inv_map = inverse_depth[:, ii].reshape(B * E, 1, height, width)
        sample_pixels = pixels.view(1, Hs, Ws, 2).expand(B * E, -1, -1, -1)
        inv_src = grid_sample_from_pixel_centers(
            inv_map,
            sample_pixels,
            height=height,
            width=width,
        ).view(B, E, Hs, Ws)

    p = pixels.view(1, 1, Hs, Ws, 2).expand(B, E, -1, -1, -1)
    bearing_i = erp_pixel_to_bearing(p.reshape(B * E, Hs * Ws, 2), height, width)
    bearing_i = bearing_i.view(B, E, Hs, Ws, 3)
    xyz_i = bearing_i / inv_src.clamp_min(1e-6).unsqueeze(-1)

    T_ji = torch.linalg.inv(poses_c2w[:, jj]) @ poses_c2w[:, ii]
    xyz_j = torch.einsum("beij,behwj->behwi", T_ji[..., :3, :3], xyz_i) + T_ji[
        ..., :3, 3
    ].view(B, E, 1, 1, 3)
    bearing_j = F.normalize(xyz_j, dim=-1, eps=1e-12)
    target_pixels = bearing_to_erp_pixel(bearing_j.reshape(B * E, Hs * Ws, 3), height, width)
    return target_pixels.view(B, E, Hs, Ws, 2)


def projective_flow_from_depth(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    c2w_i: torch.Tensor,
    c2w_j: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project sparse source pixels/depth and return wrapped flow plus target pixels."""
    B, P = depth.shape
    p = pixels.to(device=depth.device, dtype=depth.dtype).unsqueeze(0).expand(B, -1, -1)
    bearing_i = erp_pixel_to_bearing(p, height, width)
    xyz_i = bearing_i * depth.clamp_min(1e-6).unsqueeze(-1)
    ones = torch.ones(B, P, 1, device=depth.device, dtype=depth.dtype)
    world = torch.einsum("bij,bnj->bni", c2w_i, torch.cat([xyz_i, ones], dim=-1))[..., :3]
    cam_j = torch.einsum(
        "bij,bnj->bni",
        torch.linalg.inv(c2w_j),
        torch.cat([world, ones], dim=-1),
    )[..., :3]
    bearing_j = F.normalize(cam_j, dim=-1, eps=1e-12)
    target_pixels = bearing_to_erp_pixel(bearing_j, height, width)
    return seam_aware_delta(p, target_pixels, width), target_pixels


def spherical_reprojection_residual(
    source_pixels: torch.Tensor,
    inverse_depth: torch.Tensor,
    T_ji: torch.Tensor,
    *,
    height: int,
    width: int,
    target_delta: Optional[torch.Tensor] = None,
    target_pixels: Optional[torch.Tensor] = None,
    target_bearing: Optional[torch.Tensor] = None,
    residual_mode: str = "pixel",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, predicted pixels, and target pixels/bearings for one edge batch."""
    if source_pixels.ndim == 2:
        pixels_b = source_pixels.unsqueeze(0)
    elif source_pixels.ndim == 3:
        pixels_b = source_pixels
    else:
        raise ValueError(f"Expected pixels as Nx2 or BxNx2, got {tuple(source_pixels.shape)}")

    B, P, _ = pixels_b.shape
    inv = inverse_depth.reshape(B, P).to(device=pixels_b.device, dtype=pixels_b.dtype)
    T = T_ji.to(device=pixels_b.device, dtype=pixels_b.dtype)
    if T.ndim == 2:
        T = T.unsqueeze(0).expand(B, -1, -1)

    b_i = erp_pixel_to_bearing(pixels_b, height, width)
    xyz_i = b_i / inv.clamp_min(1e-6).unsqueeze(-1)
    xyz_j = torch.einsum("bij,bnj->bni", T[..., :3, :3], xyz_i) + T[..., :3, 3].unsqueeze(-2)
    b_pred = F.normalize(xyz_j, dim=-1, eps=1e-12)
    pred_pixels = bearing_to_erp_pixel(b_pred, height, width)

    if target_bearing is None:
        if target_pixels is None:
            if target_delta is None:
                raise ValueError("Provide target_delta, target_pixels, or target_bearing.")
            delta = target_delta
            if delta.ndim == 3 and delta.shape[1] == 2:
                delta = delta.permute(0, 2, 1)
            delta = delta.reshape(B, P, 2).to(device=pixels_b.device, dtype=pixels_b.dtype)
            target_pixels = pixels_b + delta
        else:
            target_pixels = target_pixels.reshape(B, P, 2).to(
                device=pixels_b.device, dtype=pixels_b.dtype
            )
        target_pixels = target_pixels.clone()
        target_pixels[..., 0] = torch.remainder(target_pixels[..., 0], float(width))
        target_bearing = erp_pixel_to_bearing(target_pixels, height, width)
    else:
        target_bearing = target_bearing.reshape(B, P, 3).to(
            device=pixels_b.device, dtype=pixels_b.dtype
        )
        target_pixels = bearing_to_erp_pixel(target_bearing, height, width)

    mode = str(residual_mode).lower()
    if mode in ("pixel", "erp_pixel", "feature_pixel"):
        residual = seam_aware_delta(pred_pixels, target_pixels, width)
    elif mode in ("tangent", "angular", "sphere"):
        residual = spherical_log_residual(target_bearing, b_pred)
    else:
        raise ValueError(f"Unsupported residual_mode: {residual_mode}")
    return residual, pred_pixels, target_pixels
