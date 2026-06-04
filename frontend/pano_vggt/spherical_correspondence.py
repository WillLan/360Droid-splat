"""Spherical correspondence helpers for PanoVGGT-M3-Sphere contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    spherical_log_residual,
)

from .grid_utils import (
    feature_uv_to_image_uv,
    image_uv_to_feature_uv,
    make_feature_grid,
    normalize_hw,
)


@dataclass
class SphericalCorrespondenceBatch:
    """Dense spherical correspondences for a batch of frame-pair edges."""

    src_indices: torch.Tensor
    tgt_indices: torch.Tensor
    src_uv: torch.Tensor
    tgt_uv: torch.Tensor
    src_bearing: torch.Tensor
    tgt_bearing: torch.Tensor
    valid_mask: torch.Tensor
    depth_consistency: torch.Tensor
    angular_baseline: torch.Tensor
    latitude_weight: torch.Tensor
    metadata: dict[str, Any]


def spherical_tangent_residual(target_bearing: torch.Tensor, predicted_bearing: torch.Tensor) -> torch.Tensor:
    """Return ``Log_target_bearing(predicted_bearing)`` as an ``R^2`` residual."""

    return spherical_log_residual(target_bearing, predicted_bearing)


def _normalize_depths(depths: torch.Tensor) -> torch.Tensor:
    if depths.ndim == 3:
        depths = depths.unsqueeze(1)
    if depths.ndim != 4 or depths.shape[1] != 1:
        raise ValueError(f"depths must have shape Nx1xHxW or NxHxW, got {tuple(depths.shape)}")
    return depths.float()


def _normalize_pairs(pair_indices: torch.Tensor | list[tuple[int, int]], *, device: torch.device) -> torch.Tensor:
    pairs = torch.as_tensor(pair_indices, device=device, dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pair_indices must have shape Ex2, got {tuple(pairs.shape)}")
    return pairs


def _sample_indices(total: int, samples_per_edge: int | None, *, device: torch.device) -> torch.Tensor:
    if samples_per_edge is None or int(samples_per_edge) >= total:
        return torch.arange(total, device=device)
    count = max(1, int(samples_per_edge))
    step = max(1, math.ceil(float(total) / float(count)))
    return torch.arange(0, total, step, device=device)[:count]


def _sample_scalar_map(maps: torch.Tensor, uv: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if maps.ndim != 4 or maps.shape[1] != 1:
        raise ValueError(f"maps must have shape Bx1xHxW, got {tuple(maps.shape)}")
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError(f"uv must have shape BxSx2, got {tuple(uv.shape)}")
    norm_x = 2.0 * (uv[..., 0] - 0.5) / max(width - 1, 1) - 1.0
    norm_y = 2.0 * (uv[..., 1] - 0.5) / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        maps,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, :, 0]


def generate_gt_spherical_correspondences(
    depths: torch.Tensor,
    poses_c2w: torch.Tensor,
    pair_indices: torch.Tensor | list[tuple[int, int]],
    feature_hw: tuple[int, int],
    image_hw: tuple[int, int],
    samples_per_edge: int | None = None,
    depth_consistency_rel: float = 0.03,
    depth_consistency_abs: float = 0.05,
    min_baseline_deg: float = 0.05,
    max_baseline_deg: float = 60.0,
    use_wraparound: bool = True,
) -> SphericalCorrespondenceBatch:
    """Generate GT spherical correspondences from depth and camera poses.

    Coordinates are feature-grid UV pixel centers unless explicitly named
    ``image`` internally. The residual target is the target bearing on ``S^2``;
    ERP pixel coordinates are used only to sample depth and report target UV.
    """

    depth_t = _normalize_depths(depths)
    device = depth_t.device
    dtype = depth_t.dtype
    poses = poses_c2w.to(device=device, dtype=dtype)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses_c2w must have shape Nx4x4, got {tuple(poses.shape)}")

    feature_size = normalize_hw(feature_hw, name="feature_hw")
    image_size = normalize_hw(image_hw, name="image_hw")
    height, width = int(image_size[0]), int(image_size[1])
    if tuple(depth_t.shape[-2:]) != image_size:
        raise ValueError(f"depths image size {tuple(depth_t.shape[-2:])} does not match image_hw {image_size}")

    pairs = _normalize_pairs(pair_indices, device=device)
    if pairs.numel() == 0:
        raise ValueError("pair_indices must contain at least one edge.")
    if int(pairs.min()) < 0 or int(pairs.max()) >= int(depth_t.shape[0]):
        raise ValueError("pair_indices contain frame indices outside the depth/pose batch.")

    full_feature_grid = make_feature_grid(feature_size, device=device, dtype=dtype).view(-1, 2)
    selected = _sample_indices(full_feature_grid.shape[0], samples_per_edge, device=device)
    src_feature_uv_base = full_feature_grid[selected]

    edge_count = int(pairs.shape[0])
    sample_count = int(src_feature_uv_base.shape[0])
    src_feature_uv = src_feature_uv_base.view(1, sample_count, 2).expand(edge_count, -1, -1)
    src_image_uv = feature_uv_to_image_uv(src_feature_uv, feature_size, image_size)

    src_idx = pairs[:, 0]
    tgt_idx = pairs[:, 1]
    src_depth = _sample_scalar_map(depth_t[src_idx], src_image_uv, height=height, width=width)
    src_bearing = erp_pixel_to_bearing(src_image_uv, height, width).to(device=device, dtype=dtype)
    src_cam = src_bearing * src_depth.clamp_min(1.0e-6).unsqueeze(-1)

    src_pose = poses[src_idx]
    tgt_pose = poses[tgt_idx]
    world = torch.einsum("eij,esj->esi", src_pose[:, :3, :3], src_cam) + src_pose[:, :3, 3].view(edge_count, 1, 3)
    tgt_cam = torch.einsum(
        "eij,esj->esi",
        tgt_pose[:, :3, :3].transpose(-1, -2),
        world - tgt_pose[:, :3, 3].view(edge_count, 1, 3),
    )
    projected_tgt_depth = torch.linalg.norm(tgt_cam, dim=-1)
    tgt_bearing = torch.nn.functional.normalize(tgt_cam, dim=-1, eps=1.0e-12)

    tgt_image_uv = bearing_to_erp_pixel(tgt_bearing, height, width, wrap=use_wraparound).to(dtype=dtype)
    if use_wraparound:
        tgt_image_uv = tgt_image_uv.clone()
        tgt_image_uv[..., 0] = torch.remainder(tgt_image_uv[..., 0], float(width))
    tgt_feature_uv = image_uv_to_feature_uv(tgt_image_uv, feature_size, image_size)
    if use_wraparound:
        tgt_feature_uv = tgt_feature_uv.clone()
        tgt_feature_uv[..., 0] = torch.remainder(tgt_feature_uv[..., 0], float(feature_size[1]))

    tgt_depth = _sample_scalar_map(depth_t[tgt_idx], tgt_image_uv, height=height, width=width)
    depth_error = (tgt_depth - projected_tgt_depth).abs()
    depth_ok = (
        torch.isfinite(tgt_depth)
        & torch.isfinite(projected_tgt_depth)
        & (tgt_depth > 0.0)
        & (depth_error <= float(depth_consistency_abs) + float(depth_consistency_rel) * tgt_depth.abs().clamp_min(1.0e-6))
    )

    src_center = src_pose[:, :3, 3].view(edge_count, 1, 3)
    tgt_center = tgt_pose[:, :3, 3].view(edge_count, 1, 3)
    src_ray_world = F.normalize(world - src_center, dim=-1, eps=1.0e-12)
    tgt_ray_world = F.normalize(world - tgt_center, dim=-1, eps=1.0e-12)
    dot = (src_ray_world * tgt_ray_world).sum(dim=-1).clamp(-1.0, 1.0)
    angular_baseline = torch.acos(dot)
    baseline_deg = torch.rad2deg(angular_baseline)
    baseline_ok = (baseline_deg >= float(min_baseline_deg)) & (baseline_deg <= float(max_baseline_deg))

    horizontal_ok = torch.ones_like(baseline_ok, dtype=torch.bool)
    if not use_wraparound:
        horizontal_ok = (tgt_feature_uv[..., 0] >= 0.0) & (tgt_feature_uv[..., 0] < float(feature_size[1]))
    vertical_ok = (tgt_feature_uv[..., 1] >= 0.0) & (tgt_feature_uv[..., 1] < float(feature_size[0]))
    src_ok = torch.isfinite(src_depth) & (src_depth > 0.0)
    finite_ok = (
        torch.isfinite(src_feature_uv).all(dim=-1)
        & torch.isfinite(tgt_feature_uv).all(dim=-1)
        & torch.isfinite(src_bearing).all(dim=-1)
        & torch.isfinite(tgt_bearing).all(dim=-1)
    )
    valid_mask = src_ok & depth_ok & baseline_ok & horizontal_ok & vertical_ok & finite_ok

    phi = math.pi * (src_image_uv[..., 1] / float(height) - 0.5)
    latitude_weight = torch.cos(phi).clamp_min(0.0)

    src_indices = src_idx.view(edge_count, 1).expand(edge_count, sample_count)
    tgt_indices = tgt_idx.view(edge_count, 1).expand(edge_count, sample_count)
    return SphericalCorrespondenceBatch(
        src_indices=src_indices,
        tgt_indices=tgt_indices,
        src_uv=src_feature_uv,
        tgt_uv=tgt_feature_uv,
        src_bearing=src_bearing,
        tgt_bearing=tgt_bearing,
        valid_mask=valid_mask,
        depth_consistency=depth_ok,
        angular_baseline=angular_baseline,
        latitude_weight=latitude_weight,
        metadata={
            "feature_hw": feature_size,
            "image_hw": image_size,
            "samples_per_edge": sample_count,
            "depth_consistency_rel": float(depth_consistency_rel),
            "depth_consistency_abs": float(depth_consistency_abs),
            "min_baseline_deg": float(min_baseline_deg),
            "max_baseline_deg": float(max_baseline_deg),
            "use_wraparound": bool(use_wraparound),
        },
    )
