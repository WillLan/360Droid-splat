"""Spherical pseudo correspondence generation for Stage 1A."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .spherical_erp import (
    DEFAULT_ERP_HEIGHT,
    DEFAULT_ERP_WIDTH,
    sample_erp_with_wrap,
)
from .spherical_projection import project_source_to_target_erp


@dataclass
class SphericalCorrespondence:
    """Sampled pseudo correspondences between ERP source and target views."""

    src_view: torch.Tensor
    tgt_view: torch.Tensor
    src_uv: torch.Tensor
    tgt_uv: torch.Tensor
    src_ray: torch.Tensor
    tgt_ray: torch.Tensor
    valid_mask: torch.Tensor
    visibility: torch.Tensor
    weight: torch.Tensor


def _normalize_depth(depth: torch.Tensor) -> tuple[torch.Tensor, bool]:
    value = depth.float() if not depth.is_floating_point() else depth
    if value.ndim == 3:
        return value.unsqueeze(0).unsqueeze(2), True
    if value.ndim == 4:
        if int(value.shape[1]) == 1:
            return value.unsqueeze(0), True
        return value.unsqueeze(2), False
    if value.ndim == 5 and int(value.shape[2]) == 1:
        return value, False
    raise ValueError(
        "depth must have shape VxHxW, Vx1xHxW, BxVxHxW, or BxVx1xHxW; "
        f"got {tuple(depth.shape)}."
    )


def _normalize_poses(poses_c2w: torch.Tensor, batch_size: int, num_views: int) -> tuple[torch.Tensor, bool]:
    poses = poses_c2w.float() if not poses_c2w.is_floating_point() else poses_c2w
    if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
        if int(poses.shape[0]) != num_views:
            raise ValueError(f"poses view count {int(poses.shape[0])} does not match depth view count {num_views}.")
        return poses.unsqueeze(0).expand(batch_size, -1, -1, -1), True
    if poses.ndim == 4 and poses.shape[-2:] == (4, 4):
        if int(poses.shape[0]) != batch_size or int(poses.shape[1]) != num_views:
            raise ValueError(
                f"poses shape {tuple(poses.shape)} does not match depth batch/view {(batch_size, num_views)}."
            )
        return poses, False
    raise ValueError(f"poses_c2w must have shape Vx4x4 or BxVx4x4, got {tuple(poses.shape)}.")


def _default_pairs(num_views: int, device: torch.device) -> torch.Tensor:
    if num_views < 2:
        raise ValueError("At least two views are required when pair_indices is not provided.")
    return torch.tensor([(idx, idx + 1) for idx in range(num_views - 1)], device=device, dtype=torch.long)


def _normalize_pairs(
    pair_indices: torch.Tensor | list[tuple[int, int]] | None,
    *,
    batch_size: int,
    num_views: int,
    device: torch.device,
) -> torch.Tensor:
    if pair_indices is None:
        pairs = _default_pairs(num_views, device)
    else:
        pairs = torch.as_tensor(pair_indices, device=device, dtype=torch.long)
    if pairs.ndim == 2 and int(pairs.shape[1]) == 2:
        pairs = pairs.unsqueeze(0).expand(batch_size, -1, -1)
    elif pairs.ndim != 3 or int(pairs.shape[-1]) != 2 or int(pairs.shape[0]) != batch_size:
        raise ValueError(f"pair_indices must have shape Ex2 or BxEx2, got {tuple(pairs.shape)}.")
    if int(pairs.min()) < 0 or int(pairs.max()) >= num_views:
        raise ValueError("pair_indices contain view indices outside the depth/pose tensors.")
    return pairs


def _latitude_weights(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.arange(height, device=device, dtype=dtype) + 0.5
    latitude = math.pi * (rows / float(height) - 0.5)
    weights = torch.cos(latitude).clamp_min(0.0).view(height, 1).expand(height, width)
    return weights.reshape(-1)


def _sample_query_uv(
    *,
    height: int,
    width: int,
    count: int,
    sampling: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    total = int(height) * int(width)
    sample_count = min(max(1, int(count)), total)
    mode = str(sampling).lower()
    if mode in {"cosine_latitude_weighted", "cosine"}:
        weights = _latitude_weights(height, width, device=device, dtype=torch.float32)
        indices = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
    elif mode in {"uniform", "random"}:
        indices = torch.randperm(total, device=device, generator=generator)[:sample_count]
    elif mode in {"grid", "linspace"}:
        indices = torch.linspace(0, total - 1, steps=sample_count, device=device).round().long()
    else:
        raise ValueError(f"Unsupported correspondence sampling mode: {sampling!r}.")
    v = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype) + 0.5
    u = torch.remainder(indices, width).to(dtype=dtype) + 0.5
    return torch.stack([u, v], dim=-1)


def _normalize_query_uv(
    query_uv: torch.Tensor | None,
    *,
    batch_size: int,
    edge_count: int,
    height: int,
    width: int,
    count: int,
    sampling: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if query_uv is None:
        base = _sample_query_uv(
            height=height,
            width=width,
            count=count,
            sampling=sampling,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return base.view(1, 1, -1, 2).expand(batch_size, edge_count, -1, -1)
    value = query_uv.to(device=device, dtype=dtype)
    if value.shape[-1] != 2:
        raise ValueError(f"query_uv must end with dimension 2, got {tuple(value.shape)}.")
    if value.ndim == 2:
        return value.view(1, 1, -1, 2).expand(batch_size, edge_count, -1, -1)
    if value.ndim == 3 and int(value.shape[0]) == edge_count:
        return value.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if value.ndim == 4 and int(value.shape[0]) == batch_size and int(value.shape[1]) == edge_count:
        return value
    raise ValueError(
        "query_uv must have shape Sx2, ExSx2, or BxExSx2; "
        f"got {tuple(value.shape)}."
    )


def _gather_frames(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_views = int(value.shape[0]), int(value.shape[1])
    flat = value.reshape(batch_size * num_views, *value.shape[2:])
    batch_offsets = torch.arange(batch_size, device=value.device, dtype=torch.long).view(batch_size, 1) * num_views
    return flat[(batch_offsets + indices.long()).reshape(-1)]


def _latitude_weight_for_uv(src_uv: torch.Tensor, height: int) -> torch.Tensor:
    latitude = math.pi * (src_uv[..., 1] / float(height) - 0.5)
    return torch.cos(latitude).clamp_min(0.0)


def generate_spherical_pseudo_correspondence(
    depth: torch.Tensor,
    poses_c2w: torch.Tensor,
    pair_indices: torch.Tensor | list[tuple[int, int]] | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    num_query_per_pair: int = 2048,
    sampling: str = "cosine_latitude_weighted",
    min_depth: float = 0.05,
    max_depth: float = 100.0,
    visibility_rel_thresh: float = 0.05,
    query_uv: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SphericalCorrespondence:
    """Generate spherical pseudo correspondences from initial pose and depth."""

    depth_bv, squeeze_batch = _normalize_depth(depth)
    batch_size, num_views, _, depth_h, depth_w = (int(dim) for dim in depth_bv.shape)
    h = int(height) if height is not None else depth_h
    w = int(width) if width is not None else depth_w
    if (h, w) != (depth_h, depth_w):
        raise ValueError(f"Requested ERP size {(h, w)} does not match depth shape {(depth_h, depth_w)}.")
    device, dtype = depth_bv.device, depth_bv.dtype
    poses_bv, _ = _normalize_poses(poses_c2w.to(device=device), batch_size, num_views)
    poses_bv = poses_bv.to(device=device, dtype=dtype)
    pairs = _normalize_pairs(pair_indices, batch_size=batch_size, num_views=num_views, device=device)
    edge_count = int(pairs.shape[1])
    src_idx = pairs[..., 0]
    tgt_idx = pairs[..., 1]
    src_uv = _normalize_query_uv(
        query_uv,
        batch_size=batch_size,
        edge_count=edge_count,
        height=h,
        width=w,
        count=int(num_query_per_pair),
        sampling=sampling,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    sample_count = int(src_uv.shape[-2])

    src_depth_maps = _gather_frames(depth_bv, src_idx).reshape(batch_size * edge_count, 1, h, w)
    tgt_depth_maps = _gather_frames(depth_bv, tgt_idx).reshape(batch_size * edge_count, 1, h, w)
    flat_src_uv = src_uv.reshape(batch_size * edge_count, sample_count, 2)
    src_depth = sample_erp_with_wrap(src_depth_maps, flat_src_uv)[..., 0]

    src_poses = _gather_frames(poses_bv, src_idx).reshape(batch_size * edge_count, 4, 4)
    tgt_poses = _gather_frames(poses_bv, tgt_idx).reshape(batch_size * edge_count, 4, 4)
    projection = project_source_to_target_erp(
        flat_src_uv,
        src_depth,
        src_poses,
        tgt_poses,
        height=h,
        width=w,
    )
    target_depth = sample_erp_with_wrap(tgt_depth_maps, projection.target_uv)[..., 0]
    depth_floor = torch.as_tensor(float(min_depth), device=device, dtype=dtype)
    depth_ceiling = torch.as_tensor(float(max_depth), device=device, dtype=dtype)
    src_depth_ok = torch.isfinite(src_depth) & (src_depth >= depth_floor) & (src_depth <= depth_ceiling)
    tgt_depth_ok = torch.isfinite(target_depth) & (target_depth >= depth_floor) & (target_depth <= depth_ceiling)
    range_ok = torch.isfinite(projection.target_range) & (projection.target_range >= depth_floor)
    rel_error = (projection.target_range - target_depth).abs() / target_depth.abs().clamp_min(1.0e-12)
    visibility = range_ok & tgt_depth_ok & (rel_error < float(visibility_rel_thresh))
    finite_ok = (
        torch.isfinite(projection.source_ray).all(dim=-1)
        & torch.isfinite(projection.target_ray).all(dim=-1)
        & torch.isfinite(projection.target_uv).all(dim=-1)
    )
    valid = src_depth_ok & visibility & finite_ok

    out_shape = (batch_size, edge_count, sample_count)
    src_views = src_idx.unsqueeze(-1).expand(out_shape)
    tgt_views = tgt_idx.unsqueeze(-1).expand(out_shape)
    src_uv_out = src_uv
    tgt_uv_out = projection.target_uv.reshape(batch_size, edge_count, sample_count, 2)
    src_ray_out = projection.source_ray.reshape(batch_size, edge_count, sample_count, 3)
    tgt_ray_out = projection.target_ray.reshape(batch_size, edge_count, sample_count, 3)
    valid_out = valid.reshape(out_shape)
    visibility_out = visibility.reshape(out_shape)
    weights = _latitude_weight_for_uv(src_uv_out, h).to(dtype=dtype) * valid_out.to(dtype=dtype)

    if squeeze_batch:
        return SphericalCorrespondence(
            src_view=src_views[0],
            tgt_view=tgt_views[0],
            src_uv=src_uv_out[0],
            tgt_uv=tgt_uv_out[0],
            src_ray=src_ray_out[0],
            tgt_ray=tgt_ray_out[0],
            valid_mask=valid_out[0],
            visibility=visibility_out[0],
            weight=weights[0],
        )
    return SphericalCorrespondence(
        src_view=src_views,
        tgt_view=tgt_views,
        src_uv=src_uv_out,
        tgt_uv=tgt_uv_out,
        src_ray=src_ray_out,
        tgt_ray=tgt_ray_out,
        valid_mask=valid_out,
        visibility=visibility_out,
        weight=weights,
    )
