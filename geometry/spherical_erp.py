"""ERP spherical geometry helpers for Stage 1A.

The convention matches the existing project panoramic camera helpers:

* +X points right
* +Y points down
* +Z points forward

ERP pixel coordinates are floating point ``[u, v]`` values. Pixel centers are
represented as ``col + 0.5, row + 0.5``. Longitude wraps horizontally; latitude
is clamped by callers that sample image tensors and is never wrapped.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DEFAULT_ERP_HEIGHT = 504
DEFAULT_ERP_WIDTH = 1008


def _normalize(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return value / torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(eps)


def _validate_hw(height: int, width: int) -> tuple[int, int]:
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        raise ValueError(f"ERP height and width must be positive, got {(h, w)!r}.")
    return h, w


def wrap_longitude_pixel(u: torch.Tensor, width: int = DEFAULT_ERP_WIDTH) -> torch.Tensor:
    """Wrap horizontal ERP pixel coordinates into ``[0, width)``."""

    _, w = _validate_hw(1, width)
    return torch.remainder(u, float(w))


def erp_pixel_to_unit_ray(
    pixel: torch.Tensor,
    height: int = DEFAULT_ERP_HEIGHT,
    width: int = DEFAULT_ERP_WIDTH,
    *,
    wrap_horizontal: bool = True,
    clamp_vertical: bool = True,
) -> torch.Tensor:
    """Convert ERP pixels to unit rays using the project convention."""

    h, w = _validate_hw(height, width)
    if pixel.shape[-1] != 2:
        raise ValueError(f"pixel must end with dimension 2, got {tuple(pixel.shape)}.")
    value = pixel.float() if not pixel.is_floating_point() else pixel
    u = value[..., 0]
    v = value[..., 1]
    if wrap_horizontal:
        u = wrap_longitude_pixel(u, w)
    if clamp_vertical:
        v = v.clamp(0.0, float(h))

    longitude = 2.0 * math.pi * (u / float(w) - 0.5)
    latitude = math.pi * (v / float(h) - 0.5)
    x = torch.cos(latitude) * torch.sin(longitude)
    y = torch.sin(latitude)
    z = torch.cos(latitude) * torch.cos(longitude)
    return _normalize(torch.stack([x, y, z], dim=-1))


def unit_ray_to_erp_pixel(
    ray: torch.Tensor,
    height: int = DEFAULT_ERP_HEIGHT,
    width: int = DEFAULT_ERP_WIDTH,
    *,
    wrap_horizontal: bool = True,
    clamp_vertical: bool = True,
) -> torch.Tensor:
    """Project unit rays to ERP pixel coordinates."""

    h, w = _validate_hw(height, width)
    if ray.shape[-1] != 3:
        raise ValueError(f"ray must end with dimension 3, got {tuple(ray.shape)}.")
    bearing = _normalize(ray)
    longitude = torch.atan2(bearing[..., 0], bearing[..., 2])
    latitude = torch.asin(bearing[..., 1].clamp(-1.0, 1.0))
    u = float(w) * (longitude / (2.0 * math.pi) + 0.5)
    v = float(h) * (latitude / math.pi + 0.5)
    if wrap_horizontal:
        u = wrap_longitude_pixel(u, w)
    if clamp_vertical:
        v = v.clamp(0.0, float(h))
    return torch.stack([u, v], dim=-1)


def build_erp_ray_grid(
    height: int = DEFAULT_ERP_HEIGHT,
    width: int = DEFAULT_ERP_WIDTH,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a dense ``H x W x 3`` ERP unit-ray grid at pixel centers."""

    h, w = _validate_hw(height, width)
    ys = torch.arange(h, device=device, dtype=dtype) + 0.5
    xs = torch.arange(w, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixel = torch.stack([xx, yy], dim=-1)
    return erp_pixel_to_unit_ray(pixel, h, w)


def safe_acos_dot(ray_a: torch.Tensor, ray_b: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    """Return ``acos(dot(normalize(ray_a), normalize(ray_b)))`` in radians."""

    a = _normalize(ray_a, eps=eps)
    b = _normalize(ray_b, eps=eps)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def spherical_geodesic_distance(
    ray_a: torch.Tensor,
    ray_b: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Great-circle angular distance on the unit sphere, in radians."""

    a = _normalize(ray_a, eps=eps)
    b = _normalize(ray_b, eps=eps)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    cross_norm = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1)
    return torch.atan2(cross_norm, dot)


def circular_pad_longitude(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circularly pad the last tensor dimension, used for ERP longitude wrap."""

    amount = int(pad)
    if amount < 0:
        raise ValueError(f"pad must be non-negative, got {amount}.")
    if amount == 0:
        return x
    if x.shape[-1] == 0:
        raise ValueError("Cannot circular-pad an empty longitude dimension.")
    if amount > int(x.shape[-1]):
        repeats = int(math.ceil(float(amount) / float(x.shape[-1]))) + 1
        tiled = x.repeat_interleave(repeats, dim=-1)
        left = tiled[..., -amount:]
        right = tiled[..., :amount]
    else:
        left = x[..., -amount:]
        right = x[..., :amount]
    return torch.cat([left, x, right], dim=-1)


def _feature_to_nchw(feature_map: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    if feature_map.ndim == 3:
        return feature_map.unsqueeze(0), (), True
    if feature_map.ndim == 4:
        return feature_map, (int(feature_map.shape[0]),), False
    if feature_map.ndim == 5:
        b, v, c, h, w = (int(dim) for dim in feature_map.shape)
        return feature_map.reshape(b * v, c, h, w), (b, v), False
    raise ValueError(
        "feature_map must have shape CxHxW, NxCxHxW, or BxVxCxHxW; "
        f"got {tuple(feature_map.shape)}."
    )


def _pixel_to_flat_batches(pixel: torch.Tensor, leading_shape: tuple[int, ...], flat_count: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if pixel.shape[-1] != 2:
        raise ValueError(f"pixel must end with dimension 2, got {tuple(pixel.shape)}.")
    value = pixel.float() if not pixel.is_floating_point() else pixel
    if not leading_shape:
        points = value.reshape(1, -1, 2)
        return points, tuple(value.shape[:-1])
    if tuple(value.shape[: len(leading_shape)]) == leading_shape:
        point_shape = tuple(value.shape[len(leading_shape) : -1])
        return value.reshape(flat_count, -1, 2), point_shape
    if value.ndim >= 2 and int(value.shape[0]) == flat_count:
        point_shape = tuple(value.shape[1:-1])
        return value.reshape(flat_count, -1, 2), point_shape
    point_shape = tuple(value.shape[:-1])
    expanded = value.reshape(1, -1, 2).expand(flat_count, -1, -1)
    return expanded, point_shape


def sample_erp_with_wrap(
    feature_map: torch.Tensor,
    pixel: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample ERP feature maps at pixel coordinates with horizontal wrap.

    Supports ``CxHxW``, ``NxCxHxW``, and ``BxVxCxHxW`` maps. The returned tensor
    stores channels last: ``...xC``. Horizontal samples wrap across the seam;
    vertical coordinates are clamped to the nearest valid pixel center.
    """

    flat, leading_shape, single_map = _feature_to_nchw(feature_map)
    n, _, height, width = (int(dim) for dim in flat.shape)
    points, point_shape = _pixel_to_flat_batches(pixel.to(device=flat.device), leading_shape, n)
    points = points.to(dtype=flat.dtype)

    padded = circular_pad_longitude(flat, 1)
    padded_width = width + 2
    u = wrap_longitude_pixel(points[..., 0], width) + 1.0
    v = points[..., 1].clamp(0.5, float(height) - 0.5)
    norm_x = 2.0 * (u - 0.5) / max(padded_width - 1, 1) - 1.0
    norm_y = 2.0 * (v - 0.5) / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(n, -1, 1, 2)
    sampled = F.grid_sample(
        padded,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    values = sampled[:, :, :, 0].transpose(1, 2).reshape(*leading_shape, *point_shape, flat.shape[1])
    if single_map:
        return values.reshape(*point_shape, flat.shape[1])
    return values
