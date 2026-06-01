"""ERP spherical camera helpers using the original backend convention.

Coordinate convention matches the source 360GS-SLAM backend:

* +X points right
* +Y points down
* +Z points forward

ERP pixel coordinates are floating point ``[u, v]`` with ``u`` horizontal and
``v`` vertical.  Pixel centers should be passed as ``col + 0.5, row + 0.5``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def wrap_horizontal(u: torch.Tensor, width: int) -> torch.Tensor:
    """Wrap horizontal ERP coordinates into ``[0, width)``."""
    return torch.remainder(u, float(width))


def seam_aware_delta(source: torch.Tensor, target: torch.Tensor, width: int) -> torch.Tensor:
    """Return ``target - source`` with the horizontal delta wrapped at the seam.

    ``source`` and ``target`` can be either ``[..., 2]`` pixel tensors or scalar
    horizontal coordinates.  For pixel tensors, the vertical delta is unchanged.
    """
    delta = target - source
    if delta.shape[-1:] == (2,):
        du = torch.remainder(delta[..., 0] + float(width) * 0.5, float(width))
        du = du - float(width) * 0.5
        return torch.stack([du, delta[..., 1]], dim=-1)
    return torch.remainder(delta + float(width) * 0.5, float(width)) - float(width) * 0.5


def erp_pixel_to_bearing(pixel: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert ERP pixels to unit bearings.

    The latitude sign intentionally follows ``utils.erp_geometry`` from the
    original project: ``phi = pi * (v / H - 0.5)``.
    """
    if pixel.shape[-1] != 2:
        raise ValueError(f"Expected pixel tensor with last dim 2, got {tuple(pixel.shape)}")
    pixel = pixel.to(dtype=torch.float32) if not pixel.is_floating_point() else pixel
    u = pixel[..., 0]
    v = pixel[..., 1]
    lam = 2.0 * math.pi * (u / float(width) - 0.5)
    phi = math.pi * (v / float(height) - 0.5)
    x = torch.cos(phi) * torch.sin(lam)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(lam)
    return _normalize(torch.stack([x, y, z], dim=-1))


def bearing_to_erp_pixel(
    bearing: torch.Tensor,
    height: int,
    width: int,
    *,
    wrap: bool = True,
) -> torch.Tensor:
    """Project unit bearings back to ERP pixels."""
    if bearing.shape[-1] != 3:
        raise ValueError(
            f"Expected bearing tensor with last dim 3, got {tuple(bearing.shape)}"
        )
    b = _normalize(bearing)
    lam = torch.atan2(b[..., 0], b[..., 2])
    phi = torch.asin(b[..., 1].clamp(-1.0, 1.0))
    u = float(width) * (lam / (2.0 * math.pi) + 0.5)
    v = float(height) * (phi / math.pi + 0.5)
    if wrap:
        u = wrap_horizontal(u, width)
    return torch.stack([u, v], dim=-1)


def tangent_basis(bearing: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Build an orthonormal tangent basis at each bearing.

    Returns a tensor of shape ``[..., 3, 2]`` where the last dimension stores
    two tangent basis vectors.
    """
    b = _normalize(bearing, eps=eps)
    up = torch.zeros_like(b)
    up[..., 1] = 1.0
    right = torch.zeros_like(b)
    right[..., 0] = 1.0
    use_up = b[..., 1].abs() < 0.9
    ref = torch.where(use_up.unsqueeze(-1), up, right)
    e1 = ref - (ref * b).sum(dim=-1, keepdim=True) * b
    e1 = _normalize(e1, eps=eps)
    e2 = torch.cross(b, e1, dim=-1)
    e2 = _normalize(e2, eps=eps)
    return torch.stack([e1, e2], dim=-1)


def spherical_log_residual(
    target_bearing: torch.Tensor,
    pred_bearing: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return the tangent-plane log residual ``Log_target(pred)``.

    The result has shape ``[..., 2]`` in the tangent basis at
    ``target_bearing``.  This is the residual used by spherical BA.
    """
    target = _normalize(target_bearing)
    pred = _normalize(pred_bearing)
    dot = (target * pred).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    tangent_vec = pred - dot * target
    sin_theta = torch.sin(theta).clamp_min(eps)
    scale = torch.where(theta > eps, theta / sin_theta, torch.ones_like(theta))
    log_vec = scale * tangent_vec
    basis = tangent_basis(target, eps=eps)
    return torch.einsum("...ij,...i->...j", basis, log_vec)


def spherical_angular_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Angular distance in radians between two bearing tensors."""
    aa = _normalize(a)
    bb = _normalize(b)
    dot = (aa * bb).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dot)


def latitude_area_weight(
    height: int,
    width: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """Return ERP cos(latitude) area weights with shape ``(1, H, W)``."""
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    phi = math.pi * (v / float(height) - 0.5)
    weight = torch.cos(phi).clamp_min(0.0).view(1, height, 1).expand(1, height, width)
    if normalize:
        weight = weight / weight.mean().clamp_min(torch.finfo(dtype).eps)
    return weight


def pixel_grid(
    height: int,
    width: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dense ERP pixel-center grid with shape ``(H, W, 2)``."""
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def image_shape_from_tensor(image: torch.Tensor) -> Tuple[int, int]:
    """Return ``(H, W)`` for a ``[..., H, W]`` image tensor."""
    if image.ndim < 2:
        raise ValueError("Image tensor must have at least two dimensions.")
    return int(image.shape[-2]), int(image.shape[-1])

