"""Differentiable spherical bundle-adjustment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .spherical_camera import (
    erp_pixel_to_bearing,
    latitude_area_weight,
    seam_aware_delta,
    spherical_log_residual,
)


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    x, y, zz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([z, -zz, y], dim=-1),
            torch.stack([zz, z, -x], dim=-1),
            torch.stack([-y, x, z], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential map for ``[..., 3]`` rotation vectors."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = theta < 1e-8
    axis = omega / theta.clamp_min(1e-8)
    K = skew(axis)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    eye = eye.expand(*omega.shape[:-1], 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    K_small = skew(omega)
    return torch.where(small[..., None], eye + K_small, R)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map for ``[..., 6]`` vectors ``[tx, ty, tz, rx, ry, rz]``."""
    trans = xi[..., :3]
    omega = xi[..., 3:]
    R = so3_exp(omega)
    T = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = trans
    T[..., 3, 3] = 1.0
    return T


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return torch.einsum("...ij,...nj->...ni", R, points) + t.unsqueeze(-2)


@dataclass
class BALossOutput:
    loss: torch.Tensor
    residual: torch.Tensor
    weights: torch.Tensor
    mean_angular_deg: torch.Tensor


def _ensure_batch_pixels(pixels: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if pixels.ndim == 2:
        return pixels.unsqueeze(0), True
    if pixels.ndim == 3:
        return pixels, False
    raise ValueError(f"Expected pixels as Nx2 or BxNx2, got {tuple(pixels.shape)}")


def spherical_ba_residual(
    source_pixels: torch.Tensor,
    inverse_depth: torch.Tensor,
    T_ji: torch.Tensor,
    *,
    height: int,
    width: int,
    target_delta: Optional[torch.Tensor] = None,
    target_pixels: Optional[torch.Tensor] = None,
    target_bearing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``Log_target(b_pred)`` for spherical BA edges."""
    pixels_b, squeeze = _ensure_batch_pixels(source_pixels)
    B, N, _ = pixels_b.shape
    inv = inverse_depth.reshape(B, N).to(device=pixels_b.device, dtype=pixels_b.dtype)
    T = T_ji.to(device=pixels_b.device, dtype=pixels_b.dtype)
    if T.ndim == 2:
        T = T.unsqueeze(0).expand(B, -1, -1)

    b_i = erp_pixel_to_bearing(pixels_b, height, width)
    X_i = b_i / inv.clamp_min(1e-6).unsqueeze(-1)
    X_j = transform_points(T, X_i)
    b_pred = F.normalize(X_j, dim=-1, eps=1e-12)

    if target_bearing is None:
        if target_pixels is None:
            if target_delta is None:
                raise ValueError("Provide target_delta, target_pixels, or target_bearing.")
            delta = target_delta
            if delta.ndim == 3 and delta.shape[1] == 2:
                delta = delta.permute(0, 2, 1)
            delta = delta.reshape(B, N, 2).to(device=pixels_b.device, dtype=pixels_b.dtype)
            target_pixels = pixels_b + delta
        else:
            target_pixels = target_pixels.reshape(B, N, 2).to(
                device=pixels_b.device, dtype=pixels_b.dtype
            )
        target_pixels = target_pixels.clone()
        target_pixels[..., 0] = torch.remainder(target_pixels[..., 0], float(width))
        target_bearing = erp_pixel_to_bearing(target_pixels, height, width)
    else:
        target_bearing = target_bearing.reshape(B, N, 3).to(
            device=pixels_b.device, dtype=pixels_b.dtype
        )

    residual = spherical_log_residual(target_bearing, b_pred)
    return residual.squeeze(0) if squeeze else residual


def spherical_ba_loss(
    source_pixels: torch.Tensor,
    inverse_depth: torch.Tensor,
    T_ji: torch.Tensor,
    *,
    height: int,
    width: int,
    target_delta: Optional[torch.Tensor] = None,
    target_pixels: Optional[torch.Tensor] = None,
    target_bearing: Optional[torch.Tensor] = None,
    confidence: Optional[torch.Tensor] = None,
    robust_delta: float = 1e-2,
) -> BALossOutput:
    residual = spherical_ba_residual(
        source_pixels,
        inverse_depth,
        T_ji,
        height=height,
        width=width,
        target_delta=target_delta,
        target_pixels=target_pixels,
        target_bearing=target_bearing,
    )
    res_b = residual.unsqueeze(0) if residual.ndim == 2 else residual
    pixels_b, _ = _ensure_batch_pixels(source_pixels)
    area = latitude_area_weight(
        height, width, device=pixels_b.device, dtype=pixels_b.dtype, normalize=False
    )
    v = pixels_b[..., 1].round().long().clamp(0, height - 1)
    point_area = area[0, v, 0]
    weights = point_area
    if confidence is not None:
        weights = weights * confidence.reshape_as(weights).to(weights)
    r2 = (res_b * res_b).sum(dim=-1)
    robust = torch.sqrt(r2 + robust_delta * robust_delta) - robust_delta
    loss = (robust * weights).sum() / weights.sum().clamp_min(1e-8)
    mean_ang = torch.rad2deg(torch.sqrt(r2.detach()).mean())
    return BALossOutput(loss=loss, residual=res_b, weights=weights, mean_angular_deg=mean_ang)


class SphericalBA(nn.Module):
    """Small differentiable spherical BA module.

    This PyTorch implementation prioritizes correctness and gradients.  It is
    not a CUDA replacement for large production BA.
    """

    def __init__(self, height: int, width: int, robust_delta: float = 1e-2) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.robust_delta = float(robust_delta)

    def forward(self, **kwargs) -> BALossOutput:
        kwargs.setdefault("height", self.height)
        kwargs.setdefault("width", self.width)
        kwargs.setdefault("robust_delta", self.robust_delta)
        return spherical_ba_loss(**kwargs)

    def optimize_pose_depth(
        self,
        source_pixels: torch.Tensor,
        inverse_depth: torch.Tensor,
        target_delta: torch.Tensor,
        T_init: torch.Tensor,
        *,
        confidence: Optional[torch.Tensor] = None,
        steps: int = 3,
        lr: float = 1e-2,
        optimize_depth: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        xi = torch.zeros(
            *T_init.shape[:-2], 6, device=T_init.device, dtype=T_init.dtype, requires_grad=True
        )
        if optimize_depth:
            log_inv = inverse_depth.detach().clamp_min(1e-6).log().clone().requires_grad_(True)
            params = [xi, log_inv]
        else:
            log_inv = inverse_depth.detach().clamp_min(1e-6).log()
            params = [xi]
        opt = torch.optim.Adam(params, lr=lr)
        losses: list[float] = []
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            T = se3_exp(xi) @ T_init
            inv = log_inv.exp().clamp_min(1e-6)
            out = self(
                source_pixels=source_pixels,
                inverse_depth=inv,
                T_ji=T,
                target_delta=target_delta,
                confidence=confidence,
            )
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.detach().cpu()))
        with torch.no_grad():
            return se3_exp(xi) @ T_init, log_inv.exp().clamp_min(1e-6), losses

