"""Spherical correlation pyramid for DROID-style ERP tracking."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def coords_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y = torch.arange(height, device=device, dtype=dtype) + 0.5
    x = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy], dim=0)
    return coords.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def _norm_grid_from_coords(
    coords: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    x = coords[..., 0]
    y = coords[..., 1]
    if width <= 1:
        nx = torch.zeros_like(x)
    else:
        nx = 2.0 * (x - 0.5) / float(width - 1) - 1.0
    if height <= 1:
        ny = torch.zeros_like(y)
    else:
        ny = 2.0 * (y - 0.5) / float(height - 1) - 1.0
    return torch.stack([nx, ny], dim=-1)


def sample_erp_feature_map(feature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample ``feature`` at ``coords`` with horizontal wrap-around.

    ``coords`` has shape ``B x Hq x Wq x K x 2`` in feature-pixel indices.
    Returns ``B x C x Hq x Wq x K``.
    """
    B, C, H, W = feature.shape
    if coords.ndim != 5 or coords.shape[0] != B or coords.shape[-1] != 2:
        raise ValueError(f"Bad coords shape {tuple(coords.shape)} for feature {tuple(feature.shape)}")
    _, Hq, Wq, K, _ = coords.shape
    x = torch.remainder(coords[..., 0], max(float(W), 1.0))
    y = coords[..., 1].clamp(0.0, max(float(H - 1), 0.0))
    feature_pad = torch.cat([feature[..., -1:], feature, feature[..., :1]], dim=-1)
    grid = _norm_grid_from_coords(
        torch.stack([x + 1.0, y], dim=-1),
        height=H,
        width=W + 2,
    )
    grid = grid.reshape(B, Hq, Wq * K, 2)
    sampled = F.grid_sample(
        feature_pad,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(B, C, Hq, Wq, K)


class SphericalCorrBlock(nn.Module):
    """Local spherical correlation pyramid.

    It follows DROID's repeated correlation lookup shape while adapting the
    local window to ERP geometry and wrapping longitude at the seam.
    """

    def __init__(
        self,
        fmap0: torch.Tensor,
        fmap1: torch.Tensor,
        *,
        num_levels: int = 4,
        radius: int = 3,
        latitude_scale: bool = True,
        min_cos_latitude: float = 0.25,
    ) -> None:
        super().__init__()
        if fmap0.shape != fmap1.shape:
            raise ValueError(f"Feature shape mismatch: {tuple(fmap0.shape)} vs {tuple(fmap1.shape)}")
        self.fmap0 = F.normalize(fmap0, dim=1)
        fmap1_norm = F.normalize(fmap1, dim=1)
        self.fmap1_pyramid = [fmap1_norm]
        for _ in range(1, int(num_levels)):
            prev = self.fmap1_pyramid[-1]
            if prev.shape[-2] >= 2 and prev.shape[-1] >= 2:
                self.fmap1_pyramid.append(F.avg_pool2d(prev, 2, stride=2))
            else:
                self.fmap1_pyramid.append(prev)
        self.num_levels = int(num_levels)
        self.radius = int(radius)
        self.latitude_scale = bool(latitude_scale)
        self.min_cos_latitude = float(min_cos_latitude)
        self.out_channels = self.num_levels * (2 * self.radius + 1) ** 2

    def _offsets(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        B, _, H, W = coords.shape
        r = self.radius
        dy = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        dx = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")
        offsets = torch.stack([xx, yy], dim=-1).reshape(1, 1, 1, -1, 2)
        K = offsets.shape[-2]
        if self.latitude_scale:
            # Equal tangent distance at high latitude needs a larger longitude
            # step in ERP coordinates.
            src_y = coords[:, 1]
            H0 = max(int(H), 1)
            phi = math.pi * (src_y / float(H0) - 0.5)
            cos_lat = torch.cos(phi).abs().clamp_min(self.min_cos_latitude)
            ox = offsets[..., 0].expand(B, H, W, K) / cos_lat.unsqueeze(-1)
            oy = offsets[..., 1].expand(B, H, W, K)
            offsets = torch.stack([ox, oy], dim=-1)
        else:
            offsets = offsets.expand(B, H, W, K, 2)
        return offsets / float(2**level)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 4 or coords.shape[1] != 2:
            raise ValueError(f"Expected coords as Bx2xHxW, got {tuple(coords.shape)}")
        B, _, H, W = coords.shape
        corr_levels = []
        coords_hw = coords.permute(0, 2, 3, 1).contiguous()
        for level, fmap1 in enumerate(self.fmap1_pyramid):
            scale = float(2**level)
            coords_l = coords_hw / scale
            offsets = self._offsets(coords, level)
            query = coords_l.unsqueeze(-2) + offsets
            sampled = sample_erp_feature_map(fmap1, query)
            corr = (self.fmap0.unsqueeze(-1) * sampled).sum(dim=1)
            corr_levels.append(corr)
        out = torch.cat(corr_levels, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous() / torch.sqrt(
            torch.tensor(float(self.fmap0.shape[1]), device=coords.device, dtype=coords.dtype)
        )
