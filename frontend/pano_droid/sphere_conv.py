"""BlueHorn/SphereNet-style spherical convolution for ERP feature maps.

This module replaces the earlier padding-only ERP convolution.  The
implementation follows the method used by BlueHorn07/sphereConv-pytorch at the
operator level: build a spherical sampling pattern for an equirectangular
feature map, sample the tangent-plane kernel positions with ``grid_sample``,
then apply Conv2d weights to the sampled kernel stack.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        a, b = tuple(value)
        return int(a), int(b)
    return int(value), int(value)


def _safe_norm_to_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if width <= 1:
        nx = torch.zeros_like(x)
    else:
        nx = 2.0 * x / float(width - 1) - 1.0
    if height <= 1:
        ny = torch.zeros_like(y)
    else:
        ny = 2.0 * y / float(height - 1) - 1.0
    return torch.stack([nx, ny], dim=-1)


@lru_cache(maxsize=64)
def _offset_cache(
    kernel_h: int,
    kernel_w: int,
    dilation_h: int,
    dilation_w: int,
) -> tuple[tuple[float, float], ...]:
    cy = (kernel_h - 1) * 0.5
    cx = (kernel_w - 1) * 0.5
    offsets: list[tuple[float, float]] = []
    for iy in range(kernel_h):
        for ix in range(kernel_w):
            offsets.append(((iy - cy) * dilation_h, (ix - cx) * dilation_w))
    return tuple(offsets)


class GridGenerator:
    """Generate ERP sampling grids for a spherical tangent-plane kernel."""

    def __init__(
        self,
        height: int,
        width: int,
        kernel_size=3,
        stride=1,
        dilation=1,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

    @property
    def output_shape(self) -> tuple[int, int]:
        h = (self.height + self.stride[0] - 1) // self.stride[0]
        w = (self.width + self.stride[1] - 1) // self.stride[1]
        return h, w

    def create_sampling_grid(
        self,
        *,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a grid with shape ``(K, Hout, Wout, 2)`` in pixel indices."""
        H, W = self.height, self.width
        out_h, out_w = self.output_shape
        y = torch.arange(out_h, device=device, dtype=dtype) * self.stride[0]
        x = torch.arange(out_w, device=device, dtype=dtype) * self.stride[1]
        y = y.clamp(0, max(H - 1, 0))
        x = x.clamp(0, max(W - 1, 0))
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        lam = 2.0 * math.pi * (xx / max(float(W), 1.0) - 0.5)
        phi = math.pi * (yy / max(float(H), 1.0) - 0.5)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_lam = torch.sin(lam)
        cos_lam = torch.cos(lam)
        center = torch.stack(
            [cos_phi * sin_lam, sin_phi, cos_phi * cos_lam],
            dim=-1,
        )
        e_lon = torch.stack([cos_lam, torch.zeros_like(lam), -sin_lam], dim=-1)
        e_lat = torch.stack(
            [-sin_phi * sin_lam, cos_phi, -sin_phi * cos_lam],
            dim=-1,
        )
        lon_step = 2.0 * math.pi / max(float(W), 1.0)
        lat_step = math.pi / max(float(H), 1.0)

        grids = []
        for dy, dx in _offset_cache(
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
        ):
            tangent = float(dx) * lon_step * e_lon + float(dy) * lat_step * e_lat
            bearing = F.normalize(center + tangent, dim=-1, eps=1e-12)
            sample_lam = torch.atan2(bearing[..., 0], bearing[..., 2])
            sample_phi = torch.asin(bearing[..., 1].clamp(-1.0, 1.0))
            sample_x = float(W) * (sample_lam / (2.0 * math.pi) + 0.5)
            sample_y = float(H) * (sample_phi / math.pi + 0.5)
            sample_x = torch.remainder(sample_x, max(float(W), 1.0))
            sample_y = sample_y.clamp(0.0, max(float(H - 1), 0.0))
            grids.append(torch.stack([sample_x, sample_y], dim=-1))
        return torch.stack(grids, dim=0)


class SphereConv2d(nn.Module):
    """Drop-in Conv2d-style spherical convolution for ERP tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.stride = _pair(stride)
        self.groups = int(groups)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
        )

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> "SphereConv2d":
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            groups=conv.groups,
        )
        with torch.no_grad():
            layer.conv.weight.copy_(conv.weight)
            if conv.bias is not None and layer.conv.bias is not None:
                layer.conv.bias.copy_(conv.bias)
        return layer

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.conv.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.conv.bias

    def _sample_kernel(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        generator = GridGenerator(
            H,
            W,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        grid_px = generator.create_sampling_grid(device=x.device, dtype=x.dtype)
        K, out_h, out_w, _ = grid_px.shape

        x_pad = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        sample_x = grid_px[..., 0] + 1.0
        sample_y = grid_px[..., 1]
        grid = _safe_norm_to_grid(
            sample_x,
            sample_y,
            height=H,
            width=W + 2,
        )
        grid = grid.reshape(K * out_h, out_w, 2).unsqueeze(0).expand(B, -1, -1, -1)
        sampled = F.grid_sample(
            x_pad,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.reshape(B, C, K, out_h, out_w)
        return sampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as BxCxHxW, got {tuple(x.shape)}")
        sampled = self._sample_kernel(x)
        B, C, K, out_h, out_w = sampled.shape
        groups = self.groups
        if C % groups != 0 or self.conv.out_channels % groups != 0:
            raise ValueError("Input and output channels must be divisible by groups.")
        sampled = sampled.view(B, groups, C // groups, K, out_h, out_w)
        weight = self.conv.weight.view(
            groups,
            self.conv.out_channels // groups,
            C // groups,
            K,
        )
        out = torch.einsum("bgckhw,gock->bgohw", sampled, weight)
        out = out.reshape(B, self.conv.out_channels, out_h, out_w)
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out


def replace_3x3_conv_with_sphere(module: nn.Module) -> nn.Module:
    """Recursively replace 3x3/5x5 Conv2d layers with BlueHorn-style SphereConv2d."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size in {(3, 3), (5, 5)}:
            setattr(module, name, SphereConv2d.from_conv2d(child))
        else:
            replace_3x3_conv_with_sphere(child)
    return module

