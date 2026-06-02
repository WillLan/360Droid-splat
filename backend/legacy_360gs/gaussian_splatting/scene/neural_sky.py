"""Tiny direction-conditioned MLP that produces an ERP sky panorama.

Used in place of the 2D learnable ``_erp_sky_bg`` texture for panoramic
GS-SLAM (see ``gaussian_model.GaussianModel.get_neural_sky_background``).

The MLP takes a 3D unit direction (the ray from the body camera centre
through an ERP pixel) and returns RGB in [0, 1] via a final sigmoid.
Direction is encoded with multi-frequency sin/cos positional encoding
(NeRF-style). Inspired by Splatfacto-W (Xu et al., 2024); we keep the
implementation small enough that lazy ERP-grid caching dominates cost.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class NeuralSkyMLP(nn.Module):
    """Direction 鈫?RGB MLP with frequency-encoded inputs."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_freq: int = 6,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_freq = max(0, int(n_freq))
        in_dim = 3 + (3 * 2 * self.n_freq if self.n_freq > 0 else 0)
        layers = []
        for i in range(self.n_layers):
            in_ch = in_dim if i == 0 else self.hidden_dim
            is_last = i == self.n_layers - 1
            out_ch = 3 if is_last else self.hidden_dim
            layers.append(nn.Linear(in_ch, out_ch))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self._dir_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    @staticmethod
    def _erp_pixel_directions(
        H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-ERP-pixel unit directions in the body-camera frame.

        Convention matches utils/erp_geometry.erp_dense_pixel_center_bearings:
        column index -> azimuth in [-pi, pi); row index -> elevation in
        [pi/2, -pi/2). Returned shape: (H, W, 3).
        """
        u = torch.arange(W, device=device, dtype=dtype)
        v = torch.arange(H, device=device, dtype=dtype)
        az = (u + 0.5) / float(W) * (2.0 * math.pi) - math.pi
        el = math.pi * 0.5 - (v + 0.5) / float(H) * math.pi
        az_grid = az.view(1, W).expand(H, W)
        el_grid = el.view(H, 1).expand(H, W)
        cos_el = torch.cos(el_grid)
        d_x = cos_el * torch.sin(az_grid)
        d_y = -torch.sin(el_grid)
        d_z = cos_el * torch.cos(az_grid)
        return torch.stack((d_x, d_y, d_z), dim=-1).contiguous()

    def _freq_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq <= 0:
            return x
        freqs = (
            2.0 ** torch.arange(self.n_freq, device=x.device, dtype=x.dtype)
        ) * math.pi
        # x: (N, 3) -> xb: (N, 3, K) -> enc: (N, 3*2K)
        xb = x.unsqueeze(-1) * freqs
        enc = torch.cat((torch.sin(xb), torch.cos(xb)), dim=-1)
        enc = enc.flatten(-2)
        return torch.cat((x, enc), dim=-1)

    def forward(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the rendered sky RGB as (3, H, W) in [0, 1]."""
        H_int, W_int = int(H), int(W)
        key = (H_int, W_int, f"{device}|{dtype}")
        dirs = self._dir_cache.get(key)
        if (
            dirs is None
            or dirs.device != torch.device(device)
            or dirs.dtype != dtype
        ):
            dirs = self._erp_pixel_directions(H_int, W_int, device=device, dtype=dtype)
            self._dir_cache[key] = dirs
        feats = self._freq_encode(dirs.reshape(-1, 3))
        out = self.net(feats)
        out = torch.sigmoid(out)
        out = out.t().reshape(3, H_int, W_int).contiguous()
        return out
