"""DROID-style feature and context encoders."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(_group_count(channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1)
        self.norm1 = _norm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.norm2 = _norm(out_dim)
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, stride=stride),
                _norm(out_dim),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class BasicEncoder(nn.Module):
    """Compact DROID-style image feature encoder with stride 8 output."""

    def __init__(
        self,
        *,
        input_dim: int = 3,
        output_dim: int = 128,
        base_dim: int | None = None,
    ) -> None:
        super().__init__()
        base = int(base_dim or max(32, min(64, output_dim)))
        mid = max(base, output_dim)
        self.conv1 = nn.Conv2d(input_dim, base, 7, stride=2, padding=3)
        self.norm1 = _norm(base)
        self.layer1 = nn.Sequential(
            ResidualBlock(base, base),
            ResidualBlock(base, base),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(base, mid, stride=2),
            ResidualBlock(mid, mid),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(mid, mid, stride=2),
            ResidualBlock(mid, mid),
        )
        self.proj = nn.Conv2d(mid, output_dim, 1)
        self.output_dim = int(output_dim)
        self.stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.proj(x)


class ContextEncoder(nn.Module):
    """DROID-style context encoder returning hidden state and context input."""

    def __init__(
        self,
        *,
        input_dim: int = 3,
        hidden_dim: int = 128,
        context_dim: int = 128,
        base_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.encoder = BasicEncoder(
            input_dim=input_dim,
            output_dim=self.hidden_dim + self.context_dim,
            base_dim=base_dim,
        )

    @property
    def stride(self) -> int:
        return self.encoder.stride

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(x)
        hidden, inp = torch.split(context, [self.hidden_dim, self.context_dim], dim=1)
        return torch.tanh(hidden), F.relu(inp, inplace=False)
