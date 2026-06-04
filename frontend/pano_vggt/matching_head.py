"""Trainable PanoVGGT-M3-Sphere matching and sky-mask heads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _select_feature(feature: torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor], feature_key: str | int | None) -> torch.Tensor:
    if torch.is_tensor(feature):
        return feature
    if isinstance(feature, dict):
        if feature_key is None:
            if len(feature) != 1:
                raise ValueError("feature_key is required when feature is a dict with multiple entries.")
            return next(iter(feature.values()))
        key = str(feature_key)
        if key not in feature:
            raise KeyError(f"Feature dict does not contain key {key!r}.")
        return feature[key]
    if isinstance(feature, Sequence):
        if feature_key is None:
            idx = -1
        elif isinstance(feature_key, int):
            idx = feature_key
        else:
            idx = int(feature_key)
        return feature[idx]
    raise TypeError(f"Unsupported feature container: {type(feature)!r}")


def _flatten_feature(feature: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    if feature.ndim == 5:
        b, n, c, h, w = feature.shape
        return feature.reshape(b * n, c, h, w), (int(b), int(n))
    if feature.ndim == 4:
        return feature, None
    raise ValueError(f"feature must have shape BxNxCxHxW or MxCxHxW, got {tuple(feature.shape)}")


def _restore_feature_shape(value: torch.Tensor, batch_frames: tuple[int, int] | None) -> torch.Tensor:
    if batch_frames is None:
        return value
    b, n = batch_frames
    return value.view(b, n, value.shape[1], value.shape[2], value.shape[3])


class _ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, int(out_channels))
        while int(out_channels) % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1),
            nn.GroupNorm(groups, int(out_channels)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PanoVGGTMatchingHead(nn.Module):
    """Dense descriptor and match-confidence head.

    The output spatial resolution always follows the input feature resolution.
    For 5D input ``[B, N, C, Hf, Wf]`` the outputs are 5D. For 4D input
    ``[M, C, Hf, Wf]`` the outputs are 4D.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        descriptor_dim: int = 24,
        hidden_dim: int = 128,
        num_conv_blocks: int = 2,
        feature_key: str | int | None = None,
    ) -> None:
        super().__init__()
        if int(descriptor_dim) <= 0:
            raise ValueError("descriptor_dim must be positive.")
        if int(num_conv_blocks) <= 0:
            raise ValueError("num_conv_blocks must be positive.")
        self.feature_dim = int(feature_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_conv_blocks = int(num_conv_blocks)
        self.feature_key = feature_key

        blocks: list[nn.Module] = []
        in_dim = self.feature_dim
        for _ in range(self.num_conv_blocks):
            blocks.append(_ConvNormGELU(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.descriptor_proj = nn.Conv2d(self.hidden_dim, self.descriptor_dim, kernel_size=1)
        self.match_confidence_proj = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the matching head on a PanoVGGT feature tensor."""

        selected = _select_feature(feature, self.feature_key)
        flat, batch_frames = _flatten_feature(selected)
        if int(flat.shape[1]) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {int(flat.shape[1])}.")
        hidden = self.trunk(flat)
        descriptors = F.normalize(self.descriptor_proj(hidden), dim=1, eps=1.0e-6)
        match_confidence = torch.sigmoid(self.match_confidence_proj(hidden))
        return {
            "dense_descriptors": _restore_feature_shape(descriptors, batch_frames),
            "match_confidence": _restore_feature_shape(match_confidence, batch_frames),
        }


class PanoVGGTSkyMaskHead(nn.Module):
    """Sky-mask head attached to frozen PanoVGGT feature tensors."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        num_conv_blocks: int = 2,
        feature_key: str | int | None = None,
    ) -> None:
        super().__init__()
        if int(num_conv_blocks) <= 0:
            raise ValueError("num_conv_blocks must be positive.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_conv_blocks = int(num_conv_blocks)
        self.feature_key = feature_key
        blocks: list[nn.Module] = []
        in_dim = self.feature_dim
        for _ in range(self.num_conv_blocks):
            blocks.append(_ConvNormGELU(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.sky_proj = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return sky logits and probabilities for the input feature tensor."""

        selected = _select_feature(feature, self.feature_key)
        flat, batch_frames = _flatten_feature(selected)
        if int(flat.shape[1]) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {int(flat.shape[1])}.")
        hidden = self.trunk(flat)
        logits = self.sky_proj(hidden)
        return {
            "sky_logits": _restore_feature_shape(logits, batch_frames),
            "sky_prob": _restore_feature_shape(torch.sigmoid(logits), batch_frames),
        }


class PanoVGGTMatchingSkyHead(nn.Module):
    """Wrapper that can host separately trained matching and sky heads."""

    def __init__(
        self,
        feature_dim: int,
        *,
        descriptor_dim: int = 24,
        hidden_dim: int = 128,
        num_conv_blocks: int = 2,
        feature_key: str | int | None = None,
        train_matching: bool = True,
        train_sky: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_conv_blocks = int(num_conv_blocks)
        self.feature_key = feature_key
        self.train_matching = bool(train_matching)
        self.train_sky = bool(train_sky)
        self.matching_head = PanoVGGTMatchingHead(
            self.feature_dim,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
            num_conv_blocks=self.num_conv_blocks,
            feature_key=self.feature_key,
        )
        self.sky_head = PanoVGGTSkyMaskHead(
            self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_conv_blocks=self.num_conv_blocks,
            feature_key=self.feature_key,
        )

    def forward(self, feature: torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the union of enabled matching and sky predictions."""

        out: dict[str, torch.Tensor] = {}
        if self.train_matching:
            out.update(self.matching_head(feature))
        if self.train_sky:
            out.update(self.sky_head(feature))
        return out

    def head_config(self) -> dict[str, Any]:
        """Return serializable head construction metadata."""

        return {
            "feature_dim": self.feature_dim,
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "num_conv_blocks": self.num_conv_blocks,
            "feature_key": self.feature_key,
            "train_matching": self.train_matching,
            "train_sky": self.train_sky,
        }
