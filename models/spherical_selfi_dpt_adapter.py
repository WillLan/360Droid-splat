"""Spherical Selfi DPT-style adapter for Stage 1B."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from geometry.spherical_erp import DEFAULT_ERP_HEIGHT, DEFAULT_ERP_WIDTH
from .panovggt_feature_wrapper import normalize_stage_feature


def _num_groups(channels: int) -> int:
    groups = min(8, int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


class _ERPConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        use_circular_padding: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.use_circular_padding = bool(use_circular_padding)
        padding = 0 if self.kernel_size > 1 and self.use_circular_padding else self.kernel_size // 2
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=self.kernel_size, padding=padding)
        groups = _num_groups(int(out_channels))
        self.norm = nn.GroupNorm(groups, int(out_channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size > 1 and self.use_circular_padding:
            pad = self.kernel_size // 2
            x = F.pad(x, (pad, pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        return self.act(self.norm(self.conv(x)))


class _FusionBlock(nn.Module):
    def __init__(self, channels: int, *, use_circular_padding: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _ERPConvNormAct(channels, channels, use_circular_padding=use_circular_padding),
            _ERPConvNormAct(channels, channels, use_circular_padding=use_circular_padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SphericalSelfiDPTAdapter(nn.Module):
    """Fuse 4-stage PanoVGGT features into full-resolution ERP descriptors."""

    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...],
        *,
        hidden_dim: int = 128,
        out_dim: int = 24,
        image_height: int = DEFAULT_ERP_HEIGHT,
        image_width: int = DEFAULT_ERP_WIDTH,
        use_circular_padding: bool = True,
        norm_output: bool = True,
        token_hw: list[tuple[int, int] | None] | tuple[tuple[int, int] | None, ...] | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"SphericalSelfiDPTAdapter requires 4 input stages, got {len(in_channels)}.")
        if int(out_dim) <= 0:
            raise ValueError("out_dim must be positive.")
        if int(image_height) <= 0 or int(image_width) <= 0:
            raise ValueError("image_height and image_width must be positive.")
        self.in_channels = [int(value) for value in in_channels]
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.use_circular_padding = bool(use_circular_padding)
        self.norm_output = bool(norm_output)
        self.token_hw = list(token_hw) if token_hw is not None else [None] * 4
        if len(self.token_hw) != 4:
            raise ValueError("token_hw must contain 4 entries when provided.")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(_num_groups(self.hidden_dim), self.hidden_dim),
                    nn.GELU(),
                )
                for channels in self.in_channels
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [_FusionBlock(self.hidden_dim, use_circular_padding=self.use_circular_padding) for _ in range(4)]
        )
        self.output_refine = _FusionBlock(self.hidden_dim, use_circular_padding=self.use_circular_padding)
        self.output_proj = nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1)

    def _infer_batch_views(self, stage_features: list[Any], batch_size: int | None, num_views: int | None) -> tuple[int, int]:
        first_tensor = None
        for value in stage_features:
            if torch.is_tensor(value):
                first_tensor = value
                break
            if isinstance(value, (list, tuple)):
                tensors = [item for item in value if torch.is_tensor(item)]
                if tensors:
                    first_tensor = tensors[-1]
                    break
            if isinstance(value, dict):
                tensors = [item for item in value.values() if torch.is_tensor(item)]
                if tensors:
                    first_tensor = tensors[-1]
                    break
        if first_tensor is None:
            raise ValueError("Could not infer batch/view dimensions because no tensor feature was provided.")
        if first_tensor.ndim == 5:
            return int(first_tensor.shape[0]), int(first_tensor.shape[1])
        if first_tensor.ndim == 4 and int(first_tensor.shape[-1]) == self.in_channels[0]:
            return int(first_tensor.shape[0]), int(first_tensor.shape[1])
        if first_tensor.ndim == 4 and batch_size is None and num_views is None:
            return 1, int(first_tensor.shape[0])
        if batch_size is None or num_views is None:
            raise ValueError("batch_size and num_views are required for flattened B*V features or B*V tokens.")
        if int(batch_size) * int(num_views) != int(first_tensor.shape[0]):
            raise ValueError(
                f"batch_size*num_views={int(batch_size) * int(num_views)} does not match feature leading dimension "
                f"{int(first_tensor.shape[0])}."
            )
        return int(batch_size), int(num_views)

    def _normalize_inputs(
        self,
        stage_features: list[Any] | tuple[Any, ...],
        *,
        batch_size: int | None,
        num_views: int | None,
    ) -> list[torch.Tensor]:
        if len(stage_features) != 4:
            raise ValueError(f"Expected 4 stage features, got {len(stage_features)}.")
        b, v = self._infer_batch_views(list(stage_features), batch_size, num_views)
        normalized = [
            normalize_stage_feature(
                feature,
                batch_size=b,
                num_views=v,
                image_hw=(self.image_height, self.image_width),
                token_hw=self.token_hw[idx],
            )
            for idx, feature in enumerate(stage_features)
        ]
        for idx, (feature, channels) in enumerate(zip(normalized, self.in_channels)):
            if int(feature.shape[2]) != int(channels):
                raise ValueError(
                    f"Stage {idx} expected {channels} channels, got {int(feature.shape[2])} "
                    f"from shape {tuple(feature.shape)}."
                )
        return normalized

    @staticmethod
    def _flatten_bv(feature: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, v, c, h, w = (int(dim) for dim in feature.shape)
        return feature.reshape(b * v, c, h, w), (b, v)

    @staticmethod
    def _restore_bv(feature: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        b, v = shape
        return feature.view(b, v, int(feature.shape[1]), int(feature.shape[2]), int(feature.shape[3]))

    def forward(
        self,
        stage_features: list[Any] | tuple[Any, ...],
        *,
        batch_size: int | None = None,
        num_views: int | None = None,
    ) -> torch.Tensor:
        """Return dense spherical features with shape ``B x V x out_dim x H x W``."""

        normalized = self._normalize_inputs(stage_features, batch_size=batch_size, num_views=num_views)
        projected: list[torch.Tensor] = []
        restore_shape: tuple[int, int] | None = None
        for feature, projection in zip(normalized, self.projections):
            flat, restore_shape = self._flatten_bv(feature)
            projected.append(projection(flat))
        assert restore_shape is not None

        fused = projected[-1]
        fused = self.fusion_blocks[-1](fused)
        for idx in range(2, -1, -1):
            fused = F.interpolate(fused, size=projected[idx].shape[-2:], mode="bilinear", align_corners=False)
            fused = self.fusion_blocks[idx](fused + projected[idx])
        fused = F.interpolate(
            fused,
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        )
        fused = self.output_refine(fused)
        dense = self.output_proj(fused)
        out = self._restore_bv(dense, restore_shape)
        if self.norm_output:
            out = F.normalize(out, dim=2, eps=1.0e-6)
        return out
