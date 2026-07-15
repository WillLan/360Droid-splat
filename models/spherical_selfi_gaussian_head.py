"""Selfi-style spherical U-Net for dense per-pixel Gaussian prediction."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from geometry.spherical_erp import build_erp_ray_grid
from .per_pixel_gaussian_observation import (
    PerPixelGaussianObservation,
    normalize_quaternion,
    real_sh_basis,
)


def _num_groups(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


class ERPConv2d(nn.Module):
    """Conv2d with circular longitude and replicated latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=self.kernel_size,
            stride=int(stride),
            padding=0,
            bias=bool(bias),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        if pad > 0:
            value = F.pad(value, (pad, pad, 0, 0), mode="circular")
            value = F.pad(value, (0, 0, pad, pad), mode="replicate")
        return self.conv(value)


def erp_bilinear_resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize BCHW data with periodic horizontal interpolation."""

    if value.ndim != 4:
        raise ValueError("erp_bilinear_resize expects a BCHW tensor.")
    target_height, target_width = int(size[0]), int(size[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError("ERP resize dimensions must be positive.")
    source_height, source_width = int(value.shape[-2]), int(value.shape[-1])
    if (source_height, source_width) == (target_height, target_width):
        return value
    # Keeping the width unchanged makes this interpolation purely vertical.
    vertical = F.interpolate(
        value,
        size=(target_height, source_width),
        mode="bilinear",
        align_corners=False,
    )
    if source_width == target_width:
        return vertical
    position = (
        (torch.arange(target_width, device=value.device, dtype=torch.float32) + 0.5)
        * (float(source_width) / float(target_width))
        - 0.5
    )
    left_float = torch.floor(position)
    blend = (position - left_float).to(dtype=value.dtype).view(1, 1, 1, target_width)
    left = left_float.long().remainder(source_width)
    right = (left + 1).remainder(source_width)
    return vertical.index_select(-1, left) * (1.0 - blend) + vertical.index_select(-1, right) * blend


class _ERPConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv = ERPConv2d(in_channels, out_channels, stride=stride, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), int(out_channels))
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(value)))


class _ERPDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _ERPConvNormAct(in_channels, out_channels),
            _ERPConvNormAct(out_channels, out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class _EncoderLevel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = _ERPConvNormAct(in_channels, out_channels, stride=2)
        self.refine = _ERPDoubleConv(out_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(value))


class _DecoderLevel(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_project = _ERPConvNormAct(in_channels, out_channels)
        self.fuse = _ERPDoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = erp_bilinear_resize(value, (int(skip.shape[-2]), int(skip.shape[-1])))
        value = self.up_project(value)
        return self.fuse(torch.cat([value, skip], dim=1))


class SphericalSelfiGaussianHead(nn.Module):
    """Decode dense aligned descriptors and RGB into one Gaussian per ERP pixel."""

    def __init__(
        self,
        *,
        feature_dim: int = 24,
        channels: tuple[int, int, int, int] | list[int] = (32, 64, 128, 256),
        mlp_hidden_dim: int = 64,
        rgb_sh_degree: int = 2,
        density_sh_degree: int = 1,
        depth_residual_ratio: float = 0.25,
        initial_opacity: float = 0.10,
        min_depth: float = 1.0e-4,
        min_scale: float = 1.0e-5,
        max_scale_ratio: float = 0.25,
        latitude_cos_min: float = 1.0e-3,
        log_scale_clamp: float = 5.0,
        render_prune_fraction: float = 0.30,
        gradient_checkpointing: bool = False,
        renderer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive.")
        if len(channels) != 4 or any(int(value) <= 0 for value in channels):
            raise ValueError("channels must contain four positive values.")
        if int(rgb_sh_degree) not in {0, 1, 2} or int(density_sh_degree) not in {0, 1, 2}:
            raise ValueError("RGB and density SH degrees must be 0, 1, or 2.")
        if not 0.0 < float(depth_residual_ratio) < 1.0:
            raise ValueError("depth_residual_ratio must be in (0, 1).")
        if not 0.0 < float(initial_opacity) < 1.0:
            raise ValueError("initial_opacity must be in (0, 1).")
        self.feature_dim = int(feature_dim)
        self.channels = tuple(int(value) for value in channels)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.rgb_sh_degree = int(rgb_sh_degree)
        self.density_sh_degree = int(density_sh_degree)
        self.rgb_sh_count = (self.rgb_sh_degree + 1) ** 2
        self.density_sh_count = (self.density_sh_degree + 1) ** 2
        self.depth_residual_ratio = float(depth_residual_ratio)
        self.initial_opacity = float(initial_opacity)
        self.min_depth = float(min_depth)
        self.min_scale = float(min_scale)
        self.max_scale_ratio = float(max_scale_ratio)
        self.latitude_cos_min = float(latitude_cos_min)
        self.log_scale_clamp = float(log_scale_clamp)
        self.render_prune_fraction = float(render_prune_fraction)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.renderer_config = renderer_config

        c0, c1, c2, c3 = self.channels
        self.stem = _ERPDoubleConv(self.feature_dim + 3, c0)
        self.encoder = nn.ModuleList(
            [_EncoderLevel(c0, c1), _EncoderLevel(c1, c2), _EncoderLevel(c2, c3)]
        )
        self.decoder = nn.ModuleList(
            [_DecoderLevel(c3, c2, c2), _DecoderLevel(c2, c1, c1), _DecoderLevel(c1, c0, c0)]
        )
        self.quaternion_head = ERPConv2d(c0, 4)
        self.depth_head = ERPConv2d(c0, 1)
        self.parameter_mlp = nn.Sequential(nn.Conv2d(c0, self.mlp_hidden_dim, kernel_size=1), nn.GELU())
        self.scale_head = nn.Conv2d(self.mlp_hidden_dim, 3, kernel_size=1)
        self.rgb_sh_head = nn.Conv2d(self.mlp_hidden_dim, self.rgb_sh_count * 3, kernel_size=1)
        self.density_sh_head = nn.Conv2d(self.mlp_hidden_dim, self.density_sh_count, kernel_size=1)
        self._initialize_prediction_heads()

    @property
    def raw_output_channels(self) -> int:
        return 4 + 1 + 3 + self.rgb_sh_count * 3 + self.density_sh_count

    def head_config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "channels": list(self.channels),
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "rgb_sh_degree": self.rgb_sh_degree,
            "density_sh_degree": self.density_sh_degree,
            "depth_residual_ratio": self.depth_residual_ratio,
            "initial_opacity": self.initial_opacity,
            "min_depth": self.min_depth,
            "min_scale": self.min_scale,
            "max_scale_ratio": self.max_scale_ratio,
            "latitude_cos_min": self.latitude_cos_min,
            "log_scale_clamp": self.log_scale_clamp,
            "render_prune_fraction": self.render_prune_fraction,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def _initialize_prediction_heads(self) -> None:
        with torch.no_grad():
            self.quaternion_head.conv.weight.zero_()
            self.quaternion_head.conv.bias.zero_()
            self.quaternion_head.conv.bias[0] = 1.0
            self.depth_head.conv.weight.zero_()
            self.depth_head.conv.bias.zero_()
            self.scale_head.weight.zero_()
            self.scale_head.bias.zero_()
            self.rgb_sh_head.weight.zero_()
            self.rgb_sh_head.bias.zero_()
            self.density_sh_head.weight.zero_()
            self.density_sh_head.bias.zero_()
            density_logit = math.log(self.initial_opacity / (1.0 - self.initial_opacity))
            self.density_sh_head.bias[0] = density_logit / 0.28209479177387814

    def _run(self, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return gradient_checkpoint(module, value, use_reentrant=False)
        return module(value)

    def _run_decoder(self, module: nn.Module, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return gradient_checkpoint(module, value, skip, use_reentrant=False)
        return module(value, skip)

    def _decode_flat(
        self, flat_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode an independent flattened B*V batch into raw Gaussian fields."""

        skips = [self._run(self.stem, flat_input)]
        for level in self.encoder:
            skips.append(self._run(level, skips[-1]))
        decoded = skips[-1]
        for level, skip in zip(self.decoder, reversed(skips[:-1])):
            decoded = self._run_decoder(level, decoded, skip)
        parameters = self.parameter_mlp(decoded)
        return (
            self.quaternion_head(decoded),
            self.depth_head(decoded),
            self.scale_head(parameters),
            self.rgb_sh_head(parameters),
            self.density_sh_head(parameters),
        )

    @staticmethod
    def _normalize_depth(depth: torch.Tensor, *, batch: int, views: int) -> torch.Tensor:
        value = depth
        if value.ndim == 5 and int(value.shape[-1]) == 1:
            value = value.permute(0, 1, 4, 2, 3)
        elif value.ndim == 4 and int(value.shape[0]) == batch and int(value.shape[1]) == views:
            value = value.unsqueeze(2)
        if value.ndim != 5 or tuple(value.shape[:3]) != (batch, views, 1):
            raise ValueError(f"initial_depth must normalize to BxSx1xHxW, got {tuple(depth.shape)}.")
        return value

    def forward(
        self,
        adapter_features: torch.Tensor,
        images: torch.Tensor,
        initial_depth: torch.Tensor,
        poses_c2w: torch.Tensor,
        *,
        frame_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        flat_batch_chunk_size: int | None = None,
    ) -> PerPixelGaussianObservation:
        if adapter_features.ndim != 5:
            raise ValueError("adapter_features must have shape BxSxCxHxW.")
        if images.ndim != 5 or int(images.shape[2]) != 3:
            raise ValueError("images must have shape BxSx3xHxW.")
        batch, views, channels, height, width = (int(value) for value in adapter_features.shape)
        if channels != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {channels}.")
        if tuple(images.shape) != (batch, views, 3, height, width):
            raise ValueError("images and adapter_features must share B/S/H/W dimensions.")
        if tuple(poses_c2w.shape) != (batch, views, 4, 4):
            raise ValueError(f"poses_c2w must have shape {(batch, views, 4, 4)}.")
        depth = self._normalize_depth(initial_depth, batch=batch, views=views).to(
            device=adapter_features.device,
            dtype=adapter_features.dtype,
        )
        if tuple(depth.shape[-2:]) != (height, width):
            depth = erp_bilinear_resize(
                depth.reshape(batch * views, 1, *depth.shape[-2:]),
                (height, width),
            ).reshape(batch, views, 1, height, width)
        images = images.to(device=adapter_features.device, dtype=adapter_features.dtype).clamp(0.0, 1.0)
        features = torch.nan_to_num(adapter_features, nan=0.0, posinf=0.0, neginf=0.0)
        flat_input = torch.cat(
            [features.reshape(batch * views, channels, height, width), images.reshape(batch * views, 3, height, width)],
            dim=1,
        )
        flat_count = int(flat_input.shape[0])
        chunk_size = flat_count if flat_batch_chunk_size is None else int(flat_batch_chunk_size)
        if chunk_size <= 0:
            chunk_size = flat_count
        decoded_chunks = [
            self._decode_flat(flat_input[start : start + chunk_size])
            for start in range(0, flat_count, chunk_size)
        ]
        quaternion_raw, depth_raw, log_scale_raw, rgb_sh_raw, density_sh_raw = (
            torch.cat([chunk[index] for chunk in decoded_chunks], dim=0)
            for index in range(5)
        )
        log_scale = log_scale_raw.reshape(batch, views, 3, height, width)
        rgb_sh = rgb_sh_raw.reshape(batch, views, self.rgb_sh_count, 3, height, width)
        density_sh = density_sh_raw.reshape(batch, views, self.density_sh_count, height, width)
        quaternion = normalize_quaternion(
            quaternion_raw.reshape(batch, views, 4, height, width).permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3)

        safe_initial = torch.where(
            torch.isfinite(depth) & (depth > self.min_depth),
            depth,
            torch.ones_like(depth),
        )
        depth_residual = safe_initial * self.depth_residual_ratio * torch.tanh(
            depth_raw.reshape(batch, views, 1, height, width)
        )
        refined_depth = safe_initial + depth_residual
        pose_valid = torch.isfinite(poses_c2w).all(dim=-1).all(dim=-1).view(batch, views, 1, 1, 1)
        finite_valid = torch.isfinite(depth) & (depth > self.min_depth) & pose_valid
        if valid_mask is not None:
            mask = valid_mask
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            if mask.ndim != 5 or tuple(mask.shape[:3]) != (batch, views, 1):
                raise ValueError("valid_mask must have shape BxSx1xHxW or BxSxHxW.")
            if tuple(mask.shape[-2:]) != (height, width):
                mask = F.interpolate(
                    mask.float().reshape(batch * views, 1, *mask.shape[-2:]),
                    size=(height, width),
                    mode="nearest",
                ).reshape(batch, views, 1, height, width) > 0.5
            finite_valid = finite_valid & mask.bool().to(device=finite_valid.device)

        ray = build_erp_ray_grid(height, width, device=features.device, dtype=features.dtype)
        ys = torch.arange(height, device=features.device, dtype=features.dtype) + 0.5
        xs = torch.arange(width, device=features.device, dtype=features.dtype) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        uv = torch.stack([xx, yy], dim=-1)
        source_basis = real_sh_basis(self.density_sh_degree, ray)
        density_values = density_sh.permute(0, 1, 3, 4, 2)
        confidence = torch.sigmoid((density_values * source_basis.view(1, 1, height, width, -1)).sum(dim=-1))
        confidence = confidence.unsqueeze(2)
        confidence = torch.where(finite_valid, confidence, torch.zeros_like(confidence))
        if frame_ids is None:
            frame_ids = torch.arange(views, device=features.device, dtype=torch.long).view(1, views).expand(batch, -1)
        else:
            frame_ids = frame_ids.to(device=features.device, dtype=torch.long)

        return PerPixelGaussianObservation(
            initial_depth=safe_initial,
            depth_residual=depth_residual,
            refined_depth=refined_depth,
            poses_c2w=poses_c2w.to(device=features.device, dtype=torch.float32),
            local_quaternion=quaternion,
            log_scale_multiplier=log_scale,
            rgb_sh=rgb_sh,
            density_sh=density_sh,
            confidence=confidence,
            valid_mask=finite_valid,
            source_uv=uv,
            source_ray=ray,
            frame_ids=frame_ids,
            rgb_sh_degree=self.rgb_sh_degree,
            density_sh_degree=self.density_sh_degree,
            min_scale=self.min_scale,
            max_scale_ratio=self.max_scale_ratio,
            latitude_cos_min=self.latitude_cos_min,
            log_scale_clamp=self.log_scale_clamp,
            render_prune_fraction=self.render_prune_fraction,
            config=self.renderer_config,
        )
