"""Render-error encoding and anchor backprojection for PanoAnchorSplat."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig
from .pano_resplat_geometry import project_world_to_erp_grid
from .resplat_types import PanoRenderOutput


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _grid_sample_wrap(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"values must have shape BxCxHxW, got {tuple(values.shape)}")
    gx = torch.remainder((grid[..., 0] + 1.0) * 0.5, 1.0) * 2.0 - 1.0
    gy = grid[..., 1].clamp(-1.0, 1.0)
    wrapped = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(values, wrapped, mode="bilinear", padding_mode="border", align_corners=True)[..., 0]


class PanoAnchorRenderErrorEncoder(nn.Module):
    """Encode RGB/depth/alpha residual maps and pool them to anchor tokens."""

    def __init__(self, config: PanoAnchorSplatConfig | dict | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.error_dim = int(self.config.error_dim)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(9, self.error_dim, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(max(1, min(8, self.error_dim)), self.error_dim),
            nn.GELU(),
            nn.Conv2d(self.error_dim, self.error_dim, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(max(1, min(8, self.error_dim)), self.error_dim),
            nn.GELU(),
        )
        self.anchor_fuse = nn.Sequential(
            nn.Linear(self.error_dim + 1, self.error_dim),
            nn.LayerNorm(self.error_dim),
            nn.GELU(),
            nn.Linear(self.error_dim, self.error_dim),
        )

    def forward(
        self,
        anchors: PanoAnchorSet,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        *,
        context_depth: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, v, _c, h, w = self._validate_context(anchors, context_images, context_poses_c2w)
        render_rgb, render_depth, render_alpha = self._unpack_render_output(context_render_output, b, v, h, w)
        target = torch.nan_to_num(context_images.to(device=anchors.centers.device, dtype=anchors.centers.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_rgb = torch.nan_to_num(render_rgb.to(device=target.device, dtype=target.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_depth = torch.nan_to_num(render_depth.to(device=target.device, dtype=target.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        render_alpha = torch.nan_to_num(render_alpha.to(device=target.device, dtype=target.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        rgb_residual = render_rgb - target
        abs_residual = rgb_residual.abs()
        if context_depth is None:
            depth_residual = torch.zeros(b, v, 1, h, w, device=target.device, dtype=target.dtype)
        else:
            depth_target = context_depth.to(device=target.device, dtype=target.dtype)
            if tuple(depth_target.shape) != (b, v, 1, h, w):
                raise ValueError(f"context_depth must have shape {(b, v, 1, h, w)}, got {tuple(context_depth.shape)}")
            depth_residual = ((render_depth - depth_target) / depth_target.abs().clamp_min(1.0)).clamp(-1.0, 1.0)
        if context_valid_mask is None:
            valid_maps = torch.ones(b, v, 1, h, w, device=target.device, dtype=target.dtype)
        else:
            valid_maps = context_valid_mask
            if valid_maps.ndim == 4:
                valid_maps = valid_maps.unsqueeze(2)
            if tuple(valid_maps.shape) != (b, v, 1, h, w):
                valid_maps = F.interpolate(valid_maps.float().reshape(b * v, 1, *valid_maps.shape[-2:]), size=(h, w), mode="nearest").reshape(b, v, 1, h, w)
            valid_maps = valid_maps.to(device=target.device, dtype=target.dtype)
        encoder_in = torch.cat([rgb_residual, abs_residual, render_alpha, depth_residual, valid_maps], dim=2)
        maps = self.map_encoder(encoder_in.reshape(b * v, 9, h, w)).reshape(b, v, self.error_dim, h, w)
        maps = maps * valid_maps
        pooled, valid_ratio = self._pool_to_anchors(anchors, maps, context_poses_c2w.to(device=target.device, dtype=target.dtype), valid_maps)
        tokens = self.anchor_fuse(torch.cat([pooled, valid_ratio], dim=-1))
        debug = {
            "mean_abs_residual": abs_residual.detach().mean(),
            "anchor_error_norm": pooled.detach().norm(dim=-1).mean(),
            "anchor_error_valid_ratio": valid_ratio.detach().mean(),
        }
        return torch.where(anchors.valid_mask.unsqueeze(-1), _finite(tokens), torch.zeros_like(tokens)), debug

    def _pool_to_anchors(
        self,
        anchors: PanoAnchorSet,
        maps: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, v, c, h, w = [int(x) for x in maps.shape]
        total = maps.new_zeros(b, anchors.num_anchors, c)
        count = maps.new_zeros(b, anchors.num_anchors, 1)
        for view_idx in range(v):
            projection = project_world_to_erp_grid(
                anchors.centers.to(device=maps.device, dtype=maps.dtype),
                poses_c2w[:, view_idx],
                (h, w),
                require_forward=False,
            )
            grid = projection.grid.view(b, anchors.num_anchors, 1, 2)
            sampled = _grid_sample_wrap(maps[:, view_idx], grid).transpose(1, 2)
            valid_sample = _grid_sample_wrap(valid_maps[:, view_idx], grid).transpose(1, 2)[..., 0] > 0.5
            valid = projection.mask.to(device=maps.device) & anchors.valid_mask.to(device=maps.device) & valid_sample
            weight = valid.unsqueeze(-1).to(dtype=maps.dtype)
            total = total + sampled * weight
            count = count + weight
        return _finite(total / count.clamp_min(1.0)), (count / float(max(v, 1))).clamp(0.0, 1.0)

    @staticmethod
    def _validate_context(
        anchors: PanoAnchorSet,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if context_images.ndim != 5 or int(context_images.shape[2]) != 3:
            raise ValueError(f"context_images must have shape BxVx3xHxW, got {tuple(context_images.shape)}")
        b, v, c, h, w = [int(x) for x in context_images.shape]
        if anchors.batch_size != b:
            raise ValueError("anchors and context_images must share batch size.")
        if context_poses_c2w.ndim != 4 or tuple(context_poses_c2w.shape) != (b, v, 4, 4):
            raise ValueError(f"context_poses_c2w must have shape {(b, v, 4, 4)}, got {tuple(context_poses_c2w.shape)}")
        return b, v, c, h, w

    @staticmethod
    def _unpack_render_output(
        output: PanoRenderOutput | dict[str, Any],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, PanoRenderOutput):
            color, depth, alpha = output.color, output.depth, output.alpha
        elif isinstance(output, dict):
            color = output.get("color", output.get("render"))
            depth = output.get("depth")
            alpha = output.get("alpha", output.get("opacity"))
        else:
            raise TypeError(f"Unsupported render output type: {type(output)!r}")
        if not torch.is_tensor(color):
            raise ValueError("render output must contain color/render tensor.")
        if color.ndim == 4:
            color = color.unsqueeze(1)
        if tuple(color.shape) != (b, v, 3, h, w):
            raise ValueError(f"render color must have shape {(b, v, 3, h, w)}, got {tuple(color.shape)}")
        if not torch.is_tensor(depth):
            depth = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if not torch.is_tensor(alpha):
            alpha = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif alpha.ndim == 4:
            alpha = alpha.unsqueeze(1)
        return color, depth, alpha
