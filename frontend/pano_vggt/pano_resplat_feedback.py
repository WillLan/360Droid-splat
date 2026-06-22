"""Context-view render feedback for Pano-ReSplat refinement."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .pano_resplat_geometry import project_world_to_erp_grid
from .resplat_types import PanoGaussianState, PanoRenderOutput


class PanoRenderFeedbackEncoder(nn.Module):
    """Encode per-Gaussian feedback from context view render residuals."""

    def __init__(self, feedback_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = 14
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feedback_dim),
        )

    def forward(
        self,
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        context_depth: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if context_images.ndim != 5 or int(context_images.shape[2]) != 3:
            raise ValueError(f"context_images must have shape BxVx3xHxW, got {tuple(context_images.shape)}")
        if context_poses_c2w.ndim != 4 or tuple(context_poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"context_poses_c2w must have shape BxVx4x4, got {tuple(context_poses_c2w.shape)}")
        b, v, _, h, w = [int(x) for x in context_images.shape]
        if state.batch_size != b:
            raise ValueError("state and context_images must share batch size.")
        if tuple(context_poses_c2w.shape[:2]) != (b, v):
            raise ValueError("context_poses_c2w must share B,V with context_images.")

        render_rgb, render_depth, render_alpha = self._unpack_render_output(context_render_output, b, v, h, w)
        target = torch.nan_to_num(context_images.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_rgb = torch.nan_to_num(render_rgb.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_depth = torch.nan_to_num(render_depth.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        render_alpha = torch.nan_to_num(render_alpha.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        depth_target = None
        if context_depth is not None:
            if context_depth.ndim != 5 or tuple(context_depth.shape[:2]) != (b, v):
                raise ValueError(f"context_depth must have shape BxVx1xHxW, got {tuple(context_depth.shape)}")
            depth_target = torch.nan_to_num(context_depth.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        valid_maps = None
        if context_valid_mask is not None:
            valid_maps = self._normalize_context_mask(context_valid_mask, b, v, h, w).to(device=state.means.device, dtype=state.means.dtype)

        per_view = []
        weights = []
        valid_ratios = []
        alpha_means = []
        residual_maps = render_rgb - target
        abs_maps = residual_maps.abs()
        for view_idx in range(v):
            projection = project_world_to_erp_grid(
                state.means,
                context_poses_c2w[:, view_idx].to(device=state.means.device, dtype=state.means.dtype),
                (h, w),
            )
            grid = projection.grid.view(b, state.num_gaussians, 1, 2)
            residual = self._sample_map(residual_maps[:, view_idx], grid).transpose(1, 2)
            abs_residual = self._sample_map(abs_maps[:, view_idx], grid).transpose(1, 2)
            alpha = self._sample_map(render_alpha[:, view_idx], grid).transpose(1, 2)
            depth_render = self._sample_map(render_depth[:, view_idx], grid).transpose(1, 2)
            if valid_maps is None:
                valid_sample = torch.ones_like(alpha[..., 0], dtype=torch.bool)
            else:
                valid_sample = self._sample_map(valid_maps[:, view_idx], grid).transpose(1, 2)[..., 0] > 0.5
            if depth_target is None:
                depth_error = torch.zeros_like(depth_render)
            else:
                depth_gt = self._sample_map(depth_target[:, view_idx], grid).transpose(1, 2)
                depth_error = ((depth_render - depth_gt) / depth_gt.abs().clamp_min(1.0)).clamp(-1.0, 1.0)
            camera_center = context_poses_c2w[:, view_idx, :3, 3].to(device=state.means.device, dtype=state.means.dtype)
            view_dir = F.normalize(state.means - camera_center[:, None, :], dim=-1, eps=1.0e-6)
            valid = projection.mask & state.valid_mask & valid_sample
            weight = valid.unsqueeze(-1).to(dtype=state.means.dtype) * alpha.clamp(0.0, 1.0)
            feat = torch.cat(
                [
                    residual,
                    abs_residual,
                    alpha,
                    depth_error,
                    depth_render / depth_render.detach().abs().mean().clamp_min(1.0),
                    valid.unsqueeze(-1).to(dtype=state.means.dtype),
                    view_dir,
                ],
                dim=-1,
            )
            per_view.append(feat)
            weights.append(weight)
            valid_ratios.append(valid.to(dtype=state.means.dtype).mean())
            alpha_means.append(alpha.mean())

        stacked = torch.stack(per_view, dim=2)
        weight_t = torch.stack(weights, dim=2)
        denom = weight_t.sum(dim=2).clamp_min(1.0e-6)
        raw = (stacked * weight_t).sum(dim=2) / denom
        raw = torch.cat([raw, (weight_t.sum(dim=2) / float(max(v, 1))).clamp(0.0, 1.0)], dim=-1)
        feedback = self.encoder(torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
        debug = {
            "valid_projection_ratio": torch.stack(valid_ratios).mean().detach(),
            "mean_alpha": torch.stack(alpha_means).mean().detach(),
            "mean_abs_residual": abs_maps.mean().detach(),
            "feedback_weight_mean": weight_t.mean().detach(),
        }
        return torch.nan_to_num(feedback, nan=0.0, posinf=0.0, neginf=0.0), debug

    @staticmethod
    def _sample_map(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True)[..., 0]

    @staticmethod
    def _normalize_context_mask(mask: torch.Tensor, b: int, v: int, h: int, w: int) -> torch.Tensor:
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        if mask.ndim != 5 or tuple(mask.shape[:2]) != (b, v) or int(mask.shape[2]) != 1:
            raise ValueError(f"context_valid_mask must have shape BxVxHxW or BxVx1xHxW, got {tuple(mask.shape)}")
        if tuple(mask.shape[-2:]) != (h, w):
            mask = F.interpolate(mask.float(), size=(h, w), mode="nearest")
        return mask.float()

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
            raise TypeError(f"Unsupported context_render_output type: {type(output)!r}")
        if not torch.is_tensor(color):
            raise ValueError("context_render_output must contain color/render tensor.")
        if color.ndim == 4:
            color = color.unsqueeze(1)
        if color.ndim != 5 or tuple(color.shape[:2]) != (b, v):
            raise ValueError(f"render color must have shape BxVx3xHxW, got {tuple(color.shape)}")
        if not torch.is_tensor(depth):
            depth = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if not torch.is_tensor(alpha):
            alpha = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif alpha.ndim == 4:
            alpha = alpha.unsqueeze(1)
        return color, depth, alpha
