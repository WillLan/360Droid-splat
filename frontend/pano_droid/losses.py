"""Training losses for PanoDROID-MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .spherical_camera import latitude_area_weight, seam_aware_delta


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def erp_flow_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample ``image`` at ``pixel + flow`` with horizontal wrap-around."""
    B, C, H, W = image.shape
    y = torch.arange(H, device=image.device, dtype=image.dtype) + 0.5
    x = torch.arange(W, device=image.device, dtype=image.dtype) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid_u = torch.remainder(xx.unsqueeze(0) + flow[:, 0], float(W))
    grid_v = (yy.unsqueeze(0) + flow[:, 1]).clamp(0.5, float(H) - 0.5)
    # grid_sample uses corner coordinates.  Convert pixel centers to [-1, 1].
    norm_u = 2.0 * (grid_u - 0.5) / max(W - 1, 1) - 1.0
    norm_v = 2.0 * (grid_v - 0.5) / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_u, norm_v], dim=-1)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _smoothness(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx_wrap = x[..., :, :1] - x[..., :, -1:]
    return charbonnier(dx).mean() + charbonnier(dy).mean() + charbonnier(dx_wrap).mean()


def _pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    R = pred[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2)
    tr = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    rot = torch.acos(((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    trans = (pred[:, :3, 3] - target[:, :3, 3]).abs().mean(dim=-1)
    return (rot + trans).mean()


@dataclass
class LossWeights:
    photometric: float = 1.0
    flow: float = 0.2
    depth: float = 0.1
    pose: float = 0.1
    smooth: float = 0.02
    confidence: float = 0.005


class PanoDroidLoss(nn.Module):
    """Composite loss supporting supervised and self-supervised signals."""

    def __init__(self, weights: Optional[LossWeights] = None, eps: float = 1e-3) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.eps = float(eps)

    def forward(self, batch: dict, pred: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image0 = batch["image0"].to(pred["spherical_flow"].device)
        image1 = batch["image1"].to(pred["spherical_flow"].device)
        flow = pred["spherical_flow"]
        inv_depth = pred["inverse_depth"]
        conf = pred["confidence"].clamp(1e-4, 1.0)
        _, _, H, W = image0.shape
        area = latitude_area_weight(H, W, device=image0.device, dtype=image0.dtype)
        warped = erp_flow_warp(image1, flow)
        photo_map = charbonnier(warped - image0, self.eps).mean(dim=1, keepdim=True)
        l_photo = (photo_map * area * conf.detach()).mean()

        l_flow = image0.new_tensor(0.0)
        if "gt_flow" in batch and batch["gt_flow"] is not None:
            gt_flow = batch["gt_flow"].to(flow.device, dtype=flow.dtype)
            err = seam_aware_delta(gt_flow.permute(0, 2, 3, 1), flow.permute(0, 2, 3, 1), W)
            err = err.permute(0, 3, 1, 2)
            l_flow = (charbonnier(err, self.eps).mean(dim=1, keepdim=True) * area).mean()

        l_depth = image0.new_tensor(0.0)
        if "gt_inverse_depth" in batch and batch["gt_inverse_depth"] is not None:
            gt_inv = batch["gt_inverse_depth"].to(inv_depth.device, dtype=inv_depth.dtype)
            l_depth = (charbonnier(inv_depth - gt_inv, self.eps) * area).mean()

        l_pose = image0.new_tensor(0.0)
        if "gt_relative_pose" in batch and batch["gt_relative_pose"] is not None:
            gt_pose = batch["gt_relative_pose"].to(pred["relative_pose"].device, dtype=pred["relative_pose"].dtype)
            l_pose = _pose_loss(pred["relative_pose"], gt_pose)

        l_smooth = _smoothness(flow) + 0.25 * _smoothness(inv_depth)
        l_conf = (-torch.log(conf)).mean()
        total = (
            self.weights.photometric * l_photo
            + self.weights.flow * l_flow
            + self.weights.depth * l_depth
            + self.weights.pose * l_pose
            + self.weights.smooth * l_smooth
            + self.weights.confidence * l_conf
        )
        metrics = {
            "loss": total.detach(),
            "photometric": l_photo.detach(),
            "flow": l_flow.detach(),
            "depth": l_depth.detach(),
            "pose": l_pose.detach(),
            "smooth": l_smooth.detach(),
            "confidence": l_conf.detach(),
        }
        return total, metrics

