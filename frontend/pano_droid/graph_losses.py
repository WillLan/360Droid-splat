"""Losses for multi-frame DROID-style PanoDROID training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .spherical_ba import spherical_ba_loss
from .spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    latitude_area_weight,
    pixel_grid,
    seam_aware_delta,
)


def build_temporal_edges(n_frames: int, radius: int = 2, *, bidirectional: bool = True) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(int(n_frames)):
        for d in range(1, int(radius) + 1):
            j = i + d
            if j >= int(n_frames):
                continue
            edges.append((i, j))
            if bidirectional:
                edges.append((j, i))
    return edges


def select_training_edges(
    edges: list[tuple[int, int]],
    *,
    max_edges: int = 0,
) -> list[tuple[int, int]]:
    if max_edges and int(max_edges) > 0:
        return edges[: int(max_edges)]
    return edges


@dataclass
class GraphLossWeights:
    pose: float = 10.0
    flow: float = 0.05
    depth: float = 0.1
    residual: float = 0.01
    smooth: float = 0.02
    confidence: float = 0.005


def _coords_for_loss(
    height: int,
    width: int,
    *,
    sample_height: int,
    sample_width: int,
    device,
    dtype,
) -> torch.Tensor:
    y = torch.linspace(0.5, float(height) - 0.5, int(sample_height), device=device, dtype=dtype)
    x = torch.linspace(0.5, float(width) - 0.5, int(sample_width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _sample_chw_map(x: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Sample ``B x C x H x W`` at ``N x 2`` pixel centers -> ``B x C x N``."""
    B, C, H, W = x.shape
    coords = pixels.to(device=x.device, dtype=x.dtype)
    norm_x = 2.0 * (coords[:, 0] - 0.5) / max(W - 1, 1) - 1.0
    norm_y = 2.0 * (coords[:, 1] - 0.5) / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2).expand(B, -1, -1, -1)
    sampled = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="border")
    return sampled.squeeze(-1)


def _relative_pose_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


def _pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    R = pred[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2)
    trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    rot = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    trans = (pred[:, :3, 3] - target[:, :3, 3]).abs().mean(dim=-1)
    return (rot + trans).mean()


def _smoothness(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx_wrap = x[..., :, :1] - x[..., :, -1:]
    return torch.sqrt(dx * dx + 1e-6).mean() + torch.sqrt(dy * dy + 1e-6).mean() + torch.sqrt(dx_wrap * dx_wrap + 1e-6).mean()


def spherical_projective_flow(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    c2w_i: torch.Tensor,
    c2w_j: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source pixels/depth from camera i into camera j."""
    B, N = depth.shape
    p = pixels.to(device=depth.device, dtype=depth.dtype).unsqueeze(0).expand(B, -1, -1)
    bearing_i = erp_pixel_to_bearing(p, height, width)
    xyz_i = bearing_i * depth.clamp_min(1e-6).unsqueeze(-1)
    ones = torch.ones(B, N, 1, device=depth.device, dtype=depth.dtype)
    xyz_i_h = torch.cat([xyz_i, ones], dim=-1)
    world = torch.einsum("bij,bnj->bni", c2w_i, xyz_i_h)[..., :3]
    world_h = torch.cat([world, ones], dim=-1)
    cam_j = torch.einsum("bij,bnj->bni", torch.linalg.inv(c2w_j), world_h)[..., :3]
    bearing_j = F.normalize(cam_j, dim=-1, eps=1e-12)
    target_pixels = bearing_to_erp_pixel(bearing_j, height, width)
    flow = seam_aware_delta(p, target_pixels, width)
    return flow, target_pixels


def graph_supervised_loss(
    batch: dict,
    pred: dict,
    *,
    weights: GraphLossWeights | None = None,
    sample_height: int = 32,
    sample_width: int = 64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or GraphLossWeights()
    images = batch["images"].to(pred["spherical_flow"].device)
    depths = batch["depths"].to(pred["spherical_flow"].device)
    poses = batch["poses_c2w"].to(pred["spherical_flow"].device)
    edges = pred["edges"]
    B, _, _, H, W = images.shape
    pixels = _coords_for_loss(
        H,
        W,
        sample_height=sample_height,
        sample_width=sample_width,
        device=images.device,
        dtype=images.dtype,
    )
    area = latitude_area_weight(H, W, device=images.device, dtype=images.dtype, normalize=True)
    point_area = _sample_chw_map(area.expand(B, -1, -1, -1), pixels).squeeze(1)

    total = images.new_tensor(0.0)
    l_pose = images.new_tensor(0.0)
    l_flow = images.new_tensor(0.0)
    l_depth = images.new_tensor(0.0)
    l_residual = images.new_tensor(0.0)
    l_smooth = images.new_tensor(0.0)
    l_conf = images.new_tensor(0.0)
    valid_edges = 0

    for e, (i, j) in enumerate(edges):
        pred_flow = pred["spherical_flow"][:, e]
        pred_inv = pred["inverse_depth"][:, e]
        pred_conf = pred["confidence"][:, e].clamp(1e-4, 1.0)
        gt_depth = _sample_chw_map(depths[:, i], pixels).squeeze(1)
        valid = gt_depth > 1e-6
        if not bool(valid.any()):
            continue
        gt_flow, gt_target_pixels = spherical_projective_flow(
            pixels,
            gt_depth,
            poses[:, i],
            poses[:, j],
            height=H,
            width=W,
        )
        pred_flow_s = _sample_chw_map(pred_flow, pixels).permute(0, 2, 1)
        pred_inv_s = _sample_chw_map(pred_inv, pixels).squeeze(1)
        pred_conf_s = _sample_chw_map(pred_conf, pixels).squeeze(1)
        flow_err = seam_aware_delta(gt_flow, pred_flow_s, W)
        edge_weight = point_area * valid.to(point_area) * pred_conf_s.detach()
        l_flow = l_flow + (torch.sqrt((flow_err * flow_err).sum(dim=-1) + 1e-6) * edge_weight).sum() / edge_weight.sum().clamp_min(1e-8)
        gt_inv = torch.zeros_like(gt_depth)
        gt_inv[valid] = 1.0 / gt_depth[valid].clamp_min(1e-6)
        l_depth = l_depth + (torch.sqrt((pred_inv_s - gt_inv) ** 2 + 1e-6) * edge_weight).sum() / edge_weight.sum().clamp_min(1e-8)
        gt_rel = _relative_pose_from_c2w(poses[:, i], poses[:, j])
        l_pose = l_pose + _pose_loss(pred["relative_pose"][:, e], gt_rel)
        ba_out = spherical_ba_loss(
            pixels.unsqueeze(0).expand(B, -1, -1),
            pred_inv_s.clamp_min(1e-6),
            pred["relative_pose"][:, e],
            height=H,
            width=W,
            target_pixels=gt_target_pixels,
            confidence=pred_conf_s,
        )
        l_residual = l_residual + ba_out.loss
        l_smooth = l_smooth + _smoothness(pred_flow) + 0.25 * _smoothness(pred_inv)
        l_conf = l_conf + (-torch.log(pred_conf)).mean()
        valid_edges += 1

    denom = max(valid_edges, 1)
    l_pose = l_pose / denom
    l_flow = l_flow / denom
    l_depth = l_depth / denom
    l_residual = l_residual / denom
    l_smooth = l_smooth / denom
    l_conf = l_conf / denom
    total = (
        weights.pose * l_pose
        + weights.flow * l_flow
        + weights.depth * l_depth
        + weights.residual * l_residual
        + weights.smooth * l_smooth
        + weights.confidence * l_conf
    )
    return total, {
        "loss": total.detach(),
        "pose": l_pose.detach(),
        "flow": l_flow.detach(),
        "depth": l_depth.detach(),
        "residual": l_residual.detach(),
        "smooth": l_smooth.detach(),
        "confidence": l_conf.detach(),
        "valid_edges": torch.tensor(float(valid_edges), device=images.device),
    }

