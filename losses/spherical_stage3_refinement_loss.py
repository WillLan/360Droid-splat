"""Losses and geometry diagnostics for Stage 3 BA/refiner training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from frontend.pano_vggt.spherical_correspondence import spherical_tangent_residual
from geometry.spherical_erp import sample_erp_with_wrap
from models.per_pixel_gaussian_observation import PerPixelGaussianObservation
from models.spherical_selfi_stage3_ba import Stage3MatchCache
from .spherical_gaussian_render_loss import spherical_dssim, spherical_weighted_l1


@dataclass(frozen=True)
class Stage3LossWeights:
    dssim: float = 0.2
    geometry: float = 0.05
    depth_anchor: float = 1.0e-3
    update_regularization: float = 1.0e-4


def all_source_render_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    dssim_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if rendered.ndim != 5 or tuple(rendered.shape) != tuple(target.shape):
        raise ValueError("rendered and target must have matching BxSx3xHxW shapes.")
    batch, views = int(rendered.shape[0]), int(rendered.shape[1])
    l1_values, dssim_values = [], []
    compute_dssim = float(dssim_weight) != 0.0
    for batch_idx in range(batch):
        for view in range(views):
            l1_values.append(spherical_weighted_l1(rendered[batch_idx, view], target[batch_idx, view]))
            if compute_dssim:
                dssim_values.append(spherical_dssim(rendered[batch_idx, view], target[batch_idx, view]))
    l1 = torch.stack(l1_values).mean()
    dssim = torch.stack(dssim_values).mean() if dssim_values else l1.new_zeros(())
    return l1 + float(dssim_weight) * dssim, {"l1": l1.detach(), "dssim": dssim.detach()}


def leave_one_out_render_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    dssim_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compatibility alias for checkpoints/tests using the historical name."""

    return all_source_render_loss(rendered, target, dssim_weight=dssim_weight)


def spherical_match_geometry_loss(
    observation: PerPixelGaussianObservation,
    cache: Stage3MatchCache,
) -> torch.Tensor:
    """Differentiable S2 consistency at the cached adapter query matches."""

    batch, views = observation.batch_size, observation.num_source_views
    height, width = observation.image_size
    if cache.batch_size != batch or cache.num_views != views:
        raise ValueError("Observation and match cache dimensions do not match.")
    sampled_depth = sample_erp_with_wrap(observation.refined_depth, cache.source_uv)[..., 0]
    losses: list[torch.Tensor] = []
    for edge_idx, pair in enumerate(cache.edges.tolist()):
        src, tgt = int(pair[0]), int(pair[1])
        depth = sampled_depth[:, src]
        ray = cache.source_ray[:, src].to(depth)
        point_source = depth[..., None] * ray
        src_pose = observation.poses_c2w[:, src].to(point_source)
        tgt_pose = observation.poses_c2w[:, tgt].to(point_source)
        point_world = torch.einsum("bij,bqj->bqi", src_pose[:, :3, :3], point_source) + src_pose[:, None, :3, 3]
        point_target = torch.einsum(
            "bij,bqj->bqi",
            tgt_pose[:, :3, :3].transpose(1, 2),
            point_world - tgt_pose[:, None, :3, 3],
        )
        predicted = F.normalize(point_target, dim=-1, eps=1.0e-8)
        residual = spherical_tangent_residual(cache.target_ray[:, edge_idx].to(predicted), predicted).norm(dim=-1)
        mask = cache.valid_mask[:, edge_idx].to(device=residual.device)
        if mask.any():
            losses.append(residual[mask].mean())
    if not losses:
        return observation.refined_depth.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def build_ba_support_map(
    cache: Stage3MatchCache,
    *,
    height: int,
    width: int,
    floor: float = 0.1,
    dilation_kernel: int = 5,
) -> torch.Tensor:
    support = torch.zeros(
        cache.batch_size,
        cache.num_views,
        1,
        int(height),
        int(width),
        device=cache.source_uv.device,
        dtype=torch.float32,
    )
    x = torch.floor(cache.source_uv[..., 0]).long().remainder(int(width))
    y = torch.floor(cache.source_uv[..., 1]).long().clamp(0, int(height) - 1)
    for batch in range(cache.batch_size):
        for view in range(cache.num_views):
            valid = cache.source_valid[batch, view]
            support[batch, view, 0, y[batch, view, valid], x[batch, view, valid]] = 1.0
    kernel = max(1, int(dilation_kernel))
    if kernel > 1:
        flat = support.reshape(cache.batch_size * cache.num_views, 1, int(height), int(width))
        pad = kernel // 2
        flat = F.pad(flat, (pad, pad, 0, 0), mode="circular")
        flat = F.pad(flat, (0, 0, pad, pad), mode="replicate")
        support = F.max_pool2d(flat, kernel_size=kernel, stride=1).reshape_as(support)
    return float(floor) + (1.0 - float(floor)) * support


def ba_depth_anchor_loss(
    refined_depth: torch.Tensor,
    ba_depth: torch.Tensor,
    support: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    relative = (refined_depth - ba_depth) / ba_depth.clamp_min(1.0e-6)
    valid = valid_mask.bool() & torch.isfinite(relative)
    if not valid.any():
        return refined_depth.sum() * 0.0
    per_pixel = F.smooth_l1_loss(relative, torch.zeros_like(relative), reduction="none")
    weight = support.to(per_pixel) * valid.to(per_pixel.dtype)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


def stage3_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    observation: PerPixelGaussianObservation,
    cache: Stage3MatchCache,
    ba_depth: torch.Tensor,
    support: torch.Tensor,
    update_energy: torch.Tensor,
    *,
    anchor_prediction: torch.Tensor | None = None,
    anchor_valid_mask: torch.Tensor | None = None,
    weights: Stage3LossWeights = Stage3LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    render, render_parts = all_source_render_loss(rendered, target, dssim_weight=weights.dssim)
    geometry = spherical_match_geometry_loss(observation, cache)
    anchor = ba_depth_anchor_loss(
        observation.refined_depth if anchor_prediction is None else anchor_prediction,
        ba_depth,
        support,
        observation.valid_mask if anchor_valid_mask is None else anchor_valid_mask,
    )
    total = (
        render
        + float(weights.geometry) * geometry
        + float(weights.depth_anchor) * anchor
        + float(weights.update_regularization) * update_energy
    )
    return total, {
        "loss": total.detach(),
        "render": render.detach(),
        "l1": render_parts["l1"],
        "dssim": render_parts["dssim"],
        "geometry_rad": geometry.detach(),
        "geometry_deg": torch.rad2deg(geometry.detach()),
        "depth_anchor": anchor.detach(),
        "update_regularization": update_energy.detach(),
    }


def aligned_pose_metrics(predicted: torch.Tensor, ground_truth: torch.Tensor) -> dict[str, float]:
    """First-frame rotation/translation alignment plus positive scale fit."""

    if predicted.shape != ground_truth.shape or predicted.ndim != 3 or predicted.shape[-2:] != (4, 4):
        raise ValueError("Pose metrics expect matching Sx4x4 tensors.")
    pred, gt = predicted.detach().float(), ground_truth.detach().float()
    align_rotation = gt[0, :3, :3] @ pred[0, :3, :3].transpose(0, 1)
    pred_centers = (align_rotation @ (pred[:, :3, 3] - pred[0, :3, 3]).transpose(0, 1)).transpose(0, 1)
    gt_centers = gt[:, :3, 3] - gt[0, :3, 3]
    denominator = pred_centers.square().sum().clamp_min(1.0e-8)
    scale = (pred_centers * gt_centers).sum() / denominator
    scale = scale.clamp_min(1.0e-8)
    aligned_centers = scale * pred_centers + gt[0, :3, 3]
    aligned_rotation = align_rotation.unsqueeze(0) @ pred[:, :3, :3]
    rotation_delta = gt[:, :3, :3].transpose(1, 2) @ aligned_rotation
    trace = rotation_delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_deg = torch.rad2deg(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
    ate = (aligned_centers - gt[:, :3, 3]).norm(dim=-1)
    if int(pred.shape[0]) > 1:
        pred_relative = torch.linalg.inv(pred[:-1]) @ pred[1:]
        gt_relative = torch.linalg.inv(gt[:-1]) @ gt[1:]
        relative_rotation = gt_relative[:, :3, :3].transpose(1, 2) @ pred_relative[:, :3, :3]
        relative_trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        rpe_rotation = torch.rad2deg(torch.acos(((relative_trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
        pred_delta = aligned_centers[1:] - aligned_centers[:-1]
        gt_delta = gt[1:, :3, 3] - gt[:-1, :3, 3]
        rpe_translation = (pred_delta - gt_delta).norm(dim=-1)
        direction_valid = (pred_delta.norm(dim=-1) > 1.0e-8) & (gt_delta.norm(dim=-1) > 1.0e-8)
        if direction_valid.any():
            direction_dot = (
                F.normalize(pred_delta[direction_valid], dim=-1)
                * F.normalize(gt_delta[direction_valid], dim=-1)
            ).sum(dim=-1)
            direction_deg = torch.rad2deg(torch.acos(direction_dot.clamp(-1.0, 1.0))).mean()
        else:
            direction_deg = pred.new_tensor(0.0)
    else:
        rpe_rotation = pred.new_zeros(1)
        rpe_translation = pred.new_zeros(1)
        direction_deg = pred.new_tensor(0.0)
    return {
        "rotation_mean_deg": float(rotation_deg.mean().cpu()),
        "rotation_median_deg": float(rotation_deg.median().cpu()),
        "rotation_p90_deg": float(rotation_deg.quantile(0.9).cpu()),
        "scale_aligned_ate": float(ate.square().mean().sqrt().cpu()),
        "alignment_scale": float(scale.cpu()),
        "rpe_rotation_mean_deg": float(rpe_rotation.mean().cpu()),
        "rpe_translation": float(rpe_translation.square().mean().sqrt().cpu()),
        "translation_direction_mean_deg": float(direction_deg.cpu()),
    }


def depth_metrics(predicted: torch.Tensor, ground_truth: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    mask = valid.bool() & torch.isfinite(predicted) & torch.isfinite(ground_truth) & (ground_truth > 0)
    if not mask.any():
        return {}
    pred, gt = predicted[mask].float(), ground_truth[mask].float()
    ratio_scale = (gt.median() / pred.median().clamp_min(1.0e-8)).clamp_min(1.0e-8)

    def values(value: torch.Tensor, prefix: str) -> dict[str, float]:
        relative = (value - gt).abs() / gt.clamp_min(1.0e-8)
        rmse = (value - gt).square().mean().sqrt()
        ratio = torch.maximum(value / gt.clamp_min(1.0e-8), gt / value.clamp_min(1.0e-8))
        return {
            f"{prefix}absrel": float(relative.mean().cpu()),
            f"{prefix}rmse": float(rmse.cpu()),
            f"{prefix}delta1": float((ratio < 1.25).float().mean().cpu()),
        }

    result = values(pred, "raw_")
    result.update(values(pred * ratio_scale, "scale_aligned_"))
    return result
