"""Losses and image metrics for Stage 2 ERP Gaussian-head training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from geometry.spherical_erp import sample_erp_with_wrap
from geometry.spherical_pseudo_correspondence import generate_spherical_pseudo_correspondence
from models.per_pixel_gaussian_observation import PerPixelGaussianObservation


def latitude_area_weights(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized ERP pixel-area weights with shape ``1xHxW``."""

    rows = torch.arange(int(height), device=device, dtype=dtype) + 0.5
    latitude = math.pi * (rows / float(height) - 0.5)
    weights = torch.cos(latitude).clamp_min(0.0).view(1, int(height), 1)
    weights = weights.expand(1, int(height), int(width))
    return weights / weights.mean().clamp_min(torch.finfo(dtype).eps)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = torch.broadcast_to(weight, value.shape)
    return (value * expanded).sum() / expanded.sum().clamp_min(torch.finfo(value.dtype).eps)


def spherical_weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have identical CxHxW shapes.")
    weight = latitude_area_weights(
        prediction.shape[-2], prediction.shape[-1], device=prediction.device, dtype=prediction.dtype
    )
    if valid_mask is not None:
        mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        weight = weight * mask
    return _weighted_mean((prediction - target).abs(), weight)


def _erp_box_filter(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    radius = int(kernel_size) // 2
    padded = F.pad(value, (radius, radius, 0, 0), mode="circular")
    padded = F.pad(padded, (0, 0, radius, radius), mode="replicate")
    channels = int(value.shape[1])
    kernel = torch.ones(
        channels,
        1,
        int(kernel_size),
        int(kernel_size),
        device=value.device,
        dtype=value.dtype,
    ) / float(int(kernel_size) ** 2)
    return F.conv2d(padded, kernel, groups=channels)


def periodic_ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Return a channel-averaged ERP SSIM map with a periodic longitude seam."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have identical BxCxHxW shapes.")
    if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    mu_x = _erp_box_filter(prediction, int(kernel_size))
    mu_y = _erp_box_filter(target, int(kernel_size))
    sigma_x = _erp_box_filter(prediction.square(), int(kernel_size)) - mu_x.square()
    sigma_y = _erp_box_filter(target.square(), int(kernel_size)) - mu_y.square()
    sigma_xy = _erp_box_filter(prediction * target, int(kernel_size)) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean(dim=1, keepdim=True)


def spherical_dssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must be CxHxW.")
    ssim = periodic_ssim_map(prediction.unsqueeze(0), target.unsqueeze(0))[0]
    weight = latitude_area_weights(ssim.shape[-2], ssim.shape[-1], device=ssim.device, dtype=ssim.dtype)
    return _weighted_mean((1.0 - ssim) * 0.5, weight)


def spherical_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = latitude_area_weights(
        prediction.shape[-2], prediction.shape[-1], device=prediction.device, dtype=prediction.dtype
    )
    mse = _weighted_mean((prediction - target).square(), weight)
    return -10.0 * torch.log10(mse.clamp_min(1.0e-10))


@dataclass(frozen=True)
class Stage2GaussianLossWeights:
    rgb: float = 1.0
    depth_residual: float = 1.0e-3
    dssim: float = 0.0
    rendered_depth: float = 0.0
    geometry: float = 0.0


def spherical_pseudo_geometry_consistency_loss(
    observation: PerPixelGaussianObservation,
    *,
    batch_index: int,
    num_query_per_pair: int = 512,
    min_depth: float = 0.05,
    max_depth: float = 100.0,
    visibility_rel_thresh: float = 0.05,
) -> torch.Tensor:
    """Match refined world points along frozen Stage 1 pseudo-correspondences."""

    batch = int(batch_index)
    height, width = observation.image_size
    initial = observation.initial_depth[batch].detach().float()
    poses = observation.poses_c2w[batch].detach().float()
    with torch.no_grad():
        correspondence = generate_spherical_pseudo_correspondence(
            initial,
            poses,
            height=height,
            width=width,
            num_query_per_pair=int(num_query_per_pair),
            sampling="fibonacci_depth_filtered",
            min_depth=float(min_depth),
            max_depth=float(max_depth),
            visibility_rel_thresh=float(visibility_rel_thresh),
        )
    if correspondence.valid_mask.numel() == 0 or not bool(correspondence.valid_mask.any()):
        return observation.refined_depth[batch].sum() * 0.0
    src_view = correspondence.src_view[:, 0].long()
    tgt_view = correspondence.tgt_view[:, 0].long()
    refined = observation.refined_depth[batch].float()
    source_depth = sample_erp_with_wrap(refined.index_select(0, src_view), correspondence.src_uv)[..., 0]
    target_depth = sample_erp_with_wrap(refined.index_select(0, tgt_view), correspondence.tgt_uv)[..., 0]
    source_ray = correspondence.src_ray.to(source_depth)
    target_ray = correspondence.tgt_ray.to(target_depth)
    source_pose = observation.poses_c2w[batch].index_select(0, src_view).float()
    target_pose = observation.poses_c2w[batch].index_select(0, tgt_view).float()
    source_camera = source_depth.unsqueeze(-1) * source_ray
    target_camera = target_depth.unsqueeze(-1) * target_ray
    source_world = torch.einsum("eij,eqj->eqi", source_pose[:, :3, :3], source_camera) + source_pose[:, None, :3, 3]
    target_world = torch.einsum("eij,eqj->eqi", target_pose[:, :3, :3], target_camera) + target_pose[:, None, :3, 3]
    normalized_distance = torch.linalg.norm(source_world - target_world, dim=-1) / (
        0.5 * (source_depth + target_depth)
    ).clamp_min(1.0e-4)
    weight = correspondence.weight.to(normalized_distance) * correspondence.valid_mask.to(normalized_distance)
    return (F.smooth_l1_loss(normalized_distance, torch.zeros_like(normalized_distance), reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)


def stage2_gaussian_render_loss(
    render_packages: Iterable[dict[str, Any]],
    target_images: torch.Tensor,
    observation: PerPixelGaussianObservation,
    *,
    batch_index: int = 0,
    target_depths: torch.Tensor | None = None,
    geometry_loss: torch.Tensor | None = None,
    weights: Stage2GaussianLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate sequential source-view reconstruction losses for one batch item."""

    cfg = weights or Stage2GaussianLossWeights()
    packages = list(render_packages)
    if target_images.ndim != 4 or int(target_images.shape[0]) != len(packages):
        raise ValueError("target_images must have shape Sx3xHxW and match render_packages.")
    zero = target_images.sum() * 0.0
    rgb_terms: list[torch.Tensor] = []
    dssim_terms: list[torch.Tensor] = []
    depth_terms: list[torch.Tensor] = []
    psnr_terms: list[torch.Tensor] = []
    for view, package in enumerate(packages):
        rendered = package.get("render")
        if not torch.is_tensor(rendered):
            raise ValueError("Every renderer package must contain a tensor 'render'.")
        target = target_images[view].to(device=rendered.device, dtype=rendered.dtype)
        rgb_terms.append(spherical_weighted_l1(rendered, target))
        psnr_terms.append(spherical_psnr(rendered.detach(), target.detach()))
        if float(cfg.dssim) != 0.0:
            dssim_terms.append(spherical_dssim(rendered, target))
        if float(cfg.rendered_depth) != 0.0:
            if target_depths is None:
                raise ValueError("target_depths are required when rendered_depth weight is non-zero.")
            rendered_depth = package.get("depth")
            if not torch.is_tensor(rendered_depth):
                raise ValueError("Renderer package is missing tensor 'depth'.")
            target_depth = target_depths[view].to(device=rendered_depth.device, dtype=rendered_depth.dtype)
            if target_depth.ndim == 2:
                target_depth = target_depth.unsqueeze(0)
            alpha = package.get("alpha")
            valid = torch.isfinite(target_depth) & (target_depth > 0.0)
            if torch.is_tensor(alpha):
                valid = valid & (alpha > 1.0e-4)
            relative = (rendered_depth - target_depth).abs() / target_depth.clamp_min(1.0e-4)
            area = latitude_area_weights(
                relative.shape[-2], relative.shape[-1], device=relative.device, dtype=relative.dtype
            )
            depth_terms.append(_weighted_mean(relative, area * valid.to(relative.dtype)))

    rgb = torch.stack(rgb_terms).mean() if rgb_terms else zero
    dssim = torch.stack(dssim_terms).mean() if dssim_terms else zero
    rendered_depth_loss = torch.stack(depth_terms).mean() if depth_terms else zero
    initial = observation.initial_depth[int(batch_index)]
    residual = observation.depth_residual[int(batch_index)]
    relative_residual = residual / initial.clamp_min(1.0e-4)
    valid = observation.valid_mask[int(batch_index)].to(relative_residual.dtype)
    residual_regularizer = F.smooth_l1_loss(
        relative_residual * valid,
        torch.zeros_like(relative_residual),
        reduction="sum",
    ) / valid.sum().clamp_min(1.0)
    if float(cfg.geometry) != 0.0 and geometry_loss is None:
        raise ValueError("geometry_loss is required when geometry weight is non-zero.")
    geometry = geometry_loss if geometry_loss is not None else zero
    total = (
        float(cfg.rgb) * rgb
        + float(cfg.depth_residual) * residual_regularizer
        + float(cfg.dssim) * dssim
        + float(cfg.rendered_depth) * rendered_depth_loss
        + float(cfg.geometry) * geometry
    )
    return {
        "loss": total,
        "rgb_l1": rgb,
        "depth_residual": residual_regularizer,
        "dssim": dssim,
        "rendered_depth": rendered_depth_loss,
        "geometry": geometry,
        "psnr": torch.stack(psnr_terms).mean() if psnr_terms else zero,
    }
