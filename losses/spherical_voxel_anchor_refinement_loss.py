"""Minimal losses for the simplified Stage-3 voxel-anchor refiner."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from losses.spherical_gaussian_render_loss import spherical_dssim, spherical_weighted_l1


@dataclass(frozen=True)
class VoxelAnchorLossWeights:
    dssim: float = 0.0
    depth: float = 0.05
    alpha_hole: float = 0.05
    update_regularization: float = 1.0e-4


def _latitude_weights(reference: torch.Tensor) -> torch.Tensor:
    height = int(reference.shape[-2])
    rows = torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5
    latitude = math.pi * (rows / float(height) - 0.5)
    return torch.cos(latitude).clamp_min(0.0).view(1, 1, 1, height, 1)


def spherical_voxel_anchor_loss(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    ba0_depth: torch.Tensor,
    target_valid: torch.Tensor,
    normalized_update_energy: torch.Tensor,
    *,
    weights: VoxelAnchorLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RGB, optional DSSIM, BA0-relative depth, holes, and update energy."""

    if tuple(rendered_rgb.shape) != tuple(target_rgb.shape):
        raise ValueError("rendered_rgb and target_rgb must have identical BxSx3xHxW shapes")
    expected_scalar = (*rendered_rgb.shape[:2], 1, *rendered_rgb.shape[-2:])
    for name, value in (
        ("rendered_depth", rendered_depth),
        ("rendered_alpha", rendered_alpha),
        ("ba0_depth", ba0_depth),
        ("target_valid", target_valid),
    ):
        if tuple(value.shape) != expected_scalar:
            raise ValueError(f"{name} must have shape {expected_scalar}")

    rgb_terms = []
    dssim_terms = []
    for batch_index in range(int(rendered_rgb.shape[0])):
        for view_index in range(int(rendered_rgb.shape[1])):
            rgb_terms.append(
                spherical_weighted_l1(
                    rendered_rgb[batch_index, view_index],
                    target_rgb[batch_index, view_index],
                )
            )
            if float(weights.dssim) > 0.0:
                dssim_terms.append(
                    spherical_dssim(
                        rendered_rgb[batch_index, view_index],
                        target_rgb[batch_index, view_index],
                    )
                )
    rgb_l1 = torch.stack(rgb_terms).mean()
    dssim = (
        torch.stack(dssim_terms).mean()
        if dssim_terms
        else rendered_rgb.sum() * 0.0
    )

    finite = (
        target_valid.bool()
        & torch.isfinite(rendered_depth)
        & torch.isfinite(ba0_depth)
        & (ba0_depth > 0.0)
        & (rendered_alpha > 0.0)
    )
    area = _latitude_weights(rendered_depth)
    depth_weight = finite.to(rendered_depth.dtype) * area
    relative_depth = (rendered_depth - ba0_depth).abs() / ba0_depth.clamp_min(1.0e-6)
    depth = (relative_depth * depth_weight).sum() / depth_weight.sum().clamp_min(1.0)

    hole_weight = target_valid.to(rendered_alpha.dtype) * area
    alpha_hole = ((1.0 - rendered_alpha).clamp_min(0.0) * hole_weight).sum() / (
        hole_weight.sum().clamp_min(1.0)
    )
    total = (
        rgb_l1
        + float(weights.dssim) * dssim
        + float(weights.depth) * depth
        + float(weights.alpha_hole) * alpha_hole
        + float(weights.update_regularization) * normalized_update_energy
    )
    return total, {
        "loss": total,
        "rgb_l1": rgb_l1,
        "dssim": dssim,
        "relative_ba0_depth": depth,
        "alpha_hole": alpha_hole,
        "update_regularization": normalized_update_energy,
    }
