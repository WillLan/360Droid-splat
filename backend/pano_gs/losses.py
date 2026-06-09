"""Losses shared by the panoramic Gaussian backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from frontend.pano_droid.spherical_camera import latitude_area_weight


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


@dataclass
class BackendLossWeights:
    photometric: float = 1.0
    depth: float = 0.1
    opacity: float = 0.01
    distortion: float = 0.0
    sky_alpha: float = 0.0


def pano_photometric_loss(
    render_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    if render_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"RGB shape mismatch: {tuple(render_rgb.shape)} vs {tuple(target_rgb.shape)}"
        )
    _, H, W = render_rgb.shape
    area = latitude_area_weight(H, W, device=render_rgb.device, dtype=render_rgb.dtype)
    err = charbonnier(render_rgb - target_rgb, eps).mean(dim=0, keepdim=True)
    weight = area
    if mask is not None:
        weight = weight * mask.to(device=render_rgb.device, dtype=render_rgb.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1e-8)


def pano_depth_loss(
    render_depth: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    if render_depth.shape != target_depth.shape:
        raise ValueError(
            f"Depth shape mismatch: {tuple(render_depth.shape)} vs {tuple(target_depth.shape)}"
        )
    _, H, W = render_depth.shape
    area = latitude_area_weight(H, W, device=render_depth.device, dtype=render_depth.dtype)
    weight = area
    if confidence is not None:
        weight = weight * confidence.to(device=render_depth.device, dtype=render_depth.dtype)
    return (charbonnier(render_depth - target_depth, eps) * weight).sum() / weight.sum().clamp_min(1e-8)


def backend_render_loss(
    render_pkg: dict,
    target_rgb: torch.Tensor,
    *,
    target_depth: torch.Tensor | None = None,
    depth_confidence: torch.Tensor | None = None,
    weights: BackendLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BackendLossWeights()
    rgb = render_pkg["render"]
    total = rgb.new_tensor(0.0)
    photo = pano_photometric_loss(rgb, target_rgb)
    total = total + weights.photometric * photo

    depth = rgb.new_tensor(0.0)
    if target_depth is not None and render_pkg.get("depth") is not None:
        depth = pano_depth_loss(render_pkg["depth"], target_depth, confidence=depth_confidence)
        total = total + weights.depth * depth

    opacity_reg = rgb.new_tensor(0.0)
    alpha = render_pkg.get("alpha")
    if alpha is not None:
        opacity_reg = charbonnier(alpha).mean()
        total = total + weights.opacity * opacity_reg

    sky_alpha = rgb.new_tensor(0.0)
    sky_mask = render_pkg.get("skybox_optimization_mask")
    if weights.sky_alpha > 0.0 and alpha is not None and torch.is_tensor(sky_mask):
        mask = sky_mask.to(device=rgb.device, dtype=rgb.dtype)
        if mask.shape != alpha.shape:
            mask = torch.nn.functional.interpolate(
                mask.reshape(1, 1, *mask.shape[-2:]),
                size=alpha.shape[-2:],
                mode="nearest",
            )[0]
        denom = mask.sum().clamp_min(1.0)
        sky_alpha = ((alpha.to(rgb) ** 2) * mask).sum() / denom
        total = total + weights.sky_alpha * sky_alpha

    distortion = rgb.new_tensor(0.0)
    render_distort = render_pkg.get("render_distort")
    if render_distort is not None:
        distortion = render_distort.mean()
        total = total + weights.distortion * distortion

    return total, {
        "loss": total.detach(),
        "photometric": photo.detach(),
        "depth": depth.detach(),
        "opacity": opacity_reg.detach(),
        "sky_alpha": sky_alpha.detach(),
        "distortion": distortion.detach(),
    }
