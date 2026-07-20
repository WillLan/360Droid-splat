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
    photometric_mode: str = "charbonnier"
    rgb_l1_weight: float = 0.8
    dssim_weight: float = 0.2
    depth_loss_mode: str = "charbonnier"
    depth_residual_clamp: float = 0.20


def _dssim_loss(
    render_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    _, H, W = render_rgb.shape
    x = render_rgb.unsqueeze(0)
    y = target_rgb.unsqueeze(0)
    mask_f = None
    if mask is not None:
        mask_f = mask.to(device=render_rgb.device, dtype=render_rgb.dtype)
        if mask_f.ndim == 2:
            mask_f = mask_f.unsqueeze(0)
        x = x * mask_f.unsqueeze(0)
        y = y * mask_f.unsqueeze(0)
    kernel = 3
    padding = kernel // 2
    mu_x = torch.nn.functional.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = torch.nn.functional.avg_pool2d(y, kernel, stride=1, padding=padding)
    sigma_x = torch.nn.functional.avg_pool2d(x * x, kernel, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = torch.nn.functional.avg_pool2d(y * y, kernel, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1.0e-8)
    dssim = ((1.0 - ssim.clamp(-1.0, 1.0)) * 0.5).mean(dim=1, keepdim=True)[0]
    area = latitude_area_weight(H, W, device=render_rgb.device, dtype=render_rgb.dtype)
    weight = area if mask_f is None else area * mask_f
    return (dssim * weight).sum() / weight.sum().clamp_min(1.0e-8)


def pano_photometric_loss(
    render_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-3,
    mode: str = "charbonnier",
    rgb_l1_weight: float = 0.8,
    dssim_weight: float = 0.2,
) -> torch.Tensor:
    if render_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"RGB shape mismatch: {tuple(render_rgb.shape)} vs {tuple(target_rgb.shape)}"
        )
    _, H, W = render_rgb.shape
    area = latitude_area_weight(H, W, device=render_rgb.device, dtype=render_rgb.dtype)
    weight = area
    if mask is not None:
        weight = weight * mask.to(device=render_rgb.device, dtype=render_rgb.dtype)
    mode = str(mode or "charbonnier").lower()
    if mode in {"l1_dssim", "rgb_l1_dssim", "l1+ssim", "l1+dssim"}:
        l1 = (render_rgb - target_rgb).abs().mean(dim=0, keepdim=True)
        l1_loss = (l1 * weight).sum() / weight.sum().clamp_min(1e-8)
        dssim = _dssim_loss(render_rgb, target_rgb, mask=mask)
        return float(rgb_l1_weight) * l1_loss + float(dssim_weight) * dssim
    err = charbonnier(render_rgb - target_rgb, eps).mean(dim=0, keepdim=True)
    return (err * weight).sum() / weight.sum().clamp_min(1e-8)


def pano_depth_loss(
    render_depth: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps: float = 1e-3,
    mode: str = "charbonnier",
    residual_clamp: float = 0.20,
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
    if mask is not None:
        weight = weight * mask.to(device=render_depth.device, dtype=render_depth.dtype)
    if str(mode or "charbonnier").lower() in {"relative", "relative_clamped", "robust_relative"}:
        residual = (render_depth - target_depth) / torch.maximum(render_depth.abs(), target_depth.abs()).clamp_min(1.0e-6)
        if float(residual_clamp) > 0.0:
            residual = residual.clamp(min=-float(residual_clamp), max=float(residual_clamp))
        loss = charbonnier(residual, eps)
    else:
        loss = charbonnier(render_depth - target_depth, eps)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def backend_render_loss(
    render_pkg: dict,
    target_rgb: torch.Tensor,
    *,
    target_depth: torch.Tensor | None = None,
    depth_confidence: torch.Tensor | None = None,
    photometric_mask: torch.Tensor | None = None,
    depth_mask: torch.Tensor | None = None,
    weights: BackendLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BackendLossWeights()
    rgb = render_pkg["render"]
    total = rgb.new_tensor(0.0)
    photo = pano_photometric_loss(
        rgb,
        target_rgb,
        mask=photometric_mask,
        mode=weights.photometric_mode,
        rgb_l1_weight=weights.rgb_l1_weight,
        dssim_weight=weights.dssim_weight,
    )
    total = total + weights.photometric * photo

    depth = rgb.new_tensor(0.0)
    if target_depth is not None and render_pkg.get("depth") is not None:
        depth = pano_depth_loss(
            render_pkg["depth"],
            target_depth,
            confidence=depth_confidence,
            mask=depth_mask,
            mode=weights.depth_loss_mode,
            residual_clamp=weights.depth_residual_clamp,
        )
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
