"""Visualization helpers for Stage 2 Gaussian-head diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def _rgb_image(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().float().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _scalar_image(value: torch.Tensor, *, symmetric: bool = False) -> Image.Image:
    tensor = value.detach().float().cpu().squeeze()
    finite = torch.isfinite(tensor)
    if not bool(finite.any()):
        normalized = torch.zeros_like(tensor)
    elif symmetric:
        scale = tensor[finite].abs().quantile(0.99).clamp_min(1.0e-8)
        normalized = (tensor / (2.0 * scale) + 0.5).clamp(0.0, 1.0)
    else:
        low = tensor[finite].quantile(0.01)
        high = tensor[finite].quantile(0.99)
        normalized = ((tensor - low) / (high - low).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    red = normalized
    blue = 1.0 - normalized
    green = 1.0 - (normalized - 0.5).abs() * 2.0
    rgb = torch.stack([red, green.clamp(0.0, 1.0), blue], dim=-1)
    return Image.fromarray((rgb.numpy() * 255.0).round().astype(np.uint8), mode="RGB")


def save_stage2_render_panel(
    target: torch.Tensor,
    rendered: torch.Tensor,
    depth_residual: torch.Tensor,
    confidence: torch.Tensor,
    path: str | Path,
    *,
    title: str = "Stage 2 Gaussian head",
) -> Path:
    """Save target/render/error/depth/confidence as one compact PNG."""

    target_image = _rgb_image(target)
    render_image = _rgb_image(rendered)
    error_image = _scalar_image((target - rendered).abs().mean(dim=0))
    residual_image = _scalar_image(depth_residual, symmetric=True)
    confidence_image = _scalar_image(confidence)
    panels = [target_image, render_image, error_image, residual_image, confidence_image]
    labels = ["target", "all-source render", "RGB error", "depth residual", "confidence"]
    header = 28
    canvas = Image.new("RGB", (target_image.width * len(panels), target_image.height + header), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * target_image.width
        canvas.paste(panel, (x, header))
        draw.text((x + 5, 7), label, fill=(255, 255, 255))
    draw.text((5, target_image.height + header - 14), title, fill=(255, 255, 255))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
