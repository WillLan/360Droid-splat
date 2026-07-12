"""Visualization helper for Stage 3 BA/refiner snapshot renders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def _to_image(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().float().clamp(0.0, 1.0).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_stage3_snapshot_panel(
    path: str | Path,
    *,
    target: torch.Tensor,
    rendered_snapshots: dict[str, torch.Tensor],
    batch_index: int = 0,
    view_index: int = 0,
) -> Path:
    """Save GT followed by Initial/BA/Refine renders for one source view."""

    order = ["initial", "ba0", "refine1", "ba1", "refine2", "ba2", "refine3"]
    entries: list[tuple[str, Image.Image]] = [("GT", _to_image(target[batch_index, view_index]))]
    for name in order:
        if name in rendered_snapshots:
            entries.append((name, _to_image(rendered_snapshots[name][batch_index, view_index])) )
    label_height = 24
    width = sum(image.width for _, image in entries)
    height = max(image.height for _, image in entries) + label_height
    panel = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    offset = 0
    for label, image in entries:
        panel.paste(image, (offset, label_height))
        draw.text((offset + 4, 4), label, fill=(255, 255, 255))
        offset += image.width
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    return output

