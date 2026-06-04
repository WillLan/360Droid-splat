"""Feature-grid and ERP image-grid coordinate helpers for PanoVGGT-M3-Sphere."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def normalize_hw(value: Sequence[int] | torch.Size, *, name: str = "hw") -> tuple[int, int]:
    """Normalize a height/width-like value into a positive ``(H, W)`` tuple."""

    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {tuple(value)!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {(height, width)!r}")
    return height, width


def validate_feature_image_hw(
    feature_hw: Sequence[int] | torch.Size,
    image_hw: Sequence[int] | torch.Size,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validate feature and image grid sizes and return normalized tuples."""

    feature = normalize_hw(feature_hw, name="feature_hw")
    image = normalize_hw(image_hw, name="image_hw")
    return feature, image


def make_feature_grid(
    feature_hw: Sequence[int] | torch.Size,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a dense feature pixel-center grid with shape ``Hf x Wf x 2``."""

    height, width = normalize_hw(feature_hw, name="feature_hw")
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def feature_uv_to_image_uv(
    feature_uv: torch.Tensor,
    feature_hw: Sequence[int] | torch.Size,
    image_hw: Sequence[int] | torch.Size,
) -> torch.Tensor:
    """Map feature-grid UV coordinates to ERP image-grid UV coordinates."""

    (feature_h, feature_w), (image_h, image_w) = validate_feature_image_hw(feature_hw, image_hw)
    if feature_uv.shape[-1] != 2:
        raise ValueError(f"feature_uv must end with dimension 2, got {tuple(feature_uv.shape)}")
    scale = feature_uv.new_tensor([float(image_w) / float(feature_w), float(image_h) / float(feature_h)])
    return feature_uv * scale


def image_uv_to_feature_uv(
    image_uv: torch.Tensor,
    feature_hw: Sequence[int] | torch.Size,
    image_hw: Sequence[int] | torch.Size,
) -> torch.Tensor:
    """Map ERP image-grid UV coordinates to feature-grid UV coordinates."""

    (feature_h, feature_w), (image_h, image_w) = validate_feature_image_hw(feature_hw, image_hw)
    if image_uv.shape[-1] != 2:
        raise ValueError(f"image_uv must end with dimension 2, got {tuple(image_uv.shape)}")
    scale = image_uv.new_tensor([float(feature_w) / float(image_w), float(feature_h) / float(image_h)])
    return image_uv * scale
