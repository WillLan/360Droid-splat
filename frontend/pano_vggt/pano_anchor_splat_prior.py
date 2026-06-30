"""Offline PanoVGGT prior adapter for PanoAnchorSplat."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .pano_anchor_splat_types import PanoAnchorSplatPrior


def _first_present(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _maybe_first_batch_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"images must have shape BxVx3xHxW or Vx3xHxW, got {tuple(images.shape)}")
    if int(images.shape[2]) != 3:
        raise ValueError(f"images must have 3 channels, got {tuple(images.shape)}")
    return torch.nan_to_num(images.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        features = features.unsqueeze(0)
    if features.ndim != 5:
        raise ValueError(f"features must have shape BxVxCxHfxWf or VxCxHfxWf, got {tuple(features.shape)}")
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_depths(depths: torch.Tensor, *, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    if depths.ndim == 3:
        depths = depths.unsqueeze(0).unsqueeze(2)
    elif depths.ndim == 4:
        if int(depths.shape[-3]) == 1:
            depths = depths.unsqueeze(0)
        else:
            depths = depths.unsqueeze(2)
    if depths.ndim != 5:
        raise ValueError(f"depths must normalize to BxVx1xHxW, got {tuple(depths.shape)}")
    if int(depths.shape[2]) != 1 or tuple(depths.shape[-2:]) != (h, w):
        raise ValueError(f"depths must have shape BxVx1x{h}x{w}, got {tuple(depths.shape)}")
    return torch.nan_to_num(depths.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _normalize_world_points(world_points: torch.Tensor, *, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    if world_points.ndim == 4:
        world_points = world_points.unsqueeze(0)
    if world_points.ndim != 5 or int(world_points.shape[-1]) != 3:
        raise ValueError(f"world_points must have shape BxVxHxWx3 or VxHxWx3, got {tuple(world_points.shape)}")
    if tuple(world_points.shape[-3:-1]) != (h, w):
        raise ValueError(f"world_points must use image size {(h, w)}, got {tuple(world_points.shape)}")
    return torch.nan_to_num(world_points.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_poses(poses: torch.Tensor, *, batch_views: tuple[int, int]) -> torch.Tensor:
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    if poses.ndim != 4 or tuple(poses.shape[-2:]) != (4, 4):
        raise ValueError(f"poses_c2w must have shape BxVx4x4 or Vx4x4, got {tuple(poses.shape)}")
    if tuple(poses.shape[:2]) != batch_views:
        raise ValueError(f"poses_c2w must share B,V={batch_views}, got {tuple(poses.shape)}")
    return torch.nan_to_num(poses.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_mask(mask: torch.Tensor | None, *, batch_views: tuple[int, int], image_hw: tuple[int, int], default: bool) -> torch.Tensor:
    b, v = batch_views
    h, w = image_hw
    if mask is None:
        return torch.full((b, v, 1, h, w), bool(default), dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(2)
    elif mask.ndim == 4:
        if tuple(mask.shape[:2]) == (b, v):
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(0)
    if mask.ndim != 5 or tuple(mask.shape[:2]) != (b, v) or int(mask.shape[2]) != 1 or tuple(mask.shape[-2:]) != (h, w):
        raise ValueError(f"mask must normalize to {(b, v, 1, h, w)}, got {tuple(mask.shape)}")
    return mask.bool()


def _normalize_confidence(confidence: torch.Tensor | None, *, depths: torch.Tensor) -> torch.Tensor:
    if confidence is None:
        return (depths > 0.0).to(dtype=depths.dtype)
    if confidence.ndim == 3:
        confidence = confidence.unsqueeze(0).unsqueeze(2)
    elif confidence.ndim == 4:
        if int(confidence.shape[-3]) == 1:
            confidence = confidence.unsqueeze(0)
        else:
            confidence = confidence.unsqueeze(2)
    if tuple(confidence.shape) != tuple(depths.shape):
        raise ValueError(f"confidence must have shape {tuple(depths.shape)}, got {tuple(confidence.shape)}")
    return torch.nan_to_num(confidence.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def prior_from_mapping(mapping: Mapping[str, Any]) -> PanoAnchorSplatPrior:
    """Build a detached ``PanoAnchorSplatPrior`` from a sample/cache mapping."""

    images = _first_present(mapping, ("images", "rgb", "source_images"))
    features = _first_present(mapping, ("features", "feature_maps", "pano_features"))
    depths = _first_present(mapping, ("depths", "depth", "inverse_depths"))
    poses = _first_present(mapping, ("poses_c2w", "pose_c2w", "poses"))
    world_points = _first_present(mapping, ("world_points", "chunk_world_points", "point_maps"))
    if not torch.is_tensor(images):
        raise ValueError("PanoAnchorSplat prior requires images.")
    if not torch.is_tensor(features):
        raise ValueError("PanoAnchorSplat prior requires cached PanoVGGT features; fake descriptors are not allowed.")
    if not torch.is_tensor(depths):
        raise ValueError("PanoAnchorSplat prior requires depths/depth.")
    if not torch.is_tensor(poses):
        raise ValueError("PanoAnchorSplat prior requires poses_c2w.")
    if not torch.is_tensor(world_points):
        raise ValueError("PanoAnchorSplat prior requires world_points/chunk_world_points.")

    images_t = _normalize_images(images)
    image_hw = int(images_t.shape[-2]), int(images_t.shape[-1])
    features_t = _normalize_features(features).to(device=images_t.device)
    depths_t = _normalize_depths(depths.to(device=images_t.device) if torch.is_tensor(depths) else depths, image_hw=image_hw)
    batch_views = int(images_t.shape[0]), int(images_t.shape[1])
    poses_t = _normalize_poses(poses.to(device=images_t.device), batch_views=batch_views)
    world_t = _normalize_world_points(world_points.to(device=images_t.device), image_hw=image_hw)
    if tuple(features_t.shape[:2]) != batch_views:
        raise ValueError(f"features must share B,V={batch_views}, got {tuple(features_t.shape)}")
    if tuple(depths_t.shape[:2]) != batch_views or tuple(world_t.shape[:2]) != batch_views:
        raise ValueError("depths and world_points must share B,V with images.")

    valid = _normalize_mask(
        _first_present(mapping, ("valid_mask", "valid_depth", "valid_world_points_mask")),
        batch_views=batch_views,
        image_hw=image_hw,
        default=True,
    ).to(device=images_t.device)
    sky = _first_present(mapping, ("sky_mask", "sky_prob_mask"))
    sky_t = None if sky is None else _normalize_mask(sky, batch_views=batch_views, image_hw=image_hw, default=False).to(device=images_t.device)
    confidence_t = _normalize_confidence(
        _first_present(mapping, ("confidence", "depth_confidence", "world_points_confidence")),
        depths=depths_t,
    ).to(device=images_t.device)
    return PanoAnchorSplatPrior(
        images=images_t.detach(),
        features=features_t.detach(),
        depths=depths_t.detach(),
        poses_c2w=poses_t.detach(),
        world_points=world_t.detach(),
        valid_mask=valid.detach(),
        confidence=confidence_t.detach(),
        sky_mask=None if sky_t is None else sky_t.detach(),
        image_hw=image_hw,
        feature_hw=(int(features_t.shape[-2]), int(features_t.shape[-1])),
    )


class PanoVGGTPriorProvider(nn.Module):
    """Adapter that supplies frozen PanoVGGT priors, preferring offline cache."""

    def __init__(
        self,
        *,
        cache_root: str | Path | None = None,
        cache_key_field: str = "sequence_id",
        strict_cache: bool = True,
        detach: bool = True,
    ) -> None:
        super().__init__()
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.cache_key_field = str(cache_key_field)
        self.strict_cache = bool(strict_cache)
        self.detach_output = bool(detach)

    def forward(self, sample: PanoAnchorSplatPrior | Mapping[str, Any]) -> PanoAnchorSplatPrior:
        if isinstance(sample, PanoAnchorSplatPrior):
            return sample.detach() if self.detach_output else sample
        if not isinstance(sample, Mapping):
            raise TypeError(f"PanoVGGTPriorProvider expected mapping or PanoAnchorSplatPrior, got {type(sample)!r}")
        merged: dict[str, Any] = {}
        cache = self._maybe_load_cache(sample)
        if cache is not None:
            merged.update(cache)
        merged.update(dict(sample))
        prior = prior_from_mapping(merged)
        return prior.detach() if self.detach_output else prior

    def _maybe_load_cache(self, sample: Mapping[str, Any]) -> dict[str, Any] | None:
        direct_path = _maybe_first_batch_item(sample.get("cache_path"))
        path: Path | None = None
        if direct_path:
            path = Path(str(direct_path))
        elif self.cache_root is not None:
            key = _maybe_first_batch_item(sample.get(self.cache_key_field))
            if key is not None:
                key_s = str(key).replace("\\", "_").replace("/", "_")
                path = self.cache_root / f"{key_s}.pt"
        if path is None:
            return None
        if not path.is_file():
            if self.strict_cache:
                raise FileNotFoundError(f"PanoVGGT offline cache not found: {path}")
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"PanoVGGT offline cache must contain a mapping, got {type(payload)!r}: {path}")
        return dict(payload)
