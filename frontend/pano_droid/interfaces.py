"""Public frontend interfaces used by the SLAM system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PanoFrame:
    image: Tensor
    timestamp: float
    frame_id: int
    mask: Optional[Tensor] = None
    meta: Optional[dict] = None


@dataclass
class FrontendOutput:
    frame_id: int
    timestamp: float
    pose_c2w: Tensor
    relative_pose: Optional[Tensor]
    pose_confidence: float
    inverse_depth: Optional[Tensor]
    depth_confidence: Optional[Tensor]
    spherical_flow: Optional[Tensor]
    keyframe_score: float
    is_keyframe: bool
    ba_residual: Optional[float]
    tracking_status: str
    world_points: Optional[Tensor] = None
    world_points_confidence: Optional[Tensor] = None
    valid_world_points_mask: Optional[Tensor] = None


class PanoDROIDFrontend:
    """Base interface for a PanoDROID frontend implementation."""

    def initialize(self, sequence_meta: dict) -> None:
        raise NotImplementedError

    def track(self, frame: PanoFrame) -> FrontendOutput:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


def ensure_chw_image(image: Tensor) -> Tensor:
    """Validate and normalize an image tensor to ``CxHxW`` float format."""
    if image.ndim != 3:
        raise ValueError(f"Expected image as CxHxW, got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
    if not image.is_floating_point():
        image = image.float() / 255.0
    return image.clamp(0.0, 1.0)


def identity_pose(device=None, dtype=torch.float32) -> Tensor:
    return torch.eye(4, device=device, dtype=dtype)
