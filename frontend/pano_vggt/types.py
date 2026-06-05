"""Typed outputs for the PanoVGGT long-sequence frontend."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PanoVGGTLocalPrediction:
    """Local PanoVGGT prediction for one chunk."""

    poses_c2w: torch.Tensor
    depth: torch.Tensor
    confidence: torch.Tensor
    chunk_world_points: torch.Tensor
    local_points: torch.Tensor | None = None
    global_points: torch.Tensor | None = None
    descriptors: torch.Tensor | None = None
    dense_descriptors: torch.Tensor | None = None
    match_confidence: torch.Tensor | None = None
    static_confidence: torch.Tensor | None = None
    sky_logits: torch.Tensor | None = None
    sky_prob: torch.Tensor | None = None
    feature_hw: tuple[int, int] | None = None
    image_hw: tuple[int, int] | None = None
    descriptor_dim: int = 24
    matching_debug: dict[str, float] | None = None
    ba_residual_angular: float | None = None
    ba_valid_ratio: float | None = None
    ba_update_norm: dict[str, float] | None = None

    @property
    def point_maps(self) -> torch.Tensor:
        """Backward-compatible alias for chunk-world point maps."""

        return self.chunk_world_points
