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

    @property
    def point_maps(self) -> torch.Tensor:
        """Backward-compatible alias for chunk-world point maps."""

        return self.chunk_world_points
