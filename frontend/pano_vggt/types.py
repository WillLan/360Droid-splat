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
    point_maps: torch.Tensor
    descriptors: torch.Tensor | None = None

