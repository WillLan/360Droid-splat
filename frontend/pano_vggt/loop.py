"""Lightweight loop bookkeeping for the PanoVGGT long frontend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .alignment import SimilarityTransform


@dataclass
class PoseGraphEdge:
    source_chunk: int
    target_chunk: int
    transform: SimilarityTransform
    residual: float
    edge_type: str


class FrontendPoseGraph:
    """Minimal append-only pose graph state for frontend bookkeeping."""

    def __init__(self) -> None:
        self.edges: list[PoseGraphEdge] = []

    def add_edge(self, edge: PoseGraphEdge) -> None:
        self.edges.append(edge)


class LoopManager:
    """Descriptor-based loop candidate filter.

    The v1 backend remains append-only, so accepted loops are recorded for
    diagnostics and for future chunk transforms but do not move old Gaussians.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        min_separation: int = 3,
        score_threshold: float = 0.92,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_separation = int(min_separation)
        self.score_threshold = float(score_threshold)
        self._chunk_descriptors: list[torch.Tensor] = []

    def add_chunk(self, descriptor: torch.Tensor | None) -> int | None:
        if descriptor is None:
            self._chunk_descriptors.append(torch.empty(0))
            return None
        desc = descriptor.detach().float().flatten()
        if desc.numel() > 0:
            desc = desc / torch.linalg.norm(desc).clamp_min(1e-8)
        self._chunk_descriptors.append(desc.cpu())
        if not self.enabled or desc.numel() == 0:
            return None
        cur = len(self._chunk_descriptors) - 1
        best_idx = None
        best_score = -1.0
        for idx, old in enumerate(self._chunk_descriptors[:-1]):
            if cur - idx < self.min_separation or old.numel() != desc.numel() or old.numel() == 0:
                continue
            score = float((old.to(desc.device) * desc).sum())
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= self.score_threshold:
            return best_idx
        return None

