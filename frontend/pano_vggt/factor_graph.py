"""Dense spherical factor graph containers for PanoVGGT-M3 matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DenseSphereFactor:
    """Dense correspondence factors for one directed frame-pair edge."""

    src: int
    tgt: int
    src_uv: torch.Tensor
    tgt_uv: torch.Tensor
    src_bearing: torch.Tensor
    tgt_bearing: torch.Tensor
    weight: torch.Tensor
    match_score: torch.Tensor
    valid_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseSphereFactorGraph:
    """Lightweight factor graph for diagnostics and future spherical BA."""

    factors: list[DenseSphereFactor] = field(default_factory=list)
    edges: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def build_edges(
        num_frames: int,
        *,
        temporal_radius: int = 2,
        max_edges: int | None = None,
        manual_edges: torch.Tensor | list[tuple[int, int]] | None = None,
        bidirectional: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Build temporal, skip, and optional manual directed edges."""

        if manual_edges is not None:
            edges = torch.as_tensor(manual_edges, dtype=torch.long, device=device)
            if edges.ndim != 2 or int(edges.shape[1]) != 2:
                raise ValueError(f"manual_edges must have shape Ex2, got {tuple(edges.shape)}.")
            return _dedupe_edges(edges, num_frames=num_frames, max_edges=max_edges)

        pairs: list[tuple[int, int]] = []
        radius = max(1, int(temporal_radius))
        for src in range(int(num_frames)):
            for delta in range(1, radius + 1):
                tgt = src + delta
                if tgt >= int(num_frames):
                    continue
                pairs.append((src, tgt))
                if bidirectional:
                    pairs.append((tgt, src))
        if not pairs:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        return _dedupe_edges(torch.tensor(pairs, dtype=torch.long, device=device), num_frames=num_frames, max_edges=max_edges)

    def add_factor(self, factor: DenseSphereFactor) -> None:
        self.factors.append(factor)
        if self.edges is None:
            self.edges = torch.tensor([[factor.src, factor.tgt]], dtype=torch.long, device=factor.weight.device)
        else:
            edge = torch.tensor([[factor.src, factor.tgt]], dtype=torch.long, device=self.edges.device)
            self.edges = _dedupe_edges(torch.cat([self.edges, edge], dim=0), num_frames=max(int(self.edges.max()) + 1, factor.src + 1, factor.tgt + 1))

    @property
    def num_edges(self) -> int:
        return 0 if self.edges is None else int(self.edges.shape[0])

    @property
    def num_factors(self) -> int:
        return int(sum(int(factor.valid_mask.numel()) for factor in self.factors))

    @property
    def num_valid_factors(self) -> int:
        return int(sum(int(factor.valid_mask.bool().sum().detach().cpu()) for factor in self.factors))

    def metrics(self) -> dict[str, float]:
        """Return aggregate scalar diagnostics for the graph."""

        total = self.num_factors
        valid = self.num_valid_factors
        weights = []
        scores = []
        sky_filtered = []
        fb_pass = []
        depth_pass = []
        angular = []
        for factor in self.factors:
            mask = factor.valid_mask.bool()
            if factor.weight.numel():
                weights.append(factor.weight.detach().float().reshape(-1))
            if factor.match_score.numel():
                scores.append(factor.match_score.detach().float().reshape(-1))
            for key, parts in (
                ("sky_filtered_mask", sky_filtered),
                ("fb_pass_mask", fb_pass),
                ("depth_consistency_mask", depth_pass),
                ("angular_error_deg", angular),
            ):
                value = factor.metadata.get(key)
                if torch.is_tensor(value):
                    data = value.detach().float().reshape(-1)
                    if key == "angular_error_deg" and mask.numel() == data.numel() and mask.any():
                        data = data[mask.reshape(-1)]
                    parts.append(data)

        return {
            "num_edges": float(self.num_edges),
            "num_factors": float(total),
            "valid_factors": float(valid),
            "valid_factor_ratio": float(valid / total) if total else 0.0,
            "sky_filtered_ratio": _mean_bool_ratio(sky_filtered),
            "fb_pass_ratio": _mean_bool_ratio(fb_pass),
            "depth_consistency_pass_ratio": _mean_bool_ratio(depth_pass),
            "mean_match_score": _mean_tensor(scores),
            "mean_weight": _mean_tensor(weights),
            "mean_spherical_angular_error_deg": _mean_tensor(angular),
        }

    def to_metadata(self) -> dict[str, Any]:
        out = dict(self.metadata)
        out.update(self.metrics())
        return out


def _dedupe_edges(edges: torch.Tensor, *, num_frames: int, max_edges: int | None = None) -> torch.Tensor:
    if edges.numel() == 0:
        return edges.reshape(0, 2)
    if edges.ndim != 2 or int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape Ex2, got {tuple(edges.shape)}.")
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_frames):
        raise ValueError("DenseSphereFactorGraph edges contain frame indices outside the local window.")
    keep: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for src, tgt in edges.detach().cpu().tolist():
        pair = (int(src), int(tgt))
        if pair[0] == pair[1] or pair in seen:
            continue
        seen.add(pair)
        keep.append(pair)
        if max_edges is not None and len(keep) >= int(max_edges):
            break
    if not keep:
        return torch.empty(0, 2, dtype=torch.long, device=edges.device)
    return torch.tensor(keep, dtype=torch.long, device=edges.device)


def _mean_tensor(parts: list[torch.Tensor]) -> float:
    values = [part.reshape(-1).float() for part in parts if part.numel()]
    if not values:
        return 0.0
    cat = torch.cat(values, dim=0)
    finite = cat[torch.isfinite(cat)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.mean().detach().cpu())


def _mean_bool_ratio(parts: list[torch.Tensor]) -> float:
    values = [part.reshape(-1).float() for part in parts if part.numel()]
    if not values:
        return 0.0
    cat = torch.cat(values, dim=0)
    return float((cat > 0.5).float().mean().detach().cpu())
