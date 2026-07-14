"""Submap alignment utilities for PanoVGGT chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geometry.sim3 import sim3_components, weighted_umeyama


@dataclass
class SimilarityTransform:
    """Transform source chunk coordinates into global coordinates."""

    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    residual: float = 0.0
    inlier_ratio: float = 1.0
    accepted: bool = True

    @classmethod
    def identity(cls, *, device=None, dtype=torch.float32) -> "SimilarityTransform":
        return cls(
            scale=1.0,
            rotation=torch.eye(3, device=device, dtype=dtype),
            translation=torch.zeros(3, device=device, dtype=dtype),
        )

    def apply_points(self, points: torch.Tensor) -> torch.Tensor:
        rot = self.rotation.to(device=points.device, dtype=points.dtype)
        trans = self.translation.to(device=points.device, dtype=points.dtype)
        return float(self.scale) * torch.matmul(points, rot.T) + trans

    def apply_pose(self, c2w: torch.Tensor) -> torch.Tensor:
        rot = self.rotation.to(device=c2w.device, dtype=c2w.dtype)
        trans = self.translation.to(device=c2w.device, dtype=c2w.dtype)
        out = c2w.clone()
        out[:3, :3] = rot @ c2w[:3, :3]
        out[:3, 3] = float(self.scale) * (rot @ c2w[:3, 3]) + trans
        return out

    def as_matrix(self) -> torch.Tensor:
        mat = torch.eye(4, device=self.rotation.device, dtype=self.rotation.dtype)
        mat[:3, :3] = float(self.scale) * self.rotation
        mat[:3, 3] = self.translation
        return mat


def _umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    allow_scale: bool,
) -> SimilarityTransform:
    transform = weighted_umeyama(
        source.float(),
        target.float(),
        weights.float(),
        allow_scale=allow_scale,
    )
    scale, rotation, translation = sim3_components(transform)
    return SimilarityTransform(
        scale=float(scale.detach().cpu()),
        rotation=rotation,
        translation=translation,
    )


def _residual(transform: SimilarityTransform, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(transform.apply_points(source) - target, dim=-1)


class SubmapAligner:
    """Robust weighted Sim(3)/SE(3) alignment for overlapping point maps."""

    def __init__(
        self,
        *,
        align_mode: str = "sim3",
        max_residual: float = 0.35,
        min_inlier_ratio: float = 0.35,
        max_scale_change: float = 2.5,
        min_points: int = 32,
        return_rejected_transform: bool = False,
        irls_iterations: int = 3,
        huber_delta: float | None = None,
    ) -> None:
        mode = str(align_mode).lower()
        if mode not in {"sim3", "se3"}:
            raise ValueError(f"Unsupported align_mode: {align_mode}")
        self.align_mode = mode
        self.max_residual = float(max_residual)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_scale_change = float(max_scale_change)
        self.min_points = int(min_points)
        self.return_rejected_transform = bool(return_rejected_transform)
        self.irls_iterations = max(1, int(irls_iterations))
        self.huber_delta = None if huber_delta is None else float(huber_delta)

    def align(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> SimilarityTransform:
        if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
            raise ValueError("source and target must both have shape Nx3")
        finite = torch.isfinite(source).all(dim=1) & torch.isfinite(target).all(dim=1)
        if weights is None:
            weights = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
        else:
            weights = weights.to(device=source.device, dtype=source.dtype)
        finite &= torch.isfinite(weights) & (weights > 0)
        source = source[finite]
        target = target[finite]
        weights = weights[finite]
        if source.shape[0] < self.min_points:
            return SimilarityTransform.identity(device=source.device, dtype=source.dtype)

        transform = _umeyama(source, target, weights, allow_scale=self.align_mode == "sim3")
        gate = self.max_residual
        for _ in range(self.irls_iterations):
            dist = _residual(transform, source, target)
            med = torch.quantile(dist.detach(), 0.5)
            gate = max(self.max_residual, float(med * 2.5))
            inliers = dist <= gate
            if int(inliers.sum()) < self.min_points:
                break
            delta = (
                max(float(self.huber_delta), 1.0e-8)
                if self.huber_delta is not None
                else max(float(med * 1.4826), 1.0e-6)
            )
            huber = torch.minimum(
                torch.ones_like(dist[inliers]),
                dist[inliers].new_tensor(delta) / dist[inliers].clamp_min(1.0e-8),
            )
            transform = _umeyama(
                source[inliers],
                target[inliers],
                weights[inliers] * huber,
                allow_scale=self.align_mode == "sim3",
            )
        dist = _residual(transform, source, target)
        inliers = dist <= gate
        inlier_ratio = float(inliers.float().mean())

        residual = float(dist[inliers].mean()) if bool(inliers.any()) else float(dist.mean())
        scale_ok = (1.0 / self.max_scale_change) <= float(transform.scale) <= self.max_scale_change
        accepted = residual <= self.max_residual and inlier_ratio >= self.min_inlier_ratio and scale_ok
        transform.residual = residual
        transform.inlier_ratio = inlier_ratio
        transform.accepted = bool(accepted)
        if not transform.accepted:
            if self.return_rejected_transform:
                return transform
            fallback = SimilarityTransform.identity(device=source.device, dtype=source.dtype)
            fallback.residual = residual
            fallback.inlier_ratio = inlier_ratio
            fallback.accepted = False
            return fallback
        return transform


def sample_overlap_points(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_confidence: torch.Tensor | None = None,
    target_confidence: torch.Tensor | None = None,
    *,
    max_points: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten and subsample overlapping point maps."""

    if source_points.shape != target_points.shape:
        raise ValueError("source and target point maps must have the same shape")
    source = source_points.reshape(-1, 3)
    target = target_points.reshape(-1, 3)
    weights = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
    if source_confidence is not None:
        weights = weights * source_confidence.reshape(-1).to(weights)
    if target_confidence is not None:
        weights = weights * target_confidence.reshape(-1).to(weights)
    finite = torch.isfinite(source).all(dim=1) & torch.isfinite(target).all(dim=1) & torch.isfinite(weights) & (weights > 0)
    source = source[finite]
    target = target[finite]
    weights = weights[finite].clamp_min(0.0)
    if source.shape[0] > int(max_points):
        _, idx = torch.topk(weights, k=int(max_points), largest=True)
        source = source[idx]
        target = target[idx]
        weights = weights[idx]
    return source, target, weights
