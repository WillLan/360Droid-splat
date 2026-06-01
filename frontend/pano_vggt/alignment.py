"""Submap alignment utilities for PanoVGGT chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    source = source.float()
    target = target.float()
    weights = weights.float().clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-8)
    src_mean = (weights[:, None] * source).sum(dim=0)
    tgt_mean = (weights[:, None] * target).sum(dim=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    cov = (weights[:, None] * src_c).T @ tgt_c
    u, s, vh = torch.linalg.svd(cov)
    rot = vh.T @ u.T
    if torch.linalg.det(rot) < 0:
        vh = vh.clone()
        vh[-1] *= -1.0
        rot = vh.T @ u.T
    var_src = (weights * (src_c.square().sum(dim=1))).sum().clamp_min(1e-8)
    scale = float(s.sum() / var_src) if allow_scale else 1.0
    trans = tgt_mean - scale * (rot @ src_mean)
    return SimilarityTransform(scale=scale, rotation=rot, translation=trans)


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
    ) -> None:
        mode = str(align_mode).lower()
        if mode not in {"sim3", "se3"}:
            raise ValueError(f"Unsupported align_mode: {align_mode}")
        self.align_mode = mode
        self.max_residual = float(max_residual)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_scale_change = float(max_scale_change)
        self.min_points = int(min_points)

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
        dist = _residual(transform, source, target)
        med = torch.quantile(dist.detach(), 0.5)
        gate = max(self.max_residual, float(med * 2.5))
        inliers = dist <= gate
        inlier_ratio = float(inliers.float().mean())
        if int(inliers.sum()) >= self.min_points:
            transform = _umeyama(source[inliers], target[inliers], weights[inliers], allow_scale=self.align_mode == "sim3")
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
    finite = torch.isfinite(source).all(dim=1) & torch.isfinite(target).all(dim=1) & torch.isfinite(weights)
    source = source[finite]
    target = target[finite]
    weights = weights[finite].clamp_min(0.0)
    if source.shape[0] > int(max_points):
        _, idx = torch.topk(weights, k=int(max_points), largest=True)
        source = source[idx]
        target = target[idx]
        weights = weights[idx]
    return source, target, weights

