"""Voxel anchor construction for PanoAnchorSplat."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig, PanoAnchorSplatPrior
from .pano_resplat_geometry import erp_pixel_grid


def _group_mean(values: torch.Tensor, inverse: torch.Tensor, counts: torch.Tensor, groups: int) -> torch.Tensor:
    out = values.new_zeros((groups, *values.shape[1:]))
    out.index_add_(0, inverse, values)
    return out / counts.view(groups, *([1] * (values.ndim - 1))).clamp_min(1.0)


def _dominant_indices(valid_idx: torch.Tensor, inverse: torch.Tensor, confidence: torch.Tensor, groups: int) -> torch.Tensor:
    conf = confidence.reshape(-1)
    order_conf = torch.argsort(conf, descending=True, stable=True)
    inv_conf = inverse[order_conf]
    order_group = torch.argsort(inv_conf, stable=True)
    order = order_conf[order_group]
    grouped = inverse[order]
    first = torch.ones_like(grouped, dtype=torch.bool)
    if int(grouped.numel()) > 1:
        first[1:] = grouped[1:] != grouped[:-1]
    dominant = torch.zeros(groups, device=valid_idx.device, dtype=torch.long)
    dominant[grouped[first]] = valid_idx[order[first]]
    return dominant


class PanoVoxelAnchorBuilder(nn.Module):
    """Build compact world-space voxel anchors from frozen PanoVGGT priors."""

    def __init__(self, config: PanoAnchorSplatConfig | dict | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.last_stats: dict[str, torch.Tensor] = {}

    def forward(self, prior: PanoAnchorSplatPrior) -> PanoAnchorSet:
        anchors = self.build(prior)
        valid_count = anchors.valid_mask.sum(dim=1).to(dtype=anchors.centers.dtype)
        dense_count = prior.valid_mask.flatten(1).sum(dim=1).to(dtype=anchors.centers.dtype)
        self.last_stats = {
            "anchor_count": valid_count.mean().detach(),
            "dense_count": dense_count.mean().detach(),
            "compression_ratio": (valid_count / dense_count.clamp_min(1.0)).mean().detach(),
        }
        return anchors

    def build(self, prior: PanoAnchorSplatPrior) -> PanoAnchorSet:
        b, v, _c, h, w = [int(x) for x in prior.images.shape]
        device = prior.world_points.device
        dtype = prior.world_points.dtype
        features = prior.features.to(device=device, dtype=dtype)
        if tuple(features.shape[-2:]) != (h, w):
            flat = features.reshape(b * v, int(features.shape[2]), int(features.shape[-2]), int(features.shape[-1]))
            features = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False).reshape(b, v, int(features.shape[2]), h, w)
        feature_flat = features.permute(0, 1, 3, 4, 2).reshape(b, v * h * w, int(features.shape[2]))
        points_flat = prior.world_points.reshape(b, v * h * w, 3).to(device=device, dtype=dtype)
        conf = prior.confidence
        if conf is None:
            conf = (prior.depths > 0.0).to(dtype=dtype)
        conf_flat = conf.to(device=device, dtype=dtype).reshape(b, v * h * w, 1)
        valid = prior.valid_mask.to(device=device).reshape(b, v * h * w)
        finite = torch.isfinite(points_flat).all(dim=-1) & torch.isfinite(feature_flat).all(dim=-1)
        valid = valid & finite & (conf_flat[..., 0] > 0.0) & (prior.depths.to(device=device).reshape(b, v * h * w) > 0.0)
        if prior.sky_mask is not None:
            valid = valid & ~prior.sky_mask.to(device=device).reshape(b, v * h * w).bool()
        uv_grid = erp_pixel_grid((h, w), device=device, dtype=dtype).view(1, 1, h, w, 2).expand(b, v, h, w, 2)
        uv_flat = uv_grid.reshape(b, v * h * w, 2)
        source_ids = torch.arange(v, device=device, dtype=torch.long).view(1, v, 1, 1).expand(b, v, h, w).reshape(b, v * h * w)

        per_batch = [
            self._build_one(
                points_flat[batch_idx],
                feature_flat[batch_idx],
                conf_flat[batch_idx],
                valid[batch_idx],
                source_ids[batch_idx],
                uv_flat[batch_idx],
            )
            for batch_idx in range(b)
        ]
        max_anchors = max(1, max(int(item["centers"].shape[0]) for item in per_batch))
        max_anchors = min(max_anchors, int(self.config.effective_max_anchors))
        return self._pad(per_batch, max_anchors, feature_dim=int(features.shape[2]), device=device, dtype=dtype, image_hw=(h, w))

    def _build_one(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        source_ids: torch.Tensor,
        source_uv: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_idx = valid.nonzero(as_tuple=False).flatten()
        if int(valid_idx.numel()) == 0:
            return self._empty_one(features.shape[-1], points.device, points.dtype)
        pts = points[valid_idx]
        coords = torch.floor(pts / float(self.config.voxel_size)).to(torch.int64)
        _unique, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
        groups = int(inverse.max().item()) + 1
        counts = torch.bincount(inverse, minlength=groups).to(device=points.device, dtype=points.dtype).clamp_min(1.0)
        centers = _group_mean(pts, inverse, counts, groups)
        mean_sq = _group_mean(pts.square(), inverse, counts, groups)
        scales = (mean_sq - centers.square()).clamp_min(0.0).sqrt()
        scales = torch.maximum(scales, torch.full_like(scales, float(self.config.min_scale)))
        scales = scales.clamp(max=float(self.config.max_scale))
        feat = _group_mean(features[valid_idx], inverse, counts, groups)
        conf = _group_mean(confidence[valid_idx], inverse, counts, groups).clamp(0.0, 1.0)
        dominant = _dominant_indices(valid_idx, inverse, confidence[valid_idx], groups)
        score = (conf[..., 0] * torch.log1p(counts)).to(dtype=points.dtype)
        if groups > int(self.config.effective_max_anchors):
            keep = torch.topk(score, k=int(self.config.effective_max_anchors), largest=True, sorted=True).indices
        else:
            keep = torch.argsort(score, descending=True, stable=True)
        return {
            "centers": centers[keep],
            "scales": scales[keep],
            "features": feat[keep],
            "confidence": conf[keep],
            "counts": counts[keep].view(-1, 1),
            "source_view_ids": source_ids[dominant[keep]].to(dtype=torch.long),
            "source_uv": source_uv[dominant[keep]],
            "valid_mask": torch.ones(int(keep.numel()), device=points.device, dtype=torch.bool),
        }

    @staticmethod
    def _empty_one(feature_dim: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "centers": torch.zeros(0, 3, device=device, dtype=dtype),
            "scales": torch.zeros(0, 3, device=device, dtype=dtype),
            "features": torch.zeros(0, int(feature_dim), device=device, dtype=dtype),
            "confidence": torch.zeros(0, 1, device=device, dtype=dtype),
            "counts": torch.zeros(0, 1, device=device, dtype=dtype),
            "source_view_ids": torch.zeros(0, device=device, dtype=torch.long),
            "source_uv": torch.zeros(0, 2, device=device, dtype=dtype),
            "valid_mask": torch.zeros(0, device=device, dtype=torch.bool),
        }

    @staticmethod
    def _pad(
        batches: list[dict[str, torch.Tensor]],
        max_anchors: int,
        *,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        image_hw: tuple[int, int],
    ) -> PanoAnchorSet:
        b = len(batches)
        centers = torch.zeros(b, max_anchors, 3, device=device, dtype=dtype)
        scales = torch.zeros(b, max_anchors, 3, device=device, dtype=dtype)
        features = torch.zeros(b, max_anchors, int(feature_dim), device=device, dtype=dtype)
        confidence = torch.zeros(b, max_anchors, 1, device=device, dtype=dtype)
        counts = torch.zeros(b, max_anchors, 1, device=device, dtype=dtype)
        source_view_ids = torch.zeros(b, max_anchors, device=device, dtype=torch.long)
        source_uv = torch.zeros(b, max_anchors, 2, device=device, dtype=dtype)
        valid_mask = torch.zeros(b, max_anchors, device=device, dtype=torch.bool)
        for batch_idx, item in enumerate(batches):
            n = min(max_anchors, int(item["centers"].shape[0]))
            if n == 0:
                continue
            centers[batch_idx, :n] = item["centers"][:n]
            scales[batch_idx, :n] = item["scales"][:n]
            features[batch_idx, :n] = item["features"][:n]
            confidence[batch_idx, :n] = item["confidence"][:n]
            counts[batch_idx, :n] = item["counts"][:n]
            source_view_ids[batch_idx, :n] = item["source_view_ids"][:n]
            source_uv[batch_idx, :n] = item["source_uv"][:n]
            valid_mask[batch_idx, :n] = item["valid_mask"][:n]
        return PanoAnchorSet(
            centers=centers,
            scales=scales.clamp_min(1.0e-6),
            features=features,
            confidence=confidence,
            counts=counts,
            source_view_ids=source_view_ids,
            source_uv=source_uv,
            valid_mask=valid_mask,
            image_hw=image_hw,
        )
