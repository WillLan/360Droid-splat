"""Voxel compaction for dense Pano-ReSplat Gaussian states."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .resplat_types import PanoGaussianState


@dataclass(frozen=True)
class VoxelCompactorStats:
    dense_count: torch.Tensor
    anchor_count: torch.Tensor
    compression_ratio: torch.Tensor
    mean_voxel_count: torch.Tensor
    max_voxel_count: torch.Tensor


def _detach_state(state: PanoGaussianState) -> PanoGaussianState:
    return PanoGaussianState(
        means=state.means.detach(),
        log_scales=state.log_scales.detach(),
        rotations_unnorm=state.rotations_unnorm.detach(),
        opacity_logits=state.opacity_logits.detach(),
        sh_coeffs=state.sh_coeffs.detach(),
        latent_features=state.latent_features.detach(),
        source_view_ids=state.source_view_ids.detach(),
        source_uv=state.source_uv.detach(),
        valid_mask=state.valid_mask.detach(),
        confidence=None if state.confidence is None else state.confidence.detach(),
    )


class VoxelGaussianCompactor(nn.Module):
    """Fuse dense per-pixel Gaussians into compact voxel anchors.

    The module intentionally preserves the public ``PanoGaussianState`` schema so
    renderers, feedback encoders, and update blocks can consume compact anchors
    without a separate object type.
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.02,
        detach_input: bool = True,
        inject_anchor_stats: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if float(voxel_size) <= 0.0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.detach_input = bool(detach_input)
        self.inject_anchor_stats = bool(inject_anchor_stats)
        self.eps = float(eps)
        self.last_stats: dict[str, torch.Tensor] = {}

    def forward(self, state: PanoGaussianState) -> PanoGaussianState:
        compact, stats = self.compact(state)
        self.last_stats = {key: value.detach() for key, value in stats.items()}
        return compact

    def compact(self, state: PanoGaussianState) -> tuple[PanoGaussianState, dict[str, torch.Tensor]]:
        src = _detach_state(state) if self.detach_input else state
        b, _n, _ = src.means.shape
        device = src.means.device
        dtype = src.means.dtype
        per_batch: list[dict[str, torch.Tensor]] = []
        anchor_counts: list[torch.Tensor] = []
        dense_counts: list[torch.Tensor] = []
        mean_counts: list[torch.Tensor] = []
        max_counts: list[torch.Tensor] = []

        for batch_idx in range(b):
            compact_b = self._compact_one(src, batch_idx)
            per_batch.append(compact_b)
            count = compact_b["valid_mask"].sum()
            dense_count = compact_b["dense_count"]
            anchor_counts.append(count.to(device=device, dtype=dtype))
            dense_counts.append(dense_count.to(device=device, dtype=dtype))
            voxel_count = compact_b["voxel_count"].to(device=device, dtype=dtype)
            if voxel_count.numel() > 0:
                mean_counts.append(voxel_count.mean())
                max_counts.append(voxel_count.max())
            else:
                mean_counts.append(src.means.new_tensor(0.0))
                max_counts.append(src.means.new_tensor(0.0))

        max_anchors = max(1, max(int(item["valid_mask"].numel()) for item in per_batch))
        out = self._pad_batches(src, per_batch, max_anchors)
        dense_count_t = torch.stack(dense_counts)
        anchor_count_t = torch.stack(anchor_counts)
        stats = {
            "dense_count": dense_count_t.mean(),
            "anchor_count": anchor_count_t.mean(),
            "compression_ratio": (anchor_count_t / dense_count_t.clamp_min(1.0)).mean(),
            "mean_voxel_count": torch.stack(mean_counts).mean(),
            "max_voxel_count": torch.stack(max_counts).max(),
        }
        return out, stats

    def _compact_one(self, state: PanoGaussianState, batch_idx: int) -> dict[str, torch.Tensor]:
        means = state.means[batch_idx]
        valid = state.valid_mask[batch_idx].bool()
        finite = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(state.log_scales[batch_idx]).all(dim=-1)
            & torch.isfinite(state.rotations_unnorm[batch_idx]).all(dim=-1)
            & torch.isfinite(state.opacity_logits[batch_idx]).all(dim=-1)
            & torch.isfinite(state.sh_coeffs[batch_idx]).flatten(1).all(dim=-1)
            & torch.isfinite(state.latent_features[batch_idx]).all(dim=-1)
        )
        if state.confidence is not None:
            finite = finite & torch.isfinite(state.confidence[batch_idx]).all(dim=-1)
        valid = valid & finite
        valid_idx = valid.nonzero(as_tuple=False).flatten()
        dense_count = valid_idx.numel()
        if dense_count == 0:
            return self._empty_one(state, batch_idx)

        coords = torch.floor(means[valid_idx] / float(self.voxel_size)).to(torch.int64)
        _unique, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
        anchors = int(inverse.max().item()) + 1
        counts_long = torch.bincount(inverse, minlength=anchors)
        counts = counts_long.to(device=means.device, dtype=means.dtype).clamp_min(1.0)

        confidence = self._dense_confidence(state, batch_idx)[valid_idx]
        dominant_idx = self._dominant_indices(valid_idx, inverse, confidence, anchors)
        ref_rot = state.rotations_unnorm[batch_idx, dominant_idx]

        def group_mean(value: torch.Tensor) -> torch.Tensor:
            flat = value[valid_idx]
            out = flat.new_zeros((anchors, *flat.shape[1:]))
            out.index_add_(0, inverse, flat)
            return out / counts.view(anchors, *([1] * (flat.ndim - 1)))

        opacity = torch.sigmoid(state.opacity_logits[batch_idx, valid_idx])
        opacity_sum = opacity.new_zeros(anchors, 1)
        opacity_sum.index_add_(0, inverse, opacity)
        opacity_mean = (opacity_sum / counts.view(anchors, 1)).clamp(self.eps, 1.0 - self.eps)

        scale = torch.exp(state.log_scales[batch_idx, valid_idx])
        scale_sum = scale.new_zeros(anchors, 3)
        scale_sum.index_add_(0, inverse, scale)
        log_scales = torch.log((scale_sum / counts.view(anchors, 1)).clamp_min(self.eps))

        rotations = state.rotations_unnorm[batch_idx, valid_idx]
        ref_per_point = ref_rot[inverse]
        sign = torch.where((rotations * ref_per_point).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        rot_sum = rotations.new_zeros(anchors, 4)
        rot_sum.index_add_(0, inverse, rotations * sign)
        rotations_mean = rot_sum / counts.view(anchors, 1)

        sh = group_mean(state.sh_coeffs[batch_idx])
        latent = group_mean(state.latent_features[batch_idx])
        conf_mean = group_mean(self._dense_confidence(state, batch_idx))
        if self.inject_anchor_stats and latent.shape[-1] > 0:
            log_count = torch.log1p(counts).view(anchors, 1)
            log_count = log_count / log_count.max().clamp_min(1.0)
            latent = latent.clone()
            latent[:, 0:1] = log_count.to(dtype=latent.dtype)
            if latent.shape[-1] > 1:
                latent[:, 1:2] = conf_mean.to(dtype=latent.dtype)

        return {
            "means": group_mean(state.means[batch_idx]),
            "log_scales": log_scales,
            "rotations_unnorm": rotations_mean,
            "opacity_logits": torch.logit(opacity_mean),
            "sh_coeffs": sh,
            "latent_features": latent,
            "source_view_ids": state.source_view_ids[batch_idx, dominant_idx].to(dtype=torch.long),
            "source_uv": state.source_uv[batch_idx, dominant_idx],
            "valid_mask": torch.ones(anchors, device=means.device, dtype=torch.bool),
            "confidence": conf_mean,
            "dense_count": torch.as_tensor(float(dense_count), device=means.device, dtype=means.dtype),
            "voxel_count": counts,
        }

    def _empty_one(self, state: PanoGaussianState, batch_idx: int) -> dict[str, torch.Tensor]:
        device = state.means.device
        dtype = state.means.dtype
        sh_dim = state.sh_dim
        latent_dim = state.latent_dim
        return {
            "means": torch.zeros(0, 3, device=device, dtype=dtype),
            "log_scales": torch.zeros(0, 3, device=device, dtype=dtype),
            "rotations_unnorm": torch.zeros(0, 4, device=device, dtype=dtype),
            "opacity_logits": torch.zeros(0, 1, device=device, dtype=dtype),
            "sh_coeffs": torch.zeros(0, 3, sh_dim, device=device, dtype=dtype),
            "latent_features": torch.zeros(0, latent_dim, device=device, dtype=dtype),
            "source_view_ids": torch.zeros(0, device=device, dtype=torch.long),
            "source_uv": torch.zeros(0, 2, device=device, dtype=dtype),
            "valid_mask": torch.zeros(0, device=device, dtype=torch.bool),
            "confidence": torch.zeros(0, 1, device=device, dtype=dtype),
            "dense_count": torch.as_tensor(0.0, device=device, dtype=dtype),
            "voxel_count": torch.zeros(0, device=device, dtype=dtype),
        }

    def _pad_batches(
        self,
        state: PanoGaussianState,
        batches: list[dict[str, torch.Tensor]],
        max_anchors: int,
    ) -> PanoGaussianState:
        b = len(batches)
        device = state.means.device
        dtype = state.means.dtype
        sh_dim = state.sh_dim
        latent_dim = state.latent_dim

        means = torch.zeros(b, max_anchors, 3, device=device, dtype=dtype)
        log_scales = torch.full((b, max_anchors, 3), math.log(max(self.eps, 1.0e-12)), device=device, dtype=dtype)
        rotations = torch.zeros(b, max_anchors, 4, device=device, dtype=dtype)
        rotations[..., 0] = 1.0
        opacity = torch.zeros(b, max_anchors, 1, device=device, dtype=dtype)
        sh = torch.zeros(b, max_anchors, 3, sh_dim, device=device, dtype=dtype)
        latent = torch.zeros(b, max_anchors, latent_dim, device=device, dtype=dtype)
        source_ids = torch.zeros(b, max_anchors, device=device, dtype=torch.long)
        source_uv = torch.zeros(b, max_anchors, 2, device=device, dtype=dtype)
        valid = torch.zeros(b, max_anchors, device=device, dtype=torch.bool)
        confidence = torch.zeros(b, max_anchors, 1, device=device, dtype=dtype)

        for batch_idx, item in enumerate(batches):
            n = int(item["valid_mask"].numel())
            if n == 0:
                continue
            means[batch_idx, :n] = item["means"]
            log_scales[batch_idx, :n] = item["log_scales"]
            rotations[batch_idx, :n] = item["rotations_unnorm"]
            opacity[batch_idx, :n] = item["opacity_logits"]
            sh[batch_idx, :n] = item["sh_coeffs"]
            latent[batch_idx, :n] = item["latent_features"]
            source_ids[batch_idx, :n] = item["source_view_ids"]
            source_uv[batch_idx, :n] = item["source_uv"]
            valid[batch_idx, :n] = item["valid_mask"]
            confidence[batch_idx, :n] = item["confidence"]

        return PanoGaussianState(
            means=torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(log_scales, nan=math.log(max(self.eps, 1.0e-12)), posinf=0.0, neginf=math.log(max(self.eps, 1.0e-12))),
            rotations_unnorm=torch.nan_to_num(rotations, nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(opacity, nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(sh, nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=source_ids,
            source_uv=source_uv,
            valid_mask=valid,
            confidence=confidence,
        )

    @staticmethod
    def _dominant_indices(valid_idx: torch.Tensor, inverse: torch.Tensor, confidence: torch.Tensor, anchors: int) -> torch.Tensor:
        conf = confidence.reshape(-1)
        order_conf = torch.argsort(conf, descending=True, stable=True)
        inv_conf = inverse[order_conf]
        order_group = torch.argsort(inv_conf, stable=True)
        order = order_conf[order_group]
        groups = inverse[order]
        first = torch.ones_like(groups, dtype=torch.bool)
        if int(groups.numel()) > 1:
            first[1:] = groups[1:] != groups[:-1]
        dominant = torch.zeros(anchors, device=valid_idx.device, dtype=torch.long)
        dominant[groups[first]] = valid_idx[order[first]]
        return dominant

    @staticmethod
    def _dense_confidence(state: PanoGaussianState, batch_idx: int) -> torch.Tensor:
        if state.confidence is not None:
            return state.confidence[batch_idx].to(device=state.means.device, dtype=state.means.dtype)
        return torch.sigmoid(state.opacity_logits[batch_idx]).to(device=state.means.device, dtype=state.means.dtype)
