"""Panoramic anchor feature encoder for PanoAnchorSplat."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class PanoAnchorFeatureEncoder(nn.Module):
    """Fuse voxel anchor geometry and PanoVGGT feature summaries.

    Input anchors are ``B x A``.  Output tokens have shape ``B x A x anchor_dim``.
    The raw PanoVGGT feature channel count can vary because it is pooled to a
    fixed number of bins before learned fusion.
    """

    def __init__(self, config: PanoAnchorSplatConfig | dict | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.raw_feature_bins = max(8, int(self.config.raw_feature_bins))
        geom_dim = 14
        geom_hidden = max(32, int(self.config.anchor_dim) // 4)
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden),
            nn.LayerNorm(geom_hidden),
            nn.GELU(),
            nn.Linear(geom_hidden, 32),
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.raw_feature_bins + 32, int(self.config.anchor_dim)),
            nn.LayerNorm(int(self.config.anchor_dim)),
            nn.GELU(),
            nn.Linear(int(self.config.anchor_dim), int(self.config.anchor_dim)),
            nn.LayerNorm(int(self.config.anchor_dim)),
        )

    def forward(self, anchors: PanoAnchorSet) -> torch.Tensor:
        raw = self._compress_raw_features(_finite(anchors.features), self.raw_feature_bins)
        geom = self._geometry_features(anchors)
        token = self.fuse(torch.cat([raw, self.geom_mlp(geom)], dim=-1))
        return torch.where(anchors.valid_mask.unsqueeze(-1), _finite(token), torch.zeros_like(token))

    @staticmethod
    def _compress_raw_features(features: torch.Tensor, bins: int) -> torch.Tensor:
        b, a, c = [int(x) for x in features.shape]
        if c == int(bins):
            return features
        pooled = F.adaptive_avg_pool1d(features.reshape(b * a, 1, c), int(bins))
        return pooled.reshape(b, a, int(bins))

    @staticmethod
    def _geometry_features(anchors: PanoAnchorSet) -> torch.Tensor:
        centers = _finite(anchors.centers)
        center_norm = centers.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        centers_n = centers / center_norm
        scales = anchors.scales.clamp_min(1.0e-6)
        log_scales = torch.log(scales)
        confidence = anchors.confidence.clamp(0.0, 1.0)
        counts = torch.log1p(anchors.counts.clamp_min(0.0))
        counts = counts / counts.detach().amax(dim=1, keepdim=True).clamp_min(1.0)
        uv = anchors.source_uv
        if anchors.image_hw is not None:
            h, w = anchors.image_hw
            uv_x = (uv[..., 0:1] / float(max(w - 1, 1))).clamp(0.0, 1.0)
            uv_y = (uv[..., 1:2] / float(max(h - 1, 1))).clamp(0.0, 1.0)
        else:
            denom = uv.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            uv_x, uv_y = (uv / denom).split(1, dim=-1)
        uv_feat = torch.cat(
            [
                torch.sin(uv_x * (2.0 * torch.pi)),
                torch.cos(uv_x * (2.0 * torch.pi)),
                torch.sin(uv_y * torch.pi),
                torch.cos(uv_y * torch.pi),
            ],
            dim=-1,
        )
        source = anchors.source_view_ids.to(dtype=centers.dtype).unsqueeze(-1)
        denom_source = source.detach().amax(dim=1, keepdim=True).clamp_min(1.0)
        source = source / denom_source
        valid = anchors.valid_mask.to(dtype=centers.dtype).unsqueeze(-1)
        return _finite(torch.cat([centers_n, log_scales, confidence, counts, uv_feat, source, valid], dim=-1))
