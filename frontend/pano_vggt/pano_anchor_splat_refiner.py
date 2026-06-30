"""Gaussian refiner-lite for PanoAnchorSplat."""

from __future__ import annotations

import torch
from torch import nn

from .pano_anchor_splat_decoder import ChunkedSelfAttentionBlock
from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig
from .resplat_types import PanoGaussianState


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class PanoAnchorGaussianRefinerLite(nn.Module):
    """Small anchor-token refiner driven by render-error tokens."""

    def __init__(self, config: PanoAnchorSplatConfig | dict | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.anchor_proj = nn.Linear(int(self.config.anchor_dim), int(self.config.refiner_dim))
        self.error_proj = nn.Linear(int(self.config.error_dim), int(self.config.refiner_dim))
        self.error_blocks = nn.ModuleList(
            [
                ChunkedSelfAttentionBlock(
                    int(self.config.refiner_dim),
                    int(self.config.decoder_heads),
                    chunk_size=int(self.config.decoder_chunk_size),
                    num_global_tokens=int(self.config.num_global_tokens),
                )
                for _ in range(max(0, int(self.config.error_transformer_depth)))
            ]
        )
        self.point_blocks = nn.ModuleList(
            [
                ChunkedSelfAttentionBlock(
                    int(self.config.refiner_dim),
                    int(self.config.decoder_heads),
                    chunk_size=int(self.config.decoder_chunk_size),
                    num_global_tokens=int(self.config.num_global_tokens),
                )
                for _ in range(max(0, int(self.config.point_transformer_depth)))
            ]
        )
        self.delta_dim = 3 + 3 + 4 + 1 + 3 * int(self.config.sh_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(self.config.refiner_dim)),
            nn.Linear(int(self.config.refiner_dim), int(self.config.refiner_dim)),
            nn.GELU(),
            nn.Linear(int(self.config.refiner_dim), int(self.config.gaussians_per_anchor) * self.delta_dim),
        )
        last = self.delta_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        state: PanoGaussianState,
        anchors: PanoAnchorSet,
        anchor_tokens: torch.Tensor,
        error_tokens: torch.Tensor,
    ) -> tuple[PanoGaussianState, dict[str, torch.Tensor]]:
        if tuple(anchor_tokens.shape[:2]) != (anchors.batch_size, anchors.num_anchors):
            raise ValueError("anchor_tokens must share B,A with anchors.")
        if tuple(error_tokens.shape[:2]) != (anchors.batch_size, anchors.num_anchors):
            raise ValueError("error_tokens must share B,A with anchors.")
        err = self.error_proj(error_tokens)
        for block in self.error_blocks:
            err = block(err, anchors.valid_mask)
        x = self.anchor_proj(anchor_tokens) + err
        for block in self.point_blocks:
            x = block(x, anchors.valid_mask)
        raw = self.delta_head(x).view(
            anchors.batch_size,
            anchors.num_anchors,
            int(self.config.gaussians_per_anchor),
            self.delta_dim,
        )
        refined = self._apply_deltas(state, raw)
        metrics = {
            "refiner_mean_delta_abs": (refined.means - state.means).detach().abs().mean(),
            "refiner_log_scale_delta_abs": (refined.log_scales - state.log_scales).detach().abs().mean(),
            "refiner_opacity_delta_abs": (refined.opacity_logits - state.opacity_logits).detach().abs().mean(),
            "refiner_sh_delta_abs": (refined.sh_coeffs - state.sh_coeffs).detach().abs().mean(),
        }
        return refined, metrics

    def _apply_deltas(self, state: PanoGaussianState, raw: torch.Tensor) -> PanoGaussianState:
        b, a, k, _ = [int(x) for x in raw.shape]
        cursor = 0
        d_mean = torch.tanh(raw[..., cursor : cursor + 3]) * float(self.config.mean_delta_limit)
        cursor += 3
        d_scale = torch.tanh(raw[..., cursor : cursor + 3]) * float(self.config.log_scale_delta_limit)
        cursor += 3
        d_rot = torch.tanh(raw[..., cursor : cursor + 4]) * float(self.config.rotation_delta_limit)
        cursor += 4
        d_opacity = torch.tanh(raw[..., cursor : cursor + 1]) * float(self.config.opacity_delta_limit)
        cursor += 1
        d_sh = torch.tanh(raw[..., cursor : cursor + 3 * int(self.config.sh_dim)]).view(b, a, k, 3, int(self.config.sh_dim))
        d_sh = d_sh * float(self.config.sh_delta_limit)
        n_total = a * k
        n = state.num_gaussians
        d_mean_f = d_mean.reshape(b, n_total, 3)[:, :n]
        d_scale_f = d_scale.reshape(b, n_total, 3)[:, :n]
        d_rot_f = d_rot.reshape(b, n_total, 4)[:, :n]
        d_opacity_f = d_opacity.reshape(b, n_total, 1)[:, :n]
        d_sh_f = d_sh.reshape(b, n_total, 3, int(self.config.sh_dim))[:, :n]
        return PanoGaussianState(
            means=_finite(state.means + d_mean_f),
            log_scales=_finite((state.log_scales + d_scale_f).clamp(float(self.config.log_min_scale), float(self.config.log_max_scale))),
            rotations_unnorm=_finite(state.rotations_unnorm + d_rot_f),
            opacity_logits=_finite(state.opacity_logits + d_opacity_f),
            sh_coeffs=_finite(state.sh_coeffs + d_sh_f),
            latent_features=state.latent_features,
            source_view_ids=state.source_view_ids,
            source_uv=state.source_uv,
            valid_mask=state.valid_mask,
            confidence=state.confidence,
        )
