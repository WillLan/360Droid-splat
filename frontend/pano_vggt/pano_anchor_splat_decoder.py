"""Lightweight AnchorSplat Gaussian decoder."""

from __future__ import annotations

import torch
from torch import nn

from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig
from .resplat_types import PanoGaussianState


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _compatible_heads(dim: int, heads: int) -> int:
    heads = max(1, int(heads))
    while heads > 1 and int(dim) % heads != 0:
        heads -= 1
    return heads


class ChunkedSelfAttentionBlock(nn.Module):
    """Windowed self-attention over long anchor token sequences."""

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        chunk_size: int = 2048,
        num_global_tokens: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.chunk_size = max(1, int(chunk_size))
        self.num_global_tokens = max(0, int(num_global_tokens))
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, _compatible_heads(self.dim, heads), batch_first=True)
        self.norm2 = nn.LayerNorm(self.dim)
        hidden = max(self.dim, int(round(self.dim * float(mlp_ratio))))
        self.mlp = nn.Sequential(nn.Linear(self.dim, hidden), nn.GELU(), nn.Linear(hidden, self.dim))
        if self.num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.zeros(self.num_global_tokens, self.dim))
            nn.init.normal_(self.global_tokens, std=0.02)
        else:
            self.register_parameter("global_tokens", None)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape BxAxC, got {tuple(tokens.shape)}")
        b, a, c = [int(x) for x in tokens.shape]
        if c != self.dim:
            raise ValueError(f"Expected token dim={self.dim}, got {c}")
        y = self.norm1(tokens)
        attended = torch.zeros_like(y)
        for start in range(0, a, self.chunk_size):
            end = min(a, start + self.chunk_size)
            chunk = y[:, start:end]
            chunk_valid = valid_mask[:, start:end].bool()
            if self.global_tokens is not None:
                global_tokens = self.global_tokens.to(device=tokens.device, dtype=tokens.dtype).view(1, self.num_global_tokens, c).expand(b, -1, -1)
                attn_input = torch.cat([global_tokens, chunk], dim=1)
                pad = torch.cat(
                    [
                        torch.zeros(b, self.num_global_tokens, device=tokens.device, dtype=torch.bool),
                        ~chunk_valid,
                    ],
                    dim=1,
                )
                offset = self.num_global_tokens
            else:
                attn_input = chunk
                pad = ~chunk_valid
                offset = 0
                if bool(pad.all()):
                    continue
            out, _ = self.attn(attn_input, attn_input, attn_input, key_padding_mask=pad, need_weights=False)
            attended[:, start:end] = out[:, offset:]
        x = tokens + attended
        x = x + self.mlp(self.norm2(x))
        return torch.where(valid_mask.unsqueeze(-1), _finite(x), torch.zeros_like(x))


class PanoAnchorGaussianDecoder(nn.Module):
    """Decode anchor tokens into SH2 Gaussian states."""

    def __init__(self, config: PanoAnchorSplatConfig | dict | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.input_proj = nn.Linear(int(self.config.anchor_dim), int(self.config.decoder_dim))
        self.blocks = nn.ModuleList(
            [
                ChunkedSelfAttentionBlock(
                    int(self.config.decoder_dim),
                    int(self.config.decoder_heads),
                    chunk_size=int(self.config.decoder_chunk_size),
                    num_global_tokens=int(self.config.num_global_tokens),
                )
                for _ in range(max(0, int(self.config.decoder_depth)))
            ]
        )
        self.gaussian_param_dim = 3 + 3 + 4 + 1 + 3 * int(self.config.sh_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(int(self.config.decoder_dim)),
            nn.Linear(int(self.config.decoder_dim), int(self.config.decoder_dim)),
            nn.GELU(),
            nn.Linear(int(self.config.decoder_dim), int(self.config.gaussians_per_anchor) * self.gaussian_param_dim),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.bias)

    def forward(self, anchors: PanoAnchorSet, anchor_tokens: torch.Tensor) -> PanoGaussianState:
        if anchor_tokens.ndim != 3 or tuple(anchor_tokens.shape[:2]) != (anchors.batch_size, anchors.num_anchors):
            raise ValueError("anchor_tokens must have shape BxAxC and share B,A with anchors.")
        x = self.input_proj(anchor_tokens)
        for block in self.blocks:
            x = block(x, anchors.valid_mask)
        raw = self.head(x).view(
            anchors.batch_size,
            anchors.num_anchors,
            int(self.config.gaussians_per_anchor),
            self.gaussian_param_dim,
        )
        return self._materialize(anchors, x, raw)

    def _materialize(self, anchors: PanoAnchorSet, tokens: torch.Tensor, raw: torch.Tensor) -> PanoGaussianState:
        b, a, k, _ = [int(x) for x in raw.shape]
        cursor = 0
        offset = torch.tanh(raw[..., cursor : cursor + 3])
        cursor += 3
        scale_delta = torch.tanh(raw[..., cursor : cursor + 3])
        cursor += 3
        rotation_delta = raw[..., cursor : cursor + 4]
        cursor += 4
        opacity = raw[..., cursor : cursor + 1]
        cursor += 1
        sh = raw[..., cursor : cursor + 3 * int(self.config.sh_dim)].view(b, a, k, 3, int(self.config.sh_dim))

        base_scale = anchors.scales.clamp(float(self.config.min_scale), float(self.config.max_scale)).unsqueeze(2)
        means = anchors.centers.unsqueeze(2) + offset * base_scale * float(self.config.max_offset_ratio)
        log_scales = torch.log(base_scale).expand(-1, -1, k, -1) + scale_delta * 0.5
        log_scales = log_scales.clamp(float(self.config.log_min_scale), float(self.config.log_max_scale))
        rotations = rotation_delta.clone()
        rotations[..., 0] = rotations[..., 0] + 1.0
        latent = tokens.unsqueeze(2).expand(-1, -1, k, -1)
        source_view_ids = anchors.source_view_ids.unsqueeze(-1).expand(-1, -1, k)
        source_uv = anchors.source_uv.unsqueeze(2).expand(-1, -1, k, -1)
        valid = anchors.valid_mask.unsqueeze(-1).expand(-1, -1, k)
        confidence = anchors.confidence.unsqueeze(2).expand(-1, -1, k, -1)

        n_total = a * k
        max_gaussians = min(int(self.config.max_gaussians), n_total)
        return PanoGaussianState(
            means=_finite(means.reshape(b, n_total, 3)[:, :max_gaussians]),
            log_scales=_finite(log_scales.reshape(b, n_total, 3)[:, :max_gaussians]),
            rotations_unnorm=_finite(rotations.reshape(b, n_total, 4)[:, :max_gaussians]),
            opacity_logits=_finite(opacity.reshape(b, n_total, 1)[:, :max_gaussians]),
            sh_coeffs=_finite(sh.reshape(b, n_total, 3, int(self.config.sh_dim))[:, :max_gaussians]),
            latent_features=_finite(latent.reshape(b, n_total, int(tokens.shape[-1]))[:, :max_gaussians]),
            source_view_ids=source_view_ids.reshape(b, n_total)[:, :max_gaussians].to(dtype=torch.long),
            source_uv=_finite(source_uv.reshape(b, n_total, 2)[:, :max_gaussians]),
            valid_mask=valid.reshape(b, n_total)[:, :max_gaussians].bool(),
            confidence=confidence.reshape(b, n_total, 1)[:, :max_gaussians],
        )
