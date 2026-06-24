"""Recurrent context-feedback Gaussian update blocks for Pano-ReSplat."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .pano_point_transformer import PanoKNNTransformerBlock
from .resplat_types import PanoGaussianState


@dataclass(frozen=True)
class PanoGaussianUpdateLimits:
    mean: float = 0.02
    log_scale: float = 0.05
    rotation: float = 0.05
    opacity: float = 0.25
    sh: float = 0.10
    latent: float = 0.10
    min_scale: float = 1.0e-5
    max_scale: float = 0.50


class PanoGaussianUpdateBlock(nn.Module):
    """Update Gaussian state from context-only feedback."""

    def __init__(
        self,
        *,
        feedback_dim: int,
        latent_dim: int,
        sh_dim: int = 1,
        hidden_dim: int = 64,
        knn: int = 8,
        num_heads: int = 4,
        limits: PanoGaussianUpdateLimits | None = None,
        max_knn_points: int = 2048,
        chunk_size: int | None = None,
        attn_proj_channels: int | None = None,
        mlp_ratio: float = 2.0,
        num_basic_refine_blocks: int = 1,
        knn_backend: str = "cdist",
        strict_knn_backend: bool = False,
        cache_knn: bool = False,
        detach_feedback: bool = True,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.latent_dim = int(latent_dim)
        self.sh_dim = int(sh_dim)
        self.hidden_dim = int(hidden_dim)
        self.limits = limits or PanoGaussianUpdateLimits()
        self.cache_knn = bool(cache_knn)
        self.detach_feedback = bool(detach_feedback)
        self.gradient_checkpoint = bool(gradient_checkpoint)
        self.num_basic_refine_blocks = max(1, int(num_basic_refine_blocks))
        input_dim = 3 + 3 + 4 + 1 + 3 * self.sh_dim + self.latent_dim + self.feedback_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.transformers = nn.ModuleList(
            [
                PanoKNNTransformerBlock(
                    self.hidden_dim,
                    num_heads=num_heads,
                    knn=knn,
                    mlp_ratio=mlp_ratio,
                    max_knn_points=max_knn_points,
                    chunk_size=chunk_size,
                    attn_proj_channels=attn_proj_channels,
                    knn_backend=knn_backend,
                    strict_knn_backend=strict_knn_backend,
                )
                for _ in range(self.num_basic_refine_blocks)
            ]
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3 + 3 + 4 + 1 + 3 * self.sh_dim + self.latent_dim),
        )
        self._zero_init_delta()

    @property
    def transformer(self) -> PanoKNNTransformerBlock:
        return self.transformers[0]

    def _zero_init_delta(self) -> None:
        last = self.delta[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        state: PanoGaussianState,
        feedback: torch.Tensor,
    ) -> tuple[PanoGaussianState, dict[str, torch.Tensor]]:
        if feedback.ndim != 3 or tuple(feedback.shape[:2]) != tuple(state.means.shape[:2]):
            raise ValueError(f"feedback must have shape BxNxCfb, got {tuple(feedback.shape)}")
        if int(feedback.shape[-1]) != self.feedback_dim:
            raise ValueError(f"Expected feedback_dim={self.feedback_dim}, got {int(feedback.shape[-1])}")
        if state.latent_dim != self.latent_dim:
            raise ValueError(f"Expected state latent_dim={self.latent_dim}, got {state.latent_dim}")
        if state.sh_dim != self.sh_dim:
            raise ValueError(f"Expected state sh_dim={self.sh_dim}, got {state.sh_dim}")

        means_prev = state.means.detach()
        log_scales_prev = state.log_scales.detach()
        rotations_prev = state.rotations_unnorm.detach()
        opacity_prev = state.opacity_logits.detach()
        sh_prev = state.sh_coeffs.detach()
        latent_prev = state.latent_features.detach()
        feedback_prev = feedback.detach() if self.detach_feedback else feedback
        confidence = (
            torch.ones_like(opacity_prev)
            if state.confidence is None
            else state.confidence.detach().to(device=state.means.device, dtype=state.means.dtype)
        )
        x = torch.cat(
            [
                means_prev,
                log_scales_prev,
                rotations_prev,
                opacity_prev,
                sh_prev.reshape(state.batch_size, state.num_gaussians, -1),
                latent_prev,
                feedback_prev,
                confidence,
            ],
            dim=-1,
        )
        feat = self.input_proj(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        knn_cache = self.transformers[0].compute_knn_cache(means_prev, state.valid_mask) if self.cache_knn else None
        for block in self.transformers:
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                def _run_block(feat_in: torch.Tensor, *, current_block: PanoKNNTransformerBlock = block) -> torch.Tensor:
                    return current_block(means_prev, feat_in, state.valid_mask, knn_cache=knn_cache)

                feat = checkpoint(_run_block, feat, use_reentrant=False)
            elif self.cache_knn:
                feat = block(means_prev, feat, state.valid_mask, knn_cache=knn_cache)
            else:
                feat = block(means_prev, feat, state.valid_mask)
        raw = self.delta(feat)
        cursor = 0
        mean_delta = torch.tanh(raw[..., cursor : cursor + 3]) * float(self.limits.mean)
        cursor += 3
        scale_delta = torch.tanh(raw[..., cursor : cursor + 3]) * float(self.limits.log_scale)
        cursor += 3
        rot_delta = torch.tanh(raw[..., cursor : cursor + 4]) * float(self.limits.rotation)
        cursor += 4
        opacity_delta = torch.tanh(raw[..., cursor : cursor + 1]) * float(self.limits.opacity)
        cursor += 1
        sh_delta = torch.tanh(raw[..., cursor : cursor + 3 * self.sh_dim]).view(state.batch_size, state.num_gaussians, 3, self.sh_dim)
        sh_delta = sh_delta * float(self.limits.sh)
        cursor += 3 * self.sh_dim
        latent_delta = torch.tanh(raw[..., cursor : cursor + self.latent_dim]) * float(self.limits.latent)

        valid = state.valid_mask.unsqueeze(-1).to(dtype=state.means.dtype)
        new_state = PanoGaussianState(
            means=torch.nan_to_num(state.means + mean_delta * valid, nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(
                (state.log_scales + scale_delta * valid).clamp(
                    math.log(float(self.limits.min_scale)),
                    math.log(float(self.limits.max_scale)),
                ),
                nan=math.log(float(self.limits.min_scale)),
                posinf=math.log(float(self.limits.max_scale)),
                neginf=math.log(float(self.limits.min_scale)),
            ),
            rotations_unnorm=torch.nan_to_num(state.rotations_unnorm + rot_delta * valid, nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(state.opacity_logits + opacity_delta * valid, nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(state.sh_coeffs + sh_delta * valid.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(state.latent_features + latent_delta * valid, nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=state.source_view_ids,
            source_uv=state.source_uv,
            valid_mask=state.valid_mask,
            confidence=state.confidence,
        )
        metrics = {
            "mean_delta_abs": mean_delta.abs().mean().detach(),
            "log_scale_delta_abs": scale_delta.abs().mean().detach(),
            "rotation_delta_abs": rot_delta.abs().mean().detach(),
            "opacity_delta_abs": opacity_delta.abs().mean().detach(),
            "sh_delta_abs": sh_delta.abs().mean().detach(),
            "latent_delta_abs": latent_delta.abs().mean().detach(),
        }
        return new_state, metrics
