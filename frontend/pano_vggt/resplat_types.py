"""Basic Pano-ReSplat Gaussian state and renderer-compatible materialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from backend.pano_gs.adapter import SH_C0
from .gaussian_head import ExplicitGaussianSet


@dataclass
class PanoGaussianState:
    """Batched feed-forward Gaussian state before renderer materialization.

    Shapes:
    - means: B x N x 3
    - log_scales: B x N x 3
    - rotations_unnorm: B x N x 4
    - opacity_logits: B x N x 1
    - sh_coeffs: B x N x 3 x SH_DIM
    - latent_features: B x N x C
    - source_view_ids: B x N
    - source_uv: B x N x 2
    - valid_mask: B x N
    - confidence: optional B x N x 1
    """

    means: torch.Tensor
    log_scales: torch.Tensor
    rotations_unnorm: torch.Tensor
    opacity_logits: torch.Tensor
    sh_coeffs: torch.Tensor
    latent_features: torch.Tensor
    source_view_ids: torch.Tensor
    source_uv: torch.Tensor
    valid_mask: torch.Tensor
    confidence: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[1])

    @property
    def sh_dim(self) -> int:
        return int(self.sh_coeffs.shape[-1])

    @property
    def latent_dim(self) -> int:
        return int(self.latent_features.shape[-1])

    def validate(self) -> None:
        if self.means.ndim != 3 or int(self.means.shape[-1]) != 3:
            raise ValueError(f"means must have shape BxNx3, got {tuple(self.means.shape)}")
        b, n, _ = [int(v) for v in self.means.shape]
        expected = {
            "log_scales": (b, n, 3),
            "rotations_unnorm": (b, n, 4),
            "opacity_logits": (b, n, 1),
            "source_view_ids": (b, n),
            "source_uv": (b, n, 2),
            "valid_mask": (b, n),
        }
        values = {
            "log_scales": self.log_scales,
            "rotations_unnorm": self.rotations_unnorm,
            "opacity_logits": self.opacity_logits,
            "source_view_ids": self.source_view_ids,
            "source_uv": self.source_uv,
            "valid_mask": self.valid_mask,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")
        if self.sh_coeffs.ndim != 4 or tuple(self.sh_coeffs.shape[:3]) != (b, n, 3):
            raise ValueError(f"sh_coeffs must have shape BxNx3xSH_DIM, got {tuple(self.sh_coeffs.shape)}")
        if self.latent_features.ndim != 3 or tuple(self.latent_features.shape[:2]) != (b, n):
            raise ValueError(f"latent_features must have shape BxNxC, got {tuple(self.latent_features.shape)}")
        if self.confidence is not None and tuple(self.confidence.shape) != (b, n, 1):
            raise ValueError(f"confidence must have shape {(b, n, 1)}, got {tuple(self.confidence.shape)}")

    def to(self, *args: Any, **kwargs: Any) -> "PanoGaussianState":
        """Return a copy with tensor fields moved through ``Tensor.to``."""

        means = self.means.to(*args, **kwargs)
        return replace(
            self,
            means=means,
            log_scales=self.log_scales.to(*args, **kwargs),
            rotations_unnorm=self.rotations_unnorm.to(*args, **kwargs),
            opacity_logits=self.opacity_logits.to(*args, **kwargs),
            sh_coeffs=self.sh_coeffs.to(*args, **kwargs),
            latent_features=self.latent_features.to(*args, **kwargs),
            source_view_ids=self.source_view_ids.to(device=means.device),
            source_uv=self.source_uv.to(*args, **kwargs),
            valid_mask=self.valid_mask.to(device=means.device),
            confidence=None if self.confidence is None else self.confidence.to(*args, **kwargs),
        )


@dataclass
class PanoRenderOutput:
    """Batched render output from a Pano-ReSplat renderer adapter."""

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    extras: dict[str, Any]


@dataclass
class ExplicitSHGaussianSet(ExplicitGaussianSet):
    """Explicit Gaussian set that preserves predicted SH coefficients.

    The existing renderer consumes ``get_*`` properties.  This subclass keeps
    that protocol while allowing ReSplat states to carry SH coefficients in the
    state layout ``G x 3 x SH_DIM``.
    """

    sh_coefficients: torch.Tensor | None = None

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        if self.sh_coefficients is None:
            return super().get_sh_coefficients
        return self.sh_coefficients.permute(0, 2, 1).contiguous()


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize quaternions and replace degenerate rows with identity."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dim 4, got {tuple(quaternion.shape)}")
    quat = torch.nan_to_num(quaternion, nan=0.0, posinf=0.0, neginf=0.0)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    return torch.where(norm > float(eps), quat / norm.clamp_min(float(eps)), identity)


def state_to_explicit_gaussian_set(
    state: PanoGaussianState,
    batch_index: int = 0,
    *,
    config: dict[str, Any] | None = None,
    min_scale: float = 1.0e-5,
    max_scale: float | None = None,
    active_sh_degree: int | None = None,
) -> ExplicitGaussianSet:
    """Materialize one batch item as a renderer-compatible explicit Gaussian set."""

    idx = int(batch_index)
    if idx < 0 or idx >= state.batch_size:
        raise IndexError(f"batch_index={idx} out of range for batch size {state.batch_size}")
    valid = state.valid_mask[idx].bool()
    scale_values = torch.exp(state.log_scales[idx])
    finite = (
        torch.isfinite(state.means[idx]).all(dim=-1)
        & torch.isfinite(state.log_scales[idx]).all(dim=-1)
        & torch.isfinite(scale_values).all(dim=-1)
        & torch.isfinite(state.opacity_logits[idx]).all(dim=-1)
        & torch.isfinite(state.sh_coeffs[idx]).all(dim=(-1, -2))
    )
    keep = valid & finite
    device = state.means.device
    dtype = state.means.dtype
    sh_dim = max(1, state.sh_dim)
    max_sh_degree = max(0, int(round(math.sqrt(sh_dim) - 1)))
    active_degree = max_sh_degree if active_sh_degree is None else int(active_sh_degree)
    if not bool(keep.any()):
        empty = torch.zeros(0, device=device, dtype=dtype)
        return ExplicitSHGaussianSet(
            xyz=empty.view(0, 3),
            scaling=empty.view(0, 3),
            rotation=empty.view(0, 4),
            opacity=empty.view(0, 1),
            features=empty.view(0, 3),
            config=config,
            active_sh_degree=active_degree,
            max_sh_degree=max_sh_degree,
            sh_coefficients=empty.view(0, 3, sh_dim),
        )

    scaling = scale_values[keep]
    sh_coeffs = torch.nan_to_num(state.sh_coeffs[idx][keep].to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    dc_rgb = (sh_coeffs[..., 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
    return ExplicitSHGaussianSet(
        xyz=torch.nan_to_num(state.means[idx][keep].to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0),
        scaling=torch.nan_to_num(scaling.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0),
        rotation=normalize_quaternion(state.rotations_unnorm[idx][keep].to(device=device, dtype=dtype)),
        opacity=torch.sigmoid(torch.nan_to_num(state.opacity_logits[idx][keep].to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)).clamp(0.0, 1.0),
        features=dc_rgb,
        config=config,
        active_sh_degree=active_degree,
        max_sh_degree=max_sh_degree,
        sh_coefficients=sh_coeffs,
    )
