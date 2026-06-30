"""Typed configuration and containers for the PanoAnchorSplat frontend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import torch

from .resplat_types import PanoGaussianState, PanoRenderOutput


def _as_tuple_hw(value: Any | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if len(value) != 2:
        raise ValueError(f"image_hw must contain two values, got {value!r}")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class PanoAnchorSplatConfig:
    """Configuration for a local-window PanoAnchorSplat feed-forward model."""

    enabled: bool = False
    sh_degree: int = 2
    min_sh_degree: int = 2
    gaussians_per_anchor: int = 4
    max_anchors: int = 25000
    max_gaussians: int = 100000
    anchor_dim: int = 192
    decoder_dim: int = 192
    decoder_depth: int = 6
    decoder_heads: int = 6
    refiner_dim: int = 192
    error_dim: int = 128
    error_transformer_depth: int = 1
    point_transformer_depth: int = 2
    dtype: str = "bf16"
    batch_size_per_gpu: int = 1
    grad_accum_steps: int = 4
    voxel_size: float = 0.05
    min_scale: float = 0.002
    max_scale: float = 0.50
    max_offset_ratio: float = 0.75
    decoder_chunk_size: int = 2048
    num_global_tokens: int = 4
    raw_feature_bins: int = 160
    mean_delta_limit: float = 0.02
    log_scale_delta_limit: float = 0.05
    rotation_delta_limit: float = 0.05
    opacity_delta_limit: float = 0.25
    sh_delta_limit: float = 0.10

    def __post_init__(self) -> None:
        if int(self.min_sh_degree) < 2:
            raise ValueError("PanoAnchorSplat requires min_sh_degree >= 2.")
        if int(self.sh_degree) < int(self.min_sh_degree):
            raise ValueError("PanoAnchorSplat requires sh_degree >= min_sh_degree.")
        if int(self.gaussians_per_anchor) <= 0:
            raise ValueError("gaussians_per_anchor must be positive.")
        if int(self.max_anchors) <= 0 or int(self.max_gaussians) <= 0:
            raise ValueError("max_anchors and max_gaussians must be positive.")
        if int(self.effective_max_anchors) <= 0:
            raise ValueError("max_gaussians is too small for gaussians_per_anchor.")
        for name in ("anchor_dim", "decoder_dim", "refiner_dim", "error_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if float(self.voxel_size) <= 0.0:
            raise ValueError("voxel_size must be positive.")
        if float(self.min_scale) <= 0.0 or float(self.max_scale) <= float(self.min_scale):
            raise ValueError("Expected 0 < min_scale < max_scale.")
        _ = self.torch_dtype

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "PanoAnchorSplatConfig":
        """Create a config while ignoring unrelated parent config keys."""

        if config is None:
            return cls()
        valid = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in dict(config).items() if key in valid}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sh_dim(self) -> int:
        return int((int(self.sh_degree) + 1) ** 2)

    @property
    def effective_max_anchors(self) -> int:
        return min(int(self.max_anchors), int(self.max_gaussians) // int(self.gaussians_per_anchor))

    @property
    def torch_dtype(self) -> torch.dtype:
        value = str(self.dtype).lower()
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        if value in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported PanoAnchorSplat dtype={self.dtype!r}")

    @property
    def log_min_scale(self) -> float:
        return math.log(float(self.min_scale))

    @property
    def log_max_scale(self) -> float:
        return math.log(float(self.max_scale))


@dataclass
class PanoAnchorSplatPrior:
    """Detached PanoVGGT geometry/features for one local window.

    Shapes:
    - images: B x V x 3 x H x W
    - features: B x V x C x Hf x Wf
    - depths/confidence/valid_mask/sky_mask: B x V x 1 x H x W
    - poses_c2w: B x V x 4 x 4
    - world_points: B x V x H x W x 3
    """

    images: torch.Tensor
    features: torch.Tensor
    depths: torch.Tensor
    poses_c2w: torch.Tensor
    world_points: torch.Tensor
    valid_mask: torch.Tensor
    confidence: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    image_hw: tuple[int, int] | None = None
    feature_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.image_hw = _as_tuple_hw(self.image_hw) or (int(self.images.shape[-2]), int(self.images.shape[-1]))
        self.feature_hw = _as_tuple_hw(self.feature_hw) or (int(self.features.shape[-2]), int(self.features.shape[-1]))
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def num_views(self) -> int:
        return int(self.images.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def validate(self) -> None:
        if self.images.ndim != 5 or int(self.images.shape[2]) != 3:
            raise ValueError(f"images must have shape BxVx3xHxW, got {tuple(self.images.shape)}")
        b, v, _, h, w = [int(x) for x in self.images.shape]
        if self.features.ndim != 5 or tuple(self.features.shape[:2]) != (b, v):
            raise ValueError(f"features must have shape BxVxCxHfxWf, got {tuple(self.features.shape)}")
        if tuple(self.depths.shape) != (b, v, 1, h, w):
            raise ValueError(f"depths must have shape {(b, v, 1, h, w)}, got {tuple(self.depths.shape)}")
        if tuple(self.poses_c2w.shape) != (b, v, 4, 4):
            raise ValueError(f"poses_c2w must have shape {(b, v, 4, 4)}, got {tuple(self.poses_c2w.shape)}")
        if tuple(self.world_points.shape) != (b, v, h, w, 3):
            raise ValueError(f"world_points must have shape {(b, v, h, w, 3)}, got {tuple(self.world_points.shape)}")
        if tuple(self.valid_mask.shape) != (b, v, 1, h, w):
            raise ValueError(f"valid_mask must have shape {(b, v, 1, h, w)}, got {tuple(self.valid_mask.shape)}")
        if self.confidence is not None and tuple(self.confidence.shape) != (b, v, 1, h, w):
            raise ValueError(f"confidence must have shape {(b, v, 1, h, w)}, got {tuple(self.confidence.shape)}")
        if self.sky_mask is not None and tuple(self.sky_mask.shape) != (b, v, 1, h, w):
            raise ValueError(f"sky_mask must have shape {(b, v, 1, h, w)}, got {tuple(self.sky_mask.shape)}")

    def detach(self) -> "PanoAnchorSplatPrior":
        return PanoAnchorSplatPrior(
            images=self.images.detach(),
            features=self.features.detach(),
            depths=self.depths.detach(),
            poses_c2w=self.poses_c2w.detach(),
            world_points=self.world_points.detach(),
            valid_mask=self.valid_mask.detach(),
            confidence=None if self.confidence is None else self.confidence.detach(),
            sky_mask=None if self.sky_mask is None else self.sky_mask.detach(),
            image_hw=self.image_hw,
            feature_hw=self.feature_hw,
        )


@dataclass
class PanoAnchorSet:
    """Compact voxel anchors consumed by the AnchorSplat decoder.

    Shapes:
    - centers/scales: B x A x 3
    - features: B x A x C
    - confidence/counts: B x A x 1
    - source_view_ids: B x A
    - source_uv: B x A x 2 in ERP pixel coordinates
    - valid_mask: B x A
    """

    centers: torch.Tensor
    scales: torch.Tensor
    features: torch.Tensor
    confidence: torch.Tensor
    counts: torch.Tensor
    source_view_ids: torch.Tensor
    source_uv: torch.Tensor
    valid_mask: torch.Tensor
    image_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.image_hw = _as_tuple_hw(self.image_hw)
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.centers.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(self.centers.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    def validate(self) -> None:
        if self.centers.ndim != 3 or int(self.centers.shape[-1]) != 3:
            raise ValueError(f"centers must have shape BxAx3, got {tuple(self.centers.shape)}")
        b, a, _ = [int(x) for x in self.centers.shape]
        expected = {
            "scales": (b, a, 3),
            "confidence": (b, a, 1),
            "counts": (b, a, 1),
            "source_view_ids": (b, a),
            "source_uv": (b, a, 2),
            "valid_mask": (b, a),
        }
        values = {
            "scales": self.scales,
            "confidence": self.confidence,
            "counts": self.counts,
            "source_view_ids": self.source_view_ids,
            "source_uv": self.source_uv,
            "valid_mask": self.valid_mask,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")
        if self.features.ndim != 3 or tuple(self.features.shape[:2]) != (b, a):
            raise ValueError(f"features must have shape BxAxC, got {tuple(self.features.shape)}")


@dataclass
class PanoAnchorSplatOutput:
    """Forward output from ``PanoAnchorSplatFrontend``."""

    anchors: PanoAnchorSet
    init_state: PanoGaussianState
    final_state: PanoGaussianState
    target_render: PanoRenderOutput | None
    context_renders: list[PanoRenderOutput]
    debug: dict[str, Any]
