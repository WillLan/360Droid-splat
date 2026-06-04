"""Configuration parsing for the gated PanoVGGT-M3-Sphere extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"PanoVGGT.{name} must be a mapping.")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class MatchingHeadConfig:
    """Configuration for the future dense matching head contract."""

    enabled: bool = False
    checkpoint: str | None = None
    descriptor_dim: int = 24
    feature_hook: str | None = None


@dataclass(frozen=True)
class DenseMatchingConfig:
    """Configuration for future pose-guided dense matching."""

    enabled: bool = False
    search_radius: int = 4
    topk: int = 1
    min_match_confidence: float = 0.2
    min_static_confidence: float = 0.2
    max_factors: int = 65536
    use_wraparound: bool = True


@dataclass(frozen=True)
class DenseBAConfig:
    """Configuration for future spherical dense bundle adjustment."""

    enabled: bool = False
    residual_mode: str = "tangent"
    debug_pixel_residual: bool = False
    iters: int = 3
    lm: float = 1.0e-4
    fixed_frames: int = 1
    sample_stride: int = 1
    max_pose_step: float = 0.05
    max_depth_step: float = 0.10


@dataclass(frozen=True)
class InferenceWindowConfig:
    """Configuration for the future local inference/refinement window."""

    size: int = 4
    overlap: int = 2
    temporal_radius: int = 2
    max_edges: int = 24


@dataclass(frozen=True)
class M3SphereConfig:
    """Top-level PanoVGGT-M3-Sphere configuration."""

    enabled: bool = False
    descriptor_dim: int = 24
    matching_head: MatchingHeadConfig = MatchingHeadConfig()
    dense_matching: DenseMatchingConfig = DenseMatchingConfig()
    dense_ba: DenseBAConfig = DenseBAConfig()
    inference_window: InferenceWindowConfig = InferenceWindowConfig()


def parse_m3_sphere_config(config: dict[str, Any]) -> M3SphereConfig:
    """Parse ``PanoVGGT`` M3-Sphere configuration from a full or nested config."""

    pano_cfg = config.get("PanoVGGT", config)
    if pano_cfg is None:
        pano_cfg = {}
    if not isinstance(pano_cfg, dict):
        raise ValueError("PanoVGGT config must be a mapping.")

    m3_raw = _section(pano_cfg, "M3Sphere")
    head_raw = _section(pano_cfg, "MatchingHead")
    dense_raw = _section(pano_cfg, "DenseMatching")
    ba_raw = _section(pano_cfg, "DenseBA")
    window_raw = _section(pano_cfg, "InferenceWindow")

    descriptor_dim = _positive_int(
        head_raw.get("descriptor_dim", m3_raw.get("descriptor_dim", 24)),
        name="PanoVGGT.MatchingHead.descriptor_dim",
    )
    m3_descriptor_dim = _positive_int(
        m3_raw.get("descriptor_dim", descriptor_dim),
        name="PanoVGGT.M3Sphere.descriptor_dim",
    )

    matching_head = MatchingHeadConfig(
        enabled=bool(head_raw.get("enabled", False)),
        checkpoint=head_raw.get("checkpoint"),
        descriptor_dim=descriptor_dim,
        feature_hook=head_raw.get("feature_hook"),
    )
    dense_matching = DenseMatchingConfig(
        enabled=bool(dense_raw.get("enabled", False)),
        search_radius=_positive_int(dense_raw.get("search_radius", 4), name="PanoVGGT.DenseMatching.search_radius"),
        topk=_positive_int(dense_raw.get("topk", 1), name="PanoVGGT.DenseMatching.topk"),
        min_match_confidence=float(dense_raw.get("min_match_confidence", 0.2)),
        min_static_confidence=float(dense_raw.get("min_static_confidence", 0.2)),
        max_factors=_positive_int(dense_raw.get("max_factors", 65536), name="PanoVGGT.DenseMatching.max_factors"),
        use_wraparound=bool(dense_raw.get("use_wraparound", True)),
    )
    dense_ba = DenseBAConfig(
        enabled=bool(ba_raw.get("enabled", False)),
        residual_mode=str(ba_raw.get("residual_mode", "tangent")),
        debug_pixel_residual=bool(ba_raw.get("debug_pixel_residual", False)),
        iters=_positive_int(ba_raw.get("iters", 3), name="PanoVGGT.DenseBA.iters"),
        lm=float(ba_raw.get("lm", 1.0e-4)),
        fixed_frames=_positive_int(ba_raw.get("fixed_frames", 1), name="PanoVGGT.DenseBA.fixed_frames"),
        sample_stride=_positive_int(ba_raw.get("sample_stride", 1), name="PanoVGGT.DenseBA.sample_stride"),
        max_pose_step=float(ba_raw.get("max_pose_step", 0.05)),
        max_depth_step=float(ba_raw.get("max_depth_step", 0.10)),
    )
    inference_window = InferenceWindowConfig(
        size=_positive_int(window_raw.get("size", 4), name="PanoVGGT.InferenceWindow.size"),
        overlap=int(window_raw.get("overlap", 2)),
        temporal_radius=_positive_int(window_raw.get("temporal_radius", 2), name="PanoVGGT.InferenceWindow.temporal_radius"),
        max_edges=_positive_int(window_raw.get("max_edges", 24), name="PanoVGGT.InferenceWindow.max_edges"),
    )
    if inference_window.overlap < 0:
        raise ValueError("PanoVGGT.InferenceWindow.overlap must be non-negative.")

    return M3SphereConfig(
        enabled=bool(m3_raw.get("enabled", False)),
        descriptor_dim=m3_descriptor_dim,
        matching_head=matching_head,
        dense_matching=dense_matching,
        dense_ba=dense_ba,
        inference_window=inference_window,
    )
