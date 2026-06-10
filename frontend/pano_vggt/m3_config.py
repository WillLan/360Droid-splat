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


def _nonnegative_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class MatchingHeadConfig:
    """Configuration for the gated dense matching and sky heads."""

    enabled: bool = False
    checkpoint: str | None = None
    matching_checkpoint: str | None = None
    sky_checkpoint: str | None = None
    descriptor_dim: int = 24
    feature_hook: str | None = None
    feature_key: str | int | None = None
    strict: bool = True
    allow_fake_matching: bool = False
    fake_feature_stride: int = 4


@dataclass(frozen=True)
class DenseMatchingConfig:
    """Configuration for pose-guided dense matching."""

    enabled: bool = False
    search_radius: int = 4
    topk: int = 1
    min_match_confidence: float = 0.2
    min_static_confidence: float = 0.2
    min_match_score: float = 0.0
    max_factors: int = 8192
    max_samples_per_edge: int | None = None
    use_wraparound: bool = True
    forward_backward: bool = True
    fb_tolerance: float = 1.5
    use_depth_consistency: bool = True
    depth_consistency_rel: float = 0.03
    depth_consistency_abs: float = 0.05


@dataclass(frozen=True)
class DenseBAConfig:
    """Configuration for spherical tangent dense bundle adjustment."""

    enabled: bool = False
    residual_mode: str = "tangent"
    debug_pixel_residual: bool = False
    shadow_mode: bool = True
    mode: str = "local_chunk"
    solver_mode: str = "pose_only_factor_graph"
    fixed_policy: str = "first_frame"
    iters: int = 3
    lm: float = 1.0e-4
    fixed_frames: int = 1
    sample_stride: int = 1
    min_valid_factor_ratio: float = 0.03
    min_num_factors: int = 256
    huber_delta_deg: float = 0.5
    pose_prior_weight: float = 1.0e-3
    depth_prior_weight: float = 1.0e-2
    max_pose_step: float = 0.05
    max_depth_step: float = 0.10
    max_pose_update_deg: float = 5.0
    max_logdepth_update: float = 0.35
    fallback_if_residual_worse: bool = True
    residual_worse_tolerance: float = 1.05
    factor_chunk_size: int = 2048
    max_ba_factors: int = 8192
    max_depth_variables: int = 0
    max_solver_sec: float = 8.0
    optimize_pose: bool = True
    optimize_depth: bool = False
    history_keyframes: int = 8
    depth_update_policy: str = "max"
    logdepth_update_quantile: float = 1.0
    line_search: bool = False


@dataclass(frozen=True)
class JointInferenceConfig:
    """Configuration for recent-history joint PanoVGGT inference."""

    enabled: bool = False
    history_policy: str = "recent"
    min_history_frames: int = 0
    max_history_frames: int = 3


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for history-aware chunk alignment."""

    use_common_history: bool = False
    history_point_budget_ratio: float = 0.5
    exclude_sky: bool = True
    sky_threshold: float = 0.5


@dataclass(frozen=True)
class KeyframeAnchorConfig:
    """Configuration for previous-keyframe anchor matching."""

    enabled: bool = False
    prepend_previous_keyframe: bool = True
    pair_confidence_mode: str = "product"
    cell_pair_conf_threshold: float = 0.25
    frame_mean_pair_conf_threshold: float = 0.30
    frame_low_pair_conf_ratio: float = 0.45
    match_coverage_threshold: float = 0.0
    translation_threshold: float = 0.75
    translation_depth_ratio_threshold: float = 0.08
    m3_score_threshold: float = -1.0
    min_keyframe_interval: int = 0
    max_keyframe_interval: int = 0
    sky_threshold: float = 0.5


@dataclass(frozen=True)
class KeyframeGraphConfig:
    """Configuration for the gated persistent keyframe correspondence graph."""

    enabled: bool = False
    current_to_last_ba: bool = True
    adjacent_edges: bool = True
    retrieval_edges: bool = False
    loop_edges: bool = False
    adjacent_history: int = 1
    window_keyframes: int = 16
    optimize_every_keyframes: int = 1
    fixed_keyframes: int = 1
    min_valid_factors: int = 256
    min_valid_factor_ratio: float = 0.03
    max_factors_per_edge: int = 8192
    publish_pose_updates: bool = False


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
    keyframe_anchor: KeyframeAnchorConfig = KeyframeAnchorConfig()
    keyframe_graph: KeyframeGraphConfig = KeyframeGraphConfig()
    inference_window: InferenceWindowConfig = InferenceWindowConfig()
    joint_inference: JointInferenceConfig = JointInferenceConfig()
    alignment: AlignmentConfig = AlignmentConfig()


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
    keyframe_anchor_raw = _section(pano_cfg, "KeyframeAnchor")
    keyframe_graph_raw = _section(pano_cfg, "KeyframeGraph")
    window_raw = _section(pano_cfg, "InferenceWindow")
    joint_raw = _section(pano_cfg, "JointInference")
    alignment_raw = _section(pano_cfg, "Alignment")

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
        matching_checkpoint=head_raw.get("matching_checkpoint"),
        sky_checkpoint=head_raw.get("sky_checkpoint"),
        descriptor_dim=descriptor_dim,
        feature_hook=head_raw.get("feature_hook"),
        feature_key=head_raw.get("feature_key"),
        strict=bool(head_raw.get("strict", True)),
        allow_fake_matching=bool(head_raw.get("allow_fake_matching", False)),
        fake_feature_stride=_positive_int(
            head_raw.get("fake_feature_stride", 4),
            name="PanoVGGT.MatchingHead.fake_feature_stride",
        ),
    )
    search_radius_raw = dense_raw.get("search_radius", dense_raw.get("radius", 4))
    forward_backward_raw = dense_raw.get("forward_backward", dense_raw.get("forward_backward_check", True))
    depth_consistency_raw = dense_raw.get("use_depth_consistency", dense_raw.get("depth_consistency_check", True))
    fb_tolerance_raw = dense_raw.get("fb_tolerance", dense_raw.get("fb_thresh_deg", 1.5))
    max_samples_per_edge_raw = dense_raw.get("max_samples_per_edge")
    dense_matching = DenseMatchingConfig(
        enabled=bool(dense_raw.get("enabled", False)),
        search_radius=_positive_int(search_radius_raw, name="PanoVGGT.DenseMatching.search_radius"),
        topk=_positive_int(dense_raw.get("topk", 1), name="PanoVGGT.DenseMatching.topk"),
        min_match_confidence=float(dense_raw.get("min_match_confidence", 0.2)),
        min_static_confidence=float(dense_raw.get("min_static_confidence", 0.2)),
        min_match_score=float(dense_raw.get("min_match_score", 0.0)),
        max_factors=_positive_int(dense_raw.get("max_factors", 8192), name="PanoVGGT.DenseMatching.max_factors"),
        max_samples_per_edge=(
            None
            if max_samples_per_edge_raw is None
            else _positive_int(max_samples_per_edge_raw, name="PanoVGGT.DenseMatching.max_samples_per_edge")
        ),
        use_wraparound=bool(dense_raw.get("use_wraparound", True)),
        forward_backward=bool(forward_backward_raw),
        fb_tolerance=float(fb_tolerance_raw),
        use_depth_consistency=bool(depth_consistency_raw),
        depth_consistency_rel=float(dense_raw.get("depth_consistency_rel", 0.03)),
        depth_consistency_abs=float(dense_raw.get("depth_consistency_abs", 0.05)),
    )
    dense_ba = DenseBAConfig(
        enabled=bool(ba_raw.get("enabled", False)),
        residual_mode=str(ba_raw.get("residual_mode", "tangent")),
        debug_pixel_residual=bool(ba_raw.get("debug_pixel_residual", False)),
        shadow_mode=bool(ba_raw.get("shadow_mode", True)),
        mode=str(ba_raw.get("mode", "local_chunk")),
        solver_mode=str(ba_raw.get("solver_mode", "pose_only_factor_graph")),
        fixed_policy=str(ba_raw.get("fixed_policy", "first_frame")),
        iters=_positive_int(ba_raw.get("iters", 3), name="PanoVGGT.DenseBA.iters"),
        lm=float(ba_raw.get("lm", 1.0e-4)),
        fixed_frames=_positive_int(ba_raw.get("fixed_frames", 1), name="PanoVGGT.DenseBA.fixed_frames"),
        sample_stride=_positive_int(ba_raw.get("sample_stride", 1), name="PanoVGGT.DenseBA.sample_stride"),
        min_valid_factor_ratio=float(ba_raw.get("min_valid_factor_ratio", 0.03)),
        min_num_factors=_positive_int(ba_raw.get("min_num_factors", 256), name="PanoVGGT.DenseBA.min_num_factors"),
        huber_delta_deg=float(ba_raw.get("huber_delta_deg", 0.5)),
        pose_prior_weight=float(ba_raw.get("pose_prior_weight", 1.0e-3)),
        depth_prior_weight=float(ba_raw.get("depth_prior_weight", 1.0e-2)),
        max_pose_step=float(ba_raw.get("max_pose_step", 0.05)),
        max_depth_step=float(ba_raw.get("max_depth_step", 0.10)),
        max_pose_update_deg=float(ba_raw.get("max_pose_update_deg", 5.0)),
        max_logdepth_update=float(ba_raw.get("max_logdepth_update", 0.35)),
        fallback_if_residual_worse=bool(ba_raw.get("fallback_if_residual_worse", True)),
        residual_worse_tolerance=float(ba_raw.get("residual_worse_tolerance", 1.05)),
        factor_chunk_size=_positive_int(ba_raw.get("factor_chunk_size", 2048), name="PanoVGGT.DenseBA.factor_chunk_size"),
        max_ba_factors=_positive_int(ba_raw.get("max_ba_factors", 8192), name="PanoVGGT.DenseBA.max_ba_factors"),
        max_depth_variables=_nonnegative_int(
            ba_raw.get("max_depth_variables", 0),
            name="PanoVGGT.DenseBA.max_depth_variables",
        ),
        max_solver_sec=float(ba_raw.get("max_solver_sec", 8.0)),
        optimize_pose=bool(ba_raw.get("optimize_pose", True)),
        optimize_depth=bool(ba_raw.get("optimize_depth", False)),
        history_keyframes=_nonnegative_int(
            ba_raw.get("history_keyframes", 8),
            name="PanoVGGT.DenseBA.history_keyframes",
        ),
        depth_update_policy=str(ba_raw.get("depth_update_policy", "max")),
        logdepth_update_quantile=float(ba_raw.get("logdepth_update_quantile", 1.0)),
        line_search=bool(ba_raw.get("line_search", False)),
    )
    keyframe_anchor = KeyframeAnchorConfig(
        enabled=bool(keyframe_anchor_raw.get("enabled", False)),
        prepend_previous_keyframe=bool(keyframe_anchor_raw.get("prepend_previous_keyframe", True)),
        pair_confidence_mode=str(keyframe_anchor_raw.get("pair_confidence_mode", "product")),
        cell_pair_conf_threshold=float(keyframe_anchor_raw.get("cell_pair_conf_threshold", 0.25)),
        frame_mean_pair_conf_threshold=float(keyframe_anchor_raw.get("frame_mean_pair_conf_threshold", 0.30)),
        frame_low_pair_conf_ratio=float(keyframe_anchor_raw.get("frame_low_pair_conf_ratio", 0.45)),
        match_coverage_threshold=float(keyframe_anchor_raw.get("match_coverage_threshold", 0.0)),
        translation_threshold=float(keyframe_anchor_raw.get("translation_threshold", 0.75)),
        translation_depth_ratio_threshold=float(keyframe_anchor_raw.get("translation_depth_ratio_threshold", 0.08)),
        m3_score_threshold=float(keyframe_anchor_raw.get("m3_score_threshold", -1.0)),
        min_keyframe_interval=_nonnegative_int(
            keyframe_anchor_raw.get("min_keyframe_interval", 0),
            name="PanoVGGT.KeyframeAnchor.min_keyframe_interval",
        ),
        max_keyframe_interval=_nonnegative_int(
            keyframe_anchor_raw.get("max_keyframe_interval", 0),
            name="PanoVGGT.KeyframeAnchor.max_keyframe_interval",
        ),
        sky_threshold=float(keyframe_anchor_raw.get("sky_threshold", 0.5)),
    )
    keyframe_graph = KeyframeGraphConfig(
        enabled=bool(keyframe_graph_raw.get("enabled", False)),
        current_to_last_ba=bool(keyframe_graph_raw.get("current_to_last_ba", True)),
        adjacent_edges=bool(keyframe_graph_raw.get("adjacent_edges", True)),
        retrieval_edges=bool(keyframe_graph_raw.get("retrieval_edges", False)),
        loop_edges=bool(keyframe_graph_raw.get("loop_edges", False)),
        adjacent_history=_positive_int(
            keyframe_graph_raw.get("adjacent_history", 1),
            name="PanoVGGT.KeyframeGraph.adjacent_history",
        ),
        window_keyframes=_positive_int(
            keyframe_graph_raw.get("window_keyframes", 16),
            name="PanoVGGT.KeyframeGraph.window_keyframes",
        ),
        optimize_every_keyframes=_positive_int(
            keyframe_graph_raw.get("optimize_every_keyframes", 1),
            name="PanoVGGT.KeyframeGraph.optimize_every_keyframes",
        ),
        fixed_keyframes=_positive_int(
            keyframe_graph_raw.get("fixed_keyframes", 1),
            name="PanoVGGT.KeyframeGraph.fixed_keyframes",
        ),
        min_valid_factors=_positive_int(
            keyframe_graph_raw.get("min_valid_factors", 256),
            name="PanoVGGT.KeyframeGraph.min_valid_factors",
        ),
        min_valid_factor_ratio=float(keyframe_graph_raw.get("min_valid_factor_ratio", 0.03)),
        max_factors_per_edge=_positive_int(
            keyframe_graph_raw.get("max_factors_per_edge", 8192),
            name="PanoVGGT.KeyframeGraph.max_factors_per_edge",
        ),
        publish_pose_updates=bool(keyframe_graph_raw.get("publish_pose_updates", False)),
    )
    inference_window = InferenceWindowConfig(
        size=_positive_int(window_raw.get("size", 4), name="PanoVGGT.InferenceWindow.size"),
        overlap=int(window_raw.get("overlap", 2)),
        temporal_radius=_positive_int(window_raw.get("temporal_radius", 2), name="PanoVGGT.InferenceWindow.temporal_radius"),
        max_edges=_positive_int(window_raw.get("max_edges", 24), name="PanoVGGT.InferenceWindow.max_edges"),
    )
    if inference_window.overlap < 0:
        raise ValueError("PanoVGGT.InferenceWindow.overlap must be non-negative.")
    joint_inference = JointInferenceConfig(
        enabled=bool(joint_raw.get("enabled", False)),
        history_policy=str(joint_raw.get("history_policy", "recent")),
        min_history_frames=_nonnegative_int(
            joint_raw.get("min_history_frames", 0),
            name="PanoVGGT.JointInference.min_history_frames",
        ),
        max_history_frames=_nonnegative_int(
            joint_raw.get("max_history_frames", 3),
            name="PanoVGGT.JointInference.max_history_frames",
        ),
    )
    alignment = AlignmentConfig(
        use_common_history=bool(alignment_raw.get("use_common_history", False)),
        history_point_budget_ratio=float(alignment_raw.get("history_point_budget_ratio", 0.5)),
        exclude_sky=bool(alignment_raw.get("exclude_sky", True)),
        sky_threshold=float(alignment_raw.get("sky_threshold", keyframe_anchor.sky_threshold)),
    )

    return M3SphereConfig(
        enabled=bool(m3_raw.get("enabled", False)),
        descriptor_dim=m3_descriptor_dim,
        matching_head=matching_head,
        dense_matching=dense_matching,
        dense_ba=dense_ba,
        keyframe_anchor=keyframe_anchor,
        keyframe_graph=keyframe_graph,
        inference_window=inference_window,
        joint_inference=joint_inference,
        alignment=alignment,
    )
