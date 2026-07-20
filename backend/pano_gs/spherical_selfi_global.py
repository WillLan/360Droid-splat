"""Coordinator for window Sim(3), panorama loops, and the single explicit map."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable

import torch

from frontend.pano_vggt.alignment import SubmapAligner
from frontend.spherical_selfi.panorama_loop import PanoramaLoopDetector
from frontend.spherical_selfi.window_packet import (
    BoundaryMatchBlock,
    ChunkStrideMatchBlock,
    LocalGaussianWindowPacket,
    chunk_stride_matches_from_cache,
)
from geometry.pose import invert_c2w
from geometry.panorama_loop_contracts import (
    DenseSphericalLoopMeasurement,
    LoopPoseMeasurement,
    PanoramaLoopVerification,
)
from geometry.spherical_pseudo_correspondence import sample_joint_valid_fibonacci_uv
from geometry.spherical_erp import sample_erp_with_wrap
from geometry.sim3 import (
    apply_sim3,
    apply_sim3_to_c2w,
    canonicalize_c2w,
    canonicalize_sim3,
    rebase_c2w_to_sim3_anchor,
    sim3_components,
    sim3_from_components,
    sim3_identity,
    sim3_inverse,
    sim3_log,
    weighted_umeyama,
)
from models.spherical_selfi_stage3_ba import build_stage3_match_cache
from models.spherical_voxel_anchor_refiner import voxelize_per_pixel_gaussians

from .adapter import PanoRenderCamera, PFGS360Renderer
from .mapper import PanoGaussianMap, PanoGaussianMapper
from .sim3_graph import (
    CoincidentPanoramaFactor,
    DenseSphericalFactorBlock,
    GlobalSim3FactorGraph,
    Sim3GraphEdge,
    Sim3GraphOptimizeResult,
    s2_log_tangent_coordinates,
)
from .stage2_global_fusion import Stage2GlobalMapFusion


@dataclass
class GlobalWindowBackendResult:
    window_id: int
    aligned: bool
    loop_accepted: int
    graph: Sim3GraphOptimizeResult | None
    fusion: dict[str, int | float]
    correction: dict[str, int]
    map_optimization: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameGeometryUpdate:
    """Private global-geometry update for one frontend frame."""

    frame_id: int
    pose_c2w: torch.Tensor
    depth_scale: float
    owner_window_id: int
    depth_owner_window_id: int
    pose_owner_node_id: int | None = None
    depth_scales_by_window: dict[int, float] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class FrameGeometryUpdateBatch:
    """Atomic private snapshot of every admitted frame geometry."""

    revision: int
    complete_snapshot: bool
    updates: dict[int, FrameGeometryUpdate]
    affected_node_ids: tuple[int, ...] = ()
    reason: str = "geometry_refresh"


@dataclass(frozen=True)
class PoseStateCandidate:
    """One fully materialized canonical pose state awaiting transaction commit."""

    revision: int
    reason: str
    affected_node_ids: tuple[int, ...]
    affected_frame_ids: tuple[int, ...]
    affected_window_ids: tuple[int, ...]
    affected_submap_ids: tuple[int, ...]
    frame_global_poses: dict[int, torch.Tensor]
    window_transforms: dict[int, torch.Tensor]
    packet_variant_count: int


@dataclass(frozen=True)
class PoseStateConsistencyReport:
    """Hard consistency check across canonical state and all derived caches."""

    candidate_revision: int
    committed_revision: int
    accepted: bool
    frame_count: int
    packet_variant_count: int
    submap_transform_count: int
    lazy_owner_count: int
    mapper_pose_count: int
    max_matrix_error: float
    max_rotation_error_deg: float
    max_center_error: float
    max_submap_matrix_error: float
    max_lazy_owner_matrix_error: float
    max_mapper_matrix_error: float
    non_finite_count: int
    revision_mismatch_count: int
    reason: str

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "canonical_pose_state_consistency",
            "accepted": bool(self.accepted),
            "candidate_revision": int(self.candidate_revision),
            "committed_revision": int(self.committed_revision),
            "frame_count": int(self.frame_count),
            "packet_variant_count": int(self.packet_variant_count),
            "submap_transform_count": int(self.submap_transform_count),
            "lazy_owner_count": int(self.lazy_owner_count),
            "mapper_pose_count": int(self.mapper_pose_count),
            "max_matrix_error": float(self.max_matrix_error),
            "max_rotation_error_deg": float(self.max_rotation_error_deg),
            "max_center_error": float(self.max_center_error),
            "max_submap_matrix_error": float(self.max_submap_matrix_error),
            "max_lazy_owner_matrix_error": float(
                self.max_lazy_owner_matrix_error
            ),
            "max_mapper_matrix_error": float(
                self.max_mapper_matrix_error
            ),
            "non_finite_count": int(self.non_finite_count),
            "revision_mismatch_count": int(
                self.revision_mismatch_count
            ),
            "reason": str(self.reason),
        }


@dataclass
class _PreparedPacketCandidate:
    """Backend-owned packet state prepared before RGB registration."""

    packet: LocalGaussianWindowPacket
    start_transform: torch.Tensor | None = None
    alignment_diagnostics: dict[str, Any] = field(default_factory=dict)
    aligned: bool = False
    refined_packet: bool = False
    refiner_pending: bool = False
    canonicalized: bool = False


@dataclass(frozen=True)
class ChunkStrideHoldout:
    source: int
    target: int
    edge_type: str
    source_bearing: torch.Tensor
    target_bearing: torch.Tensor
    source_depth: torch.Tensor
    target_depth: torch.Tensor
    initial_angular_median_deg: float
    initial_relative_depth_median: float


@dataclass
class HierarchicalSubmapRecord:
    """One bounded local graph represented by a global Sim(3) node."""

    submap_id: int
    anchor_node_id: int
    window_ids: list[int] = field(default_factory=list)
    boundary_node_ids: list[int] = field(default_factory=list)
    local_window_transforms: dict[int, torch.Tensor] = field(default_factory=dict)
    local_boundary_transforms: dict[int, torch.Tensor] = field(default_factory=dict)
    frozen: bool = False
    compressed_dense_factors: int = 0


@dataclass(frozen=True)
class RenderedSharedFrame:
    depth: torch.Tensor
    alpha: torch.Tensor
    anchor_visibility: torch.Tensor
    render_seconds: float


@dataclass(frozen=True)
class OverlapFrameGeometry:
    """Same-frame geometry expressed in the previous/current chunk anchors."""

    frame_id: int
    previous_index: int
    current_index: int
    bearing: torch.Tensor
    uv: torch.Tensor
    previous_depth: torch.Tensor
    current_depth: torch.Tensor
    previous_points: torch.Tensor
    current_points: torch.Tensor
    previous_pose: torch.Tensor
    current_pose: torch.Tensor
    holdout_mask: torch.Tensor
    previous_render: RenderedSharedFrame | None = None
    current_render: RenderedSharedFrame | None = None
    previous_valid_image: torch.Tensor | None = None
    current_valid_image: torch.Tensor | None = None
    sky_union_image: torch.Tensor | None = None


@dataclass(frozen=True)
class KnownPoseBridgeFrame:
    """One overlap frame linking a local whole-chunk map to the global map."""

    frame_id: int
    previous_index: int
    current_index: int
    bearing: torch.Tensor
    uv: torch.Tensor
    global_depth: torch.Tensor
    current_depth: torch.Tensor
    source_depth_previous_owner: torch.Tensor
    previous_local_pose: torch.Tensor
    current_local_pose: torch.Tensor
    known_global_pose: torch.Tensor
    holdout_mask: torch.Tensor
    inlier_mask: torch.Tensor
    global_render: RenderedSharedFrame
    previous_render: RenderedSharedFrame
    current_render: RenderedSharedFrame
    global_valid_image: torch.Tensor
    current_valid_image: torch.Tensor
    global_previous_consistency_image: torch.Tensor
    sky_union_image: torch.Tensor
    global_previous_consistency_ratio: float


@dataclass(frozen=True)
class KnownPoseBridgeSolution:
    packet: LocalGaussianWindowPacket
    owner_transform: torch.Tensor
    relative_measurement: torch.Tensor
    diagnostics: dict[str, Any]


class SphericalSelfiGlobalBackend:
    _MAP_TRANSACTION_METADATA = (
        "_anchor_level",
        "_anchor_voxel_size",
        "_anchor_grid_coord",
        "_anchor_obs_count",
        "_anchor_conf_accum",
        "_anchor_birth_frame",
        "_anchor_last_seen_kf",
        "_anchor_last_update_kf_ord",
        "_anchor_source_window_id",
        "_anchor_source_frame_start",
        "_anchor_source_frame_end",
        "_anchor_inlier_obs",
        "_anchor_outlier_obs",
        "_anchor_owner_window_id",
        "_anchor_quality",
        "_anchor_visibility_count",
        "_anchor_render_error_ema",
        "_anchor_depth_selected_levels",
    )

    def __init__(
        self,
        gaussian_map: PanoGaussianMap,
        *,
        mapper: PanoGaussianMapper | None = None,
        renderer: PFGS360Renderer | None = None,
        pose_canonicalized_packet_refiner: (
            Callable[[LocalGaussianWindowPacket], LocalGaussianWindowPacket]
            | None
        ) = None,
        packet_refiner_release: Callable[[int], None] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.map = gaussian_map
        self.mapper = mapper
        self.renderer = renderer if renderer is not None else getattr(mapper, "renderer", None)
        self.pose_canonicalized_packet_refiner = (
            pose_canonicalized_packet_refiner
        )
        self.packet_refiner_release = packet_refiner_release
        self.config = dict(config or {})
        graph_cfg = dict(self.config.get("global_graph", {}) or {})
        self.global_graph_optimization_enabled = bool(
            graph_cfg.get("optimization_enabled", True)
        )
        loop_cfg = dict(self.config.get("loop_closure", {}) or {})
        descriptor_cfg = dict(loop_cfg.get("descriptor", {}) or {})
        retrieval_cfg = dict(loop_cfg.get("retrieval", {}) or {})
        verification_cfg = dict(loop_cfg.get("verification", {}) or {})
        robust_loop_cfg = dict(self.config.get("robust_loop", {}) or {})
        hierarchical_cfg = dict(self.config.get("hierarchical_submaps", {}) or {})
        fusion_cfg = dict(self.config.get("voxel_fusion", {}) or {})
        optimize_cfg = dict(self.config.get("map_optimization", {}) or {})
        two_stage_cfg = dict(optimize_cfg.get("two_stage", {}) or {})
        pose_tracking_cfg = dict(
            two_stage_cfg.get("prefusion_pose_tracking", {}) or {}
        )
        gaussian_mapping_cfg = dict(
            two_stage_cfg.get("postfusion_gaussian_mapping", {}) or {}
        )
        lazy_map_cfg = dict(optimize_cfg.get("lazy_submap_transforms", {}) or {})
        validation_cfg = dict(self.config.get("geometry_validation", {}) or {})
        seam_check_cfg = dict(
            self.config.get("post_optimization_seam_check", {}) or {}
        )
        rendered_alignment_cfg = dict(
            self.config.get("rendered_overlap_alignment", {}) or {}
        )
        insertion_dedup_cfg = dict(self.config.get("insertion_dedup", {}) or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.voxel_anchor_refiner_enabled = bool(
            self.config.get("_voxel_anchor_refiner_enabled", False)
        )
        self.rendered_overlap_alignment_enabled = bool(
            rendered_alignment_cfg.get("enabled", False)
        )
        self.rendered_overlap_alignment_mode = str(
            rendered_alignment_cfg.get("mode", "shared_frame_scale_only")
        ).strip().lower()
        if (
            self.rendered_overlap_alignment_enabled
            and self.rendered_overlap_alignment_mode
            not in {
                "shared_frame_scale_only",
                "two_frame_scale_pose",
                "two_frame_full_sim3",
                "two_frame_bridge_depth_scale",
                "two_frame_bridge_pose_scale",
            }
        ):
            raise ValueError(
                "rendered_overlap_alignment.mode must be "
                "'shared_frame_scale_only', 'two_frame_scale_pose', "
                "'two_frame_full_sim3', 'two_frame_bridge_depth_scale', "
                "or 'two_frame_bridge_pose_scale'"
            )
        self.two_frame_known_pose_bridge_enabled = (
            self.rendered_overlap_alignment_enabled
            and self.rendered_overlap_alignment_mode
            in {
                "two_frame_bridge_depth_scale",
                "two_frame_bridge_pose_scale",
            }
        )
        self.two_frame_bridge_depth_scale_enabled = (
            self.rendered_overlap_alignment_mode
            == "two_frame_bridge_depth_scale"
        )
        self.two_frame_bridge_pose_scale_enabled = (
            self.rendered_overlap_alignment_mode
            == "two_frame_bridge_pose_scale"
        )
        self.two_frame_overlap_enabled = (
            self.rendered_overlap_alignment_enabled
            and self.rendered_overlap_alignment_mode
            in {
                "two_frame_scale_pose",
                "two_frame_full_sim3",
                "two_frame_bridge_depth_scale",
                "two_frame_bridge_pose_scale",
            }
        )
        self.two_frame_full_sim3_enabled = (
            self.two_frame_overlap_enabled
            and self.rendered_overlap_alignment_mode == "two_frame_full_sim3"
        )
        self.two_frame_scale_pose_enabled = (
            self.two_frame_overlap_enabled
            and self.rendered_overlap_alignment_mode == "two_frame_scale_pose"
        )
        self.rendered_alignment_min_points = max(
            3, int(rendered_alignment_cfg.get("min_points", 256))
        )
        self.rendered_alignment_max_points = max(
            self.rendered_alignment_min_points,
            int(rendered_alignment_cfg.get("max_points", 4096)),
        )
        self.rendered_alignment_min_points_per_frame = max(
            3,
            int(
                rendered_alignment_cfg.get(
                    "min_points_per_frame",
                    self.rendered_alignment_min_points,
                )
            ),
        )
        self.rendered_alignment_max_points_per_frame = max(
            self.rendered_alignment_min_points_per_frame,
            int(
                rendered_alignment_cfg.get(
                    "max_points_per_frame",
                    max(1, self.rendered_alignment_max_points // 2),
                )
            ),
        )
        self.rendered_alignment_alpha_threshold = float(
            rendered_alignment_cfg.get("alpha_threshold", 0.05)
        )
        self.rendered_alignment_min_confidence = float(
            rendered_alignment_cfg.get("min_confidence", 0.05)
        )
        self.rendered_alignment_min_inlier_ratio = float(
            rendered_alignment_cfg.get("min_inlier_ratio", 0.35)
        )
        self.rendered_alignment_max_median_relative_error = float(
            rendered_alignment_cfg.get("max_median_relative_error", 0.10)
        )
        self.rendered_alignment_max_scale_change = float(
            rendered_alignment_cfg.get("max_scale_change", 2.5)
        )
        self.rendered_alignment_irls_iterations = max(
            1, int(rendered_alignment_cfg.get("irls_iterations", 5))
        )
        self.rendered_alignment_holdout_stride = max(
            2, int(rendered_alignment_cfg.get("holdout_stride", 5))
        )
        self.rendered_alignment_covariance_min_ratio = float(
            rendered_alignment_cfg.get("covariance_min_ratio", 1.0e-4)
        )
        self.rendered_alignment_max_rotation_correction_deg = float(
            rendered_alignment_cfg.get("max_rotation_correction_deg", 10.0)
        )
        self.rendered_alignment_max_translation_correction = float(
            rendered_alignment_cfg.get("max_translation_correction", 1.0)
        )
        self.rendered_alignment_max_shared_rotation_error_deg = float(
            rendered_alignment_cfg.get("max_shared_rotation_error_deg", 2.0)
        )
        self.rendered_alignment_max_shared_center_error = float(
            rendered_alignment_cfg.get("max_shared_center_error", 0.15)
        )
        self.rendered_alignment_global_map_consistency_error = float(
            rendered_alignment_cfg.get(
                "global_map_consistency_max_relative_error", 0.15
            )
        )
        self.rendered_alignment_global_map_min_consistency_ratio = float(
            rendered_alignment_cfg.get("global_map_min_consistency_ratio", 0.35)
        )
        self.rendered_alignment_pose_baseline_min = float(
            rendered_alignment_cfg.get("pose_baseline_min", 1.0e-3)
        )
        self.post_refiner_scale_recheck_enabled = bool(
            rendered_alignment_cfg.get("post_refiner_scale_recheck", True)
        )
        self.post_refiner_scale_rerun_threshold = float(
            rendered_alignment_cfg.get("post_refiner_scale_rerun_threshold", 0.02)
        )
        self.post_refiner_scale_max_relative_change = float(
            rendered_alignment_cfg.get("post_refiner_scale_max_relative_change", 0.10)
        )
        if not 0.0 <= self.rendered_alignment_alpha_threshold <= 1.0:
            raise ValueError(
                "rendered_overlap_alignment.alpha_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.rendered_alignment_min_confidence <= 1.0:
            raise ValueError(
                "rendered_overlap_alignment.min_confidence must be in [0, 1]"
            )
        if not 0.0 <= self.rendered_alignment_min_inlier_ratio <= 1.0:
            raise ValueError(
                "rendered_overlap_alignment.min_inlier_ratio must be in [0, 1]"
            )
        if (
            not math.isfinite(
                self.rendered_alignment_max_median_relative_error
            )
            or self.rendered_alignment_max_median_relative_error < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_median_relative_error must be non-negative"
            )
        if (
            not math.isfinite(self.rendered_alignment_max_scale_change)
            or self.rendered_alignment_max_scale_change < 1.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_scale_change must be at least 1"
            )
        if (
            not math.isfinite(self.rendered_alignment_covariance_min_ratio)
            or self.rendered_alignment_covariance_min_ratio < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.covariance_min_ratio must be non-negative"
            )
        if (
            not math.isfinite(
                self.rendered_alignment_max_rotation_correction_deg
            )
            or self.rendered_alignment_max_rotation_correction_deg < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_rotation_correction_deg "
                "must be non-negative"
            )
        if (
            not math.isfinite(
                self.rendered_alignment_max_translation_correction
            )
            or self.rendered_alignment_max_translation_correction < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_translation_correction "
                "must be non-negative"
            )
        if (
            not math.isfinite(
                self.rendered_alignment_max_shared_rotation_error_deg
            )
            or self.rendered_alignment_max_shared_rotation_error_deg < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_shared_rotation_error_deg "
                "must be non-negative"
            )
        if (
            not math.isfinite(self.rendered_alignment_max_shared_center_error)
            or self.rendered_alignment_max_shared_center_error < 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.max_shared_center_error "
                "must be non-negative"
            )
        if not (
            math.isfinite(self.rendered_alignment_global_map_consistency_error)
            and self.rendered_alignment_global_map_consistency_error >= 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment."
                "global_map_consistency_max_relative_error must be non-negative"
            )
        if not 0.0 <= self.rendered_alignment_global_map_min_consistency_ratio <= 1.0:
            raise ValueError(
                "rendered_overlap_alignment.global_map_min_consistency_ratio "
                "must be in [0, 1]"
            )
        if not (
            math.isfinite(self.rendered_alignment_pose_baseline_min)
            and self.rendered_alignment_pose_baseline_min > 0.0
        ):
            raise ValueError(
                "rendered_overlap_alignment.pose_baseline_min must be positive"
            )
        if not (
            0.0 <= self.post_refiner_scale_rerun_threshold
            <= self.post_refiner_scale_max_relative_change
        ):
            raise ValueError(
                "post-refiner scale thresholds must satisfy "
                "0 <= rerun_threshold <= max_relative_change"
            )
        self.rendered_alignment_failure_policy = str(
            rendered_alignment_cfg.get("failure_policy", "error")
        ).strip().lower()
        allowed_failure_policies = (
            {"error", "scale_pose_then_error"}
            if self.two_frame_overlap_enabled
            else {"error"}
        )
        if self.rendered_overlap_alignment_enabled and (
            self.rendered_alignment_failure_policy
            not in allowed_failure_policies
        ):
            raise ValueError(
                "Unsupported rendered_overlap_alignment.failure_policy for "
                f"mode {self.rendered_overlap_alignment_mode!r}"
            )
        self.insertion_dedup_enabled = bool(
            insertion_dedup_cfg.get("enabled", False)
        )
        self.insertion_dedup_visible_only = bool(
            insertion_dedup_cfg.get("visible_only", True)
        )
        self.insertion_dedup_same_level_only = bool(
            insertion_dedup_cfg.get("same_level_only", True)
        )
        self.insertion_dedup_radius_voxels = float(
            insertion_dedup_cfg.get("radius_voxels", 1.0)
        )
        if self.insertion_dedup_enabled and (
            not math.isfinite(self.insertion_dedup_radius_voxels)
            or self.insertion_dedup_radius_voxels <= 0.0
        ):
            raise ValueError("insertion_dedup.radius_voxels must be positive")
        self.insertion_dedup_compare_existing_only = bool(
            insertion_dedup_cfg.get("compare_existing_only", True)
        )
        self.insertion_dedup_permanent_drop = bool(
            insertion_dedup_cfg.get("permanent_drop", True)
        )
        self.insertion_dedup_update_existing_statistics = bool(
            insertion_dedup_cfg.get("update_existing_statistics", True)
        )
        self.insertion_dedup_require_new_frame_support = bool(
            insertion_dedup_cfg.get("require_new_frame_support", False)
        )
        self.insertion_dedup_max_new_gaussians_per_chunk = max(
            0,
            int(insertion_dedup_cfg.get("max_new_gaussians_per_chunk", 0)),
        )
        self.insertion_dedup_coverage_coarse_cell_size = float(
            insertion_dedup_cfg.get("coverage_coarse_cell_size", 0.64)
        )
        self.insertion_dedup_log_posthash_coverage = bool(
            insertion_dedup_cfg.get("log_posthash_coverage", False)
        )
        if self.insertion_dedup_max_new_gaussians_per_chunk > 0 and (
            not math.isfinite(self.insertion_dedup_coverage_coarse_cell_size)
            or self.insertion_dedup_coverage_coarse_cell_size <= 0.0
        ):
            raise ValueError(
                "insertion_dedup.coverage_coarse_cell_size must be positive"
            )
        if self.insertion_dedup_enabled and not all(
            (
                self.insertion_dedup_visible_only,
                self.insertion_dedup_same_level_only,
                self.insertion_dedup_compare_existing_only,
                self.insertion_dedup_permanent_drop,
            )
        ):
            raise ValueError(
                "The supported insertion_dedup mode requires visible_only, "
                "same_level_only, compare_existing_only, and permanent_drop"
            )
        self.node_mode = str(graph_cfg.get("node_mode", "window_anchor")).lower()
        if self.node_mode not in {
            "window_anchor",
            "boundary_frame",
            "chunk_first_stride",
        }:
            raise ValueError(
                "global_graph.node_mode must be 'window_anchor', "
                "'boundary_frame', or 'chunk_first_stride'"
            )
        self.chunk_first_stride_graph = self.node_mode == "chunk_first_stride"
        if (
            self.chunk_first_stride_graph
            and self.rendered_overlap_alignment_enabled
            and self.rendered_overlap_alignment_mode
            != "two_frame_bridge_depth_scale"
        ):
            raise ValueError(
                "chunk_first_stride requires "
                "rendered_overlap_alignment.mode=two_frame_bridge_depth_scale; "
                "pose-baseline bridge ablations must use node_mode=boundary_frame"
            )
        self.boundary_frame_graph = self.node_mode in {
            "boundary_frame",
            "chunk_first_stride",
        }
        self.hierarchical_submaps_enabled = bool(hierarchical_cfg.get("enabled", False))
        if self.hierarchical_submaps_enabled and not self.boundary_frame_graph:
            raise ValueError("hierarchical_submaps currently requires global_graph.node_mode=boundary_frame")
        self.windows_per_submap = max(1, int(hierarchical_cfg.get("windows_per_submap", 5)))
        self.shared_submap_boundary_nodes = max(
            0, int(hierarchical_cfg.get("shared_boundary_nodes", 1))
        )
        self.compress_frozen_dense_factors = bool(
            hierarchical_cfg.get("compress_frozen_dense_factors", True)
        )
        self.local_camera_model = str(
            hierarchical_cfg.get(
                "local_camera_model",
                "se3_shared_scale" if self.hierarchical_submaps_enabled else "sim3_per_node",
            )
        ).lower()
        if self.local_camera_model not in {"se3_shared_scale", "sim3_per_node"}:
            raise ValueError(
                "hierarchical_submaps.local_camera_model must be "
                "'se3_shared_scale' or 'sim3_per_node'"
            )
        self.allow_unaligned_fallback = bool(graph_cfg.get("allow_unaligned_fallback", False))
        self.allow_boundary_matching_fallback = bool(
            graph_cfg.get("allow_boundary_matching_fallback", False)
        )
        self.expected_overlap_frames = int(graph_cfg.get("expected_overlap_frames", 1))
        self.enforce_exact_overlap = bool(graph_cfg.get("enforce_exact_overlap", True))
        stride_cfg = dict(graph_cfg.get("chunk_stride", {}) or {})
        self.chunk_stride_target_index = max(
            1, int(stride_cfg.get("target_index", 2))
        )
        self.chunk_stride_holdout_stride = max(
            2, int(stride_cfg.get("holdout_stride", 5))
        )
        self.chunk_stride_irls_iterations = max(
            1, int(stride_cfg.get("irls_iterations", 5))
        )
        skip_cfg = dict(graph_cfg.get("skip_edge", {}) or {})
        self.chunk_skip_enabled = bool(skip_cfg.get("enabled", False))
        self.chunk_skip_num_queries = max(
            3,
            int(skip_cfg.get("num_queries", 2048)),
        )
        self.chunk_skip_temperature = float(skip_cfg.get("temperature", 0.07))
        self.chunk_skip_query_chunk_size = max(
            1, int(skip_cfg.get("query_chunk_size", 32))
        )
        self.chunk_skip_oversample_factor = max(
            1, int(skip_cfg.get("fibonacci_oversample_factor", 8))
        )
        self.chunk_skip_area_correction = bool(
            skip_cfg.get("use_spherical_area_correction", True)
        )
        self.chunk_skip_forward_backward = bool(
            skip_cfg.get("forward_backward", True)
        )
        self.chunk_skip_fb_tolerance_deg = float(
            skip_cfg.get("fb_tolerance_deg", 1.0)
        )
        self.chunk_skip_min_factor_weight = float(
            skip_cfg.get("min_factor_weight", 0.01)
        )
        self.chunk_skip_subpixel_refine_radius = max(
            0, min(4, int(skip_cfg.get("subpixel_refine_radius", 1)))
        )
        self.graph_optimization_trigger = str(
            graph_cfg.get("optimization_trigger", "periodic_and_loop")
        ).strip().lower()
        if self.graph_optimization_trigger not in {
            "periodic_and_loop",
            "loop_only",
        }:
            raise ValueError(
                "global_graph.optimization_trigger must be "
                "'periodic_and_loop' or 'loop_only'"
            )
        if self.chunk_first_stride_graph:
            if self.expected_overlap_frames != 2:
                raise ValueError(
                    "chunk_first_stride requires expected_overlap_frames=2"
                )
            if self.chunk_stride_target_index != 2:
                raise ValueError(
                    "The supported size=4/stride=2 graph requires "
                    "global_graph.chunk_stride.target_index=2"
                )
        if self.two_frame_overlap_enabled:
            if not self.boundary_frame_graph:
                raise ValueError(
                    "Two-frame rendered alignment requires "
                    "global_graph.node_mode=boundary_frame"
                )
            if self.expected_overlap_frames != 2:
                raise ValueError(
                    "Two-frame rendered alignment requires "
                    "expected_overlap_frames=2"
                )
        self.fibonacci_seed = int(graph_cfg.get("fibonacci_seed", 123))
        self.fibonacci_oversample_factor = max(1, int(graph_cfg.get("fibonacci_oversample_factor", 8)))
        self.fibonacci_min_depth = float(graph_cfg.get("min_depth", 0.05))
        self.fibonacci_max_depth = float(graph_cfg.get("max_depth", 20.0))
        self.sky_threshold = float(graph_cfg.get("sky_threshold", 0.5))
        self.depth_factor_weight = float(graph_cfg.get("depth_factor_weight", 0.1))
        self.s2_huber_delta_deg = float(graph_cfg.get("s2_huber_delta_deg", 1.0))
        self.min_match_cosine = float(graph_cfg.get("min_match_cosine", 0.45))
        self.max_match_entropy = float(graph_cfg.get("max_match_entropy", 0.95))
        self.min_dense_factors = max(3, int(graph_cfg.get("min_dense_factors", 32)))
        self.normalize_dense_information_by_count = bool(
            graph_cfg.get("normalize_dense_information_by_count", False)
        )
        self.dense_information_reference_count = max(
            1.0, float(graph_cfg.get("dense_information_reference_count", 64.0))
        )
        self.recent_optimization_windows = max(2, int(graph_cfg.get("recent_windows", 32)))
        self.global_ba_start_nodes = max(2, int(graph_cfg.get("optimization_start_nodes", 6)))
        self.global_ba_interval_edges = max(1, int(graph_cfg.get("optimization_interval_edges", 3)))
        self.global_ba_active_nodes = max(2, int(graph_cfg.get("active_nodes", 6)))
        self.max_overlap_residual = float(
            graph_cfg.get("max_overlap_residual", 0.35)
        )
        self.min_overlap_inlier_ratio = float(
            graph_cfg.get("min_overlap_inlier_ratio", 0.35)
        )
        self.max_overlap_scale_change = float(
            graph_cfg.get("max_scale_change", 2.5)
        )
        self.map_steps_per_window = max(0, int(optimize_cfg.get("steps_per_window", 0)))
        self.two_stage_map_optimization_enabled = bool(
            two_stage_cfg.get("enabled", False)
        )
        self.prefusion_pose_tracking_enabled = bool(
            pose_tracking_cfg.get("enabled", True)
        )
        self.prefusion_pose_tracking_config = pose_tracking_cfg
        self.postfusion_gaussian_mapping_config = gaussian_mapping_cfg
        if self.two_stage_map_optimization_enabled:
            if not self.chunk_first_stride_graph:
                raise ValueError(
                    "map_optimization.two_stage requires chunk_first_stride graph mode"
                )
            if self.mapper is None:
                raise ValueError(
                    "map_optimization.two_stage requires a backend Mapper"
                )
        self.map_optimize_recent_windows = max(
            1,
            int(optimize_cfg.get("recent_window_count", 1)),
        )
        self.map_optimize_photometric_only = bool(
            optimize_cfg.get("photometric_only", False)
        )
        self.map_optimize_skybox = bool(
            optimize_cfg.get("optimize_skybox", True)
        )
        self.map_steps_on_loop = max(
            0,
            int(optimize_cfg.get("extra_steps_on_loop", optimize_cfg.get("steps_on_loop", 0))),
        )
        self.loop_neighborhood_refinement_enabled = bool(
            optimize_cfg.get("loop_neighborhood_refinement", False)
        )
        self.loop_neighborhood_submap_radius = max(
            0, int(optimize_cfg.get("loop_neighborhood_submap_radius", 1))
        )
        self.loop_seam_dedup_enabled = bool(
            optimize_cfg.get("loop_seam_deduplication", False)
        )
        self.final_map_steps = max(0, int(optimize_cfg.get("final_steps", 0)))
        self.map_optimize_config = optimize_cfg
        self.lazy_submap_transforms_enabled = bool(lazy_map_cfg.get("enabled", False))
        if self.lazy_submap_transforms_enabled and not self.hierarchical_submaps_enabled:
            raise ValueError(
                "map_optimization.lazy_submap_transforms requires hierarchical_submaps.enabled"
            )
        if self.voxel_anchor_refiner_enabled:
            if not self.boundary_frame_graph:
                raise ValueError(
                    "VoxelAnchorRefiner requires global_graph.node_mode=boundary_frame"
                )
            if not self.rendered_overlap_alignment_enabled:
                raise ValueError(
                    "VoxelAnchorRefiner requires rendered_overlap_alignment.enabled=true"
                )
            if not self.insertion_dedup_enabled:
                raise ValueError(
                    "VoxelAnchorRefiner requires insertion_dedup.enabled=true"
                )
            if self.renderer is None:
                raise ValueError(
                    "VoxelAnchorRefiner rendered alignment requires a PFGS360 renderer"
                )
            if not self.lazy_submap_transforms_enabled:
                raise ValueError(
                    "VoxelAnchorRefiner requires "
                    "map_optimization.lazy_submap_transforms.enabled=true"
                )
        self.geometry_validation_enabled = bool(validation_cfg.get("enabled", True))
        self.geometry_tolerance = float(validation_cfg.get("tolerance", 1.0e-5))
        self.geometry_rollback_on_failure = bool(validation_cfg.get("rollback_on_failure", True))
        self.post_optimization_seam_check_enabled = bool(
            seam_check_cfg.get("enabled", False)
        )
        self.robust_loop_mode = str(robust_loop_cfg.get("mode", "off")).lower()
        if self.robust_loop_mode not in {"off", "dcs"}:
            raise ValueError("robust_loop.mode must be 'off' or 'dcs'")
        self.loop_transaction_enabled = bool(
            robust_loop_cfg.get("transactional", self.robust_loop_mode != "off")
        )
        self.loop_dcs_phi = (
            float(robust_loop_cfg.get("dcs_phi", 25.0))
            if self.robust_loop_mode == "dcs"
            else None
        )
        self.loop_min_dcs_scale = float(robust_loop_cfg.get("min_commit_dcs_scale", 0.05))
        self.loop_max_nonloop_objective_ratio = max(
            1.0, float(robust_loop_cfg.get("max_nonloop_objective_ratio", 1.05))
        )
        self.loop_nonloop_objective_tolerance = max(
            0.0, float(robust_loop_cfg.get("nonloop_objective_tolerance", 1.0e-6))
        )
        self.loop_path_max_rotation = math.radians(
            float(robust_loop_cfg.get("path_max_rotation_deg", 45.0))
        )
        self.loop_path_max_translation = float(
            robust_loop_cfg.get("path_max_translation", 5.0)
        )
        self.loop_path_max_log_scale = float(
            robust_loop_cfg.get("path_max_log_scale", math.log(3.0))
        )
        self.insert_loop_pose_factor = bool(
            loop_cfg.get("insert_pose_factor", False)
        )
        self.lifecycle_prune_interval = max(
            0, int(fusion_cfg.get("lifecycle_prune_interval_windows", 0))
        )
        self.lifecycle_max_stale_frames = max(0, int(fusion_cfg.get("max_stale_frames", 0)))
        max_render_error = fusion_cfg.get("max_render_error")
        self.lifecycle_max_render_error = (
            float("inf") if max_render_error is None else float(max_render_error)
        )
        self.graph = GlobalSim3FactorGraph(
            damping=float(graph_cfg.get("damping", 1.0e-4)),
            max_iterations=int(graph_cfg.get("iterations", 8)),
            pcg_iterations=int(graph_cfg.get("pcg_iterations", 64)),
            pcg_tolerance=float(graph_cfg.get("pcg_tolerance", 1.0e-6)),
            lm_max_trials=int(graph_cfg.get("lm_max_trials", 6)),
            lm_acceptance_eta=float(graph_cfg.get("lm_acceptance_eta", 1.0e-4)),
            lm_damping_min=float(graph_cfg.get("lm_damping_min", 1.0e-8)),
            lm_damping_max=float(graph_cfg.get("lm_damping_max", 1.0e8)),
            lm_diagonal_floor=float(graph_cfg.get("lm_diagonal_floor", 1.0e-6)),
            dense_linearization_chunk_size=int(
                graph_cfg.get("dense_linearization_chunk_size", 512)
            ),
            lock_scale_updates=(
                self.hierarchical_submaps_enabled
                and self.local_camera_model == "se3_shared_scale"
            ),
            analytic_dense_linearization=bool(
                graph_cfg.get("analytic_dense_linearization", False)
            ),
            restrict_objective_to_active_factors=bool(
                graph_cfg.get("restrict_objective_to_active_factors", False)
            ),
            optimization_enabled=self.global_graph_optimization_enabled,
        )
        self.submap_graph = (
            GlobalSim3FactorGraph(
                damping=float(hierarchical_cfg.get("damping", graph_cfg.get("damping", 1.0e-4))),
                max_iterations=int(
                    hierarchical_cfg.get("iterations", graph_cfg.get("iterations", 8))
                ),
                pcg_iterations=int(
                    hierarchical_cfg.get("pcg_iterations", graph_cfg.get("pcg_iterations", 64))
                ),
                pcg_tolerance=float(
                    hierarchical_cfg.get("pcg_tolerance", graph_cfg.get("pcg_tolerance", 1.0e-6))
                ),
                lm_max_trials=int(graph_cfg.get("lm_max_trials", 6)),
                lm_acceptance_eta=float(graph_cfg.get("lm_acceptance_eta", 1.0e-4)),
                lm_damping_min=float(graph_cfg.get("lm_damping_min", 1.0e-8)),
                lm_damping_max=float(graph_cfg.get("lm_damping_max", 1.0e8)),
                lm_diagonal_floor=float(graph_cfg.get("lm_diagonal_floor", 1.0e-6)),
                dense_linearization_chunk_size=int(
                    hierarchical_cfg.get(
                        "dense_linearization_chunk_size",
                        graph_cfg.get("dense_linearization_chunk_size", 512),
                    )
                ),
                analytic_dense_linearization=bool(
                    hierarchical_cfg.get(
                        "analytic_dense_linearization",
                        graph_cfg.get("analytic_dense_linearization", False),
                    )
                ),
                restrict_objective_to_active_factors=bool(
                    hierarchical_cfg.get(
                        "restrict_objective_to_active_factors",
                        graph_cfg.get("restrict_objective_to_active_factors", False),
                    )
                ),
                optimization_enabled=self.global_graph_optimization_enabled,
            )
            if self.hierarchical_submaps_enabled
            else None
        )
        self.overlap_aligner = SubmapAligner(
            align_mode="sim3",
            max_residual=self.max_overlap_residual,
            min_inlier_ratio=self.min_overlap_inlier_ratio,
            max_scale_change=self.max_overlap_scale_change,
            min_points=int(graph_cfg.get("min_overlap_points", 32)),
            return_rejected_transform=True,
            irls_iterations=int(graph_cfg.get("umeyama_irls_iterations", 3)),
            huber_delta=graph_cfg.get("umeyama_huber_delta"),
        )
        self.max_overlap_points = max(
            32,
            int(graph_cfg.get("overlap_num_queries", graph_cfg.get("max_overlap_points", 4096))),
        )
        self.loop_detector = PanoramaLoopDetector(
            top_k=int(retrieval_cfg.get("top_k", loop_cfg.get("top_k", 5))),
            exclude_recent_windows=int(loop_cfg.get("exclude_recent_windows", 3)),
            min_retrieval_score=float(loop_cfg.get("min_retrieval_score", 0.35)),
            min_match_cosine=float(loop_cfg.get("min_match_cosine", 0.45)),
            min_matches=int(loop_cfg.get("min_matches", 32)),
            max_matches=int(loop_cfg.get("max_matches", 512)),
            min_inlier_ratio=float(loop_cfg.get("min_inlier_ratio", 0.30)),
            max_alignment_residual=float(loop_cfg.get("max_alignment_residual", 0.35)),
            max_scale_change=float(loop_cfg.get("max_scale_change", 2.5)),
            coincident_translation_threshold=float(loop_cfg.get("coincident_translation_threshold", 0.15)),
            coincident_rotation_residual_deg=float(loop_cfg.get("coincident_rotation_residual_deg", 2.0)),
            rotation_ransac_iterations=int(loop_cfg.get("rotation_ransac_iterations", 128)),
            factor_queries_per_direction=int(graph_cfg.get("factor_queries_per_direction", 2048)),
            fibonacci_oversample_factor=self.fibonacci_oversample_factor,
            fibonacci_seed=self.fibonacci_seed,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            sky_threshold=self.sky_threshold,
            max_match_entropy=float(graph_cfg.get("max_match_entropy", 0.95)),
            forward_backward=bool(graph_cfg.get("forward_backward", True)),
            fb_tolerance_deg=float(graph_cfg.get("fb_tolerance_deg", 1.0)),
            min_factor_weight=float(graph_cfg.get("min_factor_weight", 0.01)),
            target_area_correction=bool(graph_cfg.get("target_area_correction", True)),
            depth_factor_weight=self.depth_factor_weight,
            s2_huber_delta_deg=self.s2_huber_delta_deg,
            descriptor_mode=str(descriptor_cfg.get("mode", "latitude_bands")),
            candidate_nms_radius=int(retrieval_cfg.get("candidate_nms_radius", 2)),
            max_verified_candidates=int(retrieval_cfg.get("max_verified_candidates", 3)),
            max_accepted_loops=int(retrieval_cfg.get("max_accepted_loops", 1)),
            verification_mode=str(verification_cfg.get("mode", "legacy")),
            rotation_inlier_threshold_deg=float(
                verification_cfg.get("rotation_inlier_threshold_deg", 2.0)
            ),
            min_rotation_inlier_ratio=verification_cfg.get("min_rotation_inlier_ratio"),
            min_spherical_coverage_bins=int(
                verification_cfg.get("min_spherical_coverage_bins", 6)
            ),
            coverage_latitude_bins=int(verification_cfg.get("coverage_latitude_bins", 3)),
            coverage_longitude_bins=int(verification_cfg.get("coverage_longitude_bins", 4)),
            max_rotation_consistency_deg=float(
                verification_cfg.get("max_rotation_consistency_deg", 3.0)
            ),
            max_normalized_alignment_residual=float(
                verification_cfg.get("max_normalized_alignment_residual", 0.10)
            ),
            loop_dcs_phi=self.loop_dcs_phi,
        )
        self.fusion = Stage2GlobalMapFusion(
            gaussian_map,
            voxel_sizes=tuple(fusion_cfg.get("voxel_sizes", (0.04, 0.08, 0.16, 0.32))),
            min_confidence=float(fusion_cfg.get("min_confidence", 0.05)),
            min_opacity=float(fusion_cfg.get("min_opacity", 0.02)),
            max_total_gaussians=int(fusion_cfg.get("max_total_gaussians", 0)),
            coverage_aware_budget=bool(
                fusion_cfg.get("coverage_aware_budget", False)
            ),
            coverage_coarse_cell_size=float(
                fusion_cfg.get("coverage_coarse_cell_size", 0.64)
            ),
            lazy_owner_transforms=self.lazy_submap_transforms_enabled,
        )
        self.packets: dict[int, LocalGaussianWindowPacket] = {}
        self.accepted_loop_pairs: set[tuple[int, int]] = set()
        self.submaps: dict[int, HierarchicalSubmapRecord] = {}
        self.window_to_submap: dict[int, int] = {}
        self._active_submap_id: int | None = None
        self._last_full_packet: LocalGaussianWindowPacket | None = None
        self.window_order: list[int] = []
        self.frame_owner_window: dict[int, int] = {}
        self.frame_depth_owner_window: dict[int, int] = {}
        self.frame_windows: dict[int, set[int]] = {}
        self.frame_pose_owner_node: dict[int, int] = {}
        self.frame_local_pose_in_owner: dict[int, torch.Tensor] = {}
        self.window_anchor_nodes: dict[int, int] = {}
        self.window_end_nodes: dict[int, int] = {}
        self.boundary_node_order: list[int] = []
        self._chunk_stride_holdouts: dict[
            tuple[int, int, str], ChunkStrideHoldout
        ] = {}
        self._chunk_node_initial_scale: dict[int, float] = {}
        self._sequential_edges_since_optimization = 0
        self._has_run_global_ba = False
        self._geometry_updates: dict[int, FrameGeometryUpdate] = {}
        self._geometry_revision = 0
        self._pose_state_revision = 0
        self._window_pose_revision: dict[int, int] = {}
        self._mapper_pose_revision_by_frame: dict[int, int] = {}
        self._last_pose_state_diagnostic: dict[str, Any] | None = None
        self._last_mapper_committed_state_diagnostic: dict[str, Any] | None = None
        self._pending_geometry_batch: FrameGeometryUpdateBatch | None = None
        self._pending_map_optimization: list[tuple[int, tuple[int, ...], int]] = []
        self._optimization_packets: dict[int, LocalGaussianWindowPacket] = {}
        self._prepared_packet_candidates: dict[int, _PreparedPacketCandidate] = {}
        self._prepared_packet_candidate_order: list[int] = []
        self._pending_seam_owner_windows: set[int] = set()
        self._last_rendered_overlap_diagnostic: dict[str, torch.Tensor] | None = None
        self._last_overlap_alignment_failure: dict[str, Any] | None = None
        self.results: list[GlobalWindowBackendResult] = []

    @staticmethod
    def _snapshot_graph_state(graph: GlobalSim3FactorGraph | None) -> dict[str, Any] | None:
        if graph is None:
            return None
        return {
            "nodes": {
                int(node): transform.clone()
                for node, transform in graph.nodes.items()
            },
            "edges": list(graph.edges),
            "edge_metadata": [
                copy.deepcopy(getattr(edge, "metadata", None))
                for edge in graph.edges
            ],
            "fixed_node_id": graph.fixed_node_id,
            "damping": float(graph.damping),
            "last_pcg_iterations": int(graph._last_pcg_iterations),
            "last_pcg_relative_residual": float(
                graph._last_pcg_relative_residual
            ),
            "last_normal_condition_estimate": float(
                graph._last_normal_condition_estimate
            ),
        }

    @staticmethod
    def _restore_graph_state(
        graph: GlobalSim3FactorGraph | None,
        state: dict[str, Any] | None,
    ) -> None:
        if graph is None or state is None:
            return
        graph.nodes = state["nodes"]
        graph.edges = state["edges"]
        for edge, metadata in zip(graph.edges, state["edge_metadata"]):
            if metadata is not None and hasattr(edge, "metadata"):
                edge.metadata.clear()
                edge.metadata.update(metadata)
        graph.fixed_node_id = state["fixed_node_id"]
        graph.damping = state["damping"]
        graph._last_pcg_iterations = state["last_pcg_iterations"]
        graph._last_pcg_relative_residual = state[
            "last_pcg_relative_residual"
        ]
        graph._last_normal_condition_estimate = state[
            "last_normal_condition_estimate"
        ]

    def _snapshot_boundary_transaction(
        self,
        *,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
    ) -> dict[str, Any]:
        loop_database = getattr(self.loop_detector, "_descriptor_database", None)
        packet_variants: list[
            tuple[
                LocalGaussianWindowPacket,
                torch.Tensor,
                Any,
                Any,
                dict[str, Any],
            ]
        ] = []
        for candidate in (
            list(self.packets.values())
            + list(self._optimization_packets.values())
            + ([self._last_full_packet] if self._last_full_packet is not None else [])
            + list(extra_packets)
        ):
            if all(id(candidate) != id(item[0]) for item in packet_variants):
                packet_variants.append(
                    (
                        candidate,
                        candidate.local_poses_c2w.detach().clone(),
                        candidate.observation,
                        candidate.anchor_observation,
                        dict(candidate.metadata),
                    )
                )
        return {
            "graph": self._snapshot_graph_state(self.graph),
            "submap_graph": self._snapshot_graph_state(self.submap_graph),
            "packets": dict(self.packets),
            "accepted_loop_pairs": set(self.accepted_loop_pairs),
            "submaps": copy.deepcopy(self.submaps),
            "window_to_submap": dict(self.window_to_submap),
            "active_submap_id": self._active_submap_id,
            "last_full_packet": self._last_full_packet,
            "window_order": list(self.window_order),
            "frame_owner_window": dict(self.frame_owner_window),
            "frame_depth_owner_window": dict(self.frame_depth_owner_window),
            "frame_windows": {
                int(frame): set(windows)
                for frame, windows in self.frame_windows.items()
            },
            "frame_pose_owner_node": dict(self.frame_pose_owner_node),
            "frame_local_pose_in_owner": {
                int(frame): pose.clone()
                for frame, pose in self.frame_local_pose_in_owner.items()
            },
            "window_anchor_nodes": dict(self.window_anchor_nodes),
            "window_end_nodes": dict(self.window_end_nodes),
            "boundary_node_order": list(self.boundary_node_order),
            "chunk_stride_holdouts": dict(self._chunk_stride_holdouts),
            "chunk_node_initial_scale": dict(self._chunk_node_initial_scale),
            "sequential_edges": self._sequential_edges_since_optimization,
            "has_run_global_ba": self._has_run_global_ba,
            "geometry_updates": dict(self._geometry_updates),
            "geometry_revision": int(self._geometry_revision),
            "pose_state_revision": int(self._pose_state_revision),
            "window_pose_revision": dict(self._window_pose_revision),
            "mapper_pose_revision_by_frame": dict(
                self._mapper_pose_revision_by_frame
            ),
            "last_pose_state_diagnostic": copy.deepcopy(
                self._last_pose_state_diagnostic
            ),
            "last_mapper_committed_state_diagnostic": copy.deepcopy(
                self._last_mapper_committed_state_diagnostic
            ),
            "pending_geometry_batch": self._pending_geometry_batch,
            "pending_map_optimization": list(self._pending_map_optimization),
            "optimization_packets": dict(self._optimization_packets),
            "packet_geometry_variants": packet_variants,
            "pending_seam_owner_windows": set(
                self._pending_seam_owner_windows
            ),
            "results": list(self.results),
            "rendered_overlap_diagnostic": self._last_rendered_overlap_diagnostic,
            "overlap_alignment_failure": self._last_overlap_alignment_failure,
            "fusion_depth_selected_mode": self.fusion._depth_selected_mode,
            "fusion_last_pre_cap_count": self.fusion.last_pre_cap_count,
            "fusion_last_saturated": self.fusion.last_saturated,
            "map_parameters": dict(self.map._parameters),
            "map_metadata": {
                name: getattr(self.map, name)
                for name in self._MAP_TRANSACTION_METADATA
            },
            "map_lazy_reference": {
                int(owner): value.clone()
                for owner, value in self.map._lazy_owner_reference_transforms.items()
            },
            "map_lazy_current": {
                int(owner): value.clone()
                for owner, value in self.map._lazy_owner_current_transforms.items()
            },
            "map_lazy_sh_cache": dict(self.map._lazy_sh_rotation_cache),
            "loop_memory": list(self.loop_detector.memory),
            "loop_descriptor_offsets": list(
                getattr(self.loop_detector, "_descriptor_offsets", [])
            ),
            "loop_descriptor_rows": int(
                getattr(self.loop_detector, "_descriptor_rows", 0)
            ),
            "loop_descriptor_database": loop_database,
            "mapper_optimizer": (
                None if self.mapper is None else self.mapper.optimizer
            ),
            "mapper_geometry_state": (
                None
                if self.mapper is None
                or not callable(
                    getattr(
                        self.mapper,
                        "snapshot_frontend_geometry_state",
                        None,
                    )
                )
                else self.mapper.snapshot_frontend_geometry_state()
            ),
            "mapper_anchor_count": (
                None
                if self.mapper is None
                else int(self.mapper.stats.n_anchors)
            ),
        }

    def _restore_boundary_transaction(self, state: dict[str, Any]) -> None:
        self._restore_graph_state(self.graph, state["graph"])
        self._restore_graph_state(self.submap_graph, state["submap_graph"])
        self.packets = state["packets"]
        self.accepted_loop_pairs = state["accepted_loop_pairs"]
        self.submaps = state["submaps"]
        self.window_to_submap = state["window_to_submap"]
        self._active_submap_id = state["active_submap_id"]
        self._last_full_packet = state["last_full_packet"]
        self.window_order = state["window_order"]
        self.frame_owner_window = state["frame_owner_window"]
        self.frame_depth_owner_window = state["frame_depth_owner_window"]
        self.frame_windows = state["frame_windows"]
        self.frame_pose_owner_node = state["frame_pose_owner_node"]
        self.frame_local_pose_in_owner = state[
            "frame_local_pose_in_owner"
        ]
        self.window_anchor_nodes = state["window_anchor_nodes"]
        self.window_end_nodes = state["window_end_nodes"]
        self.boundary_node_order = state["boundary_node_order"]
        self._chunk_stride_holdouts = state["chunk_stride_holdouts"]
        self._chunk_node_initial_scale = state["chunk_node_initial_scale"]
        self._sequential_edges_since_optimization = state["sequential_edges"]
        self._has_run_global_ba = state["has_run_global_ba"]
        self._geometry_updates = state["geometry_updates"]
        self._geometry_revision = state["geometry_revision"]
        self._pose_state_revision = state["pose_state_revision"]
        self._window_pose_revision = state["window_pose_revision"]
        self._mapper_pose_revision_by_frame = state[
            "mapper_pose_revision_by_frame"
        ]
        self._last_pose_state_diagnostic = state[
            "last_pose_state_diagnostic"
        ]
        self._last_mapper_committed_state_diagnostic = state[
            "last_mapper_committed_state_diagnostic"
        ]
        self._pending_geometry_batch = state["pending_geometry_batch"]
        self._pending_map_optimization = state["pending_map_optimization"]
        self._optimization_packets = state["optimization_packets"]
        for (
            variant,
            local_poses,
            observation,
            anchor_observation,
            metadata,
        ) in state.get(
            "packet_geometry_variants", ()
        ):
            variant.local_poses_c2w = local_poses
            variant.observation = observation
            variant.anchor_observation = anchor_observation
            variant.metadata.clear()
            variant.metadata.update(metadata)
        self._pending_seam_owner_windows = state[
            "pending_seam_owner_windows"
        ]
        self.results = state["results"]
        self._last_rendered_overlap_diagnostic = state[
            "rendered_overlap_diagnostic"
        ]
        self._last_overlap_alignment_failure = state[
            "overlap_alignment_failure"
        ]
        self.fusion._depth_selected_mode = state["fusion_depth_selected_mode"]
        self.fusion.last_pre_cap_count = state["fusion_last_pre_cap_count"]
        self.fusion.last_saturated = state["fusion_last_saturated"]
        self.map._parameters.clear()
        self.map._parameters.update(state["map_parameters"])
        for name, value in state["map_metadata"].items():
            setattr(self.map, name, value)
        self.map._lazy_owner_reference_transforms = state[
            "map_lazy_reference"
        ]
        self.map._lazy_owner_current_transforms = state["map_lazy_current"]
        self.map._lazy_sh_rotation_cache = state["map_lazy_sh_cache"]
        self.loop_detector.memory = state["loop_memory"]
        self.loop_detector._descriptor_offsets = state[
            "loop_descriptor_offsets"
        ]
        self.loop_detector._descriptor_rows = state["loop_descriptor_rows"]
        self.loop_detector._descriptor_database = state[
            "loop_descriptor_database"
        ]
        if self.mapper is not None:
            mapper_geometry_state = state.get("mapper_geometry_state")
            if mapper_geometry_state is not None and callable(
                getattr(
                    self.mapper,
                    "restore_frontend_geometry_state",
                    None,
                )
            ):
                self.mapper.restore_frontend_geometry_state(
                    mapper_geometry_state
                )
            if hasattr(self.mapper, "optimizer"):
                self.mapper.optimizer = state["mapper_optimizer"]
            if hasattr(self.mapper, "stats"):
                self.mapper.stats.n_anchors = state["mapper_anchor_count"]

    def _dense_factor_information_options(self) -> dict[str, bool | float]:
        return {
            "normalize_information_by_count": self.normalize_dense_information_by_count,
            "information_reference_count": self.dense_information_reference_count,
        }

    @staticmethod
    def _node_from_local_pose(
        anchor_to_global: torch.Tensor,
        local_pose_c2w: torch.Tensor,
    ) -> torch.Tensor:
        global_pose = apply_sim3_to_c2w(
            anchor_to_global.to(local_pose_c2w), local_pose_c2w
        )
        scale, _, _ = sim3_components(anchor_to_global)
        return sim3_from_components(
            scale,
            global_pose[:3, :3],
            global_pose[:3, 3],
        )

    def _window_anchor_transforms(self) -> dict[int, torch.Tensor]:
        if self.hierarchical_submaps_enabled:
            assert self.submap_graph is not None
            transforms: dict[int, torch.Tensor] = {}
            for window_id, submap_id in self.window_to_submap.items():
                record = self.submaps[int(submap_id)]
                local = record.local_window_transforms.get(int(window_id))
                if local is None or int(submap_id) not in self.submap_graph.nodes:
                    continue
                transforms[int(window_id)] = (
                    self.submap_graph.transform(int(submap_id)).to(local) @ local
                ).clone()
            return transforms
        if not self.boundary_frame_graph:
            return {
                int(window_id): self.graph.transform(int(window_id)).clone()
                for window_id in self.window_order
                if int(window_id) in self.graph.nodes
            }
        return {
            int(window_id): self.graph.transform(int(anchor_node)).clone()
            for window_id, anchor_node in self.window_anchor_nodes.items()
            if int(anchor_node) in self.graph.nodes
        }

    def _recent_boundary_window_nodes(
        self,
        *,
        include_window_id: int | None = None,
    ) -> list[int]:
        """Return the configured recent graph nodes for local optimization."""

        if self.chunk_first_stride_graph:
            # In chunk-first mode every boundary entry is already one canonical
            # chunk node.  Selecting by windows would include the target node of
            # the newest window as an extra seventh node when active_nodes=6.
            return [
                int(node)
                for node in self.boundary_node_order
                if int(node) in self.graph.nodes
            ][-self.global_ba_active_nodes :]

        ordered_windows = list(self.window_order)
        if (
            include_window_id is not None
            and int(include_window_id) not in ordered_windows
        ):
            ordered_windows.append(int(include_window_id))
        selected = ordered_windows[-self.global_ba_active_nodes :]
        nodes: list[int] = []
        for window_id in selected:
            next_node = self.window_end_nodes.get(int(window_id))
            for node in (
                self.window_anchor_nodes.get(int(window_id)),
                next_node,
            ):
                if (
                    node is not None
                    and int(node) in self.graph.nodes
                    and int(node) not in nodes
                ):
                    nodes.append(int(node))
        return nodes

    def _overlap_seam_diagnostics(
        self,
        *,
        affected_window_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        if self.chunk_first_stride_graph:
            del affected_window_ids
            return dict(
                self._last_pose_state_diagnostic
                or {
                    "enabled": True,
                    "mode": "canonical_pose_state_consistency",
                    "accepted": True,
                    "candidate_revision": int(self._pose_state_revision),
                    "committed_revision": int(self._pose_state_revision),
                    "reason": "no_pose_candidate",
                }
            )
        rotation_errors: list[float] = []
        center_errors: list[float] = []
        factor_count = 0
        for factor in self.graph.edges:
            if not isinstance(factor, CoincidentPanoramaFactor):
                continue
            if factor.edge_type not in {
                "overlap_shared_pose_consistency",
                "overlap_known_pose_bridge_pose_consistency",
            }:
                continue
            if factor.source not in self.graph.nodes or factor.target not in self.graph.nodes:
                continue
            source_pose = apply_sim3_to_c2w(
                self.graph.transform(factor.source).to(factor.source_local_pose),
                factor.source_local_pose,
            )
            target_pose = apply_sim3_to_c2w(
                self.graph.transform(factor.target).to(factor.target_local_pose),
                factor.target_local_pose,
            )
            rotation_errors.append(
                self._rotation_error_deg(
                    source_pose[:3, :3], target_pose[:3, :3]
                )
            )
            center_errors.append(
                float(
                    torch.linalg.norm(
                        source_pose[:3, 3] - target_pose[:3, 3]
                    )
                    .detach()
                    .cpu()
                )
            )
            factor_count += 1
        max_rotation = max(rotation_errors, default=0.0)
        max_center = max(center_errors, default=0.0)
        return {
            "enabled": self.post_optimization_seam_check_enabled,
            "quality_gating_enabled": False,
            "factor_count": factor_count,
            "max_rotation_error_deg": max_rotation,
            "max_center_error": max_center,
            "rotation_errors_deg": rotation_errors,
            "center_errors": center_errors,
            "accepted": True,
        }

    @staticmethod
    def _owner_transforms_changed(
        old: dict[int, torch.Tensor],
        new: dict[int, torch.Tensor],
        *,
        tolerance: float = 1.0e-8,
    ) -> bool:
        for owner in set(old) & set(new):
            if not torch.allclose(
                old[owner].to(new[owner]),
                new[owner],
                atol=float(tolerance),
                rtol=float(tolerance),
            ):
                return True
        return False

    def _register_hierarchical_window(
        self,
        window_id: int,
        start_node: int,
        end_node: int,
    ) -> None:
        if not self.hierarchical_submaps_enabled:
            return
        assert self.submap_graph is not None
        active = (
            None
            if self._active_submap_id is None
            else self.submaps[self._active_submap_id]
        )
        if active is None or active.frozen or len(active.window_ids) >= self.windows_per_submap:
            submap_id = len(self.submaps)
            anchor_transform = self.graph.transform(int(start_node)).clone()
            record = HierarchicalSubmapRecord(
                submap_id=submap_id,
                anchor_node_id=int(start_node),
            )
            self.submaps[submap_id] = record
            self._active_submap_id = submap_id
            self.submap_graph.add_node(submap_id, anchor_transform)
            if submap_id > 0:
                previous_id = submap_id - 1
                previous = self.submap_graph.transform(previous_id)
                measurement = sim3_inverse(previous) @ anchor_transform.to(previous)
                information = previous.new_tensor([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 5.0])
                self.submap_graph.add_edge(
                    Sim3GraphEdge(
                        source=previous_id,
                        target=submap_id,
                        measurement_target_to_source=measurement,
                        information_diag=information,
                        edge_type="submap_sequential",
                        metadata={
                            "submap_anchor_node": int(start_node),
                            "independent_chunk_anchor": bool(
                                self.two_frame_overlap_enabled
                            ),
                            "windows_per_submap": self.windows_per_submap,
                        },
                    )
                )
            active = record
        active.window_ids.append(int(window_id))
        for node in (int(start_node), int(end_node)):
            if node not in active.boundary_node_ids:
                active.boundary_node_ids.append(node)
        self.window_to_submap[int(window_id)] = int(active.submap_id)
        self._update_submap_local_geometry(active.submap_id)

    def _update_submap_local_geometry(self, submap_id: int) -> None:
        if not self.hierarchical_submaps_enabled:
            return
        assert self.submap_graph is not None
        record = self.submaps[int(submap_id)]
        submap_transform = self.submap_graph.transform(int(submap_id))
        inverse = sim3_inverse(submap_transform)
        for node in record.boundary_node_ids:
            if int(node) in self.graph.nodes:
                record.local_boundary_transforms[int(node)] = (
                    inverse @ self.graph.transform(int(node)).to(inverse)
                ).detach()
        for window_id in record.window_ids:
            anchor_node = self.window_anchor_nodes[int(window_id)]
            record.local_window_transforms[int(window_id)] = (
                inverse @ self.graph.transform(int(anchor_node)).to(inverse)
            ).detach()

    def _freeze_active_submap_if_ready(self) -> bool:
        if not self.hierarchical_submaps_enabled or self._active_submap_id is None:
            return False
        record = self.submaps[self._active_submap_id]
        if len(record.window_ids) < self.windows_per_submap:
            return False
        self._update_submap_local_geometry(record.submap_id)
        if self.compress_frozen_dense_factors:
            record.compressed_dense_factors = self._compress_frozen_submap_factors(record)
        record.frozen = True
        self._active_submap_id = None
        return True

    def _compress_frozen_submap_factors(
        self,
        record: HierarchicalSubmapRecord,
    ) -> int:
        """Replace frozen local dense factors with compact relative Sim(3) summaries."""

        boundary_nodes = {int(node) for node in record.boundary_node_ids}
        if not boundary_nodes:
            return 0
        frozen_predecessor_nodes = {
            int(node)
            for submap_id, candidate in self.submaps.items()
            if int(submap_id) != int(record.submap_id) and candidate.frozen
            for node in candidate.boundary_node_ids
        }
        retained: list[
            DenseSphericalFactorBlock | Sim3GraphEdge | CoincidentPanoramaFactor
        ] = []
        internal_holdouts_to_release: list[tuple[int, int, str]] = []
        compressed = 0
        for factor in self.graph.edges:
            source = int(factor.source)
            target = int(factor.target)
            within_record = (
                source in boundary_nodes and target in boundary_nodes
            )
            cross_frozen_boundary = (
                isinstance(factor, DenseSphericalFactorBlock)
                and factor.edge_type
                in {
                    "overlap_dense_spherical",
                }
                and (
                    (
                        source in frozen_predecessor_nodes
                        and target in boundary_nodes
                    )
                    or (
                        target in frozen_predecessor_nodes
                        and source in boundary_nodes
                    )
                )
            )
            should_compress = (
                isinstance(factor, DenseSphericalFactorBlock)
                and (
                    (
                        factor.edge_type
                        in {
                            "chunk_stride_dense_spherical",
                            "chunk_skip_dense_spherical",
                        }
                        and within_record
                    )
                    or (
                        factor.edge_type
                        in {
                            "boundary_dense_spherical",
                            "overlap_dense_spherical",
                        }
                        and (within_record or cross_frozen_boundary)
                    )
                )
            )
            if not should_compress:
                retained.append(factor)
                continue
            source_transform = self.graph.transform(int(factor.source))
            target_transform = self.graph.transform(int(factor.target)).to(source_transform)
            count = max(1, int(factor.source_depth.numel()))
            # Confidence is deliberately sub-linear and capped: more matches
            # improve certainty without allowing a single dense block to
            # dominate the complete graph solely because it is larger.
            confidence = min(8.0, max(1.0, math.sqrt(float(count) / 64.0)))
            s2_information_scale = max(
                0.0, float(factor.s2_information_scale)
            )
            depth_information_scale = max(
                0.0, float(factor.depth_information_scale)
            )
            information = source_transform.new_tensor(
                [
                    depth_information_scale,
                    depth_information_scale,
                    depth_information_scale,
                    2.0 * s2_information_scale,
                    2.0 * s2_information_scale,
                    2.0 * s2_information_scale,
                    0.5 * depth_information_scale,
                ]
            ) * confidence
            retained.append(
                Sim3GraphEdge(
                    source=int(factor.source),
                    target=int(factor.target),
                    measurement_target_to_source=(
                        sim3_inverse(source_transform) @ target_transform
                    ).detach(),
                    information_diag=information,
                    edge_type="compressed_dense_summary",
                    metadata={
                        "submap_id": int(record.submap_id),
                        "source_edge_type": factor.edge_type,
                        "source_correspondence_count": count,
                        "information_confidence": confidence,
                        "s2_information_scale": s2_information_scale,
                        "depth_information_scale": depth_information_scale,
                    },
                )
            )
            if within_record and factor.edge_type in {
                "chunk_stride_dense_spherical",
                "chunk_skip_dense_spherical",
            }:
                internal_holdouts_to_release.append(
                    (source, target, str(factor.edge_type))
                )
            compressed += 1
        self.graph.edges = retained
        for key in internal_holdouts_to_release:
            self._chunk_stride_holdouts.pop(key, None)
        return compressed

    def _apply_submap_graph_to_boundary_graph(self) -> None:
        if not self.hierarchical_submaps_enabled:
            return
        assert self.submap_graph is not None
        # Overlap-2 submaps use independent chunk-anchor nodes. Legacy
        # overlap-1 records may still share a boundary node, in which case the
        # later submap intentionally writes that node last.
        for submap_id in sorted(self.submaps):
            record = self.submaps[submap_id]
            transform = self.submap_graph.transform(submap_id)
            for node, local in record.local_boundary_transforms.items():
                self.graph.nodes[int(node)] = (transform.to(local) @ local).detach()

    @staticmethod
    def _rescale_packet_geometry(
        packet: LocalGaussianWindowPacket,
        scale: float,
    ) -> None:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid chunk scale normalization: {value}")
        if abs(value - 1.0) <= 1.0e-8:
            return
        poses = packet.local_poses_c2w.detach().clone()
        poses[:, :3, 3] *= value
        depth = packet.observation.refined_depth.detach().clone() * value
        packet.local_poses_c2w = poses
        packet.observation = packet.observation.with_geometry(
            poses_c2w=poses.unsqueeze(0).to(packet.observation.poses_c2w),
            refined_depth=depth.to(packet.observation.refined_depth),
        )
        if packet.pre_depth_shift_depth is not None:
            packet.pre_depth_shift_depth = packet.pre_depth_shift_depth * value
        if packet.anchor_observation is not None:
            packet.anchor_observation = packet.anchor_observation.rescale_geometry(value)
        packet.metadata["global_alignment_local_scale"] = value

    @staticmethod
    def _rescaled_packet_copy(
        packet: LocalGaussianWindowPacket,
        scale: float,
    ) -> LocalGaussianWindowPacket:
        """Return an isolated packet whose complete geometry uses ``scale``."""

        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid chunk scale normalization: {value}")
        poses = packet.local_poses_c2w.detach().clone()
        poses[:, :3, 3] *= value
        poses[0] = torch.eye(4, device=poses.device, dtype=poses.dtype)
        depth = packet.observation.refined_depth.detach().clone() * value
        initial_depth = packet.observation.initial_depth.detach().clone() * value
        geometry = packet.observation.with_geometry(
            poses_c2w=poses.unsqueeze(0).to(packet.observation.poses_c2w),
            refined_depth=depth.to(packet.observation.refined_depth),
        )
        observation = replace(
            geometry,
            initial_depth=initial_depth.to(geometry.initial_depth),
            depth_residual=(
                depth.to(geometry.refined_depth)
                - initial_depth.to(geometry.refined_depth)
            ),
        )
        metadata = dict(packet.metadata)
        metadata["global_alignment_local_scale"] = value
        return replace(
            packet,
            local_poses_c2w=poses,
            observation=observation,
            pre_depth_shift_depth=(
                None
                if packet.pre_depth_shift_depth is None
                else packet.pre_depth_shift_depth.detach().clone() * value
            ),
            anchor_observation=(
                None
                if packet.anchor_observation is None
                else packet.anchor_observation.rescale_geometry(value)
            ),
            metadata=metadata,
        )

    @staticmethod
    def _packet_uses_voxel_refiner(packet: LocalGaussianWindowPacket) -> bool:
        return bool(
            packet.metadata.get("voxel_anchor_refiner_requested", False)
            or packet.metadata.get("voxel_anchor_refiner_enabled", False)
        )

    def _validate_refined_packet(self, packet: LocalGaussianWindowPacket) -> bool:
        refined = self._packet_uses_voxel_refiner(packet)
        if self.voxel_anchor_refiner_enabled and not refined:
            raise RuntimeError(
                "VoxelAnchorRefiner is enabled but the frontend packet is not marked refined"
            )
        if not refined:
            return False
        if packet.anchor_observation is None:
            raise RuntimeError(
                "A Refiner-routed packet must contain provisional or final anchors"
            )
        pending = bool(packet.metadata.get("voxel_anchor_refiner_pending", False))
        if pending and self.pose_canonicalized_packet_refiner is None:
            raise RuntimeError(
                "A deferred Refiner packet requires a pose-canonicalized "
                "packet refiner callback"
            )
        if pending and not self.two_frame_known_pose_bridge_enabled:
            raise RuntimeError(
                "Deferred post-bridge Refiner packets require a "
                "two_frame_bridge_* alignment mode"
            )
        if not self.rendered_overlap_alignment_enabled:
            raise RuntimeError(
                "Refined packets require rendered_overlap_alignment.enabled=true"
            )
        if not self.insertion_dedup_enabled:
            raise RuntimeError(
                "Refined packets require insertion_dedup.enabled=true"
            )
        if self.renderer is None:
            raise RuntimeError("Rendered refined-anchor alignment requires a renderer")
        return True

    def _finalize_pose_canonicalized_refiner_packet(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> LocalGaussianWindowPacket:
        if not bool(packet.metadata.get("voxel_anchor_refiner_pending", False)):
            if not bool(packet.metadata.get("voxel_anchor_refiner_enabled", False)):
                raise RuntimeError(
                    "Refiner-routed packet is neither pending nor finalized"
                )
            return packet
        callback = self.pose_canonicalized_packet_refiner
        if callback is None:
            raise RuntimeError("Deferred Refiner callback is unavailable")
        refined = callback(packet)
        if refined.anchor_observation is None or not bool(
            refined.metadata.get("voxel_anchor_refiner_enabled", False)
        ):
            raise RuntimeError(
                "Pose-canonicalized Refiner callback did not return final anchors"
            )
        if bool(refined.metadata.get("voxel_anchor_refiner_pending", False)):
            raise RuntimeError("Pose-canonicalized Refiner packet is still pending")
        return refined

    def _prepare_chunk_first_two_stage_candidate(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> _PreparedPacketCandidate:
        """Canonicalize one packet without mutating graph, map, or history."""

        if not self.chunk_first_stride_graph:
            raise RuntimeError(
                "Two-stage packet preparation requires chunk_first_stride"
            )
        refined_packet = self._validate_refined_packet(packet)
        refiner_pending = bool(
            packet.metadata.get("voxel_anchor_refiner_pending", False)
        )
        if refined_packet:
            packet = self._rescaled_packet_copy(packet, 1.0)
        window_id = int(packet.window_id)
        if window_id in self.packets:
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")
        if len(packet.frame_ids) <= self.chunk_stride_target_index:
            raise ValueError(
                "chunk_first_stride requires the configured next-anchor frame"
            )
        start_frame = int(packet.frame_ids[0])
        if not self.window_order:
            start_transform = sim3_identity(
                device=packet.local_poses_c2w.device,
                dtype=packet.local_poses_c2w.dtype,
            )
            diagnostics: dict[str, Any] = {
                "reason": "first_window",
                "mode": self.rendered_overlap_alignment_mode,
                "shared_scale": 1.0,
                "s_shared": 1.0,
                "absolute_scale": 1.0,
                "s_absolute": 1.0,
                "chunk_scale_normalization": 1.0,
                "c": 1.0,
                "accepted": True,
            }
        else:
            previous_id = int(self.window_order[-1])
            previous_packet = self._last_full_packet
            if (
                previous_packet is None
                or int(previous_packet.window_id) != previous_id
            ):
                raise RuntimeError(
                    "The previous full-resolution window packet is unavailable"
                )
            if start_frame not in self.graph.nodes:
                raise RuntimeError(
                    f"Canonical chunk-first node {start_frame} is missing"
                )
            previous_transform = self._window_anchor_transforms()[
                previous_id
            ].to(packet.local_poses_c2w)
            local_scale, diagnostics = self._estimate_canonical_ba_overlap_scale(
                previous_packet,
                packet,
            )
            if local_scale is None:
                raise RuntimeError(
                    f"Window {window_id} BA-overlap scale failed: "
                    f"{diagnostics.get('reason', 'unknown')}"
                )
            normalized = self._rescaled_packet_copy(packet, local_scale)
            start_transform = self.graph.transform(start_frame).clone().to(
                normalized.local_poses_c2w
            )
            overlap = self._overlap_frame_ids(previous_packet, normalized)
            known_global_poses = tuple(
                self._known_overlap_global_pose(
                    previous_packet,
                    frame_id,
                    previous_transform,
                )
                for frame_id in overlap
            )
            predicted_second = apply_sim3_to_c2w(
                start_transform,
                normalized.local_poses_c2w[1],
            )
            overlap_rotation_error = self._rotation_error_deg(
                known_global_poses[1][:3, :3].to(predicted_second),
                predicted_second[:3, :3],
            )
            overlap_center_error = float(
                torch.linalg.norm(
                    known_global_poses[1][:3, 3].to(predicted_second)
                    - predicted_second[:3, 3]
                )
                .detach()
                .cpu()
            )
            diagnostics = dict(diagnostics)
            diagnostics.update(
                {
                    "graph_role": "diagnostic_only_no_factor",
                    "existing_node_id": start_frame,
                    "raw_ba_to_canonical_rotation_error_deg": (
                        overlap_rotation_error
                    ),
                    "raw_ba_to_canonical_center_error": overlap_center_error,
                    "node_sim3_scale_updated": False,
                    "quality_gating_enabled": False,
                    "accepted": True,
                    "reason": "accepted_without_overlap_pose_gate",
                }
            )
            packet = self._canonicalize_packet_from_two_known_poses(
                normalized,
                start_transform,
                (known_global_poses[0], known_global_poses[1]),
            )
            packet.metadata["global_alignment_local_scale"] = float(local_scale)
        packet.metadata["two_stage_candidate_prepared"] = True
        packet.metadata["two_stage_candidate_map_anchor_count"] = int(
            self.map.anchor_count()
        )
        return _PreparedPacketCandidate(
            packet=packet,
            start_transform=start_transform.detach().clone(),
            alignment_diagnostics=dict(diagnostics),
            aligned=True,
            refined_packet=bool(refined_packet),
            refiner_pending=bool(refiner_pending),
            canonicalized=True,
        )

    def prepare_packet_candidate(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> int:
        """Prepare an immutable backend candidate before its RGB is registered."""

        window_id = int(packet.window_id)
        if (
            window_id in self._prepared_packet_candidates
            or window_id in self.packets
        ):
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")
        use_two_stage = bool(
            self.two_stage_map_optimization_enabled
            and self.boundary_frame_graph
            and self.chunk_first_stride_graph
            and self._packet_uses_voxel_refiner(packet)
            and len(packet.frame_ids) == 4
        )
        try:
            candidate = (
                self._prepare_chunk_first_two_stage_candidate(packet)
                if use_two_stage
                else _PreparedPacketCandidate(packet=packet)
            )
        except Exception:
            if (
                self._packet_uses_voxel_refiner(packet)
                and self.packet_refiner_release is not None
            ):
                self.packet_refiner_release(window_id)
            raise
        self._prepared_packet_candidates[window_id] = candidate
        self._prepared_packet_candidate_order.append(window_id)
        return window_id

    def pending_packet_candidate_ids(self) -> tuple[int, ...]:
        return tuple(self._prepared_packet_candidate_order)

    def track_and_commit_candidate(
        self,
        window_id: int,
    ) -> GlobalWindowBackendResult:
        """Track, refine, fuse, and atomically commit one prepared candidate."""

        value = int(window_id)
        candidate = self._prepared_packet_candidates.get(value)
        if candidate is None:
            raise KeyError(f"Unknown prepared packet candidate {value}")
        try:
            if candidate.canonicalized and self.mapper is not None:
                prepared = self.mapper.prepare_spherical_selfi_window(
                    candidate.packet.frame_ids
                )
                if prepared != len(candidate.packet.frame_ids):
                    raise RuntimeError(
                        f"window {value} has "
                        f"{prepared}/{len(candidate.packet.frame_ids)} "
                        "registered RGB observations"
                    )
            if self.boundary_frame_graph:
                return self._process_boundary_packet(
                    candidate.packet,
                    prepared_candidate=(
                        candidate if candidate.canonicalized else None
                    ),
                )
            return self._process_window_anchor_packet(candidate.packet)
        finally:
            self._prepared_packet_candidates.pop(value, None)
            self._prepared_packet_candidate_order = [
                item
                for item in self._prepared_packet_candidate_order
                if int(item) != value
            ]

    @staticmethod
    def _single_camera_render_tensor(
        package: dict[str, Any],
        name: str,
    ) -> torch.Tensor:
        value = package.get(name)
        if not torch.is_tensor(value):
            raise RuntimeError(f"Renderer package is missing tensor {name!r}")
        if value.ndim == 4:
            if int(value.shape[0]) != 1:
                raise RuntimeError(f"Single-camera {name} must have a leading size of one")
            value = value[0]
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or int(value.shape[0]) != 1:
            raise RuntimeError(
                f"Single-camera {name} must have shape 1xHxW, got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _single_camera_visibility(
        package: dict[str, Any],
        *,
        expected_count: int,
        require_accumulated: bool = False,
    ) -> torch.Tensor:
        accumulated = package.get("accum_visible")
        if require_accumulated and not torch.is_tensor(accumulated):
            raise RuntimeError(
                "Rendered alignment/hash requires gsplat360 accum_visible"
            )
        value = (
            accumulated
            if torch.is_tensor(accumulated)
            else package.get("visibility_filter")
        )
        if not torch.is_tensor(value):
            raise RuntimeError("Renderer package is missing visibility_filter")
        if value.ndim == 2:
            if int(value.shape[0]) != 1:
                raise RuntimeError(
                    "Single-camera visibility_filter must have a leading size of one"
                )
            value = value[0]
        value = value.bool().reshape(-1)
        if int(value.numel()) != int(expected_count):
            raise RuntimeError(
                "Renderer visibility_filter does not match the Gaussian count: "
                f"{int(value.numel())} != {int(expected_count)}"
            )
        return value

    def _render_refined_anchor_frame(
        self,
        packet: LocalGaussianWindowPacket,
        frame_id: int,
        *,
        exclude_target_only_anchors: bool = False,
    ) -> RenderedSharedFrame:
        if self.renderer is None or packet.anchor_observation is None:
            raise RuntimeError("Refined-anchor rendering requires anchors and a renderer")
        anchor = packet.anchor_observation
        height, width = anchor.image_size
        frame_index = packet.frame_index(int(frame_id))
        camera_pose = anchor.local_poses_c2w[0, frame_index].to(
            device=anchor.xyz.device,
            dtype=anchor.xyz.dtype,
        )
        camera = PanoRenderCamera(height, width, camera_pose)
        selected_indices = None
        if exclude_target_only_anchors:
            membership = anchor.membership
            if int(membership.anchor_index.numel()) == 0:
                raise RuntimeError(
                    "Non-self overlap rendering requires provisional anchor membership"
                )
            supported_elsewhere = torch.zeros(
                anchor.num_anchors,
                device=anchor.xyz.device,
                dtype=torch.bool,
            )
            other = membership.source_view_index != int(frame_index)
            if bool(other.any()):
                supported_elsewhere[
                    membership.anchor_index[other].to(supported_elsewhere.device)
                ] = True
            selected_indices = torch.nonzero(
                supported_elsewhere
                & (anchor.batch_index == 0),
                as_tuple=False,
            ).flatten()
            if int(selected_indices.numel()) == 0:
                raise RuntimeError(
                    f"Frame {frame_id} has no non-self-supported provisional anchors"
                )
        explicit = anchor.materialize_batch(
            [camera],
            batch_index=0,
            anchor_indices=selected_indices,
        )
        if anchor.xyz.device.type == "cuda":
            torch.cuda.synchronize(anchor.xyz.device)
        start = time.perf_counter()
        with torch.inference_mode():
            package = self.renderer.render_cameras([camera], explicit)
        if anchor.xyz.device.type == "cuda":
            torch.cuda.synchronize(anchor.xyz.device)
        elapsed = float(time.perf_counter() - start)
        explicit_visibility = self._single_camera_visibility(
            package,
            expected_count=int(explicit.anchor_indices.numel()),
            require_accumulated=isinstance(self.renderer, PFGS360Renderer),
        )
        visibility = torch.zeros(
            anchor.num_anchors,
            device=anchor.xyz.device,
            dtype=torch.bool,
        )
        visibility[explicit.anchor_indices] = explicit_visibility.to(visibility.device)
        return RenderedSharedFrame(
            depth=self._single_camera_render_tensor(package, "depth"),
            alpha=self._single_camera_render_tensor(package, "alpha"),
            anchor_visibility=visibility,
            render_seconds=elapsed,
        )

    def _render_refined_anchor_shared_frame(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> RenderedSharedFrame:
        """Legacy first-frame renderer used by scale-only alignment."""

        return self._render_refined_anchor_frame(packet, int(packet.frame_ids[0]))

    def _render_global_pose_frame(
        self,
        pose_c2w: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> RenderedSharedFrame:
        if self.renderer is None:
            raise RuntimeError("Global shared-frame rendering requires a renderer")
        height, width = (int(value) for value in image_size)
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        c2w = pose_c2w.to(device=device, dtype=dtype)
        if tuple(c2w.shape) != (4, 4):
            raise ValueError("Global render pose must have shape 4x4")
        camera = PanoRenderCamera(height, width, c2w)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            package = self.renderer.render_cameras([camera], self.map)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - start)
        return RenderedSharedFrame(
            depth=self._single_camera_render_tensor(package, "depth"),
            alpha=self._single_camera_render_tensor(package, "alpha"),
            anchor_visibility=self._single_camera_visibility(
                package,
                expected_count=self.map.anchor_count(),
                require_accumulated=isinstance(self.renderer, PFGS360Renderer),
            ),
            render_seconds=elapsed,
        )

    def _render_global_shared_frame(
        self,
        shared_transform: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> RenderedSharedFrame:
        if self.renderer is None:
            raise RuntimeError("Global shared-frame rendering requires a renderer")
        height, width = (int(value) for value in image_size)
        device, dtype = self.map.xyz.device, self.map.xyz.dtype
        _, rotation, translation = sim3_components(
            shared_transform.to(device=device, dtype=dtype)
        )
        c2w = torch.eye(4, device=device, dtype=dtype)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = translation
        return self._render_global_pose_frame(
            c2w,
            image_size=(height, width),
        )

    def _estimate_rendered_depth_scale(
        self,
        local_depth: torch.Tensor,
        global_depth: torch.Tensor,
        *,
        local_valid: torch.Tensor,
        global_valid: torch.Tensor,
        local_sky_probability: torch.Tensor | None,
        global_sky_probability: torch.Tensor | None,
        shared_scale: float,
        seed: int,
    ) -> tuple[float | None, dict[str, Any], torch.Tensor, torch.Tensor]:
        """Estimate the absolute local-to-world scale from shared-frame depth."""

        shared = float(shared_scale)
        if not math.isfinite(shared) or shared <= 0.0:
            raise ValueError("The shared graph-node scale must be positive and finite")
        solve_start = time.perf_counter()
        samples = sample_joint_valid_fibonacci_uv(
            local_depth,
            global_depth,
            count=self.rendered_alignment_max_points,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            source_valid=local_valid,
            target_valid=global_valid,
            source_sky_probability=local_sky_probability,
            target_sky_probability=global_sky_probability,
            sky_threshold=self.sky_threshold,
            seed=int(seed),
        )
        count = int(samples.source_depth.numel())
        diagnostics: dict[str, Any] = {
            "mode": "shared_frame_scale_only",
            "shared_scale": shared,
            "s_shared": shared,
            "valid_points": count,
            "fibonacci_seed": int(samples.seed),
            "fibonacci_longitude_phase": float(samples.longitude_phase),
        }
        empty_inliers = torch.zeros(count, device=samples.source_depth.device, dtype=torch.bool)
        if count < self.rendered_alignment_min_points:
            diagnostics.update(
                {
                    "reason": "insufficient_rendered_overlap_support",
                    "accepted": False,
                    "scale_solve_seconds": float(time.perf_counter() - solve_start),
                }
            )
            return None, diagnostics, samples.uv, empty_inliers

        log_ratio = (
            samples.target_depth.clamp_min(1.0e-8).log()
            - samples.source_depth.clamp_min(1.0e-8).log()
        )
        estimate = log_ratio.median()
        for _ in range(3):
            residual = log_ratio - estimate
            centered = residual - residual.median()
            mad = centered.abs().median()
            robust_sigma = (1.4826 * mad).clamp_min(1.0e-6)
            huber_delta = (1.345 * robust_sigma).clamp_min(1.0e-4)
            weights = torch.where(
                residual.abs() <= huber_delta,
                torch.ones_like(residual),
                huber_delta / residual.abs().clamp_min(1.0e-8),
            )
            estimate = (weights * log_ratio).sum() / weights.sum().clamp_min(1.0e-8)

        absolute_scale_tensor = estimate.exp()
        absolute_scale = float(absolute_scale_tensor.detach().cpu())
        correction = absolute_scale / shared
        aligned_local = samples.source_depth * absolute_scale_tensor
        relative_error = (
            (aligned_local - samples.target_depth).abs()
            / samples.target_depth.abs().clamp_min(1.0e-6)
        )
        inliers = (
            relative_error <= self.rendered_alignment_max_median_relative_error
        )
        inlier_ratio = float(inliers.float().mean().detach().cpu())
        median_error = float(relative_error.median().detach().cpu())
        p90_error = float(
            torch.quantile(relative_error.float(), 0.90).detach().cpu()
        )
        scale_ok = (
            math.isfinite(correction)
            and 1.0 / self.rendered_alignment_max_scale_change
            <= correction
            <= self.rendered_alignment_max_scale_change
        )
        accepted = (
            scale_ok
            and inlier_ratio >= self.rendered_alignment_min_inlier_ratio
            and median_error <= self.rendered_alignment_max_median_relative_error
        )
        diagnostics.update(
            {
                "absolute_scale": absolute_scale,
                "s_absolute": absolute_scale,
                "chunk_scale_normalization": correction,
                "c": correction,
                "inlier_points": int(inliers.sum().item()),
                "inlier_ratio": inlier_ratio,
                "median_relative_error": median_error,
                "p90_relative_error": p90_error,
                "accepted": bool(accepted),
                "reason": "accepted" if accepted else "rendered_scale_gate_rejected",
                "scale_solve_seconds": float(time.perf_counter() - solve_start),
            }
        )
        return (
            correction if accepted else None,
            diagnostics,
            samples.uv,
            inliers,
        )

    def _rendered_shared_frame_alignment(
        self,
        previous: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
        shared_transform: torch.Tensor,
    ) -> tuple[float | None, dict[str, Any]]:
        if target.anchor_observation is None:
            raise RuntimeError("Rendered overlap alignment requires refined anchors")
        frame_id = int(target.frame_ids[0])
        if int(previous.frame_ids[-1]) != frame_id:
            raise RuntimeError(
                "Rendered overlap alignment requires previous last == current first"
            )
        previous_index = previous.frame_index(frame_id)
        target_index = target.frame_index(frame_id)
        local_render = self._render_refined_anchor_shared_frame(target)
        global_render = self._render_global_shared_frame(
            shared_transform,
            image_size=target.anchor_observation.image_size,
        )
        global_depth = global_render.depth.to(local_render.depth)
        global_alpha = global_render.alpha.to(local_render.alpha)
        local_semantic_valid = (
            target.finite_gaussian_mask[0, target_index]
            & target.static_mask[0, target_index]
            & target.geometry_consistency[0, target_index]
            & ~target.sky_mask[0, target_index]
        ).to(local_render.depth.device)
        global_semantic_valid = (
            previous.finite_gaussian_mask[0, previous_index]
            & previous.static_mask[0, previous_index]
            & previous.geometry_consistency[0, previous_index]
            & ~previous.sky_mask[0, previous_index]
        ).to(local_render.depth.device)
        local_valid = (
            local_semantic_valid
            & torch.isfinite(local_render.depth)
            & (local_render.depth > 0.0)
            & torch.isfinite(local_render.alpha)
            & (local_render.alpha >= self.rendered_alignment_alpha_threshold)
        )
        global_valid = (
            global_semantic_valid
            & torch.isfinite(global_depth)
            & (global_depth > 0.0)
            & torch.isfinite(global_alpha)
            & (global_alpha >= self.rendered_alignment_alpha_threshold)
        )
        shared_scale, _, _ = sim3_components(shared_transform)
        seed = (
            self.fibonacci_seed
            + 1_000_003 * int(previous.window_id)
            + 10_007 * int(target.window_id)
            + 101 * frame_id
        ) & 0x7FFFFFFF
        correction, diagnostics, sampled_uv, inliers = (
            self._estimate_rendered_depth_scale(
                local_render.depth,
                global_depth,
                local_valid=local_valid,
                global_valid=global_valid,
                local_sky_probability=target.sky_prob[0, target_index].to(
                    local_render.depth
                ),
                global_sky_probability=previous.sky_prob[0, previous_index].to(
                    local_render.depth
                ),
                shared_scale=float(shared_scale.detach().cpu()),
                seed=seed,
            )
        )
        diagnostics.update(
            {
                "source_window_id": int(previous.window_id),
                "target_window_id": int(target.window_id),
                "shared_frame_id": frame_id,
                "local_render_seconds": local_render.render_seconds,
                "global_render_seconds": global_render.render_seconds,
                "render_seconds": (
                    local_render.render_seconds + global_render.render_seconds
                ),
                "canonical_rotation_mismatch_deg": 0.0,
                "canonical_translation_mismatch": 0.0,
            }
        )
        inlier_map = torch.zeros_like(local_render.depth, dtype=torch.bool)
        if int(sampled_uv.numel()) > 0 and bool(inliers.any()):
            uv = sampled_uv[inliers]
            columns = torch.floor(uv[:, 0]).long().clamp(
                0, int(inlier_map.shape[-1]) - 1
            )
            rows = torch.floor(uv[:, 1]).long().clamp(
                0, int(inlier_map.shape[-2]) - 1
            )
            inlier_map[0, rows, columns] = True
        sky_union = (
            target.sky_mask[0, target_index]
            | previous.sky_mask[0, previous_index].to(target.sky_mask.device)
        )
        absolute_scale = float(diagnostics.get("absolute_scale", 1.0))
        aligned_local_depth = local_render.depth * absolute_scale
        relative_error = (
            (aligned_local_depth - global_depth).abs()
            / global_depth.abs().clamp_min(1.0e-6)
        )
        self._last_rendered_overlap_diagnostic = {
            "local_depth": local_render.depth.detach().cpu().float(),
            "aligned_local_depth": aligned_local_depth.detach().cpu().float(),
            "global_depth": global_depth.detach().cpu().float(),
            "relative_error": relative_error.detach().cpu().float(),
            "local_alpha": local_render.alpha.detach().cpu().float(),
            "global_alpha": global_alpha.detach().cpu().float(),
            "sky_mask": sky_union.detach().cpu().bool(),
            "valid_mask": (local_valid & global_valid).detach().cpu().bool(),
            "inlier_mask": inlier_map.detach().cpu().bool(),
        }
        return correction, diagnostics

    @staticmethod
    def _rotation_error_deg(
        reference: torch.Tensor,
        estimate: torch.Tensor,
    ) -> float:
        relative = reference.transpose(-1, -2) @ estimate
        cosine = ((relative.diagonal().sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
        return float(torch.rad2deg(torch.acos(cosine)).detach().cpu())

    @staticmethod
    def _pose_state_rotation_error_deg(
        reference: torch.Tensor,
        estimate: torch.Tensor,
    ) -> float:
        """Stable near-zero SO(3) error for strict cache consistency checks."""

        relative = (
            reference.detach().double().transpose(-1, -2)
            @ estimate.detach().double()
        )
        skew = torch.stack(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        sine = 0.5 * torch.linalg.norm(skew)
        cosine = 0.5 * (torch.trace(relative) - 1.0)
        return float(
            torch.rad2deg(torch.atan2(sine, cosine)).detach().cpu()
        )

    @staticmethod
    def _average_rotations(rotations: list[torch.Tensor]) -> torch.Tensor:
        if not rotations:
            raise ValueError("At least one rotation is required")
        matrix = torch.stack(rotations, dim=0).mean(dim=0)
        u, _, vh = torch.linalg.svd(matrix)
        correction = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        correction[-1, -1] = torch.where(
            torch.linalg.det(u @ vh) < 0.0,
            matrix.new_tensor(-1.0),
            matrix.new_tensor(1.0),
        )
        return u @ correction @ vh

    @staticmethod
    def _weighted_point_singular_values(
        points: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        normalized = weights / weights.sum().clamp_min(1.0e-8)
        mean = (normalized[:, None] * points).sum(dim=0)
        centered = points - mean
        covariance = (normalized[:, None] * centered).T @ centered
        return torch.linalg.svdvals(covariance)

    def _overlap_frame_ids(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
    ) -> tuple[int, ...]:
        overlap = tuple(sorted(set(previous.frame_ids) & set(current.frame_ids)))
        if self.enforce_exact_overlap and len(overlap) != self.expected_overlap_frames:
            raise RuntimeError(
                "Unexpected overlap count: "
                f"expected {self.expected_overlap_frames}, got {len(overlap)} "
                f"for windows {previous.window_id}->{current.window_id}"
            )
        return overlap

    @staticmethod
    def _weighted_median_1d(
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return a deterministic weighted median for finite one-dimensional data."""

        value = values.reshape(-1)
        weight = weights.to(value).reshape(-1).clamp_min(0.0)
        if int(value.numel()) == 0 or int(weight.numel()) != int(value.numel()):
            raise ValueError("Weighted median requires equally sized non-empty inputs")
        valid = torch.isfinite(value) & torch.isfinite(weight) & (weight > 0.0)
        if not bool(valid.any()):
            raise ValueError("Weighted median requires positive finite support")
        value = value[valid]
        weight = weight[valid]
        order = torch.argsort(value, stable=True)
        sorted_value = value.index_select(0, order)
        sorted_weight = weight.index_select(0, order)
        threshold = 0.5 * sorted_weight.sum()
        index = int(
            torch.searchsorted(
                sorted_weight.cumsum(dim=0),
                threshold,
                right=False,
            )
            .clamp_max(int(sorted_value.numel()) - 1)
            .item()
        )
        return sorted_value[index]

    def _estimate_canonical_ba_overlap_scale(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
    ) -> tuple[float | None, dict[str, Any]]:
        """Estimate packet normalization using BA depths only, never map renders."""

        overlap = self._overlap_frame_ids(previous, current)
        if len(overlap) != 2:
            return None, {
                "reason": "two_ba_overlap_frames_required",
                "overlap_frame_ids": list(overlap),
                "accepted": False,
            }
        log_ratio_parts: list[torch.Tensor] = []
        frame_ids: list[int] = []
        frame_counts: list[int] = []
        frame_confidence_support: list[int] = []
        frame_confidence_rejected: list[int] = []
        frame_index_parts: list[torch.Tensor] = []
        seeds: list[int] = []
        for frame_slot, frame_id in enumerate(overlap):
            previous_index = previous.frame_index(frame_id)
            current_index = current.frame_index(frame_id)
            previous_depth = previous.observation.refined_depth[
                0, previous_index
            ].detach()
            current_depth = current.observation.refined_depth[
                0, current_index
            ].detach().to(previous_depth)
            previous_semantic_valid = (
                previous.finite_gaussian_mask[0, previous_index]
                & previous.static_mask[0, previous_index]
                & previous.geometry_consistency[0, previous_index]
                & ~previous.sky_mask[0, previous_index]
            ).detach()
            current_semantic_valid = (
                current.finite_gaussian_mask[0, current_index]
                & current.static_mask[0, current_index]
                & current.geometry_consistency[0, current_index]
                & ~current.sky_mask[0, current_index]
            ).detach().to(previous_semantic_valid.device)
            previous_confidence = previous.observation.confidence[
                0, previous_index
            ].detach().to(previous_depth)
            current_confidence = current.observation.confidence[
                0, current_index
            ].detach().to(previous_depth)
            previous_confident = (
                torch.isfinite(previous_confidence)
                & (
                    previous_confidence
                    >= self.rendered_alignment_min_confidence
                )
            )
            current_confident = (
                torch.isfinite(current_confidence)
                & (
                    current_confidence
                    >= self.rendered_alignment_min_confidence
                )
            )
            semantic_joint = (
                previous_semantic_valid & current_semantic_valid
            )
            confident_joint = previous_confident & current_confident
            previous_valid = previous_semantic_valid & previous_confident
            current_valid = current_semantic_valid & current_confident
            frame_confidence_support.append(
                int((semantic_joint & confident_joint).sum().detach().cpu())
            )
            frame_confidence_rejected.append(
                int((semantic_joint & ~confident_joint).sum().detach().cpu())
            )
            seed = (
                self.fibonacci_seed
                + 1_000_003 * int(previous.window_id)
                + 10_007 * int(current.window_id)
                + 101 * int(frame_id)
            ) & 0x7FFFFFFF
            samples = sample_joint_valid_fibonacci_uv(
                current_depth,
                previous_depth,
                count=self.rendered_alignment_max_points_per_frame,
                oversample_factor=self.fibonacci_oversample_factor,
                min_depth=self.fibonacci_min_depth,
                max_depth=self.fibonacci_max_depth,
                source_valid=current_valid,
                target_valid=previous_valid,
                source_sky_probability=current.sky_prob[
                    0, current_index
                ].detach().to(previous_depth),
                target_sky_probability=previous.sky_prob[
                    0, previous_index
                ].detach(),
                sky_threshold=self.sky_threshold,
                seed=seed,
            )
            count = int(samples.source_depth.numel())
            if count < self.rendered_alignment_min_points_per_frame:
                return None, {
                    "reason": "insufficient_ba_overlap_depth_support",
                    "failed_frame_id": int(frame_id),
                    "valid_points": count,
                    "accepted": False,
                    "global_render_used_for_scale": False,
                }
            log_ratio_parts.append(
                samples.target_depth.clamp_min(1.0e-8).log()
                - samples.source_depth.clamp_min(1.0e-8).log()
            )
            frame_index_parts.append(
                torch.full(
                    (count,),
                    frame_slot,
                    device=samples.source_depth.device,
                    dtype=torch.long,
                )
            )
            frame_ids.append(int(frame_id))
            frame_counts.append(count)
            seeds.append(int(samples.seed))

        log_ratio = torch.cat(log_ratio_parts, dim=0)
        frame_index = torch.cat(frame_index_parts, dim=0)
        base_weight = torch.zeros_like(log_ratio)
        for frame_slot in range(2):
            rows = frame_index == frame_slot
            base_weight[rows] = 0.5 / float(rows.sum().item())
        estimate = self._weighted_median_1d(log_ratio, base_weight)
        robust_weight = base_weight
        mad = log_ratio.new_tensor(0.0)
        for _ in range(5):
            residual = log_ratio - estimate
            centered = residual - self._weighted_median_1d(
                residual, base_weight
            )
            mad = self._weighted_median_1d(centered.abs(), base_weight)
            sigma = (1.4826 * mad).clamp_min(1.0e-6)
            delta = (1.345 * sigma).clamp_min(1.0e-4)
            huber = torch.minimum(
                torch.ones_like(residual),
                delta / residual.abs().clamp_min(1.0e-8),
            )
            robust_weight = base_weight * huber
            estimate = (
                robust_weight * log_ratio
            ).sum() / robust_weight.sum().clamp_min(1.0e-8)
        scale = float(estimate.exp().detach().cpu())
        relative_error = (log_ratio - estimate).exp().sub(1.0).abs()
        inliers = relative_error <= self.rendered_alignment_max_median_relative_error
        inlier_ratio = float(
            (base_weight * inliers.to(base_weight)).sum().detach().cpu()
        )
        median_error = sum(
            float(relative_error[frame_index == slot].median().detach().cpu())
            for slot in range(2)
        ) / 2.0
        per_frame_inlier = [
            float(inliers[frame_index == slot].float().mean().detach().cpu())
            for slot in range(2)
        ]
        accepted = (
            math.isfinite(scale)
            and 1.0 / self.rendered_alignment_max_scale_change
            <= scale
            <= self.rendered_alignment_max_scale_change
            and inlier_ratio >= self.rendered_alignment_min_inlier_ratio
            and median_error
            <= self.rendered_alignment_max_median_relative_error
        )
        diagnostics = {
            "mode": "canonical_ba_overlap_depth_only",
            "source_window_id": int(previous.window_id),
            "target_window_id": int(current.window_id),
            "overlap_frame_ids": frame_ids,
            "per_frame_valid_points": frame_counts,
            "min_confidence": self.rendered_alignment_min_confidence,
            "per_frame_confidence_support_pixels": frame_confidence_support,
            "per_frame_confidence_rejected_pixels": frame_confidence_rejected,
            "valid_points": int(log_ratio.numel()),
            "per_frame_inlier_ratio": per_frame_inlier,
            "inlier_ratio": inlier_ratio,
            "median_relative_error": median_error,
            "log_ratio_mad": float(mad.detach().cpu()),
            "chunk_scale_normalization": scale,
            "c": scale,
            "irls_iterations": 5,
            "frame_weight": 0.5,
            "fibonacci_seeds": seeds,
            "global_render_used_for_scale": False,
            "post_refiner_scale_recheck": False,
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else "ba_overlap_scale_gate_rejected",
        }
        return (scale if accepted else None), diagnostics

    def _known_overlap_global_pose(
        self,
        previous: LocalGaussianWindowPacket,
        frame_id: int,
        previous_owner_transform: torch.Tensor,
    ) -> torch.Tensor:
        """Return the already-admitted global SE(3) pose of an overlap frame."""

        frame = int(frame_id)
        if self.chunk_first_stride_graph and frame in self.frame_pose_owner_node:
            owner_node = int(self.frame_pose_owner_node[frame])
            local_pose = self.frame_local_pose_in_owner[frame]
            if owner_node not in self.graph.nodes:
                raise RuntimeError(
                    f"Canonical pose owner node {owner_node} for frame {frame} is missing"
                )
            return apply_sim3_to_c2w(
                self.graph.transform(owner_node).to(local_pose), local_pose
            )
        if frame in self.graph.nodes:
            node = self.graph.transform(frame)
            _, rotation, translation = sim3_components(node)
            pose = torch.eye(4, device=node.device, dtype=node.dtype)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            return pose
        index = previous.frame_index(frame)
        return apply_sim3_to_c2w(
            previous_owner_transform.to(previous.local_poses_c2w),
            previous.local_poses_c2w[index],
        )

    @staticmethod
    def _bridge_owner_from_first_pose(
        scale: float,
        local_first_pose: torch.Tensor,
        known_global_first_pose: torch.Tensor,
    ) -> torch.Tensor:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Known-pose bridge scale must be positive and finite")
        local = local_first_pose.to(known_global_first_pose)
        rotation = (
            known_global_first_pose[:3, :3]
            @ local[:3, :3].transpose(0, 1)
        )
        translation = (
            known_global_first_pose[:3, 3]
            - value * (rotation @ local[:3, 3])
        )
        return sim3_from_components(value, rotation, translation)

    @staticmethod
    def _canonicalize_packet_from_two_known_poses(
        packet: LocalGaussianWindowPacket,
        owner_transform: torch.Tensor,
        known_global_poses: tuple[torch.Tensor, torch.Tensor],
    ) -> LocalGaussianWindowPacket:
        """Pin both overlap poses and preserve the second-to-tail local motion."""

        if len(packet.frame_ids) < 2:
            raise ValueError("Known-pose bridge requires at least two packet frames")
        original = packet.local_poses_c2w.detach().clone()
        corrected = original.clone()
        corrected[0] = rebase_c2w_to_sim3_anchor(
            owner_transform.to(known_global_poses[0]),
            known_global_poses[0],
        ).to(corrected)
        corrected[0] = torch.eye(
            4, device=corrected.device, dtype=corrected.dtype
        )
        corrected[1] = rebase_c2w_to_sim3_anchor(
            owner_transform.to(known_global_poses[1]),
            known_global_poses[1],
        ).to(corrected)
        tail_from_second = invert_c2w(original[1])
        for index in range(2, len(packet.frame_ids)):
            corrected[index] = corrected[1] @ tail_from_second @ original[index]
        observation = packet.observation.with_geometry(
            poses_c2w=corrected.unsqueeze(0).to(packet.observation.poses_c2w)
        )
        metadata = dict(packet.metadata)
        metadata["known_pose_bridge_canonicalized"] = True
        metadata["known_pose_bridge_tail_reference_index"] = 1
        # Provisional anchors use the pre-canonical trajectory and are never
        # allowed to reach fusion.  The deferred callback re-voxelizes from
        # ``observation`` before running the learned Refiner.
        return replace(
            packet,
            local_poses_c2w=corrected,
            observation=observation,
            anchor_observation=None,
            metadata=metadata,
        )

    def _collect_known_pose_bridge_frame(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        frame_id: int,
        *,
        previous_owner_transform: torch.Tensor,
        exclude_current_target_only: bool,
    ) -> KnownPoseBridgeFrame:
        if previous.anchor_observation is None or current.anchor_observation is None:
            raise RuntimeError(
                "Known-pose bridge requires previous final and current whole-chunk anchors"
            )
        previous_index = previous.frame_index(frame_id)
        current_index = current.frame_index(frame_id)
        known_global_pose = self._known_overlap_global_pose(
            previous,
            frame_id,
            previous_owner_transform,
        )
        global_render = self._render_global_pose_frame(
            known_global_pose,
            image_size=current.anchor_observation.image_size,
        )
        previous_render = self._render_refined_anchor_frame(previous, frame_id)
        current_render = self._render_refined_anchor_frame(
            current,
            frame_id,
            exclude_target_only_anchors=exclude_current_target_only,
        )
        device_depth = current_render.depth
        global_depth = global_render.depth.to(device_depth)
        previous_depth = previous_render.depth.to(device_depth)
        current_depth = current_render.depth
        global_alpha = global_render.alpha.to(device_depth)
        previous_alpha = previous_render.alpha.to(device_depth)
        current_alpha = current_render.alpha
        previous_semantic = (
            previous.finite_gaussian_mask[0, previous_index]
            & previous.static_mask[0, previous_index]
            & previous.geometry_consistency[0, previous_index]
            & ~previous.sky_mask[0, previous_index]
        ).to(device_depth.device)
        current_semantic = (
            current.finite_gaussian_mask[0, current_index]
            & current.static_mask[0, current_index]
            & current.geometry_consistency[0, current_index]
            & ~current.sky_mask[0, current_index]
        ).to(device_depth.device)

        def render_valid(
            depth: torch.Tensor,
            alpha: torch.Tensor,
            semantic: torch.Tensor,
        ) -> torch.Tensor:
            return (
                semantic
                & torch.isfinite(depth)
                & (depth > 0.0)
                & torch.isfinite(alpha)
                & (alpha >= self.rendered_alignment_alpha_threshold)
            )

        global_valid_base = render_valid(
            global_depth, global_alpha, previous_semantic
        )
        previous_valid = render_valid(
            previous_depth, previous_alpha, previous_semantic
        )
        current_valid = render_valid(
            current_depth, current_alpha, current_semantic
        )
        previous_scale, _, _ = sim3_components(previous_owner_transform)
        previous_world_depth = previous_depth * previous_scale.to(previous_depth)
        consistency_error = (
            (previous_world_depth - global_depth).abs()
            / global_depth.abs().clamp_min(1.0e-6)
        )
        consistency_support = global_valid_base & previous_valid
        consistency = (
            consistency_support
            & (
                consistency_error
                <= self.rendered_alignment_global_map_consistency_error
            )
        )
        support_count = int(consistency_support.sum().item())
        consistency_ratio = (
            0.0
            if support_count == 0
            else float(consistency.sum().float().item() / support_count)
        )
        global_valid = global_valid_base & consistency
        seed = (
            self.fibonacci_seed
            + 1_000_003 * int(previous.window_id)
            + 10_007 * int(current.window_id)
            + 101 * int(frame_id)
        ) & 0x7FFFFFFF
        samples = sample_joint_valid_fibonacci_uv(
            global_depth,
            current_depth,
            count=self.rendered_alignment_max_points_per_frame,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            source_valid=global_valid,
            target_valid=current_valid,
            source_sky_probability=previous.sky_prob[
                0, previous_index
            ].detach().to(device_depth),
            target_sky_probability=current.sky_prob[
                0, current_index
            ].detach().to(device_depth),
            sky_threshold=self.sky_threshold,
            seed=seed,
        )
        count = int(samples.source_depth.numel())
        if count < self.rendered_alignment_min_points_per_frame:
            raise RuntimeError(
                f"Frame {frame_id} has only {count} valid known-pose bridge points; "
                f"{self.rendered_alignment_min_points_per_frame} required"
            )
        row = torch.arange(count, device=samples.bearing.device, dtype=torch.long)
        hashed = (row * 1_103_515_245 + int(seed) * 12_345) & 0x7FFFFFFF
        holdout = (hashed % self.rendered_alignment_holdout_stride) == 0
        if not bool(holdout.any()):
            holdout[-1] = True
        if bool(holdout.all()):
            holdout[0] = False
        previous_local_pose = rebase_c2w_to_sim3_anchor(
            previous_owner_transform.to(known_global_pose),
            known_global_pose,
        ).to(samples.bearing)
        return KnownPoseBridgeFrame(
            frame_id=int(frame_id),
            previous_index=previous_index,
            current_index=current_index,
            bearing=samples.bearing.detach(),
            uv=samples.uv.detach(),
            global_depth=samples.source_depth.detach(),
            current_depth=samples.target_depth.detach(),
            source_depth_previous_owner=(
                samples.source_depth / previous_scale.to(samples.source_depth)
            ).detach(),
            previous_local_pose=previous_local_pose.detach(),
            current_local_pose=current.local_poses_c2w[current_index]
            .to(samples.bearing)
            .detach(),
            known_global_pose=known_global_pose.to(samples.bearing).detach(),
            holdout_mask=holdout,
            inlier_mask=torch.ones(count, device=samples.bearing.device, dtype=torch.bool),
            global_render=global_render,
            previous_render=previous_render,
            current_render=current_render,
            global_valid_image=global_valid.detach(),
            current_valid_image=current_valid.detach(),
            global_previous_consistency_image=consistency.detach(),
            sky_union_image=(
                previous.sky_mask[0, previous_index]
                | current.sky_mask[0, current_index].to(previous.sky_mask.device)
            ).detach().to(device_depth.device),
            global_previous_consistency_ratio=consistency_ratio,
        )

    def _collect_known_pose_bridge_frames(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        previous_owner_transform: torch.Tensor,
        *,
        exclude_current_target_only: bool,
    ) -> list[KnownPoseBridgeFrame]:
        overlap = self._overlap_frame_ids(previous, current)
        if len(overlap) != 2:
            raise RuntimeError("Known-pose bridge requires exactly two overlap frames")
        frames = [
            self._collect_known_pose_bridge_frame(
                previous,
                current,
                frame_id,
                previous_owner_transform=previous_owner_transform,
                exclude_current_target_only=exclude_current_target_only,
            )
            for frame_id in overlap
        ]
        ratios = [
            float(frame.global_previous_consistency_ratio) for frame in frames
        ]
        balanced_ratio = sum(ratios) / float(len(ratios))
        if (
            balanced_ratio
            < self.rendered_alignment_global_map_min_consistency_ratio
        ):
            formatted = ", ".join(f"{value:.3f}" for value in ratios)
            raise RuntimeError(
                "Two-frame balanced global/previous map consistency ratio "
                f"{balanced_ratio:.3f} is below "
                f"{self.rendered_alignment_global_map_min_consistency_ratio:.3f} "
                f"(per-frame: [{formatted}])"
            )
        return frames

    @staticmethod
    def _bridge_balanced_weights(
        frames: list[KnownPoseBridgeFrame],
        masks: list[torch.Tensor],
        robust_parts: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(frames) != len(masks) or not frames:
            raise ValueError("Bridge weights require one mask per frame")
        pieces: list[torch.Tensor] = []
        frame_weight = 1.0 / float(len(frames))
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            count = int(mask.sum().item())
            if count <= 0:
                raise ValueError(
                    f"Bridge frame {frame.frame_id} has no selected samples"
                )
            values = (
                torch.ones(
                    count,
                    device=frame.current_depth.device,
                    dtype=frame.current_depth.dtype,
                )
                if robust_parts is None
                else robust_parts[index].to(frame.current_depth)[mask]
            ).clamp_min(0.0)
            if not bool((values > 0.0).any()):
                raise ValueError(
                    f"Bridge frame {frame.frame_id} has no positive weights"
                )
            pieces.append(
                frame_weight * values / values.sum().clamp_min(1.0e-8)
            )
        return torch.cat(pieces, dim=0)

    def _estimate_known_pose_bridge_scale(
        self,
        frames: list[KnownPoseBridgeFrame],
        previous_owner_transform: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[float | None, list[torch.Tensor], dict[str, Any]]:
        if len(frames) != 2:
            raise ValueError("Known-pose bridge scale requires two frames")
        train_masks = [~frame.holdout_mask for frame in frames]
        previous_scale = float(
            sim3_components(previous_owner_transform)[0].detach().cpu()
        )
        per_frame_depth_scales = [
            float(
                (
                    frame.global_depth[mask].clamp_min(1.0e-8).log()
                    - frame.current_depth[mask].clamp_min(1.0e-8).log()
                )
                .median()
                .exp()
                .detach()
                .cpu()
            )
            for frame, mask in zip(frames, train_masks)
        ]
        diagnostics: dict[str, Any] = {
            "bridge_scale_mode": str(mode),
            "bridge_previous_owner_scale": previous_scale,
            "bridge_per_frame_depth_scale": per_frame_depth_scales,
            "bridge_global_previous_consistency_ratio": [
                frame.global_previous_consistency_ratio for frame in frames
            ],
            "bridge_global_previous_consistency_balanced_ratio": sum(
                float(frame.global_previous_consistency_ratio)
                for frame in frames
            )
            / float(len(frames)),
        }
        if mode == "depth":
            log_ratio = torch.cat(
                [
                    frame.global_depth[mask].clamp_min(1.0e-8).log()
                    - frame.current_depth[mask].clamp_min(1.0e-8).log()
                    for frame, mask in zip(frames, train_masks)
                ],
                dim=0,
            )
            weights = self._bridge_balanced_weights(frames, train_masks)
            estimate = (weights * log_ratio).sum() / weights.sum().clamp_min(
                1.0e-8
            )
            for _ in range(self.rendered_alignment_irls_iterations):
                residual = log_ratio - estimate
                centered = residual - residual.median()
                mad = centered.abs().median()
                delta = (1.345 * 1.4826 * mad).clamp_min(1.0e-4)
                huber = torch.minimum(
                    torch.ones_like(residual),
                    delta / residual.abs().clamp_min(1.0e-8),
                )
                robust_parts: list[torch.Tensor] = []
                offset = 0
                for frame, mask in zip(frames, train_masks):
                    count = int(mask.sum().item())
                    full = torch.zeros_like(frame.current_depth)
                    full[mask] = huber[offset : offset + count]
                    robust_parts.append(full)
                    offset += count
                weights = self._bridge_balanced_weights(
                    frames, train_masks, robust_parts
                )
                estimate = (
                    weights * log_ratio
                ).sum() / weights.sum().clamp_min(1.0e-8)
            absolute_scale = float(estimate.exp().detach().cpu())
            frame_scale_disagreement = (
                max(per_frame_depth_scales) / max(min(per_frame_depth_scales), 1.0e-8)
                - 1.0
            )
            diagnostics["bridge_frame_scale_disagreement"] = float(
                frame_scale_disagreement
            )
        elif mode == "pose_baseline":
            global_baseline = float(
                torch.linalg.norm(
                    frames[1].known_global_pose[:3, 3]
                    - frames[0].known_global_pose[:3, 3]
                )
                .detach()
                .cpu()
            )
            local_baseline = float(
                torch.linalg.norm(
                    frames[1].current_local_pose[:3, 3]
                    - frames[0].current_local_pose[:3, 3]
                )
                .detach()
                .cpu()
            )
            diagnostics.update(
                {
                    "bridge_global_pose_baseline": global_baseline,
                    "bridge_local_pose_baseline": local_baseline,
                }
            )
            if (
                global_baseline < self.rendered_alignment_pose_baseline_min
                or local_baseline < self.rendered_alignment_pose_baseline_min
            ):
                diagnostics.update(
                    {
                        "accepted": False,
                        "reason": "pose_baseline_degenerate",
                    }
                )
                return None, [
                    torch.zeros_like(frame.current_depth, dtype=torch.bool)
                    for frame in frames
                ], diagnostics
            absolute_scale = global_baseline / local_baseline
        else:
            raise ValueError(f"Unsupported known-pose bridge scale mode {mode!r}")

        relative_owner_scale = absolute_scale / max(previous_scale, 1.0e-8)
        strict_error_threshold = self.rendered_alignment_max_median_relative_error
        consensus_error_threshold = max(
            0.20,
            min(0.30, 1.5 * strict_error_threshold),
        )
        inlier_masks: list[torch.Tensor] = []
        consensus_inlier_masks: list[torch.Tensor] = []
        train_errors: list[torch.Tensor] = []
        holdout_errors: list[torch.Tensor] = []
        per_frame_ratios: list[float] = []
        per_frame_medians: list[float] = []
        for frame in frames:
            error = (
                (absolute_scale * frame.current_depth - frame.global_depth).abs()
                / frame.global_depth.abs().clamp_min(1.0e-6)
            )
            inliers = (
                error <= strict_error_threshold
            )
            inlier_masks.append(inliers)
            consensus_inlier_masks.append(error <= consensus_error_threshold)
            train_errors.append(error[~frame.holdout_mask])
            holdout_errors.append(error[frame.holdout_mask])
            per_frame_ratios.append(float(inliers.float().mean().detach().cpu()))
            per_frame_medians.append(float(error.median().detach().cpu()))
        train_ratio = sum(
            float(
                (error <= self.rendered_alignment_max_median_relative_error)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in train_errors
        ) / float(len(train_errors))
        holdout_ratio = sum(
            float(
                (error <= self.rendered_alignment_max_median_relative_error)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in holdout_errors
        ) / float(len(holdout_errors))
        train_median = sum(
            float(error.median().detach().cpu()) for error in train_errors
        ) / float(len(train_errors))
        holdout_median = sum(
            float(error.median().detach().cpu()) for error in holdout_errors
        ) / float(len(holdout_errors))
        scale_ok = (
            math.isfinite(absolute_scale)
            and 1.0 / self.rendered_alignment_max_scale_change
            <= relative_owner_scale
            <= self.rendered_alignment_max_scale_change
        )
        depth_gate = (
            train_ratio >= self.rendered_alignment_min_inlier_ratio
            and holdout_ratio >= self.rendered_alignment_min_inlier_ratio
            and train_median
            <= self.rendered_alignment_max_median_relative_error
            and holdout_median
            <= self.rendered_alignment_max_median_relative_error
        )
        consensus_train_ratio = sum(
            float(
                (error <= consensus_error_threshold)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in train_errors
        ) / float(len(train_errors))
        consensus_holdout_ratio = sum(
            float(
                (error <= consensus_error_threshold)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in holdout_errors
        ) / float(len(holdout_errors))
        depth_consensus_gate = bool(
            mode == "depth"
            and frame_scale_disagreement <= 0.10
            and 0.80 <= relative_owner_scale <= 1.25
            and train_median <= consensus_error_threshold
            and holdout_median <= consensus_error_threshold
            and consensus_train_ratio >= self.rendered_alignment_min_inlier_ratio
            and consensus_holdout_ratio >= self.rendered_alignment_min_inlier_ratio
        )
        # Pose-baseline is an intentional ablation: rendered depth remains a
        # diagnostic/factor selector but cannot change or veto its scale.
        accepted = scale_ok and (
            (depth_gate or depth_consensus_gate) if mode == "depth" else True
        )
        reason = (
            "accepted"
            if accepted
            else ("scale_gate_rejected" if not scale_ok else "depth_gate_rejected")
        )
        diagnostics.update(
            {
                "absolute_scale": absolute_scale,
                "s_absolute": absolute_scale,
                "measurement_scale": relative_owner_scale,
                "bridge_relative_owner_scale": relative_owner_scale,
                "bridge_train_inlier_ratio": train_ratio,
                "bridge_holdout_inlier_ratio": holdout_ratio,
                "bridge_train_median_relative_error": train_median,
                "bridge_holdout_median_relative_error": holdout_median,
                "bridge_per_frame_inlier_ratio": per_frame_ratios,
                "bridge_per_frame_median_relative_error": per_frame_medians,
                "bridge_depth_gate_passed": bool(depth_gate),
                "bridge_depth_consensus_gate_passed": depth_consensus_gate,
                "bridge_depth_consensus_fallback_used": bool(
                    depth_consensus_gate and not depth_gate
                ),
                "bridge_depth_consensus_error_threshold": float(
                    consensus_error_threshold
                ),
                "bridge_depth_consensus_train_inlier_ratio": (
                    consensus_train_ratio
                ),
                "bridge_depth_consensus_holdout_inlier_ratio": (
                    consensus_holdout_ratio
                ),
                "bridge_factor_relative_error_threshold": float(
                    consensus_error_threshold
                    if depth_consensus_gate and not depth_gate
                    else strict_error_threshold
                ),
                "accepted": bool(accepted),
                "reason": reason,
            }
        )
        return (
            absolute_scale if accepted else None,
            (
                consensus_inlier_masks
                if depth_consensus_gate and not depth_gate
                else inlier_masks
            ),
            diagnostics,
        )

    def _solve_known_pose_bridge(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        previous_owner_transform: torch.Tensor,
    ) -> KnownPoseBridgeSolution:
        started = time.perf_counter()
        frames = self._collect_known_pose_bridge_frames(
            previous,
            current,
            previous_owner_transform,
            exclude_current_target_only=True,
        )
        mode = (
            "depth"
            if self.two_frame_bridge_depth_scale_enabled
            else "pose_baseline"
        )
        scale, inlier_masks, diagnostics = self._estimate_known_pose_bridge_scale(
            frames,
            previous_owner_transform,
            mode=mode,
        )
        if scale is None:
            diagnostics["alignment_seconds"] = float(
                time.perf_counter() - started
            )
            diagnostics.update(
                {
                    "mode": self.rendered_overlap_alignment_mode,
                    "alignment_method": f"known_pose_bridge_{mode}",
                    "source_window_id": int(previous.window_id),
                    "target_window_id": int(current.window_id),
                    "overlap_frame_ids": [frame.frame_id for frame in frames],
                    "per_frame_valid_points": [
                        int(frame.current_depth.numel()) for frame in frames
                    ],
                    "valid_points": sum(
                        int(frame.current_depth.numel()) for frame in frames
                    ),
                    "accepted": False,
                }
            )
            candidate_scale = diagnostics.get("absolute_scale")
            if isinstance(candidate_scale, (int, float)) and math.isfinite(
                float(candidate_scale)
            ):
                self._set_known_pose_bridge_diagnostic(
                    frames,
                    float(candidate_scale),
                    inlier_masks,
                )
            self._last_overlap_alignment_failure = copy.deepcopy(diagnostics)
            raise RuntimeError(
                "Known-pose bridge scale failed: "
                f"{diagnostics.get('reason', 'unknown')}"
            )
        known_global_poses = (
            frames[0].known_global_pose,
            frames[1].known_global_pose,
        )
        owner = self._bridge_owner_from_first_pose(
            scale,
            current.local_poses_c2w[0],
            known_global_poses[0],
        )
        corrected = self._canonicalize_packet_from_two_known_poses(
            current,
            owner,
            known_global_poses,
        )
        relative_measurement = (
            sim3_inverse(previous_owner_transform.to(owner)) @ owner
        )
        predicted = corrected.global_poses(owner.to(corrected.local_poses_c2w))
        rotation_errors = [
            self._rotation_error_deg(expected[:3, :3], actual[:3, :3])
            for expected, actual in zip(known_global_poses, predicted[:2])
        ]
        center_errors = [
            float(
                torch.linalg.norm(expected[:3, 3] - actual[:3, 3])
                .detach()
                .cpu()
            )
            for expected, actual in zip(known_global_poses, predicted[:2])
        ]
        diagnostics.update(
            {
                "mode": self.rendered_overlap_alignment_mode,
                "alignment_method": f"known_pose_bridge_{mode}",
                "source_window_id": int(previous.window_id),
                "target_window_id": int(current.window_id),
                "overlap_frame_ids": [frame.frame_id for frame in frames],
                "per_frame_valid_points": [
                    int(frame.current_depth.numel()) for frame in frames
                ],
                "valid_points": sum(
                    int(frame.current_depth.numel()) for frame in frames
                ),
                "bridge_shared_rotation_errors_deg": rotation_errors,
                "bridge_shared_center_errors": center_errors,
                "chunk_scale_normalization": 1.0,
                "canonical_rotation_mismatch_deg": max(rotation_errors),
                "canonical_translation_mismatch": max(center_errors),
                "accepted": True,
                "reason": "accepted",
                "alignment_seconds": float(time.perf_counter() - started),
                "render_seconds": sum(
                    frame.global_render.render_seconds
                    + frame.previous_render.render_seconds
                    + frame.current_render.render_seconds
                    for frame in frames
                ),
            }
        )
        return KnownPoseBridgeSolution(
            packet=corrected,
            owner_transform=owner.detach(),
            relative_measurement=relative_measurement.detach(),
            diagnostics=diagnostics,
        )

    def _set_known_pose_bridge_diagnostic(
        self,
        frames: list[KnownPoseBridgeFrame],
        absolute_scale: float,
        inlier_masks: list[torch.Tensor],
    ) -> None:
        panels: dict[str, list[torch.Tensor]] = {
            "local_depth": [],
            "aligned_local_depth": [],
            "global_depth": [],
            "relative_error": [],
            "local_alpha": [],
            "global_alpha": [],
            "sky_mask": [],
            "valid_mask": [],
            "inlier_mask": [],
        }
        frame_ids: list[int] = []
        for frame, inliers in zip(frames, inlier_masks):
            local = frame.current_render.depth
            global_depth = frame.global_render.depth.to(local)
            aligned = local * float(absolute_scale)
            relative_error = (
                (aligned - global_depth).abs()
                / global_depth.abs().clamp_min(1.0e-6)
            )
            inlier_image = torch.zeros_like(local, dtype=torch.bool)
            if int(frame.uv.numel()) > 0 and bool(inliers.any()):
                uv = frame.uv[inliers]
                columns = torch.floor(uv[:, 0]).long().clamp(
                    0, int(local.shape[-1]) - 1
                )
                rows = torch.floor(uv[:, 1]).long().clamp(
                    0, int(local.shape[-2]) - 1
                )
                inlier_image[0, rows, columns] = True
            panels["local_depth"].append(local.detach().cpu().float())
            panels["aligned_local_depth"].append(
                aligned.detach().cpu().float()
            )
            panels["global_depth"].append(
                global_depth.detach().cpu().float()
            )
            panels["relative_error"].append(
                relative_error.detach().cpu().float()
            )
            panels["local_alpha"].append(
                frame.current_render.alpha.detach().cpu().float()
            )
            panels["global_alpha"].append(
                frame.global_render.alpha.detach().cpu().float()
            )
            panels["sky_mask"].append(frame.sky_union_image.detach().cpu())
            panels["valid_mask"].append(
                (
                    frame.global_valid_image
                    & frame.current_valid_image
                )
                .detach()
                .cpu()
            )
            panels["inlier_mask"].append(inlier_image.detach().cpu())
            frame_ids.append(frame.frame_id)
        self._last_rendered_overlap_diagnostic = {
            name: torch.stack(values, dim=0)
            for name, values in panels.items()
        }
        self._last_rendered_overlap_diagnostic["frame_ids"] = torch.tensor(
            frame_ids, dtype=torch.long
        )

    def _known_pose_bridge_constraints(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        *,
        previous_anchor_node: int,
        current_anchor_node: int,
        previous_owner_transform: torch.Tensor,
        current_owner_transform: torch.Tensor,
        alignment_diagnostics: dict[str, Any],
    ) -> tuple[
        Sim3GraphEdge,
        tuple[DenseSphericalFactorBlock, ...],
        tuple[CoincidentPanoramaFactor, ...],
        dict[str, Any],
    ]:
        frames = self._collect_known_pose_bridge_frames(
            previous,
            current,
            previous_owner_transform,
            exclude_current_target_only=False,
        )
        absolute_scale = float(
            sim3_components(current_owner_transform)[0].detach().cpu()
        )
        factor_error_threshold = float(
            alignment_diagnostics.get(
                "bridge_factor_relative_error_threshold",
                self.rendered_alignment_max_median_relative_error,
            )
        )
        inlier_masks: list[torch.Tensor] = []
        per_frame_ratio: list[float] = []
        per_frame_median: list[float] = []
        for frame in frames:
            error = (
                (absolute_scale * frame.current_depth - frame.global_depth).abs()
                / frame.global_depth.abs().clamp_min(1.0e-6)
            )
            mask = error <= factor_error_threshold
            inlier_masks.append(mask)
            per_frame_ratio.append(float(mask.float().mean().detach().cpu()))
            per_frame_median.append(float(error.median().detach().cpu()))
        self._set_known_pose_bridge_diagnostic(
            frames, absolute_scale, inlier_masks
        )
        measurement = (
            sim3_inverse(previous_owner_transform.to(current_owner_transform))
            @ current_owner_transform
        )
        total_inliers = sum(int(mask.sum().item()) for mask in inlier_masks)
        confidence = min(
            8.0,
            max(1.0, math.sqrt(float(max(1, total_inliers)) / 64.0)),
        )
        information = measurement.new_tensor(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.75]
        ) * confidence
        diagnostics = dict(alignment_diagnostics)
        diagnostics.update(
            {
                "post_refiner_per_frame_inlier_ratio": per_frame_ratio,
                "post_refiner_per_frame_median_relative_error": per_frame_median,
                "post_refiner_valid_points": sum(
                    int(frame.current_depth.numel()) for frame in frames
                ),
                "post_refiner_factor_relative_error_threshold": (
                    factor_error_threshold
                ),
                "post_refiner_depth_gate_passed": bool(
                    min(per_frame_ratio, default=0.0)
                    >= self.rendered_alignment_min_inlier_ratio
                    and max(per_frame_median, default=float("inf"))
                    <= self.rendered_alignment_max_median_relative_error
                ),
            }
        )
        edge = Sim3GraphEdge(
            source=int(previous_anchor_node),
            target=int(current_anchor_node),
            measurement_target_to_source=measurement.detach(),
            information_diag=information.detach(),
            edge_type="overlap_known_pose_bridge_sim3",
            metadata=dict(diagnostics),
        )
        dense_factors: list[DenseSphericalFactorBlock] = []
        pose_factors: list[CoincidentPanoramaFactor] = []
        for frame, strict_inliers in zip(frames, inlier_masks):
            keep = strict_inliers
            relaxed = False
            if int(keep.sum().item()) < self.min_dense_factors:
                # The owner measurement and the two coincident-pose factors
                # already carry the accepted bridge. Refiner surface changes
                # must not disconnect the graph merely because fewer than the
                # preferred number of rendered-depth samples remain. Keep the
                # geometrically valid samples and let the factor's robust
                # residual downweight disagreement.
                keep = torch.ones_like(strict_inliers)
                relaxed = True
            frame_diagnostics = {
                "source_window_id": int(previous.window_id),
                "target_window_id": int(current.window_id),
                "source_frame_id": int(frame.frame_id),
                "target_frame_id": int(frame.frame_id),
                "overlap_frame_id": int(frame.frame_id),
                "num_matches": int(keep.sum().item()),
                "alignment_method": diagnostics.get("alignment_method"),
                "weight_mode": "equal_solid_angle_per_frame",
                "pose_baseline_relaxed_depth_selection": relaxed,
                "post_refiner_relaxed_depth_selection": relaxed,
            }
            dense_factors.append(
                DenseSphericalFactorBlock(
                    source=int(previous_anchor_node),
                    target=int(current_anchor_node),
                    source_local_pose=frame.previous_local_pose.detach(),
                    target_local_pose=frame.current_local_pose.detach(),
                    source_bearing=frame.bearing[keep].detach(),
                    target_bearing=frame.bearing[keep].detach(),
                    source_depth=frame.source_depth_previous_owner[keep].detach(),
                    target_depth=frame.current_depth[keep].detach(),
                    factor_weight=torch.ones(
                        int(keep.sum().item()),
                        device=frame.bearing.device,
                        dtype=frame.bearing.dtype,
                    ),
                    depth_factor_weight=self.depth_factor_weight,
                    s2_huber_delta_deg=self.s2_huber_delta_deg,
                    use_depth=True,
                    edge_type="overlap_known_pose_bridge_dense",
                    **self._dense_factor_information_options(),
                    metadata=frame_diagnostics,
                )
            )
            pose_factors.append(
                CoincidentPanoramaFactor(
                    source=int(previous_anchor_node),
                    target=int(current_anchor_node),
                    source_local_pose=frame.previous_local_pose.detach(),
                    target_local_pose=frame.current_local_pose.detach(),
                    measured_source_to_target_rotation=torch.eye(
                        3,
                        device=frame.bearing.device,
                        dtype=frame.bearing.dtype,
                    ),
                    center_weight=confidence,
                    rotation_weight=confidence,
                    edge_type="overlap_known_pose_bridge_pose_consistency",
                    metadata=frame_diagnostics,
                )
            )
        if len(dense_factors) != 2 or len(pose_factors) != 2:
            raise RuntimeError("Known-pose bridge must create two overlap factors")
        return edge, tuple(dense_factors), tuple(pose_factors), diagnostics

    def _collect_overlap_frame_geometry(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        frame_id: int,
        *,
        use_rendered_anchors: bool,
    ) -> OverlapFrameGeometry:
        previous_index = previous.frame_index(frame_id)
        current_index = current.frame_index(frame_id)
        previous_render = None
        current_render = None
        if use_rendered_anchors:
            previous_render = self._render_refined_anchor_frame(previous, frame_id)
            current_render = self._render_refined_anchor_frame(current, frame_id)
            previous_depth = previous_render.depth
            current_depth = current_render.depth.to(previous_depth)
            previous_alpha = previous_render.alpha
            current_alpha = current_render.alpha.to(previous_alpha)
        else:
            previous_depth = previous.observation.refined_depth[
                0, previous_index
            ].detach()
            current_depth = current.observation.refined_depth[
                0, current_index
            ].detach().to(previous_depth)
            previous_alpha = torch.ones_like(previous_depth)
            current_alpha = torch.ones_like(previous_depth)

        previous_semantic_valid = (
            previous.finite_gaussian_mask[0, previous_index]
            & previous.static_mask[0, previous_index]
            & previous.geometry_consistency[0, previous_index]
            & ~previous.sky_mask[0, previous_index]
        ).to(previous_depth.device)
        current_semantic_valid = (
            current.finite_gaussian_mask[0, current_index]
            & current.static_mask[0, current_index]
            & current.geometry_consistency[0, current_index]
            & ~current.sky_mask[0, current_index]
        ).to(previous_depth.device)
        previous_valid = (
            previous_semantic_valid
            & torch.isfinite(previous_depth)
            & (previous_depth > 0.0)
            & torch.isfinite(previous_alpha)
            & (previous_alpha >= self.rendered_alignment_alpha_threshold)
        )
        current_valid = (
            current_semantic_valid
            & torch.isfinite(current_depth)
            & (current_depth > 0.0)
            & torch.isfinite(current_alpha)
            & (current_alpha >= self.rendered_alignment_alpha_threshold)
        )
        seed = (
            self.fibonacci_seed
            + 1_000_003 * int(previous.window_id)
            + 10_007 * int(current.window_id)
            + 101 * int(frame_id)
        ) & 0x7FFFFFFF
        samples = sample_joint_valid_fibonacci_uv(
            previous_depth,
            current_depth,
            count=self.rendered_alignment_max_points_per_frame,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            source_valid=previous_valid,
            target_valid=current_valid,
            source_sky_probability=previous.sky_prob[
                0, previous_index
            ].detach().to(previous_depth),
            target_sky_probability=current.sky_prob[
                0, current_index
            ].detach().to(previous_depth),
            sky_threshold=self.sky_threshold,
            seed=seed,
        )
        count = int(samples.source_depth.numel())
        if count < self.rendered_alignment_min_points_per_frame:
            raise RuntimeError(
                f"Frame {frame_id} has only {count} valid overlap points; "
                f"{self.rendered_alignment_min_points_per_frame} required"
            )
        previous_pose = previous.local_poses_c2w[previous_index].to(
            samples.bearing
        )
        current_pose = current.local_poses_c2w[current_index].to(samples.bearing)
        previous_camera = samples.bearing * samples.source_depth[:, None]
        current_camera = samples.bearing * samples.target_depth[:, None]
        previous_points = (
            previous_camera @ previous_pose[:3, :3].transpose(0, 1)
            + previous_pose[:3, 3]
        )
        current_points = (
            current_camera @ current_pose[:3, :3].transpose(0, 1)
            + current_pose[:3, 3]
        )
        row = torch.arange(count, device=samples.bearing.device, dtype=torch.long)
        hashed = (
            row * 1_103_515_245
            + int(seed) * 12_345
        ) & 0x7FFFFFFF
        holdout = (hashed % self.rendered_alignment_holdout_stride) == 0
        if not bool(holdout.any()):
            holdout[-1] = True
        if bool(holdout.all()):
            holdout[0] = False
        return OverlapFrameGeometry(
            frame_id=int(frame_id),
            previous_index=previous_index,
            current_index=current_index,
            bearing=samples.bearing.detach(),
            uv=samples.uv.detach(),
            previous_depth=samples.source_depth.detach(),
            current_depth=samples.target_depth.detach(),
            previous_points=previous_points.detach(),
            current_points=current_points.detach(),
            previous_pose=previous_pose.detach(),
            current_pose=current_pose.detach(),
            holdout_mask=holdout,
            previous_render=previous_render,
            current_render=current_render,
            previous_valid_image=previous_valid.detach(),
            current_valid_image=current_valid.detach(),
            sky_union_image=(
                previous.sky_mask[0, previous_index]
                | current.sky_mask[0, current_index].to(
                    previous.sky_mask.device
                )
            )
            .detach()
            .to(previous_depth.device),
        )

    @staticmethod
    def _balanced_frame_weights(
        frames: list[OverlapFrameGeometry],
        masks: list[torch.Tensor],
        *,
        point_weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(frames) != len(masks) or not frames:
            raise ValueError("Balanced overlap weights require matching frame masks")
        if point_weights is not None and len(point_weights) != len(frames):
            raise ValueError("Point weights must match the overlap frame count")
        pieces = []
        frame_weight = 1.0 / float(len(frames))
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            count = int(mask.sum().item())
            if count <= 0:
                raise ValueError(f"Overlap frame {frame.frame_id} has no selected points")
            if point_weights is None:
                selected = torch.ones(
                    count,
                    device=frame.current_points.device,
                    dtype=frame.current_points.dtype,
                )
            else:
                selected = point_weights[index][mask].to(
                    device=frame.current_points.device,
                    dtype=frame.current_points.dtype,
                )
            selected = selected.clamp_min(0.0)
            if not bool((selected > 0.0).any()):
                raise ValueError(
                    f"Overlap frame {frame.frame_id} has no positive robust weights"
                )
            pieces.append(
                frame_weight
                * selected
                / selected.sum().clamp_min(1.0e-8)
            )
        return torch.cat(pieces, dim=0)

    def _pose_prior_from_overlap(
        self,
        frames: list[OverlapFrameGeometry],
        scale: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        rotations = [
            frame.previous_pose[:3, :3]
            @ frame.current_pose[:3, :3].transpose(0, 1)
            for frame in frames
        ]
        rotation = self._average_rotations(rotations)
        translations = [
            frame.previous_pose[:3, 3]
            - float(scale) * (rotation @ frame.current_pose[:3, 3])
            for frame in frames
        ]
        translation = torch.stack(translations, dim=0).mean(dim=0)
        pair_rotation = (
            0.0
            if len(rotations) < 2
            else self._rotation_error_deg(rotations[0], rotations[1])
        )
        pair_translation = (
            0.0
            if len(translations) < 2
            else float(
                torch.linalg.norm(translations[0] - translations[1])
                .detach()
                .cpu()
            )
        )
        return (
            sim3_from_components(float(scale), rotation, translation),
            {
                "pose_pair_rotation_deg": pair_rotation,
                "pose_pair_translation": pair_translation,
                "pose_frame_scale": [float(scale) for _ in frames],
                "pose_frame_rotation_deg": [
                    self._rotation_error_deg(
                        torch.eye(
                            3,
                            device=value.device,
                            dtype=value.dtype,
                        ),
                        value,
                    )
                    for value in rotations
                ],
                "pose_frame_translation": [
                    [float(component) for component in value.detach().cpu()]
                    for value in translations
                ],
                "pose_frame_translation_norm": [
                    float(torch.linalg.norm(value).detach().cpu())
                    for value in translations
                ],
            },
        )

    def _shared_pose_errors(
        self,
        transform: torch.Tensor,
        frames: list[OverlapFrameGeometry],
    ) -> tuple[list[float], list[float]]:
        rotation_errors: list[float] = []
        center_errors: list[float] = []
        for frame in frames:
            predicted = apply_sim3_to_c2w(
                transform.to(frame.current_pose),
                frame.current_pose,
            )
            rotation_errors.append(
                self._rotation_error_deg(
                    frame.previous_pose[:3, :3],
                    predicted[:3, :3],
                )
            )
            center_errors.append(
                float(
                    torch.linalg.norm(
                        predicted[:3, 3] - frame.previous_pose[:3, 3]
                    )
                    .detach()
                    .cpu()
                )
            )
        return rotation_errors, center_errors

    def _fit_two_frame_full_sim3(
        self,
        frames: list[OverlapFrameGeometry],
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], dict[str, Any]]:
        train_masks = [~frame.holdout_mask for frame in frames]
        current_train = torch.cat(
            [
                frame.current_points[mask]
                for frame, mask in zip(frames, train_masks)
            ],
            dim=0,
        )
        previous_train = torch.cat(
            [
                frame.previous_points[mask]
                for frame, mask in zip(frames, train_masks)
            ],
            dim=0,
        )
        base_weights = self._balanced_frame_weights(frames, train_masks)
        raw_current_singular = self._weighted_point_singular_values(
            current_train, base_weights
        )
        raw_previous_singular = self._weighted_point_singular_values(
            previous_train, base_weights
        )
        raw_current_ratio = float(
            (
                raw_current_singular[-1]
                / raw_current_singular[0].clamp_min(1.0e-8)
            )
            .detach()
            .cpu()
        )
        raw_previous_ratio = float(
            (
                raw_previous_singular[-1]
                / raw_previous_singular[0].clamp_min(1.0e-8)
            )
            .detach()
            .cpu()
        )
        diagnostics: dict[str, Any] = {
            "full_sim3_raw_current_singular_values": [
                float(value) for value in raw_current_singular.detach().cpu()
            ],
            "full_sim3_raw_previous_singular_values": [
                float(value) for value in raw_previous_singular.detach().cpu()
            ],
            "full_sim3_raw_current_covariance_ratio": raw_current_ratio,
            "full_sim3_raw_previous_covariance_ratio": raw_previous_ratio,
        }
        try:
            transform = weighted_umeyama(
                current_train,
                previous_train,
                base_weights,
                allow_scale=True,
            )
            for _ in range(self.rendered_alignment_irls_iterations):
                residual = torch.linalg.norm(
                    apply_sim3(transform, current_train) - previous_train,
                    dim=-1,
                )
                median = residual.median()
                gate = max(
                    self.max_overlap_residual,
                    float((2.5 * median).detach().cpu()),
                )
                inliers = residual <= gate
                if int(inliers.sum().item()) < self.overlap_aligner.min_points:
                    break
                centered = residual[inliers] - residual[inliers].median()
                mad = centered.abs().median()
                delta = (
                    max(float(self.overlap_aligner.huber_delta), 1.0e-8)
                    if self.overlap_aligner.huber_delta is not None
                    else max(float((1.4826 * mad).detach().cpu()), 1.0e-6)
                )
                huber = torch.minimum(
                    torch.ones_like(residual),
                    residual.new_tensor(delta)
                    / residual.clamp_min(1.0e-8),
                )
                robust_parts: list[torch.Tensor] = []
                offset = 0
                for mask in train_masks:
                    count = int(mask.sum().item())
                    robust_parts.append(
                        (
                            huber[offset : offset + count]
                            * inliers[offset : offset + count].to(huber)
                        )
                    )
                    offset += count
                robust_weights = self._balanced_frame_weights(
                    frames,
                    train_masks,
                    point_weights=[
                        torch.zeros(
                            int(frame.current_points.shape[0]),
                            device=frame.current_points.device,
                            dtype=frame.current_points.dtype,
                        ).masked_scatter(mask, values)
                        for frame, mask, values in zip(
                            frames,
                            train_masks,
                            robust_parts,
                        )
                    ],
                )
                if int((robust_weights > 0.0).sum().item()) < 3:
                    break
                transform = weighted_umeyama(
                    current_train,
                    previous_train,
                    robust_weights,
                    allow_scale=True,
                )
        except (RuntimeError, ValueError) as exc:
            diagnostics.update(
                {
                    "full_sim3_accepted": False,
                    "full_sim3_reason": "umeyama_failed",
                    "full_sim3_error": repr(exc),
                }
            )
            return None, [
                torch.zeros(
                    int(frame.current_points.shape[0]),
                    device=frame.current_points.device,
                    dtype=torch.bool,
                )
                for frame in frames
            ], diagnostics

        scale, rotation, translation = sim3_components(transform)
        all_inlier_masks = []
        per_frame_ratios = []
        per_frame_residuals = []
        train_residual_parts: list[torch.Tensor] = []
        holdout_residual_parts: list[torch.Tensor] = []
        for frame in frames:
            residual = torch.linalg.norm(
                apply_sim3(transform, frame.current_points)
                - frame.previous_points,
                dim=-1,
            )
            mask = residual <= self.max_overlap_residual
            all_inlier_masks.append(mask)
            per_frame_ratios.append(float(mask.float().mean().detach().cpu()))
            per_frame_residuals.append(float(residual.median().detach().cpu()))
            train_residual_parts.append(residual[~frame.holdout_mask])
            holdout_residual_parts.append(residual[frame.holdout_mask])
        covariance_masks = [
            (~frame.holdout_mask) & inliers
            for frame, inliers in zip(frames, all_inlier_masks)
        ]
        try:
            covariance_weights = self._balanced_frame_weights(
                frames,
                covariance_masks,
            )
            current_covariance_points = torch.cat(
                [
                    frame.current_points[mask]
                    for frame, mask in zip(frames, covariance_masks)
                ],
                dim=0,
            )
            previous_covariance_points = torch.cat(
                [
                    frame.previous_points[mask]
                    for frame, mask in zip(frames, covariance_masks)
                ],
                dim=0,
            )
            current_singular = self._weighted_point_singular_values(
                current_covariance_points,
                covariance_weights,
            )
            previous_singular = self._weighted_point_singular_values(
                previous_covariance_points,
                covariance_weights,
            )
            current_ratio = float(
                (
                    current_singular[-1]
                    / current_singular[0].clamp_min(1.0e-8)
                )
                .detach()
                .cpu()
            )
            previous_ratio = float(
                (
                    previous_singular[-1]
                    / previous_singular[0].clamp_min(1.0e-8)
                )
                .detach()
                .cpu()
            )
        except ValueError:
            current_singular = current_train.new_zeros(3)
            previous_singular = previous_train.new_zeros(3)
            current_ratio = 0.0
            previous_ratio = 0.0
        diagnostics.update(
            {
                "full_sim3_current_singular_values": [
                    float(value) for value in current_singular.detach().cpu()
                ],
                "full_sim3_previous_singular_values": [
                    float(value) for value in previous_singular.detach().cpu()
                ],
                "full_sim3_current_covariance_ratio": current_ratio,
                "full_sim3_previous_covariance_ratio": previous_ratio,
            }
        )
        train_ratio = sum(
            float(
                (part <= self.max_overlap_residual)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for part in train_residual_parts
        ) / float(len(train_residual_parts))
        train_mean = sum(
            float(
                (
                    part[part <= self.max_overlap_residual].mean()
                    if bool((part <= self.max_overlap_residual).any())
                    else part.mean()
                )
                .detach()
                .cpu()
            )
            for part in train_residual_parts
        ) / float(len(train_residual_parts))
        holdout_ratio = sum(
            float(
                (part <= self.max_overlap_residual)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for part in holdout_residual_parts
        ) / float(len(holdout_residual_parts))
        holdout_median = sum(
            float(part.median().detach().cpu())
            for part in holdout_residual_parts
        ) / float(len(holdout_residual_parts))
        scale_value = float(scale.detach().cpu())
        pose_prior, pose_pair = self._pose_prior_from_overlap(
            frames, scale_value
        )
        _, prior_rotation, prior_translation = sim3_components(
            pose_prior.to(transform)
        )
        rotation_correction = self._rotation_error_deg(
            prior_rotation, rotation
        )
        translation_correction = float(
            torch.linalg.norm(translation - prior_translation).detach().cpu()
        )
        shared_rotation_errors, shared_center_errors = self._shared_pose_errors(
            transform, frames
        )
        scale_ok = (
            math.isfinite(scale_value)
            and 1.0 / self.rendered_alignment_max_scale_change
            <= scale_value
            <= self.rendered_alignment_max_scale_change
        )
        covariance_ok = (
            current_ratio >= self.rendered_alignment_covariance_min_ratio
            and previous_ratio >= self.rendered_alignment_covariance_min_ratio
        )
        accepted = (
            scale_ok
            and covariance_ok
            and train_ratio >= self.rendered_alignment_min_inlier_ratio
            and train_mean <= self.max_overlap_residual
            and holdout_ratio >= self.rendered_alignment_min_inlier_ratio
            and holdout_median <= self.max_overlap_residual
            and rotation_correction
            <= self.rendered_alignment_max_rotation_correction_deg
            and translation_correction
            <= self.rendered_alignment_max_translation_correction
            and max(shared_rotation_errors, default=0.0)
            <= self.rendered_alignment_max_shared_rotation_error_deg
            and max(shared_center_errors, default=0.0)
            <= self.rendered_alignment_max_shared_center_error
        )
        if not covariance_ok:
            reason = "covariance_degenerate"
        elif not scale_ok:
            reason = "scale_gate_rejected"
        elif train_ratio < self.rendered_alignment_min_inlier_ratio:
            reason = "training_inlier_gate_rejected"
        elif train_mean > self.max_overlap_residual:
            reason = "training_residual_gate_rejected"
        elif (
            holdout_ratio < self.rendered_alignment_min_inlier_ratio
            or holdout_median > self.max_overlap_residual
        ):
            reason = "holdout_gate_rejected"
        elif (
            rotation_correction
            > self.rendered_alignment_max_rotation_correction_deg
            or translation_correction
            > self.rendered_alignment_max_translation_correction
        ):
            reason = "pose_prior_correction_gate_rejected"
        elif (
            max(shared_rotation_errors, default=0.0)
            > self.rendered_alignment_max_shared_rotation_error_deg
            or max(shared_center_errors, default=0.0)
            > self.rendered_alignment_max_shared_center_error
        ):
            reason = "shared_pose_gate_rejected"
        else:
            reason = "accepted"
        diagnostics.update(
            {
                "full_sim3_scale": scale_value,
                "full_sim3_rotation_deg": float(
                    torch.rad2deg(
                        torch.linalg.norm(sim3_log(transform)[3:6])
                    )
                    .detach()
                    .cpu()
                ),
                "full_sim3_translation_norm": float(
                    torch.linalg.norm(translation).detach().cpu()
                ),
                "full_sim3_train_inlier_ratio": train_ratio,
                "full_sim3_train_mean_residual": train_mean,
                "full_sim3_holdout_inlier_ratio": holdout_ratio,
                "full_sim3_holdout_median_residual": holdout_median,
                "full_sim3_rotation_correction_deg": rotation_correction,
                "full_sim3_translation_correction": translation_correction,
                "full_sim3_shared_rotation_errors_deg": shared_rotation_errors,
                "full_sim3_shared_center_errors": shared_center_errors,
                "full_sim3_per_frame_inlier_ratio": per_frame_ratios,
                "full_sim3_per_frame_median_residual": per_frame_residuals,
                "full_sim3_accepted": bool(accepted),
                "full_sim3_reason": reason,
                **pose_pair,
            }
        )
        return (
            transform.detach() if accepted else None,
            all_inlier_masks,
            diagnostics,
        )

    def _fit_two_frame_scale_pose_fallback(
        self,
        frames: list[OverlapFrameGeometry],
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], dict[str, Any]]:
        train_masks = [~frame.holdout_mask for frame in frames]
        log_ratios = torch.cat(
            [
                frame.previous_depth[mask].clamp_min(1.0e-8).log()
                - frame.current_depth[mask].clamp_min(1.0e-8).log()
                for frame, mask in zip(frames, train_masks)
            ],
            dim=0,
        )
        base_weights = self._balanced_frame_weights(frames, train_masks)
        estimate = (base_weights * log_ratios).sum() / base_weights.sum().clamp_min(
            1.0e-8
        )
        for _ in range(self.rendered_alignment_irls_iterations):
            residual = log_ratios - estimate
            centered = residual - residual.median()
            mad = centered.abs().median()
            delta = (1.345 * 1.4826 * mad).clamp_min(1.0e-4)
            huber = torch.minimum(
                torch.ones_like(residual),
                delta / residual.abs().clamp_min(1.0e-8),
            )
            robust_parts: list[torch.Tensor] = []
            offset = 0
            for mask in train_masks:
                count = int(mask.sum().item())
                robust_parts.append(huber[offset : offset + count])
                offset += count
            weights = self._balanced_frame_weights(
                frames,
                train_masks,
                point_weights=[
                    torch.zeros(
                        int(frame.current_points.shape[0]),
                        device=frame.current_points.device,
                        dtype=frame.current_points.dtype,
                    ).masked_scatter(mask, values)
                    for frame, mask, values in zip(
                        frames,
                        train_masks,
                        robust_parts,
                    )
                ],
            )
            estimate = (weights * log_ratios).sum() / weights.sum().clamp_min(
                1.0e-8
            )
        scale = float(estimate.exp().detach().cpu())
        transform, pose_pair = self._pose_prior_from_overlap(frames, scale)
        all_inlier_masks: list[torch.Tensor] = []
        relative_error_parts: list[torch.Tensor] = []
        holdout_error_parts: list[torch.Tensor] = []
        per_frame_ratios = []
        for frame in frames:
            relative_error = (
                (scale * frame.current_depth - frame.previous_depth).abs()
                / frame.previous_depth.abs().clamp_min(1.0e-6)
            )
            inliers = (
                relative_error
                <= self.rendered_alignment_max_median_relative_error
            )
            all_inlier_masks.append(inliers)
            relative_error_parts.append(relative_error[~frame.holdout_mask])
            holdout_error_parts.append(relative_error[frame.holdout_mask])
            per_frame_ratios.append(
                float(inliers.float().mean().detach().cpu())
            )
        shared_rotation_errors, shared_center_errors = self._shared_pose_errors(
            transform, frames
        )
        scale_ok = (
            math.isfinite(scale)
            and 1.0 / self.rendered_alignment_max_scale_change
            <= scale
            <= self.rendered_alignment_max_scale_change
        )
        train_ratio = sum(
            float(
                (
                    error
                    <= self.rendered_alignment_max_median_relative_error
                )
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in relative_error_parts
        ) / float(len(relative_error_parts))
        holdout_ratio = sum(
            float(
                (
                    error
                    <= self.rendered_alignment_max_median_relative_error
                )
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for error in holdout_error_parts
        ) / float(len(holdout_error_parts))
        train_median = sum(
            float(error.median().detach().cpu())
            for error in relative_error_parts
        ) / float(len(relative_error_parts))
        holdout_median = sum(
            float(error.median().detach().cpu())
            for error in holdout_error_parts
        ) / float(len(holdout_error_parts))
        accepted = (
            scale_ok
            and train_ratio >= self.rendered_alignment_min_inlier_ratio
            and holdout_ratio >= self.rendered_alignment_min_inlier_ratio
            and train_median
            <= self.rendered_alignment_max_median_relative_error
            and holdout_median
            <= self.rendered_alignment_max_median_relative_error
            and max(shared_rotation_errors, default=0.0)
            <= self.rendered_alignment_max_shared_rotation_error_deg
            and max(shared_center_errors, default=0.0)
            <= self.rendered_alignment_max_shared_center_error
        )
        if not scale_ok:
            reason = "scale_gate_rejected"
        elif train_ratio < self.rendered_alignment_min_inlier_ratio:
            reason = "training_inlier_gate_rejected"
        elif (
            train_median
            > self.rendered_alignment_max_median_relative_error
        ):
            reason = "training_residual_gate_rejected"
        elif (
            holdout_ratio < self.rendered_alignment_min_inlier_ratio
            or holdout_median
            > self.rendered_alignment_max_median_relative_error
        ):
            reason = "holdout_gate_rejected"
        elif (
            max(shared_rotation_errors, default=0.0)
            > self.rendered_alignment_max_shared_rotation_error_deg
            or max(shared_center_errors, default=0.0)
            > self.rendered_alignment_max_shared_center_error
        ):
            reason = "shared_pose_gate_rejected"
        else:
            reason = "accepted"
        diagnostics = {
            "fallback_scale": scale,
            "fallback_train_inlier_ratio": train_ratio,
            "fallback_holdout_inlier_ratio": holdout_ratio,
            "fallback_train_median_relative_error": train_median,
            "fallback_holdout_median_relative_error": holdout_median,
            "fallback_per_frame_inlier_ratio": per_frame_ratios,
            "fallback_shared_rotation_errors_deg": shared_rotation_errors,
            "fallback_shared_center_errors": shared_center_errors,
            "fallback_accepted": bool(accepted),
            "fallback_reason": reason,
            **pose_pair,
        }
        return (
            transform.detach() if accepted else None,
            all_inlier_masks,
            diagnostics,
        )

    def _set_two_frame_rendered_diagnostic(
        self,
        frames: list[OverlapFrameGeometry],
        transform: torch.Tensor,
        inlier_masks: list[torch.Tensor],
    ) -> None:
        rendered = [
            (frame, inliers)
            for frame, inliers in zip(frames, inlier_masks)
            if frame.previous_render is not None
            and frame.current_render is not None
        ]
        if not rendered:
            self._last_rendered_overlap_diagnostic = None
            return
        scale, _, _ = sim3_components(transform)
        panels: dict[str, list[torch.Tensor]] = {
            "local_depth": [],
            "aligned_local_depth": [],
            "global_depth": [],
            "relative_error": [],
            "local_alpha": [],
            "global_alpha": [],
            "sky_mask": [],
            "valid_mask": [],
            "inlier_mask": [],
        }
        frame_ids = []
        for frame, inliers in rendered:
            assert frame.previous_render is not None
            assert frame.current_render is not None
            current_depth = frame.current_render.depth
            previous_depth = frame.previous_render.depth.to(current_depth)
            aligned = current_depth * scale.to(current_depth)
            relative_error = (
                (aligned - previous_depth).abs()
                / previous_depth.abs().clamp_min(1.0e-6)
            )
            inlier_map = torch.zeros_like(current_depth, dtype=torch.bool)
            if bool(inliers.any()):
                uv = frame.uv[inliers]
                columns = torch.floor(uv[:, 0]).long().clamp(
                    0, int(inlier_map.shape[-1]) - 1
                )
                rows = torch.floor(uv[:, 1]).long().clamp(
                    0, int(inlier_map.shape[-2]) - 1
                )
                inlier_map[0, rows, columns] = True
            panels["local_depth"].append(current_depth.detach().cpu().float())
            panels["aligned_local_depth"].append(aligned.detach().cpu().float())
            panels["global_depth"].append(previous_depth.detach().cpu().float())
            panels["relative_error"].append(
                relative_error.detach().cpu().float()
            )
            panels["local_alpha"].append(
                frame.current_render.alpha.detach().cpu().float()
            )
            panels["global_alpha"].append(
                frame.previous_render.alpha.detach().cpu().float()
            )
            previous_packet_valid = frame.previous_valid_image
            current_packet_valid = frame.current_valid_image
            assert previous_packet_valid is not None
            assert current_packet_valid is not None
            panels["valid_mask"].append(
                (previous_packet_valid & current_packet_valid)
                .detach()
                .cpu()
                .bool()
            )
            sky_union = frame.sky_union_image
            assert sky_union is not None
            panels["sky_mask"].append(
                sky_union.detach().cpu().bool()
            )
            panels["inlier_mask"].append(inlier_map.detach().cpu().bool())
            frame_ids.append(frame.frame_id)
        self._last_rendered_overlap_diagnostic = {
            name: torch.stack(values, dim=0)
            for name, values in panels.items()
        }
        self._last_rendered_overlap_diagnostic["frame_ids"] = torch.tensor(
            frame_ids, dtype=torch.long
        )

    def _two_frame_overlap_constraints(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
        *,
        previous_anchor_node: int,
        current_anchor_node: int,
        use_rendered_anchors: bool,
    ) -> tuple[
        Sim3GraphEdge | None,
        tuple[DenseSphericalFactorBlock, ...],
        tuple[CoincidentPanoramaFactor, ...],
        dict[str, Any],
    ]:
        solve_start = time.perf_counter()
        overlap = self._overlap_frame_ids(previous, current)
        if len(overlap) != 2:
            return None, (), (), {
                "mode": self.rendered_overlap_alignment_mode,
                "accepted": False,
                "reason": "two_overlap_frames_required",
                "overlap_frame_ids": list(overlap),
            }
        frames: list[OverlapFrameGeometry] = []
        try:
            for frame_id in overlap:
                frames.append(
                    self._collect_overlap_frame_geometry(
                        previous,
                        current,
                        frame_id,
                        use_rendered_anchors=use_rendered_anchors,
                    )
                )
        except RuntimeError as exc:
            if frames and use_rendered_anchors:
                reference = frames[0].current_points
                self._set_two_frame_rendered_diagnostic(
                    frames,
                    sim3_identity(
                        device=reference.device,
                        dtype=reference.dtype,
                    ),
                    [
                        torch.zeros(
                            int(frame.current_points.shape[0]),
                            device=frame.current_points.device,
                            dtype=torch.bool,
                        )
                        for frame in frames
                    ],
                )
            return None, (), (), {
                "mode": self.rendered_overlap_alignment_mode,
                "accepted": False,
                "reason": "insufficient_two_frame_support",
                "overlap_frame_ids": list(overlap),
                "error": str(exc),
                "alignment_seconds": float(time.perf_counter() - solve_start),
            }

        if self.two_frame_scale_pose_enabled:
            measurement, inlier_masks, fallback_diagnostics = (
                self._fit_two_frame_scale_pose_fallback(frames)
            )
            full_diagnostics: dict[str, Any] = {
                "full_sim3_accepted": False,
                "full_sim3_reason": "disabled_by_mode",
            }
            method = "scale_pose_only"
        else:
            measurement, inlier_masks, full_diagnostics = (
                self._fit_two_frame_full_sim3(frames)
            )
            fallback_diagnostics = {}
            method = "full_sim3"
            if (
                measurement is None
                and self.rendered_alignment_failure_policy
                == "scale_pose_then_error"
            ):
                measurement, inlier_masks, fallback_diagnostics = (
                    self._fit_two_frame_scale_pose_fallback(frames)
                )
                method = "scale_pose_fallback"
        diagnostics: dict[str, Any] = {
            "mode": self.rendered_overlap_alignment_mode,
            "source_window_id": int(previous.window_id),
            "target_window_id": int(current.window_id),
            "overlap_frame_ids": list(overlap),
            "valid_points": sum(
                int(frame.current_points.shape[0]) for frame in frames
            ),
            "per_frame_valid_points": [
                int(frame.current_points.shape[0]) for frame in frames
            ],
            "alignment_method": method,
            "accepted": measurement is not None,
            "reason": (
                "accepted"
                if measurement is not None
                else fallback_diagnostics.get(
                    "fallback_reason",
                    full_diagnostics.get("full_sim3_reason", "alignment_failed"),
                )
            ),
            "render_seconds": sum(
                (
                    0.0
                    if frame.previous_render is None
                    else frame.previous_render.render_seconds
                )
                + (
                    0.0
                    if frame.current_render is None
                    else frame.current_render.render_seconds
                )
                for frame in frames
            ),
            "alignment_seconds": float(time.perf_counter() - solve_start),
            **full_diagnostics,
            **fallback_diagnostics,
        }
        per_frame_inlier_ratio = [
            float(mask.float().mean().detach().cpu())
            for mask in inlier_masks
        ]
        diagnostics["per_frame_inlier_ratio"] = per_frame_inlier_ratio
        diagnostics["inlier_ratio"] = sum(per_frame_inlier_ratio) / float(
            len(per_frame_inlier_ratio)
        )
        if measurement is None:
            if use_rendered_anchors:
                candidate_scale = float(
                    fallback_diagnostics.get(
                        "fallback_scale",
                        full_diagnostics.get("full_sim3_scale", 1.0),
                    )
                )
                reference = frames[0].current_points
                diagnostic_transform = sim3_from_components(
                    candidate_scale,
                    torch.eye(
                        3,
                        device=reference.device,
                        dtype=reference.dtype,
                    ),
                    torch.zeros(
                        3,
                        device=reference.device,
                        dtype=reference.dtype,
                    ),
                )
                self._set_two_frame_rendered_diagnostic(
                    frames,
                    diagnostic_transform,
                    inlier_masks,
                )
            return None, (), (), diagnostics
        scale, rotation, translation = sim3_components(measurement)
        diagnostics.update(
            {
                "measurement_scale": float(scale.detach().cpu()),
                "measurement_rotation_deg": float(
                    torch.rad2deg(
                        torch.linalg.norm(sim3_log(measurement)[3:6])
                    )
                    .detach()
                    .cpu()
                ),
                "measurement_translation_norm": float(
                    torch.linalg.norm(translation).detach().cpu()
                ),
                "chunk_scale_normalization": 1.0,
                "canonical_rotation_mismatch_deg": 0.0,
                "canonical_translation_mismatch": 0.0,
            }
        )
        self._set_two_frame_rendered_diagnostic(
            frames, measurement, inlier_masks
        )
        total_inliers = sum(int(mask.sum().item()) for mask in inlier_masks)
        confidence = min(
            8.0,
            max(1.0, math.sqrt(float(max(1, total_inliers)) / 64.0)),
        )
        information = measurement.new_tensor(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.75]
        ) * confidence
        sim3_edge = Sim3GraphEdge(
            source=int(previous_anchor_node),
            target=int(current_anchor_node),
            measurement_target_to_source=measurement.detach(),
            information_diag=information.detach(),
            edge_type="overlap_two_frame_sim3",
            metadata=dict(diagnostics),
        )
        dense_factors: list[DenseSphericalFactorBlock] = []
        pose_factors: list[CoincidentPanoramaFactor] = []
        for frame, inliers in zip(frames, inlier_masks):
            keep = inliers
            if int(keep.sum().item()) < self.min_dense_factors:
                continue
            frame_diagnostics = {
                "source_window_id": int(previous.window_id),
                "target_window_id": int(current.window_id),
                "source_frame_id": int(frame.frame_id),
                "target_frame_id": int(frame.frame_id),
                "overlap_frame_id": int(frame.frame_id),
                "num_matches": int(keep.sum().item()),
                "alignment_method": method,
                "weight_mode": "equal_solid_angle_per_frame",
            }
            dense_factors.append(
                DenseSphericalFactorBlock(
                    source=int(previous_anchor_node),
                    target=int(current_anchor_node),
                    source_local_pose=frame.previous_pose.detach(),
                    target_local_pose=frame.current_pose.detach(),
                    source_bearing=frame.bearing[keep].detach(),
                    target_bearing=frame.bearing[keep].detach(),
                    source_depth=frame.previous_depth[keep].detach(),
                    target_depth=frame.current_depth[keep].detach(),
                    factor_weight=torch.ones(
                        int(keep.sum().item()),
                        device=frame.bearing.device,
                        dtype=frame.bearing.dtype,
                    ),
                    depth_factor_weight=self.depth_factor_weight,
                    s2_huber_delta_deg=self.s2_huber_delta_deg,
                    use_depth=True,
                    edge_type="overlap_dense_spherical",
                    **self._dense_factor_information_options(),
                    metadata=frame_diagnostics,
                )
            )
            pose_factors.append(
                CoincidentPanoramaFactor(
                    source=int(previous_anchor_node),
                    target=int(current_anchor_node),
                    source_local_pose=frame.previous_pose.detach(),
                    target_local_pose=frame.current_pose.detach(),
                    measured_source_to_target_rotation=torch.eye(
                        3,
                        device=frame.bearing.device,
                        dtype=frame.bearing.dtype,
                    ),
                    center_weight=confidence,
                    rotation_weight=confidence,
                    edge_type="overlap_shared_pose_consistency",
                    metadata=frame_diagnostics,
                )
            )
        if len(dense_factors) != 2 or len(pose_factors) != 2:
            diagnostics.update(
                {
                    "accepted": False,
                    "reason": "insufficient_inlier_factors_after_alignment",
                }
            )
            return None, (), (), diagnostics
        return (
            sim3_edge,
            tuple(dense_factors),
            tuple(pose_factors),
            diagnostics,
        )

    def consume_rendered_overlap_diagnostic(
        self,
    ) -> dict[str, torch.Tensor] | None:
        diagnostic = self._last_rendered_overlap_diagnostic
        self._last_rendered_overlap_diagnostic = None
        return diagnostic

    def consume_overlap_alignment_failure(
        self,
    ) -> dict[str, Any] | None:
        diagnostic = self._last_overlap_alignment_failure
        self._last_overlap_alignment_failure = None
        return diagnostic

    @staticmethod
    def _rebuild_packet_anchor_geometry(packet: LocalGaussianWindowPacket) -> bool:
        """Re-voxelize anchors after a non-uniform per-frame depth replacement."""

        if packet.anchor_observation is None:
            return False
        if bool(packet.metadata.get("voxel_anchor_refiner_enabled", False)):
            raise RuntimeError(
                "Refined anchor packets must never be re-voxelized in the backend"
            )
        config = packet.anchor_observation.config
        observation = packet.observation
        images = observation.refined_depth.new_zeros(
            observation.batch_size,
            observation.num_source_views,
            3,
            *observation.image_size,
        )
        packet.anchor_observation = voxelize_per_pixel_gaussians(
            observation,
            packet.adapter_features.to(observation.refined_depth),
            images,
            config,
            valid_mask=packet.finite_gaussian_mask.to(observation.refined_depth.device),
        ).detach_for_backend()
        packet.metadata["anchor_geometry_rebuilt_after_depth_sync"] = True
        return True

    @classmethod
    def _replace_packet_depth(
        cls,
        packet: LocalGaussianWindowPacket,
        depth: torch.Tensor,
        *,
        rebuild_anchors: bool = True,
    ) -> None:
        target = depth.detach().to(packet.observation.refined_depth)
        if tuple(target.shape) != tuple(packet.observation.refined_depth.shape):
            raise ValueError("Replacement packet depth must match refined_depth shape")
        packet.observation = packet.observation.with_geometry(refined_depth=target.clone())
        if rebuild_anchors:
            cls._rebuild_packet_anchor_geometry(packet)

    @classmethod
    def _restore_packet_pre_shift_depth(
        cls,
        packet: LocalGaussianWindowPacket,
    ) -> bool:
        if packet.pre_depth_shift_depth is None:
            return False
        cls._replace_packet_depth(packet, packet.pre_depth_shift_depth)
        packet.pre_depth_shift_depth = None
        packet.metadata["depth_shift_rollback"] = True
        return True

    @classmethod
    def _synchronize_shared_canonical_depth(
        cls,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
        frame_id: int,
    ) -> None:
        source_index = source.frame_index(frame_id)
        target_index = target.frame_index(frame_id)
        source_depth = source.observation.refined_depth[0, source_index]
        target_depth = target.observation.refined_depth
        if tuple(source_depth.shape) != tuple(target_depth[0, target_index].shape):
            raise ValueError("Canonical shared-frame depth requires matching ERP resolution")
        synchronized = target_depth.detach().clone()
        synchronized[0, target_index] = source_depth.to(synchronized)
        cls._replace_packet_depth(target, synchronized)
        target.metadata["canonical_shared_depth_frame_id"] = int(frame_id)
        target.metadata["canonical_shared_depth_owner_window"] = int(source.window_id)

    def _shared_frame_alignment(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        overlap = sorted(set(source.frame_ids) & set(target.frame_ids))
        if self.enforce_exact_overlap and len(overlap) != self.expected_overlap_frames:
            return None, {
                "reason": "unexpected_overlap_count",
                "overlap_frame_ids": overlap,
                "expected_overlap_frames": self.expected_overlap_frames,
            }
        if len(overlap) != 1:
            return None, {"reason": "single_overlap_required", "overlap_frame_ids": overlap}
        frame_id = int(overlap[0])
        source_index = source.frame_index(frame_id)
        target_index = target.frame_index(frame_id)
        source_depth = source.observation.refined_depth[0, source_index].detach()
        target_depth = target.observation.refined_depth[0, target_index].detach().to(source_depth)
        edge_seed = (
            self.fibonacci_seed
            + 1_000_003 * int(source.window_id)
            + 10_007 * int(target.window_id)
            + 101 * frame_id
        ) & 0x7FFFFFFF
        samples = sample_joint_valid_fibonacci_uv(
            source_depth,
            target_depth,
            count=self.max_overlap_points,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            source_valid=source.finite_gaussian_mask[0, source_index].detach(),
            target_valid=target.finite_gaussian_mask[0, target_index].detach().to(source_depth.device),
            source_sky_probability=source.sky_prob[0, source_index].detach(),
            target_sky_probability=target.sky_prob[0, target_index].detach().to(source_depth.device),
            sky_threshold=self.sky_threshold,
            seed=edge_seed,
        )
        if int(samples.uv.shape[0]) < self.overlap_aligner.min_points:
            return None, {
                "reason": "insufficient_joint_fibonacci_support",
                "overlap_frame_ids": overlap,
                "overlap_points": int(samples.uv.shape[0]),
                "fibonacci_seed": edge_seed,
            }

        source_pose = source.local_poses_c2w[source_index].to(samples.bearing)
        target_pose = target.local_poses_c2w[target_index].to(samples.bearing)
        source_camera = samples.bearing * samples.source_depth[:, None]
        target_camera = samples.bearing * samples.target_depth[:, None]
        source_points = source_camera @ source_pose[:3, :3].transpose(0, 1) + source_pose[:3, 3]
        target_points = target_camera @ target_pose[:3, :3].transpose(0, 1) + target_pose[:3, 3]
        # Fibonacci samples are equal-solid-angle. All accepted points enter
        # Umeyama with equal measurement weight; only robust residual gating
        # inside SubmapAligner may change their influence.
        alignment = self.overlap_aligner.align(
            target_points,
            source_points,
            torch.ones(int(target_points.shape[0]), device=target_points.device, dtype=target_points.dtype),
        )
        diagnostics = {
            "overlap_frame_ids": overlap,
            "source_frame_id": frame_id,
            "target_frame_id": frame_id,
            "overlap_points": int(target_points.shape[0]),
            "overlap_residual": float(alignment.residual),
            "overlap_inlier_ratio": float(alignment.inlier_ratio),
            "fibonacci_seed": int(samples.seed),
            "fibonacci_longitude_phase": float(samples.longitude_phase),
            "descriptor_gate": False,
            "weight_mode": "fibonacci_equal_joint_geometry_mask",
        }
        if not alignment.accepted:
            diagnostics["reason"] = "overlap_alignment_rejected"
            return None, diagnostics
        return alignment.as_matrix().detach(), diagnostics

    def _fallback_boundary_matches(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> BoundaryMatchBlock | None:
        if not self.allow_boundary_matching_fallback:
            return None
        from models.spherical_selfi_stage3_ba import build_stage3_match_cache

        generator = torch.Generator(device=packet.adapter_features.device)
        generator.manual_seed((self.fibonacci_seed + int(packet.window_id)) & 0x7FFFFFFF)
        cache = build_stage3_match_cache(
            packet.adapter_features,
            packet.observation.refined_depth,
            num_queries=min(self.max_overlap_points, int(packet.observation.image_size[0] * packet.observation.image_size[1])),
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            fibonacci_oversample_factor=self.fibonacci_oversample_factor,
            use_spherical_area_correction=True,
            forward_backward=bool(
                self.config.get("global_graph", {}).get("forward_backward", True)
            ),
            fb_tolerance_deg=float(self.config.get("global_graph", {}).get("fb_tolerance_deg", 1.0)),
            min_factor_weight=0.0,
            edge_topology="all_directed",
            static_valid_mask=packet.finite_gaussian_mask,
            generator=generator,
        )
        last = cache.num_views - 1
        entropy_scale = max(math.log(max(2, packet.observation.image_size[0] * packet.observation.image_size[1])), 1.0e-8)
        pieces: dict[str, list[torch.Tensor]] = {
            "source_uv": [], "target_uv": [], "source_bearing": [], "target_bearing": [],
            "top1_cosine": [], "top2_margin": [], "normalized_entropy": [],
        }
        for edge_index, pair in enumerate(cache.edges.detach().cpu().tolist()):
            src, tgt = int(pair[0]), int(pair[1])
            if (src, tgt) not in {(0, last), (last, 0)}:
                continue
            keep = cache.valid_mask[0, edge_index]
            if src == 0:
                values = (
                    cache.source_uv[0, 0, keep], cache.target_uv[0, edge_index, keep],
                    cache.source_ray[0, 0, keep], cache.target_ray[0, edge_index, keep],
                )
            else:
                values = (
                    cache.target_uv[0, edge_index, keep], cache.source_uv[0, last, keep],
                    cache.target_ray[0, edge_index, keep], cache.source_ray[0, last, keep],
                )
            for name, value in zip(
                ("source_uv", "target_uv", "source_bearing", "target_bearing"), values
            ):
                pieces[name].append(value)
            pieces["top1_cosine"].append(cache.top1_cosine[0, edge_index, keep])
            pieces["top2_margin"].append(cache.top2_margin[0, edge_index, keep])
            pieces["normalized_entropy"].append(
                (cache.entropy[0, edge_index, keep] / entropy_scale).clamp(0.0, 1.0)
            )
        if not pieces["source_uv"]:
            return None
        return BoundaryMatchBlock(
            **{name: torch.cat(value, dim=0).detach().clone() for name, value in pieces.items()}
        )

    @staticmethod
    def _balanced_chunk_stride_weights(
        direction: torch.Tensor,
        selected: torch.Tensor,
        *,
        dtype: torch.dtype,
        quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Balance query directions while retaining soft row quality."""

        weights = torch.zeros(
            int(direction.numel()),
            device=direction.device,
            dtype=dtype,
        )
        present = [
            value
            for value in (0, 1)
            if bool((selected & (direction == value)).any())
        ]
        if not present:
            raise RuntimeError("Chunk-stride weights require at least one row")
        row_quality = (
            torch.ones_like(direction, dtype=dtype)
            if quality is None
            else quality.to(device=direction.device, dtype=dtype).reshape(-1)
        )
        if int(row_quality.numel()) != int(direction.numel()):
            raise ValueError("Chunk-stride quality must match correspondence count")
        row_quality = row_quality.clamp_min(0.0)
        direction_weight = 1.0 / float(len(present))
        for value in present:
            rows = selected & (direction == value)
            count = int(rows.sum().item())
            values = row_quality[rows]
            total = values.sum()
            if float(total.detach().cpu()) <= 1.0e-12:
                weights[rows] = direction_weight / float(count)
            else:
                weights[rows] = direction_weight * values / total
        return weights

    @staticmethod
    def _chunk_stride_holdout_mask(
        direction: torch.Tensor,
        *,
        source_frame: int,
        target_frame: int,
        stride: int,
    ) -> torch.Tensor:
        """Return a deterministic split for every available query direction."""

        count = int(direction.numel())
        rows = torch.arange(count, device=direction.device, dtype=torch.int64)
        hashed = (
            rows * 1_000_003
            + int(source_frame) * 10_007
            + int(target_frame) * 101
            + direction.to(torch.int64) * 97
        )
        holdout = (hashed.remainder(int(stride))) == 0
        for value in torch.unique(direction).tolist():
            members = torch.nonzero(direction == value, as_tuple=False).flatten()
            if int(members.numel()) < 2:
                holdout[members] = False
                continue
            if not bool(holdout.index_select(0, members).any()):
                holdout[members[-1]] = True
            if bool(holdout.index_select(0, members).all()):
                holdout[members[0]] = False
        # Umeyama needs at least three optimization rows. With very sparse
        # support, prefer using every finite correspondence over manufacturing
        # a holdout that would make the dense edge impossible to construct.
        if int((~holdout).sum().item()) < 3:
            holdout.zero_()
        return holdout

    @staticmethod
    def _chunk_stride_alignment_errors(
        transform_target_to_source: torch.Tensor,
        source_bearing: torch.Tensor,
        target_bearing: torch.Tensor,
        source_depth: torch.Tensor,
        target_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_points = target_bearing * target_depth[:, None]
        predicted_source = apply_sim3(
            transform_target_to_source.to(target_points), target_points
        )
        predicted_depth = torch.linalg.norm(predicted_source, dim=-1).clamp_min(
            1.0e-8
        )
        predicted_bearing = predicted_source / predicted_depth[:, None]
        cosine = (predicted_bearing * source_bearing).sum(dim=-1).clamp(-1.0, 1.0)
        angular_deg = torch.rad2deg(torch.acos(cosine))
        relative_depth = (
            (predicted_depth - source_depth).abs()
            / source_depth.abs().clamp_min(1.0e-6)
        )
        return angular_deg, relative_depth

    @staticmethod
    def _spherical_coverage_ratio(bearing: torch.Tensor) -> float:
        value = bearing / torch.linalg.norm(
            bearing, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        latitude = torch.asin(value[:, 1].clamp(-1.0, 1.0))
        longitude = torch.atan2(value[:, 0], value[:, 2])
        latitude_bin = torch.floor(
            (latitude + 0.5 * math.pi) / math.pi * 3.0
        ).long().clamp(0, 2)
        longitude_bin = torch.floor(
            (longitude + math.pi) / (2.0 * math.pi) * 8.0
        ).long().clamp(0, 7)
        occupied = torch.unique(latitude_bin * 8 + longitude_bin)
        return float(occupied.numel()) / 24.0

    def _chunk_stride_pose_fallback(
        self,
        packet: LocalGaussianWindowPacket,
        *,
        target_index: int,
        edge_type: str,
        diagnostics: dict[str, Any],
    ) -> tuple[Sim3GraphEdge, torch.Tensor, None, dict[str, Any]]:
        """Keep the chunk graph connected with canonical BA odometry.

        The fallback is used only when a finite dense S2+depth factor cannot be
        formed. It deliberately carries no scale information: packet overlap
        depth has already canonicalized the local unit and pure-chain node
        scale remains graph-owned.
        """

        if not 0 < int(target_index) < len(packet.frame_ids):
            raise RuntimeError(
                "Chunk-stride BA fallback target index is outside the packet"
            )
        source_pose = packet.local_poses_c2w[0]
        target_pose = packet.local_poses_c2w[int(target_index)]
        relative_pose = canonicalize_c2w(
            invert_c2w(source_pose) @ target_pose
        )
        if tuple(relative_pose.shape) != (4, 4) or not bool(
            torch.isfinite(relative_pose).all()
        ):
            raise RuntimeError(
                "Chunk-stride BA fallback requires a finite canonical pose"
            )
        measurement = sim3_from_components(
            1.0,
            relative_pose[:3, :3],
            relative_pose[:3, 3],
        )
        metadata = dict(diagnostics)
        metadata.update(
            {
                "accepted": True,
                "quality_gating_enabled": False,
                "fallback_used": True,
                "fallback_reason": str(
                    diagnostics.get("reason", "dense_factor_unavailable")
                ),
                "reason": "canonical_ba_pose_fallback",
                "factor_representation": "sim3_pose_fallback",
                "source_frame_id": int(packet.frame_ids[0]),
                "target_frame_id": int(packet.frame_ids[int(target_index)]),
            }
        )
        factor = Sim3GraphEdge(
            source=int(packet.frame_ids[0]),
            target=int(packet.frame_ids[int(target_index)]),
            measurement_target_to_source=measurement.detach(),
            information_diag=measurement.new_tensor(
                [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0]
            ),
            edge_type=str(edge_type),
            metadata=metadata,
        )
        return factor, measurement.detach(), None, metadata

    def _chunk_stride_factor(
        self,
        packet: LocalGaussianWindowPacket,
        *,
        edge_type: str = "chunk_stride_dense_spherical",
        expected_target_index: int | None = None,
        validate_local_ba: bool = True,
        reference_measurement: torch.Tensor | None = None,
        allow_pose_fallback: bool = True,
    ) -> tuple[
        DenseSphericalFactorBlock | Sim3GraphEdge | None,
        torch.Tensor | None,
        ChunkStrideHoldout | None,
        dict[str, Any],
    ]:
        """Fit one independently matched chunk-anchor S²+depth factor."""

        matches = packet.chunk_stride_matches
        fallback_target_index = int(
            self.chunk_stride_target_index
            if expected_target_index is None
            else expected_target_index
        )
        target_index = (
            fallback_target_index
            if matches is None
            else int(matches.target_index)
        )
        ba_residual = packet.metadata.get(
            "local_ba_final_median_residual_deg"
        )
        ba_trust_touched = bool(
            packet.metadata.get("local_ba_trust_region_touched", False)
        )
        ba_quality_diagnostic = {
            "local_ba_validation_requested": bool(validate_local_ba),
            "local_ba_accepted": packet.metadata.get("local_ba_accepted"),
            "local_ba_final_median_residual_deg": ba_residual,
            "local_ba_trust_region_touched": ba_trust_touched,
            "quality_gating_enabled": False,
        }
        if matches is None:
            diagnostics = {
                **ba_quality_diagnostic,
                "reason": "chunk_stride_matches_unavailable",
                "raw_chunk_stride_matches": 0,
                "hard_gated_chunk_stride_matches": 0,
                "edge_type": str(edge_type),
            }
            if allow_pose_fallback:
                return self._chunk_stride_pose_fallback(
                    packet,
                    target_index=fallback_target_index,
                    edge_type=edge_type,
                    diagnostics=diagnostics,
                )
            return None, None, None, diagnostics
        if (
            int(matches.source_index) != 0
            or target_index >= len(packet.frame_ids)
            or target_index <= 0
            or (
                expected_target_index is not None
                and target_index != int(expected_target_index)
            )
        ):
            diagnostics = {
                **ba_quality_diagnostic,
                "reason": "unexpected_chunk_stride_match_indices",
                "source_index": int(matches.source_index),
                "target_index": int(matches.target_index),
            }
            if allow_pose_fallback:
                return self._chunk_stride_pose_fallback(
                    packet,
                    target_index=fallback_target_index,
                    edge_type=edge_type,
                    diagnostics=diagnostics,
                )
            return None, None, None, diagnostics

        source_depth = sample_erp_with_wrap(
            packet.observation.refined_depth[0, 0], matches.source_uv
        )[..., 0]
        target_depth = sample_erp_with_wrap(
            packet.observation.refined_depth[0, target_index], matches.target_uv
        )[..., 0]

        def sampled_mask(value: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
            return sample_erp_with_wrap(value.float(), uv)[..., 0] >= 0.5

        source_valid = (
            sampled_mask(packet.finite_gaussian_mask[0, 0], matches.source_uv)
            & sampled_mask(packet.static_mask[0, 0], matches.source_uv)
            & sampled_mask(packet.geometry_consistency[0, 0], matches.source_uv)
        )
        target_valid = (
            sampled_mask(
                packet.finite_gaussian_mask[0, target_index], matches.target_uv
            )
            & sampled_mask(packet.static_mask[0, target_index], matches.target_uv)
            & sampled_mask(
                packet.geometry_consistency[0, target_index], matches.target_uv
            )
        )
        source_sky = sample_erp_with_wrap(
            packet.sky_prob[0, 0], matches.source_uv
        )[..., 0]
        target_sky = sample_erp_with_wrap(
            packet.sky_prob[0, target_index], matches.target_uv
        )[..., 0]
        keep = (
            source_valid
            & target_valid
            & (source_sky < self.sky_threshold)
            & (target_sky < self.sky_threshold)
            & torch.isfinite(source_depth)
            & torch.isfinite(target_depth)
            & (source_depth >= self.fibonacci_min_depth)
            & (target_depth >= self.fibonacci_min_depth)
            & (source_depth <= self.fibonacci_max_depth)
            & (target_depth <= self.fibonacci_max_depth)
            & (matches.top1_cosine >= self.min_match_cosine)
            & (matches.normalized_entropy <= self.max_match_entropy)
        )
        source_frame = int(packet.frame_ids[0])
        target_frame = int(packet.frame_ids[target_index])
        diagnostics: dict[str, Any] = {
            **ba_quality_diagnostic,
            "source_window_id": int(packet.window_id),
            "target_window_id": int(packet.window_id),
            "source_frame_id": source_frame,
            "target_frame_id": target_frame,
            "raw_chunk_stride_matches": int(matches.count),
            "hard_gated_chunk_stride_matches": int(keep.sum().item()),
            "sky_rejected": int(
                (
                    (source_sky >= self.sky_threshold)
                    | (target_sky >= self.sky_threshold)
                ).sum().item()
            ),
            "weight_mode": "bidirectional_half_balanced",
            "umeyama_iterations": int(self.chunk_stride_irls_iterations),
            "edge_type": str(edge_type),
            "independent_correspondences": True,
        }
        if int(keep.sum().item()) < 3:
            diagnostics["reason"] = "insufficient_finite_chunk_stride_matches"
            if allow_pose_fallback:
                return self._chunk_stride_pose_fallback(
                    packet,
                    target_index=fallback_target_index,
                    edge_type=edge_type,
                    diagnostics=diagnostics,
                )
            return None, None, None, diagnostics

        direction = matches.query_direction[keep].to(source_depth.device).long()
        source_bearing = matches.source_bearing[keep].to(source_depth)
        target_bearing = matches.target_bearing[keep].to(source_depth)
        source_depth = source_depth[keep]
        target_depth = target_depth[keep]
        descriptor_quality = (
            matches.top1_cosine[keep].to(source_depth).clamp(0.0, 1.0)
            * (
                1.0
                - matches.normalized_entropy[keep]
                .to(source_depth)
                .clamp(0.0, 1.0)
            )
        ).clamp_min(1.0e-3)
        try:
            holdout_mask = self._chunk_stride_holdout_mask(
                direction,
                source_frame=source_frame,
                target_frame=target_frame,
                stride=self.chunk_stride_holdout_stride,
            )
            train_mask = ~holdout_mask
            descriptor_reference_weight = self._balanced_chunk_stride_weights(
                direction,
                train_mask,
                dtype=source_depth.dtype,
            )
            descriptor_score = float(
                (
                    descriptor_reference_weight * descriptor_quality
                ).sum().detach().cpu()
            )
            base_weight = self._balanced_chunk_stride_weights(
                direction,
                train_mask,
                dtype=source_depth.dtype,
                quality=descriptor_quality,
            )
        except RuntimeError as error:
            diagnostics.update(
                {"reason": "chunk_stride_weighting_failed", "detail": str(error)}
            )
            if allow_pose_fallback:
                return self._chunk_stride_pose_fallback(
                    packet,
                    target_index=fallback_target_index,
                    edge_type=edge_type,
                    diagnostics=diagnostics,
                )
            return None, None, None, diagnostics

        source_points = source_bearing * source_depth[:, None]
        target_points = target_bearing * target_depth[:, None]
        train_rows = torch.nonzero(train_mask, as_tuple=False).flatten()
        robust_weight = base_weight.clone()
        try:
            measurement = weighted_umeyama(
                target_points.index_select(0, train_rows),
                source_points.index_select(0, train_rows),
                robust_weight.index_select(0, train_rows),
                allow_scale=True,
            )
            for _ in range(self.chunk_stride_irls_iterations):
                residual = torch.linalg.norm(
                    apply_sim3(measurement, target_points) - source_points,
                    dim=-1,
                )
                train_residual = residual.index_select(0, train_rows)
                train_base_weight = base_weight.index_select(0, train_rows)
                centered = train_residual - self._weighted_median_1d(
                    train_residual,
                    train_base_weight,
                )
                sigma = (
                    1.4826
                    * self._weighted_median_1d(
                        centered.abs(), train_base_weight
                    )
                ).clamp_min(1.0e-5)
                delta = (1.345 * sigma).clamp_min(1.0e-4)
                huber = torch.minimum(
                    torch.ones_like(residual),
                    delta / residual.clamp_min(1.0e-8),
                )
                robust_weight = base_weight * huber
                measurement = weighted_umeyama(
                    target_points.index_select(0, train_rows),
                    source_points.index_select(0, train_rows),
                    robust_weight.index_select(0, train_rows),
                    allow_scale=True,
                )
        except (RuntimeError, ValueError) as error:
            diagnostics.update(
                {"reason": "chunk_stride_umeyama_failed", "detail": str(error)}
            )
            if allow_pose_fallback:
                return self._chunk_stride_pose_fallback(
                    packet,
                    target_index=fallback_target_index,
                    edge_type=edge_type,
                    diagnostics=diagnostics,
                )
            return None, None, None, diagnostics

        residual = torch.linalg.norm(
            apply_sim3(measurement, target_points) - source_points,
            dim=-1,
        )
        median_depth = float(source_depth[train_mask].median().detach().cpu())
        inlier_threshold = max(1.0e-3, 0.10 * median_depth)
        inliers = train_mask & (residual <= inlier_threshold)
        inlier_ratio = float(
            (
                (base_weight * inliers.to(base_weight)).sum()
                / base_weight.sum().clamp_min(1.0e-8)
            )
            .detach()
            .cpu()
        )
        diagnostic_mask = (
            inliers if int(inliers.sum().item()) >= 3 else train_mask
        )
        inlier_weight = self._balanced_chunk_stride_weights(
            direction,
            diagnostic_mask,
            dtype=source_depth.dtype,
            quality=descriptor_quality,
        )
        source_singular = self._weighted_point_singular_values(
            source_points[diagnostic_mask], inlier_weight[diagnostic_mask]
        )
        target_singular = self._weighted_point_singular_values(
            target_points[diagnostic_mask], inlier_weight[diagnostic_mask]
        )
        source_ratio = float(
            (source_singular[-1] / source_singular[0].clamp_min(1.0e-12))
            .detach()
            .cpu()
        )
        target_ratio = float(
            (target_singular[-1] / target_singular[0].clamp_min(1.0e-12))
            .detach()
            .cpu()
        )
        covariance_ratio = min(source_ratio, target_ratio)

        scale, rotation, translation = sim3_components(measurement)
        local_pose = (
            packet.local_poses_c2w[target_index].to(measurement)
            if reference_measurement is None
            else reference_measurement.to(measurement)
        )
        rotation_error = self._rotation_error_deg(
            local_pose[:3, :3], rotation
        )
        translation_error = float(
            torch.linalg.norm(translation - local_pose[:3, 3]).detach().cpu()
        )
        holdout_available = bool(holdout_mask.any())
        train_angular, train_depth_error = self._chunk_stride_alignment_errors(
            measurement,
            source_bearing[train_mask],
            target_bearing[train_mask],
            source_depth[train_mask],
            target_depth[train_mask],
        )
        train_weight = base_weight[train_mask]
        train_angular_median = float(
            self._weighted_median_1d(
                train_angular, train_weight
            ).detach().cpu()
        )
        train_depth_median = float(
            self._weighted_median_1d(
                train_depth_error, train_weight
            ).detach().cpu()
        )
        if holdout_available:
            holdout_angular, holdout_depth = self._chunk_stride_alignment_errors(
                measurement,
                source_bearing[holdout_mask],
                target_bearing[holdout_mask],
                source_depth[holdout_mask],
                target_depth[holdout_mask],
            )
            holdout_weight = self._balanced_chunk_stride_weights(
                direction,
                holdout_mask,
                dtype=source_depth.dtype,
                quality=descriptor_quality,
            )[holdout_mask]
            holdout_angular_median = float(
                self._weighted_median_1d(
                    holdout_angular, holdout_weight
                ).detach().cpu()
            )
            holdout_depth_median = float(
                self._weighted_median_1d(
                    holdout_depth, holdout_weight
                ).detach().cpu()
            )
        else:
            holdout_angular_median = 0.0
            holdout_depth_median = 0.0
        source_coverage = self._spherical_coverage_ratio(
            source_bearing[diagnostic_mask]
        )
        target_coverage = self._spherical_coverage_ratio(
            target_bearing[diagnostic_mask]
        )
        coverage = 0.5 * (source_coverage + target_coverage)
        support_score = min(
            1.0,
            float(train_mask.sum().item())
            / float(max(self.dense_information_reference_count, 1.0)),
        )
        angular_reference = (
            holdout_angular_median
            if holdout_available
            else train_angular_median
        )
        depth_reference = (
            holdout_depth_median if holdout_available else train_depth_median
        )
        angular_scale_deg = max(float(self.s2_huber_delta_deg), 1.0e-3)
        angular_score = 1.0 / (
            1.0 + (max(0.0, angular_reference) / angular_scale_deg) ** 2
        )
        depth_score = 1.0 / (
            1.0 + (max(0.0, depth_reference) / 0.25) ** 2
        )
        common_quality = (
            max(support_score, 1.0e-6)
            * max(coverage, 1.0e-6)
            * max(descriptor_score, 1.0e-6)
        )
        s2_information_scale = max(
            0.05,
            min(
                1.0,
                (common_quality * max(angular_score, 1.0e-6)) ** 0.25,
            ),
        )
        depth_information_scale = max(
            0.05,
            min(
                1.0,
                (common_quality * max(depth_score, 1.0e-6)) ** 0.25,
            ),
        )
        information_confidence = math.sqrt(
            s2_information_scale * depth_information_scale
        )
        scale_value = float(scale.detach().cpu())
        diagnostics.update(
            {
                "reason": "accepted_without_geometric_quality_gate",
                "accepted": True,
                "quality_gating_enabled": False,
                "fallback_used": False,
                "factor_representation": "dense_spherical_depth",
                "train_matches": int(train_mask.sum().item()),
                "holdout_matches": int(holdout_mask.sum().item()),
                "holdout_available": holdout_available,
                "umeyama_inliers": int(inliers.sum().item()),
                "umeyama_inlier_ratio": inlier_ratio,
                "umeyama_inlier_threshold": inlier_threshold,
                "scale": scale_value,
                "rotation_error_vs_local_ba_deg": rotation_error,
                "translation_error_vs_local_ba": translation_error,
                "median_scene_depth": median_depth,
                "source_covariance_singular_values": [
                    float(value) for value in source_singular.detach().cpu()
                ],
                "target_covariance_singular_values": [
                    float(value) for value in target_singular.detach().cpu()
                ],
                "covariance_ratio": covariance_ratio,
                "holdout_median_angular_error_deg": holdout_angular_median,
                "holdout_median_relative_depth_error": holdout_depth_median,
                "train_median_angular_error_deg": train_angular_median,
                "train_median_relative_depth_error": train_depth_median,
                "source_spherical_coverage": source_coverage,
                "target_spherical_coverage": target_coverage,
                "support_score": support_score,
                "descriptor_score": descriptor_score,
                "angular_score": angular_score,
                "depth_score": depth_score,
                "information_confidence": information_confidence,
                "s2_information_scale": s2_information_scale,
                "depth_information_scale": depth_information_scale,
                "effective_s2_information": (
                    self.dense_information_reference_count
                    * s2_information_scale
                ),
                "effective_depth_information": (
                    self.dense_information_reference_count
                    * max(float(self.depth_factor_weight), 0.0)
                    * depth_information_scale
                ),
                "local_ba_final_median_residual_deg": (
                    None if ba_residual is None else float(ba_residual)
                ),
                "local_ba_trust_region_touched": ba_trust_touched,
            }
        )
        selected = torch.nonzero(train_mask, as_tuple=False).flatten()
        identity = torch.eye(
            4, device=source_depth.device, dtype=source_depth.dtype
        )
        factor = DenseSphericalFactorBlock(
            source=source_frame,
            target=target_frame,
            source_local_pose=identity,
            target_local_pose=identity.clone(),
            source_bearing=source_bearing.index_select(0, selected),
            target_bearing=target_bearing.index_select(0, selected),
            source_depth=source_depth.index_select(0, selected),
            target_depth=target_depth.index_select(0, selected),
            factor_weight=(
                robust_weight.index_select(0, selected)
            ),
            depth_factor_weight=self.depth_factor_weight,
            s2_huber_delta_deg=self.s2_huber_delta_deg,
            use_depth=True,
            edge_type=str(edge_type),
            s2_information_scale=s2_information_scale,
            depth_information_scale=depth_information_scale,
            **self._dense_factor_information_options(),
            metadata=dict(diagnostics),
        )
        holdout = (
            ChunkStrideHoldout(
                source=source_frame,
                target=target_frame,
                edge_type=str(edge_type),
                source_bearing=source_bearing[holdout_mask].detach().clone(),
                target_bearing=target_bearing[holdout_mask].detach().clone(),
                source_depth=source_depth[holdout_mask].detach().clone(),
                target_depth=target_depth[holdout_mask].detach().clone(),
                initial_angular_median_deg=holdout_angular_median,
                initial_relative_depth_median=holdout_depth_median,
            )
            if holdout_available
            else None
        )
        return factor, measurement.detach(), holdout, diagnostics

    def _independent_chunk_skip_factor(
        self,
        previous: LocalGaussianWindowPacket,
        current: LocalGaussianWindowPacket,
    ) -> tuple[
        DenseSphericalFactorBlock | None,
        ChunkStrideHoldout | None,
        dict[str, Any],
    ]:
        """Match Ck to Ck+2 from scratch and build an independent cycle edge."""

        if not self.chunk_skip_enabled:
            return None, None, {"enabled": False, "reason": "disabled"}
        target_index = int(self.chunk_stride_target_index)
        if target_index >= len(current.frame_ids):
            return None, None, {
                "enabled": True,
                "reason": "target_index_out_of_range",
            }
        source_frame = int(previous.frame_ids[0])
        target_frame = int(current.frame_ids[target_index])
        if source_frame not in self.graph.nodes or target_frame not in self.graph.nodes:
            return None, None, {
                "enabled": True,
                "reason": "missing_skip_endpoint",
                "source_frame_id": source_frame,
                "target_frame_id": target_frame,
            }

        # Only the immediately previous full packet is used here.  Keeping the
        # native adapter grid avoids quantizing a 2-degree holdout gate to the
        # 32x64 loop-verification grid.
        source = previous
        target = current
        if (
            source.observation.image_size != target.observation.image_size
            or tuple(source.verification_features.shape[-3:])
            != tuple(target.verification_features.shape[-3:])
        ):
            return None, None, {
                "enabled": True,
                "reason": "skip_verification_shape_mismatch",
            }

        def pair_views(name: str) -> torch.Tensor:
            left = getattr(source, name)[:, 0:1]
            right = getattr(target, name)[:, target_index : target_index + 1]
            return torch.cat([left, right.to(left)], dim=1)

        source_observation = source.observation

        def observation_views(name: str) -> torch.Tensor:
            left = getattr(source_observation, name)[:, 0:1]
            right = getattr(target.observation, name)[
                :, target_index : target_index + 1
            ]
            return torch.cat([left, right.to(left)], dim=1)

        pair_pose = torch.eye(
            4,
            device=source.local_poses_c2w.device,
            dtype=source.local_poses_c2w.dtype,
        ).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        pair_observation = replace(
            source_observation,
            initial_depth=observation_views("initial_depth"),
            depth_residual=observation_views("depth_residual"),
            refined_depth=observation_views("refined_depth"),
            poses_c2w=pair_pose.to(source_observation.poses_c2w),
            local_quaternion=observation_views("local_quaternion"),
            log_scale_multiplier=observation_views("log_scale_multiplier"),
            rgb_sh=observation_views("rgb_sh"),
            density_sh=observation_views("density_sh"),
            confidence=observation_views("confidence"),
            valid_mask=observation_views("valid_mask").bool(),
            frame_ids=torch.tensor(
                [[source_frame, target_frame]],
                device=source_observation.frame_ids.device,
                dtype=source_observation.frame_ids.dtype,
            ),
        )
        pair_features = torch.cat(
            [
                source.verification_features[:, 0:1],
                target.verification_features[
                    :, target_index : target_index + 1
                ].to(source.verification_features),
            ],
            dim=1,
        )
        pair_packet = LocalGaussianWindowPacket(
            window_id=int(current.window_id),
            anchor_frame_id=source_frame,
            frame_ids=(source_frame, target_frame),
            local_poses_c2w=pair_pose[0].float(),
            observation=pair_observation,
            adapter_features=pair_features,
            retrieval_descriptors=torch.cat(
                [
                    source.retrieval_descriptors[0:1],
                    target.retrieval_descriptors[
                        target_index : target_index + 1
                    ].to(source.retrieval_descriptors),
                ],
                dim=0,
            ),
            verification_features=pair_features,
            valid_mask=pair_views("valid_mask").bool(),
            finite_gaussian_mask=pair_views("finite_gaussian_mask").bool(),
            sky_prob=pair_views("sky_prob").float(),
            sky_mask=pair_views("sky_mask").bool(),
            static_mask=pair_views("static_mask").bool(),
            geometry_consistency=pair_views("geometry_consistency").bool(),
            metadata={
                "skip_source_window_id": int(previous.window_id),
                "skip_target_window_id": int(current.window_id),
            },
        )
        valid = (
            pair_packet.finite_gaussian_mask
            & pair_packet.static_mask
            & pair_packet.geometry_consistency
            & ~pair_packet.sky_mask
        )
        generator = torch.Generator(device=pair_features.device)
        seed = (
            self.fibonacci_seed
            + 1_000_003 * source_frame
            + 10_007 * target_frame
        ) & 0x7FFFFFFF
        generator.manual_seed(seed)
        try:
            cache = build_stage3_match_cache(
                pair_features,
                pair_observation.refined_depth,
                num_queries=self.chunk_skip_num_queries,
                min_depth=self.fibonacci_min_depth,
                max_depth=self.fibonacci_max_depth,
                temperature=self.chunk_skip_temperature,
                query_chunk_size=self.chunk_skip_query_chunk_size,
                fibonacci_oversample_factor=self.chunk_skip_oversample_factor,
                use_spherical_area_correction=self.chunk_skip_area_correction,
                forward_backward=self.chunk_skip_forward_backward,
                fb_tolerance_deg=self.chunk_skip_fb_tolerance_deg,
                min_factor_weight=self.chunk_skip_min_factor_weight,
                factor_weight_mode="descriptor_confidence",
                subpixel_refine_radius=self.chunk_skip_subpixel_refine_radius,
                edge_topology="all_directed",
                static_valid_mask=valid,
                generator=generator,
            )
        except (RuntimeError, ValueError) as error:
            return None, None, {
                "enabled": True,
                "reason": "independent_skip_matching_failed",
                "detail": str(error),
                "source_frame_id": source_frame,
                "target_frame_id": target_frame,
            }
        pair_packet.chunk_stride_matches = chunk_stride_matches_from_cache(
            cache,
            pair_observation.image_size,
            stride=1,
        )
        source_transform = self.graph.transform(source_frame)
        reference = (
            sim3_inverse(source_transform)
            @ self.graph.transform(target_frame).to(source_transform)
        )
        factor, _, holdout, diagnostics = self._chunk_stride_factor(
            pair_packet,
            edge_type="chunk_skip_dense_spherical",
            expected_target_index=1,
            validate_local_ba=False,
            reference_measurement=reference,
            allow_pose_fallback=False,
        )
        diagnostics.update(
            {
                "enabled": True,
                "matching_seed": int(seed),
                "source_window_id": int(previous.window_id),
                "target_window_id": int(current.window_id),
                "independent_from_sequential_and_overlap": True,
            }
        )
        return factor, holdout, diagnostics

    def _boundary_factor(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> tuple[DenseSphericalFactorBlock | None, dict[str, Any]]:
        matches = packet.boundary_matches or self._fallback_boundary_matches(packet)
        if matches is None:
            return None, {"reason": "boundary_matches_unavailable"}
        first_index, last_index = 0, len(packet.frame_ids) - 1
        source_depth = sample_erp_with_wrap(
            packet.observation.refined_depth[0, first_index], matches.source_uv
        )[..., 0]
        target_depth = sample_erp_with_wrap(
            packet.observation.refined_depth[0, last_index], matches.target_uv
        )[..., 0]
        source_valid = sample_erp_with_wrap(
            packet.finite_gaussian_mask[0, first_index].float(), matches.source_uv
        )[..., 0] >= 0.5
        target_valid = sample_erp_with_wrap(
            packet.finite_gaussian_mask[0, last_index].float(), matches.target_uv
        )[..., 0] >= 0.5
        source_sky = sample_erp_with_wrap(
            packet.sky_prob[0, first_index], matches.source_uv
        )[..., 0]
        target_sky = sample_erp_with_wrap(
            packet.sky_prob[0, last_index], matches.target_uv
        )[..., 0]
        keep = (
            source_valid
            & target_valid
            & (source_sky < self.sky_threshold)
            & (target_sky < self.sky_threshold)
            & torch.isfinite(source_depth)
            & torch.isfinite(target_depth)
            & (source_depth >= self.fibonacci_min_depth)
            & (target_depth >= self.fibonacci_min_depth)
            & (source_depth <= self.fibonacci_max_depth)
            & (target_depth <= self.fibonacci_max_depth)
            & (matches.top1_cosine >= self.min_match_cosine)
            & (matches.normalized_entropy <= self.max_match_entropy)
        )
        count = int(keep.sum())
        diagnostics = {
            "source_window_id": int(packet.window_id),
            "target_window_id": int(packet.window_id),
            "source_frame_id": int(packet.frame_ids[0]),
            "target_frame_id": int(packet.frame_ids[-1]),
            "raw_boundary_matches": int(matches.count),
            "hard_gated_boundary_matches": count,
            "sky_rejected": int(((source_sky >= self.sky_threshold) | (target_sky >= self.sky_threshold)).sum()),
            "weight_mode": "fibonacci_equal_after_hard_gates",
        }
        if count < self.min_dense_factors:
            diagnostics["reason"] = "insufficient_boundary_matches"
            return None, diagnostics
        identity = torch.eye(4, device=source_depth.device, dtype=source_depth.dtype)
        factor = DenseSphericalFactorBlock(
            source=int(packet.frame_ids[0]),
            target=int(packet.frame_ids[-1]),
            source_local_pose=identity,
            target_local_pose=identity.clone(),
            source_bearing=matches.source_bearing[keep].to(source_depth),
            target_bearing=matches.target_bearing[keep].to(source_depth),
            source_depth=source_depth[keep],
            target_depth=target_depth[keep],
            factor_weight=torch.ones(count, device=source_depth.device, dtype=source_depth.dtype),
            depth_factor_weight=self.depth_factor_weight,
            s2_huber_delta_deg=self.s2_huber_delta_deg,
            use_depth=True,
            edge_type="boundary_dense_spherical",
            **self._dense_factor_information_options(),
            metadata=diagnostics,
        )
        return factor, diagnostics

    @staticmethod
    def _boundary_local_pose_fallback_edge(
        packet: LocalGaussianWindowPacket,
        diagnostics: dict[str, Any],
    ) -> Sim3GraphEdge:
        """Keep a chunk connected when its optional first/last match block fails.

        The local S2 BA pose is already the accepted odometry estimate for the
        window.  It is therefore a safer bounded fallback than either dropping
        the window or leaving its end node disconnected.  Scale remains one in
        chunk coordinates; the owner Sim(3) applies the selected bridge scale
        exactly once when the chunk is placed in the global frame.
        """

        relative_pose = packet.local_poses_c2w[-1].detach()
        if relative_pose.shape != (4, 4) or not bool(
            torch.isfinite(relative_pose).all()
        ):
            raise RuntimeError("Local BA boundary pose must be a finite 4x4 matrix")
        measurement = sim3_from_components(
            1.0,
            relative_pose[:3, :3],
            relative_pose[:3, 3],
        )
        metadata = dict(diagnostics)
        metadata.update(
            {
                "accepted": True,
                "fallback_used": True,
                "fallback_reason": diagnostics.get("reason", "unknown"),
                "reason": "local_ba_pose_fallback",
                "source_frame_id": int(packet.frame_ids[0]),
                "target_frame_id": int(packet.frame_ids[-1]),
            }
        )
        return Sim3GraphEdge(
            source=int(packet.frame_ids[0]),
            target=int(packet.frame_ids[-1]),
            measurement_target_to_source=measurement,
            information_diag=measurement.new_tensor(
                [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.25]
            ),
            edge_type="boundary_local_ba_pose_fallback",
            metadata=metadata,
        )

    def _overlap_edge(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
    ) -> tuple[
        Sim3GraphEdge | None,
        DenseSphericalFactorBlock | None,
        CoincidentPanoramaFactor | None,
        dict[str, Any],
    ]:
        overlap = sorted(set(source.frame_ids) & set(target.frame_ids))
        if self.enforce_exact_overlap and len(overlap) != self.expected_overlap_frames:
            return None, None, None, {
                "reason": "unexpected_overlap_count",
                "overlap_frame_ids": overlap,
                "expected_overlap_frames": self.expected_overlap_frames,
            }
        if len(overlap) != 1:
            return None, None, None, {"reason": "single_overlap_required", "overlap_frame_ids": overlap}
        frame_id = int(overlap[0])
        source_index = source.frame_index(frame_id)
        target_index = target.frame_index(frame_id)
        source_depth = source.observation.refined_depth[0, source_index].detach()
        target_depth = target.observation.refined_depth[0, target_index].detach().to(source_depth)
        edge_seed = (
            self.fibonacci_seed
            + 1_000_003 * int(source.window_id)
            + 10_007 * int(target.window_id)
            + 101 * frame_id
        ) & 0x7FFFFFFF
        samples = sample_joint_valid_fibonacci_uv(
            source_depth,
            target_depth,
            count=self.max_overlap_points,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.fibonacci_min_depth,
            max_depth=self.fibonacci_max_depth,
            source_valid=source.finite_gaussian_mask[0, source_index].detach(),
            target_valid=target.finite_gaussian_mask[0, target_index].detach().to(source_depth.device),
            source_sky_probability=source.sky_prob[0, source_index].detach(),
            target_sky_probability=target.sky_prob[0, target_index].detach().to(source_depth.device),
            sky_threshold=self.sky_threshold,
            seed=edge_seed,
        )
        if int(samples.uv.shape[0]) < self.overlap_aligner.min_points:
            return None, None, None, {
                "reason": "insufficient_joint_fibonacci_support",
                "overlap_frame_ids": overlap,
                "overlap_points": int(samples.uv.shape[0]),
                "fibonacci_seed": edge_seed,
            }
        source_pose = source.local_poses_c2w[source_index].to(samples.bearing)
        target_pose = target.local_poses_c2w[target_index].to(samples.bearing)
        source_camera = samples.bearing * samples.source_depth[:, None]
        target_camera = samples.bearing * samples.target_depth[:, None]
        source_points = source_camera @ source_pose[:3, :3].transpose(0, 1) + source_pose[:3, 3]
        target_points = target_camera @ target_pose[:3, :3].transpose(0, 1) + target_pose[:3, 3]
        # A shared panorama supplies exact same-pixel correspondences.  The
        # joint finite/depth/sky mask above is the only hard gate; equal-solid-
        # angle Fibonacci samples enter Umeyama with equal weights.
        weights = torch.ones(
            int(target_points.shape[0]),
            device=target_points.device,
            dtype=target_points.dtype,
        )
        # Measurement maps target anchor coordinates into source anchor coordinates.
        alignment = self.overlap_aligner.align(target_points, source_points, weights)
        diagnostics = {
            "overlap_frame_ids": overlap,
            "source_frame_id": frame_id,
            "target_frame_id": frame_id,
            "overlap_points": int(weights.numel()),
            "overlap_residual": float(alignment.residual),
            "overlap_inlier_ratio": float(alignment.inlier_ratio),
            "fibonacci_seed": int(samples.seed),
            "fibonacci_longitude_phase": float(samples.longitude_phase),
            "descriptor_gate": False,
            "weight_mode": "fibonacci_equal_joint_geometry_mask",
        }
        if not alignment.accepted:
            diagnostics["reason"] = "overlap_alignment_rejected"
            return None, None, None, diagnostics
        information = source_points.new_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.75])
        information *= max(1.0, float(weights.numel()) * float(alignment.inlier_ratio))
        sim3_edge = Sim3GraphEdge(
                source=int(source.window_id),
                target=int(target.window_id),
                measurement_target_to_source=alignment.as_matrix().detach(),
                information_diag=information.detach(),
                edge_type="overlap",
                metadata=diagnostics,
            )
        dense_factor = DenseSphericalFactorBlock(
            source=int(source.window_id),
            target=int(target.window_id),
            source_local_pose=source_pose.detach(),
            target_local_pose=target_pose.detach(),
            source_bearing=samples.bearing.detach(),
            target_bearing=samples.bearing.detach(),
            source_depth=samples.source_depth.detach(),
            target_depth=samples.target_depth.detach(),
            factor_weight=weights.detach(),
            depth_factor_weight=self.depth_factor_weight,
            s2_huber_delta_deg=self.s2_huber_delta_deg,
            edge_type="overlap_dense_spherical",
            **self._dense_factor_information_options(),
            metadata=diagnostics,
        )
        shared_pose_factor = CoincidentPanoramaFactor(
            source=int(source.window_id),
            target=int(target.window_id),
            source_local_pose=source_pose.detach(),
            target_local_pose=target_pose.detach(),
            measured_source_to_target_rotation=torch.eye(3, device=source_pose.device, dtype=source_pose.dtype),
            center_weight=max(1.0, float(weights.numel()) * 0.25),
            rotation_weight=max(1.0, float(weights.numel()) * 0.25),
            edge_type="shared_frame_pose_consistency",
            metadata=diagnostics,
        )
        return sim3_edge, dense_factor, shared_pose_factor, diagnostics

    def _initial_transform(
        self,
        previous_id: int,
        edge: Sim3GraphEdge | None,
    ) -> torch.Tensor:
        previous = self.graph.transform(previous_id)
        if edge is None:
            return previous.clone()
        return previous @ edge.measurement_target_to_source.to(previous)

    def _register_chunk_stride_segments(
        self,
        packet: LocalGaussianWindowPacket,
        *,
        source_node: int,
        target_node: int,
    ) -> dict[str, Any]:
        """Register immutable two-frame pose owners for a size=4/stride=2 packet."""

        target_index = int(self.chunk_stride_target_index)
        if target_index >= len(packet.frame_ids):
            raise RuntimeError(
                "chunk_first_stride packet is missing its next anchor frame"
            )
        source_reference = packet.local_poses_c2w[0]
        target_reference = packet.local_poses_c2w[target_index]
        source_inverse = invert_c2w(source_reference)
        target_inverse = invert_c2w(target_reference)
        preserved = 0
        registered = 0
        max_rotation_mismatch = 0.0
        max_translation_mismatch = 0.0
        for index, frame_id in enumerate(packet.frame_ids):
            frame = int(frame_id)
            if index < target_index:
                owner = int(source_node)
                local_pose = source_inverse @ packet.local_poses_c2w[index]
            else:
                owner = int(target_node)
                local_pose = target_inverse @ packet.local_poses_c2w[index]
            local_pose = canonicalize_c2w(local_pose.detach())
            existing_owner = self.frame_pose_owner_node.get(frame)
            existing_pose = self.frame_local_pose_in_owner.get(frame)
            if existing_owner is not None:
                if int(existing_owner) != owner or existing_pose is None:
                    raise RuntimeError(
                        f"Frame {frame} already belongs to pose owner "
                        f"{existing_owner}, not {owner}"
                    )
                max_rotation_mismatch = max(
                    max_rotation_mismatch,
                    self._rotation_error_deg(
                        existing_pose[:3, :3].to(local_pose),
                        local_pose[:3, :3],
                    ),
                )
                max_translation_mismatch = max(
                    max_translation_mismatch,
                    float(
                        torch.linalg.norm(
                            existing_pose[:3, 3].to(local_pose)
                            - local_pose[:3, 3]
                        )
                        .detach()
                        .cpu()
                    ),
                )
                preserved += 1
                continue
            self.frame_pose_owner_node[frame] = owner
            self.frame_local_pose_in_owner[frame] = local_pose.clone()
            registered += 1
        return {
            "registered_frames": registered,
            "preserved_overlap_frames": preserved,
            "max_preserved_rotation_mismatch_deg": max_rotation_mismatch,
            "max_preserved_translation_mismatch": max_translation_mismatch,
        }

    def _chunk_stride_holdout_diagnostics(
        self,
        *,
        affected_node_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """Validate held-out sequential/skip edges touched by a transaction."""

        angular_errors: list[float] = []
        depth_errors: list[float] = []
        angular_ratios: list[float] = []
        depth_ratios: list[float] = []
        per_edge: list[dict[str, Any]] = []
        selected_nodes = (
            None
            if affected_node_ids is None
            else {int(node) for node in affected_node_ids}
        )
        for key in sorted(self._chunk_stride_holdouts):
            holdout = self._chunk_stride_holdouts[key]
            if selected_nodes is not None and {
                int(holdout.source),
                int(holdout.target),
            }.isdisjoint(selected_nodes):
                continue
            if holdout.source not in self.graph.nodes or holdout.target not in self.graph.nodes:
                per_edge.append(
                    {
                        "source": holdout.source,
                        "target": holdout.target,
                        "edge_type": holdout.edge_type,
                        "reason": "missing_graph_endpoint",
                        "accepted": True,
                        "quality_gating_enabled": False,
                    }
                )
                continue
            source_transform = self.graph.transform(holdout.source)
            relative = (
                sim3_inverse(source_transform)
                @ self.graph.transform(holdout.target).to(source_transform)
            )
            angular, depth = self._chunk_stride_alignment_errors(
                relative,
                holdout.source_bearing.to(relative),
                holdout.target_bearing.to(relative),
                holdout.source_depth.to(relative),
                holdout.target_depth.to(relative),
            )
            angular_median = float(angular.median().detach().cpu())
            depth_median = float(depth.median().detach().cpu())
            angular_ratio = angular_median / max(
                holdout.initial_angular_median_deg, 1.0e-4
            )
            depth_ratio = depth_median / max(
                holdout.initial_relative_depth_median, 1.0e-6
            )
            angular_errors.append(angular_median)
            depth_errors.append(depth_median)
            angular_ratios.append(angular_ratio)
            depth_ratios.append(depth_ratio)
            per_edge.append(
                {
                    "source": holdout.source,
                    "target": holdout.target,
                    "edge_type": holdout.edge_type,
                    "angular_median_deg": angular_median,
                    "relative_depth_median": depth_median,
                    "angular_worsening_ratio": angular_ratio,
                    "depth_worsening_ratio": depth_ratio,
                    "accepted": True,
                    "quality_gating_enabled": False,
                }
            )
        return {
            "enabled": bool(self.chunk_first_stride_graph),
            "factor_count": len(per_edge),
            "max_angular_median_deg": max(angular_errors, default=0.0),
            "max_relative_depth_median": max(depth_errors, default=0.0),
            "max_angular_worsening_ratio": max(angular_ratios, default=1.0),
            "max_depth_worsening_ratio": max(depth_ratios, default=1.0),
            "per_edge": per_edge,
            "accepted": True,
            "quality_gating_enabled": False,
        }

    def _refresh_geometry_updates(
        self,
        *,
        complete_snapshot: bool = True,
        affected_node_ids: set[int] | None = None,
        reason: str | None = None,
    ) -> None:
        if self.chunk_first_stride_graph:
            updates: dict[int, FrameGeometryUpdate] = {}
            window_transforms = self._window_anchor_transforms()
            window_scales = {
                int(window_id): float(
                    sim3_components(transform)[0].detach().cpu()
                )
                for window_id, transform in window_transforms.items()
            }
            window_input_scales = {
                int(window_id): float(
                    self.packets[int(window_id)].metadata.get(
                        "global_alignment_local_scale", 1.0
                    )
                )
                for window_id in self.window_order
                if int(window_id) in self.packets
            }
            selected_nodes = (
                None
                if affected_node_ids is None
                else {int(node) for node in affected_node_ids}
            )
            selected_frames = [
                int(frame)
                for frame in sorted(self.frame_pose_owner_node)
                if complete_snapshot
                or selected_nodes is None
                or int(self.frame_pose_owner_node[int(frame)]) in selected_nodes
                or len(self.window_order) <= 1
            ]
            for frame in selected_frames:
                owner_node = int(self.frame_pose_owner_node[frame])
                local_pose = self.frame_local_pose_in_owner[frame]
                if owner_node not in self.graph.nodes:
                    raise RuntimeError(
                        f"Frame {frame} references missing pose owner node {owner_node}"
                    )
                node_transform = self.graph.transform(owner_node).to(local_pose)
                node_scale = float(
                    sim3_components(node_transform)[0].detach().cpu()
                )
                pose = apply_sim3_to_c2w(node_transform, local_pose)
                if not bool(torch.isfinite(pose).all()) or not math.isfinite(
                    node_scale
                ):
                    raise RuntimeError(
                        f"Frame {frame} produced non-finite canonical geometry"
                    )
                owner_window = int(
                    self.frame_owner_window.get(
                        frame,
                        min(self.frame_windows.get(frame, {0})),
                    )
                )
                depth_owner = int(
                    self.frame_depth_owner_window.get(frame, owner_window)
                )
                depth_scales_by_window = {
                    int(candidate): float(window_scales[int(candidate)])
                    * float(window_input_scales.get(int(candidate), 1.0))
                    for candidate in self.frame_windows.get(frame, {owner_window})
                    if int(candidate) in window_scales
                }
                updates[frame] = FrameGeometryUpdate(
                    frame_id=frame,
                    pose_c2w=pose.detach().cpu().float(),
                    depth_scale=(
                        node_scale
                        * float(window_input_scales.get(owner_window, 1.0))
                    ),
                    owner_window_id=owner_window,
                    depth_owner_window_id=depth_owner,
                    pose_owner_node_id=owner_node,
                    depth_scales_by_window=depth_scales_by_window,
                )
            if complete_snapshot:
                self._geometry_updates = dict(updates)
            else:
                self._geometry_updates.update(updates)
            if not updates and not complete_snapshot:
                return
            self._geometry_revision += 1
            batch_updates = dict(updates)
            batch_complete = bool(complete_snapshot)
            batch_affected = set(selected_nodes or ())
            if self._pending_geometry_batch is not None:
                previous_batch = self._pending_geometry_batch
                batch_affected.update(previous_batch.affected_node_ids)
                if previous_batch.complete_snapshot or batch_complete:
                    batch_complete = True
                    batch_updates = dict(self._geometry_updates)
                else:
                    batch_updates = {
                        **previous_batch.updates,
                        **batch_updates,
                    }
            self._pending_geometry_batch = FrameGeometryUpdateBatch(
                revision=int(self._geometry_revision),
                complete_snapshot=batch_complete,
                updates=batch_updates,
                affected_node_ids=tuple(
                    sorted(
                        batch_affected
                        if batch_affected
                        else self.graph.nodes
                    )
                ),
                reason=(
                    reason
                    or (
                        "chunk_first_stride_complete_geometry_refresh"
                        if batch_complete
                        else "chunk_first_stride_incremental_geometry_refresh"
                    )
                ),
            )
            return
        if self.boundary_frame_graph:
            updates: dict[int, FrameGeometryUpdate] = {}
            window_transforms = self._window_anchor_transforms()
            window_scales = {
                int(window_id): float(
                    sim3_components(window_transforms[int(window_id)])[0].detach().cpu()
                )
                for window_id in self.window_order
                if int(window_id) in window_transforms
            }
            window_input_scales = {
                int(window_id): float(
                    self.packets[int(window_id)].metadata.get(
                        "global_alignment_local_scale", 1.0
                    )
                )
                for window_id in self.window_order
            }
            window_depth_scales = {
                int(window_id): window_scales[int(window_id)]
                * window_input_scales[int(window_id)]
                for window_id in self.window_order
            }
            for window_id in self.window_order:
                packet = self.packets[window_id]
                anchor_node = self.window_anchor_nodes[int(window_id)]
                anchor_transform = window_transforms[int(window_id)].to(packet.local_poses_c2w)
                left_anchored_poses = packet.global_poses(anchor_transform)
                for index, frame_id in enumerate(packet.frame_ids):
                    frame = int(frame_id)
                    owner = int(self.frame_owner_window.get(frame, int(window_id)))
                    if frame in self.graph.nodes:
                        node_transform = self.graph.transform(frame)
                        node_scale, node_rotation, node_translation = sim3_components(node_transform)
                        pose = torch.eye(4, device=node_transform.device, dtype=node_transform.dtype)
                        pose[:3, :3] = node_rotation
                        pose[:3, 3] = node_translation
                        depth_scale = (
                            float(node_scale.detach().cpu())
                            * window_input_scales.get(owner, 1.0)
                        )
                    else:
                        pose = left_anchored_poses[index]
                        depth_scale = float(window_depth_scales[owner])
                    updates[frame] = FrameGeometryUpdate(
                        frame_id=frame,
                        pose_c2w=pose.detach().cpu().float(),
                        depth_scale=depth_scale,
                        owner_window_id=owner,
                        depth_owner_window_id=owner,
                        depth_scales_by_window={
                            int(candidate): float(window_depth_scales[int(candidate)])
                            for candidate in self.frame_windows.get(frame, {owner})
                            if int(candidate) in window_depth_scales
                        },
                    )
            self._geometry_updates.update(updates)
            return

        updates: dict[int, FrameGeometryUpdate] = {}
        window_scales = {
            int(window_id): float(sim3_components(self.graph.transform(window_id))[0].detach().cpu())
            for window_id in self.window_order
        }
        for window_id in self.window_order:
            packet = self.packets[window_id]
            transform = self.graph.transform(window_id).to(packet.local_poses_c2w)
            poses = packet.global_poses(transform)
            for frame_id, pose in zip(packet.frame_ids, poses):
                # Later overlapping windows have the freshest multi-view estimate.
                depth_owner = int(
                    self.frame_depth_owner_window.get(
                        int(frame_id), min(self.frame_windows.get(int(frame_id), {int(window_id)}))
                    )
                )
                updates[int(frame_id)] = FrameGeometryUpdate(
                    frame_id=int(frame_id),
                    pose_c2w=pose.detach().cpu().float(),
                    depth_scale=float(window_scales.get(depth_owner, window_scales[int(window_id)])),
                    owner_window_id=int(window_id),
                    depth_owner_window_id=depth_owner,
                    depth_scales_by_window={
                        int(owner): float(window_scales[int(owner)])
                        for owner in self.frame_windows.get(int(frame_id), {int(window_id)})
                        if int(owner) in window_scales
                    },
                )
                self.frame_owner_window[int(frame_id)] = int(window_id)
        self._geometry_updates.update(updates)

    def _refresh_pose_updates(self) -> None:
        """Compatibility wrapper for older internal call sites."""

        self._refresh_geometry_updates()

    def pop_frame_geometry_updates(self) -> dict[int, FrameGeometryUpdate]:
        batch = self._pending_geometry_batch
        updates = (
            dict(batch.updates)
            if batch is not None
            else dict(self._geometry_updates)
        )
        self._geometry_updates.clear()
        self._pending_geometry_batch = None
        return updates

    def pop_frame_geometry_update_batch(self) -> FrameGeometryUpdateBatch | None:
        batch = self._pending_geometry_batch
        self._pending_geometry_batch = None
        self._geometry_updates.clear()
        return batch

    def pop_pose_updates(self) -> dict[int, torch.Tensor]:
        """Backward-compatible pose-only view of pending geometry updates."""

        return {
            int(frame_id): update.pose_c2w
            for frame_id, update in self.pop_frame_geometry_updates().items()
        }

    def _run_map_optimization(self, window_id: int, frame_ids: tuple[int, ...], steps: int) -> dict[str, float]:
        if self.mapper is None or int(steps) <= 0:
            return {}
        self.mapper.optimizer = self.map.make_optimizer(
            lr=float(self.config.get("map_optimization", {}).get("lr", 2.0e-3))
        )
        # Mapper keyframes are registered by prepare_spherical_selfi_window().
        # They are durable input state, not part of the pose proposal.  Capture
        # the rollback boundary only after registration so rejecting a proposal
        # never tries to restore a pre-registration keyframe topology.
        backend_transaction = None
        if self.chunk_first_stride_graph:
            self._last_mapper_committed_state_diagnostic = None
        try:
            live_packet = self._optimization_packets.get(int(window_id))
            if live_packet is not None and not self.boundary_frame_graph:
                graph_node = (
                    self.window_anchor_nodes[int(window_id)]
                    if self.boundary_frame_graph
                    else int(window_id)
                )
                window_scale, _, _ = sim3_components(self.graph.transform(graph_node))
                for frame_index, frame_id in enumerate(live_packet.frame_ids):
                    self.mapper.set_spherical_selfi_observation_geometry(
                        int(frame_id),
                        target_depth_local=live_packet.observation.refined_depth[0, frame_index],
                        depth_scale=float(window_scale.detach().cpu()),
                        owner_window_id=int(window_id),
                        depth_confidence=(
                            live_packet.observation.confidence[0, frame_index]
                            * live_packet.finite_gaussian_mask[0, frame_index].float()
                        ),
                        sky_mask=live_packet.sky_mask[0, frame_index],
                    )
            prepared = self.mapper.prepare_spherical_selfi_window(frame_ids)
            if prepared != len(frame_ids):
                raise RuntimeError(
                    f"window {window_id} has {prepared}/{len(frame_ids)} registered RGB observations"
                )
            if self.chunk_first_stride_graph:
                backend_transaction = self._snapshot_boundary_transaction()
            optimized_frame_ids = tuple(
                dict.fromkeys(int(frame_id) for frame_id in frame_ids)
            )
            active_owner_window_ids = self._map_optimization_window_ids(
                int(window_id)
            )
            fixed_frame_ids: list[int] = []
            gaussian_only = bool(
                self.two_stage_map_optimization_enabled
                and live_packet is not None
                and live_packet.metadata.get(
                    "two_stage_candidate_prepared", False
                )
            )
            gaussian_config = self.postfusion_gaussian_mapping_config
            replay_frame_ids: tuple[int, ...] = ()
            validation_frame_ids: tuple[int, ...] = ()
            if gaussian_only:
                primary = list(optimized_frame_ids)
                replay_frame_ids = tuple(
                    int(frame_id)
                    for frame_id in sorted(self.mapper.observations)
                    if int(frame_id) not in set(primary)
                )
                if len(primary) <= 4:
                    validation_frame_ids = tuple(primary)
                elif primary:
                    selected_indices = tuple(
                        dict.fromkeys(
                            int(round(index * (len(primary) - 1) / 3.0))
                            for index in range(4)
                        )
                    )
                    validation_frame_ids = tuple(
                        primary[index] for index in selected_indices
                    )
            settings = {
                "gaussian_lr": float(self.map_optimize_config.get("gaussian_lr", self.map_optimize_config.get("lr", 2.0e-3))),
                "separate_gaussian_lrs": bool(self.map_optimize_config.get("separate_gaussian_lrs", False)),
                "xyz_lr": float(self.map_optimize_config.get("xyz_lr", 5.0e-4)),
                "feature_lr": float(self.map_optimize_config.get("feature_lr", 2.0e-3)),
                "sh_rest_lr": float(self.map_optimize_config.get("sh_rest_lr", 1.0e-4)),
                "opacity_lr": float(self.map_optimize_config.get("opacity_lr", 1.0e-3)),
                "scaling_lr": float(self.map_optimize_config.get("scaling_lr", 1.0e-4)),
                "rotation_lr": float(self.map_optimize_config.get("rotation_lr", 1.0e-4)),
                "scale_gaussian_parameter_updates": bool(
                    self.map_optimize_config.get("scale_gaussian_parameter_updates", False)
                ),
                "pose_lr": float(self.map_optimize_config.get("pose_lr", 1.0e-3)),
                "pose_refine_enable": (
                    False
                    if gaussian_only
                    else bool(
                        self.map_optimize_config.get(
                            "pose_refine_enable", True
                        )
                    )
                ),
                "pose_prior_weight": (
                    0.0
                    if self.map_optimize_photometric_only
                    else float(self.map_optimize_config.get("pose_prior_weight", 0.0))
                ),
                "pose_grad_clip": float(self.map_optimize_config.get("pose_grad_clip", 1.0e-3)),
                "visible_neighbor_lr_scale": float(self.map_optimize_config.get("visible_neighbor_lr_scale", 0.1)),
                "sampler_seed": int(self.map_optimize_config.get("seed", 123)) + int(window_id),
                "fixed_pose_frame_ids": fixed_frame_ids,
                "active_owner_window_ids": active_owner_window_ids,
                "photometric_only": self.map_optimize_photometric_only,
                "optimize_skybox": self.map_optimize_skybox,
                "replay_frame_ids": replay_frame_ids,
                "replay_every": int(
                    gaussian_config.get("replay_every", 2)
                ),
                "validation_frame_ids": validation_frame_ids,
                "validation_interval": int(
                    gaussian_config.get("validation_interval", 25)
                ),
                "restore_best_gaussians": bool(
                    gaussian_config.get("restore_best_gaussians", True)
                ),
                "holdout_fraction": float(
                    gaussian_config.get("holdout_fraction", 0.20)
                ),
                "alpha_threshold": float(
                    gaussian_config.get("alpha_threshold", 0.05)
                ),
                "photometric_loss_mode": str(
                    gaussian_config.get(
                        "photometric_loss_mode", "charbonnier_dssim"
                    )
                ),
                "charbonnier_weight": float(
                    gaussian_config.get("charbonnier_weight", 0.85)
                ),
                "dssim_weight": float(
                    gaussian_config.get("dssim_weight", 0.15)
                ),
            }
            if gaussian_only:
                metrics = self.mapper.optimize_spherical_selfi_gaussian_only(
                    window_id=int(window_id),
                    frame_ids=list(optimized_frame_ids),
                    iters=int(steps),
                    settings=settings,
                )
            else:
                metrics = self.mapper.optimize_spherical_selfi_window(
                    window_id=int(window_id),
                    frame_ids=list(optimized_frame_ids),
                    iters=int(steps),
                    settings=settings,
                    extra_loss_fn=(
                        None
                        if self.map_optimize_photometric_only
                        else lambda trainable_pose_ids: self._joint_graph_pose_loss(
                            int(window_id), trainable_pose_ids
                        )
                    ),
                )
            metrics["pose_refine_enabled"] = float(settings["pose_refine_enable"])
            metrics["gaussian_only_mapping"] = float(gaussian_only)
            metrics["optimized_frame_count"] = float(len(optimized_frame_ids))
            metrics["active_owner_window_count"] = float(
                len(active_owner_window_ids)
            )
            metrics["photometric_only"] = float(
                self.map_optimize_photometric_only
            )
            if float(metrics.get("window_rollback", 0.0)) == 0.0:
                if gaussian_only:
                    self.mapper.commit_spherical_selfi_window()
                    return metrics
                try:
                    self._synchronize_joint_optimized_window(
                        int(window_id),
                        optimized_frame_ids=optimized_frame_ids,
                    )
                except (RuntimeError, ValueError, KeyError) as exc:
                    if backend_transaction is not None:
                        self._restore_boundary_transaction(backend_transaction)
                    if self.geometry_rollback_on_failure:
                        self.mapper.rollback_spherical_selfi_window()
                    else:
                        self.mapper.commit_spherical_selfi_window()
                    metrics["steps"] = 0.0
                    metrics["window_rollback"] = 1.0
                    metrics["geometry_sync_failed"] = 1.0
                    self.mapper.stats.notes.append(
                        f"spherical-Selfi geometry synchronization rolled back: {exc!r}"
                    )
                else:
                    self.mapper.commit_spherical_selfi_window()
                    committed = self._last_mapper_committed_state_diagnostic
                    if committed:
                        metrics["committed_pose_revision"] = float(
                            committed.get(
                                "committed_revision",
                                self._pose_state_revision,
                            )
                        )
                        metrics["committed_render_views"] = float(
                            committed.get("view_count", 0)
                        )
                        metrics["committed_render_loss"] = float(
                            committed.get("mean_loss", 0.0)
                        )
                        metrics["committed_render_psnr"] = float(
                            committed.get("mean_psnr", 0.0)
                        )
                        metrics["committed_tail_render_loss"] = float(
                            committed.get("tail_mean_loss", 0.0)
                        )
            return metrics
        except (RuntimeError, ValueError, KeyError) as exc:
            if backend_transaction is not None:
                self._restore_boundary_transaction(backend_transaction)
            if self.geometry_rollback_on_failure:
                self.mapper.rollback_spherical_selfi_window()
            else:
                self.mapper.commit_spherical_selfi_window()
            self.mapper.stats.notes.append(f"spherical-Selfi map optimization skipped: {exc!r}")
            return {"steps": 0.0, "loss": 0.0, "window_rollback": 1.0}
        finally:
            self._optimization_packets.pop(int(window_id), None)

    def _joint_graph_pose_loss(self, window_id: int, trainable_pose_ids: set[int]) -> torch.Tensor:
        """Evaluate dense graph correspondences on differentiable SE(3) camera poses."""

        if self.mapper is None:
            return self.map.get_xyz.new_zeros(())
        device, dtype = self.map.get_xyz.device, self.map.get_xyz.dtype
        lambda_s2 = float(self.map_optimize_config.get("s2_loss_weight", 0.1))
        lambda_depth = float(self.map_optimize_config.get("match_depth_loss_weight", 0.01))
        costs: list[torch.Tensor] = []

        def camera_pose(frame_id: int) -> torch.Tensor | None:
            pose_delta = self.mapper.pose_deltas.get(int(frame_id))
            if pose_delta is None:
                return None
            pose = pose_delta()
            return pose if int(frame_id) in trainable_pose_ids else pose.detach()

        packet = self.packets.get(int(window_id))
        window_frames = set() if packet is None else {int(value) for value in packet.frame_ids}
        for factor in self.graph.edges:
            if not isinstance(factor, DenseSphericalFactorBlock):
                continue
            source_frame_id = factor.metadata.get("source_frame_id")
            target_frame_id = factor.metadata.get("target_frame_id")
            if source_frame_id is None or target_frame_id is None:
                continue
            if (
                int(source_frame_id) not in window_frames
                and int(target_frame_id) not in window_frames
                and int(source_frame_id) not in trainable_pose_ids
                and int(target_frame_id) not in trainable_pose_ids
            ):
                continue
            source_pose = camera_pose(int(source_frame_id))
            target_pose = camera_pose(int(target_frame_id))
            if source_pose is None or target_pose is None:
                continue
            source_scale, _, _ = sim3_components(self.graph.transform(int(factor.source)).detach())
            target_scale, _, _ = sim3_components(self.graph.transform(int(factor.target)).detach())
            source_bearing = factor.source_bearing.to(device=device, dtype=dtype)
            target_bearing = factor.target_bearing.to(device=device, dtype=dtype)
            source_depth = factor.source_depth.to(device=device, dtype=dtype) * source_scale.to(device=device, dtype=dtype)
            expected_target_depth = factor.target_depth.to(device=device, dtype=dtype) * target_scale.to(device=device, dtype=dtype)
            weight = factor.factor_weight.to(device=device, dtype=dtype).clamp_min(0.0)
            source_pose = source_pose.to(device=device, dtype=dtype)
            target_pose = target_pose.to(device=device, dtype=dtype)
            source_camera = source_bearing * source_depth[:, None]
            world = source_camera @ source_pose[:3, :3].transpose(0, 1) + source_pose[:3, 3]
            target_camera = (world - target_pose[:3, 3]) @ target_pose[:3, :3]
            predicted_depth = torch.linalg.norm(target_camera, dim=-1).clamp_min(1.0e-8)
            predicted_bearing = target_camera / predicted_depth[:, None]
            s2 = s2_log_tangent_coordinates(target_bearing, predicted_bearing)
            s2_norm = torch.linalg.norm(s2, dim=-1)
            s2_delta = torch.as_tensor(
                torch.pi * float(factor.s2_huber_delta_deg) / 180.0,
                device=device,
                dtype=dtype,
            ).clamp_min(1.0e-8)
            s2_huber = torch.where(
                s2_norm <= s2_delta,
                0.5 * s2_norm.square(),
                s2_delta * (s2_norm - 0.5 * s2_delta),
            )
            factor_loss = lambda_s2 * (weight * s2_huber).sum() / weight.sum().clamp_min(1.0e-8)
            if factor.use_depth and lambda_depth > 0.0:
                depth_residual = torch.log(predicted_depth / expected_target_depth.clamp_min(1.0e-8))
                depth_delta = depth_residual.new_tensor(0.25)
                depth_abs = depth_residual.abs()
                depth_huber = torch.where(
                    depth_abs <= depth_delta,
                    0.5 * depth_residual.square(),
                    depth_delta * (depth_abs - 0.5 * depth_delta),
                )
                factor_loss = factor_loss + lambda_depth * (
                    weight * depth_huber
                ).sum() / weight.sum().clamp_min(1.0e-8)
            costs.append(factor_loss)
        return torch.stack(costs).mean() if costs else torch.zeros((), device=device, dtype=dtype)

    def run_pending_map_optimization(self) -> dict[str, float]:
        """Run low-rate map updates after the system registered window images."""

        if not self._pending_map_optimization and not self._pending_seam_owner_windows:
            return {}
        pending = list(self._pending_map_optimization)
        self._pending_map_optimization.clear()
        last_metrics: dict[str, float] = {}
        rolled_back = False
        for window_id, frame_ids, steps in pending:
            last_metrics = self._run_map_optimization(window_id, frame_ids, steps)
            rolled_back = rolled_back or bool(
                float(last_metrics.get("window_rollback", 0.0)) > 0.0
            )
        if self._pending_seam_owner_windows:
            seam_owners = set(self._pending_seam_owner_windows)
            self._pending_seam_owner_windows.clear()
            deduplicated = 0
            if self.loop_seam_dedup_enabled and not rolled_back:
                deduplicated = self.fusion.deduplicate_owner_neighborhood(
                    seam_owners
                )
                if deduplicated > 0 and self.mapper is not None:
                    self.mapper.optimizer = self.map.make_optimizer(
                        lr=float(
                            self.config.get("map_optimization", {}).get(
                                "lr", 2.0e-3
                            )
                        )
                    )
                    self.mapper.stats.n_anchors = self.map.anchor_count()
            last_metrics["seam_deduplicated"] = float(deduplicated)
            last_metrics["seam_refinement_rollback"] = float(rolled_back)
        return last_metrics

    def _enqueue_map_optimization(
        self,
        window_id: int,
        frame_ids: tuple[int, ...],
        steps: int,
    ) -> None:
        if int(steps) <= 0:
            return
        window = int(window_id)
        selected_frame_ids = tuple(int(value) for value in frame_ids)
        if self.map_optimize_recent_windows > 1:
            recent_frame_ids = self._map_optimization_frame_ids(window)
            if recent_frame_ids:
                selected_frame_ids = recent_frame_ids
        for index, (queued_window, queued_frames, queued_steps) in enumerate(
            self._pending_map_optimization
        ):
            if int(queued_window) == window:
                self._pending_map_optimization[index] = (
                    window,
                    tuple(
                        dict.fromkeys(
                            [int(value) for value in queued_frames]
                            + [int(value) for value in selected_frame_ids]
                        )
                    ),
                    max(int(queued_steps), int(steps)),
                )
                return
        self._pending_map_optimization.append(
            (window, selected_frame_ids, int(steps))
        )

    def _map_optimization_window_ids(self, window_id: int) -> tuple[int, ...]:
        window = int(window_id)
        try:
            index = self.window_order.index(window)
        except ValueError:
            return (window,) if window in self.packets else ()
        start = max(0, index - self.map_optimize_recent_windows + 1)
        return tuple(int(value) for value in self.window_order[start : index + 1])

    def _map_optimization_frame_ids(self, window_id: int) -> tuple[int, ...]:
        frame_ids: list[int] = []
        for selected_window in self._map_optimization_window_ids(window_id):
            packet = self._optimization_packets.get(int(selected_window))
            if packet is None:
                packet = self.packets.get(int(selected_window))
            if packet is None:
                continue
            for frame_id in packet.frame_ids:
                value = int(frame_id)
                if value not in frame_ids:
                    frame_ids.append(value)
        return tuple(frame_ids)

    def _loop_neighborhood_windows(
        self,
        accepted_loops: list[PanoramaLoopVerification],
    ) -> list[int]:
        selected: set[int] = set()
        for result in accepted_loops:
            selected.update(
                [int(result.source_window_id), int(result.target_window_id)]
            )
        if self.hierarchical_submaps_enabled:
            submap_ids = {
                int(self.window_to_submap[window_id])
                for window_id in selected
                if window_id in self.window_to_submap
            }
            expanded = {
                candidate
                for submap_id in submap_ids
                for candidate in range(
                    max(0, submap_id - self.loop_neighborhood_submap_radius),
                    min(
                        len(self.submaps),
                        submap_id + self.loop_neighborhood_submap_radius + 1,
                    ),
                )
            }
            selected.update(
                int(window_id)
                for submap_id in expanded
                for window_id in self.submaps[submap_id].window_ids
            )
        else:
            index_by_window = {
                int(window_id): index for index, window_id in enumerate(self.window_order)
            }
            for window_id in tuple(selected):
                if window_id not in index_by_window:
                    continue
                index = index_by_window[window_id]
                for offset in (-1, 1):
                    neighbor = index + offset
                    if 0 <= neighbor < len(self.window_order):
                        selected.add(int(self.window_order[neighbor]))
        return [window_id for window_id in self.window_order if int(window_id) in selected]

    def _packet_variants(
        self,
        window_id: int,
        *,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
    ) -> list[LocalGaussianWindowPacket]:
        variants: list[LocalGaussianWindowPacket] = []
        for candidate in (
            self.packets.get(int(window_id)),
            self._optimization_packets.get(int(window_id)),
            self._last_full_packet,
            *extra_packets,
        ):
            if candidate is None or int(candidate.window_id) != int(window_id):
                continue
            if all(id(candidate) != id(existing) for existing in variants):
                variants.append(candidate)
        return variants

    def _refresh_factor_local_poses(self, affected_windows: set[int]) -> None:
        for factor in self.graph.edges:
            if not isinstance(factor, (DenseSphericalFactorBlock, CoincidentPanoramaFactor)):
                continue
            if self.boundary_frame_graph:
                source_window = factor.metadata.get("source_window_id")
                target_window = factor.metadata.get("target_window_id")
                if source_window is not None and int(source_window) in affected_windows:
                    factor.source_local_pose = torch.eye(
                        4,
                        device=factor.source_local_pose.device,
                        dtype=factor.source_local_pose.dtype,
                    )
                if target_window is not None and int(target_window) in affected_windows:
                    factor.target_local_pose = torch.eye(
                        4,
                        device=factor.target_local_pose.device,
                        dtype=factor.target_local_pose.dtype,
                    )
                continue
            if int(factor.source) in affected_windows:
                packet = self.packets[int(factor.source)]
                frame_id = int(factor.metadata.get("source_frame_id", packet.frame_ids[0]))
                if frame_id in packet.frame_ids:
                    factor.source_local_pose = packet.local_poses_c2w[
                        packet.frame_index(frame_id)
                    ].detach()
            if int(factor.target) in affected_windows:
                packet = self.packets[int(factor.target)]
                frame_id = int(factor.metadata.get("target_frame_id", packet.frame_ids[0]))
                if frame_id in packet.frame_ids:
                    factor.target_local_pose = packet.local_poses_c2w[
                        packet.frame_index(frame_id)
                    ].detach()

    def _validate_pose_round_trip(
        self,
        transform: torch.Tensor,
        local_pose: torch.Tensor,
        global_pose: torch.Tensor,
        *,
        frame_id: int,
        window_id: int,
    ) -> None:
        if not self.geometry_validation_enabled:
            return
        reconstructed = apply_sim3_to_c2w(transform.to(local_pose), local_pose)
        if not bool(torch.isfinite(reconstructed).all()):
            raise RuntimeError(f"non-finite Sim(3) pose round-trip for window={window_id} frame={frame_id}")
        if not torch.allclose(
            reconstructed,
            global_pose.to(reconstructed),
            atol=self.geometry_tolerance,
            rtol=self.geometry_tolerance,
        ):
            error = float((reconstructed - global_pose.to(reconstructed)).abs().max().detach().cpu())
            raise RuntimeError(
                f"Sim(3) pose round-trip failed for window={window_id} frame={frame_id}: max_error={error:.3e}"
            )

    def _synchronize_joint_optimized_window(
        self,
        window_id: int,
        *,
        optimized_frame_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Transactionally rebase optimized SE(3) poses while graph scale stays authoritative."""

        if self.chunk_first_stride_graph:
            self._synchronize_chunk_stride_optimized_window(
                window_id,
                optimized_frame_ids=optimized_frame_ids,
            )
            return
        if self.boundary_frame_graph:
            self._synchronize_boundary_optimized_window(window_id)
            return
        if self.mapper is None or int(window_id) not in self.packets:
            return
        packet = self.packets[int(window_id)]
        optimized_by_frame: dict[int, torch.Tensor] = {}
        for frame_id in packet.frame_ids:
            pose = self.mapper.refined_pose_c2w(int(frame_id))
            if pose is None:
                raise RuntimeError(f"missing optimized pose for frame {frame_id}")
            if tuple(pose.shape) != (4, 4) or not bool(torch.isfinite(pose).all()):
                raise RuntimeError(f"invalid optimized pose for frame {frame_id}")
            optimized_by_frame[int(frame_id)] = canonicalize_c2w(pose.float())

        old_nodes = {node: value.clone() for node, value in self.graph.nodes.items()}
        geometry_snapshot = dict(self._geometry_updates)
        affected_windows = {
            int(owner)
            for frame_id in optimized_by_frame
            for owner in self.frame_windows.get(int(frame_id), {int(window_id)})
        }
        packet_snapshots = {
            id(variant): (
                variant,
                variant.local_poses_c2w.clone(),
                variant.observation,
            )
            for owner in affected_windows
            for variant in self._packet_variants(owner)
        }

        try:
            self.graph.nodes = {
                int(node): canonicalize_sim3(transform)
                for node, transform in self.graph.nodes.items()
            }
            current_sim3 = self.graph.transform(int(window_id))
            scale, _, _ = sim3_components(current_sim3)
            anchor_pose = optimized_by_frame[int(packet.frame_ids[0])].to(current_sim3)
            rebased = sim3_from_components(
                scale,
                anchor_pose[:3, :3],
                anchor_pose[:3, 3],
            )
            self.graph.nodes[int(window_id)] = rebased.detach()

            for owner in affected_windows:
                transform = self.graph.transform(owner)
                for variant in self._packet_variants(owner):
                    local_poses = variant.local_poses_c2w.clone()
                    changed = False
                    for frame_id, global_pose in optimized_by_frame.items():
                        if frame_id not in variant.frame_ids:
                            continue
                        index = variant.frame_index(frame_id)
                        local = rebase_c2w_to_sim3_anchor(
                            transform.to(local_poses), global_pose.to(local_poses)
                        )
                        if owner == int(window_id) and index == 0:
                            local = torch.eye(4, device=local.device, dtype=local.dtype)
                        self._validate_pose_round_trip(
                            transform,
                            local,
                            global_pose,
                            frame_id=frame_id,
                            window_id=owner,
                        )
                        local_poses[index] = local
                        changed = True
                    if changed:
                        variant.local_poses_c2w = local_poses.detach()
                        variant.observation = variant.observation.with_geometry(
                            poses_c2w=local_poses.unsqueeze(0).to(variant.observation.poses_c2w)
                        )

            self._refresh_factor_local_poses(affected_windows)
            graph_reference = {node: value.clone() for node, value in self.graph.nodes.items()}
            active = self.window_order[-self.recent_optimization_windows :]
            graph_result = self.graph.optimize(active)
            if not torch.isfinite(torch.tensor(graph_result.final_objective)):
                raise RuntimeError("graph optimization produced a non-finite objective")
            if graph_result.final_objective > graph_result.initial_objective + 1.0e-10:
                raise RuntimeError("graph optimization increased its robust objective")
            self.fusion.apply_owner_corrections(graph_reference, self.graph.nodes)
            self._refresh_geometry_updates()
        except Exception:
            self.graph.nodes = {node: value.clone() for node, value in old_nodes.items()}
            self._geometry_updates = geometry_snapshot
            for variant, local_poses, observation in packet_snapshots.values():
                variant.local_poses_c2w = local_poses
                variant.observation = observation
            self._refresh_factor_local_poses(affected_windows)
            raise

    def _synchronize_chunk_packet_variants(
        self,
        affected_windows: set[int],
        *,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
        revision: int | None = None,
    ) -> int:
        """Make every retained packet mirror the canonical rigid segments."""

        window_transforms = self._window_anchor_transforms()
        synchronized = 0
        for window_id in sorted(int(value) for value in affected_windows):
            if window_id not in window_transforms:
                continue
            owner_transform = window_transforms[window_id]
            for variant in self._packet_variants(
                window_id,
                extra_packets=extra_packets,
            ):
                local_poses = variant.local_poses_c2w.detach().clone()
                changed = False
                for index, frame_id in enumerate(variant.frame_ids):
                    frame = int(frame_id)
                    owner_node = self.frame_pose_owner_node.get(frame)
                    canonical_local = self.frame_local_pose_in_owner.get(frame)
                    if (
                        owner_node is None
                        or canonical_local is None
                        or int(owner_node) not in self.graph.nodes
                    ):
                        continue
                    global_pose = apply_sim3_to_c2w(
                        self.graph.transform(int(owner_node)).to(canonical_local),
                        canonical_local,
                    )
                    local_poses[index] = canonicalize_c2w(
                        rebase_c2w_to_sim3_anchor(
                            owner_transform.to(global_pose),
                            global_pose,
                        )
                    ).to(local_poses)
                    changed = True
                if changed:
                    variant.local_poses_c2w = local_poses.detach()
                    variant.observation = variant.observation.with_geometry(
                        poses_c2w=local_poses.unsqueeze(0).to(
                            variant.observation.poses_c2w
                        )
                    )
                    if variant.anchor_observation is not None:
                        variant.anchor_observation = replace(
                            variant.anchor_observation,
                            local_poses_c2w=local_poses.unsqueeze(0).to(
                                variant.anchor_observation.local_poses_c2w
                            ),
                        )
                    if revision is not None:
                        variant.metadata["canonical_pose_revision"] = int(
                            revision
                        )
                    synchronized += 1
        return synchronized

    def _pose_state_affected_sets(
        self,
        affected_node_ids: set[int],
        *,
        affected_submap_ids: set[int] | None = None,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        nodes = {int(value) for value in affected_node_ids}
        submaps = {
            int(value) for value in (affected_submap_ids or set())
        }
        windows: set[int] = set()
        for submap_id, record in self.submaps.items():
            if (
                int(submap_id) in submaps
                or not nodes.isdisjoint(
                    int(value) for value in record.boundary_node_ids
                )
            ):
                submaps.add(int(submap_id))
                nodes.update(int(value) for value in record.boundary_node_ids)
                windows.update(int(value) for value in record.window_ids)
        frames = {
            int(frame_id)
            for frame_id, owner in self.frame_pose_owner_node.items()
            if int(owner) in nodes
        }
        for frame_id in frames:
            windows.update(
                int(value) for value in self.frame_windows.get(frame_id, set())
            )
        windows.update(
            int(window_id)
            for window_id, anchor_node in self.window_anchor_nodes.items()
            if int(anchor_node) in nodes
        )
        for packet in extra_packets:
            windows.add(int(packet.window_id))
            frames.update(
                int(frame_id)
                for frame_id in packet.frame_ids
                if int(frame_id) in self.frame_pose_owner_node
                and int(frame_id) in self.frame_local_pose_in_owner
            )
        for window_id in tuple(windows):
            submap_id = self.window_to_submap.get(int(window_id))
            if submap_id is not None:
                submaps.add(int(submap_id))
        return nodes, frames, windows, submaps

    def _canonical_frame_global_poses(
        self,
        frame_ids: set[int],
    ) -> dict[int, torch.Tensor]:
        poses: dict[int, torch.Tensor] = {}
        for frame_id in sorted(int(value) for value in frame_ids):
            owner_node = self.frame_pose_owner_node.get(frame_id)
            local_pose = self.frame_local_pose_in_owner.get(frame_id)
            if (
                owner_node is None
                or local_pose is None
                or int(owner_node) not in self.graph.nodes
            ):
                raise RuntimeError(
                    f"Frame {frame_id} has no complete canonical pose state"
                )
            pose = apply_sim3_to_c2w(
                self.graph.transform(int(owner_node)).to(local_pose),
                local_pose,
            )
            if not bool(torch.isfinite(pose).all()):
                raise RuntimeError(
                    f"Frame {frame_id} has a non-finite canonical pose"
                )
            poses[frame_id] = pose.detach().clone()
        return poses

    def _refresh_affected_submap_local_geometry(
        self,
        submap_ids: set[int],
    ) -> None:
        if not self.hierarchical_submaps_enabled:
            return
        for submap_id in sorted(int(value) for value in submap_ids):
            if submap_id in self.submaps:
                self._update_submap_local_geometry(submap_id)

    def _synchronize_mapper_pose_cache(
        self,
        frame_global_poses: dict[int, torch.Tensor],
        *,
        revision: int,
    ) -> int:
        if self.mapper is None or not frame_global_poses:
            return 0
        apply_state = getattr(
            self.mapper,
            "apply_canonical_pose_state",
            None,
        )
        if not callable(apply_state):
            return 0
        applied = apply_state(
            frame_global_poses,
            revision=int(revision),
        )
        for frame_id in frame_global_poses:
            if (
                int(frame_id) in getattr(self.mapper, "pose_deltas", {})
                or int(frame_id) in getattr(self.mapper, "observations", {})
            ):
                self._mapper_pose_revision_by_frame[int(frame_id)] = int(
                    revision
                )
        return int(applied)

    def _pose_state_consistency_report(
        self,
        candidate: PoseStateCandidate,
        *,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
    ) -> PoseStateConsistencyReport:
        matrix_errors: list[float] = []
        rotation_errors: list[float] = []
        center_errors: list[float] = []
        submap_errors: list[float] = []
        lazy_owner_errors: list[float] = []
        mapper_errors: list[float] = []
        non_finite_count = 0
        revision_mismatch_count = 0
        packet_variant_count = 0
        submap_transform_count = 0
        lazy_owner_count = 0
        mapper_pose_count = 0

        def matrix_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
            nonlocal non_finite_count
            actual_value = actual.detach().to(expected)
            expected_value = expected.detach()
            if not bool(torch.isfinite(actual_value).all()) or not bool(
                torch.isfinite(expected_value).all()
            ):
                non_finite_count += 1
                return float("inf")
            return float(
                (actual_value - expected_value).abs().max().detach().cpu()
            )

        for window_id in candidate.affected_window_ids:
            window_transform = candidate.window_transforms.get(int(window_id))
            if window_transform is None:
                continue
            for packet in self._packet_variants(
                int(window_id),
                extra_packets=extra_packets,
            ):
                packet_variant_count += 1
                if int(
                    packet.metadata.get("canonical_pose_revision", -1)
                ) != int(candidate.revision):
                    revision_mismatch_count += 1
                packet_poses = packet.global_poses(
                    window_transform.to(packet.local_poses_c2w)
                )
                for index, frame_id in enumerate(packet.frame_ids):
                    expected = candidate.frame_global_poses.get(int(frame_id))
                    if expected is None:
                        continue
                    actual = packet_poses[index]
                    matrix_errors.append(matrix_error(actual, expected))
                    rotation_errors.append(
                        self._pose_state_rotation_error_deg(
                            actual[:3, :3].to(expected),
                            expected[:3, :3],
                        )
                    )
                    center_errors.append(
                        float(
                            torch.linalg.norm(
                                actual[:3, 3].to(expected)
                                - expected[:3, 3]
                            )
                            .detach()
                            .cpu()
                        )
                    )

        if self.hierarchical_submaps_enabled:
            assert self.submap_graph is not None
            for submap_id in candidate.affected_submap_ids:
                if (
                    int(submap_id) not in self.submaps
                    or int(submap_id) not in self.submap_graph.nodes
                ):
                    continue
                record = self.submaps[int(submap_id)]
                submap_transform = self.submap_graph.transform(int(submap_id))
                for node_id, local in record.local_boundary_transforms.items():
                    if int(node_id) not in self.graph.nodes:
                        continue
                    reconstructed = submap_transform.to(local) @ local
                    submap_errors.append(
                        matrix_error(
                            reconstructed,
                            self.graph.transform(int(node_id)),
                        )
                    )
                    submap_transform_count += 1
                for window_id, local in record.local_window_transforms.items():
                    anchor_node = self.window_anchor_nodes.get(int(window_id))
                    if anchor_node is None or int(anchor_node) not in self.graph.nodes:
                        continue
                    reconstructed = submap_transform.to(local) @ local
                    submap_errors.append(
                        matrix_error(
                            reconstructed,
                            self.graph.transform(int(anchor_node)),
                        )
                    )
                    submap_transform_count += 1

        lazy_current = self.map._lazy_owner_current_transforms
        for window_id, expected in candidate.window_transforms.items():
            current = lazy_current.get(int(window_id))
            if current is None:
                continue
            lazy_owner_errors.append(matrix_error(current, expected))
            lazy_owner_count += 1

        if self.mapper is not None and callable(
            getattr(self.mapper, "apply_canonical_pose_state", None)
        ):
            mapper_observations = getattr(self.mapper, "observations", {})
            refined_pose = getattr(self.mapper, "refined_pose_c2w", None)
            for frame_id, expected in candidate.frame_global_poses.items():
                observation = mapper_observations.get(int(frame_id))
                refined = (
                    refined_pose(int(frame_id))
                    if callable(refined_pose)
                    else None
                )
                if observation is None and refined is None:
                    continue
                mapper_pose_count += 1
                if refined is not None:
                    mapper_errors.append(matrix_error(refined, expected))
                if observation is not None:
                    mapper_errors.append(
                        matrix_error(observation.pose_c2w, expected)
                    )
                    if int(observation.pose_revision) != int(
                        candidate.revision
                    ):
                        revision_mismatch_count += 1
                if int(
                    self._mapper_pose_revision_by_frame.get(
                        int(frame_id), -1
                    )
                ) != int(candidate.revision):
                    revision_mismatch_count += 1

        max_matrix = max(matrix_errors, default=0.0)
        max_rotation = max(rotation_errors, default=0.0)
        max_center = max(center_errors, default=0.0)
        max_submap = max(submap_errors, default=0.0)
        max_lazy = max(lazy_owner_errors, default=0.0)
        max_mapper = max(mapper_errors, default=0.0)
        accepted = bool(
            non_finite_count == 0
            and revision_mismatch_count == 0
            and max_matrix <= 1.0e-5
            and max_rotation <= 1.0e-4
            and max_center <= 1.0e-5
            and max_submap <= 1.0e-5
            and max_lazy <= 1.0e-5
            and max_mapper <= 1.0e-5
        )
        reason = "accepted"
        if non_finite_count:
            reason = "non_finite_pose_state"
        elif revision_mismatch_count:
            reason = "pose_revision_mismatch"
        elif not accepted:
            reason = "pose_round_trip_mismatch"
        return PoseStateConsistencyReport(
            candidate_revision=int(candidate.revision),
            committed_revision=int(self._pose_state_revision),
            accepted=accepted,
            frame_count=len(candidate.frame_global_poses),
            packet_variant_count=packet_variant_count,
            submap_transform_count=submap_transform_count,
            lazy_owner_count=lazy_owner_count,
            mapper_pose_count=mapper_pose_count,
            max_matrix_error=max_matrix,
            max_rotation_error_deg=max_rotation,
            max_center_error=max_center,
            max_submap_matrix_error=max_submap,
            max_lazy_owner_matrix_error=max_lazy,
            max_mapper_matrix_error=max_mapper,
            non_finite_count=non_finite_count,
            revision_mismatch_count=revision_mismatch_count,
            reason=reason,
        )

    def _materialize_pose_state_candidate(
        self,
        *,
        affected_node_ids: set[int],
        old_window_transforms: dict[int, torch.Tensor],
        reason: str,
        affected_submap_ids: set[int] | None = None,
        extra_packets: tuple[LocalGaussianWindowPacket, ...] = (),
    ) -> tuple[
        PoseStateCandidate,
        PoseStateConsistencyReport,
        dict[str, int],
    ]:
        nodes, frames, windows, submaps = self._pose_state_affected_sets(
            affected_node_ids,
            affected_submap_ids=affected_submap_ids,
            extra_packets=extra_packets,
        )
        revision = max(
            int(self._geometry_revision),
            int(self._pose_state_revision),
        ) + 1
        self._refresh_affected_submap_local_geometry(submaps)
        all_window_transforms = self._window_anchor_transforms()
        missing_windows = windows.difference(all_window_transforms)
        if missing_windows:
            raise RuntimeError(
                "Canonical pose candidate is missing window transforms: "
                f"{sorted(missing_windows)}"
            )
        frame_global_poses = self._canonical_frame_global_poses(frames)
        packet_variant_count = self._synchronize_chunk_packet_variants(
            windows,
            extra_packets=extra_packets,
            revision=revision,
        )
        candidate = PoseStateCandidate(
            revision=revision,
            reason=str(reason),
            affected_node_ids=tuple(sorted(nodes)),
            affected_frame_ids=tuple(sorted(frames)),
            affected_window_ids=tuple(sorted(windows)),
            affected_submap_ids=tuple(sorted(submaps)),
            frame_global_poses=frame_global_poses,
            window_transforms={
                int(window_id): all_window_transforms[int(window_id)].clone()
                for window_id in windows
            },
            packet_variant_count=packet_variant_count,
        )
        owner_sync_required = bool(
            self.fusion.lazy_owner_transforms
            or self._owner_transforms_changed(
                old_window_transforms,
                all_window_transforms,
            )
        )
        correction = (
            self.fusion.apply_owner_corrections(
                old_window_transforms,
                all_window_transforms,
            )
            if owner_sync_required
            else {"moved": 0, "deduplicated": 0}
        )
        self._synchronize_mapper_pose_cache(
            frame_global_poses,
            revision=revision,
        )
        report = self._pose_state_consistency_report(
            candidate,
            extra_packets=extra_packets,
        )
        self._last_pose_state_diagnostic = report.as_diagnostics()
        if not report.accepted:
            raise RuntimeError(
                "Canonical pose-state consistency failed: "
                f"{report.reason}; matrix={report.max_matrix_error:.3e}, "
                f"rotation={report.max_rotation_error_deg:.3e}deg, "
                f"center={report.max_center_error:.3e}, "
                f"submap={report.max_submap_matrix_error:.3e}, "
                f"lazy_owner={report.max_lazy_owner_matrix_error:.3e}, "
                f"mapper={report.max_mapper_matrix_error:.3e}, "
                f"revision_mismatches={report.revision_mismatch_count}"
            )
        self._pose_state_revision = int(revision)
        for window_id in windows:
            self._window_pose_revision[int(window_id)] = int(revision)
        report = replace(report, committed_revision=int(revision))
        self._last_pose_state_diagnostic = report.as_diagnostics()
        self._last_pose_state_diagnostic["transaction_committed"] = True
        return candidate, report, correction

    def _committed_mapper_render_diagnostics(
        self,
        frame_ids: set[int],
        *,
        revision: int,
    ) -> dict[str, Any]:
        """Render the map after owner/camera state has reached one revision."""

        if self.mapper is None:
            return {
                "enabled": False,
                "candidate_revision": int(revision),
                "committed_revision": int(self._pose_state_revision),
                "reason": "mapper_unavailable",
            }
        render_diagnostic = getattr(
            self.mapper,
            "render_keyframe_diagnostic",
            None,
        )
        if not callable(render_diagnostic):
            return {
                "enabled": False,
                "candidate_revision": int(revision),
                "committed_revision": int(self._pose_state_revision),
                "reason": "mapper_render_diagnostic_unavailable",
            }
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for frame_id in sorted(int(value) for value in frame_ids):
            diagnostic = render_diagnostic(frame_id)
            if diagnostic is None:
                continue
            owner_node = self.frame_pose_owner_node.get(frame_id)
            rows.append(
                {
                    "frame_id": frame_id,
                    "pose_owner_node_id": (
                        None if owner_node is None else int(owner_node)
                    ),
                    "is_segment_tail": bool(
                        owner_node is not None and int(owner_node) != frame_id
                    ),
                    "loss": float(diagnostic.loss),
                    "psnr": float(diagnostic.psnr),
                    "anchor_count": int(diagnostic.anchor_count),
                }
            )
        losses = [float(row["loss"]) for row in rows]
        psnrs = [float(row["psnr"]) for row in rows]
        tail_rows = [row for row in rows if bool(row["is_segment_tail"])]
        return {
            "enabled": True,
            "candidate_revision": int(revision),
            "committed_revision": int(self._pose_state_revision),
            "view_count": len(rows),
            "tail_view_count": len(tail_rows),
            "mean_loss": (
                sum(losses) / len(losses) if losses else 0.0
            ),
            "mean_psnr": (
                sum(psnrs) / len(psnrs) if psnrs else 0.0
            ),
            "tail_mean_loss": (
                sum(float(row["loss"]) for row in tail_rows)
                / len(tail_rows)
                if tail_rows
                else 0.0
            ),
            "render_seconds": float(time.perf_counter() - started),
            "per_frame": rows,
            "reason": "committed_pose_owner_state",
        }

    def _synchronize_chunk_stride_optimized_window_impl(
        self,
        window_id: int,
        *,
        optimized_frame_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Validate and publish one mapper pose proposal to canonical segments."""

        if self.mapper is None or int(window_id) not in self.packets:
            return
        packet = self.packets[int(window_id)]
        selected_frame_ids = (
            tuple(int(value) for value in packet.frame_ids)
            if optimized_frame_ids is None
            else tuple(dict.fromkeys(int(value) for value in optimized_frame_ids))
        )
        optimized_by_frame: dict[int, torch.Tensor] = {}
        for frame_id in selected_frame_ids:
            pose = self.mapper.refined_pose_c2w(int(frame_id))
            if pose is None:
                raise RuntimeError(f"missing optimized pose for frame {frame_id}")
            if tuple(pose.shape) != (4, 4) or not bool(torch.isfinite(pose).all()):
                raise RuntimeError(f"invalid optimized pose for frame {frame_id}")
            if int(frame_id) not in self.frame_pose_owner_node:
                raise RuntimeError(
                    f"frame {frame_id} has no canonical chunk-stride owner"
                )
            optimized_by_frame[int(frame_id)] = canonicalize_c2w(pose.float())

        affected_nodes = {
            int(self.frame_pose_owner_node[int(frame_id)])
            for frame_id in optimized_by_frame
        }
        affected_factors = tuple(
            factor
            for factor in self.graph.edges
            if getattr(factor, "edge_type", "")
            in {
                "chunk_stride_dense_spherical",
                "chunk_skip_dense_spherical",
            }
            and (
                int(factor.source) in affected_nodes
                or int(factor.target) in affected_nodes
            )
        )
        sequential_factors = tuple(
            factor
            for factor in affected_factors
            if factor.edge_type == "chunk_stride_dense_spherical"
        )
        skip_factors = tuple(
            factor
            for factor in affected_factors
            if factor.edge_type == "chunk_skip_dense_spherical"
        )
        sequence_before = float(
            self.graph.objective(factors=sequential_factors).detach().cpu()
        )
        skip_before = float(
            self.graph.objective(factors=skip_factors).detach().cpu()
        )
        old_window_transforms = self._window_anchor_transforms()

        # Node scales remain graph-owned.  The mapper proposes only global
        # SE(3) R/t; both frames in each rigid segment are then reconstructed
        # from the same canonical owner node.
        for owner in sorted(affected_nodes):
            if owner not in optimized_by_frame:
                continue
            current = self.graph.transform(owner)
            scale, _, _ = sim3_components(current)
            pose = optimized_by_frame[owner].to(current)
            self.graph.nodes[owner] = sim3_from_components(
                scale,
                pose[:3, :3],
                pose[:3, 3],
            ).detach()
        for frame_id, global_pose in optimized_by_frame.items():
            owner = int(self.frame_pose_owner_node[frame_id])
            owner_transform = self.graph.transform(owner)
            if frame_id == owner:
                local_pose = torch.eye(
                    4,
                    device=owner_transform.device,
                    dtype=owner_transform.dtype,
                )
            else:
                local_pose = rebase_c2w_to_sim3_anchor(
                    owner_transform,
                    global_pose.to(owner_transform),
                )
            self._validate_pose_round_trip(
                owner_transform,
                local_pose,
                global_pose,
                frame_id=frame_id,
                window_id=window_id,
            )
            self.frame_local_pose_in_owner[frame_id] = local_pose.detach().clone()

        sequence_after = float(
            self.graph.objective(factors=sequential_factors).detach().cpu()
        )
        skip_after = float(
            self.graph.objective(factors=skip_factors).detach().cpu()
        )
        sequence_ratio = sequence_after / max(sequence_before, 1.0e-12)
        skip_ratio = skip_after / max(skip_before, 1.0e-12)
        holdout = self._chunk_stride_holdout_diagnostics(
            affected_node_ids=affected_nodes
        )
        if (
            not math.isfinite(sequence_after)
            or not math.isfinite(skip_after)
        ):
            raise RuntimeError(
                "mapper pose proposal produced a non-finite graph objective: "
                f"sequence_ratio={sequence_ratio:.6f}, "
                f"skip_ratio={skip_ratio:.6f}"
            )
        candidate, report, _ = self._materialize_pose_state_candidate(
            affected_node_ids=affected_nodes,
            old_window_transforms=old_window_transforms,
            reason="mapper_pose_candidate",
        )
        self._last_mapper_committed_state_diagnostic = (
            self._committed_mapper_render_diagnostics(
                set(candidate.affected_frame_ids),
                revision=int(candidate.revision),
            )
        )
        self._last_mapper_committed_state_diagnostic[
            "pose_state_consistency"
        ] = report.as_diagnostics()
        self._last_mapper_committed_state_diagnostic[
            "geometric_quality_diagnostics"
        ] = {
            "quality_gating_enabled": False,
            "sequence_objective_before": sequence_before,
            "sequence_objective_after": sequence_after,
            "sequence_objective_ratio": sequence_ratio,
            "skip_objective_before": skip_before,
            "skip_objective_after": skip_after,
            "skip_objective_ratio": skip_ratio,
            "holdout": holdout,
        }
        self._refresh_geometry_updates(
            complete_snapshot=True,
            affected_node_ids=affected_nodes,
            reason="mapper_pose_transaction_commit",
        )

    def _synchronize_chunk_stride_optimized_window(
        self,
        window_id: int,
        *,
        optimized_frame_ids: tuple[int, ...] | None = None,
    ) -> None:
        transaction = self._snapshot_boundary_transaction()
        try:
            self._synchronize_chunk_stride_optimized_window_impl(
                window_id,
                optimized_frame_ids=optimized_frame_ids,
            )
        except Exception:
            self._restore_boundary_transaction(transaction)
            raise

    def _synchronize_boundary_optimized_window(self, window_id: int) -> None:
        """Synchronize global render-optimized poses with boundary nodes and packets."""

        if self.mapper is None or int(window_id) not in self.packets:
            return
        packet = self.packets[int(window_id)]
        optimized_by_frame: dict[int, torch.Tensor] = {}
        for frame_id in packet.frame_ids:
            pose = self.mapper.refined_pose_c2w(int(frame_id))
            if pose is None:
                raise RuntimeError(f"missing optimized pose for frame {frame_id}")
            if tuple(pose.shape) != (4, 4) or not bool(torch.isfinite(pose).all()):
                raise RuntimeError(f"invalid optimized pose for frame {frame_id}")
            optimized_by_frame[int(frame_id)] = canonicalize_c2w(pose.float())

        affected_windows = {
            int(owner)
            for frame_id in optimized_by_frame
            for owner in self.frame_windows.get(int(frame_id), {int(window_id)})
        }
        old_nodes = {node: value.clone() for node, value in self.graph.nodes.items()}
        geometry_snapshot = dict(self._geometry_updates)
        packet_snapshots = {
            id(variant): (variant, variant.local_poses_c2w.clone(), variant.observation)
            for owner in affected_windows
            for variant in self._packet_variants(owner)
        }
        factor_snapshots = {
            id(factor): (
                factor,
                factor.source_local_pose.clone(),
                factor.target_local_pose.clone(),
            )
            for factor in self.graph.edges
            if isinstance(factor, (DenseSphericalFactorBlock, CoincidentPanoramaFactor))
        }

        try:
            # The graph/map feedback loop crosses a float32 matrix boundary.
            # Retract every node before using R.T as R^-1 so numerical shear
            # cannot accumulate from one window to the next.
            self.graph.nodes = {
                int(node): canonicalize_sim3(transform)
                for node, transform in self.graph.nodes.items()
            }
            # Boundary-node scale remains the Sim(3) gauge authority; render BA
            # contributes only the optimized global SE(3) rotation/translation.
            for frame_id, global_pose in optimized_by_frame.items():
                if int(frame_id) not in self.graph.nodes:
                    continue
                current = self.graph.transform(int(frame_id))
                scale, _, _ = sim3_components(current)
                pose = global_pose.to(current)
                self.graph.nodes[int(frame_id)] = sim3_from_components(
                    scale,
                    pose[:3, :3],
                    pose[:3, 3],
                ).detach()

            for owner in affected_windows:
                anchor_node = self.window_anchor_nodes[int(owner)]
                transform = self.graph.transform(anchor_node)
                for variant in self._packet_variants(owner):
                    local_poses = variant.local_poses_c2w.clone()
                    changed = False
                    for frame_id, global_pose in optimized_by_frame.items():
                        if frame_id not in variant.frame_ids:
                            continue
                        index = variant.frame_index(frame_id)
                        local = rebase_c2w_to_sim3_anchor(
                            transform.to(local_poses), global_pose.to(local_poses)
                        )
                        if int(frame_id) == int(anchor_node):
                            local = torch.eye(4, device=local.device, dtype=local.dtype)
                        self._validate_pose_round_trip(
                            transform,
                            local,
                            global_pose,
                            frame_id=frame_id,
                            window_id=owner,
                        )
                        local_poses[index] = local
                        changed = True
                    if changed:
                        variant.local_poses_c2w = local_poses.detach()
                        variant.observation = variant.observation.with_geometry(
                            poses_c2w=local_poses.unsqueeze(0).to(
                                variant.observation.poses_c2w
                            )
                        )

            self._refresh_factor_local_poses(affected_windows)
            if self.hierarchical_submaps_enabled:
                for submap_id in {
                    self.window_to_submap[owner]
                    for owner in affected_windows
                    if owner in self.window_to_submap
                }:
                    self._update_submap_local_geometry(submap_id)
            # The graph BA cadence stays governed by optimization_start_nodes /
            # optimization_interval_edges.  Synchronization itself must not turn
            # the configured periodic graph BA into an every-window solve.
            self._refresh_geometry_updates()
        except Exception:
            self.graph.nodes = {node: value.clone() for node, value in old_nodes.items()}
            self._geometry_updates = geometry_snapshot
            for variant, local_poses, observation in packet_snapshots.values():
                variant.local_poses_c2w = local_poses
                variant.observation = observation
            for factor, source_pose, target_pose in factor_snapshots.values():
                factor.source_local_pose = source_pose
                factor.target_local_pose = target_pose
            raise

    @staticmethod
    def _canonical_loop_pair(result: PanoramaLoopVerification) -> tuple[int, int]:
        return tuple(sorted((int(result.source_window_id), int(result.target_window_id))))

    def _materialize_loop_pose_factor(
        self,
        measurement: LoopPoseMeasurement,
    ) -> Sim3GraphEdge | CoincidentPanoramaFactor:
        """Convert a verified backend-neutral measurement into a graph factor."""

        if measurement.kind == "sim3":
            assert measurement.measurement_target_to_source is not None
            assert measurement.information_diag is not None
            information = measurement.information_diag
            if self.normalize_dense_information_by_count:
                count = max(
                    1.0,
                    float(
                        measurement.metadata.get(
                            "verified_num_matches",
                            measurement.metadata.get("num_matches", 1.0),
                        )
                    ),
                )
                information = information * (
                    self.dense_information_reference_count / count
                )
            return Sim3GraphEdge(
                source=int(measurement.source),
                target=int(measurement.target),
                measurement_target_to_source=measurement.measurement_target_to_source,
                information_diag=information,
                edge_type=measurement.edge_type,
                robust_delta=measurement.robust_delta,
                dcs_phi=measurement.dcs_phi,
                metadata=dict(measurement.metadata),
            )
        assert measurement.source_local_pose is not None
        assert measurement.target_local_pose is not None
        assert measurement.measured_source_to_target_rotation is not None
        center_weight = float(measurement.center_weight)
        rotation_weight = float(measurement.rotation_weight)
        if self.normalize_dense_information_by_count:
            count = max(
                1.0,
                float(
                    measurement.metadata.get(
                        "verified_num_matches",
                        measurement.metadata.get("num_matches", 1.0),
                    )
                ),
            )
            normalization = self.dense_information_reference_count / count
            center_weight *= normalization
            rotation_weight *= normalization
        return CoincidentPanoramaFactor(
            source=int(measurement.source),
            target=int(measurement.target),
            source_local_pose=measurement.source_local_pose,
            target_local_pose=measurement.target_local_pose,
            measured_source_to_target_rotation=measurement.measured_source_to_target_rotation,
            center_weight=center_weight,
            rotation_weight=rotation_weight,
            robust_delta=float(measurement.robust_delta),
            edge_type=measurement.edge_type,
            dcs_phi=measurement.dcs_phi,
            metadata=dict(measurement.metadata),
        )

    def _materialize_dense_loop_factor(
        self,
        measurement: DenseSphericalLoopMeasurement,
    ) -> DenseSphericalFactorBlock:
        return DenseSphericalFactorBlock(
            source=int(measurement.source),
            target=int(measurement.target),
            source_local_pose=measurement.source_local_pose,
            target_local_pose=measurement.target_local_pose,
            source_bearing=measurement.source_bearing,
            target_bearing=measurement.target_bearing,
            source_depth=measurement.source_depth,
            target_depth=measurement.target_depth,
            factor_weight=measurement.factor_weight,
            depth_factor_weight=float(measurement.depth_factor_weight),
            s2_huber_delta_deg=float(measurement.s2_huber_delta_deg),
            use_depth=bool(measurement.use_depth),
            robust_delta=float(measurement.robust_delta),
            edge_type=measurement.edge_type,
            dcs_phi=measurement.dcs_phi,
            **self._dense_factor_information_options(),
            metadata=dict(measurement.metadata),
        )

    def _loop_path_consistency(
        self,
        result: PanoramaLoopVerification,
        *,
        source_node: int,
        target_node: int,
        graph: GlobalSim3FactorGraph | None = None,
        measurement: LoopPoseMeasurement | None = None,
    ) -> tuple[bool, dict[str, float | str]]:
        """Compare a verified loop measurement with the graph's current path."""

        factor = result.factor if measurement is None else measurement
        if factor is None:
            return False, {"reason": "missing_loop_summary_factor"}
        active_graph = self.graph if graph is None else graph
        source_transform = active_graph.transform(source_node)
        target_transform = active_graph.transform(target_node).to(source_transform)
        if factor.kind == "sim3":
            assert factor.measurement_target_to_source is not None
            predicted = sim3_inverse(source_transform) @ target_transform
            error = sim3_inverse(factor.measurement_target_to_source.to(predicted)) @ predicted
            tangent = sim3_log(error)
            translation = float(tangent[:3].norm().detach().cpu())
            rotation = float(tangent[3:6].norm().detach().cpu())
            log_scale = abs(float(tangent[6].detach().cpu()))
        elif factor.kind == "coincident":
            assert factor.source_local_pose is not None
            assert factor.target_local_pose is not None
            assert factor.measured_source_to_target_rotation is not None
            source_pose = factor.source_local_pose.to(source_transform)
            target_pose = factor.target_local_pose.to(source_transform)
            source_center = apply_sim3(source_transform, source_pose[:3, 3])
            target_center = apply_sim3(target_transform, target_pose[:3, 3])
            _, source_rotation, _ = sim3_components(source_transform)
            _, target_rotation, _ = sim3_components(target_transform)
            predicted_rotation = (
                source_rotation @ source_pose[:3, :3]
            ).transpose(0, 1) @ (target_rotation @ target_pose[:3, :3])
            measured = factor.measured_source_to_target_rotation.to(predicted_rotation)
            rotation_error = measured.transpose(0, 1) @ predicted_rotation
            rotation = float(sim3_log(sim3_from_components(
                rotation_error.new_tensor(1.0), rotation_error, rotation_error.new_zeros(3)
            ))[3:6].norm().detach().cpu())
            translation = float((target_center - source_center).norm().detach().cpu())
            log_scale = 0.0
        diagnostics: dict[str, float | str] = {
            "translation_error": translation,
            "rotation_error_deg": math.degrees(rotation),
            "log_scale_error": log_scale,
        }
        accepted = (
            math.isfinite(translation)
            and math.isfinite(rotation)
            and math.isfinite(log_scale)
            and translation <= self.loop_path_max_translation
            and rotation <= self.loop_path_max_rotation
            and log_scale <= self.loop_path_max_log_scale
        )
        diagnostics["reason"] = "accepted" if accepted else "path_inconsistent"
        return accepted, diagnostics

    @staticmethod
    def _compose_submap_local_pose(
        window_to_submap: torch.Tensor,
        camera_to_window: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compose a window-local SE(3) pose with a local Sim(3) gauge."""

        scale, rotation, translation = sim3_components(window_to_submap)
        pose = camera_to_window.to(window_to_submap)
        composed = torch.eye(4, device=pose.device, dtype=pose.dtype)
        composed[:3, :3] = rotation @ pose[:3, :3]
        composed[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
        return composed, float(scale.detach().cpu())

    def _loop_measurement_for_submaps(
        self,
        result: PanoramaLoopVerification,
    ) -> LoopPoseMeasurement | None:
        measurement = result.factor
        if measurement is None:
            return None
        source_window = int(result.source_window_id)
        target_window = int(result.target_window_id)
        source_submap = self.window_to_submap.get(source_window)
        target_submap = self.window_to_submap.get(target_window)
        if source_submap is None or target_submap is None or source_submap == target_submap:
            return None
        source_local = self.submaps[source_submap].local_window_transforms[source_window]
        target_local = self.submaps[target_submap].local_window_transforms[target_window].to(source_local)
        if measurement.kind == "sim3":
            assert measurement.measurement_target_to_source is not None
            assert measurement.information_diag is not None
            submap_measurement = (
                source_local
                @ measurement.measurement_target_to_source.to(source_local)
                @ sim3_inverse(target_local)
            )
            return LoopPoseMeasurement(
                kind="sim3",
                source=int(source_submap),
                target=int(target_submap),
                edge_type=measurement.edge_type,
                measurement_target_to_source=submap_measurement,
                information_diag=measurement.information_diag,
                robust_delta=measurement.robust_delta,
                dcs_phi=measurement.dcs_phi,
                metadata=dict(measurement.metadata),
            )
        assert measurement.source_local_pose is not None
        assert measurement.target_local_pose is not None
        assert measurement.measured_source_to_target_rotation is not None
        source_pose, _ = self._compose_submap_local_pose(
            source_local, measurement.source_local_pose
        )
        target_pose, _ = self._compose_submap_local_pose(
            target_local, measurement.target_local_pose
        )
        return LoopPoseMeasurement(
            kind="coincident",
            source=int(source_submap),
            target=int(target_submap),
            edge_type=measurement.edge_type,
            source_local_pose=source_pose,
            target_local_pose=target_pose,
            measured_source_to_target_rotation=measurement.measured_source_to_target_rotation,
            center_weight=measurement.center_weight,
            rotation_weight=measurement.rotation_weight,
            robust_delta=measurement.robust_delta,
            dcs_phi=measurement.dcs_phi,
            metadata=dict(measurement.metadata),
        )

    def _loop_measurement_for_boundary_nodes(
        self,
        result: PanoramaLoopVerification,
    ) -> LoopPoseMeasurement | None:
        measurement = result.factor
        if measurement is None:
            return None
        source = self.window_anchor_nodes.get(int(result.source_window_id))
        target = self.window_anchor_nodes.get(int(result.target_window_id))
        if source is None or target is None or int(source) == int(target):
            return None
        values = {
            "kind": measurement.kind,
            "source": int(source),
            "target": int(target),
            "edge_type": measurement.edge_type,
            "robust_delta": measurement.robust_delta,
            "dcs_phi": measurement.dcs_phi,
            "metadata": dict(measurement.metadata),
        }
        if measurement.kind == "sim3":
            return LoopPoseMeasurement(
                **values,
                measurement_target_to_source=measurement.measurement_target_to_source,
                information_diag=measurement.information_diag,
            )
        return LoopPoseMeasurement(
            **values,
            source_local_pose=measurement.source_local_pose,
            target_local_pose=measurement.target_local_pose,
            measured_source_to_target_rotation=(
                measurement.measured_source_to_target_rotation
            ),
            center_weight=measurement.center_weight,
            rotation_weight=measurement.rotation_weight,
        )

    def _merge_loop_submap_dense_factors(
        self,
        result: PanoramaLoopVerification,
    ) -> DenseSphericalFactorBlock | None:
        source_window = int(result.source_window_id)
        target_window = int(result.target_window_id)
        source_submap = self.window_to_submap.get(source_window)
        target_submap = self.window_to_submap.get(target_window)
        if source_submap is None or target_submap is None or source_submap == target_submap:
            return None
        source_local = self.submaps[source_submap].local_window_transforms[source_window]
        target_local = self.submaps[target_submap].local_window_transforms[target_window].to(source_local)
        source_bearing_parts: list[torch.Tensor] = []
        target_bearing_parts: list[torch.Tensor] = []
        source_depth_parts: list[torch.Tensor] = []
        target_depth_parts: list[torch.Tensor] = []
        weight_parts: list[torch.Tensor] = []
        source_pose_reference: torch.Tensor | None = None
        target_pose_reference: torch.Tensor | None = None
        reference: DenseSphericalLoopMeasurement | None = None
        use_depth = False
        for factor in result.dense_factors:
            if int(factor.source) == source_window and int(factor.target) == target_window:
                source_bearing, target_bearing = factor.source_bearing, factor.target_bearing
                source_depth, target_depth = factor.source_depth, factor.target_depth
                source_pose_raw, target_pose_raw = factor.source_local_pose, factor.target_local_pose
            elif int(factor.source) == target_window and int(factor.target) == source_window:
                source_bearing, target_bearing = factor.target_bearing, factor.source_bearing
                source_depth, target_depth = factor.target_depth, factor.source_depth
                source_pose_raw, target_pose_raw = factor.target_local_pose, factor.source_local_pose
            else:
                continue
            source_pose, source_scale = self._compose_submap_local_pose(
                source_local, source_pose_raw
            )
            target_pose, target_scale = self._compose_submap_local_pose(
                target_local, target_pose_raw
            )
            if source_pose_reference is None:
                source_pose_reference = source_pose
                target_pose_reference = target_pose
            elif not (
                torch.allclose(source_pose_reference, source_pose, atol=1.0e-4, rtol=1.0e-4)
                and torch.allclose(target_pose_reference, target_pose, atol=1.0e-4, rtol=1.0e-4)
            ):
                raise ValueError("Bidirectional loop blocks disagree on their submap-local frame poses")
            source_bearing_parts.append(source_bearing)
            target_bearing_parts.append(target_bearing)
            source_depth_parts.append(source_depth * source_scale)
            target_depth_parts.append(target_depth * target_scale)
            weight_parts.append(factor.factor_weight)
            reference = factor
            use_depth = use_depth or bool(factor.use_depth)
        if reference is None or source_pose_reference is None or target_pose_reference is None:
            return None
        source_bearing = torch.cat(source_bearing_parts, dim=0)
        target_bearing = torch.cat([value.to(source_bearing) for value in target_bearing_parts], dim=0)
        source_depth = torch.cat([value.to(source_bearing) for value in source_depth_parts], dim=0)
        target_depth = torch.cat([value.to(source_bearing) for value in target_depth_parts], dim=0)
        weight = torch.cat([value.to(source_bearing) for value in weight_parts], dim=0)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "source_window_id": source_window,
                "target_window_id": target_window,
                "source_submap_id": int(source_submap),
                "target_submap_id": int(target_submap),
                "num_matches": int(source_depth.numel()),
                "hierarchical_submap_factor": True,
            }
        )
        return DenseSphericalFactorBlock(
            source=int(source_submap),
            target=int(target_submap),
            source_local_pose=source_pose_reference,
            target_local_pose=target_pose_reference,
            source_bearing=source_bearing,
            target_bearing=target_bearing,
            source_depth=source_depth,
            target_depth=target_depth,
            factor_weight=weight,
            depth_factor_weight=float(reference.depth_factor_weight),
            s2_huber_delta_deg=float(reference.s2_huber_delta_deg),
            use_depth=use_depth,
            robust_delta=float(reference.robust_delta),
            edge_type="submap_loop_dense_spherical",
            dcs_phi=reference.dcs_phi,
            **self._dense_factor_information_options(),
            metadata=metadata,
        )

    @staticmethod
    def _reject_loop_result(
        result: PanoramaLoopVerification,
        reason: str,
        **metadata: Any,
    ) -> None:
        result.accepted = False
        result.reason = str(reason)
        result.metadata.update(metadata)

    def _loop_transaction_commit_ok(
        self,
        graph_result: Sim3GraphOptimizeResult,
        factors: list[DenseSphericalFactorBlock | Sim3GraphEdge | CoincidentPanoramaFactor],
        *,
        nonloop_objective_before: float | None = None,
        nonloop_objective_after: float | None = None,
    ) -> tuple[bool, float, float]:
        scales = [
            float(factor.metadata.get("dcs_scale", 1.0))
            for factor in factors
            if factor.dcs_phi is not None
        ]
        minimum_scale = min(scales, default=1.0)
        graph_converged = bool(graph_result.accepted) or graph_result.reason in {
            "converged_gradient",
            "converged_step",
        }
        nonloop_ratio = 1.0
        nonloop_safe = True
        if nonloop_objective_before is not None and nonloop_objective_after is not None:
            baseline = max(float(nonloop_objective_before), 0.0)
            final = float(nonloop_objective_after)
            nonloop_ratio = final / max(baseline, self.loop_nonloop_objective_tolerance)
            nonloop_safe = (
                math.isfinite(final)
                and final
                <= baseline * self.loop_max_nonloop_objective_ratio
                + self.loop_nonloop_objective_tolerance
            )
        accepted = (
            graph_converged
            and math.isfinite(graph_result.final_objective)
            and graph_result.final_objective <= graph_result.initial_objective + 1.0e-9
            and minimum_scale >= self.loop_min_dcs_scale
            and nonloop_safe
        )
        return accepted, minimum_scale, nonloop_ratio

    def _merge_loop_dense_factors(
        self,
        result: PanoramaLoopVerification,
    ) -> DenseSphericalFactorBlock | None:
        source_window = int(result.source_window_id)
        target_window = int(result.target_window_id)
        if source_window not in self.window_anchor_nodes or target_window not in self.window_anchor_nodes:
            return None
        source_bearing_parts: list[torch.Tensor] = []
        target_bearing_parts: list[torch.Tensor] = []
        source_depth_parts: list[torch.Tensor] = []
        target_depth_parts: list[torch.Tensor] = []
        use_depth = False
        reference: DenseSphericalLoopMeasurement | None = None
        for factor in result.dense_factors:
            if int(factor.source) == source_window and int(factor.target) == target_window:
                source_bearing_parts.append(factor.source_bearing)
                target_bearing_parts.append(factor.target_bearing)
                source_depth_parts.append(factor.source_depth)
                target_depth_parts.append(factor.target_depth)
            elif int(factor.source) == target_window and int(factor.target) == source_window:
                # Swap reverse queries into the canonical loop source->target
                # direction so one observation creates one graph factor block.
                source_bearing_parts.append(factor.target_bearing)
                target_bearing_parts.append(factor.source_bearing)
                source_depth_parts.append(factor.target_depth)
                target_depth_parts.append(factor.source_depth)
            else:
                continue
            reference = factor
            use_depth = use_depth or bool(factor.use_depth)
        if reference is None or not source_bearing_parts:
            return None
        source_bearing = torch.cat(source_bearing_parts, dim=0)
        target_bearing = torch.cat(
            [value.to(source_bearing) for value in target_bearing_parts], dim=0
        )
        source_depth = torch.cat(
            [value.to(source_bearing) for value in source_depth_parts], dim=0
        )
        target_depth = torch.cat(
            [value.to(source_bearing) for value in target_depth_parts], dim=0
        )
        identity = torch.eye(4, device=source_bearing.device, dtype=source_bearing.dtype)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "source_window_id": source_window,
                "target_window_id": target_window,
                "source_frame_id": int(self.window_anchor_nodes[source_window]),
                "target_frame_id": int(self.window_anchor_nodes[target_window]),
                "num_matches": int(source_depth.numel()),
                "weight_mode": "fibonacci_equal_after_hard_gates",
                "bidirectional_matches_merged": True,
            }
        )
        return DenseSphericalFactorBlock(
            source=int(self.window_anchor_nodes[source_window]),
            target=int(self.window_anchor_nodes[target_window]),
            source_local_pose=identity,
            target_local_pose=identity.clone(),
            source_bearing=source_bearing,
            target_bearing=target_bearing,
            source_depth=source_depth,
            target_depth=target_depth,
            factor_weight=torch.ones_like(source_depth),
            depth_factor_weight=reference.depth_factor_weight,
            s2_huber_delta_deg=reference.s2_huber_delta_deg,
            use_depth=use_depth,
            robust_delta=reference.robust_delta,
            edge_type="loop_dense_spherical",
            dcs_phi=reference.dcs_phi,
            **self._dense_factor_information_options(),
            metadata=metadata,
        )

    def _run_prefusion_pose_tracking(
        self,
        packet: LocalGaussianWindowPacket,
        start_transform: torch.Tensor,
    ) -> tuple[LocalGaussianWindowPacket, dict[str, Any]]:
        """Refine only the two new global poses against the committed map."""

        if (
            not self.two_stage_map_optimization_enabled
            or not self.prefusion_pose_tracking_enabled
        ):
            return packet, {
                "enabled": False,
                "reason": "prefusion_pose_tracking_disabled",
            }
        if self.mapper is None:
            raise RuntimeError("Pre-fusion pose tracking requires a Mapper")
        if not self.window_order:
            return packet, {
                "enabled": True,
                "accepted": True,
                "steps": 0,
                "reason": "first_window_has_no_committed_map",
            }
        if len(packet.frame_ids) != 4:
            raise RuntimeError(
                "Pre-fusion pose tracking currently requires a four-frame packet"
            )
        owner_rows = getattr(self.map, "_anchor_owner_window_id", None)
        if torch.is_tensor(owner_rows) and bool(
            (owner_rows == int(packet.window_id)).any().detach().cpu()
        ):
            raise RuntimeError(
                "Candidate owner is already present in the committed map before tracking"
            )
        initial_global_poses = packet.global_poses(start_transform)
        prepared = self.mapper.prepare_spherical_selfi_window(packet.frame_ids)
        if prepared != len(packet.frame_ids):
            raise RuntimeError(
                f"window {packet.window_id} has {prepared}/{len(packet.frame_ids)} "
                "registered RGB observations before pose-only tracking"
            )
        provisional_revision = max(
            int(self._pose_state_revision),
            int(self._geometry_revision),
        ) + 1
        self.mapper.apply_canonical_pose_state(
            {
                int(frame_id): initial_global_poses[index]
                for index, frame_id in enumerate(packet.frame_ids)
            },
            revision=provisional_revision,
        )
        config = self.prefusion_pose_tracking_config
        pyramid_steps = (
            (
                float(config.get("quarter_resolution_scale", 0.25)),
                int(config.get("quarter_resolution_steps", 12)),
            ),
            (
                float(config.get("half_resolution_scale", 0.5)),
                int(config.get("half_resolution_steps", 18)),
            ),
            (1.0, int(config.get("full_resolution_steps", 30))),
        )
        new_frame_ids = tuple(int(value) for value in packet.frame_ids[2:4])
        metrics = self.mapper.optimize_spherical_selfi_pose_only(
            frame_ids=new_frame_ids,
            pyramid_steps=pyramid_steps,
            pose_lr=float(config.get("pose_lr", 2.0e-4)),
            pose_grad_clip=float(config.get("pose_grad_clip", 1.0e-3)),
            alpha_threshold=float(config.get("alpha_threshold", 0.05)),
            holdout_fraction=float(config.get("holdout_fraction", 0.20)),
            validation_interval=int(config.get("validation_interval", 10)),
            min_validation_improvement=float(
                config.get("min_validation_improvement", 0.0)
            ),
            photometric_mode=str(
                config.get("photometric_loss_mode", "charbonnier_dssim")
            ),
            charbonnier_weight=float(config.get("charbonnier_weight", 0.85)),
            dssim_weight=float(config.get("dssim_weight", 0.15)),
        )
        local_poses = packet.local_poses_c2w.detach().clone()
        # The two overlap poses are immutable anchors for this tracking pass.
        for index, frame_id in enumerate(packet.frame_ids[2:4], start=2):
            refined_global = self.mapper.refined_pose_c2w(int(frame_id))
            if refined_global is None:
                raise RuntimeError(
                    f"Pose-only tracking did not retain frame {int(frame_id)}"
                )
            tracking_anchor = start_transform.to(
                device=refined_global.device,
                dtype=refined_global.dtype,
            )
            local_poses[index] = canonicalize_c2w(
                rebase_c2w_to_sim3_anchor(
                    tracking_anchor,
                    refined_global.to(tracking_anchor),
                )
            ).to(local_poses)
        observation = packet.observation.with_geometry(
            poses_c2w=local_poses.unsqueeze(0).to(
                packet.observation.poses_c2w
            )
        )
        anchor_observation = packet.anchor_observation
        if anchor_observation is not None:
            anchor_observation = replace(
                anchor_observation,
                local_poses_c2w=local_poses.unsqueeze(0).to(
                    anchor_observation.local_poses_c2w
                ),
            )
        metadata = dict(packet.metadata)
        metadata["prefusion_pose_tracking_enabled"] = True
        metadata["prefusion_pose_tracking_revision"] = int(
            provisional_revision
        )
        metadata["prefusion_pose_tracking_metrics"] = dict(metrics)
        tracked = replace(
            packet,
            local_poses_c2w=local_poses,
            observation=observation,
            anchor_observation=anchor_observation,
            metadata=metadata,
        )
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "accepted": bool(metrics.get("validation_improved", 0.0)),
            "candidate_owner_excluded": True,
            "fixed_overlap_frame_ids": [
                int(value) for value in packet.frame_ids[:2]
            ],
            "optimized_new_frame_ids": list(new_frame_ids),
            "candidate_revision": int(provisional_revision),
            **metrics,
        }
        return tracked, diagnostics

    def _process_boundary_packet_impl(
        self,
        packet: LocalGaussianWindowPacket,
        *,
        prepared_candidate: _PreparedPacketCandidate | None = None,
    ) -> GlobalWindowBackendResult:
        if not self.enabled:
            raise RuntimeError("SphericalSelfiGlobalBackend is disabled")
        window_id = int(packet.window_id)
        if window_id in self.packets:
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")
        if len(packet.frame_ids) < 2:
            raise ValueError("Boundary-frame graph requires at least two frames per window")
        if (
            self.chunk_first_stride_graph
            and len(packet.frame_ids) <= self.chunk_stride_target_index
        ):
            raise ValueError(
                "chunk_first_stride requires the configured next-anchor frame"
            )
        if prepared_candidate is not None:
            packet = prepared_candidate.packet
            refined_packet = bool(prepared_candidate.refined_packet)
            refiner_pending = bool(prepared_candidate.refiner_pending)
        else:
            refined_packet = self._validate_refined_packet(packet)
            refiner_pending = bool(
                packet.metadata.get("voxel_anchor_refiner_pending", False)
            )
            if refined_packet:
                # Keep the frontend-owned packet immutable even for the first
                # chunk or for failures after alignment has succeeded.
                packet = self._rescaled_packet_copy(packet, 1.0)

        start_frame = int(packet.frame_ids[0])
        end_frame = int(packet.frame_ids[-1])
        alignment_diagnostics: dict[str, Any] = {}
        sequential_overlap_edge: Sim3GraphEdge | None = None
        sequential_overlap_dense: tuple[DenseSphericalFactorBlock, ...] = ()
        sequential_overlap_pose: tuple[CoincidentPanoramaFactor, ...] = ()
        previous_packet = self._last_full_packet
        if prepared_candidate is not None:
            if prepared_candidate.start_transform is None:
                raise RuntimeError("Prepared packet has no canonical transform")
            aligned = bool(prepared_candidate.aligned)
            start_transform = prepared_candidate.start_transform.clone().to(
                packet.local_poses_c2w
            )
            alignment_diagnostics = dict(
                prepared_candidate.alignment_diagnostics
            )
        elif not self.window_order:
            aligned = True
            start_transform = sim3_identity(device=packet.local_poses_c2w.device)
            alignment_diagnostics = {
                "reason": "first_window",
                "mode": (
                    self.rendered_overlap_alignment_mode
                    if self.two_frame_overlap_enabled
                    else (
                        "shared_frame_scale_only"
                        if refined_packet
                        else "legacy_sim3"
                    )
                ),
                "shared_scale": 1.0,
                "s_shared": 1.0,
                "absolute_scale": 1.0,
                "s_absolute": 1.0,
                "chunk_scale_normalization": 1.0,
                "c": 1.0,
                "accepted": True,
            }
            if refiner_pending:
                packet = self._finalize_pose_canonicalized_refiner_packet(packet)
        else:
            previous_id = int(self.window_order[-1])
            if previous_packet is None or int(previous_packet.window_id) != previous_id:
                raise RuntimeError("The previous full-resolution window packet is unavailable")
            if self.chunk_first_stride_graph:
                if start_frame not in self.graph.nodes:
                    raise RuntimeError(
                        f"Canonical chunk-first node {start_frame} is missing"
                    )
                previous_transform = self._window_anchor_transforms()[
                    previous_id
                ].to(packet.local_poses_c2w)
                bridge_input_packet = packet
                local_scale, alignment_diagnostics = (
                    self._estimate_canonical_ba_overlap_scale(
                        previous_packet,
                        bridge_input_packet,
                    )
                )
                if local_scale is None:
                    raise RuntimeError(
                        f"Window {window_id} BA-overlap scale failed: "
                        f"{alignment_diagnostics.get('reason', 'unknown')}"
                    )
                normalized_packet = self._rescaled_packet_copy(
                    bridge_input_packet, local_scale
                )
                start_transform = self.graph.transform(start_frame).clone().to(
                    normalized_packet.local_poses_c2w
                )
                overlap = self._overlap_frame_ids(
                    previous_packet,
                    normalized_packet,
                )
                known_global_poses = tuple(
                    self._known_overlap_global_pose(
                        previous_packet,
                        frame_id,
                        previous_transform,
                    )
                    for frame_id in overlap
                )
                predicted_second = apply_sim3_to_c2w(
                    start_transform,
                    normalized_packet.local_poses_c2w[1],
                )
                overlap_rotation_error = self._rotation_error_deg(
                    known_global_poses[1][:3, :3].to(predicted_second),
                    predicted_second[:3, :3],
                )
                overlap_center_error = float(
                    torch.linalg.norm(
                        known_global_poses[1][:3, 3].to(predicted_second)
                        - predicted_second[:3, 3]
                    )
                    .detach()
                    .cpu()
                )
                alignment_diagnostics.update(
                    {
                        "graph_role": "diagnostic_only_no_factor",
                        "existing_node_id": start_frame,
                        "raw_ba_to_canonical_rotation_error_deg": (
                            overlap_rotation_error
                        ),
                        "raw_ba_to_canonical_center_error": overlap_center_error,
                        "node_sim3_scale_updated": False,
                        "quality_gating_enabled": False,
                        "accepted": True,
                        "reason": "accepted_without_overlap_pose_gate",
                    }
                )
                packet = self._canonicalize_packet_from_two_known_poses(
                    normalized_packet,
                    start_transform,
                    (known_global_poses[0], known_global_poses[1]),
                )
                if refined_packet:
                    packet = self._finalize_pose_canonicalized_refiner_packet(
                        packet
                    )
                packet.metadata["global_alignment_local_scale"] = float(
                    local_scale
                )
                aligned = True
            elif self.two_frame_known_pose_bridge_enabled:
                previous_anchor = self.window_anchor_nodes[previous_id]
                previous_transform = self._window_anchor_transforms()[
                    previous_id
                ].to(packet.local_poses_c2w)
                bridge_input_packet = packet
                bridge = self._solve_known_pose_bridge(
                    previous_packet,
                    bridge_input_packet,
                    previous_transform,
                )
                packet = self._finalize_pose_canonicalized_refiner_packet(
                    bridge.packet
                )
                start_transform = bridge.owner_transform.to(
                    packet.local_poses_c2w
                )
                alignment_diagnostics = dict(bridge.diagnostics)
                if (
                    self.two_frame_bridge_depth_scale_enabled
                    and self.post_refiner_scale_recheck_enabled
                ):
                    post_frames = self._collect_known_pose_bridge_frames(
                        previous_packet,
                        packet,
                        previous_transform,
                        exclude_current_target_only=False,
                    )
                    post_scale, _, post_diagnostics = (
                        self._estimate_known_pose_bridge_scale(
                            post_frames,
                            previous_transform,
                            mode="depth",
                        )
                    )
                    post_scale_recheck_accepted = post_scale is not None
                    if post_scale is None:
                        post_scale = float(
                            sim3_components(start_transform)[0].detach().cpu()
                        )
                    alignment_diagnostics.update(
                        {
                            "post_refiner_scale_recheck_accepted": bool(
                                post_scale_recheck_accepted
                            ),
                            "post_refiner_scale_recheck_reason": str(
                                post_diagnostics.get("reason", "unknown")
                            ),
                            "post_refiner_candidate_scale": float(
                                post_diagnostics.get("absolute_scale", post_scale)
                            ),
                        }
                    )
                    pre_scale = float(
                        sim3_components(start_transform)[0].detach().cpu()
                    )
                    relative_change = abs(post_scale / pre_scale - 1.0)
                    alignment_diagnostics.update(
                        {
                            "post_refiner_scale": float(post_scale),
                            "post_refiner_scale_relative_change": float(
                                relative_change
                            ),
                            "post_refiner_scale_rerun": False,
                            **{
                                f"post_refiner_{key}": value
                                for key, value in post_diagnostics.items()
                                if key not in {"accepted", "reason"}
                            },
                        }
                    )
                    if (
                        relative_change
                        > self.post_refiner_scale_max_relative_change
                    ):
                        raise RuntimeError(
                            "Post-Refiner scale changed by "
                            f"{relative_change:.3f}, exceeding "
                            f"{self.post_refiner_scale_max_relative_change:.3f}"
                        )
                    if (
                        relative_change
                        > self.post_refiner_scale_rerun_threshold
                    ):
                        known_global = (
                            post_frames[0].known_global_pose,
                            post_frames[1].known_global_pose,
                        )
                        start_transform = self._bridge_owner_from_first_pose(
                            post_scale,
                            bridge_input_packet.local_poses_c2w[0],
                            known_global[0],
                        ).to(packet.local_poses_c2w)
                        rerun_packet = (
                            self._canonicalize_packet_from_two_known_poses(
                                bridge_input_packet,
                                start_transform,
                                known_global,
                            )
                        )
                        packet = self._finalize_pose_canonicalized_refiner_packet(
                            rerun_packet
                        )
                        alignment_diagnostics[
                            "post_refiner_scale_rerun"
                        ] = True
                        alignment_diagnostics["absolute_scale"] = float(
                            post_scale
                        )
                        alignment_diagnostics["s_absolute"] = float(post_scale)
                        previous_scale = float(
                            sim3_components(previous_transform)[0]
                            .detach()
                            .cpu()
                        )
                        alignment_diagnostics["measurement_scale"] = float(
                            post_scale / previous_scale
                        )
                        final_frames = self._collect_known_pose_bridge_frames(
                            previous_packet,
                            packet,
                            previous_transform,
                            exclude_current_target_only=False,
                        )
                        final_scale, _, final_scale_diagnostics = (
                            self._estimate_known_pose_bridge_scale(
                                final_frames,
                                previous_transform,
                                mode="depth",
                            )
                        )
                        final_scale_recheck_accepted = final_scale is not None
                        if final_scale is None:
                            final_scale = float(post_scale)
                        alignment_diagnostics.update(
                            {
                                "post_refiner_final_scale_recheck_accepted": bool(
                                    final_scale_recheck_accepted
                                ),
                                "post_refiner_final_scale_recheck_reason": str(
                                    final_scale_diagnostics.get(
                                        "reason", "unknown"
                                    )
                                ),
                                "post_refiner_final_candidate_scale": float(
                                    final_scale_diagnostics.get(
                                        "absolute_scale", final_scale
                                    )
                                ),
                            }
                        )
                        final_relative_change = abs(
                            final_scale / float(post_scale) - 1.0
                        )
                        alignment_diagnostics[
                            "post_refiner_final_scale"
                        ] = float(final_scale)
                        alignment_diagnostics[
                            "post_refiner_final_scale_relative_change"
                        ] = float(final_relative_change)
                        if (
                            final_relative_change
                            > self.post_refiner_scale_max_relative_change
                        ):
                            raise RuntimeError(
                                "Final post-Refiner scale remained unstable: "
                                f"{final_relative_change:.3f}"
                            )
                (
                    sequential_overlap_edge,
                    sequential_overlap_dense,
                    sequential_overlap_pose,
                    alignment_diagnostics,
                ) = self._known_pose_bridge_constraints(
                    previous_packet,
                    packet,
                    previous_anchor_node=previous_anchor,
                    current_anchor_node=start_frame,
                    previous_owner_transform=previous_transform,
                    current_owner_transform=start_transform,
                    alignment_diagnostics=alignment_diagnostics,
                )
                packet.metadata["global_alignment_local_scale"] = 1.0
                aligned = True
            elif self.two_frame_overlap_enabled:
                previous_anchor = self.window_anchor_nodes[previous_id]
                (
                    sequential_overlap_edge,
                    sequential_overlap_dense,
                    sequential_overlap_pose,
                    alignment_diagnostics,
                ) = self._two_frame_overlap_constraints(
                    previous_packet,
                    packet,
                    previous_anchor_node=previous_anchor,
                    current_anchor_node=start_frame,
                    use_rendered_anchors=refined_packet,
                )
                if sequential_overlap_edge is None:
                    self._last_overlap_alignment_failure = copy.deepcopy(
                        alignment_diagnostics
                    )
                    raise RuntimeError(
                        f"Window {window_id} two-frame alignment failed: "
                        f"{alignment_diagnostics.get('reason', 'unknown')}"
                    )
                previous_transform = self.graph.transform(previous_anchor)
                start_transform = (
                    previous_transform
                    @ sequential_overlap_edge.measurement_target_to_source.to(
                        previous_transform
                    )
                )
                packet.metadata["global_alignment_local_scale"] = 1.0
                aligned = True
            else:
                if int(previous_packet.frame_ids[-1]) != start_frame:
                    raise RuntimeError(
                        f"Boundary continuity violated: previous end={previous_packet.frame_ids[-1]} "
                        f"current start={start_frame}"
                    )
                if start_frame not in self.graph.nodes:
                    raise RuntimeError(f"Shared boundary node {start_frame} is missing")
            if (
                refined_packet
                and not self.two_frame_overlap_enabled
                and not self.chunk_first_stride_graph
            ):
                start_transform = self.graph.transform(start_frame).clone()
                local_scale, alignment_diagnostics = (
                    self._rendered_shared_frame_alignment(
                        previous_packet,
                        packet,
                        start_transform,
                    )
                )
                if local_scale is None:
                    raise RuntimeError(
                        f"Window {window_id} rendered shared-frame scale alignment failed: "
                        f"{alignment_diagnostics.get('reason', 'unknown')}"
                    )
                packet = self._rescaled_packet_copy(packet, local_scale)
                packet.pre_depth_shift_depth = None
                aligned = True
            elif (
                not refined_packet
                and not self.two_frame_overlap_enabled
                and not self.chunk_first_stride_graph
            ):
                alignment_attempts: list[dict[str, Any]] = []
                measurement, attempt_diagnostics = self._shared_frame_alignment(
                    previous_packet, packet
                )
                alignment_attempts.append(
                    {"stage": "ba_pose_shifted_depth", **attempt_diagnostics}
                )
                recovery_stage = "ba_pose_shifted_depth"
                if measurement is None and self._restore_packet_pre_shift_depth(packet):
                    measurement, attempt_diagnostics = self._shared_frame_alignment(
                        previous_packet, packet
                    )
                    alignment_attempts.append(
                        {"stage": "ba_pose_depth_shift_rollback", **attempt_diagnostics}
                    )
                    recovery_stage = "ba_pose_depth_shift_rollback"
                if measurement is None:
                    self._synchronize_shared_canonical_depth(
                        previous_packet,
                        packet,
                        start_frame,
                    )
                    measurement, attempt_diagnostics = self._shared_frame_alignment(
                        previous_packet, packet
                    )
                    alignment_attempts.append(
                        {"stage": "canonical_depth_retry", **attempt_diagnostics}
                    )
                    recovery_stage = "canonical_depth_retry"
                alignment_diagnostics = dict(attempt_diagnostics)
                alignment_diagnostics.update(
                    {
                        "alignment_attempts": alignment_attempts,
                        "alignment_recovery_stage": recovery_stage,
                        "depth_shift_rollback": bool(
                            packet.metadata.get("depth_shift_rollback", False)
                        ),
                    }
                )
                if measurement is None:
                    if not self.allow_unaligned_fallback:
                        raise RuntimeError(
                            f"Window {window_id} cannot be aligned to previous window {previous_id}: "
                            f"{alignment_diagnostics.get('reason', 'unknown')}"
                        )
                    start_transform = self.graph.transform(start_frame).clone()
                    aligned = False
                else:
                    previous_anchor = self.window_anchor_nodes[previous_id]
                    raw_start_transform = (
                        self.graph.transform(previous_anchor)
                        @ measurement.to(self.graph.transform(previous_anchor))
                    )
                    start_transform = self.graph.transform(start_frame).clone()
                    canonicalization = sim3_inverse(start_transform) @ raw_start_transform
                    scale, rotation, translation = sim3_components(canonicalization)
                    local_scale = float(scale.detach().cpu())
                    self._rescale_packet_geometry(packet, local_scale)
                    self._synchronize_shared_canonical_depth(
                        previous_packet,
                        packet,
                        start_frame,
                    )
                    packet.pre_depth_shift_depth = None
                    rotation_trace = rotation.diagonal().sum()
                    rotation_angle = torch.acos(
                        ((rotation_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
                    )
                    alignment_diagnostics.update(
                        {
                            "chunk_scale_normalization": local_scale,
                            "canonical_rotation_mismatch_deg": float(
                                torch.rad2deg(rotation_angle).detach().cpu()
                            ),
                            "canonical_translation_mismatch": float(
                                translation.norm().detach().cpu()
                            ),
                        }
                    )
                    aligned = True

        prefusion_pose_diagnostics: dict[str, Any] = {
            "enabled": False,
            "reason": "legacy_single_stage_packet",
        }
        if prepared_candidate is not None:
            packet, prefusion_pose_diagnostics = (
                self._run_prefusion_pose_tracking(packet, start_transform)
            )
            alignment_diagnostics["prefusion_pose_tracking"] = dict(
                prefusion_pose_diagnostics
            )

        # Recovery state is needed only while this packet is the incoming
        # target.  Once admitted it becomes the canonical source for the next
        # boundary and the dense backup can be released.
        packet.pre_depth_shift_depth = None
        skip_factor_added = False

        if self.chunk_first_stride_graph:
            (
                stride_factor,
                stride_measurement,
                stride_holdout,
                boundary_diagnostics,
            ) = self._chunk_stride_factor(
                packet,
                expected_target_index=self.chunk_stride_target_index,
            )
            if (
                stride_factor is None
                or stride_measurement is None
            ):
                raise RuntimeError(
                    f"Window {window_id} cannot construct a finite stride edge: "
                    f"{boundary_diagnostics.get('reason', 'unknown')}"
                )
            next_frame = int(packet.frame_ids[self.chunk_stride_target_index])
            if not self.window_order:
                self.graph.add_node(start_frame, start_transform)
                self.boundary_node_order.append(start_frame)
                self._chunk_node_initial_scale[start_frame] = float(
                    sim3_components(start_transform)[0].detach().cpu()
                )
            elif start_frame not in self.graph.nodes:
                raise RuntimeError(
                    f"Expected existing chunk-first node {start_frame}"
                )
            if next_frame in self.graph.nodes:
                raise RuntimeError(
                    f"Next chunk-first node {next_frame} already exists before "
                    f"window {window_id}"
                )
            start_node_transform = self.graph.transform(start_frame)
            stride_ba_pose = canonicalize_c2w(
                invert_c2w(packet.local_poses_c2w[0])
                @ packet.local_poses_c2w[self.chunk_stride_target_index]
            ).to(start_node_transform)
            # Packet-to-packet overlap depth has already canonicalized the
            # packet.  Pure-chain initialization therefore uses the
            # canonical BA (0 -> stride) SE(3) motion and inherits the parent
            # node scale exactly.  Umeyama s/R/t remains a correspondence
            # quality, direction and holdout diagnostic only.
            next_transform = (
                start_node_transform
                @ stride_ba_pose
            )
            boundary_diagnostics["node_initialization_source"] = (
                "canonical_ba_stride_pose"
            )
            boundary_diagnostics["umeyama_used_for_node_initialization"] = False
            self.graph.add_node(next_frame, next_transform)
            self.boundary_node_order.append(next_frame)
            self._chunk_node_initial_scale[next_frame] = float(
                sim3_components(next_transform)[0].detach().cpu()
            )
            self.graph.add_edge(stride_factor)
            if stride_holdout is not None:
                self._chunk_stride_holdouts[
                    (start_frame, next_frame, stride_holdout.edge_type)
                ] = stride_holdout
            skip_diagnostics: dict[str, Any] = {
                "enabled": bool(self.chunk_skip_enabled),
                "reason": "first_window",
            }
            if self.window_order and self._last_full_packet is not None:
                skip_factor, skip_holdout, skip_diagnostics = (
                    self._independent_chunk_skip_factor(
                        self._last_full_packet,
                        packet,
                    )
                )
                if skip_factor is not None and skip_holdout is not None:
                    self.graph.add_edge(skip_factor)
                    self._chunk_stride_holdouts[
                        (
                            skip_holdout.source,
                            skip_holdout.target,
                            skip_holdout.edge_type,
                        )
                    ] = skip_holdout
                    skip_factor_added = True
            boundary_diagnostics["skip_edge"] = skip_diagnostics
            self.window_anchor_nodes[window_id] = start_frame
            self.frame_owner_window.setdefault(start_frame, window_id)
            self.frame_depth_owner_window.setdefault(start_frame, window_id)
            self.frame_owner_window.setdefault(next_frame, window_id)
            self.frame_depth_owner_window.setdefault(next_frame, window_id)
            segment_diagnostics = self._register_chunk_stride_segments(
                packet,
                source_node=start_frame,
                target_node=next_frame,
            )
            boundary_diagnostics.update(segment_diagnostics)
            self._sequential_edges_since_optimization += 1
            self._register_hierarchical_window(
                window_id, start_frame, next_frame
            )
        else:
            boundary_factor, boundary_diagnostics = self._boundary_factor(packet)
            boundary_pose_fallback: Sim3GraphEdge | None = None
            if boundary_factor is None:
                if self.two_frame_known_pose_bridge_enabled:
                    boundary_pose_fallback = self._boundary_local_pose_fallback_edge(
                        packet,
                        boundary_diagnostics,
                    )
                    boundary_diagnostics = dict(boundary_pose_fallback.metadata)
                elif not self.allow_unaligned_fallback:
                    raise RuntimeError(
                        f"Window {window_id} has no valid first/last spherical factor: "
                        f"{boundary_diagnostics.get('reason', 'unknown')}"
                    )

            if (
                self.two_frame_overlap_enabled
                and self.window_order
                and start_frame in self.graph.nodes
            ):
                raise ValueError(
                    f"Independent overlap-2 anchor node {start_frame} already exists "
                    f"before window {window_id}"
                )
            if not self.window_order:
                self.graph.add_node(start_frame, start_transform)
                self.boundary_node_order.append(start_frame)
            elif start_frame not in self.graph.nodes:
                self.graph.add_node(start_frame, start_transform)
                self.boundary_node_order.append(start_frame)
            self.window_anchor_nodes[window_id] = start_frame
            if self.two_frame_overlap_enabled:
                self.frame_owner_window[start_frame] = window_id
                self.frame_depth_owner_window[start_frame] = window_id
            else:
                self.frame_owner_window.setdefault(start_frame, window_id)
                self.frame_depth_owner_window.setdefault(start_frame, window_id)

            if end_frame in self.graph.nodes and end_frame != start_frame:
                raise ValueError(
                    f"Boundary frame node {end_frame} already exists before window {window_id}"
                )
            if end_frame not in self.graph.nodes:
                end_transform = self._node_from_local_pose(
                    self.graph.transform(start_frame), packet.local_poses_c2w[-1]
                )
                self.graph.add_node(end_frame, end_transform)
                self.boundary_node_order.append(end_frame)
            self.window_end_nodes[window_id] = end_frame
            self.frame_owner_window.setdefault(end_frame, window_id)
            self.frame_depth_owner_window.setdefault(end_frame, window_id)
            if sequential_overlap_edge is not None:
                self.graph.add_edge(sequential_overlap_edge)
            for factor in sequential_overlap_dense:
                self.graph.add_edge(factor)
            for factor in sequential_overlap_pose:
                self.graph.add_edge(factor)
            if boundary_factor is not None:
                self.graph.add_edge(boundary_factor)
            elif boundary_pose_fallback is not None:
                self.graph.add_edge(boundary_pose_fallback)
            if (
                boundary_factor is not None
                or boundary_pose_fallback is not None
                or sequential_overlap_edge is not None
            ):
                self._sequential_edges_since_optimization += 1
            self._register_hierarchical_window(window_id, start_frame, end_frame)

        # Loop retrieval operates on window packets. Verified pose and dense
        # spherical measurements are both re-keyed to boundary/submap nodes;
        # count normalization prevents the correlated blocks from being
        # over-weighted merely because they contain many correspondences.
        loop_results = self.loop_detector.detect(packet)
        loop_graph = (
            self.submap_graph
            if self.hierarchical_submaps_enabled
            else self.graph
        )
        assert loop_graph is not None
        accepted_loops: list[PanoramaLoopVerification] = []
        pending_loop_factors: list[
            DenseSphericalFactorBlock | Sim3GraphEdge | CoincidentPanoramaFactor
        ] = []
        loop_edge_start = len(loop_graph.edges)
        for loop_result in loop_results:
            if not loop_result.accepted or not loop_result.dense_factors:
                continue
            pair = self._canonical_loop_pair(loop_result)
            if pair in self.accepted_loop_pairs:
                self._reject_loop_result(loop_result, "duplicate_loop_pair")
                continue
            loop_measurement = (
                self._loop_measurement_for_submaps(loop_result)
                if self.hierarchical_submaps_enabled
                else self._loop_measurement_for_boundary_nodes(loop_result)
            )
            if loop_measurement is None:
                self._reject_loop_result(loop_result, "intra_submap_or_missing_loop_measurement")
                continue
            if self.hierarchical_submaps_enabled:
                source_node = int(loop_measurement.source)
                target_node = int(loop_measurement.target)
            else:
                source_node = self.window_anchor_nodes.get(int(loop_result.source_window_id))
                target_node = self.window_anchor_nodes.get(int(loop_result.target_window_id))
            if self.loop_transaction_enabled:
                if source_node is None or target_node is None:
                    self._reject_loop_result(loop_result, "missing_loop_graph_endpoint")
                    continue
                path_ok, path_diagnostics = self._loop_path_consistency(
                    loop_result,
                    source_node=int(source_node),
                    target_node=int(target_node),
                    graph=loop_graph,
                    measurement=loop_measurement,
                )
                loop_result.metadata["path_consistency"] = path_diagnostics
                if not path_ok:
                    self._reject_loop_result(loop_result, "path_inconsistent")
                    continue
            merged = (
                self._merge_loop_submap_dense_factors(loop_result)
                if self.hierarchical_submaps_enabled
                else self._merge_loop_dense_factors(loop_result)
            )
            if merged is not None and merged.source != merged.target:
                pose_factor = (
                    self._materialize_loop_pose_factor(loop_measurement)
                    if self.insert_loop_pose_factor
                    else None
                )
                if pose_factor is not None:
                    loop_graph.add_edge(pose_factor)
                loop_graph.add_edge(merged)
                accepted_loops.append(loop_result)
                if pose_factor is not None:
                    pending_loop_factors.append(pose_factor)
                pending_loop_factors.append(merged)

        old_window_transforms = self._window_anchor_transforms()
        chunk_sequence_factors = tuple(
            factor
            for factor in self.graph.edges
            if getattr(factor, "edge_type", "")
            == "chunk_stride_dense_spherical"
        )
        chunk_sequence_objective_before = (
            float(
                self.graph.objective(factors=chunk_sequence_factors)
                .detach()
                .cpu()
            )
            if self.chunk_first_stride_graph and chunk_sequence_factors
            else 0.0
        )
        pre_loop_nodes = {node: value.clone() for node, value in loop_graph.nodes.items()}
        pre_boundary_graph_state = self._snapshot_graph_state(self.graph)
        pre_submap_graph_state = self._snapshot_graph_state(self.submap_graph)
        graph_result: Sim3GraphOptimizeResult | None = None
        seam_diagnostics: dict[str, Any] = {
            "enabled": bool(
                self.chunk_first_stride_graph
                or self.post_optimization_seam_check_enabled
            ),
            "mode": (
                "canonical_pose_state_consistency"
                if self.chunk_first_stride_graph
                else "legacy_overlap_seam"
            ),
            "enforced": bool(self.chunk_first_stride_graph),
            "quality_gating_enabled": False,
            "accepted": True,
            "factor_count": 0,
            "candidate_revision": int(self._pose_state_revision),
            "committed_revision": int(self._pose_state_revision),
        }
        recent_window_count = len(self.window_order) + 1
        periodic_start_ready = (
            len(self.boundary_node_order) >= self.global_ba_start_nodes
            if self.chunk_first_stride_graph
            else (
                recent_window_count >= self.global_ba_start_nodes
                if self.two_frame_overlap_enabled
                else len(self.boundary_node_order) >= self.global_ba_start_nodes
            )
        )
        should_optimize_recent = (
            self.global_graph_optimization_enabled
            and self.graph_optimization_trigger == "periodic_and_loop"
            and periodic_start_ready
            and (
                not self._has_run_global_ba
                or self._sequential_edges_since_optimization >= self.global_ba_interval_edges
            )
        )

        def optimize_recent_periodic_graph() -> Sim3GraphOptimizeResult:
            active = (
                self._recent_boundary_window_nodes(
                    include_window_id=window_id
                )
                if self.two_frame_overlap_enabled
                or self.chunk_first_stride_graph
                else self.boundary_node_order[-self.global_ba_active_nodes :]
            )
            periodic_scale_lock = bool(self.graph.lock_scale_updates)
            try:
                if self.chunk_first_stride_graph:
                    # Adjacent stride edges do not make per-node scale
                    # independently observable.  Periodic local BA therefore
                    # refines only R/t; accepted loop transactions retain their
                    # separate log-scale path.
                    self.graph.lock_scale_updates = True
                result = self.graph.optimize(
                    active,
                    fixed_node_ids={active[0]},
                )
            finally:
                self.graph.lock_scale_updates = periodic_scale_lock
            boundary_diagnostics["periodic_optimization"] = {
                "trigger": "recent_chunk_cadence",
                "active_node_ids": [int(node) for node in active],
                "fixed_node_id": int(active[0]),
                "scale_locked": bool(self.chunk_first_stride_graph),
                "attempted": True,
                "accepted": bool(result.accepted),
                "reason": str(result.reason),
            }
            self._has_run_global_ba = True
            self._sequential_edges_since_optimization = 0
            return result

        if accepted_loops and not self.global_graph_optimization_enabled:
            for loop_result in accepted_loops:
                self.accepted_loop_pairs.add(
                    self._canonical_loop_pair(loop_result)
                )
                loop_result.metadata["graph_transaction"] = {
                    "enabled": False,
                    "committed": False,
                    "reason": "global_graph_optimization_disabled",
                }
        elif accepted_loops:
            nonloop_factors = tuple(loop_graph.edges[:loop_edge_start])
            nonloop_before = float(
                loop_graph.objective(factors=nonloop_factors).detach().cpu()
            )
            loop_scale_lock = bool(loop_graph.lock_scale_updates)
            try:
                if self.chunk_first_stride_graph:
                    loop_graph.lock_scale_updates = False
                graph_result = loop_graph.optimize()
            finally:
                loop_graph.lock_scale_updates = loop_scale_lock
            commit = True
            minimum_dcs_scale = 1.0
            nonloop_objective_ratio = 1.0
            cumulative_scale_ratio = 1.0
            if self.loop_transaction_enabled:
                nonloop_after = float(
                    loop_graph.objective(factors=nonloop_factors).detach().cpu()
                )
                (
                    commit,
                    minimum_dcs_scale,
                    nonloop_objective_ratio,
                ) = self._loop_transaction_commit_ok(
                    graph_result,
                    list(pending_loop_factors),
                    nonloop_objective_before=nonloop_before,
                    nonloop_objective_after=nonloop_after,
                )
            if self.chunk_first_stride_graph and loop_graph is self.graph:
                for node, initial_scale in self._chunk_node_initial_scale.items():
                    if int(node) not in self.graph.nodes:
                        continue
                    current_scale = float(
                        sim3_components(self.graph.transform(int(node)))[0]
                        .detach()
                        .cpu()
                    )
                    ratio = max(
                        current_scale / max(initial_scale, 1.0e-12),
                        initial_scale / max(current_scale, 1.0e-12),
                    )
                    cumulative_scale_ratio = max(
                        cumulative_scale_ratio, ratio
                    )
            for loop_result in accepted_loops:
                loop_result.metadata["graph_transaction"] = {
                    "enabled": self.loop_transaction_enabled,
                    "committed": bool(commit),
                    "minimum_dcs_scale": float(minimum_dcs_scale),
                    "nonloop_objective_ratio": float(nonloop_objective_ratio),
                    "max_cumulative_scale_ratio": float(
                        cumulative_scale_ratio
                    ),
                    "cumulative_scale_gating_enabled": False,
                    "initial_objective": float(graph_result.initial_objective),
                    "final_objective": float(graph_result.final_objective),
                }
            if commit:
                for loop_result in accepted_loops:
                    self.accepted_loop_pairs.add(self._canonical_loop_pair(loop_result))
                self._has_run_global_ba = True
                self._sequential_edges_since_optimization = 0
                if self.hierarchical_submaps_enabled:
                    self._apply_submap_graph_to_boundary_graph()
            else:
                loop_graph.nodes = pre_loop_nodes
                del loop_graph.edges[loop_edge_start:]
                for loop_result in accepted_loops:
                    self._reject_loop_result(loop_result, "loop_transaction_rejected")
                accepted_loops = []
                graph_result = replace(
                    graph_result,
                    accepted=False,
                    final_objective=graph_result.initial_objective,
                    max_update_norm=0.0,
                    reason="loop_transaction_rejected",
                )
                if should_optimize_recent:
                    graph_result = optimize_recent_periodic_graph()
        elif (
            skip_factor_added
            and self.chunk_first_stride_graph
            and self.global_graph_optimization_enabled
        ):
            active = self._recent_boundary_window_nodes(
                include_window_id=window_id
            )
            for node in (start_frame, next_frame):
                if int(node) not in active:
                    active.append(int(node))
            sequence_factors = tuple(
                factor
                for factor in self.graph.edges
                if getattr(factor, "edge_type", "")
                == "chunk_stride_dense_spherical"
            )
            sequence_before = float(
                self.graph.objective(factors=sequence_factors).detach().cpu()
            )
            scale_lock = bool(self.graph.lock_scale_updates)
            try:
                # A pure chain never enters this branch. The independent skip
                # edge creates the first observable cycle, at which point R/t
                # and unconstrained log-scale steps may be optimized together.
                self.graph.lock_scale_updates = False
                graph_result = self.graph.optimize(
                    active,
                    fixed_node_ids={active[0]},
                )
            finally:
                self.graph.lock_scale_updates = scale_lock
            sequence_after = float(
                self.graph.objective(factors=sequence_factors).detach().cpu()
            )
            sequence_ratio = sequence_after / max(sequence_before, 1.0e-12)
            cumulative_scale_ratio = 1.0
            for node in active:
                initial_scale = self._chunk_node_initial_scale.get(int(node))
                if initial_scale is None:
                    continue
                current_scale = float(
                    sim3_components(self.graph.transform(int(node)))[0]
                    .detach()
                    .cpu()
                )
                ratio = max(
                    current_scale / max(initial_scale, 1.0e-12),
                    initial_scale / max(current_scale, 1.0e-12),
                )
                cumulative_scale_ratio = max(cumulative_scale_ratio, ratio)
            cycle_commit = bool(graph_result.accepted)
            boundary_diagnostics["cycle_optimization"] = {
                "trigger": "independent_skip_edge",
                "committed": bool(cycle_commit),
                "quality_gating_enabled": False,
                "sequence_objective_before": sequence_before,
                "sequence_objective_after": sequence_after,
                "sequence_objective_ratio": sequence_ratio,
                "max_cumulative_scale_ratio": cumulative_scale_ratio,
            }
            if cycle_commit:
                self._has_run_global_ba = True
                self._sequential_edges_since_optimization = 0
            else:
                self._restore_graph_state(
                    self.graph, pre_boundary_graph_state
                )
                graph_result = replace(
                    graph_result,
                    accepted=False,
                    final_objective=graph_result.initial_objective,
                    max_update_norm=0.0,
                    reason="chunk_cycle_transaction_rejected",
                )
        elif should_optimize_recent:
            graph_result = optimize_recent_periodic_graph()
        chunk_scale_diagnostics: dict[str, Any] = {
            "enabled": bool(self.chunk_first_stride_graph),
            "accepted": True,
            "max_cumulative_scale_ratio": 1.0,
        }
        if graph_result is not None and self.chunk_first_stride_graph:
            maximum_ratio = 1.0
            for node, initial_scale in self._chunk_node_initial_scale.items():
                if int(node) not in self.graph.nodes:
                    continue
                current_scale = float(
                    sim3_components(self.graph.transform(int(node)))[0]
                    .detach()
                    .cpu()
                )
                ratio = max(
                    current_scale / max(initial_scale, 1.0e-12),
                    initial_scale / max(current_scale, 1.0e-12),
                )
                maximum_ratio = max(maximum_ratio, ratio)
            chunk_scale_diagnostics = {
                "enabled": True,
                "accepted": True,
                "quality_gating_enabled": False,
                "max_cumulative_scale_ratio": maximum_ratio,
            }
        chunk_sequence_diagnostics: dict[str, Any] = {
            "enabled": bool(self.chunk_first_stride_graph),
            "enforced": False,
            "quality_gating_enabled": False,
            "accepted": True,
            "factor_count": len(chunk_sequence_factors),
            "objective_before": chunk_sequence_objective_before,
            "objective_after": chunk_sequence_objective_before,
            "objective_ratio": 1.0,
        }
        if graph_result is not None and self.chunk_first_stride_graph:
            sequence_after = (
                float(
                    self.graph.objective(factors=chunk_sequence_factors)
                    .detach()
                    .cpu()
                )
                if chunk_sequence_factors
                else 0.0
            )
            sequence_ratio = sequence_after / max(
                chunk_sequence_objective_before, 1.0e-12
            )
            chunk_sequence_diagnostics = {
                "enabled": True,
                "enforced": False,
                "quality_gating_enabled": False,
                "accepted": True,
                "factor_count": len(chunk_sequence_factors),
                "objective_before": chunk_sequence_objective_before,
                "objective_after": sequence_after,
                "objective_ratio": sequence_ratio,
                "objective_finite": math.isfinite(sequence_after),
            }
        if (
            graph_result is not None
            and not self.chunk_first_stride_graph
            and self.post_optimization_seam_check_enabled
        ):
            seam_diagnostics = self._overlap_seam_diagnostics()
            seam_diagnostics["enforced"] = False
        stride_holdout_diagnostics: dict[str, Any] = {
            "enabled": bool(self.chunk_first_stride_graph),
            "enforced": False,
            "quality_gating_enabled": False,
            "accepted": True,
            "factor_count": 0,
        }
        if graph_result is not None and self.chunk_first_stride_graph:
            affected_holdout_nodes = {
                int(node) for node in graph_result.optimized_node_ids
            }
            if (
                self.hierarchical_submaps_enabled
                and loop_graph is self.submap_graph
            ):
                affected_holdout_nodes = {
                    int(node)
                    for submap_id in graph_result.optimized_node_ids
                    if int(submap_id) in self.submaps
                    for node in self.submaps[int(submap_id)].boundary_node_ids
                }
            stride_holdout_diagnostics = (
                self._chunk_stride_holdout_diagnostics(
                    affected_node_ids=affected_holdout_nodes
                )
            )
            stride_holdout_diagnostics["enforced"] = False
        correction: dict[str, int] = {"moved": 0, "deduplicated": 0}
        if (
            graph_result is not None
            and bool(graph_result.accepted)
            and self.chunk_first_stride_graph
        ):
            affected_submap_ids: set[int] = set()
            if (
                self.hierarchical_submaps_enabled
                and loop_graph is self.submap_graph
            ):
                affected_submap_ids = {
                    int(value) for value in graph_result.optimized_node_ids
                }
                affected_pose_nodes = {
                    int(node)
                    for submap_id in affected_submap_ids
                    if submap_id in self.submaps
                    for node in self.submaps[submap_id].boundary_node_ids
                }
            else:
                affected_pose_nodes = {
                    int(value) for value in graph_result.optimized_node_ids
                }
            if affected_pose_nodes:
                _, state_report, correction = (
                    self._materialize_pose_state_candidate(
                        affected_node_ids=affected_pose_nodes,
                        affected_submap_ids=affected_submap_ids,
                        old_window_transforms=old_window_transforms,
                        reason="chunk_graph_optimization_candidate",
                        extra_packets=(packet,),
                    )
                )
                seam_diagnostics = state_report.as_diagnostics()
                seam_diagnostics["enforced"] = True
                seam_diagnostics["transaction_committed"] = True
        elif prepared_candidate is not None and self.chunk_first_stride_graph:
            # A pure-chain append still commits new canonical camera state.
            # Materialize that state before Refiner/Hash/fusion so every
            # downstream consumer observes one shared pose revision even when
            # no graph LM happened in this window.
            _, state_report, correction = self._materialize_pose_state_candidate(
                affected_node_ids={int(start_frame), int(next_frame)},
                old_window_transforms=old_window_transforms,
                reason="prefusion_pose_tracking_candidate",
                extra_packets=(packet,),
            )
            seam_diagnostics = state_report.as_diagnostics()
            seam_diagnostics["enforced"] = True
            seam_diagnostics["transaction_committed"] = True
        if self.hierarchical_submaps_enabled and self._active_submap_id is not None:
            self._update_submap_local_geometry(self._active_submap_id)
        submap_frozen = self._freeze_active_submap_if_ready()
        new_window_transforms = self._window_anchor_transforms()
        if not self.chunk_first_stride_graph:
            correction = (
                self.fusion.apply_owner_corrections(
                    old_window_transforms,
                    new_window_transforms,
                )
                if graph_result is not None
                and self._owner_transforms_changed(
                    old_window_transforms, new_window_transforms
                )
                else {"moved": 0, "deduplicated": 0}
            )

        window_transform = self._window_anchor_transforms()[window_id]
        if (
            prepared_candidate is not None
            and refined_packet
            and refiner_pending
        ):
            # Refiner must observe the final packet pose after both RGB
            # tracking and any graph LM accepted in this transaction.
            packet = self._finalize_pose_canonicalized_refiner_packet(packet)
            if bool(packet.metadata.get("voxel_anchor_refiner_pending", False)):
                raise RuntimeError(
                    "Final Refiner still reports a pending packet"
                )
            packet.metadata["refiner_pose_revision"] = int(
                self._pose_state_revision
            )
            packet.metadata["refiner_after_prefusion_tracking"] = True
        if refined_packet:
            prepare_start = time.perf_counter()
            prepared = self.fusion.prepare_packet_batch(packet, window_transform)
            prepare_seconds = float(time.perf_counter() - prepare_start)
            support_requested = len(prepared.batch)
            support_kept = support_requested
            if (
                self.insertion_dedup_require_new_frame_support
            ):
                source_view_mask = packet.metadata.get(
                    "voxel_anchor_source_view_mask"
                )
                if not torch.is_tensor(source_view_mask):
                    raise RuntimeError(
                        "New-frame-supported fusion requires the Refiner "
                        "source-view support mask"
                    )
                if prepared.source_anchor_indices is None:
                    raise RuntimeError(
                        "New-frame-supported fusion requires source anchor indices"
                    )
                previous_packet = self._last_full_packet
                overlap_ids = (
                    set()
                    if previous_packet is None
                    else set(self._overlap_frame_ids(previous_packet, packet))
                )
                new_indices = [
                    index
                    for index, frame_id in enumerate(packet.frame_ids)
                    if int(frame_id) not in overlap_ids
                ]
                if not new_indices:
                    raise RuntimeError(
                        "The incoming packet has no non-overlap frame support"
                    )
                new_bits = sum(1 << int(index) for index in new_indices)
                support = source_view_mask.to(
                    device=prepared.source_anchor_indices.device,
                    dtype=torch.long,
                )
                source_rows = prepared.source_anchor_indices.to(support.device)
                keep_support = (
                    support.index_select(0, source_rows) & int(new_bits)
                ) != 0
                selected = torch.nonzero(
                    keep_support, as_tuple=False
                ).flatten().to(prepared.batch.xyz.device)
                prepared = prepared.index(selected)
                support_kept = len(prepared.batch)
            hash_stats: dict[str, int | float] = {
                "hash_requested": int(prepared.requested),
                "hash_candidates": len(prepared.batch),
                "hash_visible_incoming": 0,
                "hash_visible_existing": 0,
                "hash_hits": 0,
                "hash_kept": len(prepared.batch),
                "hash_radius_voxels": self.insertion_dedup_radius_voxels,
                "hash_visibility_views": 0,
                "new_frame_support_requested": support_requested,
                "new_frame_support_kept": support_kept,
                "new_frame_support_dropped": (
                    support_requested - support_kept
                ),
            }
            for level in range(len(self.fusion.voxel_sizes)):
                level_incoming = int(
                    (prepared.batch.level == level).sum().detach().cpu()
                )
                hash_stats[f"hash_level_{level}_incoming"] = level_incoming
                hash_stats[f"hash_level_{level}_visible"] = 0
                hash_stats[f"hash_level_{level}_hits"] = 0
                hash_stats[f"hash_level_{level}_kept"] = level_incoming
            evidence_update = None
            insertion_render_seconds = 0.0
            hash_seconds = 0.0
            hash_visibility_views = 0
            posthash_coverage_context: tuple[
                tuple[int, ...],
                torch.Tensor,
                set[int],
                tuple[int, int],
            ] | None = None
            has_existing_map = self.map.anchor_count() > 0
            require_four_view_admission = bool(
                self.insertion_dedup_require_new_frame_support
                and len(packet.frame_ids) == 4
            )
            render_insertion_visibility = bool(
                require_four_view_admission
                or (self.insertion_dedup_enabled and has_existing_map)
            )
            if render_insertion_visibility:
                assert packet.anchor_observation is not None
                use_all_packet_views = bool(
                    require_four_view_admission
                    or self.chunk_first_stride_graph
                )
                if require_four_view_admission or self.two_frame_overlap_enabled:
                    if use_all_packet_views:
                        visibility_frame_ids = tuple(
                            int(value) for value in packet.frame_ids
                        )
                    else:
                        previous_packet = self._last_full_packet
                        if previous_packet is None:
                            raise RuntimeError(
                                "Two-frame insertion dedup requires the previous packet"
                            )
                        visibility_frame_ids = self._overlap_frame_ids(
                            previous_packet, packet
                        )
                    global_poses = packet.global_poses(window_transform)
                    incoming_visibility = torch.zeros(
                        packet.anchor_observation.num_anchors,
                        device=packet.anchor_observation.xyz.device,
                        dtype=torch.bool,
                    )
                    existing_visibility = torch.zeros(
                        self.map.anchor_count(),
                        device=self.map.xyz.device,
                        dtype=torch.bool,
                    )
                    diagnostic_overlap_ids = (
                        set(
                            self._overlap_frame_ids(
                                previous_packet, packet
                            )
                        )
                        if self.chunk_first_stride_graph
                        and previous_packet is not None
                        else set()
                    )
                    if (
                        self.insertion_dedup_log_posthash_coverage
                        and has_existing_map
                    ):
                        posthash_coverage_context = (
                            tuple(int(value) for value in visibility_frame_ids),
                            global_poses.detach().clone(),
                            set(diagnostic_overlap_ids),
                            tuple(
                                int(value)
                                for value in packet.anchor_observation.image_size
                            ),
                        )
                    diagnostic_rows: list[dict[str, torch.Tensor]] = []
                    coverage_by_group: dict[str, list[tuple[float, ...]]] = {
                        "overlap": [],
                        "new_frame": [],
                    }
                    owner_scale = sim3_components(window_transform)[0]
                    for frame_id in visibility_frame_ids:
                        incoming_render = self._render_refined_anchor_frame(
                            packet, frame_id
                        )
                        existing_render = (
                            self._render_global_pose_frame(
                                global_poses[packet.frame_index(frame_id)],
                                image_size=packet.anchor_observation.image_size,
                            )
                            if has_existing_map
                            else None
                        )
                        incoming_visibility |= (
                            incoming_render.anchor_visibility.to(
                                incoming_visibility.device
                            )
                        )
                        if existing_render is not None:
                            existing_visibility |= (
                                existing_render.anchor_visibility.to(
                                    existing_visibility.device
                                )
                            )
                        insertion_render_seconds += incoming_render.render_seconds
                        if existing_render is not None:
                            insertion_render_seconds += existing_render.render_seconds
                        incoming_depth_valid = (
                            torch.isfinite(incoming_render.depth)
                            & (incoming_render.depth > 0.0)
                        )
                        incoming_alpha_valid = (
                            torch.isfinite(incoming_render.alpha)
                            & (
                                incoming_render.alpha
                                >= self.rendered_alignment_alpha_threshold
                            )
                        )
                        existing_depth_valid = (
                            torch.isfinite(existing_render.depth)
                            & (existing_render.depth > 0.0)
                            if existing_render is not None
                            else torch.zeros_like(incoming_depth_valid)
                        )
                        existing_alpha_valid = (
                            torch.isfinite(existing_render.alpha)
                            & (
                                existing_render.alpha
                                >= self.rendered_alignment_alpha_threshold
                            )
                            if existing_render is not None
                            else torch.zeros_like(incoming_alpha_valid)
                        )
                        coverage_values = (
                            float(incoming_depth_valid.float().mean().detach().cpu()),
                            float(incoming_alpha_valid.float().mean().detach().cpu()),
                            float(
                                (incoming_depth_valid & incoming_alpha_valid)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            ),
                            float(existing_depth_valid.float().mean().detach().cpu()),
                            float(existing_alpha_valid.float().mean().detach().cpu()),
                            float(
                                (existing_depth_valid & existing_alpha_valid)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            ),
                        )
                        coverage_group = (
                            "overlap"
                            if int(frame_id) in diagnostic_overlap_ids
                            else "new_frame"
                        )
                        coverage_by_group[coverage_group].append(coverage_values)
                        coverage_names = (
                            "incoming_depth",
                            "incoming_alpha",
                            "incoming_valid",
                            "existing_depth",
                            "existing_alpha",
                            "existing_valid",
                        )
                        for name, value in zip(coverage_names, coverage_values):
                            hash_stats[
                                f"prehash_view_{int(frame_id)}_{name}_coverage"
                            ] = value
                        if (
                            int(frame_id) in diagnostic_overlap_ids
                            and existing_render is not None
                            and previous_packet is not None
                        ):
                            frame_index = packet.frame_index(int(frame_id))
                            previous_index = previous_packet.frame_index(
                                int(frame_id)
                            )
                            local_depth = incoming_render.depth
                            aligned_local_depth = (
                                local_depth * owner_scale.to(local_depth)
                            )
                            global_depth = existing_render.depth.to(local_depth)
                            local_alpha = incoming_render.alpha.to(local_depth)
                            global_alpha = existing_render.alpha.to(local_depth)
                            sky = (
                                packet.sky_mask[0, frame_index]
                                | previous_packet.sky_mask[
                                    0, previous_index
                                ].to(packet.sky_mask.device)
                            ).to(local_depth.device)
                            semantic_valid = (
                                packet.finite_gaussian_mask[0, frame_index]
                                & packet.static_mask[0, frame_index]
                                & packet.geometry_consistency[0, frame_index]
                                & previous_packet.finite_gaussian_mask[
                                    0, previous_index
                                ].to(packet.finite_gaussian_mask.device)
                                & previous_packet.static_mask[
                                    0, previous_index
                                ].to(packet.static_mask.device)
                                & previous_packet.geometry_consistency[
                                    0, previous_index
                                ].to(packet.geometry_consistency.device)
                            ).to(local_depth.device)
                            valid = (
                                semantic_valid
                                & ~sky
                                & torch.isfinite(aligned_local_depth)
                                & torch.isfinite(global_depth)
                                & (aligned_local_depth > 0.0)
                                & (global_depth > 0.0)
                                & torch.isfinite(local_alpha)
                                & torch.isfinite(global_alpha)
                                & (
                                    local_alpha
                                    >= self.rendered_alignment_alpha_threshold
                                )
                                & (
                                    global_alpha
                                    >= self.rendered_alignment_alpha_threshold
                                )
                            )
                            relative_error = torch.where(
                                valid,
                                (aligned_local_depth - global_depth).abs()
                                / global_depth.abs().clamp_min(1.0e-6),
                                torch.full_like(global_depth, torch.nan),
                            )
                            diagnostic_rows.append(
                                {
                                    "frame_id": torch.tensor(
                                        int(frame_id), dtype=torch.long
                                    ),
                                    "local_depth": local_depth.detach(),
                                    "aligned_local_depth": (
                                        aligned_local_depth.detach()
                                    ),
                                    "global_depth": global_depth.detach(),
                                    "relative_error": relative_error.detach(),
                                    "local_alpha": local_alpha.detach(),
                                    "global_alpha": global_alpha.detach(),
                                    "sky_mask": sky.detach(),
                                    "valid_mask": valid.detach(),
                                    "inlier_mask": (
                                        valid
                                        & (
                                            relative_error
                                            <= self.rendered_alignment_global_map_consistency_error
                                        )
                                    ).detach(),
                                }
                            )
                    hash_visibility_views = len(visibility_frame_ids)
                    if diagnostic_rows:
                        valid_errors = torch.cat(
                            [
                                row["relative_error"][row["valid_mask"]]
                                for row in diagnostic_rows
                                if bool(row["valid_mask"].any())
                            ],
                            dim=0,
                        ) if any(
                            bool(row["valid_mask"].any())
                            for row in diagnostic_rows
                        ) else window_transform.new_empty(0)
                        diagnostic_inlier_ratio = (
                            float(
                                (
                                    valid_errors
                                    <= self.rendered_alignment_global_map_consistency_error
                                )
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            )
                            if int(valid_errors.numel()) > 0
                            else 0.0
                        )
                        diagnostic_median = (
                            float(valid_errors.median().detach().cpu())
                            if int(valid_errors.numel()) > 0
                            else float("nan")
                        )
                        self._last_rendered_overlap_diagnostic = {
                            name: torch.stack(
                                [row[name].detach().cpu() for row in diagnostic_rows]
                            )
                            for name in (
                                "local_depth",
                                "aligned_local_depth",
                                "global_depth",
                                "relative_error",
                                "local_alpha",
                                "global_alpha",
                                "sky_mask",
                                "valid_mask",
                                "inlier_mask",
                            )
                        }
                        self._last_rendered_overlap_diagnostic["frame_ids"] = (
                            torch.stack(
                                [row["frame_id"] for row in diagnostic_rows]
                            )
                        )
                        alignment_diagnostics.update(
                            {
                                "global_render_used_for_scale": False,
                                "global_render_diagnostic_only": True,
                                "global_render_diagnostic_valid_points": int(
                                    valid_errors.numel()
                                ),
                                "global_render_diagnostic_inlier_ratio": (
                                    diagnostic_inlier_ratio
                                ),
                                "global_render_diagnostic_median_relative_error": (
                                    diagnostic_median
                                ),
                            }
                        )
                    coverage_names = (
                        "incoming_depth",
                        "incoming_alpha",
                        "incoming_valid",
                        "existing_depth",
                        "existing_alpha",
                        "existing_valid",
                    )
                    for group, rows in coverage_by_group.items():
                        hash_stats[f"prehash_{group}_view_count"] = len(rows)
                        if not rows:
                            continue
                        for index, name in enumerate(coverage_names):
                            hash_stats[
                                f"prehash_{group}_{name}_coverage"
                            ] = sum(row[index] for row in rows) / len(rows)
                    if require_four_view_admission:
                        if prepared.source_anchor_indices is None:
                            raise RuntimeError(
                                "Four-view incoming admission requires source anchor indices"
                            )
                        admission_requested = len(prepared.batch)
                        source_rows = prepared.source_anchor_indices.to(
                            incoming_visibility.device
                        )
                        admitted = incoming_visibility.index_select(
                            0, source_rows
                        )
                        selected = torch.nonzero(
                            admitted, as_tuple=False
                        ).flatten().to(prepared.batch.xyz.device)
                        prepared = prepared.index(selected)
                        admission_kept = len(prepared.batch)
                        hash_stats.update(
                            {
                                "incoming_visibility_admission_requested": (
                                    admission_requested
                                ),
                                "incoming_visibility_admission_kept": (
                                    admission_kept
                                ),
                                "incoming_visibility_admission_dropped": (
                                    admission_requested - admission_kept
                                ),
                                "incoming_visibility_admission_views": (
                                    len(visibility_frame_ids)
                                ),
                                "hash_candidates": admission_kept,
                                "hash_visible_incoming": admission_kept,
                                "hash_kept": admission_kept,
                            }
                        )
                        for level in range(len(self.fusion.voxel_sizes)):
                            level_incoming = int(
                                (prepared.batch.level == level)
                                .sum()
                                .detach()
                                .cpu()
                            )
                            hash_stats[f"hash_level_{level}_incoming"] = (
                                level_incoming
                            )
                            hash_stats[f"hash_level_{level}_visible"] = (
                                level_incoming
                            )
                            hash_stats[f"hash_level_{level}_kept"] = (
                                level_incoming
                            )
                else:
                    incoming_render = self._render_refined_anchor_shared_frame(
                        packet
                    )
                    existing_render = self._render_global_shared_frame(
                        self.graph.transform(start_frame),
                        image_size=packet.anchor_observation.image_size,
                    )
                    incoming_visibility = incoming_render.anchor_visibility
                    existing_visibility = existing_render.anchor_visibility
                    insertion_render_seconds = (
                        incoming_render.render_seconds
                        + existing_render.render_seconds
                    )
                    hash_visibility_views = 1
                hash_stats["hash_visibility_views"] = hash_visibility_views
                if self.insertion_dedup_enabled and has_existing_map:
                    hash_start = time.perf_counter()
                    prehash_diagnostics = dict(hash_stats)
                    prepared, filtered_hash_stats, evidence_update = (
                        self.fusion.filter_against_visible_map(
                            prepared,
                            incoming_anchor_visibility=incoming_visibility,
                            existing_anchor_visibility=existing_visibility,
                            radius_voxels=self.insertion_dedup_radius_voxels,
                            update_existing_statistics=(
                                self.insertion_dedup_update_existing_statistics
                            ),
                        )
                    )
                    hash_stats = {
                        **prehash_diagnostics,
                        **filtered_hash_stats,
                    }
                    hash_stats["hash_visibility_views"] = hash_visibility_views
                    hash_seconds = float(time.perf_counter() - hash_start)
            prepared, incoming_budget_stats = (
                self.fusion.limit_prepared_incoming_by_coverage(
                    prepared,
                    max_new_gaussians=(
                        self.insertion_dedup_max_new_gaussians_per_chunk
                    ),
                    coarse_cell_size=(
                        self.insertion_dedup_coverage_coarse_cell_size
                    ),
                )
            )
            hash_stats.update(incoming_budget_stats)
            anchors_before_commit = self.map.anchor_count()
            commit_start = time.perf_counter()
            fusion_stats = self.fusion.commit_prepared_packet(
                packet,
                window_transform,
                prepared,
                evidence_update=evidence_update,
                extra_stats={
                    **hash_stats,
                    "fusion_prepare_seconds": prepare_seconds,
                    "insertion_render_seconds": insertion_render_seconds,
                    "insertion_hash_seconds": hash_seconds,
                    "refiner_seconds": float(
                        packet.metadata.get("voxel_anchor_refiner_seconds", 0.0)
                    ),
                },
            )
            fusion_stats["fusion_commit_seconds"] = float(
                time.perf_counter() - commit_start
            )
            fusion_stats["chunk_anchor_delta"] = (
                self.map.anchor_count() - anchors_before_commit
            )
            if posthash_coverage_context is not None:
                (
                    coverage_frame_ids,
                    coverage_global_poses,
                    coverage_overlap_ids,
                    coverage_image_size,
                ) = posthash_coverage_context
                posthash_render_start = time.perf_counter()
                posthash_by_group: dict[str, list[tuple[float, float, float]]] = {
                    "overlap": [],
                    "new_frame": [],
                }
                for frame_id in coverage_frame_ids:
                    frame_index = packet.frame_index(frame_id)
                    rendered = self._render_global_pose_frame(
                        coverage_global_poses[frame_index],
                        image_size=coverage_image_size,
                    )
                    depth_valid = torch.isfinite(rendered.depth) & (
                        rendered.depth > 0.0
                    )
                    alpha_valid = torch.isfinite(rendered.alpha) & (
                        rendered.alpha
                        >= self.rendered_alignment_alpha_threshold
                    )
                    coverage_values = (
                        float(depth_valid.float().mean().detach().cpu()),
                        float(alpha_valid.float().mean().detach().cpu()),
                        float(
                            (depth_valid & alpha_valid)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        ),
                    )
                    group = (
                        "overlap"
                        if frame_id in coverage_overlap_ids
                        else "new_frame"
                    )
                    posthash_by_group[group].append(coverage_values)
                    for name, value in zip(
                        ("global_depth", "global_alpha", "global_valid"),
                        coverage_values,
                    ):
                        fusion_stats[
                            f"posthash_view_{frame_id}_{name}_coverage"
                        ] = value
                    prehash_valid = fusion_stats.get(
                        f"prehash_view_{frame_id}_existing_valid_coverage"
                    )
                    if prehash_valid is not None:
                        fusion_stats[
                            f"posthash_view_{frame_id}_valid_coverage_delta"
                        ] = coverage_values[2] - float(prehash_valid)
                for group, rows in posthash_by_group.items():
                    fusion_stats[f"posthash_{group}_view_count"] = len(rows)
                    if not rows:
                        continue
                    for index, name in enumerate(
                        ("global_depth", "global_alpha", "global_valid")
                    ):
                        fusion_stats[
                            f"posthash_{group}_{name}_coverage"
                        ] = sum(row[index] for row in rows) / len(rows)
                    prehash_group_valid = fusion_stats.get(
                        f"prehash_{group}_existing_valid_coverage"
                    )
                    if prehash_group_valid is not None:
                        fusion_stats[
                            f"posthash_{group}_valid_coverage_delta"
                        ] = (
                            fusion_stats[
                                f"posthash_{group}_global_valid_coverage"
                            ]
                            - float(prehash_group_valid)
                        )
                fusion_stats["posthash_coverage_views"] = len(
                    coverage_frame_ids
                )
                fusion_stats["posthash_coverage_render_seconds"] = float(
                    time.perf_counter() - posthash_render_start
                )
        else:
            fusion_stats = self.fusion.fuse_packet(
                packet,
                window_transform,
            )

        fusion_stats["pose_state_candidate_revision"] = int(
            seam_diagnostics.get(
                "candidate_revision", self._pose_state_revision
            )
        )
        fusion_stats["pose_state_committed_revision"] = int(
            self._pose_state_revision
        )

        compact_packet = packet.compact_for_memory()
        self.packets[window_id] = compact_packet
        self._last_full_packet = packet
        self.window_order.append(window_id)
        for frame_id in packet.frame_ids:
            frame = int(frame_id)
            self.frame_windows.setdefault(frame, set()).add(window_id)
            self.frame_owner_window.setdefault(frame, window_id)
            self.frame_depth_owner_window.setdefault(frame, window_id)

        if self.lifecycle_prune_interval > 0 and (
            len(self.window_order) % self.lifecycle_prune_interval == 0
        ):
            fusion_stats["lifecycle_pruned"] = self.fusion.prune_lifecycle(
                current_frame=end_frame,
                max_stale_frames=self.lifecycle_max_stale_frames,
                max_render_error=self.lifecycle_max_render_error,
            )
        self.loop_detector.add(compact_packet)
        if self.chunk_first_stride_graph:
            graph_changed_history = bool(
                graph_result is not None and graph_result.accepted
            )
            self._refresh_geometry_updates(
                complete_snapshot=graph_changed_history,
                affected_node_ids=(
                    set(self.graph.nodes)
                    if graph_changed_history
                    else (
                        {int(start_frame), int(next_frame)}
                        if len(self.window_order) == 1
                        else {int(next_frame)}
                    )
                ),
                reason=(
                    "chunk_graph_optimization_commit"
                    if graph_changed_history
                    else "chunk_pure_chain_append"
                ),
            )
        else:
            self._refresh_geometry_updates()

        if self.loop_neighborhood_refinement_enabled and accepted_loops:
            self._enqueue_map_optimization(
                window_id, packet.frame_ids, self.map_steps_per_window
            )
            refinement_frames: list[int] = []
            refinement_windows = self._loop_neighborhood_windows(accepted_loops)
            self._pending_seam_owner_windows.update(refinement_windows)
            for refinement_window in refinement_windows:
                refinement_packet = self.packets.get(int(refinement_window))
                if refinement_packet is not None:
                    refinement_frames.extend(
                        int(value) for value in refinement_packet.frame_ids
                    )
            self._enqueue_map_optimization(
                window_id,
                tuple(dict.fromkeys(refinement_frames)),
                self.map_steps_on_loop,
            )
            if self.map_steps_per_window > 0 or self.map_steps_on_loop > 0:
                self._optimization_packets[window_id] = packet
        else:
            map_steps = self.map_steps_per_window + (
                self.map_steps_on_loop if accepted_loops else 0
            )
            if map_steps > 0:
                self._enqueue_map_optimization(window_id, packet.frame_ids, map_steps)
                self._optimization_packets[window_id] = packet
        if self.mapper is not None:
            self.mapper.optimizer = self.map.make_optimizer(
                lr=float(self.config.get("map_optimization", {}).get("lr", 2.0e-3))
            )
            self.mapper.stats.n_anchors = self.map.anchor_count()

        result = GlobalWindowBackendResult(
            window_id=window_id,
            aligned=aligned,
            loop_accepted=len(accepted_loops),
            graph=graph_result,
            fusion=fusion_stats,
            correction=correction,
            map_optimization={},
            diagnostics={
                "alignment": alignment_diagnostics,
                "boundary_factor": boundary_diagnostics,
                "loops": [self._loop_summary(value) for value in loop_results],
                "graph_node_mode": self.node_mode,
                "global_graph_optimization_enabled": (
                    self.global_graph_optimization_enabled
                ),
                "post_optimization_validation_enforced": (
                    False
                ),
                "global_ba_scheduled": graph_result is not None,
                "hierarchical_submaps_enabled": self.hierarchical_submaps_enabled,
                "local_camera_model": self.local_camera_model,
                "submap_id": self.window_to_submap.get(window_id),
                "submap_frozen": bool(submap_frozen),
                "submap_count": len(self.submaps),
                "compressed_dense_factors": (
                    self.submaps[self.window_to_submap[window_id]].compressed_dense_factors
                    if window_id in self.window_to_submap
                    else 0
                ),
                "post_optimization_seam_check": seam_diagnostics,
                "pose_state_consistency": seam_diagnostics,
                "chunk_stride_holdout_check": stride_holdout_diagnostics,
                "chunk_scale_check": chunk_scale_diagnostics,
                "chunk_sequence_objective_check": (
                    chunk_sequence_diagnostics
                ),
                "prefusion_pose_tracking": prefusion_pose_diagnostics,
            },
        )
        self.results.append(result)
        return result

    def _process_boundary_packet(
        self,
        packet: LocalGaussianWindowPacket,
        *,
        prepared_candidate: _PreparedPacketCandidate | None = None,
    ) -> GlobalWindowBackendResult:
        if (
            not self._packet_uses_voxel_refiner(packet)
            and not self.two_frame_overlap_enabled
            and not self.chunk_first_stride_graph
        ):
            return self._process_boundary_packet_impl(
                packet,
                prepared_candidate=prepared_candidate,
            )
        transaction = self._snapshot_boundary_transaction(
            extra_packets=(packet,),
        )
        try:
            return self._process_boundary_packet_impl(
                packet,
                prepared_candidate=prepared_candidate,
            )
        except Exception as exc:
            failure_diagnostic = self._last_rendered_overlap_diagnostic
            failure_alignment = self._last_overlap_alignment_failure
            failure_pose_state = self._last_pose_state_diagnostic
            if (
                failure_alignment is None
                and self.two_frame_known_pose_bridge_enabled
            ):
                failure_alignment = {
                    "mode": self.rendered_overlap_alignment_mode,
                    "window_id": int(packet.window_id),
                    "accepted": False,
                    "reason": "known_pose_bridge_exception",
                    "error": repr(exc),
                }
            self._restore_boundary_transaction(transaction)
            self._last_rendered_overlap_diagnostic = failure_diagnostic
            self._last_overlap_alignment_failure = failure_alignment
            if failure_pose_state is not None:
                failure_pose_state = dict(failure_pose_state)
                failure_pose_state["consistency_accepted"] = bool(
                    failure_pose_state.get("accepted", False)
                )
                failure_pose_state["accepted"] = False
                failure_pose_state["transaction_committed"] = False
                failure_pose_state["committed_revision"] = int(
                    self._pose_state_revision
                )
                failure_pose_state["reason"] = (
                    "pose_candidate_transaction_rolled_back"
                )
            self._last_pose_state_diagnostic = failure_pose_state
            raise
        finally:
            if (
                self._packet_uses_voxel_refiner(packet)
                and self.packet_refiner_release is not None
            ):
                self.packet_refiner_release(int(packet.window_id))

    def process_packet(self, packet: LocalGaussianWindowPacket) -> GlobalWindowBackendResult:
        if self.boundary_frame_graph:
            return self._process_boundary_packet(packet)
        return self._process_window_anchor_packet(packet)

    def _process_window_anchor_packet(self, packet: LocalGaussianWindowPacket) -> GlobalWindowBackendResult:
        if not self.enabled:
            raise RuntimeError("SphericalSelfiGlobalBackend is disabled")
        window_id = int(packet.window_id)
        if window_id in self.packets or window_id in self.graph.nodes:
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")

        sequential_edge = None
        sequential_dense_factor = None
        shared_pose_factor = None
        sequential_extra_factors: tuple[DenseSphericalFactorBlock, ...] = ()
        alignment_diagnostics: dict[str, Any] = {}
        if not self.window_order:
            initial = sim3_identity(device=packet.local_poses_c2w.device)
            aligned = True
        else:
            previous_id = self.window_order[-1]
            previous_packet = self._last_full_packet
            if previous_packet is None or int(previous_packet.window_id) != int(previous_id):
                raise RuntimeError("The previous full-resolution window packet is unavailable")
            sequential_edge, sequential_dense_factor, shared_pose_factor, alignment_diagnostics = self._overlap_edge(previous_packet, packet)
            if sequential_edge is None:
                fallback = self.loop_detector.verify_pair(
                    previous_packet,
                    packet,
                    retrieval_score=1.0,
                    edge_type="sequential",
                )
                if (
                    fallback.factor is not None
                    and fallback.factor.kind == "sim3"
                    and fallback.accepted
                ):
                    sequential_edge = self._materialize_loop_pose_factor(fallback.factor)
                    assert isinstance(sequential_edge, Sim3GraphEdge)
                    sequential_extra_factors = tuple(
                        self._materialize_dense_loop_factor(value)
                        for value in fallback.dense_factors
                    )
                    alignment_diagnostics["fallback"] = fallback.reason
                elif not self.allow_unaligned_fallback:
                    raise RuntimeError(
                        f"Window {window_id} cannot be aligned to previous window {previous_id}: "
                        f"{alignment_diagnostics.get('reason', fallback.reason)}"
                    )
            initial = self._initial_transform(previous_id, sequential_edge)
            aligned = sequential_edge is not None

        self.graph.add_node(window_id, initial)
        if sequential_edge is not None:
            self.graph.add_edge(sequential_edge)
        if sequential_dense_factor is not None:
            self.graph.add_edge(sequential_dense_factor)
        if shared_pose_factor is not None:
            self.graph.add_edge(shared_pose_factor)
        for factor in sequential_extra_factors:
            self.graph.add_edge(factor)

        loop_results = self.loop_detector.detect(packet)
        accepted_loops: list[PanoramaLoopVerification] = []
        pending_loop_factors: list[
            DenseSphericalFactorBlock | Sim3GraphEdge | CoincidentPanoramaFactor
        ] = []
        loop_edge_start = len(self.graph.edges)
        for result in loop_results:
            if not result.accepted or result.factor is None:
                continue
            pair = self._canonical_loop_pair(result)
            if pair in self.accepted_loop_pairs:
                self._reject_loop_result(result, "duplicate_loop_pair")
                continue
            if self.loop_transaction_enabled:
                path_ok, path_diagnostics = self._loop_path_consistency(
                    result,
                    source_node=int(result.source_window_id),
                    target_node=int(result.target_window_id),
                )
                result.metadata["path_consistency"] = path_diagnostics
                if not path_ok:
                    self._reject_loop_result(result, "path_inconsistent")
                    continue
            pose_factor = self._materialize_loop_pose_factor(result.factor)
            self.graph.add_edge(pose_factor)
            pending_loop_factors.append(pose_factor)
            for dense_measurement in result.dense_factors:
                dense_factor = self._materialize_dense_loop_factor(dense_measurement)
                self.graph.add_edge(dense_factor)
                pending_loop_factors.append(dense_factor)
            accepted_loops.append(result)

        old_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        pre_loop_nodes = {node: value.clone() for node, value in self.graph.nodes.items()}
        if accepted_loops:
            nonloop_factors = tuple(self.graph.edges[:loop_edge_start])
            nonloop_before = float(
                self.graph.objective(factors=nonloop_factors).detach().cpu()
            )
            graph_result = self.graph.optimize()
            commit = True
            minimum_dcs_scale = 1.0
            nonloop_objective_ratio = 1.0
            if self.loop_transaction_enabled:
                nonloop_after = float(
                    self.graph.objective(factors=nonloop_factors).detach().cpu()
                )
                (
                    commit,
                    minimum_dcs_scale,
                    nonloop_objective_ratio,
                ) = self._loop_transaction_commit_ok(
                    graph_result,
                    pending_loop_factors,
                    nonloop_objective_before=nonloop_before,
                    nonloop_objective_after=nonloop_after,
                )
            for loop_result in accepted_loops:
                loop_result.metadata["graph_transaction"] = {
                    "enabled": self.loop_transaction_enabled,
                    "committed": bool(commit),
                    "minimum_dcs_scale": float(minimum_dcs_scale),
                    "nonloop_objective_ratio": float(nonloop_objective_ratio),
                    "initial_objective": float(graph_result.initial_objective),
                    "final_objective": float(graph_result.final_objective),
                }
            if commit:
                for loop_result in accepted_loops:
                    self.accepted_loop_pairs.add(self._canonical_loop_pair(loop_result))
            else:
                self.graph.nodes = pre_loop_nodes
                del self.graph.edges[loop_edge_start:]
                for loop_result in accepted_loops:
                    self._reject_loop_result(loop_result, "loop_transaction_rejected")
                accepted_loops = []
                graph_result = self.graph.optimize(
                    self.window_order[-self.recent_optimization_windows :] + [window_id]
                )
        else:
            graph_result = self.graph.optimize(self.window_order[-self.recent_optimization_windows :] + [window_id])
        new_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        correction = (
            self.fusion.apply_owner_corrections(old_transforms, new_transforms)
            if self._owner_transforms_changed(old_transforms, new_transforms)
            else {"moved": 0, "deduplicated": 0}
        )

        compact_packet = packet.compact_for_memory()
        self.packets[window_id] = compact_packet
        self._last_full_packet = packet
        self.window_order.append(window_id)
        for frame_id in packet.frame_ids:
            self.frame_windows.setdefault(int(frame_id), set()).add(int(window_id))
            self.frame_depth_owner_window.setdefault(int(frame_id), int(window_id))
        fusion_stats = self.fusion.fuse_packet(packet, self.graph.transform(window_id))
        if self.lifecycle_prune_interval > 0 and (
            len(self.window_order) % self.lifecycle_prune_interval == 0
        ):
            fusion_stats["lifecycle_pruned"] = self.fusion.prune_lifecycle(
                current_frame=int(packet.frame_ids[-1]),
                max_stale_frames=self.lifecycle_max_stale_frames,
                max_render_error=self.lifecycle_max_render_error,
            )
        self.loop_detector.add(compact_packet)
        self._refresh_pose_updates()
        if self.loop_neighborhood_refinement_enabled and accepted_loops:
            self._enqueue_map_optimization(
                window_id, packet.frame_ids, self.map_steps_per_window
            )
            refinement_frames: list[int] = []
            refinement_windows = self._loop_neighborhood_windows(accepted_loops)
            self._pending_seam_owner_windows.update(refinement_windows)
            for refinement_window in refinement_windows:
                refinement_packet = self.packets.get(int(refinement_window))
                if refinement_packet is not None:
                    refinement_frames.extend(
                        int(value) for value in refinement_packet.frame_ids
                    )
            self._enqueue_map_optimization(
                window_id,
                tuple(dict.fromkeys(refinement_frames)),
                self.map_steps_on_loop,
            )
            if self.map_steps_per_window > 0 or self.map_steps_on_loop > 0:
                self._optimization_packets[window_id] = packet
        else:
            map_steps = self.map_steps_per_window + (
                self.map_steps_on_loop if accepted_loops else 0
            )
            if map_steps > 0:
                self._enqueue_map_optimization(window_id, packet.frame_ids, map_steps)
                self._optimization_packets[window_id] = packet
        map_metrics: dict[str, float] = {}
        if self.mapper is not None:
            self.mapper.optimizer = self.map.make_optimizer(
                lr=float(self.config.get("map_optimization", {}).get("lr", 2.0e-3))
            )
            self.mapper.stats.n_anchors = self.map.anchor_count()

        result = GlobalWindowBackendResult(
            window_id=window_id,
            aligned=aligned,
            loop_accepted=len(accepted_loops),
            graph=graph_result,
            fusion=fusion_stats,
            correction=correction,
            map_optimization=map_metrics,
            diagnostics={
                "alignment": alignment_diagnostics,
                "loops": [self._loop_summary(value) for value in loop_results],
            },
        )
        self.results.append(result)
        return result

    @staticmethod
    def _loop_summary(result: PanoramaLoopVerification) -> dict[str, Any]:
        summary = {
            "source_window_id": result.source_window_id,
            "target_window_id": result.target_window_id,
            "accepted": result.accepted,
            "reason": result.reason,
            "retrieval_score": result.retrieval_score,
            "yaw_shift_columns": result.yaw_shift_columns,
            "num_matches": result.num_matches,
            "inlier_ratio": result.inlier_ratio,
            "residual": result.residual,
        }
        diagnostic_keys = (
            "source_frame_index",
            "target_frame_index",
            "source_frame_id",
            "target_frame_id",
            "rotation_inlier_ratio",
            "rotation_ransac_residual",
            "rotation_inlier_count",
            "spherical_coverage_bins",
            "source_spherical_coverage_bins",
            "target_spherical_coverage_bins",
            "rotation_consistency_deg",
            "normalized_alignment_residual",
            "alignment_scale",
            "verified_num_matches",
            "path_consistency",
            "graph_transaction",
        )
        summary["verification"] = {
            key: result.metadata[key]
            for key in diagnostic_keys
            if key in result.metadata
        }
        return summary

    def finalize(self) -> dict[str, Any]:
        if not self.window_order:
            return {}
        if self.boundary_frame_graph:
            pending_metrics = self.run_pending_map_optimization()
            old_transforms = self._window_anchor_transforms()
            if self.chunk_first_stride_graph:
                # Every admitted skip/loop cycle has already passed the
                # sequential-objective and holdout transaction at insertion
                # time.  Re-running an unconditional legacy final LM here
                # would bypass those gates and can move the complete history
                # after the last observable cycle.  Publish the accepted graph
                # as-is instead.
                objective = float(self.graph.objective().detach().cpu())
                graph_result = Sim3GraphOptimizeResult(
                    accepted=False,
                    iterations=0,
                    initial_objective=objective,
                    final_objective=objective,
                    max_update_norm=0.0,
                    optimized_node_ids=(),
                    reason="chunk_first_stride_no_unvalidated_finalize_lm",
                    final_damping=float(self.graph.damping),
                )
            elif self.hierarchical_submaps_enabled:
                assert self.submap_graph is not None
                if self._active_submap_id is not None:
                    self._update_submap_local_geometry(self._active_submap_id)
                graph_result = self.submap_graph.optimize()
                if self.global_graph_optimization_enabled:
                    self._apply_submap_graph_to_boundary_graph()
            else:
                graph_result = self.graph.optimize()
            new_transforms = self._window_anchor_transforms()
            correction = (
                self.fusion.apply_owner_corrections(
                    old_transforms,
                    new_transforms,
                )
                if self._owner_transforms_changed(old_transforms, new_transforms)
                else {"moved": 0, "deduplicated": 0}
            )
            self._refresh_geometry_updates()
            map_metrics = self._run_map_optimization(
                int(self.window_order[-1]),
                tuple(self.frame_owner_window),
                self.final_map_steps,
            )
            if not map_metrics:
                map_metrics = pending_metrics
            return {
                "graph_initial_objective": graph_result.initial_objective,
                "graph_final_objective": graph_result.final_objective,
                "graph_iterations": graph_result.iterations,
                "graph_reason": graph_result.reason,
                "global_graph_optimization_enabled": (
                    self.global_graph_optimization_enabled
                ),
                "graph_node_mode": self.node_mode,
                "hierarchical_submaps_enabled": self.hierarchical_submaps_enabled,
                "local_camera_model": self.local_camera_model,
                "submap_nodes": (
                    len(self.submap_graph.nodes)
                    if self.submap_graph is not None
                    else 0
                ),
                "compressed_dense_factors": sum(
                    record.compressed_dense_factors
                    for record in self.submaps.values()
                ),
                "pcg_iterations": graph_result.pcg_iterations,
                "pcg_relative_residual": graph_result.pcg_relative_residual,
                "normal_condition_estimate": graph_result.normal_condition_estimate,
                "rejected_lm_trials": graph_result.rejected_trials,
                "moved_gaussians": correction.get("moved", 0),
                "deduplicated_gaussians": correction.get("deduplicated", 0),
                "anchors": self.map.anchor_count(),
                "map_saturated": int(
                    any(int(result.fusion.get("map_saturated", 0)) > 0 for result in self.results)
                ),
                "map_optimization": map_metrics,
            }
        pending_metrics = self.run_pending_map_optimization()
        old_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        graph_result = self.graph.optimize()
        correction = (
            self.fusion.apply_owner_corrections(old_transforms, self.graph.nodes)
            if self._owner_transforms_changed(old_transforms, self.graph.nodes)
            else {"moved": 0, "deduplicated": 0}
        )
        self._refresh_pose_updates()
        map_metrics = self._run_map_optimization(
            int(self.window_order[-1]), tuple(self.frame_owner_window), self.final_map_steps
        )
        if not map_metrics:
            map_metrics = pending_metrics
        return {
            "graph_initial_objective": graph_result.initial_objective,
            "graph_final_objective": graph_result.final_objective,
            "graph_iterations": graph_result.iterations,
            "moved_gaussians": correction.get("moved", 0),
            "deduplicated_gaussians": correction.get("deduplicated", 0),
            "anchors": self.map.anchor_count(),
            "map_saturated": int(
                any(int(result.fusion.get("map_saturated", 0)) > 0 for result in self.results)
            ),
            "map_optimization": map_metrics,
        }
