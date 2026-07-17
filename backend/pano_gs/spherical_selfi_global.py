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
from frontend.spherical_selfi.window_packet import BoundaryMatchBlock, LocalGaussianWindowPacket
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
    rebase_c2w_to_sim3_anchor,
    sim3_components,
    sim3_from_components,
    sim3_identity,
    sim3_inverse,
    sim3_log,
    weighted_umeyama,
)
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
    depth_scales_by_window: dict[int, float] = field(default_factory=dict, compare=False)


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
        loop_cfg = dict(self.config.get("loop_closure", {}) or {})
        descriptor_cfg = dict(loop_cfg.get("descriptor", {}) or {})
        retrieval_cfg = dict(loop_cfg.get("retrieval", {}) or {})
        verification_cfg = dict(loop_cfg.get("verification", {}) or {})
        robust_loop_cfg = dict(self.config.get("robust_loop", {}) or {})
        hierarchical_cfg = dict(self.config.get("hierarchical_submaps", {}) or {})
        fusion_cfg = dict(self.config.get("voxel_fusion", {}) or {})
        optimize_cfg = dict(self.config.get("map_optimization", {}) or {})
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
        if self.node_mode not in {"window_anchor", "boundary_frame"}:
            raise ValueError("global_graph.node_mode must be 'window_anchor' or 'boundary_frame'")
        self.boundary_frame_graph = self.node_mode == "boundary_frame"
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
        self.min_match_margin = float(graph_cfg.get("min_match_margin", 0.01))
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
        self.post_optimization_seam_max_rotation_deg = float(
            seam_check_cfg.get("max_rotation_error_deg", 2.0)
        )
        self.post_optimization_seam_max_center_error = float(
            seam_check_cfg.get("max_center_error", 0.15)
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
            max_translation_update=float(graph_cfg.get("max_translation_update", 1.0)),
            max_rotation_update_deg=float(graph_cfg.get("max_rotation_update_deg", 10.0)),
            max_log_scale_update=float(graph_cfg.get("max_log_scale_update", 0.25)),
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
                max_translation_update=float(graph_cfg.get("max_translation_update", 1.0)),
                max_rotation_update_deg=float(graph_cfg.get("max_rotation_update_deg", 10.0)),
                max_log_scale_update=float(graph_cfg.get("max_log_scale_update", 0.25)),
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
            min_match_margin=float(graph_cfg.get("min_match_margin", 0.01)),
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
        self.window_anchor_nodes: dict[int, int] = {}
        self.window_end_nodes: dict[int, int] = {}
        self.boundary_node_order: list[int] = []
        self._sequential_edges_since_optimization = 0
        self._has_run_global_ba = False
        self._geometry_updates: dict[int, FrameGeometryUpdate] = {}
        self._pending_map_optimization: list[tuple[int, tuple[int, ...], int]] = []
        self._optimization_packets: dict[int, LocalGaussianWindowPacket] = {}
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

    def _snapshot_boundary_transaction(self) -> dict[str, Any]:
        loop_database = getattr(self.loop_detector, "_descriptor_database", None)
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
            "window_anchor_nodes": dict(self.window_anchor_nodes),
            "window_end_nodes": dict(self.window_end_nodes),
            "boundary_node_order": list(self.boundary_node_order),
            "sequential_edges": self._sequential_edges_since_optimization,
            "has_run_global_ba": self._has_run_global_ba,
            "geometry_updates": dict(self._geometry_updates),
            "pending_map_optimization": list(self._pending_map_optimization),
            "optimization_packets": dict(self._optimization_packets),
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
        self.window_anchor_nodes = state["window_anchor_nodes"]
        self.window_end_nodes = state["window_end_nodes"]
        self.boundary_node_order = state["boundary_node_order"]
        self._sequential_edges_since_optimization = state["sequential_edges"]
        self._has_run_global_ba = state["has_run_global_ba"]
        self._geometry_updates = state["geometry_updates"]
        self._pending_map_optimization = state["pending_map_optimization"]
        self._optimization_packets = state["optimization_packets"]
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
            self.mapper.optimizer = state["mapper_optimizer"]
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
        """Return start/end graph nodes for the most recent configured windows."""

        ordered_windows = list(self.window_order)
        if (
            include_window_id is not None
            and int(include_window_id) not in ordered_windows
        ):
            ordered_windows.append(int(include_window_id))
        selected = ordered_windows[-self.global_ba_active_nodes :]
        nodes: list[int] = []
        for window_id in selected:
            for node in (
                self.window_anchor_nodes.get(int(window_id)),
                self.window_end_nodes.get(int(window_id)),
            ):
                if (
                    node is not None
                    and int(node) in self.graph.nodes
                    and int(node) not in nodes
                ):
                    nodes.append(int(node))
        return nodes

    def _overlap_seam_diagnostics(self) -> dict[str, Any]:
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
        accepted = (
            max_rotation <= self.post_optimization_seam_max_rotation_deg
            and max_center <= self.post_optimization_seam_max_center_error
        )
        return {
            "enabled": self.post_optimization_seam_check_enabled,
            "factor_count": factor_count,
            "max_rotation_error_deg": max_rotation,
            "max_center_error": max_center,
            "rotation_errors_deg": rotation_errors,
            "center_errors": center_errors,
            "accepted": bool(accepted),
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
        compressed = 0
        for factor in self.graph.edges:
            source = int(factor.source)
            target = int(factor.target)
            within_record = (
                source in boundary_nodes and target in boundary_nodes
            )
            cross_frozen_boundary = (
                isinstance(factor, DenseSphericalFactorBlock)
                and factor.edge_type == "overlap_dense_spherical"
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
                and factor.edge_type
                in {"boundary_dense_spherical", "overlap_dense_spherical"}
                and (within_record or cross_frozen_boundary)
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
            information = source_transform.new_tensor(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5]
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
                    },
                )
            )
            compressed += 1
        self.graph.edges = retained
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

    def _known_overlap_global_pose(
        self,
        previous: LocalGaussianWindowPacket,
        frame_id: int,
        previous_owner_transform: torch.Tensor,
    ) -> torch.Tensor:
        """Return the already-admitted global SE(3) pose of an overlap frame."""

        frame = int(frame_id)
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
        if (
            consistency_ratio
            < self.rendered_alignment_global_map_min_consistency_ratio
        ):
            raise RuntimeError(
                f"Frame {frame_id} global/previous map consistency ratio "
                f"{consistency_ratio:.3f} is below "
                f"{self.rendered_alignment_global_map_min_consistency_ratio:.3f}"
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
        return [
            self._collect_known_pose_bridge_frame(
                previous,
                current,
                frame_id,
                previous_owner_transform=previous_owner_transform,
                exclude_current_target_only=exclude_current_target_only,
            )
            for frame_id in overlap
        ]

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
        inlier_masks: list[torch.Tensor] = []
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
                error <= self.rendered_alignment_max_median_relative_error
            )
            inlier_masks.append(inliers)
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
        # Pose-baseline is an intentional ablation: rendered depth remains a
        # diagnostic/factor selector but cannot change or veto its scale.
        accepted = scale_ok and (depth_gate if mode == "depth" else True)
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
                "accepted": bool(accepted),
                "reason": reason,
            }
        )
        return (
            absolute_scale if accepted else None,
            inlier_masks,
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
        inlier_masks: list[torch.Tensor] = []
        per_frame_ratio: list[float] = []
        per_frame_median: list[float] = []
        for frame in frames:
            error = (
                (absolute_scale * frame.current_depth - frame.global_depth).abs()
                / frame.global_depth.abs().clamp_min(1.0e-6)
            )
            mask = error <= self.rendered_alignment_max_median_relative_error
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
                if self.two_frame_bridge_depth_scale_enabled:
                    raise RuntimeError(
                        f"Frame {frame.frame_id} has insufficient post-Refiner "
                        "depth inliers for bridge factors"
                    )
                # Pose-baseline is deliberately independent of rendered depth;
                # keep all geometrically valid samples and let the graph's
                # robust depth residual downweight disagreement.
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
            & (matches.top2_margin >= self.min_match_margin)
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

    def _refresh_geometry_updates(self) -> None:
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
        updates = dict(self._geometry_updates)
        self._geometry_updates.clear()
        return updates

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
            fixed_frame_ids: list[int] = []
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
                "pose_refine_enable": bool(
                    self.map_optimize_config.get("pose_refine_enable", True)
                ),
                "pose_prior_weight": float(self.map_optimize_config.get("pose_prior_weight", 0.0)),
                "pose_grad_clip": float(self.map_optimize_config.get("pose_grad_clip", 1.0e-3)),
                "visible_neighbor_lr_scale": float(self.map_optimize_config.get("visible_neighbor_lr_scale", 0.1)),
                "sampler_seed": int(self.map_optimize_config.get("seed", 123)) + int(window_id),
                "fixed_pose_frame_ids": fixed_frame_ids,
            }
            metrics = self.mapper.optimize_spherical_selfi_window(
                window_id=int(window_id),
                frame_ids=list(frame_ids),
                iters=int(steps),
                settings=settings,
                extra_loss_fn=(
                    lambda trainable_pose_ids: self._joint_graph_pose_loss(
                        int(window_id), trainable_pose_ids
                    )
                ),
            )
            metrics["pose_refine_enabled"] = float(settings["pose_refine_enable"])
            if float(metrics.get("window_rollback", 0.0)) == 0.0:
                try:
                    self._synchronize_joint_optimized_window(int(window_id))
                except (RuntimeError, ValueError, KeyError) as exc:
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
            return metrics
        except (RuntimeError, ValueError, KeyError) as exc:
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
        for index, (queued_window, queued_frames, queued_steps) in enumerate(
            self._pending_map_optimization
        ):
            if int(queued_window) == window:
                self._pending_map_optimization[index] = (
                    window,
                    tuple(
                        dict.fromkeys(
                            [int(value) for value in queued_frames]
                            + [int(value) for value in frame_ids]
                        )
                    ),
                    max(int(queued_steps), int(steps)),
                )
                return
        self._pending_map_optimization.append((window, tuple(frame_ids), int(steps)))

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

    def _packet_variants(self, window_id: int) -> list[LocalGaussianWindowPacket]:
        variants: list[LocalGaussianWindowPacket] = []
        for candidate in (
            self.packets.get(int(window_id)),
            self._optimization_packets.get(int(window_id)),
            self._last_full_packet,
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

    def _synchronize_joint_optimized_window(self, window_id: int) -> None:
        """Transactionally rebase optimized SE(3) poses while graph scale stays authoritative."""

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
            optimized_by_frame[int(frame_id)] = pose.float()

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
            optimized_by_frame[int(frame_id)] = pose.float()

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

    def _process_boundary_packet_impl(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> GlobalWindowBackendResult:
        if not self.enabled:
            raise RuntimeError("SphericalSelfiGlobalBackend is disabled")
        window_id = int(packet.window_id)
        if window_id in self.packets:
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")
        if len(packet.frame_ids) < 2:
            raise ValueError("Boundary-frame graph requires at least two frames per window")
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
        if not self.window_order:
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
            previous_packet = self._last_full_packet
            if previous_packet is None or int(previous_packet.window_id) != previous_id:
                raise RuntimeError("The previous full-resolution window packet is unavailable")
            if self.two_frame_known_pose_bridge_enabled:
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
                    if post_scale is None:
                        raise RuntimeError(
                            "Post-Refiner bridge scale validation failed: "
                            f"{post_diagnostics.get('reason', 'unknown')}"
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
                        if final_scale is None:
                            raise RuntimeError(
                                "Final post-Refiner scale validation failed: "
                                f"{final_scale_diagnostics.get('reason', 'unknown')}"
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
            if refined_packet and not self.two_frame_overlap_enabled:
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
            elif not refined_packet and not self.two_frame_overlap_enabled:
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

        # Recovery state is needed only while this packet is the incoming
        # target.  Once admitted it becomes the canonical source for the next
        # boundary and the dense backup can be released.
        packet.pre_depth_shift_depth = None

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
            raise ValueError(f"Boundary frame node {end_frame} already exists before window {window_id}")
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
        pre_loop_nodes = {node: value.clone() for node, value in loop_graph.nodes.items()}
        pre_boundary_graph_state = self._snapshot_graph_state(self.graph)
        pre_submap_graph_state = self._snapshot_graph_state(self.submap_graph)
        graph_result: Sim3GraphOptimizeResult | None = None
        seam_diagnostics: dict[str, Any] = {
            "enabled": self.post_optimization_seam_check_enabled,
            "accepted": True,
            "factor_count": 0,
        }
        recent_window_count = len(self.window_order) + 1
        should_optimize_recent = (
            (
                recent_window_count >= self.global_ba_start_nodes
                if self.two_frame_overlap_enabled
                else len(self.boundary_node_order) >= self.global_ba_start_nodes
            )
            and (
                not self._has_run_global_ba
                or self._sequential_edges_since_optimization >= self.global_ba_interval_edges
            )
        )
        if accepted_loops:
            nonloop_factors = tuple(loop_graph.edges[:loop_edge_start])
            nonloop_before = float(
                loop_graph.objective(factors=nonloop_factors).detach().cpu()
            )
            graph_result = loop_graph.optimize()
            commit = True
            minimum_dcs_scale = 1.0
            nonloop_objective_ratio = 1.0
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
                    active = (
                        self._recent_boundary_window_nodes(
                            include_window_id=window_id
                        )
                        if self.two_frame_overlap_enabled
                        else self.boundary_node_order[
                            -self.global_ba_active_nodes :
                        ]
                    )
                    graph_result = self.graph.optimize(active, fixed_node_ids={active[0]})
                    self._has_run_global_ba = True
                    self._sequential_edges_since_optimization = 0
        elif should_optimize_recent:
            active = (
                self._recent_boundary_window_nodes(
                    include_window_id=window_id
                )
                if self.two_frame_overlap_enabled
                else self.boundary_node_order[-self.global_ba_active_nodes :]
            )
            graph_result = self.graph.optimize(
                active,
                fixed_node_ids={active[0]},
            )
            self._has_run_global_ba = True
            self._sequential_edges_since_optimization = 0
        if graph_result is not None and self.post_optimization_seam_check_enabled:
            seam_diagnostics = self._overlap_seam_diagnostics()
            if not bool(seam_diagnostics["accepted"]):
                self._restore_graph_state(
                    self.graph, pre_boundary_graph_state
                )
                self._restore_graph_state(
                    self.submap_graph, pre_submap_graph_state
                )
                if accepted_loops:
                    del loop_graph.edges[loop_edge_start:]
                    for loop_result in accepted_loops:
                        self.accepted_loop_pairs.discard(
                            self._canonical_loop_pair(loop_result)
                        )
                        self._reject_loop_result(
                            loop_result,
                            "post_optimization_seam_check_rejected",
                        )
                    accepted_loops = []
                graph_result = replace(
                    graph_result,
                    accepted=False,
                    final_objective=graph_result.initial_objective,
                    max_update_norm=0.0,
                    reason="post_optimization_seam_check_rejected",
                )
        if self.hierarchical_submaps_enabled and self._active_submap_id is not None:
            self._update_submap_local_geometry(self._active_submap_id)
        submap_frozen = self._freeze_active_submap_if_ready()
        new_window_transforms = self._window_anchor_transforms()
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
        if refined_packet:
            prepare_start = time.perf_counter()
            prepared = self.fusion.prepare_packet_batch(packet, window_transform)
            prepare_seconds = float(time.perf_counter() - prepare_start)
            support_requested = len(prepared.batch)
            support_kept = support_requested
            if (
                self.insertion_dedup_require_new_frame_support
                and self.window_order
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
                if previous_packet is None:
                    raise RuntimeError(
                        "New-frame-supported fusion requires the previous packet"
                    )
                overlap_ids = set(
                    self._overlap_frame_ids(previous_packet, packet)
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
            if self.insertion_dedup_enabled and self.map.anchor_count() > 0:
                assert packet.anchor_observation is not None
                if self.two_frame_overlap_enabled:
                    previous_packet = self._last_full_packet
                    if previous_packet is None:
                        raise RuntimeError(
                            "Two-frame insertion dedup requires the previous packet"
                        )
                    overlap_ids = self._overlap_frame_ids(
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
                    for frame_id in overlap_ids:
                        incoming_render = self._render_refined_anchor_frame(
                            packet, frame_id
                        )
                        existing_render = self._render_global_pose_frame(
                            global_poses[packet.frame_index(frame_id)],
                            image_size=packet.anchor_observation.image_size,
                        )
                        incoming_visibility |= (
                            incoming_render.anchor_visibility.to(
                                incoming_visibility.device
                            )
                        )
                        existing_visibility |= (
                            existing_render.anchor_visibility.to(
                                existing_visibility.device
                            )
                        )
                        insertion_render_seconds += (
                            incoming_render.render_seconds
                            + existing_render.render_seconds
                        )
                    hash_visibility_views = len(overlap_ids)
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
                hash_start = time.perf_counter()
                prepared, hash_stats, evidence_update = (
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
                hash_stats["hash_visibility_views"] = hash_visibility_views
                hash_seconds = float(time.perf_counter() - hash_start)
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
        else:
            fusion_stats = self.fusion.fuse_packet(
                packet,
                window_transform,
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
                "graph_node_mode": "boundary_frame",
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
            },
        )
        self.results.append(result)
        return result

    def _process_boundary_packet(
        self,
        packet: LocalGaussianWindowPacket,
    ) -> GlobalWindowBackendResult:
        if (
            not self._packet_uses_voxel_refiner(packet)
            and not self.two_frame_overlap_enabled
        ):
            return self._process_boundary_packet_impl(packet)
        transaction = self._snapshot_boundary_transaction()
        try:
            return self._process_boundary_packet_impl(packet)
        except Exception as exc:
            failure_diagnostic = self._last_rendered_overlap_diagnostic
            failure_alignment = self._last_overlap_alignment_failure
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
            if self.hierarchical_submaps_enabled:
                assert self.submap_graph is not None
                if self._active_submap_id is not None:
                    self._update_submap_local_geometry(self._active_submap_id)
                graph_result = self.submap_graph.optimize()
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
                "graph_node_mode": "boundary_frame",
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
