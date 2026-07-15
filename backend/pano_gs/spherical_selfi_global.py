"""Coordinator for window Sim(3), panorama loops, and the single explicit map."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch

from frontend.pano_vggt.alignment import SubmapAligner
from frontend.spherical_selfi.panorama_loop import PanoramaLoopDetector
from frontend.spherical_selfi.window_packet import BoundaryMatchBlock, LocalGaussianWindowPacket
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
)
from models.spherical_voxel_anchor_refiner import voxelize_per_pixel_gaussians

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
    """One five-window local graph represented by a global Sim(3) node."""

    submap_id: int
    anchor_node_id: int
    window_ids: list[int] = field(default_factory=list)
    boundary_node_ids: list[int] = field(default_factory=list)
    local_window_transforms: dict[int, torch.Tensor] = field(default_factory=dict)
    local_boundary_transforms: dict[int, torch.Tensor] = field(default_factory=dict)
    frozen: bool = False
    compressed_dense_factors: int = 0


class SphericalSelfiGlobalBackend:
    def __init__(
        self,
        gaussian_map: PanoGaussianMap,
        *,
        mapper: PanoGaussianMapper | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.map = gaussian_map
        self.mapper = mapper
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
        self.enabled = bool(self.config.get("enabled", False))
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
        self.allow_unaligned_fallback = bool(graph_cfg.get("allow_unaligned_fallback", False))
        self.allow_boundary_matching_fallback = bool(
            graph_cfg.get("allow_boundary_matching_fallback", False)
        )
        self.expected_overlap_frames = int(graph_cfg.get("expected_overlap_frames", 1))
        self.enforce_exact_overlap = bool(graph_cfg.get("enforce_exact_overlap", True))
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
        self.final_map_steps = max(0, int(optimize_cfg.get("final_steps", 0)))
        self.map_optimize_config = optimize_cfg
        self.lazy_submap_transforms_enabled = bool(lazy_map_cfg.get("enabled", False))
        if self.lazy_submap_transforms_enabled and not self.hierarchical_submaps_enabled:
            raise ValueError(
                "map_optimization.lazy_submap_transforms requires hierarchical_submaps.enabled"
            )
        self.geometry_validation_enabled = bool(validation_cfg.get("enabled", True))
        self.geometry_tolerance = float(validation_cfg.get("tolerance", 1.0e-5))
        self.geometry_rollback_on_failure = bool(validation_cfg.get("rollback_on_failure", True))
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
            )
            if self.hierarchical_submaps_enabled
            else None
        )
        self.overlap_aligner = SubmapAligner(
            align_mode="sim3",
            max_residual=float(graph_cfg.get("max_overlap_residual", 0.35)),
            min_inlier_ratio=float(graph_cfg.get("min_overlap_inlier_ratio", 0.35)),
            max_scale_change=float(graph_cfg.get("max_scale_change", 2.5)),
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
        self.boundary_node_order: list[int] = []
        self._sequential_edges_since_optimization = 0
        self._has_run_global_ba = False
        self._geometry_updates: dict[int, FrameGeometryUpdate] = {}
        self._pending_map_optimization: list[tuple[int, tuple[int, ...], int]] = []
        self._optimization_packets: dict[int, LocalGaussianWindowPacket] = {}
        self.results: list[GlobalWindowBackendResult] = []

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
                            "shared_boundary_node": int(start_node),
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
        retained: list[
            DenseSphericalFactorBlock | Sim3GraphEdge | CoincidentPanoramaFactor
        ] = []
        compressed = 0
        for factor in self.graph.edges:
            should_compress = (
                isinstance(factor, DenseSphericalFactorBlock)
                and factor.edge_type == "boundary_dense_spherical"
                and int(factor.source) in boundary_nodes
                and int(factor.target) in boundary_nodes
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
        # Later submaps own the shared boundary node, which keeps the next
        # active submap's anchor exactly synchronized after a global update.
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
    def _rebuild_packet_anchor_geometry(packet: LocalGaussianWindowPacket) -> bool:
        """Re-voxelize anchors after a non-uniform per-frame depth replacement."""

        if packet.anchor_observation is None:
            return False
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

        if not self._pending_map_optimization:
            return {}
        pending = list(self._pending_map_optimization)
        self._pending_map_optimization.clear()
        last_metrics: dict[str, float] = {}
        for window_id, frame_ids, steps in pending:
            last_metrics = self._run_map_optimization(window_id, frame_ids, steps)
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
            # the configured five-window graph BA into an every-window solve.
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

    @staticmethod
    def _materialize_loop_pose_factor(
        measurement: LoopPoseMeasurement,
    ) -> Sim3GraphEdge | CoincidentPanoramaFactor:
        """Convert a verified backend-neutral measurement into a graph factor."""

        if measurement.kind == "sim3":
            assert measurement.measurement_target_to_source is not None
            assert measurement.information_diag is not None
            return Sim3GraphEdge(
                source=int(measurement.source),
                target=int(measurement.target),
                measurement_target_to_source=measurement.measurement_target_to_source,
                information_diag=measurement.information_diag,
                edge_type=measurement.edge_type,
                robust_delta=measurement.robust_delta,
                dcs_phi=measurement.dcs_phi,
                metadata=dict(measurement.metadata),
            )
        assert measurement.source_local_pose is not None
        assert measurement.target_local_pose is not None
        assert measurement.measured_source_to_target_rotation is not None
        return CoincidentPanoramaFactor(
            source=int(measurement.source),
            target=int(measurement.target),
            source_local_pose=measurement.source_local_pose,
            target_local_pose=measurement.target_local_pose,
            measured_source_to_target_rotation=measurement.measured_source_to_target_rotation,
            center_weight=float(measurement.center_weight),
            rotation_weight=float(measurement.rotation_weight),
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

    def _process_boundary_packet(
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

        start_frame = int(packet.frame_ids[0])
        end_frame = int(packet.frame_ids[-1])
        alignment_diagnostics: dict[str, Any] = {}
        if not self.window_order:
            aligned = True
            start_transform = sim3_identity(device=packet.local_poses_c2w.device)
            alignment_diagnostics = {"reason": "first_window", "chunk_scale_normalization": 1.0}
        else:
            previous_id = int(self.window_order[-1])
            previous_packet = self._last_full_packet
            if previous_packet is None or int(previous_packet.window_id) != previous_id:
                raise RuntimeError("The previous full-resolution window packet is unavailable")
            if int(previous_packet.frame_ids[-1]) != start_frame:
                raise RuntimeError(
                    f"Boundary continuity violated: previous end={previous_packet.frame_ids[-1]} "
                    f"current start={start_frame}"
                )
            if start_frame not in self.graph.nodes:
                raise RuntimeError(f"Shared boundary node {start_frame} is missing")
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
        if boundary_factor is None and not self.allow_unaligned_fallback:
            raise RuntimeError(
                f"Window {window_id} has no valid first/last spherical factor: "
                f"{boundary_diagnostics.get('reason', 'unknown')}"
            )

        if not self.window_order:
            self.graph.add_node(start_frame, start_transform)
            self.boundary_node_order.append(start_frame)
        elif start_frame not in self.graph.nodes:
            self.graph.add_node(start_frame, start_transform)
            self.boundary_node_order.append(start_frame)
        self.window_anchor_nodes[window_id] = start_frame

        if end_frame in self.graph.nodes and end_frame != start_frame:
            raise ValueError(f"Boundary frame node {end_frame} already exists before window {window_id}")
        if end_frame not in self.graph.nodes:
            end_transform = self._node_from_local_pose(
                self.graph.transform(start_frame), packet.local_poses_c2w[-1]
            )
            self.graph.add_node(end_frame, end_transform)
            self.boundary_node_order.append(end_frame)
        if boundary_factor is not None:
            self.graph.add_edge(boundary_factor)
            self._sequential_edges_since_optimization += 1
        self._register_hierarchical_window(window_id, start_frame, end_frame)

        # Loop retrieval operates on window packets, but accepted dense factors
        # are re-keyed to the corresponding boundary-frame nodes. The correlated
        # Sim(3)/coincident summary factor is deliberately not inserted.
        loop_results = self.loop_detector.detect(packet)
        loop_graph = (
            self.submap_graph
            if self.hierarchical_submaps_enabled
            else self.graph
        )
        assert loop_graph is not None
        accepted_loops: list[PanoramaLoopVerification] = []
        pending_loop_factors: list[DenseSphericalFactorBlock] = []
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
                else loop_result.factor
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
                loop_graph.add_edge(merged)
                accepted_loops.append(loop_result)
                pending_loop_factors.append(merged)

        old_window_transforms = self._window_anchor_transforms()
        pre_loop_nodes = {node: value.clone() for node, value in loop_graph.nodes.items()}
        graph_result: Sim3GraphOptimizeResult | None = None
        should_optimize_recent = (
            len(self.boundary_node_order) >= self.global_ba_start_nodes
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
                    active = self.boundary_node_order[-self.global_ba_active_nodes :]
                    graph_result = self.graph.optimize(active, fixed_node_ids={active[0]})
                    self._has_run_global_ba = True
                    self._sequential_edges_since_optimization = 0
        elif should_optimize_recent:
            active = self.boundary_node_order[-self.global_ba_active_nodes :]
            graph_result = self.graph.optimize(
                active,
                fixed_node_ids={active[0]},
            )
            self._has_run_global_ba = True
            self._sequential_edges_since_optimization = 0
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
            else {"moved": 0, "deduplicated": 0}
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

        fusion_stats = self.fusion.fuse_packet(
            packet,
            self._window_anchor_transforms()[window_id],
        )
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
            for refinement_window in self._loop_neighborhood_windows(accepted_loops):
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
                "submap_id": self.window_to_submap.get(window_id),
                "submap_frozen": bool(submap_frozen),
                "submap_count": len(self.submaps),
                "compressed_dense_factors": (
                    self.submaps[self.window_to_submap[window_id]].compressed_dense_factors
                    if window_id in self.window_to_submap
                    else 0
                ),
            },
        )
        self.results.append(result)
        return result

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
        correction = self.fusion.apply_owner_corrections(old_transforms, new_transforms)

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
            for refinement_window in self._loop_neighborhood_windows(accepted_loops):
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
            correction = self.fusion.apply_owner_corrections(
                old_transforms,
                new_transforms,
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
        correction = self.fusion.apply_owner_corrections(old_transforms, self.graph.nodes)
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
