"""Coordinator for window Sim(3), panorama loops, and the single explicit map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from frontend.pano_vggt.alignment import SubmapAligner
from frontend.spherical_selfi.panorama_loop import PanoramaLoopDetector, PanoramaLoopVerification
from frontend.spherical_selfi.window_packet import LocalGaussianWindowPacket
from geometry.sim3 import sim3_identity

from .mapper import PanoGaussianMap, PanoGaussianMapper
from .sim3_graph import GlobalSim3FactorGraph, Sim3GraphEdge, Sim3GraphOptimizeResult
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
        fusion_cfg = dict(self.config.get("voxel_fusion", {}) or {})
        optimize_cfg = dict(self.config.get("map_optimization", {}) or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.allow_unaligned_fallback = bool(graph_cfg.get("allow_unaligned_fallback", False))
        self.recent_optimization_windows = max(2, int(graph_cfg.get("recent_windows", 32)))
        self.map_steps_per_window = max(0, int(optimize_cfg.get("steps_per_window", 0)))
        self.map_steps_on_loop = max(0, int(optimize_cfg.get("steps_on_loop", 0)))
        self.final_map_steps = max(0, int(optimize_cfg.get("final_steps", 0)))
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
        )
        self.overlap_aligner = SubmapAligner(
            align_mode="sim3",
            max_residual=float(graph_cfg.get("max_overlap_residual", 0.35)),
            min_inlier_ratio=float(graph_cfg.get("min_overlap_inlier_ratio", 0.35)),
            max_scale_change=float(graph_cfg.get("max_scale_change", 2.5)),
            min_points=int(graph_cfg.get("min_overlap_points", 32)),
            return_rejected_transform=True,
        )
        self.max_overlap_points = max(32, int(graph_cfg.get("max_overlap_points", 4096)))
        self.loop_detector = PanoramaLoopDetector(
            top_k=int(loop_cfg.get("top_k", 5)),
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
        )
        self.fusion = Stage2GlobalMapFusion(
            gaussian_map,
            voxel_sizes=tuple(fusion_cfg.get("voxel_sizes", (0.04, 0.08, 0.16, 0.32))),
            min_confidence=float(fusion_cfg.get("min_confidence", 0.05)),
            min_opacity=float(fusion_cfg.get("min_opacity", 0.02)),
            max_total_gaussians=int(fusion_cfg.get("max_total_gaussians", 0)),
        )
        self.packets: dict[int, LocalGaussianWindowPacket] = {}
        self._last_full_packet: LocalGaussianWindowPacket | None = None
        self.window_order: list[int] = []
        self.frame_owner_window: dict[int, int] = {}
        self._pose_updates: dict[int, torch.Tensor] = {}
        self._pending_map_optimization: list[tuple[tuple[int, ...], int]] = []
        self.results: list[GlobalWindowBackendResult] = []

    def _overlap_edge(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
    ) -> tuple[Sim3GraphEdge | None, dict[str, Any]]:
        overlap = sorted(set(source.frame_ids) & set(target.frame_ids))
        source_parts, target_parts, weight_parts = [], [], []
        for frame_id in overlap:
            source_index = source.frame_index(frame_id)
            target_index = target.frame_index(frame_id)
            source_points = source.local_points(source_index).detach()
            target_points = target.local_points(target_index).detach().to(source_points)
            if source_points.shape != target_points.shape:
                continue
            source_conf = source.observation.confidence[0, source_index, 0].detach().to(source_points)
            target_conf = target.observation.confidence[0, target_index, 0].detach().to(source_points)
            source_valid = source.valid_mask[0, source_index, 0].to(source_points.device)
            target_valid = target.valid_mask[0, target_index, 0].to(source_points.device)
            height = int(source_points.shape[0])
            rows = torch.arange(height, device=source_points.device, dtype=source_points.dtype) + 0.5
            area = torch.cos(torch.pi * (rows / float(height) - 0.5)).clamp_min(1.0e-4)
            weight = source_conf * target_conf * area[:, None]
            valid = (
                source_valid
                & target_valid
                & torch.isfinite(source_points).all(dim=-1)
                & torch.isfinite(target_points).all(dim=-1)
                & torch.isfinite(weight)
                & (weight > 0.0)
            )
            source_parts.append(source_points[valid])
            target_parts.append(target_points[valid])
            weight_parts.append(weight[valid])
        if not source_parts:
            return None, {"reason": "no_compatible_overlap", "overlap_frame_ids": overlap}
        source_points = torch.cat(source_parts, dim=0)
        target_points = torch.cat(target_parts, dim=0)
        weights = torch.cat(weight_parts, dim=0)
        if int(weights.numel()) > self.max_overlap_points:
            selected = torch.topk(weights, k=self.max_overlap_points, largest=True).indices
            source_points, target_points, weights = source_points[selected], target_points[selected], weights[selected]
        # Measurement maps target anchor coordinates into source anchor coordinates.
        alignment = self.overlap_aligner.align(target_points, source_points, weights)
        diagnostics = {
            "overlap_frame_ids": overlap,
            "overlap_points": int(weights.numel()),
            "overlap_residual": float(alignment.residual),
            "overlap_inlier_ratio": float(alignment.inlier_ratio),
        }
        if not alignment.accepted:
            diagnostics["reason"] = "overlap_alignment_rejected"
            return None, diagnostics
        information = source_points.new_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.75])
        information *= max(1.0, float(weights.numel()) * float(alignment.inlier_ratio))
        return (
            Sim3GraphEdge(
                source=int(source.window_id),
                target=int(target.window_id),
                measurement_target_to_source=alignment.as_matrix().detach(),
                information_diag=information.detach(),
                edge_type="overlap",
                metadata=diagnostics,
            ),
            diagnostics,
        )

    def _initial_transform(
        self,
        previous_id: int,
        edge: Sim3GraphEdge | None,
    ) -> torch.Tensor:
        previous = self.graph.transform(previous_id)
        if edge is None:
            return previous.clone()
        return previous @ edge.measurement_target_to_source.to(previous)

    def _refresh_pose_updates(self) -> None:
        updates: dict[int, torch.Tensor] = {}
        for window_id in self.window_order:
            packet = self.packets[window_id]
            poses = packet.global_poses(self.graph.transform(window_id).to(packet.local_poses_c2w))
            for frame_id, pose in zip(packet.frame_ids, poses):
                # Later overlapping windows have the freshest multi-view estimate.
                updates[int(frame_id)] = pose.detach().cpu().float()
                self.frame_owner_window[int(frame_id)] = int(window_id)
        self._pose_updates.update(updates)

    def pop_pose_updates(self) -> dict[int, torch.Tensor]:
        updates = dict(self._pose_updates)
        self._pose_updates.clear()
        return updates

    def _run_map_optimization(self, frame_ids: tuple[int, ...], steps: int) -> dict[str, float]:
        if self.mapper is None or int(steps) <= 0:
            return {}
        self.mapper.optimizer = self.map.make_optimizer(
            lr=float(self.config.get("map_optimization", {}).get("lr", 2.0e-3))
        )
        try:
            return self.mapper.optimize_resplat_global_window(frame_ids=list(frame_ids), iters=int(steps))
        except (RuntimeError, ValueError) as exc:
            self.mapper.stats.notes.append(f"spherical-Selfi map optimization skipped: {exc!r}")
            return {"steps": 0.0, "loss": 0.0}

    def run_pending_map_optimization(self) -> dict[str, float]:
        """Run low-rate map updates after the system registered window images."""

        if not self._pending_map_optimization:
            return {}
        pending = list(self._pending_map_optimization)
        self._pending_map_optimization.clear()
        last_metrics: dict[str, float] = {}
        for frame_ids, steps in pending:
            last_metrics = self._run_map_optimization(frame_ids, steps)
        return last_metrics

    def process_packet(self, packet: LocalGaussianWindowPacket) -> GlobalWindowBackendResult:
        if not self.enabled:
            raise RuntimeError("SphericalSelfiGlobalBackend is disabled")
        window_id = int(packet.window_id)
        if window_id in self.packets or window_id in self.graph.nodes:
            raise ValueError(f"Duplicate local Gaussian window id {window_id}")

        sequential_edge = None
        alignment_diagnostics: dict[str, Any] = {}
        if not self.window_order:
            initial = sim3_identity(device=packet.local_poses_c2w.device)
            aligned = True
        else:
            previous_id = self.window_order[-1]
            previous_packet = self._last_full_packet
            if previous_packet is None or int(previous_packet.window_id) != int(previous_id):
                raise RuntimeError("The previous full-resolution window packet is unavailable")
            sequential_edge, alignment_diagnostics = self._overlap_edge(previous_packet, packet)
            if sequential_edge is None:
                fallback = self.loop_detector.verify_pair(
                    previous_packet,
                    packet,
                    retrieval_score=1.0,
                    edge_type="sequential",
                )
                if isinstance(fallback.factor, Sim3GraphEdge) and fallback.accepted:
                    sequential_edge = fallback.factor
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

        loop_results = self.loop_detector.detect(packet)
        accepted_loops = [result for result in loop_results if result.accepted and result.factor is not None]
        for result in accepted_loops:
            self.graph.add_edge(result.factor)

        old_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        if accepted_loops:
            graph_result = self.graph.optimize()
        else:
            graph_result = self.graph.optimize(self.window_order[-self.recent_optimization_windows :] + [window_id])
        new_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        correction = self.fusion.apply_owner_corrections(old_transforms, new_transforms)

        compact_packet = packet.compact_for_memory()
        self.packets[window_id] = compact_packet
        self._last_full_packet = packet
        self.window_order.append(window_id)
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
        map_steps = self.map_steps_on_loop if accepted_loops else self.map_steps_per_window
        if map_steps > 0:
            self._pending_map_optimization.append((packet.frame_ids, map_steps))
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
        return {
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

    def finalize(self) -> dict[str, Any]:
        if not self.window_order:
            return {}
        pending_metrics = self.run_pending_map_optimization()
        old_transforms = {node: value.clone() for node, value in self.graph.nodes.items()}
        graph_result = self.graph.optimize()
        correction = self.fusion.apply_owner_corrections(old_transforms, self.graph.nodes)
        self._refresh_pose_updates()
        map_metrics = self._run_map_optimization(tuple(self.frame_owner_window), self.final_map_steps)
        if not map_metrics:
            map_metrics = pending_metrics
        return {
            "graph_initial_objective": graph_result.initial_objective,
            "graph_final_objective": graph_result.final_objective,
            "graph_iterations": graph_result.iterations,
            "moved_gaussians": correction.get("moved", 0),
            "deduplicated_gaussians": correction.get("deduplicated", 0),
            "anchors": self.map.anchor_count(),
            "map_optimization": map_metrics,
        }
