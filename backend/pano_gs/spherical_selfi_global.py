"""Coordinator for window Sim(3), panorama loops, and the single explicit map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from frontend.pano_vggt.alignment import SubmapAligner
from frontend.spherical_selfi.panorama_loop import PanoramaLoopDetector, PanoramaLoopVerification
from frontend.spherical_selfi.window_packet import LocalGaussianWindowPacket
from geometry.spherical_pseudo_correspondence import sample_joint_valid_fibonacci_uv
from geometry.spherical_erp import sample_erp_with_wrap
from geometry.sim3 import (
    apply_sim3_to_c2w,
    rebase_c2w_to_sim3_anchor,
    sim3_components,
    sim3_from_components,
    sim3_identity,
)

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
        validation_cfg = dict(self.config.get("geometry_validation", {}) or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.allow_unaligned_fallback = bool(graph_cfg.get("allow_unaligned_fallback", False))
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
        self.recent_optimization_windows = max(2, int(graph_cfg.get("recent_windows", 32)))
        self.map_steps_per_window = max(0, int(optimize_cfg.get("steps_per_window", 0)))
        self.map_steps_on_loop = max(
            0,
            int(optimize_cfg.get("extra_steps_on_loop", optimize_cfg.get("steps_on_loop", 0))),
        )
        self.final_map_steps = max(0, int(optimize_cfg.get("final_steps", 0)))
        self.map_optimize_config = optimize_cfg
        self.geometry_validation_enabled = bool(validation_cfg.get("enabled", True))
        self.geometry_tolerance = float(validation_cfg.get("tolerance", 1.0e-5))
        self.geometry_rollback_on_failure = bool(validation_cfg.get("rollback_on_failure", True))
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
        self.max_overlap_points = max(
            32,
            int(graph_cfg.get("overlap_num_queries", graph_cfg.get("max_overlap_points", 4096))),
        )
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
        self.frame_depth_owner_window: dict[int, int] = {}
        self.frame_windows: dict[int, set[int]] = {}
        self._geometry_updates: dict[int, FrameGeometryUpdate] = {}
        self._pending_map_optimization: list[tuple[int, tuple[int, ...], int]] = []
        self._optimization_packets: dict[int, LocalGaussianWindowPacket] = {}
        self.results: list[GlobalWindowBackendResult] = []

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
            source_confidence=source.observation.confidence[0, source_index].detach(),
            target_confidence=target.observation.confidence[0, target_index].detach().to(source_depth.device),
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
        source_feature = source.adapter_features[0, source_index].detach().to(samples.bearing)
        target_feature = target.adapter_features[0, target_index].detach().to(samples.bearing)

        def feature_uv(feature: torch.Tensor, observation_hw: tuple[int, int]) -> torch.Tensor:
            feature_h, feature_w = int(feature.shape[-2]), int(feature.shape[-1])
            image_h, image_w = observation_hw
            uv = samples.uv.clone()
            uv[:, 0] *= float(feature_w) / float(image_w)
            uv[:, 1] *= float(feature_h) / float(image_h)
            return uv

        source_descriptor = torch.nn.functional.normalize(
            sample_erp_with_wrap(source_feature, feature_uv(source_feature, source.observation.image_size)),
            dim=-1,
            eps=1.0e-8,
        )
        target_descriptor = torch.nn.functional.normalize(
            sample_erp_with_wrap(target_feature, feature_uv(target_feature, target.observation.image_size)),
            dim=-1,
            eps=1.0e-8,
        )
        descriptor_cosine = (source_descriptor * target_descriptor).sum(dim=-1).clamp(-1.0, 1.0)
        descriptor_consistency = torch.sigmoid(10.0 * (descriptor_cosine - self.min_match_cosine))
        weights = (
            samples.source_confidence
            * samples.target_confidence
            * (1.0 - samples.source_sky_probability)
            * (1.0 - samples.target_sky_probability)
            * descriptor_consistency
        ).clamp_min(0.0)
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
            "mean_overlap_descriptor_cosine": float(descriptor_cosine.mean().detach().cpu()),
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
            if live_packet is not None:
                window_scale, _, _ = sim3_components(self.graph.transform(int(window_id)))
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
            fixed_frame_ids = [int(self.packets[self.window_order[0]].frame_ids[0])] if self.window_order else []
            settings = {
                "gaussian_lr": float(self.map_optimize_config.get("gaussian_lr", self.map_optimize_config.get("lr", 2.0e-3))),
                "pose_lr": float(self.map_optimize_config.get("pose_lr", 1.0e-3)),
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
                extra_loss_fn=lambda trainable_pose_ids: self._joint_graph_pose_loss(
                    int(window_id), trainable_pose_ids
                ),
            )
            if float(metrics.get("window_rollback", 0.0)) == 0.0:
                try:
                    self._synchronize_joint_optimized_window(int(window_id))
                except (RuntimeError, ValueError) as exc:
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
        except (RuntimeError, ValueError) as exc:
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

        for factor in self.graph.edges:
            if not isinstance(factor, DenseSphericalFactorBlock):
                continue
            if int(factor.source) != int(window_id) and int(factor.target) != int(window_id):
                continue
            source_frame_id = factor.metadata.get("source_frame_id")
            target_frame_id = factor.metadata.get("target_frame_id")
            if source_frame_id is None or target_frame_id is None:
                continue
            source_pose = camera_pose(int(source_frame_id))
            target_pose = camera_pose(int(target_frame_id))
            if source_pose is None or target_pose is None:
                continue
            from geometry.sim3 import sim3_components

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
            for variant, local_poses, observation in packet_snapshots.values():
                variant.local_poses_c2w = local_poses
                variant.observation = observation
            self._refresh_factor_local_poses(affected_windows)
            raise

    def process_packet(self, packet: LocalGaussianWindowPacket) -> GlobalWindowBackendResult:
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
                if isinstance(fallback.factor, Sim3GraphEdge) and fallback.accepted:
                    sequential_edge = fallback.factor
                    sequential_extra_factors = fallback.dense_factors
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
        accepted_loops = [result for result in loop_results if result.accepted and result.factor is not None]
        for result in accepted_loops:
            self.graph.add_edge(result.factor)
            for dense_factor in result.dense_factors:
                self.graph.add_edge(dense_factor)

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
        map_steps = self.map_steps_per_window + (self.map_steps_on_loop if accepted_loops else 0)
        if map_steps > 0:
            self._pending_map_optimization.append((window_id, packet.frame_ids, map_steps))
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
