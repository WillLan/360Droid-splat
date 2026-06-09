"""PanoVGGT-M3 dense BA refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

from .dense_matcher import PoseGuidedDenseMatcher
from .factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from .keyframe_memory import KeyframeMemory, KeyframeRecord
from .m3_config import M3SphereConfig
from .spherical_dense_ba import SphericalTangentDenseBA, SphericalTangentDenseBAOutput
from .types import PanoVGGTLocalPrediction


@dataclass(frozen=True)
class DenseBARefinerStats:
    enabled: bool = False
    shadow_mode: bool = True
    mode: str = "local_chunk"
    success: bool = False
    used_refined: bool = False
    fallback_reason: str | None = None
    mean_residual_deg: float = 0.0
    median_residual_deg: float = 0.0
    initial_mean_residual_deg: float = 0.0
    valid_factor_ratio: float = 0.0
    num_factors: int = 0
    valid_factors: int = 0
    history_keyframes: int = 0
    history_factors: int = 0
    pose_update_norm: dict[str, float] = field(default_factory=dict)
    depth_update_norm: dict[str, float] = field(default_factory=dict)
    solver_mode: str = ""
    used_factors: int = 0
    num_pose_variables: int = 0
    num_depth_variables: int = 0
    pose_solve_sec: float = 0.0
    stopped_by_time_budget: bool = False

    @property
    def status_suffix(self) -> str:
        if not self.enabled:
            return ""
        mode = "shadow" if self.shadow_mode else "active"
        if self.success:
            return f"|dense_ba_{mode}_success"
        reason = self.fallback_reason or "unknown"
        return f"|dense_ba_{mode}_fallback:{reason}"

    def as_debug(self) -> dict[str, float]:
        return {
            "dense_ba_enabled": float(self.enabled),
            "dense_ba_shadow_mode": float(self.shadow_mode),
            "dense_ba_success": float(self.success),
            "dense_ba_used_refined": float(self.used_refined),
            "dense_ba_history_keyframes": float(self.history_keyframes),
            "dense_ba_history_factors": float(self.history_factors),
            "dense_ba_mean_residual_deg": float(self.mean_residual_deg),
            "dense_ba_median_residual_deg": float(self.median_residual_deg),
            "dense_ba_initial_mean_residual_deg": float(self.initial_mean_residual_deg),
            "dense_ba_valid_factor_ratio": float(self.valid_factor_ratio),
            "dense_ba_num_factors": float(self.num_factors),
            "dense_ba_valid_factors": float(self.valid_factors),
            "dense_ba_pose_update_mean": float(self.pose_update_norm.get("mean", 0.0)),
            "dense_ba_pose_update_max": float(self.pose_update_norm.get("max", 0.0)),
            "dense_ba_pose_update_rot_max_deg": float(self.pose_update_norm.get("rot_max_deg", 0.0)),
            "dense_ba_pose_update_trans_max": float(self.pose_update_norm.get("trans_max", 0.0)),
            "dense_ba_depth_update_mean": float(self.depth_update_norm.get("mean", 0.0)),
            "dense_ba_depth_update_max": float(self.depth_update_norm.get("max", 0.0)),
            "dense_ba_solver_pose_only": float(str(self.solver_mode) == "pose_only_factor_graph"),
            "dense_ba_used_factors": float(self.used_factors),
            "dense_ba_num_pose_variables": float(self.num_pose_variables),
            "dense_ba_num_depth_variables": float(self.num_depth_variables),
            "dense_ba_pose_solve_sec": float(self.pose_solve_sec),
            "dense_ba_stopped_by_time_budget": float(self.stopped_by_time_budget),
        }


class PanoVGGTDenseBARefiner:
    """Config-gated local dense BA refiner for PanoVGGT predictions."""

    def __init__(self, config: M3SphereConfig) -> None:
        self.config = config
        self.solver = SphericalTangentDenseBA(
            factor_chunk_size=config.dense_ba.factor_chunk_size,
            huber_delta_deg=config.dense_ba.huber_delta_deg,
        )
        self.last_factor_graph: DenseSphereFactorGraph | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.dense_ba.enabled)

    @property
    def shadow_mode(self) -> bool:
        return bool(self.config.dense_ba.shadow_mode)

    def refine(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        *,
        factor_graph: DenseSphereFactorGraph | None = None,
        keyframe_memory: KeyframeMemory | None = None,
        current_start: int = 0,
        current_count: int | None = None,
        fixed_frames_override: int | None = None,
    ) -> tuple[PanoVGGTLocalPrediction, DenseBARefinerStats]:
        """Return a refined prediction or the original prediction with fallback stats."""

        self.last_factor_graph = None
        if not self.enabled:
            return pred, self._stats(enabled=False, success=False, reason="disabled")
        if self.config.dense_ba.residual_mode != "tangent":
            return pred, self._stats(success=False, reason="unsupported_residual_mode")
        mode = str(self.config.dense_ba.mode).lower()
        if mode not in {"local_chunk", "history_window", "keyframe_graph"}:
            return pred, self._stats(success=False, reason="unsupported_mode")
        missing = self._missing_prediction_fields(pred)
        if missing:
            return pred, self._stats(success=False, reason=f"missing_{missing}")
        feature_hw = pred.feature_hw
        image_hw = pred.image_hw
        assert feature_hw is not None and image_hw is not None

        graph = factor_graph if factor_graph is not None else self._build_factor_graph(pred)
        solver_pred = pred
        fixed_frames = self._fixed_frames(frame_ids)
        explicit_current_start = max(0, int(current_start))
        explicit_current_count = current_count
        current_start = explicit_current_start
        history_keyframes = 0
        if fixed_frames_override is not None:
            fixed_frames = max(1, min(int(fixed_frames_override), int(pred.poses_c2w.shape[0])))
        if mode in {"history_window", "keyframe_graph"} and explicit_current_start == 0:
            prepared = self._prepare_history_window(
                pred,
                frame_ids,
                factor_graph=graph,
                keyframe_memory=keyframe_memory,
            )
            if prepared is not None:
                solver_pred, graph, current_start, fixed_frames, history_keyframes = prepared
        self.last_factor_graph = graph
        if graph is None or not graph.factors:
            return pred, self._stats(success=False, reason="empty_factors")

        graph_metrics = graph.metrics()
        graph_metrics["history_keyframes"] = float(history_keyframes)
        graph_metrics["history_factors"] = float(graph.metadata.get("history_factors", 0.0))
        num_factors = int(graph_metrics.get("num_factors", 0.0))
        valid_factors = int(graph_metrics.get("valid_factors", 0.0))
        valid_ratio = float(graph_metrics.get("valid_factor_ratio", 0.0))
        if valid_factors < int(self.config.dense_ba.min_num_factors):
            return pred, self._stats(success=False, reason="too_few_factors", graph_metrics=graph_metrics)
        if valid_ratio < float(self.config.dense_ba.min_valid_factor_ratio):
            return pred, self._stats(success=False, reason="low_valid_factor_ratio", graph_metrics=graph_metrics)

        depth_low = F.interpolate(solver_pred.depth.float(), size=feature_hw, mode="bilinear", align_corners=False).clamp_min(1.0e-6)
        log_inv_depth_low = depth_low.reciprocal().clamp_min(1.0e-6).log()
        output = self.solver(
            solver_pred.poses_c2w.to(device=solver_pred.depth.device, dtype=solver_pred.depth.dtype),
            log_inv_depth_low.to(device=solver_pred.depth.device, dtype=solver_pred.depth.dtype),
            graph,
            fixed_frames=fixed_frames,
            iters=self.config.dense_ba.iters,
            damping=self.config.dense_ba.lm,
            optimize_pose=self.config.dense_ba.optimize_pose,
            optimize_depth=self.config.dense_ba.optimize_depth,
            pose_prior_weight=self.config.dense_ba.pose_prior_weight,
            depth_prior_weight=self.config.dense_ba.depth_prior_weight,
            max_pose_update_deg=self.config.dense_ba.max_pose_update_deg,
            max_logdepth_update=self.config.dense_ba.max_logdepth_update,
            line_search=self.config.dense_ba.line_search,
            solver_mode=self.config.dense_ba.solver_mode,
            max_ba_factors=self.config.dense_ba.max_ba_factors,
            max_depth_variables=self.config.dense_ba.max_depth_variables,
            max_solver_sec=self.config.dense_ba.max_solver_sec,
        )
        if output.failed:
            return pred, self._stats(success=False, reason=str(output.debug.get("fallback_reason", "solver_failed")), graph_metrics=graph_metrics, output=output)
        fallback_reason = self._validate_output(solver_pred, log_inv_depth_low, output, fixed_frames=fixed_frames)
        if fallback_reason is not None:
            return pred, self._stats(success=False, reason=fallback_reason, graph_metrics=graph_metrics, output=output)

        initial = float(output.debug.get("initial_mean_angular_residual_deg", output.mean_angular_residual_deg))
        if (
            self.config.dense_ba.fallback_if_residual_worse
            and output.mean_angular_residual_deg > initial * float(self.config.dense_ba.residual_worse_tolerance)
        ):
            return pred, self._stats(success=False, reason="residual_worse", graph_metrics=graph_metrics, output=output)

        refined_full = self._rebuild_prediction(solver_pred, log_inv_depth_low, output)
        if explicit_current_start > 0 or explicit_current_count is not None:
            refined = self._slice_prediction(refined_full, explicit_current_start, explicit_current_count)
        else:
            refined = self._slice_current_prediction(pred, refined_full, current_start)
        stats = self._stats(success=True, reason=None, graph_metrics=graph_metrics, output=output)
        refined = replace(
            refined,
            matching_debug={
                **(pred.matching_debug or {}),
                **graph_metrics,
                **stats.as_debug(),
            },
        )
        return refined, stats

    def _missing_prediction_fields(self, pred: PanoVGGTLocalPrediction) -> str | None:
        required = {
            "dense_descriptors": pred.dense_descriptors,
            "match_confidence": pred.match_confidence,
            "sky_prob": pred.sky_prob,
            "feature_hw": pred.feature_hw,
            "image_hw": pred.image_hw,
        }
        for name, value in required.items():
            if value is None:
                return name
        return None

    def _build_factor_graph(self, pred: PanoVGGTLocalPrediction) -> DenseSphereFactorGraph | None:
        if pred.dense_descriptors is None or pred.match_confidence is None or pred.sky_prob is None or pred.feature_hw is None or pred.image_hw is None:
            return None
        dense = self.config.dense_matching
        matcher = PoseGuidedDenseMatcher(
            search_radius=dense.search_radius,
            topk=dense.topk,
            min_match_confidence=dense.min_match_confidence,
            min_static_confidence=dense.min_static_confidence,
            min_match_score=dense.min_match_score,
            max_factors=dense.max_factors,
            max_samples_per_edge=dense.max_samples_per_edge,
            use_wraparound=dense.use_wraparound,
            forward_backward=dense.forward_backward,
            fb_tolerance=dense.fb_tolerance,
            use_depth_consistency=dense.use_depth_consistency,
            depth_consistency_rel=dense.depth_consistency_rel,
            depth_consistency_abs=dense.depth_consistency_abs,
        )
        edges = DenseSphereFactorGraph.build_edges(
            int(pred.poses_c2w.shape[0]),
            temporal_radius=self.config.inference_window.temporal_radius,
            max_edges=self.config.inference_window.max_edges,
            device=pred.depth.device,
        )
        return matcher.match(
            poses_c2w=pred.poses_c2w,
            depth=pred.depth,
            dense_descriptors=pred.dense_descriptors,
            match_confidence=pred.match_confidence,
            sky_prob=pred.sky_prob,
            static_confidence=pred.static_confidence,
            image_hw=pred.image_hw,
            feature_hw=pred.feature_hw,
            edge_pairs=edges,
        )

    def _prepare_history_window(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        *,
        factor_graph: DenseSphereFactorGraph | None,
        keyframe_memory: KeyframeMemory | None,
    ) -> tuple[PanoVGGTLocalPrediction, DenseSphereFactorGraph, int, int, int] | None:
        if keyframe_memory is None or int(self.config.dense_ba.history_keyframes) <= 0:
            return None
        records = self._compatible_history_records(keyframe_memory, pred, frame_ids)
        if not records:
            return None

        history_count = len(records)
        solver_pred = self._augment_prediction_with_history(pred, records)
        shifted_local = self._shift_factor_graph(factor_graph, offset=history_count) if factor_graph is not None else None
        history_graph = self._build_history_factor_graph(solver_pred, history_count=history_count)

        factors: list[DenseSphereFactor] = []
        edge_parts: list[torch.Tensor] = []
        if shifted_local is not None:
            factors.extend(shifted_local.factors)
            if shifted_local.edges is not None:
                edge_parts.append(shifted_local.edges)
        if history_graph is not None:
            factors.extend(history_graph.factors)
            if history_graph.edges is not None:
                edge_parts.append(history_graph.edges)
        if not factors:
            return None
        edges = torch.cat(edge_parts, dim=0) if edge_parts else None
        history_factors = 0 if history_graph is None else history_graph.num_factors
        graph = DenseSphereFactorGraph(
            factors=factors,
            edges=edges,
            metadata={
                **(shifted_local.metadata if shifted_local is not None else {}),
                "mode": "history_window",
                "history_keyframes": float(history_count),
                "history_factors": float(history_factors),
            },
        )
        fixed_current = self._fixed_frames(frame_ids)
        fixed_frames = min(int(solver_pred.poses_c2w.shape[0]), history_count + fixed_current)
        return solver_pred, graph, history_count, fixed_frames, history_count

    def _compatible_history_records(
        self,
        keyframe_memory: KeyframeMemory,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
    ) -> list[KeyframeRecord]:
        assert pred.dense_descriptors is not None
        assert pred.match_confidence is not None
        assert pred.sky_prob is not None
        assert pred.feature_hw is not None
        current_ids = {int(frame_id) for frame_id in frame_ids}
        feature_hw = tuple(int(v) for v in pred.feature_hw)
        desc_dim = int(pred.dense_descriptors.shape[1])
        records: list[KeyframeRecord] = []
        for record in keyframe_memory.recent(int(self.config.dense_ba.history_keyframes)):
            if int(record.frame_id) in current_ids:
                continue
            if record.feature_hw is not None and tuple(int(v) for v in record.feature_hw) != feature_hw:
                continue
            if record.pose_c2w.shape != (4, 4) or not torch.isfinite(record.pose_c2w).all():
                continue
            depth = _as_frame_tensor(record.depth_low)
            desc = _as_frame_tensor(record.dense_descriptors)
            match = _as_frame_tensor(record.match_confidence)
            sky = _as_frame_tensor(record.sky_prob)
            if depth is None or desc is None or match is None or sky is None:
                continue
            if int(depth.shape[0]) != 1 or not torch.isfinite(depth).all():
                continue
            if int(desc.shape[0]) != desc_dim or tuple(desc.shape[-2:]) != feature_hw:
                continue
            if int(match.shape[0]) != 1 or tuple(match.shape[-2:]) != feature_hw:
                continue
            if int(sky.shape[0]) != 1 or tuple(sky.shape[-2:]) != feature_hw:
                continue
            if record.static_confidence is not None:
                static = _as_frame_tensor(record.static_confidence)
                if static is None or int(static.shape[0]) != 1 or tuple(static.shape[-2:]) != feature_hw:
                    continue
            records.append(record)
        return records

    def _augment_prediction_with_history(
        self,
        pred: PanoVGGTLocalPrediction,
        records: list[KeyframeRecord],
    ) -> PanoVGGTLocalPrediction:
        assert pred.dense_descriptors is not None
        assert pred.match_confidence is not None
        assert pred.sky_prob is not None
        device = pred.depth.device
        dtype = pred.depth.dtype
        depth_hw = tuple(int(v) for v in pred.depth.shape[-2:])
        poses = torch.stack([record.pose_c2w.to(device=device, dtype=dtype) for record in records], dim=0)
        hist_depth = torch.stack(
            [_resize_frame_depth(_as_frame_tensor(record.depth_low), depth_hw, device=device, dtype=dtype) for record in records],
            dim=0,
        )
        hist_conf = torch.ones_like(hist_depth)
        hist_local = _build_local_points(hist_depth)
        hist_world = _local_points_to_world(hist_local, poses)

        hist_desc = torch.stack(
            [_as_frame_tensor(record.dense_descriptors).to(device=device, dtype=pred.dense_descriptors.dtype) for record in records],
            dim=0,
        )
        hist_match = torch.stack(
            [_as_frame_tensor(record.match_confidence).to(device=device, dtype=pred.match_confidence.dtype) for record in records],
            dim=0,
        )
        hist_sky = torch.stack(
            [_as_frame_tensor(record.sky_prob).to(device=device, dtype=pred.sky_prob.dtype) for record in records],
            dim=0,
        )
        static_confidence = None
        if pred.static_confidence is not None and all(record.static_confidence is not None for record in records):
            hist_static = torch.stack(
                [_as_frame_tensor(record.static_confidence).to(device=device, dtype=pred.static_confidence.dtype) for record in records],
                dim=0,
            )
            static_confidence = torch.cat([hist_static, pred.static_confidence], dim=0)
        current_local = pred.local_points if pred.local_points is not None else _build_local_points(pred.depth)
        return replace(
            pred,
            poses_c2w=torch.cat([poses, pred.poses_c2w.to(device=device, dtype=dtype)], dim=0),
            depth=torch.cat([hist_depth, pred.depth.to(device=device, dtype=dtype)], dim=0),
            confidence=torch.cat([hist_conf, pred.confidence.to(device=device, dtype=dtype)], dim=0),
            chunk_world_points=torch.cat([hist_world, pred.chunk_world_points.to(device=device, dtype=dtype)], dim=0),
            local_points=torch.cat([hist_local, current_local.to(device=device, dtype=dtype)], dim=0),
            dense_descriptors=torch.cat([hist_desc, pred.dense_descriptors.to(device=device, dtype=hist_desc.dtype)], dim=0),
            match_confidence=torch.cat([hist_match, pred.match_confidence.to(device=device, dtype=hist_match.dtype)], dim=0),
            static_confidence=static_confidence,
            sky_prob=torch.cat([hist_sky, pred.sky_prob.to(device=device, dtype=hist_sky.dtype)], dim=0),
        )

    def _build_history_factor_graph(
        self,
        pred: PanoVGGTLocalPrediction,
        *,
        history_count: int,
    ) -> DenseSphereFactorGraph | None:
        if history_count <= 0:
            return None
        current_count = int(pred.poses_c2w.shape[0]) - int(history_count)
        if current_count <= 0:
            return None
        dense = self.config.dense_matching
        matcher = PoseGuidedDenseMatcher(
            search_radius=dense.search_radius,
            topk=dense.topk,
            min_match_confidence=dense.min_match_confidence,
            min_static_confidence=dense.min_static_confidence,
            min_match_score=dense.min_match_score,
            max_factors=dense.max_factors,
            max_samples_per_edge=dense.max_samples_per_edge,
            use_wraparound=dense.use_wraparound,
            forward_backward=dense.forward_backward,
            fb_tolerance=dense.fb_tolerance,
            use_depth_consistency=dense.use_depth_consistency,
            depth_consistency_rel=dense.depth_consistency_rel,
            depth_consistency_abs=dense.depth_consistency_abs,
        )
        edges = torch.tensor(
            [
                (int(history_count) + current_idx, hist_idx)
                for current_idx in range(current_count)
                for hist_idx in range(int(history_count))
            ],
            dtype=torch.long,
            device=pred.depth.device,
        )
        return matcher.match(
            poses_c2w=pred.poses_c2w,
            depth=pred.depth,
            dense_descriptors=pred.dense_descriptors,
            match_confidence=pred.match_confidence,
            sky_prob=pred.sky_prob,
            static_confidence=pred.static_confidence,
            image_hw=pred.image_hw,
            feature_hw=pred.feature_hw,
            edge_pairs=edges,
        )

    @staticmethod
    def _shift_factor_graph(
        graph: DenseSphereFactorGraph,
        *,
        offset: int,
    ) -> DenseSphereFactorGraph:
        if int(offset) == 0:
            return graph
        factors = [replace(factor, src=int(factor.src) + int(offset), tgt=int(factor.tgt) + int(offset)) for factor in graph.factors]
        edges = None if graph.edges is None else graph.edges + int(offset)
        return DenseSphereFactorGraph(factors=factors, edges=edges, metadata=dict(graph.metadata))

    def _fixed_frames(self, frame_ids: tuple[int, ...]) -> int:
        fixed = int(self.config.dense_ba.fixed_frames)
        if self.config.dense_ba.fixed_policy == "first_frame":
            return max(1, min(fixed, len(frame_ids)))
        return max(1, min(fixed, len(frame_ids)))

    def _validate_output(
        self,
        pred: PanoVGGTLocalPrediction,
        log_inv_depth_low: torch.Tensor,
        output: SphericalTangentDenseBAOutput,
        *,
        fixed_frames: int,
    ) -> str | None:
        if not torch.isfinite(output.poses_c2w).all() or not torch.isfinite(output.inverse_depth).all():
            return "non_finite_outputs"
        if not torch.allclose(output.poses_c2w[:fixed_frames], pred.poses_c2w[:fixed_frames].to(output.poses_c2w), atol=1.0e-5):
            return "fixed_frame_changed"
        log_delta = output.log_inv_depth - log_inv_depth_low.to(output.log_inv_depth)
        if not torch.isfinite(log_delta).all():
            return "non_finite_depth_update"
        update_abs = log_delta.abs().detach().reshape(-1)
        if update_abs.numel():
            q = min(1.0, max(0.0, float(self.config.dense_ba.logdepth_update_quantile)))
            update_q = float(torch.quantile(update_abs.float().cpu(), q).item())
            if q < 1.0 and update_q > float(self.config.dense_ba.max_logdepth_update) + 1.0e-5:
                return "depth_update_quantile_too_large"
        if output.pose_update_norm.get("rot_max_deg", 0.0) > float(self.config.dense_ba.max_pose_update_deg) + 0.2:
            return "pose_update_too_large"
        return None

    def _rebuild_prediction(
        self,
        pred: PanoVGGTLocalPrediction,
        log_inv_depth_low: torch.Tensor,
        output: SphericalTangentDenseBAOutput,
    ) -> PanoVGGTLocalPrediction:
        log_delta_low = output.log_inv_depth - log_inv_depth_low.to(output.log_inv_depth)
        log_delta_full = F.interpolate(
            log_delta_low.float(),
            size=pred.depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_inv = pred.depth.float().clamp_min(1.0e-6).reciprocal() * torch.exp(log_delta_full).clamp_min(1.0e-6)
        refined_depth = refined_inv.clamp_min(1.0e-6).reciprocal()
        local_points = _build_local_points(refined_depth)
        chunk_world_points = _local_points_to_world(local_points, output.poses_c2w.to(refined_depth))
        return replace(
            pred,
            poses_c2w=output.poses_c2w.to(pred.poses_c2w),
            depth=refined_depth.to(pred.depth),
            local_points=local_points.to(pred.depth),
            chunk_world_points=chunk_world_points.to(pred.depth),
            ba_residual_angular=float(math.radians(output.mean_angular_residual_deg)),
            ba_valid_ratio=float(output.valid_mask.float().mean().detach().cpu()) if output.valid_mask.numel() else 0.0,
            ba_update_norm={
                "pose_mean": float(output.pose_update_norm.get("mean", 0.0)),
                "pose_max": float(output.pose_update_norm.get("max", 0.0)),
                "depth_mean": float(output.depth_update_norm.get("mean", 0.0)),
                "depth_max": float(output.depth_update_norm.get("max", 0.0)),
            },
        )

    def _slice_current_prediction(
        self,
        original: PanoVGGTLocalPrediction,
        refined: PanoVGGTLocalPrediction,
        current_start: int,
    ) -> PanoVGGTLocalPrediction:
        if int(current_start) == 0:
            return refined
        count = int(original.poses_c2w.shape[0])
        idx = slice(int(current_start), int(current_start) + count)
        return replace(
            original,
            poses_c2w=refined.poses_c2w[idx].to(original.poses_c2w),
            depth=refined.depth[idx].to(original.depth),
            confidence=original.confidence,
            chunk_world_points=refined.chunk_world_points[idx].to(original.chunk_world_points),
            local_points=None if refined.local_points is None else refined.local_points[idx].to(original.depth),
            ba_residual_angular=refined.ba_residual_angular,
            ba_valid_ratio=refined.ba_valid_ratio,
            ba_update_norm=refined.ba_update_norm,
        )

    def _slice_prediction(
        self,
        pred: PanoVGGTLocalPrediction,
        start: int,
        count: int | None,
    ) -> PanoVGGTLocalPrediction:
        n = int(pred.poses_c2w.shape[0])
        start_i = max(0, min(int(start), n))
        count_i = n - start_i if count is None else max(0, min(int(count), n - start_i))
        idx = slice(start_i, start_i + count_i)
        return replace(
            pred,
            poses_c2w=pred.poses_c2w[idx],
            depth=pred.depth[idx],
            confidence=pred.confidence[idx],
            chunk_world_points=pred.chunk_world_points[idx],
            local_points=None if pred.local_points is None else pred.local_points[idx],
            global_points=None if pred.global_points is None else pred.global_points[idx],
            descriptors=None if pred.descriptors is None else pred.descriptors[idx],
            dense_descriptors=None if pred.dense_descriptors is None else pred.dense_descriptors[idx],
            match_confidence=None if pred.match_confidence is None else pred.match_confidence[idx],
            static_confidence=None if pred.static_confidence is None else pred.static_confidence[idx],
            sky_logits=None if pred.sky_logits is None else pred.sky_logits[idx],
            sky_prob=None if pred.sky_prob is None else pred.sky_prob[idx],
        )

    def _stats(
        self,
        *,
        enabled: bool | None = None,
        success: bool,
        reason: str | None,
        graph_metrics: dict[str, float] | None = None,
        output: SphericalTangentDenseBAOutput | None = None,
    ) -> DenseBARefinerStats:
        metrics = graph_metrics or {}
        initial = 0.0
        debug = {}
        if output is not None:
            debug = dict(output.debug)
            initial = float(debug.get("initial_mean_angular_residual_deg", 0.0))
        return DenseBARefinerStats(
            enabled=self.enabled if enabled is None else bool(enabled),
            shadow_mode=self.shadow_mode,
            mode=str(self.config.dense_ba.mode),
            success=bool(success),
            used_refined=bool(success and not self.shadow_mode),
            fallback_reason=reason,
            mean_residual_deg=0.0 if output is None else float(output.mean_angular_residual_deg),
            median_residual_deg=0.0 if output is None else float(output.median_angular_residual_deg),
            initial_mean_residual_deg=initial,
            valid_factor_ratio=float(metrics.get("valid_factor_ratio", 0.0)),
            num_factors=int(metrics.get("num_factors", 0.0)),
            valid_factors=int(metrics.get("valid_factors", 0.0)),
            history_keyframes=int(metrics.get("history_keyframes", 0.0)),
            history_factors=int(metrics.get("history_factors", 0.0)),
            pose_update_norm={} if output is None else dict(output.pose_update_norm),
            depth_update_norm={} if output is None else dict(output.depth_update_norm),
            solver_mode=str(debug.get("solver_mode", self.config.dense_ba.solver_mode)),
            used_factors=int(debug.get("used_factors", metrics.get("valid_factors", 0.0))),
            num_pose_variables=int(debug.get("num_pose_variables", 0)),
            num_depth_variables=int(debug.get("num_depth_variables", 0)),
            pose_solve_sec=float(debug.get("pose_solve_sec", 0.0)),
            stopped_by_time_budget=bool(debug.get("stopped_by_time_budget", False)),
        )


def _build_local_points(depth: torch.Tensor) -> torch.Tensor:
    n, _, height, width = depth.shape
    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    bearing = erp_pixel_to_bearing(grid, height, width).to(device=depth.device, dtype=depth.dtype)
    return bearing.unsqueeze(0) * depth[:, 0].unsqueeze(-1)


def _local_points_to_world(local_points: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    n = local_points.shape[0]
    rot = poses_c2w[:, :3, :3]
    trans = poses_c2w[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", rot, local_points) + trans.view(n, 1, 1, 3)


def _as_frame_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    out = value.detach()
    if out.ndim == 4 and int(out.shape[0]) == 1:
        out = out[0]
    if out.ndim == 2:
        out = out.unsqueeze(0)
    if out.ndim != 3:
        return None
    return out


def _resize_frame_depth(
    value: torch.Tensor | None,
    size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None or int(value.shape[0]) != 1:
        raise ValueError("History keyframe depth must have shape 1xHxW or 1x1xHxW.")
    depth = value.to(device=device, dtype=dtype).clamp_min(1.0e-6)
    if tuple(depth.shape[-2:]) == tuple(size):
        return depth
    return F.interpolate(
        depth.unsqueeze(0),
        size=tuple(int(v) for v in size),
        mode="bilinear",
        align_corners=False,
    )[0].clamp_min(1.0e-6)
