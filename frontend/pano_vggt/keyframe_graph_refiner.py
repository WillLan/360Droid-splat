"""Persistent keyframe correspondence graph and BA helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

from .dense_matcher import PoseGuidedDenseMatcher
from .factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from .keyframe_memory import KeyframeCorrespondenceEdge, KeyframeCorrespondenceGraph, KeyframeMemory, KeyframeRecord
from .m3_config import M3SphereConfig
from .spherical_dense_ba import SphericalTangentDenseBA
from .types import PanoVGGTLocalPrediction


@dataclass(frozen=True)
class KeyframeGraphBAStats:
    """Scalar diagnostics for keyframe graph BA stages."""

    enabled: bool = False
    stage: str = ""
    success: bool = False
    used_refined: bool = False
    fallback_reason: str | None = None
    num_edges: int = 0
    num_factors: int = 0
    valid_factors: int = 0
    valid_factor_ratio: float = 0.0
    mean_residual_deg: float = 0.0
    pose_update_norm: dict[str, float] | None = None

    def as_debug(self) -> dict[str, float]:
        pose = self.pose_update_norm or {}
        return {
            f"keyframe_graph_{self.stage}_enabled": float(self.enabled),
            f"keyframe_graph_{self.stage}_success": float(self.success),
            f"keyframe_graph_{self.stage}_used_refined": float(self.used_refined),
            f"keyframe_graph_{self.stage}_num_edges": float(self.num_edges),
            f"keyframe_graph_{self.stage}_num_factors": float(self.num_factors),
            f"keyframe_graph_{self.stage}_valid_factors": float(self.valid_factors),
            f"keyframe_graph_{self.stage}_valid_factor_ratio": float(self.valid_factor_ratio),
            f"keyframe_graph_{self.stage}_mean_residual_deg": float(self.mean_residual_deg),
            f"keyframe_graph_{self.stage}_pose_update_mean": float(pose.get("mean", 0.0)),
            f"keyframe_graph_{self.stage}_pose_update_max": float(pose.get("max", 0.0)),
            f"keyframe_graph_{self.stage}_pose_update_rot_max_deg": float(pose.get("rot_max_deg", 0.0)),
        }


class PanoVGGTKeyframeGraphRefiner:
    """Config-gated keyframe graph correspondence and pose-only BA refiner."""

    def __init__(self, config: M3SphereConfig) -> None:
        self.config = config
        self.solver = SphericalTangentDenseBA(
            factor_chunk_size=config.dense_ba.factor_chunk_size,
            huber_delta_deg=config.dense_ba.huber_delta_deg,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.keyframe_graph.enabled)

    def refine_current_to_last(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        *,
        new_local_indices: Iterable[int],
        last_keyframe: KeyframeRecord | None,
    ) -> tuple[PanoVGGTLocalPrediction, KeyframeGraphBAStats]:
        """Refine post-alignment current chunk frames against the last keyframe."""

        if not (self.enabled and self.config.keyframe_graph.current_to_last_ba):
            return pred, self._stats(stage="current_to_last", success=False, reason="disabled")
        indices = tuple(int(idx) for idx in new_local_indices)
        if not indices:
            return pred, self._stats(stage="current_to_last", success=False, reason="empty_current_frames")
        if last_keyframe is None:
            return pred, self._stats(stage="current_to_last", success=False, reason="missing_last_keyframe")
        missing = _missing_prediction_fields(pred)
        if missing is not None:
            return pred, self._stats(stage="current_to_last", success=False, reason=f"missing_{missing}")
        if not _record_has_dense_fields(last_keyframe):
            return pred, self._stats(stage="current_to_last", success=False, reason="incomplete_last_keyframe")

        try:
            solver_pred = _prediction_from_last_and_current(pred, indices, last_keyframe)
            graph = self._match_graph(
                solver_pred,
                [(local_idx + 1, 0) for local_idx in range(len(indices))],
            )
        except (RuntimeError, ValueError) as exc:
            return pred, self._stats(stage="current_to_last", success=False, reason=f"build_failed:{exc}")
        ok, stats = self._gate_graph(graph, stage="current_to_last")
        if not ok:
            return pred, stats

        output = self._solve(solver_pred, graph, fixed_frames=1)
        if output.failed:
            return pred, self._stats(
                stage="current_to_last",
                success=False,
                reason=str(output.debug.get("fallback_reason", "solver_failed")),
                graph=graph,
                output=output,
            )
        refined = _merge_current_subset(pred, solver_pred, output.poses_c2w, indices)
        stats = self._stats(stage="current_to_last", success=True, reason=None, graph=graph, output=output)
        refined = replace(
            refined,
            matching_debug={
                **(pred.matching_debug or {}),
                **graph.metrics(),
                **stats.as_debug(),
            },
        )
        return refined, stats

    def build_adjacent_edge(
        self,
        *,
        source: KeyframeRecord,
        target: KeyframeRecord,
        edge_type: str = "adjacent",
    ) -> tuple[KeyframeCorrespondenceEdge | None, KeyframeGraphBAStats]:
        """Build a persistent source-keyframe to target-keyframe dense edge."""

        if not self.enabled:
            return None, self._stats(stage="edge", success=False, reason="disabled")
        if edge_type == "adjacent" and not self.config.keyframe_graph.adjacent_edges:
            return None, self._stats(stage="edge", success=False, reason="adjacent_edges_disabled")
        if edge_type == "retrieval" and not self.config.keyframe_graph.retrieval_edges:
            return None, self._stats(stage="edge", success=False, reason="retrieval_edges_disabled")
        if edge_type == "loop" and not self.config.keyframe_graph.loop_edges:
            return None, self._stats(stage="edge", success=False, reason="loop_edges_disabled")
        if not (_record_has_dense_fields(source) and _record_has_dense_fields(target)):
            return None, self._stats(stage="edge", success=False, reason="incomplete_keyframe")
        if tuple(source.feature_hw or ()) != tuple(target.feature_hw or ()):
            return None, self._stats(stage="edge", success=False, reason="feature_hw_mismatch")

        try:
            pred = _prediction_from_records([target, source])
            graph = self._match_graph(pred, [(1, 0)])
        except (RuntimeError, ValueError) as exc:
            return None, self._stats(stage="edge", success=False, reason=f"build_failed:{exc}")
        ok, stats = self._gate_graph(graph, stage="edge")
        if not ok:
            return None, stats
        if not graph.factors:
            return None, self._stats(stage="edge", success=False, reason="empty_factors", graph=graph)
        metrics = graph.metrics()
        return (
            KeyframeCorrespondenceEdge(
                src_kf_id=int(source.frame_id),
                tgt_kf_id=int(target.frame_id),
                edge_type=str(edge_type),
                factor=graph.factors[0],
                metrics=metrics,
            ),
            self._stats(stage="edge", success=True, reason=None, graph=graph),
        )

    def optimize_keyframe_graph(
        self,
        *,
        memory: KeyframeMemory,
        graph: KeyframeCorrespondenceGraph,
    ) -> tuple[dict[int, torch.Tensor], KeyframeGraphBAStats]:
        """Run pose-only BA over recent persistent keyframe graph edges."""

        if not self.enabled:
            return {}, self._stats(stage="global", success=False, reason="disabled")
        records = list(memory.recent(self.config.keyframe_graph.window_keyframes))
        if len(records) <= int(self.config.keyframe_graph.fixed_keyframes):
            return {}, self._stats(stage="global", success=False, reason="too_few_keyframes")
        records = _compatible_records(records)
        if len(records) <= int(self.config.keyframe_graph.fixed_keyframes):
            return {}, self._stats(stage="global", success=False, reason="too_few_compatible_keyframes")
        node_ids = {int(record.frame_id) for record in records}
        graph_edges = graph.edges_for_nodes(node_ids)
        if not graph_edges:
            return {}, self._stats(stage="global", success=False, reason="empty_graph_edges")
        id_to_idx = {int(record.frame_id): idx for idx, record in enumerate(records)}
        factors: list[DenseSphereFactor] = []
        edge_pairs: list[tuple[int, int]] = []
        for edge in graph_edges:
            src = id_to_idx.get(int(edge.src_kf_id))
            tgt = id_to_idx.get(int(edge.tgt_kf_id))
            if src is None or tgt is None:
                continue
            factors.append(replace(edge.factor, src=src, tgt=tgt))
            edge_pairs.append((src, tgt))
        if not factors:
            return {}, self._stats(stage="global", success=False, reason="empty_local_edges")
        factor_graph = DenseSphereFactorGraph(
            factors=factors,
            edges=torch.tensor(edge_pairs, dtype=torch.long) if edge_pairs else None,
            metadata={"mode": "keyframe_graph"},
        )
        ok, stats = self._gate_graph(factor_graph, stage="global")
        if not ok:
            return {}, stats

        pred = _prediction_from_records(records)
        fixed = max(1, min(int(self.config.keyframe_graph.fixed_keyframes), int(pred.poses_c2w.shape[0])))
        output = self._solve(pred, factor_graph, fixed_frames=fixed)
        if output.failed:
            return {}, self._stats(
                stage="global",
                success=False,
                reason=str(output.debug.get("fallback_reason", "solver_failed")),
                graph=factor_graph,
                output=output,
            )
        updates = {
            int(record.frame_id): output.poses_c2w[idx].detach().cpu().float()
            for idx, record in enumerate(records)
            if idx >= fixed
        }
        return updates, self._stats(stage="global", success=True, reason=None, graph=factor_graph, output=output)

    def _match_graph(self, pred: PanoVGGTLocalPrediction, edge_pairs: list[tuple[int, int]]) -> DenseSphereFactorGraph:
        assert pred.dense_descriptors is not None
        assert pred.match_confidence is not None
        assert pred.sky_prob is not None
        assert pred.feature_hw is not None
        assert pred.image_hw is not None
        dense = self.config.dense_matching
        matcher = PoseGuidedDenseMatcher(
            search_radius=dense.search_radius,
            topk=dense.topk,
            min_match_confidence=dense.min_match_confidence,
            min_static_confidence=dense.min_static_confidence,
            min_match_score=dense.min_match_score,
            max_factors=max(1, int(self.config.keyframe_graph.max_factors_per_edge) * max(1, len(edge_pairs))),
            max_samples_per_edge=dense.max_samples_per_edge,
            use_wraparound=dense.use_wraparound,
            forward_backward=dense.forward_backward,
            fb_tolerance=dense.fb_tolerance,
            use_depth_consistency=dense.use_depth_consistency,
            depth_consistency_rel=dense.depth_consistency_rel,
            depth_consistency_abs=dense.depth_consistency_abs,
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
            edge_pairs=torch.tensor(edge_pairs, dtype=torch.long, device=pred.depth.device),
        )

    def _gate_graph(self, graph: DenseSphereFactorGraph, *, stage: str) -> tuple[bool, KeyframeGraphBAStats]:
        metrics = graph.metrics()
        valid = int(metrics.get("valid_factors", 0.0))
        ratio = float(metrics.get("valid_factor_ratio", 0.0))
        if valid < int(self.config.keyframe_graph.min_valid_factors):
            return False, self._stats(stage=stage, success=False, reason="too_few_factors", graph=graph)
        if ratio < float(self.config.keyframe_graph.min_valid_factor_ratio):
            return False, self._stats(stage=stage, success=False, reason="low_valid_factor_ratio", graph=graph)
        return True, self._stats(stage=stage, success=True, reason=None, graph=graph)

    def _solve(self, pred: PanoVGGTLocalPrediction, graph: DenseSphereFactorGraph, *, fixed_frames: int):
        assert pred.feature_hw is not None
        depth_low = F.interpolate(
            pred.depth.float(),
            size=tuple(int(v) for v in pred.feature_hw),
            mode="bilinear",
            align_corners=False,
        ).clamp_min(1.0e-6)
        log_inv_depth = depth_low.reciprocal().clamp_min(1.0e-6).log()
        return self.solver(
            pred.poses_c2w.to(device=pred.depth.device, dtype=pred.depth.dtype),
            log_inv_depth.to(device=pred.depth.device, dtype=pred.depth.dtype),
            graph,
            fixed_frames=fixed_frames,
            iters=self.config.dense_ba.iters,
            damping=self.config.dense_ba.lm,
            optimize_pose=True,
            optimize_depth=False,
            pose_prior_weight=self.config.dense_ba.pose_prior_weight,
            depth_prior_weight=self.config.dense_ba.depth_prior_weight,
            max_pose_update_deg=self.config.dense_ba.max_pose_update_deg,
            max_logdepth_update=self.config.dense_ba.max_logdepth_update,
            line_search=self.config.dense_ba.line_search,
            solver_mode="pose_only_factor_graph",
            max_ba_factors=self.config.keyframe_graph.max_factors_per_edge * max(1, graph.num_edges),
            max_depth_variables=0,
            max_solver_sec=self.config.dense_ba.max_solver_sec,
        )

    def _stats(
        self,
        *,
        stage: str,
        success: bool,
        reason: str | None,
        graph: DenseSphereFactorGraph | None = None,
        output=None,
    ) -> KeyframeGraphBAStats:
        metrics = graph.metrics() if graph is not None else {}
        return KeyframeGraphBAStats(
            enabled=self.enabled,
            stage=str(stage),
            success=bool(success),
            used_refined=bool(success and stage in {"current_to_last", "global"}),
            fallback_reason=reason,
            num_edges=int(metrics.get("num_edges", 0.0)),
            num_factors=int(metrics.get("num_factors", 0.0)),
            valid_factors=int(metrics.get("valid_factors", 0.0)),
            valid_factor_ratio=float(metrics.get("valid_factor_ratio", 0.0)),
            mean_residual_deg=0.0 if output is None else float(output.mean_angular_residual_deg),
            pose_update_norm=None if output is None else dict(output.pose_update_norm),
        )


def _missing_prediction_fields(pred: PanoVGGTLocalPrediction) -> str | None:
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


def _record_has_dense_fields(record: KeyframeRecord) -> bool:
    return (
        record.pose_c2w is not None
        and record.depth_low is not None
        and record.dense_descriptors is not None
        and record.match_confidence is not None
        and record.sky_prob is not None
        and record.feature_hw is not None
        and record.image_hw is not None
    )


def _compatible_records(records: list[KeyframeRecord]) -> list[KeyframeRecord]:
    if not records:
        return []
    base = records[-1]
    if not _record_has_dense_fields(base):
        return []
    feature_hw = tuple(int(v) for v in base.feature_hw)
    image_hw = tuple(int(v) for v in base.image_hw)
    desc_dim = int(_as_frame_tensor(base.dense_descriptors).shape[0])
    out = []
    for record in records:
        if not _record_has_dense_fields(record):
            continue
        desc = _as_frame_tensor(record.dense_descriptors)
        if desc is None or int(desc.shape[0]) != desc_dim:
            continue
        if tuple(int(v) for v in record.feature_hw) != feature_hw:
            continue
        if tuple(int(v) for v in record.image_hw) != image_hw:
            continue
        out.append(record)
    return out


def _prediction_from_last_and_current(
    pred: PanoVGGTLocalPrediction,
    indices: tuple[int, ...],
    last_keyframe: KeyframeRecord,
) -> PanoVGGTLocalPrediction:
    assert pred.dense_descriptors is not None
    assert pred.match_confidence is not None
    assert pred.sky_prob is not None
    assert pred.feature_hw is not None
    assert pred.image_hw is not None
    device = pred.depth.device
    dtype = pred.depth.dtype
    depth_hw = tuple(int(v) for v in pred.depth.shape[-2:])
    last_depth = _resize_frame_depth(_as_frame_tensor(last_keyframe.depth_low), depth_hw, device=device, dtype=dtype)
    poses = torch.cat(
        [
            last_keyframe.pose_c2w.to(device=device, dtype=dtype).view(1, 4, 4),
            pred.poses_c2w[list(indices)].to(device=device, dtype=dtype),
        ],
        dim=0,
    )
    depth = torch.cat([last_depth.unsqueeze(0), pred.depth[list(indices)].to(device=device, dtype=dtype)], dim=0)
    confidence = torch.cat([torch.ones_like(last_depth).unsqueeze(0), pred.confidence[list(indices)].to(device=device, dtype=dtype)], dim=0)
    local = _build_local_points(depth)
    points = _local_points_to_world(local, poses)
    dense = torch.cat(
        [
            _as_frame_tensor(last_keyframe.dense_descriptors).to(device=device, dtype=pred.dense_descriptors.dtype).unsqueeze(0),
            pred.dense_descriptors[list(indices)].to(device=device, dtype=pred.dense_descriptors.dtype),
        ],
        dim=0,
    )
    match = torch.cat(
        [
            _as_frame_tensor(last_keyframe.match_confidence).to(device=device, dtype=pred.match_confidence.dtype).unsqueeze(0),
            pred.match_confidence[list(indices)].to(device=device, dtype=pred.match_confidence.dtype),
        ],
        dim=0,
    )
    sky = torch.cat(
        [
            _as_frame_tensor(last_keyframe.sky_prob).to(device=device, dtype=pred.sky_prob.dtype).unsqueeze(0),
            pred.sky_prob[list(indices)].to(device=device, dtype=pred.sky_prob.dtype),
        ],
        dim=0,
    )
    static_confidence = None
    if pred.static_confidence is not None and last_keyframe.static_confidence is not None:
        static_confidence = torch.cat(
            [
                _as_frame_tensor(last_keyframe.static_confidence).to(device=device, dtype=pred.static_confidence.dtype).unsqueeze(0),
                pred.static_confidence[list(indices)].to(device=device, dtype=pred.static_confidence.dtype),
            ],
            dim=0,
        )
    return PanoVGGTLocalPrediction(
        poses_c2w=poses,
        depth=depth,
        confidence=confidence,
        chunk_world_points=points,
        local_points=local,
        dense_descriptors=dense,
        match_confidence=match,
        static_confidence=static_confidence,
        sky_prob=sky,
        feature_hw=pred.feature_hw,
        image_hw=pred.image_hw,
        descriptor_dim=pred.descriptor_dim,
    )


def _prediction_from_records(records: list[KeyframeRecord]) -> PanoVGGTLocalPrediction:
    if not records:
        raise ValueError("records must not be empty.")
    base = records[-1]
    feature_hw = tuple(int(v) for v in base.feature_hw)
    image_hw = tuple(int(v) for v in base.image_hw)
    device = base.pose_c2w.device
    dtype = base.pose_c2w.dtype
    depth_hw = tuple(int(v) for v in _as_frame_tensor(base.depth_low).shape[-2:])
    poses = torch.stack([record.pose_c2w.to(device=device, dtype=dtype) for record in records], dim=0)
    depth = torch.stack(
        [_resize_frame_depth(_as_frame_tensor(record.depth_low), depth_hw, device=device, dtype=dtype) for record in records],
        dim=0,
    )
    confidence = torch.ones_like(depth)
    local = _build_local_points(depth)
    points = _local_points_to_world(local, poses)
    dense = torch.stack(
        [_as_frame_tensor(record.dense_descriptors).to(device=device, dtype=dtype) for record in records],
        dim=0,
    )
    match = torch.stack(
        [_as_frame_tensor(record.match_confidence).to(device=device, dtype=dtype) for record in records],
        dim=0,
    )
    sky = torch.stack(
        [_as_frame_tensor(record.sky_prob).to(device=device, dtype=dtype) for record in records],
        dim=0,
    )
    static_confidence = None
    if all(record.static_confidence is not None for record in records):
        static_confidence = torch.stack(
            [_as_frame_tensor(record.static_confidence).to(device=device, dtype=dtype) for record in records],
            dim=0,
        )
    return PanoVGGTLocalPrediction(
        poses_c2w=poses,
        depth=depth,
        confidence=confidence,
        chunk_world_points=points,
        local_points=local,
        dense_descriptors=dense,
        match_confidence=match,
        static_confidence=static_confidence,
        sky_prob=sky,
        feature_hw=feature_hw,
        image_hw=image_hw,
        descriptor_dim=int(dense.shape[1]),
    )


def _merge_current_subset(
    original: PanoVGGTLocalPrediction,
    solver_pred: PanoVGGTLocalPrediction,
    refined_poses: torch.Tensor,
    indices: tuple[int, ...],
) -> PanoVGGTLocalPrediction:
    poses = original.poses_c2w.clone()
    poses[list(indices)] = refined_poses[1 : 1 + len(indices)].to(poses)
    local = original.local_points
    if local is None:
        local = _build_local_points(original.depth)
    points = original.chunk_world_points.clone()
    selected_local = local[list(indices)].to(device=poses.device, dtype=poses.dtype)
    selected_poses = poses[list(indices)]
    refined_points = _local_points_to_world(selected_local, selected_poses)
    points[list(indices)] = refined_points.to(points)
    _ = solver_pred
    return replace(
        original,
        poses_c2w=poses,
        local_points=local,
        chunk_world_points=points,
    )


def _build_local_points(depth: torch.Tensor) -> torch.Tensor:
    n, _, height, width = depth.shape
    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    bearing = erp_pixel_to_bearing(grid, height, width).to(device=depth.device, dtype=depth.dtype)
    return bearing.unsqueeze(0) * depth[:, 0].unsqueeze(-1)


def _local_points_to_world(local_points: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    rot = poses_c2w[:, :3, :3]
    trans = poses_c2w[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", rot, local_points) + trans.view(int(local_points.shape[0]), 1, 1, 3)


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
        raise ValueError("Keyframe depth must have shape 1xHxW or 1x1xHxW.")
    depth = value.to(device=device, dtype=dtype).clamp_min(1.0e-6)
    if tuple(depth.shape[-2:]) == tuple(size):
        return depth
    return F.interpolate(
        depth.unsqueeze(0),
        size=tuple(int(v) for v in size),
        mode="bilinear",
        align_corners=False,
    )[0].clamp_min(1.0e-6)
