import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from frontend.pano_vggt.alignment import SimilarityTransform
from frontend.pano_vggt.dense_ba_refiner import DenseBARefinerStats, PanoVGGTDenseBARefiner
from frontend.pano_vggt.engine import PanoVGGTInferenceEngine
from frontend.pano_vggt.factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from frontend.pano_vggt.keyframe_graph_refiner import PanoVGGTKeyframeGraphRefiner
from frontend.pano_vggt.keyframe_memory import (
    KeyframeCorrespondenceEdge,
    KeyframeCorrespondenceGraph,
    KeyframeMemory,
    KeyframeRecord,
)
from frontend.pano_vggt.m3_config import parse_m3_sphere_config
from frontend.pano_vggt.spherical_correspondence import generate_gt_spherical_correspondences
from frontend.pano_vggt.spherical_dense_ba import SphericalTangentDenseBA
from frontend.pano_vggt.tracker import PanoVGGTLongTracker
from frontend.pano_vggt.types import PanoVGGTLocalPrediction
from system.pano_droid_gs_slam import SlamRuntimeLogger, _summarize_dense_ba_stats


def _poses(tx: float = 0.2) -> torch.Tensor:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = float(tx)
    return poses


def _depth(feature_hw: tuple[int, int], value: float = 2.0) -> torch.Tensor:
    return torch.full((2, 1, feature_hw[0], feature_hw[1]), float(value))


def _graph_from_truth(
    poses: torch.Tensor,
    depth: torch.Tensor,
    *,
    feature_hw: tuple[int, int] = (5, 8),
    samples: int = 16,
) -> DenseSphereFactorGraph:
    corr = generate_gt_spherical_correspondences(
        depth,
        poses,
        [(0, 1)],
        feature_hw,
        feature_hw,
        samples_per_edge=samples,
        depth_consistency_abs=10.0,
        depth_consistency_rel=10.0,
        min_baseline_deg=0.0,
        max_baseline_deg=180.0,
    )
    valid = corr.valid_mask[0]
    assert bool(valid.any())
    factor = DenseSphereFactor(
        src=0,
        tgt=1,
        src_uv=corr.src_uv[0],
        tgt_uv=corr.tgt_uv[0],
        src_bearing=corr.src_bearing[0],
        tgt_bearing=corr.tgt_bearing[0],
        weight=torch.ones_like(valid, dtype=torch.float32),
        match_score=torch.ones_like(valid, dtype=torch.float32),
        valid_mask=valid,
        metadata={"depth_consistency_mask": corr.depth_consistency[0]},
    )
    return DenseSphereFactorGraph(factors=[factor], edges=torch.tensor([[0, 1]], dtype=torch.long))


def _prediction(
    *,
    poses: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    dense: bool = True,
    sky: bool = True,
    feature_hw: tuple[int, int] = (5, 8),
) -> PanoVGGTLocalPrediction:
    poses_t = _poses(0.05) if poses is None else poses
    depth_t = _depth(feature_hw, 2.0) if depth is None else depth
    local_points = torch.zeros(2, feature_hw[0], feature_hw[1], 3)
    pred = PanoVGGTLocalPrediction(
        poses_c2w=poses_t.clone(),
        depth=depth_t.clone(),
        confidence=torch.ones_like(depth_t),
        chunk_world_points=local_points.clone(),
        local_points=local_points.clone(),
        feature_hw=feature_hw if dense else None,
        image_hw=feature_hw if dense else None,
    )
    if dense:
        pred.dense_descriptors = torch.ones(2, 4, feature_hw[0], feature_hw[1])
        pred.match_confidence = torch.ones(2, 1, feature_hw[0], feature_hw[1])
    if sky:
        pred.sky_prob = torch.zeros(2, 1, feature_hw[0], feature_hw[1])
    return pred


def _descriptor_grid(
    feature_hw: tuple[int, int],
    *,
    frames: int,
    channels: int = 4,
) -> torch.Tensor:
    h, w = feature_hw
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xx = xx.float() / max(1, w - 1)
    yy = yy.float() / max(1, h - 1)
    base = torch.stack([xx, yy, 1.0 - xx, 1.0 - yy], dim=0)
    if channels > 4:
        base = torch.cat([base, torch.ones(channels - 4, h, w)], dim=0)
    base = torch.nn.functional.normalize(base[:channels], dim=0)
    return base.unsqueeze(0).repeat(int(frames), 1, 1, 1)


def _prediction_n(
    n: int,
    *,
    feature_hw: tuple[int, int] = (5, 8),
    tx_step: float = 0.05,
) -> PanoVGGTLocalPrediction:
    poses = torch.eye(4).repeat(int(n), 1, 1)
    for idx in range(int(n)):
        poses[idx, 0, 3] = float(idx) * float(tx_step)
    depth = torch.full((int(n), 1, feature_hw[0], feature_hw[1]), 2.0)
    local_points = torch.zeros(int(n), feature_hw[0], feature_hw[1], 3)
    return PanoVGGTLocalPrediction(
        poses_c2w=poses,
        depth=depth,
        confidence=torch.ones_like(depth),
        chunk_world_points=local_points.clone(),
        local_points=local_points.clone(),
        dense_descriptors=_descriptor_grid(feature_hw, frames=int(n)),
        match_confidence=torch.ones(int(n), 1, feature_hw[0], feature_hw[1]),
        sky_prob=torch.zeros(int(n), 1, feature_hw[0], feature_hw[1]),
        feature_hw=feature_hw,
        image_hw=feature_hw,
    )


def _record_from_prediction(pred: PanoVGGTLocalPrediction, idx: int, frame_id: int) -> KeyframeRecord:
    assert pred.dense_descriptors is not None
    assert pred.match_confidence is not None
    assert pred.sky_prob is not None
    return KeyframeRecord(
        frame_id=int(frame_id),
        pose_c2w=pred.poses_c2w[int(idx)].detach().cpu().float(),
        depth_low=pred.depth[int(idx)].detach().cpu().float(),
        dense_descriptors=pred.dense_descriptors[int(idx)].detach().cpu().float(),
        match_confidence=pred.match_confidence[int(idx)].detach().cpu().float(),
        sky_prob=pred.sky_prob[int(idx)].detach().cpu().float(),
        feature_hw=tuple(int(v) for v in pred.feature_hw),
        image_hw=tuple(int(v) for v in pred.image_hw),
        global_points=pred.chunk_world_points[int(idx)].detach().cpu().float(),
        frozen=True,
    )


def _m3_config(
    *,
    shadow: bool = False,
    min_num_factors: int = 1,
    residual_worse: bool = False,
    mode: str = "local_chunk",
    history_keyframes: int | None = None,
) -> dict:
    dense_ba = {
        "enabled": True,
        "shadow_mode": shadow,
        "mode": mode,
        "iters": 3,
        "min_num_factors": min_num_factors,
        "min_valid_factor_ratio": 0.1,
        "huber_delta_deg": 10.0,
        "pose_prior_weight": 0.0,
        "depth_prior_weight": 0.0,
        "max_pose_update_deg": 20.0,
        "max_logdepth_update": 0.5,
        "fallback_if_residual_worse": residual_worse,
        "residual_worse_tolerance": 1.0,
        "factor_chunk_size": 64,
    }
    if history_keyframes is not None:
        dense_ba["history_keyframes"] = history_keyframes
    return {
        "PanoVGGT": {
            "M3Sphere": {"enabled": True},
            "DenseMatching": {
                "enabled": False,
                "max_samples_per_edge": 16,
                "search_radius": 1,
                "use_depth_consistency": False,
            },
            "DenseBA": dense_ba,
        }
    }


def _keyframe_graph_config() -> dict:
    cfg = _m3_config(shadow=False, min_num_factors=1)
    cfg["PanoVGGT"]["DenseMatching"].update(
        {
            "enabled": True,
            "min_match_confidence": 0.0,
            "min_static_confidence": 0.0,
            "min_match_score": 0.0,
            "forward_backward": False,
            "use_depth_consistency": False,
            "max_samples_per_edge": 32,
        }
    )
    cfg["PanoVGGT"]["DenseBA"].update(
        {
            "iters": 3,
            "pose_prior_weight": 0.0,
            "depth_prior_weight": 0.0,
            "factor_chunk_size": 64,
            "max_solver_sec": 2.0,
        }
    )
    cfg["PanoVGGT"]["KeyframeGraph"] = {
        "enabled": True,
        "current_to_last_ba": True,
        "adjacent_edges": True,
        "retrieval_edges": False,
        "loop_edges": False,
        "adjacent_history": 1,
        "window_keyframes": 4,
        "optimize_every_keyframes": 1,
        "fixed_keyframes": 1,
        "min_valid_factors": 1,
        "min_valid_factor_ratio": 0.0,
        "max_factors_per_edge": 64,
        "publish_pose_updates": True,
    }
    return cfg


def _minimal_factor_graph(src: int = 1, tgt: int = 0, factors: int = 4) -> DenseSphereFactorGraph:
    valid = torch.ones(int(factors), dtype=torch.bool)
    uv = torch.stack(
        [
            torch.linspace(0.5, 1.5, steps=int(factors)),
            torch.linspace(0.5, 1.5, steps=int(factors)),
        ],
        dim=-1,
    )
    bearing = torch.nn.functional.normalize(torch.ones(int(factors), 3), dim=-1)
    factor = DenseSphereFactor(
        src=int(src),
        tgt=int(tgt),
        src_uv=uv,
        tgt_uv=uv.clone(),
        src_bearing=bearing,
        tgt_bearing=bearing.clone(),
        weight=torch.ones(int(factors)),
        match_score=torch.ones(int(factors)),
        valid_mask=valid,
    )
    return DenseSphereFactorGraph(factors=[factor], edges=torch.tensor([[int(src), int(tgt)]], dtype=torch.long))


def test_spherical_dense_ba_two_frame_pose_perturbation_residual_decreases():
    feature_hw = (5, 8)
    true_poses = _poses(0.2)
    true_depth = _depth(feature_hw, 2.0)
    graph = _graph_from_truth(true_poses, true_depth, feature_hw=feature_hw)
    init_poses = _poses(0.05)
    log_inv_depth = torch.log(true_depth.reciprocal())

    solver = SphericalTangentDenseBA(factor_chunk_size=64, huber_delta_deg=10.0)
    before = solver(init_poses, log_inv_depth, graph, fixed_frames=1, iters=0, optimize_depth=False)
    after = solver(init_poses, log_inv_depth, graph, fixed_frames=1, iters=3, optimize_depth=False)

    assert not after.failed
    assert after.mean_angular_residual_deg < before.mean_angular_residual_deg * 0.1
    assert torch.allclose(after.poses_c2w[1, :3, 3], true_poses[1, :3, 3], atol=1e-3)


def test_spherical_dense_ba_fixed_first_frame_unchanged():
    feature_hw = (5, 8)
    graph = _graph_from_truth(_poses(0.2), _depth(feature_hw, 2.0), feature_hw=feature_hw)
    init_poses = _poses(0.05)
    log_inv_depth = torch.log(_depth(feature_hw, 2.0).reciprocal())

    out = SphericalTangentDenseBA(factor_chunk_size=64, huber_delta_deg=10.0)(
        init_poses,
        log_inv_depth,
        graph,
        fixed_frames=1,
        iters=3,
        optimize_depth=False,
    )

    assert not out.failed
    assert torch.allclose(out.poses_c2w[0], init_poses[0], atol=1e-7)


def test_spherical_dense_ba_depth_only_perturbation_residual_decreases():
    feature_hw = (5, 8)
    true_poses = _poses(0.2)
    true_depth = _depth(feature_hw, 2.0)
    graph = _graph_from_truth(true_poses, true_depth, feature_hw=feature_hw)
    wrong_log_depth = torch.log(_depth(feature_hw, 1.5).reciprocal())

    solver = SphericalTangentDenseBA(factor_chunk_size=64, huber_delta_deg=10.0)
    before = solver(true_poses, wrong_log_depth, graph, fixed_frames=2, iters=0, optimize_pose=False)
    after = solver(true_poses, wrong_log_depth, graph, fixed_frames=2, iters=5, optimize_pose=False)

    assert not after.failed
    assert after.mean_angular_residual_deg < before.mean_angular_residual_deg * 0.1
    factor = graph.factors[0]
    uv = factor.src_uv[factor.valid_mask]
    src_x = uv[:, 0].floor().long().clamp(0, feature_hw[1] - 1)
    src_y = uv[:, 1].floor().long().clamp(0, feature_hw[0] - 1)
    refined_mean = after.inverse_depth[0, 0, src_y, src_x].mean()
    initial_mean = torch.tensor(1.0 / 1.5)
    assert (refined_mean - 0.5).abs() < (initial_mean - 0.5).abs()
    assert refined_mean < 0.55


def test_spherical_dense_ba_seam_case_stable():
    feature_hw = (4, 8)
    poses = _poses(0.2)
    depth = _depth(feature_hw, 2.0)
    graph = _graph_from_truth(poses, depth, feature_hw=feature_hw, samples=8)
    factor = graph.factors[0]
    factor.src_uv = factor.src_uv.clone()
    factor.tgt_uv = factor.tgt_uv.clone()
    factor.src_uv[:, 0] = torch.remainder(factor.src_uv[:, 0] + float(feature_hw[1]) - 0.2, float(feature_hw[1]))
    factor.tgt_uv[:, 0] = torch.remainder(factor.tgt_uv[:, 0] + float(feature_hw[1]) - 0.1, float(feature_hw[1]))

    out = SphericalTangentDenseBA(factor_chunk_size=64, huber_delta_deg=10.0)(
        _poses(0.05),
        torch.log(depth.reciprocal()),
        graph,
        fixed_frames=1,
        iters=1,
        optimize_depth=False,
    )

    assert torch.isfinite(out.residual_angular).all()
    assert torch.isfinite(out.poses_c2w).all()


def test_spherical_dense_ba_empty_or_invalid_factors_safe_fallback():
    invalid = DenseSphereFactor(
        src=0,
        tgt=1,
        src_uv=torch.zeros(3, 2),
        tgt_uv=torch.zeros(3, 2),
        src_bearing=torch.zeros(3, 3),
        tgt_bearing=torch.zeros(3, 3),
        weight=torch.zeros(3),
        match_score=torch.zeros(3),
        valid_mask=torch.zeros(3, dtype=torch.bool),
    )

    out = SphericalTangentDenseBA()(
        _poses(0.1),
        torch.log(_depth((3, 4), 2.0).reciprocal()),
        DenseSphereFactorGraph([invalid]),
        fixed_frames=1,
    )

    assert out.failed
    assert out.debug["fallback_reason"] == "empty_or_invalid_factors"


def test_spherical_dense_ba_source_uses_s2_tangent_residual_not_pixel_delta():
    source = inspect.getsource(SphericalTangentDenseBA._residual_for_state)
    assert "spherical_tangent_residual" in source
    assert "seam_aware_delta" not in source


def test_dense_ba_refiner_missing_descriptors_or_sky_fallback():
    refiner = PanoVGGTDenseBARefiner(parse_m3_sphere_config(_m3_config()))
    pred = _prediction(dense=False)

    refined, stats = refiner.refine(pred, (0, 1))

    assert refined is pred
    assert not stats.success
    assert stats.fallback_reason == "missing_dense_descriptors"

    pred_missing_sky = _prediction(sky=False)
    refined, stats = refiner.refine(pred_missing_sky, (0, 1))
    assert refined is pred_missing_sky
    assert stats.fallback_reason == "missing_sky_prob"


def test_dense_ba_refiner_synthetic_prediction_returns_refined_prediction_and_finite_points():
    true_poses = _poses(0.2)
    true_depth = _depth((5, 8), 2.0)
    graph = _graph_from_truth(true_poses, true_depth)
    pred = _prediction(poses=_poses(0.05), depth=true_depth)
    refiner = PanoVGGTDenseBARefiner(parse_m3_sphere_config(_m3_config()))

    refined, stats = refiner.refine(pred, (0, 1), factor_graph=graph)

    assert stats.success
    assert stats.used_refined
    assert refined is not pred
    assert refined.ba_residual_angular is not None
    assert refined.ba_valid_ratio is not None
    assert torch.isfinite(refined.chunk_world_points).all()
    assert torch.allclose(refined.poses_c2w[1, :3, 3], true_poses[1, :3, 3], atol=1e-3)
    assert torch.allclose(refined.depth, pred.depth)
    assert stats.solver_mode == "pose_only_factor_graph"
    assert stats.num_depth_variables == 0
    assert stats.used_factors > 0
    assert stats.pose_solve_sec >= 0.0
    assert stats.depth_update_norm.get("max", 0.0) == 0.0


def test_pose_only_dense_ba_does_not_build_depth_variables():
    feature_hw = (5, 8)
    true_poses = _poses(0.2)
    graph = _graph_from_truth(true_poses, _depth(feature_hw, 2.0), feature_hw=feature_hw)
    init_poses = _poses(0.05)
    log_inv_depth = torch.log(_depth(feature_hw, 2.0).reciprocal())

    out = SphericalTangentDenseBA(factor_chunk_size=64, huber_delta_deg=10.0)(
        init_poses,
        log_inv_depth,
        graph,
        fixed_frames=1,
        iters=3,
        optimize_pose=True,
        optimize_depth=True,
        solver_mode="pose_only_factor_graph",
        max_ba_factors=8,
    )

    assert not out.failed
    assert out.debug["solver_mode"] == "pose_only_factor_graph"
    assert out.debug["num_depth_variables"] == 0
    assert out.debug["used_factors"] == 8
    assert torch.allclose(out.log_inv_depth, log_inv_depth)
    assert out.depth_update_norm.get("max", 0.0) == 0.0


def test_keyframe_correspondence_graph_keeps_sparse_adjacent_edges():
    factor = _minimal_factor_graph().factors[0]
    graph = KeyframeCorrespondenceGraph(max_edges=4)
    graph.add_edge(
        KeyframeCorrespondenceEdge(
            src_kf_id=2,
            tgt_kf_id=1,
            edge_type="adjacent",
            factor=factor,
            metrics={"valid_factors": 4.0, "valid_factor_ratio": 1.0, "mean_weight": 1.0},
        )
    )
    graph.add_edge(
        KeyframeCorrespondenceEdge(
            src_kf_id=2,
            tgt_kf_id=1,
            edge_type="adjacent",
            factor=factor,
            metrics={"valid_factors": 8.0, "valid_factor_ratio": 1.0, "mean_weight": 1.0},
        )
    )

    assert len(graph) == 1
    assert len(graph.edges_for_nodes({1, 2})) == 1
    assert len(graph.edges_for_nodes({0, 2})) == 0
    metrics = graph.metrics()
    assert metrics["keyframe_graph_adjacent_edges"] == 1.0
    assert metrics["keyframe_graph_retrieval_edges"] == 0.0
    assert metrics["keyframe_graph_loop_edges"] == 0.0


def test_current_to_last_refiner_updates_only_selected_new_frames():
    refiner = PanoVGGTKeyframeGraphRefiner(parse_m3_sphere_config(_keyframe_graph_config()))
    pred = _prediction_n(3)
    last = _record_from_prediction(pred, 0, frame_id=99)
    graph = _minimal_factor_graph(src=1, tgt=0, factors=4)

    refiner._match_graph = lambda solver_pred, edge_pairs: graph

    def fake_solve(solver_pred, graph_arg, *, fixed_frames):
        _ = graph_arg
        assert fixed_frames == 1
        poses = solver_pred.poses_c2w.clone()
        poses[1, 0, 3] += 0.4
        return SimpleNamespace(
            failed=False,
            poses_c2w=poses,
            mean_angular_residual_deg=0.1,
            pose_update_norm={"mean": 0.4, "max": 0.4, "rot_max_deg": 0.0},
        )

    refiner._solve = fake_solve

    refined, stats = refiner.refine_current_to_last(
        pred,
        (10, 11, 12),
        new_local_indices=(2,),
        last_keyframe=last,
    )

    assert stats.success
    assert stats.valid_factors == 4
    assert torch.allclose(refined.poses_c2w[0], pred.poses_c2w[0])
    assert torch.allclose(refined.poses_c2w[1], pred.poses_c2w[1])
    assert torch.allclose(refined.poses_c2w[2, :3, 3], pred.poses_c2w[2, :3, 3] + torch.tensor([0.4, 0.0, 0.0]))


def test_keyframe_graph_refiner_builds_adjacent_edge_from_records():
    refiner = PanoVGGTKeyframeGraphRefiner(parse_m3_sphere_config(_keyframe_graph_config()))
    pred = _prediction_n(2, tx_step=0.0)
    previous = _record_from_prediction(pred, 0, frame_id=0)
    current = _record_from_prediction(pred, 1, frame_id=1)

    edge, stats = refiner.build_adjacent_edge(source=current, target=previous)

    assert stats.success
    assert edge is not None
    assert edge.src_kf_id == 1
    assert edge.tgt_kf_id == 0
    assert edge.edge_type == "adjacent"
    assert edge.metrics["valid_factors"] >= 1.0


def test_keyframe_graph_ba_updates_only_non_fixed_keyframe_pose():
    cfg = _keyframe_graph_config()
    refiner = PanoVGGTKeyframeGraphRefiner(parse_m3_sphere_config(cfg))
    true_poses = _poses(0.2)
    init_poses = _poses(0.05)
    depth = _depth((5, 8), 2.0)
    truth_graph = _graph_from_truth(true_poses, depth, samples=24)
    pred = _prediction(poses=init_poses, depth=depth)
    memory = KeyframeMemory(max_keyframes=4)
    record0 = _record_from_prediction(pred, 0, frame_id=0)
    record1 = _record_from_prediction(pred, 1, frame_id=1)
    record0.pose_c2w = true_poses[0].clone()
    memory.add(record0)
    memory.add(record1)
    graph = KeyframeCorrespondenceGraph(max_edges=4)
    graph.add_edge(
        KeyframeCorrespondenceEdge(
            src_kf_id=0,
            tgt_kf_id=1,
            edge_type="adjacent",
            factor=truth_graph.factors[0],
            metrics=truth_graph.metrics(),
        )
    )

    updates, stats = refiner.optimize_keyframe_graph(memory=memory, graph=graph)

    assert stats.success
    assert 0 not in updates
    assert 1 in updates
    assert abs(float(updates[1][0, 3]) - 0.2) < abs(float(init_poses[1, 0, 3]) - 0.2)


def test_keyframe_graph_disabled_keeps_frontend_output_api_and_empty_graph():
    pred0 = _prediction()
    tracker, _ = _tracker_with_fake_alignment(pred0, _m3_config(shadow=False))

    outputs = _feed_two_frames(tracker)

    assert outputs
    assert len(tracker.keyframe_correspondence_graph) == 0
    assert tracker.pop_keyframe_graph_pose_updates() == {}
    assert set(FrontendOutput.__dataclass_fields__) == {
        "frame_id",
        "timestamp",
        "pose_c2w",
        "relative_pose",
        "pose_confidence",
        "inverse_depth",
        "depth_confidence",
        "spherical_flow",
        "keyframe_score",
        "is_keyframe",
        "ba_residual",
        "tracking_status",
        "world_points",
        "world_points_confidence",
        "valid_world_points_mask",
    }


def test_dense_ba_refiner_history_window_adds_keyframe_anchor_factors():
    feature_hw = (4, 8)
    pred = _prediction(feature_hw=feature_hw)
    memory = KeyframeMemory(max_keyframes=4)
    memory.add(
        KeyframeRecord(
            frame_id=-1,
            pose_c2w=torch.eye(4),
            depth_low=torch.full((1, feature_hw[0], feature_hw[1]), 2.0),
            dense_descriptors=torch.ones(4, feature_hw[0], feature_hw[1]),
            match_confidence=torch.ones(1, feature_hw[0], feature_hw[1]),
            sky_prob=torch.zeros(1, feature_hw[0], feature_hw[1]),
            feature_hw=feature_hw,
            image_hw=feature_hw,
            frozen=True,
        )
    )
    cfg = _m3_config(mode="history_window", history_keyframes=1, min_num_factors=1)
    cfg["PanoVGGT"]["DenseBA"]["optimize_depth"] = False
    refiner = PanoVGGTDenseBARefiner(parse_m3_sphere_config(cfg))

    refined, stats = refiner.refine(pred, (0, 1), keyframe_memory=memory)

    assert stats.success
    assert refined.poses_c2w.shape == pred.poses_c2w.shape
    assert stats.history_keyframes == 1
    assert stats.history_factors > 0
    assert refiner.last_factor_graph is not None
    assert any(int(factor.tgt) == 0 and int(factor.src) >= 1 for factor in refiner.last_factor_graph.factors)


def test_dense_ba_refiner_residual_worse_threshold_triggers_fallback():
    true_poses = _poses(0.2)
    true_depth = _depth((5, 8), 2.0)
    graph = _graph_from_truth(true_poses, true_depth)
    pred = _prediction(poses=true_poses, depth=true_depth)
    cfg = _m3_config(residual_worse=True)
    cfg["PanoVGGT"]["DenseBA"]["iters"] = 1
    cfg["PanoVGGT"]["DenseBA"]["residual_worse_tolerance"] = 0.0
    refiner = PanoVGGTDenseBARefiner(parse_m3_sphere_config(cfg))

    refined, stats = refiner.refine(pred, (0, 1), factor_graph=graph)

    assert refined is pred
    assert not stats.success
    assert stats.fallback_reason == "residual_worse"


class _StaticEngine(PanoVGGTInferenceEngine):
    def __init__(self, pred: PanoVGGTLocalPrediction) -> None:
        self.pred = pred
        self.last_dense_factor_graph = None

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        _ = images
        return self.pred

    def load_checkpoint(self, path: str) -> None:
        _ = path


class _RecordingAnchorEngine(PanoVGGTInferenceEngine):
    def __init__(
        self,
        *,
        feature_hw: tuple[int, int] = (2, 4),
        current_conf_by_call: tuple[float, ...] = (1.0, 1.0),
        translation_by_call: tuple[float, ...] = (0.0, 0.05),
        sky_high_cell: tuple[int, int] | None = None,
    ) -> None:
        self.feature_hw = feature_hw
        self.current_conf_by_call = current_conf_by_call
        self.translation_by_call = translation_by_call
        self.sky_high_cell = sky_high_cell
        self.batch_sizes: list[int] = []
        self.calls = 0
        self.last_dense_factor_graph = None

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        n = int(images.shape[0])
        self.batch_sizes.append(n)
        call_idx = self.calls
        self.calls += 1
        h, w = self.feature_hw
        poses = torch.eye(4).view(1, 4, 4).repeat(n, 1, 1)
        poses[:, 0, 3] = float(self.translation_by_call[min(call_idx, len(self.translation_by_call) - 1)])
        depth = torch.full((n, 1, h, w), 2.0)
        confidence = torch.ones_like(depth)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        points = torch.stack([xx.float(), yy.float(), torch.ones(h, w)], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
        descriptors = torch.ones(n, 4, h, w)
        descriptors = torch.nn.functional.normalize(descriptors, dim=1)
        match_conf = torch.ones(n, 1, h, w)
        if call_idx > 0:
            current_conf = float(self.current_conf_by_call[min(call_idx, len(self.current_conf_by_call) - 1)])
            start = 1 if n > 1 else 0
            match_conf[start:] = current_conf
        sky_prob = torch.zeros(n, 1, h, w)
        if self.sky_high_cell is not None and call_idx > 0:
            row, col = self.sky_high_cell
            start = 1 if n > 1 else 0
            sky_prob[start:, :, row, col] = 0.95
        return PanoVGGTLocalPrediction(
            poses_c2w=poses,
            depth=depth,
            confidence=confidence,
            chunk_world_points=points,
            local_points=points,
            dense_descriptors=descriptors,
            match_confidence=match_conf,
            sky_prob=sky_prob,
            feature_hw=self.feature_hw,
            image_hw=self.feature_hw,
        )

    def load_checkpoint(self, path: str) -> None:
        _ = path


class _FakeRefiner:
    def __init__(self, refined: PanoVGGTLocalPrediction, stats: DenseBARefinerStats) -> None:
        self.refined = refined
        self.stats = stats
        self.calls = 0
        self.memory_lengths: list[int] = []

    def refine(self, pred, frame_ids, *, factor_graph=None, keyframe_memory=None, **kwargs):
        _ = pred, frame_ids, factor_graph, keyframe_memory, kwargs
        self.calls += 1
        self.memory_lengths.append(0 if keyframe_memory is None else len(keyframe_memory))
        return self.refined, self.stats


def _tracker_with_fake_alignment(pred: PanoVGGTLocalPrediction, cfg: dict) -> tuple[PanoVGGTLongTracker, list[PanoVGGTLocalPrediction]]:
    tracker = PanoVGGTLongTracker(
        engine=_StaticEngine(pred),
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=2,
        overlap=0,
        emit_delay=0,
        min_overlap_points=1,
        require_aligned_world_points=False,
        emit_unaligned=True,
    )
    captured = []

    def fake_align(aligned_pred, frame_ids, **kwargs):
        _ = kwargs
        _ = frame_ids
        captured.append(aligned_pred)
        return SimilarityTransform.identity(device=aligned_pred.depth.device, dtype=aligned_pred.depth.dtype)

    tracker._align_chunk = fake_align
    return tracker, captured


def _feed_two_frames(tracker: PanoVGGTLongTracker):
    for idx in range(2):
        tracker.track(PanoFrame(image=torch.zeros(3, 5, 8), timestamp=float(idx), frame_id=idx))
    return tracker.pop_ready_outputs()


def _anchor_cfg(anchor_overrides: dict | None = None) -> dict:
    cfg = {
        "PanoVGGT": {
            "M3Sphere": {"enabled": True},
            "KeyframeAnchor": {
                "enabled": True,
                "prepend_previous_keyframe": True,
                "cell_pair_conf_threshold": 0.25,
                "frame_mean_pair_conf_threshold": 0.30,
                "frame_low_pair_conf_ratio": 0.45,
                "translation_threshold": 0.75,
                "translation_depth_ratio_threshold": 0.08,
                "sky_threshold": 0.5,
            },
            "DenseMatching": {
                "enabled": True,
                "search_radius": 1,
                "min_match_confidence": 0.0,
                "min_static_confidence": 0.0,
                "min_match_score": 0.0,
                "forward_backward": False,
                "use_depth_consistency": False,
            },
            "DenseBA": {"enabled": False},
        }
    }
    if anchor_overrides:
        cfg["PanoVGGT"]["KeyframeAnchor"].update(anchor_overrides)
    return cfg


def _tracker_for_anchor_tests(
    engine: _RecordingAnchorEngine,
    *,
    novel: bool = True,
    anchor_overrides: dict | None = None,
    novel_insert_options: dict | None = None,
) -> PanoVGGTLongTracker:
    cfg = _anchor_cfg(anchor_overrides)
    novel_insert_options = novel_insert_options or {}
    tracker = PanoVGGTLongTracker(
        engine=engine,
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=2,
        overlap=0,
        emit_delay=0,
        min_overlap_points=1,
        require_aligned_world_points=False,
        emit_unaligned=True,
        novel_insertion_enabled=novel,
        **novel_insert_options,
    )

    def fake_align(aligned_pred, frame_ids, **kwargs):
        _ = kwargs
        _ = frame_ids
        return SimilarityTransform.identity(device=aligned_pred.depth.device, dtype=aligned_pred.depth.dtype)

    tracker._align_chunk = fake_align
    return tracker


def test_tracker_disabled_path_status_unchanged():
    pred = _prediction(dense=False)
    cfg = {"PanoVGGT": {"M3Sphere": {"enabled": False}, "DenseBA": {"enabled": False}}}
    tracker, captured = _tracker_with_fake_alignment(pred, cfg)

    outputs = _feed_two_frames(tracker)

    assert len(captured) == 1
    assert captured[0] is pred
    assert outputs
    assert all(out.tracking_status == "tracked_panovggt_long" for out in outputs)
    assert tracker.last_dense_ba_stats is not None
    assert tracker.last_dense_ba_stats.enabled is False
    assert tracker.pop_keyframe_decisions() == []


def test_replace_fuse_keyframes_follow_new_block_last_policy():
    cfg = _anchor_cfg(
        {
            "translation_threshold": 0.0,
            "translation_depth_ratio_threshold": 0.0,
            "min_keyframe_interval": 0,
            "max_keyframe_interval": 0,
        }
    )
    tracker = PanoVGGTLongTracker(
        engine=_RecordingAnchorEngine(
            current_conf_by_call=(1.0, 1.0, 1.0, 1.0),
            translation_by_call=(0.0, 0.1, 0.2, 0.3),
        ),
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=4,
        overlap=2,
        emit_delay=0,
        min_overlap_points=1,
        require_aligned_world_points=False,
        emit_unaligned=True,
        novel_insertion_enabled=True,
        novel_insertion_strategy="pfgs360_replace_fuse",
        novel_insert_keyframe_policy="new_block_last",
        novel_insert_keyframe_block_size=4,
    )

    def fake_align(aligned_pred, frame_ids, **kwargs):
        _ = aligned_pred, frame_ids, kwargs
        return SimilarityTransform.identity(device=torch.device("cpu"), dtype=torch.float32)

    tracker._align_chunk = fake_align
    outputs = []
    for idx in range(8):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
        outputs.extend(tracker.pop_ready_outputs())

    keyframes = [int(out.frame_id) for out in outputs if out.is_keyframe]
    decisions = tracker.pop_keyframe_decisions()
    accepted = [int(item["frame_id"]) for item in decisions if item.get("accepted")]

    assert keyframes == [3, 7]
    assert accepted == [3, 7]
    assert all(
        item.get("chunk_keyframe_policy") == "new_block_last"
        for item in decisions
        if int(item["frame_id"]) in {0, 1, 2, 3, 4, 5, 6, 7}
    )


def test_keyframe_anchor_uses_sidepath_without_prepending_main_forward():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 1.0))
    tracker = _tracker_for_anchor_tests(engine)

    _feed_two_frames(tracker)
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    tracker.pop_ready_outputs()

    assert engine.batch_sizes == [2, 2, 3]


def test_joint_inference_uses_recent_history_keyframes_once():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 1.0))
    cfg = _anchor_cfg(
        {
            "prepend_previous_keyframe": False,
            "min_keyframe_interval": 4,
            "max_keyframe_interval": 8,
        }
    )
    cfg["PanoVGGT"]["JointInference"] = {
        "enabled": True,
        "history_policy": "recent",
        "max_history_frames": 3,
    }
    tracker = PanoVGGTLongTracker(
        engine=engine,
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=2,
        overlap=0,
        emit_delay=0,
        min_overlap_points=1,
        require_aligned_world_points=False,
        emit_unaligned=True,
    )

    def fake_align(aligned_pred, frame_ids, **kwargs):
        assert kwargs["history_records"]
        return SimilarityTransform.identity(device=aligned_pred.depth.device, dtype=aligned_pred.depth.dtype)

    tracker._align_chunk = fake_align
    for frame_id in (0, 4, 8, 12):
        tracker.keyframe_memory.add(
            KeyframeRecord(
                frame_id=frame_id,
                pose_c2w=torch.eye(4),
                depth_low=torch.ones(1, 2, 4),
                dense_descriptors=torch.ones(4, 2, 4),
                match_confidence=torch.ones(1, 2, 4),
                sky_prob=torch.zeros(1, 2, 4),
                feature_hw=(2, 4),
                image_hw=(2, 4),
                image=torch.zeros(3, 2, 4),
                global_points=torch.zeros(2, 4, 3),
            )
        )

    for idx in range(20, 22):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    outputs = tracker.pop_ready_outputs()

    assert engine.batch_sizes == [5]
    assert tracker.current_recent_history_ids == (4, 8, 12)
    assert len(outputs) == 2


def test_alignment_can_disable_common_history_and_use_only_overlap_points():
    cfg = _anchor_cfg()
    cfg["PanoVGGT"]["Alignment"] = {
        "use_common_history": False,
        "history_point_budget_ratio": 0.0,
    }
    tracker = PanoVGGTLongTracker(
        engine=_RecordingAnchorEngine(),
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=1,
        overlap=0,
        emit_delay=0,
        max_alignment_points=4,
        min_overlap_points=4,
        max_align_rmse=0.05,
        require_aligned_world_points=True,
        emit_unaligned=False,
    )
    target = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    pred = _prediction(feature_hw=(2, 2))
    pred.chunk_world_points[0] = target
    tracker.global_points_by_frame[7] = target.clone()
    history_pred = _prediction(feature_hw=(2, 2))
    history_pred.chunk_world_points[0] = target + 100.0
    history_record = KeyframeRecord(
        frame_id=3,
        pose_c2w=torch.eye(4),
        depth_low=torch.ones(1, 2, 2),
        dense_descriptors=torch.ones(4, 2, 2),
        match_confidence=torch.ones(1, 2, 2),
        sky_prob=torch.zeros(1, 2, 2),
        global_points=target - 100.0,
    )

    transform = tracker._align_chunk(
        pred,
        (7,),
        history_pred=history_pred,
        history_records=(history_record,),
    )

    assert transform.accepted
    assert tracker.last_alignment_debug.overlap_points == 4
    assert tracker.last_alignment_debug.history_points == 0


def test_alignment_excludes_sky_pixels_from_overlap_sim3():
    cfg = _anchor_cfg()
    cfg["PanoVGGT"]["Alignment"] = {
        "exclude_sky": True,
        "sky_threshold": 0.5,
    }
    tracker = PanoVGGTLongTracker(
        engine=_RecordingAnchorEngine(),
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=1,
        overlap=0,
        emit_delay=0,
        max_alignment_points=8,
        min_overlap_points=4,
        max_align_rmse=0.05,
        require_aligned_world_points=True,
        emit_unaligned=False,
    )
    yy, xx = torch.meshgrid(torch.arange(2, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij")
    target = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    source = target.clone()
    source[:, 2:] += torch.tensor([100.0, 100.0, 100.0])
    pred = _prediction(feature_hw=(2, 4))
    pred.chunk_world_points[0] = source
    pred.sky_prob[0, 0, :, 2:] = 0.95
    tracker.global_points_by_frame[7] = target.clone()

    transform = tracker._align_chunk(pred, (7,))

    assert transform.accepted
    assert tracker.last_alignment_debug.overlap_points == 4
    assert transform.residual < 1.0e-4


def test_keyframe_anchor_reuses_overlap_keyframe_without_prepending():
    engine = _RecordingAnchorEngine()
    tracker = _tracker_for_anchor_tests(engine)
    tracker.last_keyframe_id = 0
    tracker.last_keyframe_anchor = SimpleNamespace(
        frame_id=0,
        image=torch.zeros(3, 2, 4),
        pose_c2w=torch.eye(4),
    )

    _feed_two_frames(tracker)

    assert engine.batch_sizes == [2]


def test_low_pair_confidence_triggers_keyframe_and_novel_mask_filters_sky():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 0.1), sky_high_cell=(0, 0))
    tracker = _tracker_for_anchor_tests(engine)

    first = _feed_two_frames(tracker)
    assert first[0].is_keyframe
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    outputs = tracker.pop_ready_outputs()

    assert outputs[0].is_keyframe
    mask = outputs[0].valid_world_points_mask
    assert mask is not None
    assert not bool(mask[0, 0, 0])
    assert bool(mask[0, 1, 1])
    assert outputs[0].world_points_confidence is not None
    assert outputs[0].world_points_confidence[0, 1, 1] > 0.5
    decisions = tracker.pop_keyframe_decisions()
    assert any(item.get("accepted") and "frame_mean_pair_conf" in item for item in decisions)
    assert any("pair_conf_quantiles" in item for item in decisions)


def test_high_pair_confidence_and_small_translation_does_not_trigger_keyframe():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 1.0), translation_by_call=(0.0, 0.01, 0.01))
    tracker = _tracker_for_anchor_tests(engine)

    _feed_two_frames(tracker)
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    outputs = tracker.pop_ready_outputs()

    assert outputs
    assert not outputs[0].is_keyframe


def test_large_translation_triggers_keyframe_even_with_high_pair_confidence():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 1.0), translation_by_call=(0.0, 1.0, 1.0))
    tracker = _tracker_for_anchor_tests(engine)

    _feed_two_frames(tracker)
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    outputs = tracker.pop_ready_outputs()

    assert outputs[0].is_keyframe
    mask = outputs[0].valid_world_points_mask
    assert mask is not None
    assert not bool(mask.any())


def test_keyframe_anchor_min_interval_suppresses_motion_only_keyframe():
    tracker = _tracker_for_anchor_tests(
        _RecordingAnchorEngine(),
        anchor_overrides={
            "min_keyframe_interval": 4,
            "translation_threshold": 0.1,
            "translation_depth_ratio_threshold": 0.01,
        },
    )
    tracker.last_keyframe_id = 0
    tracker.last_keyframe_anchor = SimpleNamespace(
        frame_id=0,
        image=torch.zeros(3, 2, 4),
        pose_c2w=torch.eye(4),
    )
    pose = torch.eye(4)
    pose[0, 3] = 1.0

    accepted, decision = tracker._decide_keyframe(
        frame_id=2,
        pose=pose,
        inverse_depth=torch.ones(1, 2, 4),
        confidence=torch.ones(1, 2, 4),
        key_score=1.0,
        anchor_metrics=None,
    )

    assert not accepted
    assert decision["keyframe_gap"] == 2
    assert decision["suppressed_by_min_keyframe_interval"] is True
    assert "translation_depth_ratio" in decision["suppressed_reasons"]


def test_keyframe_anchor_gap_window_uses_m3_score_threshold():
    tracker = _tracker_for_anchor_tests(
        _RecordingAnchorEngine(),
        anchor_overrides={
            "min_keyframe_interval": 4,
            "max_keyframe_interval": 8,
            "m3_score_threshold": 0.43,
            "frame_mean_pair_conf_threshold": 0.0,
            "frame_low_pair_conf_ratio": 1.1,
            "translation_threshold": 99.0,
            "translation_depth_ratio_threshold": 99.0,
        },
    )
    tracker.last_keyframe_id = 0
    tracker.last_keyframe_anchor = SimpleNamespace(
        frame_id=0,
        image=torch.zeros(3, 2, 4),
        pose_c2w=torch.eye(4),
    )
    anchor_metrics = SimpleNamespace(
        anchor_frame_id=0,
        anchor_pose_c2w=torch.eye(4),
        match_coverage=0.0,
        frame_mean_pair_conf=0.5,
        low_pair_conf_ratio=0.0,
        pair_conf_quantiles={},
    )

    accepted, decision = tracker._decide_keyframe(
        frame_id=4,
        pose=torch.eye(4),
        inverse_depth=torch.ones(1, 2, 4),
        confidence=torch.ones(1, 2, 4),
        key_score=0.0,
        anchor_metrics=anchor_metrics,
    )

    assert accepted
    assert decision["reasons"] == ["m3_score"]
    assert decision["m3_score_threshold"] == 0.43
    assert decision["m3_keyframe_score"] >= 0.43


def test_keyframe_anchor_max_interval_forces_keyframe():
    tracker = _tracker_for_anchor_tests(
        _RecordingAnchorEngine(),
        anchor_overrides={"max_keyframe_interval": 4, "translation_threshold": 10.0, "translation_depth_ratio_threshold": 10.0},
    )
    tracker.last_keyframe_id = 0
    tracker.last_keyframe_anchor = SimpleNamespace(
        frame_id=0,
        image=torch.zeros(3, 2, 4),
        pose_c2w=torch.eye(4),
    )

    accepted, decision = tracker._decide_keyframe(
        frame_id=4,
        pose=torch.eye(4),
        inverse_depth=torch.ones(1, 2, 4),
        confidence=torch.ones(1, 2, 4),
        key_score=1.0,
        anchor_metrics=None,
    )

    assert accepted
    assert decision["reasons"] == ["max_keyframe_interval"]


def test_novel_insert_pair_threshold_adds_spatially_limited_candidates():
    tracker = _tracker_for_anchor_tests(
        _RecordingAnchorEngine(),
        novel_insert_options={
            "novel_pair_conf_insert_threshold": 0.95,
            "novel_insert_confidence_floor": 0.2,
            "novel_spatial_cell_size": 2,
            "novel_max_seeds_per_cell": 1,
        },
    )
    anchor_metrics = SimpleNamespace(
        pair_confidence=torch.full((1, 2, 4), 0.90),
        low_pair_conf=torch.zeros(1, 2, 4, dtype=torch.bool),
        non_sky=torch.ones(1, 2, 4, dtype=torch.bool),
    )

    mask, conf = tracker._novel_world_mask_and_confidence(
        valid_world_points_mask=torch.ones(1, 4, 8, dtype=torch.bool),
        confidence=torch.ones(1, 4, 8),
        image_size=(4, 8),
        anchor_metrics=anchor_metrics,
        sky_prob=None,
        first_keyframe=False,
    )

    assert int(mask.sum()) == 8
    assert float(conf[mask].min()) >= 0.2


def test_pfgs360_insertion_hints_do_not_replace_valid_world_mask():
    tracker = _tracker_for_anchor_tests(
        _RecordingAnchorEngine(),
        novel_insert_options={"novel_insertion_strategy": "pfgs360"},
    )
    tracker.last_keyframe_id = 0
    pose = torch.eye(4)
    pose[0, 3] = 1.0
    valid = torch.ones(1, 4, 8, dtype=torch.bool)
    valid[0, 0, 0] = False
    anchor_metrics = SimpleNamespace(
        anchor_frame_id=0,
        frame_mean_pair_conf=0.9,
        low_pair_conf_ratio=0.0,
        match_coverage=1.0,
        pair_conf_quantiles={},
        pair_confidence=torch.full((1, 2, 4), 0.9),
        low_pair_conf=torch.zeros(1, 2, 4, dtype=torch.bool),
        matched_cells=torch.ones(1, 2, 4, dtype=torch.bool),
        non_sky=torch.ones(1, 2, 4, dtype=torch.bool),
        anchor_pose_c2w=torch.eye(4),
    )

    output = tracker._make_output(
        PanoFrame(image=torch.zeros(3, 4, 8), timestamp=1.0, frame_id=1),
        pose=pose,
        inverse_depth=torch.ones(1, 4, 8),
        confidence=torch.ones(1, 4, 8),
        world_points=torch.zeros(4, 8, 3),
        valid_world_points_mask=valid,
        anchor_metrics=anchor_metrics,
        sky_prob=None,
        residual=0.0,
        status="tracked_panovggt_long",
    )
    hints = tracker.consume_insertion_hints(1)

    assert output.is_keyframe
    assert torch.equal(output.valid_world_points_mask, valid)
    assert hints is not None
    assert hints["pair_confidence"].shape == (1, 4, 8)


def test_tracker_exposes_panovggt_sky_mask_without_frontend_output_api_change():
    pred = _prediction(feature_hw=(2, 4))
    assert pred.sky_prob is not None
    pred.sky_prob[1, 0, 0, 1] = 0.95
    cfg = _anchor_cfg()
    tracker = PanoVGGTLongTracker(
        engine=_StaticEngine(pred),
        engine_config=cfg["PanoVGGT"],
        device="cpu",
        chunk_size=2,
        overlap=0,
        emit_delay=0,
        min_overlap_points=1,
        require_aligned_world_points=False,
        emit_unaligned=True,
        novel_insertion_enabled=True,
        novel_insertion_strategy="pfgs360",
    )

    def fake_align(aligned_pred, frame_ids, **kwargs):
        _ = aligned_pred, frame_ids, kwargs
        return SimilarityTransform.identity(device=pred.depth.device, dtype=pred.depth.dtype)

    tracker._align_chunk = fake_align
    for idx in range(2):
        tracker.track(PanoFrame(image=torch.zeros(3, 4, 8), timestamp=float(idx), frame_id=idx))
    outputs = tracker.pop_ready_outputs()
    mask = tracker.sky_mask_for_frame(1, image_size=(4, 8))
    hints = tracker._pfgs360_insertion_hints(
        image_size=(4, 8),
        anchor_metrics=None,
        sky_prob=pred.sky_prob[1],
        first_keyframe=True,
    )

    assert outputs
    assert mask is not None
    assert mask.shape == (1, 4, 8)
    assert bool(mask.any())
    assert "sky_mask" in hints
    assert hints["sky_mask"].shape == (1, 4, 8)


def test_tracker_shadow_mode_runs_ba_but_alignment_receives_original_prediction():
    pred0 = _prediction()
    refined = replace(pred0, poses_c2w=pred0.poses_c2w.clone())
    refined.poses_c2w[1, 0, 3] = 0.7
    cfg = _m3_config(shadow=True)
    tracker, captured = _tracker_with_fake_alignment(pred0, cfg)
    stats = DenseBARefinerStats(enabled=True, shadow_mode=True, success=True, used_refined=False)
    tracker.dense_ba_refiner = _FakeRefiner(refined, stats)

    outputs = _feed_two_frames(tracker)

    assert len(captured) == 1
    assert captured[0] is pred0
    assert tracker.dense_ba_refiner.calls == 1
    assert outputs
    assert "dense_ba_shadow_success" in outputs[0].tracking_status


def test_tracker_non_shadow_success_sends_refined_prediction_into_align_chunk():
    pred0 = _prediction()
    refined = replace(pred0, poses_c2w=pred0.poses_c2w.clone())
    refined.poses_c2w[1, 0, 3] = 0.7
    cfg = _m3_config(shadow=False)
    tracker, captured = _tracker_with_fake_alignment(pred0, cfg)
    stats = DenseBARefinerStats(enabled=True, shadow_mode=False, success=True, used_refined=True)
    tracker.dense_ba_refiner = _FakeRefiner(refined, stats)

    outputs = _feed_two_frames(tracker)

    assert len(captured) == 1
    assert captured[0] is refined
    assert outputs
    assert "dense_ba_active_success" in outputs[0].tracking_status


def test_tracker_history_window_passes_emitted_keyframes_to_refiner_memory():
    pred0 = _prediction()
    cfg = _m3_config(shadow=False, mode="history_window", history_keyframes=4)
    tracker, captured = _tracker_with_fake_alignment(pred0, cfg)
    stats = DenseBARefinerStats(
        enabled=True,
        shadow_mode=False,
        mode="history_window",
        success=False,
        used_refined=False,
        fallback_reason="empty_factors",
    )
    fake_refiner = _FakeRefiner(pred0, stats)
    tracker.dense_ba_refiner = fake_refiner

    _feed_two_frames(tracker)
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 5, 8), timestamp=float(idx), frame_id=idx))
    tracker.pop_ready_outputs()

    assert fake_refiner.memory_lengths[0] == 0
    assert fake_refiner.memory_lengths[1] >= 1
    assert len(tracker.keyframe_memory) >= 1
    assert len(captured) == 2


def test_tracker_enabled_descriptors_missing_fallback_still_emits_frontend_output():
    pred0 = _prediction(dense=False)
    cfg = _m3_config(shadow=False)
    tracker, captured = _tracker_with_fake_alignment(pred0, cfg)

    outputs = _feed_two_frames(tracker)

    assert len(captured) == 1
    assert captured[0] is pred0
    assert outputs
    assert outputs[0].tracking_status.startswith("tracked_panovggt_long|dense_ba_active_fallback:missing_dense_descriptors")


def test_slam_dense_ba_summary_aggregates_tracker_internal_stats():
    class Frontend:
        dense_ba_stats_history = [
            DenseBARefinerStats(
                enabled=True,
                shadow_mode=True,
                success=True,
                used_refined=False,
                mean_residual_deg=0.4,
                initial_mean_residual_deg=1.0,
                valid_factor_ratio=0.25,
                pose_update_norm={"mean": 0.01, "max": 0.02, "rot_max_deg": 0.3},
                depth_update_norm={"mean": 0.03, "max": 0.04},
            ),
            DenseBARefinerStats(
                enabled=True,
                shadow_mode=True,
                success=False,
                used_refined=False,
                fallback_reason="too_few_factors",
                mean_residual_deg=0.0,
                initial_mean_residual_deg=0.0,
                valid_factor_ratio=0.01,
                pose_update_norm={},
                depth_update_norm={},
            ),
        ]
        last_dense_ba_stats = dense_ba_stats_history[-1]

    summary = _summarize_dense_ba_stats(Frontend())

    assert summary["enabled"] is True
    assert summary["shadow_mode"] is True
    assert summary["chunks"] == 2
    assert summary["successes"] == 1
    assert summary["fallbacks"] == 1
    assert summary["success_ratio"] == 0.5
    assert summary["used_refined"] == 0
    assert summary["fallback_reasons"] == {"too_few_factors": 1}
    assert abs(summary["mean_valid_factor_ratio"] - 0.13) < 1e-8
    assert summary["max_pose_update"] == 0.02
    assert summary["max_depth_update"] == 0.04


def test_slam_logger_saves_minimal_m3_debug_visualizations(tmp_path):
    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": True, "m3_log_every": 1, "m3_max_matches": 4},
        },
        tmp_path,
    )
    factor = DenseSphereFactor(
        src=0,
        tgt=1,
        src_uv=torch.tensor([[1.5, 1.5], [2.5, 2.5]]),
        tgt_uv=torch.tensor([[2.5, 1.5], [3.5, 2.5]]),
        src_bearing=torch.randn(2, 3),
        tgt_bearing=torch.randn(2, 3),
        weight=torch.ones(2),
        match_score=torch.ones(2),
        valid_mask=torch.ones(2, dtype=torch.bool),
    )
    debug = {
        "chunk_index": 0,
        "stats": DenseBARefinerStats(
            enabled=True,
            shadow_mode=True,
            success=True,
            mean_residual_deg=0.25,
            initial_mean_residual_deg=0.75,
            valid_factor_ratio=0.5,
        ),
        "factor_graph": DenseSphereFactorGraph([factor]),
        "sky_prob": torch.linspace(0.0, 1.0, steps=64).view(2, 1, 4, 8),
        "feature_hw": (4, 8),
        "image_hw": (4, 8),
        "images": torch.rand(2, 3, 4, 8),
    }
    output = FrontendOutput(
        frame_id=0,
        timestamp=0.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=0.0,
        is_keyframe=False,
        ba_residual=None,
        tracking_status="tracked_panovggt_long|dense_ba_shadow_success",
    )

    logger.observe(
        output,
        PanoFrame(image=torch.rand(3, 4, 8), timestamp=0.0, frame_id=0),
        anchor_count=0,
        keyframe_count=0,
        backend_loss=None,
        m3_debug=debug,
    )

    assert (tmp_path / "visualizations" / "m3_chunk_000000_match_lines.png").is_file()
    assert (tmp_path / "visualizations" / "m3_chunk_000000_sky_prob.png").is_file()


def test_slam_logger_saves_and_logs_keyframe_opt_visualizations(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value):
            return ("image", value)

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": True, "save_kf_opt": True, "kf_opt_log_every": 1},
            "Results": {"kf_render_format": "png"},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()
    logger._step = 4
    diagnostic = SimpleNamespace(
        frame_id=3,
        target=torch.rand(3, 4, 8),
        render=torch.rand(3, 4, 8),
        depth=torch.rand(1, 4, 8).clamp_min(0.1),
        loss=0.125,
        psnr=22.5,
        anchor_count=42,
        phase="local_submap",
    )

    logger.observe_keyframe_opt(diagnostic)

    render_path = tmp_path / "kf_renders_opt" / "kf_0003.png"
    depth_path = tmp_path / "kf_depths_opt" / "kf_0003.png"
    assert render_path.is_file()
    assert depth_path.is_file()
    logged_keys = set().union(*(payload.keys() for payload, _ in logger.run.logged))
    assert "backend/kf_opt_loss" in logged_keys
    assert "backend/kf_render_opt" in logged_keys
    assert "backend/kf_depth_opt" in logged_keys


def test_slam_logger_can_disable_keyframe_opt_file_saves(tmp_path):
    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": True, "save_kf_opt": False},
        },
        tmp_path,
    )
    diagnostic = SimpleNamespace(
        frame_id=5,
        target=torch.rand(3, 4, 8),
        render=torch.rand(3, 4, 8),
        depth=torch.rand(1, 4, 8),
        loss=0.0,
        psnr=0.0,
        anchor_count=1,
        phase="local_submap",
    )

    logger.observe_keyframe_opt(diagnostic)

    assert not (tmp_path / "kf_renders_opt").exists()
    assert not (tmp_path / "kf_depths_opt").exists()


def test_slam_logger_logs_every_post_optimized_window_frame_and_preserves_overlap(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value, caption=None):
            return ("image", value, caption)

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled", "runtime_log_preset": "compact_slam"},
            "Visualization": {
                "save_local": True,
                "post_opt_all_frames": True,
                "post_opt_log_depth": False,
            },
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()

    def diagnostic(frame_id: int, loss: float, psnr: float):
        return SimpleNamespace(
            frame_id=frame_id,
            target=torch.rand(3, 4, 8),
            render=torch.rand(3, 4, 8),
            depth=torch.rand(1, 4, 8).clamp_min(0.1),
            target_depth=torch.rand(1, 4, 8).clamp_min(0.1),
            loss=loss,
            psnr=psnr,
            anchor_count=42,
            phase="feedforward_window",
        )

    logger.observe_post_optimized_window(
        [diagnostic(0, 0.2, 18.0), diagnostic(3, 0.1, 20.0)],
        window_id=0,
        step=4,
    )
    logger.observe_post_optimized_window(
        [diagnostic(3, 0.08, 21.0), diagnostic(4, 0.07, 22.0)],
        window_id=1,
        step=7,
    )

    first_overlap = (
        tmp_path
        / "visualizations"
        / "post_opt"
        / "window_000000"
        / "frame_000003_render_vs_gt.png"
    )
    second_overlap = (
        tmp_path
        / "visualizations"
        / "post_opt"
        / "window_000001"
        / "frame_000003_render_vs_gt.png"
    )
    assert first_overlap.is_file()
    assert second_overlap.is_file()
    assert not list((tmp_path / "visualizations" / "post_opt").rglob("*_depth.png"))

    media_payloads = [
        (payload, step)
        for payload, step in logger.run.logged
        if "backend/post_opt_window_frames" in payload
    ]
    assert [step for _, step in media_payloads] == [4, 7]
    assert [len(payload["backend/post_opt_window_frames"]) for payload, _ in media_payloads] == [2, 2]
    assert "window=0 frame=3" in media_payloads[0][0]["backend/post_opt_window_frames"][1][2]
    assert "window=1 frame=3" in media_payloads[1][0]["backend/post_opt_window_frames"][0][2]
    assert media_payloads[0][0]["backend/post_opt_frame_count"] == 2


def test_slam_logger_saves_new_gaussian_insertion_visualization(tmp_path):
    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": True},
        },
        tmp_path,
    )

    path = logger.observe_new_gaussians(
        frame_id=7,
        image=torch.zeros(3, 4, 8),
        source_hw=(4, 8),
        requested_idx=torch.tensor([0, 3, 9, 10]),
        inserted_idx=torch.tensor([3, 10]),
        stats={"kept": 2, "skipped_voxel": 1},
    )

    assert path is not None
    assert path.is_file()
    assert path.name == "frame_000007.png"


def test_slam_logger_logs_keyframe_mapping_diagnostics_at_explicit_step(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": False},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._step = 12
    explicit_step = 13

    logger.observe_new_gaussians(
        frame_id=7,
        image=torch.zeros(3, 4, 8),
        source_hw=(4, 8),
        requested_idx=torch.tensor([0, 3, 9, 10]),
        inserted_idx=torch.tensor([3, 10]),
        stats={"kept": 2, "skipped_voxel": 1},
        step=explicit_step,
    )

    diagnostic = SimpleNamespace(
        frame_id=7,
        render_depth=torch.ones(1, 4, 8),
        predicted_depth=torch.full((1, 4, 8), 1.2),
        rel_depth_error=torch.full((1, 4, 8), 0.2),
        missing_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_mismatch_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        render_bad_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_scale=1.0,
        depth_shift=0.0,
    )
    diagnostic.depth_mismatch_mask[..., 1, 1] = True
    diagnostic.render_bad_mask = diagnostic.depth_mismatch_mask
    logger.observe_depth_insertion_diagnostic(
        frame_id=7,
        image=torch.zeros(3, 4, 8),
        source_hw=(4, 8),
        inserted_idx=torch.tensor([9]),
        diagnostic=diagnostic,
        stats={"kept": 1},
        step=explicit_step,
    )

    assert logger._step == 12
    assert len(logger.run.logged) == 2
    assert all(step == explicit_step for _, step in logger.run.logged)
    logged_keys = set().union(*(payload.keys() for payload, _ in logger.run.logged))
    assert "mapping/new_gaussians_requested" in logged_keys
    assert "mapping/depth_insertion_need_pixels" in logged_keys


def test_slam_logger_compact_wandb_keeps_depth_insertion_and_insert_count(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value):
            return ("image", value)

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled", "runtime_log_preset": "compact_slam"},
            "Visualization": {"save_local": True, "save_kf_opt": True},
            "Results": {"kf_render_format": "png"},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()
    logger._step = 4

    logger.observe_keyframe_opt(
        SimpleNamespace(
            frame_id=3,
            target=torch.rand(3, 4, 8),
            render=torch.rand(3, 4, 8),
            depth=torch.rand(1, 4, 8).clamp_min(0.1),
            loss=0.125,
            psnr=22.5,
            anchor_count=42,
            phase="local_submap",
        ),
        step=5,
    )
    logger.observe_keyframe_inserted_gaussians(frame_id=3, inserted_count=1234, step=5)

    diagnostic = SimpleNamespace(
        frame_id=3,
        render_depth=torch.ones(1, 4, 8),
        predicted_depth=torch.full((1, 4, 8), 1.2),
        rel_depth_error=torch.full((1, 4, 8), 0.2),
        missing_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_mismatch_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        render_bad_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_scale=1.0,
        depth_shift=0.0,
    )
    diagnostic.depth_mismatch_mask[..., 1, 1] = True
    diagnostic.render_bad_mask = diagnostic.depth_mismatch_mask
    logger.observe_depth_insertion_diagnostic(
        frame_id=3,
        image=torch.zeros(3, 4, 8),
        source_hw=(4, 8),
        inserted_idx=torch.tensor([9]),
        diagnostic=diagnostic,
        stats={"kept": 1},
        step=5,
    )

    logged_keys = set().union(*(payload.keys() for payload, _ in logger.run.logged))
    assert "backend/kf_opt_loss" in logged_keys
    assert "backend/kf_opt_psnr" in logged_keys
    assert "backend/kf_render_opt" in logged_keys
    assert "backend/kf_depth_opt" in logged_keys
    assert "mapping/keyframe_inserted_gaussians" in logged_keys
    assert "mapping/new_gaussians_inserted" in logged_keys
    assert "mapping/depth_insertion" in logged_keys
    assert "mapping/depth_insertion_need_pixels" in logged_keys
    assert "mapping/depth_insertion_png" not in logged_keys


def test_slam_core_visuals_separate_pose_streams_and_filter_wandb(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value):
            return ("image", value)

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {
                "mode": "disabled",
                "runtime_log_preset": "slam_core_visuals",
            },
            "Visualization": {"save_local": True, "log_every": 1},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()
    plotted_histories = {}
    save_trajectory_panel = logger._save_trajectory_panel

    def capture_trajectory_panel(output, *, kind, pred_history):
        plotted_histories[kind] = list(pred_history)
        return save_trajectory_panel(
            output,
            kind=kind,
            pred_history=pred_history,
        )

    logger._save_trajectory_panel = capture_trajectory_panel

    def pose_x(value: float) -> torch.Tensor:
        pose = torch.eye(4)
        pose[0, 3] = float(value)
        return pose

    output = FrontendOutput(
        frame_id=1,
        timestamp=1.0,
        pose_c2w=pose_x(1.0),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=0.0,
        is_keyframe=False,
        ba_residual=None,
        tracking_status="tracked",
    )
    logger.record_frontend_raw(output)
    logger.replace_frontend_sim3_history({1: pose_x(1.5)})
    logger.replace_geometry_history(
        {1: SimpleNamespace(pose_c2w=pose_x(2.0))},
        revision=1,
        record_backend_graph=True,
    )
    logger.replace_geometry_history(
        {1: SimpleNamespace(pose_c2w=pose_x(2.5))},
        revision=2,
        record_backend_graph=False,
    )
    logger.observe(
        output,
        PanoFrame(
            image=torch.rand(3, 4, 8),
            timestamp=1.0,
            frame_id=1,
            meta={"gt_c2w": pose_x(4.0)},
        ),
        anchor_count=10,
        keyframe_count=1,
        backend_loss=0.5,
        backend_pose_c2w=pose_x(2.0),
        slam_refined_poses_c2w={1: pose_x(3.0)},
        backend_render_pkg={
            "render": torch.rand(3, 4, 8),
            "depth": torch.ones(1, 4, 8),
        },
        defer_trajectory_logging=True,
    )
    keys_before_post_photo = set().union(
        *(payload.keys() for payload, _ in logger.run.logged)
    )
    assert "frontend/trajectory_vs_gt" not in keys_before_post_photo
    assert "backend/trajectory_vs_gt" not in keys_before_post_photo
    assert "slam/trajectory_vs_gt" not in keys_before_post_photo
    logger._log_wandb_payload(
        {
            "backend/new_gaussians_per_chunk": 123,
            "backend/selfi_graph_objective": 9.0,
        },
        step=1,
    )
    logger.observe_trajectory_comparison(
        output,
        slam_refined_poses_c2w={1: pose_x(3.0)},
    )
    final_path = logger.log_final_slam_trajectory(
        [(0, pose_x(0.0)), (1, pose_x(1.0)), (2, pose_x(2.0))],
        trajectory_metrics={
            "sim3_ate_rmse": 0.12,
            "se3_ate_rmse": 0.34,
        },
        fallback_ate_rmse=None,
        step=8,
    )

    assert logger._frontend_raw_pose_history[-1][1][0] == pytest.approx(1.0)
    assert logger._frontend_sim3_pose_history[-1][1][0] == pytest.approx(1.5)
    assert logger._backend_graph_pose_history[-1][1][0] == pytest.approx(2.0)
    assert logger._backend_global_pose_history[-1][1][0] == pytest.approx(2.5)
    assert logger._slam_final_pose_history[-1][1][0] == pytest.approx(3.0)
    assert plotted_histories["frontend"][-1][1][0] == pytest.approx(1.5)
    assert plotted_histories["backend"][-1][1][0] == pytest.approx(2.0)
    assert plotted_histories["slam"][-1][1][0] == pytest.approx(3.0)
    assert (
        tmp_path
        / "visualizations"
        / "frame_000001_frontend_trajectory_vs_gt.png"
    ).is_file()
    assert (
        tmp_path
        / "visualizations"
        / "frame_000001_backend_trajectory_vs_gt.png"
    ).is_file()
    assert (
        tmp_path
        / "visualizations"
        / "frame_000001_slam_trajectory_vs_gt.png"
    ).is_file()
    assert final_path is not None
    assert Path(final_path).is_file()
    assert len(logger.run.logged) == 1
    logged_payload, logged_step = logger.run.logged[0]
    logged_keys = set(logged_payload)
    assert logged_keys == {
        "slam/frame_id",
        "slam/frame_index",
        "frontend/trajectory_vs_gt",
        "backend/trajectory_vs_gt",
        "slam/trajectory_vs_gt",
        "backend/render_vs_gt_panorama",
        "backend/new_gaussians_per_chunk",
    }
    assert logged_step == 1
    assert logger.run.summary.keys() >= {
        "slam/final_trajectory_vs_gt",
        "slam/final_ate_rmse",
        "slam/final_sim3_ate_rmse",
        "slam/final_se3_ate_rmse",
    }


def test_slam_logger_both_ate_mode_keeps_standard_primary_metric(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value):
            return ("image", value)

    logger = SlamRuntimeLogger(
        {
            "TrajectoryEvaluation": {"ate_mode": "both"},
            "WeightsAndBiases": {
                "mode": "disabled",
                "runtime_log_preset": "slam_core_visuals",
            },
            "Visualization": {"save_local": True},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()
    poses = []
    for frame_id in range(3):
        pose = torch.eye(4)
        pose[0, 3] = float(frame_id)
        poses.append((frame_id, pose))

    logger.log_final_slam_trajectory(
        poses,
        trajectory_metrics={
            "sim3_ate_rmse": 0.12,
            "se3_ate_rmse": 0.34,
            "pfgs360_ate": 0.045,
            "scale_drift_percent": 7.5,
            "path_length_scale_ratio": 1.075,
        },
        fallback_ate_rmse=None,
        step=8,
    )

    scalar_payload = logger.run.summary
    assert scalar_payload["slam/final_ate_rmse"] == pytest.approx(0.12)
    assert scalar_payload["slam/final_sim3_ate_rmse"] == pytest.approx(0.12)
    assert scalar_payload["slam/final_pfgs360_ate"] == pytest.approx(0.045)
    assert scalar_payload["slam/final_se3_ate_rmse"] == pytest.approx(0.34)
    assert scalar_payload["slam/final_scale_drift_percent"] == pytest.approx(7.5)
    assert scalar_payload["slam/final_path_length_scale_ratio"] == pytest.approx(
        1.075
    )


def test_slam_core_visuals_commit_one_step_per_frame_and_clip_future_poses(tmp_path):
    class _Run:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    class _Wandb:
        @staticmethod
        def Image(value):
            return ("image", value)

    def pose_x(value: float) -> torch.Tensor:
        pose = torch.eye(4)
        pose[0, 3] = float(value)
        return pose

    def output(frame_id: int) -> FrontendOutput:
        return FrontendOutput(
            frame_id=frame_id,
            timestamp=float(frame_id),
            pose_c2w=pose_x(float(frame_id)),
            relative_pose=None,
            pose_confidence=1.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=False,
            ba_residual=None,
            tracking_status="tracked",
        )

    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {
                "mode": "disabled",
                "runtime_log_preset": "slam_core_visuals",
            },
            # Core visuals must still be emitted for every frame.
            "Visualization": {"save_local": True, "log_every": 100},
        },
        tmp_path,
    )
    logger.run = _Run()
    logger._wandb = _Wandb()
    logger.replace_frontend_sim3_history({10: pose_x(1.0), 20: pose_x(2.0)})
    logger.replace_geometry_history(
        {
            10: SimpleNamespace(pose_c2w=pose_x(1.5)),
            20: SimpleNamespace(pose_c2w=pose_x(2.5)),
        },
        revision=1,
        record_backend_graph=True,
    )

    plotted: list[tuple[int, str, list[int]]] = []
    save_trajectory_panel = logger._save_trajectory_panel

    def capture(output_value, *, kind, pred_history):
        plotted.append(
            (
                int(output_value.frame_id),
                str(kind),
                [int(frame_id) for frame_id, _ in pred_history],
            )
        )
        return save_trajectory_panel(
            output_value,
            kind=kind,
            pred_history=pred_history,
        )

    logger._save_trajectory_panel = capture
    for frame_id in (10, 20):
        current = output(frame_id)
        logger.observe(
            current,
            PanoFrame(
                image=torch.rand(3, 4, 8),
                timestamp=float(frame_id),
                frame_id=frame_id,
                meta={"gt_c2w": pose_x(float(frame_id))},
            ),
            anchor_count=10,
            keyframe_count=1,
            backend_loss=None,
            backend_pose_c2w=pose_x(float(frame_id)),
            slam_refined_poses_c2w={
                10: pose_x(1.75),
                20: pose_x(2.75),
            },
            backend_render_pkg={
                "render": torch.rand(3, 4, 8),
                "depth": torch.ones(1, 4, 8),
            },
        )

    assert [step for _, step in logger.run.logged] == [1, 2]
    assert all(
        {
            "frontend/trajectory_vs_gt",
            "backend/trajectory_vs_gt",
            "slam/trajectory_vs_gt",
            "backend/render_vs_gt_panorama",
        }
        <= set(payload)
        for payload, _ in logger.run.logged
    )
    first_frame_histories = [ids for frame_id, _, ids in plotted if frame_id == 10]
    assert first_frame_histories
    assert all(ids == [10] for ids in first_frame_histories)


def test_slam_logger_saves_depth_insertion_diagnostic_visualization(tmp_path):
    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"mode": "disabled"},
            "Visualization": {"save_local": True},
        },
        tmp_path,
    )
    diagnostic = SimpleNamespace(
        frame_id=7,
        render_depth=torch.ones(1, 4, 8),
        predicted_depth=torch.full((1, 4, 8), 1.2),
        rel_depth_error=torch.full((1, 4, 8), 0.2),
        missing_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_mismatch_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        render_bad_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        depth_scale=1.0,
        depth_shift=0.0,
    )
    diagnostic.missing_mask[..., 0, 0] = True
    diagnostic.depth_mismatch_mask[..., 1, 1] = True
    diagnostic.render_bad_mask = diagnostic.missing_mask | diagnostic.depth_mismatch_mask

    path = logger.observe_depth_insertion_diagnostic(
        frame_id=7,
        image=torch.zeros(3, 4, 8),
        source_hw=(4, 8),
        inserted_idx=torch.tensor([0, 9]),
        diagnostic=diagnostic,
        stats={"kept": 2},
    )

    assert path is not None
    assert path.is_file()
    assert path.parent.name == "depth_insertion"
    assert path.name == "frame_000007.png"
