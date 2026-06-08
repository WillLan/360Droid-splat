import inspect
from dataclasses import replace
from types import SimpleNamespace

import torch

from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from frontend.pano_vggt.alignment import SimilarityTransform
from frontend.pano_vggt.dense_ba_refiner import DenseBARefinerStats, PanoVGGTDenseBARefiner
from frontend.pano_vggt.engine import PanoVGGTInferenceEngine
from frontend.pano_vggt.factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from frontend.pano_vggt.keyframe_memory import KeyframeMemory, KeyframeRecord
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

    def refine(self, pred, frame_ids, *, factor_graph=None, keyframe_memory=None):
        _ = pred, frame_ids, factor_graph, keyframe_memory
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

    def fake_align(aligned_pred, frame_ids):
        _ = frame_ids
        captured.append(aligned_pred)
        return SimilarityTransform.identity(device=aligned_pred.depth.device, dtype=aligned_pred.depth.dtype)

    tracker._align_chunk = fake_align
    return tracker, captured


def _feed_two_frames(tracker: PanoVGGTLongTracker):
    for idx in range(2):
        tracker.track(PanoFrame(image=torch.zeros(3, 5, 8), timestamp=float(idx), frame_id=idx))
    return tracker.pop_ready_outputs()


def _anchor_cfg() -> dict:
    return {
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


def _tracker_for_anchor_tests(engine: _RecordingAnchorEngine, *, novel: bool = True) -> PanoVGGTLongTracker:
    cfg = _anchor_cfg()
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
    )

    def fake_align(aligned_pred, frame_ids):
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


def test_keyframe_anchor_uses_sidepath_without_prepending_main_forward():
    engine = _RecordingAnchorEngine(current_conf_by_call=(1.0, 1.0, 1.0))
    tracker = _tracker_for_anchor_tests(engine)

    _feed_two_frames(tracker)
    for idx in range(2, 4):
        tracker.track(PanoFrame(image=torch.zeros(3, 2, 4), timestamp=float(idx), frame_id=idx))
    tracker.pop_ready_outputs()

    assert engine.batch_sizes == [2, 2, 3]


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
