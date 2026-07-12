from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from losses.spherical_stage3_refinement_loss import (
    aligned_pose_metrics,
    build_ba_support_map,
    leave_one_out_render_loss,
)
from models.spherical_recurrent_gaussian_refiner import (
    ReSplatErrorEncoder,
    SphericalErrorRouter,
    SphericalRecurrentGaussianRefiner,
    quaternion_exp_map,
    quaternion_log_map,
)
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead
from models.spherical_selfi_stage3_ba import (
    BlockSparseSphericalBA,
    Stage3MatchCache,
    _weighted_affine_fit,
    all_directed_pairs,
    build_stage3_match_cache,
)
from geometry.spherical_erp import build_erp_ray_grid, erp_pixel_to_unit_ray
from frontend.pano_droid.spherical_ba import se3_exp
from training.train_spherical_ba_recurrent_refiner import _ba_outer_schedule, default_config, train
from tools.generate_stage3_ba_ablation_configs import EXPERIMENTS, generate as generate_ablation_configs
from tools.generate_stage3_ba_gate_sweep import (
    HIGH_PARALLAX_VARIANTS,
    TRUST_REGION_VARIANTS,
    VARIANTS,
    generate as generate_gate_sweep,
)
from tools.summarize_stage3_ba_ablation import summarize_checkpoint
from tools.evaluate_stage3_ba import evaluate


def _observation(*, views: int = 3, height: int = 8, width: int = 16):
    torch.manual_seed(4)
    feature = torch.randn(1, views, 24, height, width)
    image = torch.rand(1, views, 3, height, width)
    depth = torch.ones(1, views, 1, height, width) * 2.0
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    poses[0, :, 0, 3] = torch.arange(views).float() * 0.05
    head = SphericalSelfiGaussianHead(channels=(8, 16, 24, 32), mlp_hidden_dim=16)
    return head(feature, image, depth, poses), feature, image


def test_observation_functional_update_preserves_provenance_and_recomputes_confidence() -> None:
    observation, _, _ = _observation()
    density = observation.density_sh.clone()
    density[:, :, 0] += 1.0
    updated = observation.with_updates(
        refined_depth=observation.refined_depth * 1.1,
        density_sh=density,
    )
    assert updated.source_uv.data_ptr() == observation.source_uv.data_ptr()
    assert updated.source_ray.data_ptr() == observation.source_ray.data_ptr()
    assert updated.frame_ids.data_ptr() == observation.frame_ids.data_ptr()
    assert updated.canonical_count == observation.canonical_count
    assert bool((updated.confidence > observation.confidence).all())


def test_all_directed_pairs_and_global_match_contract() -> None:
    observation, feature, _ = _observation(views=4, height=4, width=8)
    query_uv = torch.tensor([[[[0.5, 0.5], [3.5, 2.5]]] * 4], dtype=torch.float32)
    cache = build_stage3_match_cache(
        feature,
        observation.refined_depth,
        num_queries=2,
        query_chunk_size=1,
        query_uv=query_uv,
    )
    assert cache.edges.shape == (12, 2)
    assert torch.equal(cache.edges, all_directed_pairs(4))
    assert cache.target_uv.shape == (1, 12, 2, 2)
    assert cache.source_uv.shape == (1, 4, 2, 2)
    assert cache.num_factors == 24
    assert bool(torch.isfinite(cache.entropy).all())


def test_match_cache_filters_static_source_and_target_pixels() -> None:
    feature = torch.randn(1, 2, 8, 4, 8)
    depth = torch.ones(1, 2, 1, 4, 8)
    static = torch.ones_like(depth, dtype=torch.bool)
    static[:, 1] = False
    cache = build_stage3_match_cache(
        feature,
        depth,
        num_queries=4,
        query_chunk_size=2,
        static_valid_mask=static,
    )
    assert not bool(cache.source_valid[:, 1].any())
    edge_to_invalid_target = int(
        torch.nonzero((cache.edges[:, 0] == 0) & (cache.edges[:, 1] == 1))[0]
    )
    assert not bool(cache.target_valid[:, edge_to_invalid_target].any())
    assert not bool(cache.valid_mask.any())


def test_match_cache_reliability_fraction_keeps_top_ranked_per_edge() -> None:
    observation, feature, _ = _observation(views=2, height=4, width=8)
    cache = build_stage3_match_cache(
        feature,
        observation.refined_depth,
        num_queries=8,
        query_chunk_size=2,
        forward_backward=False,
        min_factor_weight=0.0,
        reliability_keep_fraction=0.25,
        generator=torch.Generator().manual_seed(9),
    )
    assert cache.edges.shape[0] == 2
    assert torch.equal(cache.valid_mask.sum(dim=-1), torch.full((1, 2), 2))
    for edge in range(2):
        selected = cache.factor_weight[0, edge][cache.valid_mask[0, edge]]
        rejected = cache.factor_weight[0, edge][~cache.valid_mask[0, edge]]
        assert float(selected.min()) >= float(rejected.max())


def test_match_cache_distinctiveness_margin_excludes_the_local_peak() -> None:
    height, width = 8, 16
    smooth = build_erp_ray_grid(height, width).permute(2, 0, 1)
    feature = smooth.view(1, 1, 3, height, width).repeat(1, 2, 1, 1, 1)
    depth = torch.full((1, 2, 1, height, width), 2.0)
    query_uv = torch.tensor(
        [[[[0.5, 2.5], [4.5, 2.5], [8.5, 4.5], [12.5, 4.5]]] * 2],
        dtype=torch.float32,
    )
    local_margin = build_stage3_match_cache(
        feature,
        depth,
        num_queries=4,
        query_chunk_size=2,
        query_uv=query_uv,
        forward_backward=False,
        use_spherical_area_correction=False,
        distinctiveness_exclusion_deg=0.0,
    )
    independent_margin = build_stage3_match_cache(
        feature,
        depth,
        num_queries=4,
        query_chunk_size=2,
        query_uv=query_uv,
        forward_backward=False,
        use_spherical_area_correction=False,
        distinctiveness_exclusion_deg=45.0,
    )
    torch.testing.assert_close(independent_margin.target_uv, local_margin.target_uv)
    assert bool((independent_margin.top2_margin >= local_margin.top2_margin).all())
    assert bool((independent_margin.top2_margin > local_margin.top2_margin).any())


def test_match_cache_subpixel_refinement_reduces_smooth_bearing_quantization() -> None:
    height, width = 16, 32
    smooth = build_erp_ray_grid(height, width).permute(2, 0, 1)
    feature = smooth.view(1, 1, 3, height, width).repeat(1, 2, 1, 1, 1)
    depth = torch.full((1, 2, 1, height, width), 2.0)
    query_uv = torch.tensor([[[[4.2, 6.2]], [[4.2, 6.2]]]], dtype=torch.float32)
    discrete = build_stage3_match_cache(
        feature,
        depth,
        num_queries=1,
        query_chunk_size=1,
        query_uv=query_uv,
        forward_backward=False,
        use_spherical_area_correction=False,
        subpixel_refine_radius=0,
    )
    refined = build_stage3_match_cache(
        feature,
        depth,
        num_queries=1,
        query_chunk_size=1,
        query_uv=query_uv,
        forward_backward=False,
        use_spherical_area_correction=False,
        subpixel_refine_radius=1,
    )
    expected_ray = erp_pixel_to_unit_ray(query_uv[:, :1], height, width)[0, 0, 0]
    discrete_error = torch.acos(
        (discrete.target_ray[0, 0, 0] * expected_ray).sum().clamp(-1.0, 1.0)
    )
    refined_error = torch.acos(
        (refined.target_ray[0, 0, 0] * expected_ray).sum().clamp(-1.0, 1.0)
    )
    assert float(refined_error) < float(discrete_error)
    assert 0.0 <= float(refined.target_uv[0, 0, 0, 0]) < width


def test_affine_depth_fit_recovers_scale_and_shift_with_outlier() -> None:
    source = torch.linspace(1.0, 10.0, 100)
    target = 1.2 * source - 0.3
    target[-1] = 50.0
    scale, shift = _weighted_affine_fit(source, target, torch.ones_like(source), median_depth=5.5)
    torch.testing.assert_close(scale, torch.tensor(1.2), atol=2e-2, rtol=0.0)
    torch.testing.assert_close(shift, torch.tensor(-0.3), atol=8e-2, rtol=0.0)


def test_block_sparse_ba_returns_finite_dense_contract() -> None:
    observation, feature, _ = _observation(views=3)
    cache = build_stage3_match_cache(
        feature,
        observation.refined_depth,
        num_queries=3,
        query_chunk_size=2,
        generator=torch.Generator().manual_seed(1),
    )
    solver = BlockSparseSphericalBA(iterations=1, min_factors=1, min_affine_support=2, factor_chunk_size=4)
    output = solver(observation.poses_c2w, observation.refined_depth, cache)
    assert output.dense_depth.shape == observation.refined_depth.shape
    assert output.sparse_depth.shape == (1, 3, 3)
    assert output.depth_scale.shape == (1, 3)
    assert bool(torch.isfinite(output.dense_depth).all())
    assert bool((output.dense_depth > 0).all())


def test_dense_depth_none_is_strict_identity() -> None:
    observation, feature, _ = _observation(views=3)
    cache = build_stage3_match_cache(
        feature,
        observation.refined_depth,
        num_queries=3,
        query_chunk_size=2,
        generator=torch.Generator().manual_seed(3),
    )
    solver = BlockSparseSphericalBA(
        iterations=1,
        min_factors=1,
        min_affine_support=2,
        factor_chunk_size=4,
        dense_depth_mode="none",
    )
    output = solver(observation.poses_c2w, observation.refined_depth, cache)
    assert torch.equal(output.dense_depth, observation.refined_depth)
    assert torch.equal(output.depth_scale, torch.ones_like(output.depth_scale))
    assert torch.equal(output.depth_shift, torch.zeros_like(output.depth_shift))


def test_initial_baseline_gauge_preserves_bearing_geometry() -> None:
    solver = BlockSparseSphericalBA(gauge_mode="initial_baseline")
    initial = torch.eye(4).repeat(3, 1, 1)
    initial[1, :3, 3] = torch.tensor([0.2, -0.1, 0.0])
    initial[2, :3, 3] = torch.tensor([0.8, 0.2, -0.1])
    global_scale = 1.7
    current = initial.clone()
    current[:, :3, 3] = initial[0, :3, 3] + global_scale * (
        initial[:, :3, 3] - initial[0, :3, 3]
    )
    inverse_depth = torch.tensor([0.4, 0.25, 0.5])
    current_log_inverse_depth = inverse_depth.log() - torch.log(torch.tensor(global_scale))
    source = torch.tensor([0], dtype=torch.long)
    target = torch.tensor([2], dtype=torch.long)
    depth_index = torch.tensor([0], dtype=torch.long)
    source_ray = torch.tensor([[0.2, -0.1, 0.97]])
    source_ray = torch.nn.functional.normalize(source_ray, dim=-1)
    before = solver._predicted_bearing(
        current,
        current_log_inverse_depth,
        source,
        target,
        depth_index,
        source_ray,
    )
    normalized_pose, normalized_log_depth, gauge_scale, valid = solver._apply_scale_gauge(
        current,
        current_log_inverse_depth,
        initial,
    )
    after = solver._predicted_bearing(
        normalized_pose,
        normalized_log_depth,
        source,
        target,
        depth_index,
        source_ray,
    )
    assert valid
    assert abs(gauge_scale - 1.0 / global_scale) < 1.0e-6
    initial_baseline = (initial[2, :3, 3] - initial[0, :3, 3]).norm()
    normalized_baseline = (normalized_pose[2, :3, 3] - normalized_pose[0, :3, 3]).norm()
    torch.testing.assert_close(normalized_baseline, initial_baseline, atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(after, before, atol=1.0e-6, rtol=0.0)


def test_baseline_gauge_jacobian_matches_finite_difference() -> None:
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[1, :3, 3] = torch.tensor([0.2, -0.1, 0.0])
    poses[2, :3, 3] = torch.tensor([0.8, 0.2, -0.1])
    direction = torch.tensor(
        [0.1, -0.2, 0.05, 0.03, -0.04, 0.02, -0.1, 0.05, 0.08, -0.02, 0.01, 0.04]
    )
    epsilon = 1.0e-4
    reference = int((poses[:, :3, 3] - poses[0, :3, 3]).norm(dim=-1).argmax())
    for side in ("left", "right"):
        solver = BlockSparseSphericalBA(gauge_mode="initial_baseline", pose_update_side=side)
        row = solver._baseline_gauge_jacobian(poses, poses)
        assert row is not None
        updated = poses.clone()
        for frame in range(1, 3):
            delta = epsilon * direction[(frame - 1) * 6 : frame * 6]
            updated[frame] = (
                poses[frame] @ se3_exp(delta)
                if side == "right"
                else se3_exp(delta) @ poses[frame]
            )
        before = (poses[reference, :3, 3] - poses[0, :3, 3]).norm()
        after = (updated[reference, :3, 3] - updated[0, :3, 3]).norm()
        finite_difference = (after - before) / epsilon
        torch.testing.assert_close(finite_difference, row @ direction, atol=5e-4, rtol=2e-3)


def test_right_local_rotation_keeps_camera_center_fixed() -> None:
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.8, -0.2, 0.3])
    rotation_only = torch.tensor([0.0, 0.0, 0.0, 0.1, -0.05, 0.02])
    right_updated = pose @ se3_exp(rotation_only)
    left_updated = se3_exp(rotation_only) @ pose
    torch.testing.assert_close(right_updated[:3, 3], pose[:3, 3])
    assert not torch.allclose(left_updated[:3, 3], pose[:3, 3])


def test_rotation_only_ba_requires_right_local_updates() -> None:
    with pytest.raises(ValueError, match="requires pose_update_side='right'"):
        BlockSparseSphericalBA(pose_update_side="left", pose_dof_mode="rotation_only")
    with pytest.raises(ValueError, match="requires pose_update_side='right'"):
        BlockSparseSphericalBA(pose_update_side="left", pose_dof_mode="translation_only")


def test_block_sparse_ba_recovers_known_pose_and_strictly_decreases_objective() -> None:
    height, width = 6, 12
    query_count = height * width
    poses_truth = torch.eye(4).repeat(2, 1, 1)
    poses_truth[1] = se3_exp(torch.tensor([0.20, -0.03, 0.02, 0.01, 0.08, -0.02]))
    depth = torch.linspace(1.5, 5.0, query_count).reshape(1, 1, height, width).repeat(2, 1, 1, 1)
    rays = build_erp_ray_grid(height, width).reshape(query_count, 3)
    source_ray = rays.view(1, 1, query_count, 3).repeat(1, 2, 1, 1)
    source_uv = torch.stack(
        torch.meshgrid(
            torch.arange(width).float() + 0.5,
            torch.arange(height).float() + 0.5,
            indexing="ij",
        ),
        dim=-1,
    ).permute(1, 0, 2).reshape(query_count, 2).view(1, 1, query_count, 2).repeat(1, 2, 1, 1)
    edges = all_directed_pairs(2)
    target_rays = []
    for source, target in edges.tolist():
        source_depth = depth[source, 0].reshape(-1)
        source_point = source_depth[:, None] * rays
        world_point = torch.einsum("ij,nj->ni", poses_truth[source, :3, :3], source_point) + poses_truth[source, :3, 3]
        target_point = torch.einsum(
            "ij,nj->ni",
            poses_truth[target, :3, :3].T,
            world_point - poses_truth[target, :3, 3],
        )
        target_rays.append(torch.nn.functional.normalize(target_point, dim=-1))
    target_ray = torch.stack(target_rays).unsqueeze(0)
    cache = Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=depth[:, 0].reshape(1, 2, query_count),
        source_valid=torch.ones(1, 2, query_count, dtype=torch.bool),
        edges=edges,
        target_uv=torch.zeros(1, 2, query_count, 2),
        target_ray=target_ray,
        top1_cosine=torch.ones(1, 2, query_count),
        top2_margin=torch.ones(1, 2, query_count),
        entropy=torch.zeros(1, 2, query_count),
        valid_mask=torch.ones(1, 2, query_count, dtype=torch.bool),
        factor_weight=torch.ones(1, 2, query_count),
    )
    poses_initial = poses_truth.clone()
    poses_initial[1] = se3_exp(torch.tensor([0.04, -0.02, 0.01, 0.02, -0.03, 0.015])) @ poses_initial[1]
    output = BlockSparseSphericalBA(
        iterations=8,
        damping=1e-4,
        huber_delta_deg=5.0,
        pose_prior_weight=1e-6,
        depth_prior_weight=1e-4,
        max_pose_update_deg=10.0,
        max_translation_update=0.1,
        min_factors=8,
        min_affine_support=8,
        factor_chunk_size=128,
        residual_worse_tolerance=1.0,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        lm_max_trials=8,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert bool(output.accepted[0])
    diagnostics = output.diagnostics[0]
    assert diagnostics["final_objective"] < diagnostics["initial_objective"]
    assert diagnostics["accepted_steps"] > 0
    assert diagnostics["gain_ratio_mean"] > 0.0
    assert diagnostics["final_median_residual_deg"] < 1.0e-3
    torch.testing.assert_close(
        output.poses_c2w[0, 1, :3, 3],
        poses_truth[1, :3, 3],
        atol=2e-4,
        rtol=0.0,
    )

    gauged_output = BlockSparseSphericalBA(
        iterations=8,
        damping=1e-4,
        huber_delta_deg=5.0,
        pose_prior_weight=1e-6,
        depth_prior_weight=1e-4,
        max_pose_update_deg=10.0,
        max_translation_update=0.1,
        min_factors=8,
        min_affine_support=8,
        factor_chunk_size=128,
        residual_worse_tolerance=1.0,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        gauge_mode="initial_baseline",
        lm_max_trials=8,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert bool(gauged_output.accepted[0])
    gauged_diagnostics = gauged_output.diagnostics[0]
    assert gauged_diagnostics["final_objective"] < gauged_diagnostics["initial_objective"]
    assert gauged_diagnostics["gain_ratio_mean"] > 0.0
    initial_baseline = (poses_initial[1, :3, 3] - poses_initial[0, :3, 3]).norm()
    output_baseline = (
        gauged_output.poses_c2w[0, 1, :3, 3] - gauged_output.poses_c2w[0, 0, :3, 3]
    ).norm()
    torch.testing.assert_close(output_baseline, initial_baseline, atol=2e-6, rtol=0.0)

    right_output = BlockSparseSphericalBA(
        iterations=8,
        damping=1e-4,
        huber_delta_deg=5.0,
        pose_prior_weight=1e-6,
        depth_prior_weight=1e-4,
        max_pose_update_deg=10.0,
        max_translation_update=0.1,
        min_factors=8,
        min_affine_support=8,
        factor_chunk_size=128,
        residual_worse_tolerance=1.0,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        gauge_mode="initial_baseline",
        pose_update_side="right",
        lm_max_trials=8,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert bool(right_output.accepted[0])
    assert right_output.diagnostics[0]["final_objective"] < right_output.diagnostics[0]["initial_objective"]
    right_baseline = (right_output.poses_c2w[0, 1, :3, 3] - right_output.poses_c2w[0, 0, :3, 3]).norm()
    torch.testing.assert_close(right_baseline, initial_baseline, atol=2e-6, rtol=0.0)

    rotation_only_output = BlockSparseSphericalBA(
        iterations=8,
        damping=1e-4,
        huber_delta_deg=5.0,
        pose_prior_weight=1e-6,
        depth_prior_weight=1e-4,
        max_pose_update_deg=10.0,
        min_factors=8,
        min_affine_support=8,
        factor_chunk_size=128,
        residual_worse_tolerance=1.0,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        gauge_mode="initial_baseline",
        pose_update_side="right",
        pose_dof_mode="rotation_only",
        lm_max_trials=8,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert bool(rotation_only_output.accepted[0])
    assert rotation_only_output.diagnostics[0]["final_objective"] < rotation_only_output.diagnostics[0]["initial_objective"]
    torch.testing.assert_close(
        rotation_only_output.poses_c2w[0, :, :3, 3],
        poses_initial[:, :3, 3],
        atol=1e-7,
        rtol=0.0,
    )

    translation_only_output = BlockSparseSphericalBA(
        iterations=8,
        damping=1e-4,
        huber_delta_deg=5.0,
        pose_prior_weight=1e-6,
        depth_prior_weight=1e-4,
        max_translation_update=0.1,
        min_factors=8,
        min_affine_support=8,
        factor_chunk_size=128,
        residual_worse_tolerance=1.0,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        gauge_mode="initial_baseline",
        pose_update_side="right",
        pose_dof_mode="translation_only",
        lm_max_trials=8,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert bool(translation_only_output.accepted[0])
    assert translation_only_output.diagnostics[0]["final_objective"] < translation_only_output.diagnostics[0]["initial_objective"]
    torch.testing.assert_close(
        translation_only_output.poses_c2w[0, :, :3, :3],
        poses_initial[:, :3, :3],
        atol=1e-7,
        rtol=0.0,
    )

    gated_output = BlockSparseSphericalBA(
        iterations=8,
        min_factors=8,
        factor_chunk_size=128,
        solver_mode="standard_lm",
        dense_depth_mode="none",
        min_initial_median_residual_deg=180.0,
    )(poses_initial.unsqueeze(0), depth.unsqueeze(0), cache)
    assert not bool(gated_output.accepted[0])
    assert gated_output.diagnostics[0]["reason"] == "below_min_initial_median_residual"
    torch.testing.assert_close(gated_output.poses_c2w[0], poses_initial)
    torch.testing.assert_close(
        gated_output.initial_median_residual_deg,
        gated_output.final_median_residual_deg,
    )


def test_block_sparse_ba_rejects_antipodal_factor() -> None:
    observation, _, _ = _observation(views=2)
    source_uv = torch.tensor([[[[0.5, 4.5]], [[0.5, 4.5]]]])
    source_ray = torch.tensor([[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]])
    cache = Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=torch.full((1, 2, 1), 2.0),
        source_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        edges=torch.tensor([[0, 1]]),
        target_uv=torch.tensor([[[[8.5, 4.5]]]]),
        target_ray=torch.tensor([[[[0.0, 0.0, -1.0]]]]),
        top1_cosine=torch.ones(1, 1, 1),
        top2_margin=torch.ones(1, 1, 1),
        entropy=torch.zeros(1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    output = BlockSparseSphericalBA(iterations=1, min_factors=1, min_affine_support=1)(
        observation.poses_c2w,
        observation.refined_depth,
        cache,
    )
    assert not bool(output.accepted[0])
    assert output.diagnostics[0]["reason"] == "insufficient_non_antipodal_factors"


def test_block_sparse_ba_rejects_zero_parallax_factor_when_gated() -> None:
    observation, _, _ = _observation(views=2)
    poses = observation.poses_c2w.clone()
    poses[:, 1] = poses[:, 0]
    source_uv = torch.tensor([[[[0.5, 4.5]], [[0.5, 4.5]]]])
    source_ray = torch.tensor([[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]])
    cache = Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=torch.full((1, 2, 1), 2.0),
        source_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        edges=torch.tensor([[0, 1]]),
        target_uv=torch.tensor([[[[0.5, 4.5]]]]),
        target_ray=source_ray[:, :1].clone(),
        top1_cosine=torch.ones(1, 1, 1),
        top2_margin=torch.ones(1, 1, 1),
        entropy=torch.zeros(1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        factor_weight=torch.ones(1, 1, 1),
    )
    output = BlockSparseSphericalBA(
        iterations=1,
        min_factors=1,
        min_affine_support=1,
        min_parallax_deg=1.0,
    )(poses, observation.refined_depth, cache)
    assert not bool(output.accepted[0])
    assert output.diagnostics[0]["reason"] == "insufficient_geometry_gated_factors"


def test_refiner_zero_initialization_is_identity_and_backward_is_finite() -> None:
    observation, feature, image = _observation()
    refiner = SphericalRecurrentGaussianRefiner()
    output = refiner(
        observation,
        observation,
        feature,
        image,
        torch.randn(1, 3, 32, 8, 16),
        iteration_index=0,
    )
    torch.testing.assert_close(output.observation.refined_depth, observation.refined_depth)
    torch.testing.assert_close(output.observation.local_quaternion, observation.local_quaternion)
    torch.testing.assert_close(output.observation.rgb_sh, observation.rgb_sh)
    loss = output.observation.rgb_sh.mean() + output.observation.refined_depth.mean()
    loss.backward()
    gradients = [parameter.grad for parameter in refiner.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_refiner_geometry_gate_freezes_far_pixels_but_updates_appearance() -> None:
    observation, feature, image = _observation(views=2)
    far = observation.with_updates(refined_depth=torch.full_like(observation.refined_depth, 25.0))
    refiner = SphericalRecurrentGaussianRefiner()
    with torch.no_grad():
        refiner.geometry_head[-1].bias.fill_(1.0)
        refiner.appearance_head[-1].bias.fill_(1.0)
    output = refiner(far, observation, feature, image, torch.zeros(1, 2, 32, 8, 16), iteration_index=0)
    torch.testing.assert_close(output.observation.refined_depth, far.refined_depth)
    torch.testing.assert_close(output.observation.local_quaternion, far.local_quaternion)
    torch.testing.assert_close(output.observation.log_scale_multiplier, far.log_scale_multiplier)
    assert not torch.equal(output.observation.rgb_sh, far.rgb_sh)
    assert not torch.equal(output.observation.density_sh, far.density_sh)


def test_quaternion_log_exp_round_trip() -> None:
    rotation = torch.tensor([[0.1, -0.2, 0.05]])
    reconstructed = quaternion_log_map(quaternion_exp_map(rotation))
    torch.testing.assert_close(reconstructed, rotation, atol=1e-6, rtol=1e-5)


def test_error_encoder_and_router_shapes_without_optional_resnet() -> None:
    observation, _, image = _observation()
    encoder = ReSplatErrorEncoder(use_resnet=False)
    flat = image.reshape(3, 3, 8, 16)
    reference = encoder.encode_reference(flat)
    error = encoder(flat * 0.9, reference).reshape(1, 3, 32, 2, 4)
    router = SphericalErrorRouter()
    routed = router(
        observation,
        error,
        torch.full((1, 3, 1, 8, 16), 2.0),
        torch.ones(1, 3, 1, 8, 16),
    )
    assert routed.shape == (1, 3, 32, 8, 16)
    assert bool(torch.isfinite(routed).all())


def test_support_map_has_floor_and_query_peaks() -> None:
    observation, feature, _ = _observation()
    cache = build_stage3_match_cache(feature, observation.refined_depth, num_queries=2, query_chunk_size=1)
    support = build_ba_support_map(cache, height=8, width=16, floor=0.1, dilation_kernel=1)
    assert abs(float(support.min()) - 0.1) < 1.0e-6
    assert float(support.max()) == 1.0


def test_zero_dssim_weight_skips_dssim_computation() -> None:
    rendered = torch.rand(1, 2, 3, 4, 8)
    target = torch.rand_like(rendered)
    with patch(
        "losses.spherical_stage3_refinement_loss.spherical_dssim",
        side_effect=AssertionError("DSSIM must not run when its weight is zero."),
    ):
        loss, metrics = leave_one_out_render_loss(rendered, target, dssim_weight=0.0)
    assert bool(torch.isfinite(loss))
    assert float(metrics["dssim"]) == 0.0


def test_ba_outer_schedule_is_ba0_only_by_default() -> None:
    config = default_config()
    assert _ba_outer_schedule(config) == (True, False, False)


def test_ba_ablation_generator_changes_only_declared_solver_axes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    suite_dir = tmp_path / "suite"
    manifest = generate_ablation_configs(
        root / "configs" / "stage3_spherical_ba_recurrent_refiner_omni360.yaml",
        suite_dir,
    )
    assert len(manifest["experiments"]) == len(EXPERIMENTS)
    for experiment in manifest["experiments"]:
        config_path = Path(experiment["config"])
        assert config_path.is_file()
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["ba"]["outer_schedule"] == [True, False, False]
        assert config["loss"]["dssim"] == 0.0
        assert config["train"]["max_steps"] == 200
        assert config["WeightsAndBiases"]["enabled"] is True
        assert config["Visualization"]["enabled"] is True


def test_ba_gate_sweep_generator_keeps_solver_fixed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    suite_dir = tmp_path / "gate_sweep"
    manifest = generate_gate_sweep(
        root / "configs" / "stage3_spherical_ba_recurrent_refiner_omni360.yaml",
        suite_dir,
    )
    assert len(manifest["variants"]) == len(VARIANTS)
    import yaml

    for variant in manifest["variants"]:
        config = yaml.safe_load(Path(variant["config"]).read_text(encoding="utf-8"))
        assert config["ba"]["dense_depth_mode"] == "none"
        assert config["ba"]["gauge_mode"] == "initial_baseline"
        assert config["ba"]["solver_mode"] == "standard_lm"
        assert config["matching"]["reliability_keep_fraction"] == variant["keep"]

    parallax_suite = tmp_path / "parallax_sweep"
    parallax_manifest = generate_gate_sweep(
        root / "configs" / "stage3_spherical_ba_recurrent_refiner_omni360.yaml",
        parallax_suite,
        variants=HIGH_PARALLAX_VARIANTS,
    )
    assert len(parallax_manifest["variants"]) == len(HIGH_PARALLAX_VARIANTS)
    assert all(float(variant["parallax"]) >= 2.0 for variant in parallax_manifest["variants"])

    trust_suite = tmp_path / "trust_sweep"
    trust_manifest = generate_gate_sweep(
        root / "configs" / "stage3_spherical_ba_recurrent_refiner_omni360.yaml",
        trust_suite,
        variants=TRUST_REGION_VARIANTS,
    )
    assert len(trust_manifest["variants"]) == len(TRUST_REGION_VARIANTS)
    assert any(variant.get("side") == "right" for variant in trust_manifest["variants"])


def test_ba_ablation_summary_uses_validation_snapshots(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "global_step": 200,
            "training_config": {
                "ba": {
                    "dense_depth_mode": "none",
                    "gauge_mode": "initial_baseline",
                    "solver_mode": "standard_lm",
                }
            },
            "metrics": {
                "val/initial/pose_scale_aligned_ate": 0.10,
                "val/ba0/pose_scale_aligned_ate": 0.08,
                "val/refine3/pose_scale_aligned_ate": 0.08,
                "val/initial/loo_psnr": 10.0,
                "val/ba0/loo_psnr": 10.2,
                "val/refine3/loo_psnr": 11.0,
            },
        },
        checkpoint,
    )
    row = summarize_checkpoint(checkpoint, name="combined")
    assert row["step"] == 200
    assert row["dense_depth_mode"] == "none"
    assert abs(row["ba0_delta_pose_scale_aligned_ate"] + 0.02) < 1.0e-8
    assert abs(row["ba0_delta_loo_psnr"] - 0.2) < 1.0e-8


def test_synthetic_ba_only_evaluator() -> None:
    config = default_config()
    config["stage3"]["enabled"] = True
    config["matching"].update(
        {
            "num_queries": 2,
            "query_chunk_size": 1,
            "forward_backward": False,
            "min_factor_weight": 0.0,
        }
    )
    config["ba"].update(
        {
            "iterations": 0,
            "min_factors": 1,
            "min_affine_support": 2,
            "factor_chunk_size": 4,
        }
    )
    config["dataset"]["max_val_samples"] = 1
    config["train"].update(
        {
            "batch_size": 1,
            "num_workers": 0,
            "feature_device": "cpu",
            "train_device": "cpu",
            "amp": False,
        }
    )
    result = evaluate(config, max_batches=1)
    assert result["format"] == "spherical_stage3_ba_evaluation_v1"
    assert result["num_batches"] == 1
    assert "delta/pose_scale_aligned_ate" in result["mean"]
    assert result["records"][0]["metrics"]["matching/valid_factors"] > 0


def test_pose_metric_is_zero_for_identical_poses() -> None:
    observation, _, _ = _observation()
    metrics = aligned_pose_metrics(observation.poses_c2w[0], observation.poses_c2w[0])
    assert metrics["rotation_mean_deg"] == 0.0
    assert metrics["scale_aligned_ate"] < 1.0e-7


def test_stage3_modules_do_not_import_forbidden_frontends() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "models/spherical_selfi_stage3_ba.py",
            "models/spherical_recurrent_gaussian_refiner.py",
            "training/train_spherical_ba_recurrent_refiner.py",
        )
    ).lower()
    for forbidden in ("anchor_splat", "point_transformer", "voxel_compactor", "recurrent_updater", "rae"):
        assert forbidden not in source


def test_synthetic_stage3_one_step_and_checkpoint(tmp_path: Path) -> None:
    config = default_config()
    config["stage3"]["enabled"] = True
    config["matching"].update({"num_queries": 2, "query_chunk_size": 1})
    config["ba"].update({"iterations": 0, "min_factors": 1, "min_affine_support": 2, "factor_chunk_size": 4})
    config["dataset"]["max_train_samples"] = 1
    config["train"].update(
        {
            "max_steps": 1,
            "save_interval": 1,
            "diagnostics_interval": 100,
            "val_interval": 100,
            "output_dir": str(tmp_path / "stage3"),
        }
    )
    result = train(config)
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format"] == "spherical_ba_recurrent_gaussian_refiner_v1"
    assert payload["global_step"] == 1
    assert "model" in payload and "optimizer" in payload and "scheduler" in payload
