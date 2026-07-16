from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from backend.pano_gs.mapper import MapperObservation, PanoGaussianMap, PanoGaussianMapper
from backend.pano_gs.sim3_graph import (
    CoincidentPanoramaFactor,
    DenseSphericalFactorBlock,
    GlobalSim3FactorGraph,
    Sim3GraphEdge,
    s2_log_tangent_coordinates,
)
from backend.pano_gs.spherical_selfi_global import SphericalSelfiGlobalBackend
from backend.pano_gs.stage2_global_fusion import (
    Stage2GlobalMapFusion,
    rotate_sh_coefficients,
)
from frontend.pano_droid.spherical_ba import se3_exp
from frontend.pano_droid.interfaces import PanoFrame
from frontend.spherical_selfi.panorama_loop import (
    PanoramaLoopDetector,
    PanoramaLoopVerification,
    circular_yaw_shift,
    spherical_rotation_ransac,
)
from frontend.spherical_selfi.runtime import (
    SphericalSelfiWindowFrontend,
    _split_stage3_cache_for_validation,
)
from frontend.spherical_selfi.window_packet import (
    BoundaryMatchBlock,
    LocalGaussianWindowPacket,
    build_panorama_retrieval_descriptor,
)
from geometry.sim3 import (
    apply_sim3,
    apply_sim3_to_c2w,
    sim3_exp,
    sim3_from_components,
    sim3_identity,
    sim3_inverse,
    sim3_log,
    sim3_components,
)
from geometry.panorama_loop_contracts import (
    DenseSphericalLoopMeasurement,
    LoopPoseMeasurement,
)
from geometry.spherical_erp import erp_pixel_to_unit_ray
from models.per_pixel_gaussian_observation import real_sh_basis
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead
from models.spherical_selfi_stage3_ba import Stage3MatchCache
from models.spherical_voxel_anchor_refiner import (
    VoxelAnchorConfig,
    voxelize_per_pixel_gaussians,
)
from training.train_spherical_selfi_gaussian_head import default_config as stage2_default_config


def _observation(
    poses: torch.Tensor,
    frame_ids: tuple[int, ...],
    *,
    height: int = 6,
    width: int = 12,
    feature_by_frame: dict[int, torch.Tensor] | None = None,
):
    torch.manual_seed(11)
    views = len(frame_ids)
    feature = torch.stack(
        [
            feature_by_frame[int(frame_id)]
            if feature_by_frame is not None and int(frame_id) in feature_by_frame
            else torch.randn(24, height, width)
            for frame_id in frame_ids
        ],
        dim=0,
    ).unsqueeze(0)
    image = torch.rand(1, views, 3, height, width)
    depth = torch.full((1, views, 1, height, width), 2.0)
    head = SphericalSelfiGaussianHead(channels=(8, 12, 16, 24), mlp_hidden_dim=12)
    observation = head(
        feature,
        image,
        depth,
        poses.unsqueeze(0),
        frame_ids=torch.tensor([frame_ids]),
    )
    return observation, feature


def _packet(
    window_id: int,
    poses: torch.Tensor,
    frame_ids: tuple[int, ...],
    *,
    feature_by_frame: dict[int, torch.Tensor] | None = None,
) -> LocalGaussianWindowPacket:
    observation, feature = _observation(poses, frame_ids, feature_by_frame=feature_by_frame)
    return LocalGaussianWindowPacket.from_observation(
        window_id=window_id,
        observation=observation,
        adapter_features=feature,
        frame_ids=frame_ids,
        verification_size=feature.shape[-2:],
    )


def _refined_packet(
    window_id: int,
    poses: torch.Tensor,
    frame_ids: tuple[int, ...],
    *,
    feature_by_frame: dict[int, torch.Tensor] | None = None,
) -> LocalGaussianWindowPacket:
    packet = _packet(
        window_id,
        poses,
        frame_ids,
        feature_by_frame=feature_by_frame,
    )
    anchor_config = VoxelAnchorConfig(
        use_resnet_error=False,
        pretrained_resnet=False,
    )
    images = torch.zeros(
        1,
        len(frame_ids),
        3,
        *packet.observation.image_size,
        device=packet.observation.refined_depth.device,
        dtype=packet.observation.refined_depth.dtype,
    )
    packet.anchor_observation = voxelize_per_pixel_gaussians(
        packet.observation,
        packet.adapter_features,
        images,
        anchor_config,
        valid_mask=packet.finite_gaussian_mask,
    ).detach_for_backend()
    packet.metadata["voxel_anchor_refiner_enabled"] = True
    packet.metadata["voxel_anchor_count"] = packet.anchor_observation.num_anchors
    return packet


class _SyntheticSharedDepthRenderer:
    def __init__(
        self,
        *,
        local_depth: float,
        global_depth: float,
        alpha: float = 1.0,
        fail_on_call: int | None = None,
    ):
        self.local_depth = float(local_depth)
        self.global_depth = float(global_depth)
        self.alpha = float(alpha)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def render_cameras(self, cameras, gaussians):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == int(self.fail_on_call):
            raise RuntimeError("synthetic renderer failure")
        camera = cameras[0]
        count = int(gaussians.get_xyz.shape[0])
        is_local_anchor = hasattr(gaussians, "anchor_indices")
        depth_value = self.local_depth if is_local_anchor else self.global_depth
        device, dtype = gaussians.get_xyz.device, gaussians.get_xyz.dtype
        depth = torch.full(
            (1, 1, int(camera.image_height), int(camera.image_width)),
            depth_value,
            device=device,
            dtype=dtype,
        )
        alpha = torch.full_like(depth, self.alpha)
        return {
            "depth": depth,
            "alpha": alpha,
            "visibility_filter": torch.ones(
                1,
                count,
                device=device,
                dtype=torch.bool,
            ),
        }


def test_spherical_selfi_global_config_uses_confirmed_two_stage_and_backend_rates() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "spherical_selfi_global_gs_slam.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    local_ba = config["SphericalSelfiRuntime"]["local_ba"]
    outlier = local_ba["outlier_refinement"]
    map_optimization = config["SphericalSelfiGlobalBackend"]["map_optimization"]
    graph = config["SphericalSelfiGlobalBackend"]["global_graph"]
    global_backend = config["SphericalSelfiGlobalBackend"]

    assert local_ba["iterations"] == 5
    assert local_ba["lm_max_trials"] == 8
    assert local_ba["lm_acceptance_eta"] == 1.0e-6
    assert local_ba["residual_worse_tolerance"] == 1.05
    assert local_ba["max_pose_update_deg"] == 10.0
    assert local_ba["max_translation_update"] == 0.10
    assert local_ba["max_logdepth_update"] == 0.70
    assert local_ba["min_factors"] == 128
    assert local_ba["pose_safe_two_stage"] is True
    assert local_ba["defer_dense_depth_affine"] is True
    assert local_ba["dense_depth_mode"] == "shift"
    assert local_ba["dense_depth_output_floor"] == 0.01
    assert outlier["second_stage_iterations"] == 10
    assert outlier["angular_max_deg"] == 5.0
    assert outlier["sim3_max_relative_depth"] == 0.05
    assert outlier["min_inliers"] == 128
    assert outlier["min_inlier_ratio"] == 0.20
    assert outlier["validation_stride"] == 5
    assert outlier["validation_min_inliers"] == 32
    assert outlier["validation_residual_worse_tolerance"] == 1.0
    assert outlier["validation_sim3_worse_tolerance"] == 1.05
    assert local_ba["matching"]["factor_weight_mode"] == "fibonacci_equal"
    assert graph["optimization_start_nodes"] == 6
    assert graph["optimization_interval_edges"] == 5
    assert graph["normalize_dense_information_by_count"] is True
    assert graph["analytic_dense_linearization"] is True
    assert graph["restrict_objective_to_active_factors"] is True
    assert global_backend["hierarchical_submaps"]["enabled"] is True
    assert global_backend["hierarchical_submaps"]["local_camera_model"] == "se3_shared_scale"
    assert global_backend["hierarchical_submaps"]["compress_frozen_dense_factors"] is True
    assert global_backend["loop_closure"]["descriptor"]["mode"] == "so3_sh_gram"
    assert global_backend["loop_closure"]["insert_pose_factor"] is True
    assert global_backend["loop_closure"]["verification"]["mode"] == "spherical_so3"
    assert global_backend["keyframe_selection"]["enabled"] is True
    assert global_backend["rendered_overlap_alignment"] == {
        "enabled": True,
        "mode": "shared_frame_scale_only",
        "min_points": 256,
        "max_points": 4096,
        "alpha_threshold": 0.05,
        "min_inlier_ratio": 0.35,
        "max_median_relative_error": 0.10,
        "max_scale_change": 2.5,
        "failure_policy": "error",
    }
    assert global_backend["insertion_dedup"] == {
        "enabled": True,
        "visible_only": True,
        "same_level_only": True,
        "radius_voxels": 1.0,
        "compare_existing_only": True,
        "permanent_drop": True,
        "update_existing_statistics": True,
    }
    assert map_optimization["lazy_submap_transforms"]["enabled"] is True
    assert map_optimization["loop_neighborhood_refinement"] is True
    assert map_optimization["loop_seam_deduplication"] is True
    assert map_optimization["extra_steps_on_loop"] == 20
    assert map_optimization["pose_lr"] == 2.0e-4
    assert map_optimization["pose_refine_enable"] is True
    assert map_optimization["separate_gaussian_lrs"] is True
    assert map_optimization["scale_gaussian_parameter_updates"] is True
    assert {
        name: map_optimization[name]
        for name in (
            "xyz_lr",
            "feature_lr",
            "sh_rest_lr",
            "opacity_lr",
            "scaling_lr",
            "rotation_lr",
        )
    } == {
        "xyz_lr": 5.0e-4,
        "feature_lr": 2.0e-3,
        "sh_rest_lr": 1.0e-4,
        "opacity_lr": 1.0e-3,
        "scaling_lr": 1.0e-4,
        "rotation_lr": 1.0e-4,
    }


def test_new_so3_hierarchical_features_are_default_off_for_legacy_configs() -> None:
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        config={"enabled": True},
    )

    assert backend.loop_detector.descriptor_mode == "latitude_bands"
    assert not backend.loop_detector.so3_verification
    assert not backend.hierarchical_submaps_enabled
    assert backend.submap_graph is None
    assert backend.robust_loop_mode == "off"
    assert not backend.loop_transaction_enabled
    assert not backend.lazy_submap_transforms_enabled
    assert not backend.insert_loop_pose_factor
    assert not backend.graph.analytic_dense_linearization
    assert not backend.graph.restrict_objective_to_active_factors
    assert not backend.normalize_dense_information_by_count
    assert not backend.loop_neighborhood_refinement_enabled
    assert not backend.loop_seam_dedup_enabled


def test_spherical_keyframe_policy_combines_gap_descriptor_coverage_and_parallax() -> None:
    frontend = object.__new__(SphericalSelfiWindowFrontend)
    frontend.spherical_keyframe_selection_enabled = True
    frontend.keyframe_min_gap = 2
    frontend.keyframe_max_gap = 5
    frontend.keyframe_score_threshold = 0.30
    frontend.keyframe_descriptor_weight = 0.35
    frontend.keyframe_coverage_weight = 0.20
    frontend.keyframe_parallax_weight = 0.30
    frontend.keyframe_residual_weight = 0.15
    frontend.keyframe_translation_ratio = 0.05
    frontend.keyframe_rotation_deg = 10.0
    frontend._keyframe_decisions = {}
    frontend._last_keyframe_id = None
    frontend._last_keyframe_descriptor = None
    frontend._last_keyframe_pose = None
    frontend._last_keyframe_coverage = 0.0
    valid = torch.ones(1, 4, 8, dtype=torch.bool)
    sky = torch.zeros_like(valid)
    confidence = torch.ones(1, 4, 8) * 0.9
    depth = torch.ones(1, 4, 8) * 2.0
    first_pose = torch.eye(4)
    descriptor = torch.nn.functional.normalize(torch.randn(32), dim=0)

    first = frontend._spherical_keyframe_decision(
        frame_id=0,
        descriptor=descriptor,
        pose_c2w=first_pose,
        valid_mask=valid,
        sky_mask=sky,
        confidence=confidence,
        depth=depth,
        ba_residual_deg=0.1,
    )
    too_close = frontend._spherical_keyframe_decision(
        frame_id=1,
        descriptor=-descriptor,
        pose_c2w=first_pose,
        valid_mask=valid,
        sky_mask=sky,
        confidence=confidence,
        depth=depth,
        ba_residual_deg=3.0,
    )
    moved_pose = first_pose.clone()
    moved_pose[0, 3] = 0.2
    moved = frontend._spherical_keyframe_decision(
        frame_id=2,
        descriptor=-descriptor,
        pose_c2w=moved_pose,
        valid_mask=valid,
        sky_mask=sky,
        confidence=confidence,
        depth=depth,
        ba_residual_deg=3.0,
    )

    assert first[0]
    assert not too_close[0]
    assert moved[0]
    assert moved[1] > frontend.keyframe_score_threshold


def test_stage2_validation_split_is_deterministic_disjoint_and_complete() -> None:
    queries = 20
    edges = torch.tensor([[0, 1], [1, 0]])
    valid = torch.ones(1, 2, queries, dtype=torch.bool)
    cache = Stage3MatchCache(
        source_uv=torch.zeros(1, 2, queries, 2),
        source_ray=torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3).repeat(
            1, 2, queries, 1
        ),
        source_depth=torch.ones(1, 2, queries),
        source_valid=torch.ones(1, 2, queries, dtype=torch.bool),
        edges=edges,
        target_uv=torch.zeros(1, 2, queries, 2),
        target_ray=torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3).repeat(
            1, 2, queries, 1
        ),
        top1_cosine=torch.ones(1, 2, queries),
        top2_margin=torch.ones(1, 2, queries),
        entropy=torch.zeros(1, 2, queries),
        valid_mask=valid,
        factor_weight=torch.ones(1, 2, queries),
    )

    training, validation = _split_stage3_cache_for_validation(cache, stride=5)

    assert not bool((training.valid_mask & validation.valid_mask).any())
    assert torch.equal(training.valid_mask | validation.valid_mask, valid)
    assert torch.equal(
        validation.valid_mask[0, 0],
        torch.arange(queries).remainder(5).eq(0),
    )
    assert training.metadata["factor_split"] == "stage2_training"
    assert validation.metadata["factor_split"] == "stage2_validation"


def test_sim3_exp_log_round_trip_and_graph_scale_recovery() -> None:
    tangent = torch.tensor([0.3, -0.2, 0.1, 0.03, -0.04, 0.02, math.log(1.2)])
    truth = sim3_exp(tangent)
    torch.testing.assert_close(sim3_log(truth), tangent, atol=2e-5, rtol=2e-5)

    graph = GlobalSim3FactorGraph(max_iterations=10, pcg_iterations=32)
    graph.add_node(0, sim3_identity())
    graph.add_node(1, sim3_exp(tangent + torch.tensor([0.15, 0.0, 0.0, 0.0, 0.02, 0.0, 0.1])))
    graph.add_edge(
        Sim3GraphEdge(
            source=0,
            target=1,
            measurement_target_to_source=truth,
            information_diag=torch.ones(7),
        )
    )
    result = graph.optimize()
    assert result.accepted
    assert result.final_objective < result.initial_objective
    torch.testing.assert_close(graph.transform(1), truth, atol=2e-4, rtol=2e-4)


def test_sim3_log_identity_jacobians_are_finite() -> None:
    zero = torch.zeros(7, dtype=torch.float64)

    def log_after_update(delta: torch.Tensor) -> torch.Tensor:
        return sim3_log(sim3_exp(delta))

    for jacobian in (
        torch.func.jacfwd(log_after_update)(zero),
        torch.func.jacrev(log_after_update)(zero),
    ):
        assert torch.isfinite(jacobian).all()
        torch.testing.assert_close(jacobian, torch.eye(7, dtype=zero.dtype), atol=2e-6, rtol=2e-6)


def test_identity_graph_factor_linearization_is_finite() -> None:
    graph = GlobalSim3FactorGraph(max_iterations=2)
    graph.add_node(0, sim3_identity(dtype=torch.float64))
    graph.add_node(1, sim3_identity(dtype=torch.float64))
    factor = Sim3GraphEdge(
        source=0,
        target=1,
        measurement_target_to_source=sim3_identity(dtype=torch.float64),
        information_diag=torch.ones(7, dtype=torch.float64),
    )
    graph.add_edge(factor)
    _, blocks, residual = graph._linearize_factor(factor, {1: 0})
    assert torch.isfinite(residual).all()
    assert blocks and torch.isfinite(blocks[0]).all()
    result = graph.optimize()
    assert result.reason == "converged_gradient"


def test_coincident_panorama_factor_corrects_center_rotation_without_scale() -> None:
    truth_rotation = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.35, 0.0]))[:3, :3]
    initial_rotation = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.10, 0.0]))[:3, :3]
    graph = GlobalSim3FactorGraph(max_iterations=12, pcg_iterations=48)
    graph.add_node(0, sim3_identity())
    graph.add_node(
        1,
        sim3_from_components(1.4, initial_rotation, torch.tensor([0.3, -0.2, 0.1])),
    )
    graph.add_edge(
        CoincidentPanoramaFactor(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            measured_source_to_target_rotation=truth_rotation,
            center_weight=10.0,
            rotation_weight=10.0,
        )
    )
    result = graph.optimize()
    assert result.final_objective < result.initial_objective
    scale, rotation, translation = sim3_components(graph.transform(1))
    assert abs(float(scale) - 1.4) < 1.0e-5
    assert float(translation.norm()) < 3.0e-3
    torch.testing.assert_close(rotation, truth_rotation, atol=3e-3, rtol=3e-3)


def test_dense_spherical_depth_factor_recovers_window_scale() -> None:
    height, width = 8, 16
    row, column = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    uv = torch.stack([column, row], dim=-1).reshape(-1, 2)[::2]
    bearing = erp_pixel_to_unit_ray(uv, height, width)
    graph = GlobalSim3FactorGraph(max_iterations=12, pcg_iterations=48)
    graph.add_node(0, sim3_identity())
    graph.add_node(1, sim3_from_components(1.25, torch.eye(3), torch.zeros(3)))
    graph.add_edge(
        DenseSphericalFactorBlock(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            source_bearing=bearing,
            target_bearing=bearing,
            source_depth=torch.full((bearing.shape[0],), 2.0),
            target_depth=torch.full((bearing.shape[0],), 1.0),
            factor_weight=torch.ones(bearing.shape[0]),
            depth_factor_weight=1.0,
        )
    )
    result = graph.optimize()
    assert result.accepted
    scale, rotation, translation = sim3_components(graph.transform(1))
    assert abs(float(scale) - 2.0) < 2.0e-3
    torch.testing.assert_close(rotation, torch.eye(3), atol=2e-4, rtol=2e-4)
    assert float(translation.norm()) < 2.0e-4


def test_local_se3_graph_locks_boundary_scale_for_hierarchical_submap() -> None:
    bearing = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    graph = GlobalSim3FactorGraph(
        max_iterations=8,
        pcg_iterations=32,
        lock_scale_updates=True,
    )
    graph.add_node(0, sim3_identity())
    initial = sim3_from_components(1.25, torch.eye(3), torch.zeros(3))
    graph.add_node(1, initial)
    graph.add_edge(
        DenseSphericalFactorBlock(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            source_bearing=bearing,
            target_bearing=bearing,
            source_depth=torch.full((64,), 2.0),
            target_depth=torch.ones(64),
            factor_weight=torch.ones(64),
            depth_factor_weight=1.0,
        )
    )

    graph.optimize()

    scale, _, _ = sim3_components(graph.transform(1))
    assert abs(float(scale) - 1.25) < 1.0e-6


def test_dense_spherical_factor_jacobian_matches_finite_difference() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    count = 32
    source_bearing = torch.nn.functional.normalize(
        torch.randn(count, 3, dtype=dtype), dim=-1
    )
    source_depth = torch.linspace(1.0, 4.0, count, dtype=dtype)
    truth = sim3_exp(
        torch.tensor([0.12, -0.07, 0.04, 0.03, -0.02, 0.01, 0.08], dtype=dtype)
    )
    target_point = apply_sim3(
        sim3_inverse(truth), source_bearing * source_depth[:, None]
    )
    target_depth = target_point.norm(dim=-1)
    factor = DenseSphericalFactorBlock(
        source=0,
        target=1,
        source_local_pose=torch.eye(4, dtype=dtype),
        target_local_pose=torch.eye(4, dtype=dtype),
        source_bearing=source_bearing,
        target_bearing=target_point / target_depth[:, None],
        source_depth=source_depth,
        target_depth=target_depth,
        factor_weight=torch.linspace(0.4, 1.0, count, dtype=dtype),
        depth_factor_weight=0.1,
        s2_huber_delta_deg=10.0,
    )
    source_transform = sim3_identity(dtype=dtype)
    initial = sim3_exp(
        torch.tensor([0.002, -0.001, 0.001, 0.0008, -0.0005, 0.0003, 0.001], dtype=dtype)
    ) @ truth

    def weighted_residual(delta: torch.Tensor) -> torch.Tensor:
        residual, information = GlobalSim3FactorGraph._factor_residual(
            factor,
            source_transform,
            sim3_exp(delta) @ initial,
        )
        return information.sqrt() * residual

    zero = torch.zeros(7, dtype=dtype)
    jacobian = torch.func.jacfwd(weighted_residual)(zero)
    direction = torch.tensor(
        [0.2, -0.3, 0.1, 0.1, 0.15, -0.12, 0.08], dtype=dtype
    )
    direction = direction / direction.norm()
    epsilon = 1.0e-5
    finite_difference = (
        weighted_residual(epsilon * direction)
        - weighted_residual(-epsilon * direction)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(
        jacobian @ direction,
        finite_difference,
        atol=2.0e-6,
        rtol=2.0e-5,
    )


def test_dense_spherical_analytic_normal_equations_match_autodiff() -> None:
    torch.manual_seed(17)
    count = 37
    graph = GlobalSim3FactorGraph(dense_linearization_chunk_size=13)
    graph.add_node(
        0,
        sim3_exp(torch.tensor([0.1, -0.2, 0.05, 0.03, -0.02, 0.04, 0.08])),
    )
    graph.add_node(
        1,
        sim3_exp(torch.tensor([-0.2, 0.1, 0.15, -0.04, 0.05, -0.01, -0.03])),
    )
    source_bearing = torch.nn.functional.normalize(torch.randn(count, 3), dim=-1)
    target_bearing = torch.nn.functional.normalize(torch.randn(count, 3), dim=-1)
    source_pose = torch.eye(4)
    source_pose[:3, 3] = torch.tensor([0.02, 0.01, -0.01])
    target_pose = torch.eye(4)
    target_pose[:3, 3] = torch.tensor([0.1, -0.05, 0.02])
    factor = DenseSphericalFactorBlock(
        source=0,
        target=1,
        source_local_pose=source_pose,
        target_local_pose=target_pose,
        source_bearing=source_bearing,
        target_bearing=target_bearing,
        source_depth=torch.rand(count) + 1.0,
        target_depth=torch.rand(count) + 1.0,
        factor_weight=torch.rand(count) + 0.1,
        depth_factor_weight=0.2,
        s2_huber_delta_deg=5.0,
    )
    graph.add_edge(factor)

    ids, blocks, residual = graph._linearize_factor(factor, {0: 0, 1: 1})
    autodiff_hessian = torch.stack(
        [torch.stack([first.T @ second for second in blocks]) for first in blocks]
    )
    autodiff_gradient = torch.stack([block.T @ residual for block in blocks])
    analytic_ids, analytic_hessian, analytic_gradient = (
        graph._dense_factor_normal_equations(factor, {0: 0, 1: 1})
    )

    assert analytic_ids == ids == [0, 1]
    torch.testing.assert_close(
        analytic_hessian, autodiff_hessian, atol=1.0e-5, rtol=2.0e-5
    )
    torch.testing.assert_close(
        analytic_gradient, autodiff_gradient, atol=2.0e-6, rtol=2.0e-5
    )


def test_dense_information_count_normalization_is_duplicate_invariant() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = torch.nn.functional.normalize(
        source + torch.tensor([[0.0, 0.02, 0.0], [0.01, 0.0, 0.0]]), dim=-1
    )

    def objective(repeats: int) -> float:
        graph = GlobalSim3FactorGraph()
        graph.add_node(0, sim3_identity())
        graph.add_node(1, sim3_identity())
        graph.add_edge(
            DenseSphericalFactorBlock(
                source=0,
                target=1,
                source_local_pose=torch.eye(4),
                target_local_pose=torch.eye(4),
                source_bearing=source.repeat(repeats, 1),
                target_bearing=target.repeat(repeats, 1),
                source_depth=torch.ones(2 * repeats),
                target_depth=torch.ones(2 * repeats),
                factor_weight=torch.ones(2 * repeats),
                use_depth=False,
                normalize_information_by_count=True,
                information_reference_count=64.0,
            )
        )
        return float(graph.objective())

    assert abs(objective(1) - objective(20)) < 1.0e-6


def test_s2_log_antipodal_is_finite_and_not_zero() -> None:
    base = torch.tensor([[0.0, 0.0, 1.0]])
    antipode = -base
    residual = s2_log_tangent_coordinates(base, antipode)
    assert torch.isfinite(residual).all()
    torch.testing.assert_close(residual.norm(dim=-1), torch.tensor([math.pi]))


def test_yaw_invariant_retrieval_descriptor_and_shift() -> None:
    torch.manual_seed(3)
    feature = torch.randn(1, 24, 8, 16)
    rolled = torch.roll(feature, shifts=5, dims=-1)
    first = build_panorama_retrieval_descriptor(feature)
    second = build_panorama_retrieval_descriptor(rolled)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    shift, score = circular_yaw_shift(feature[0], rolled[0])
    assert shift in {5, 11}
    assert math.isfinite(score)


def test_spherical_rotation_ransac_recovers_rotation_with_outliers() -> None:
    torch.manual_seed(23)
    target = torch.nn.functional.normalize(torch.randn(96, 3), dim=-1)
    truth = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.12, -0.18, 0.07]))[:3, :3]
    source = target @ truth.T
    source[64:] = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
    rotation, inliers, ratio, residual = spherical_rotation_ransac(
        target,
        source,
        torch.ones(96),
        threshold_rad=math.radians(2.0),
        iterations=128,
        seed=19,
    )
    torch.testing.assert_close(rotation, truth, atol=2.0e-4, rtol=2.0e-4)
    assert int(inliers[:64].sum()) == 64
    assert ratio >= 0.65
    assert residual < math.radians(0.05)


def test_so3_loop_verification_filters_rotation_outliers_before_sim3() -> None:
    torch.manual_seed(29)
    count = 128
    target_bearing = torch.nn.functional.normalize(torch.randn(count, 3), dim=-1)
    truth = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.35, -0.22, 0.18]))[:3, :3]
    source_bearing = target_bearing @ truth.T
    source_bearing[64:] = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    source_packet = _packet(0, torch.eye(4).view(1, 4, 4), (0,))
    target_packet = _packet(10, torch.eye(4).view(1, 4, 4), (10,))
    detector = PanoramaLoopDetector(
        descriptor_mode="so3_sh_gram",
        verification_mode="spherical_so3",
        min_matches=32,
        max_matches=128,
        min_inlier_ratio=0.30,
        min_rotation_inlier_ratio=0.40,
        min_spherical_coverage_bins=6,
        max_alignment_residual=0.05,
        max_normalized_alignment_residual=0.05,
        max_rotation_consistency_deg=1.0,
        rotation_ransac_iterations=256,
    )
    calls = 0

    def synthetic_matches(*args, direction: int, **kwargs):
        nonlocal calls
        calls += 1
        start = 0 if direction == 0 else 64
        stop = start + 64
        source = source_bearing[start:stop]
        target = target_bearing[start:stop]
        if direction == 1:
            source, target = target, source
        return {
            "count": 64,
            "raw_valid_count": 64,
            "seed": 100 + direction,
            "source_bearing": source,
            "target_bearing": target,
            "source_depth": torch.full((64,), 2.0),
            "target_depth": torch.full((64,), 2.0),
            "weight": torch.ones(64),
        }

    detector._fibonacci_matches = synthetic_matches
    result = detector.verify_pair(source_packet, target_packet)

    assert calls == 2
    assert result.accepted
    assert result.reason == "coincident_panorama"
    assert result.metadata["rotation_inlier_count"] == 64
    assert result.metadata["verified_num_matches"] == 64
    assert sum(int(factor.source_depth.numel()) for factor in result.dense_factors) == 64
    assert result.metadata["rotation_consistency_deg"] < 0.1


def test_rgb_sh_rotation_preserves_directional_value() -> None:
    torch.manual_seed(5)
    coefficients = torch.randn(7, 9, 3)
    rotation = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.2, -0.1, 0.05]))[:3, :3]
    rotated = rotate_sh_coefficients(coefficients, rotation, degree=2)
    directions_target = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    directions_local = directions_target @ rotation
    expected = torch.einsum("nk,bkc->bnc", real_sh_basis(2, directions_local), coefficients)
    actual = torch.einsum("nk,bkc->bnc", real_sh_basis(2, directions_target), rotated)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_stage2_voxel_fusion_keeps_unique_owner_and_moves_all_attributes() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 0.1
    packet0 = _packet(0, poses, (0, 1))
    packet1 = _packet(1, poses, (0, 1))
    config = {
        "SphericalSelfiGlobalBackend": {"enabled": True, "rgb_sh_degree": 2},
        "BackendOptimization": {"sh_degree": 2},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    fusion = Stage2GlobalMapFusion(gaussian_map, voxel_sizes=(0.5,), min_confidence=0.0, min_opacity=0.0)
    first = fusion.fuse_packet(packet0, sim3_identity())
    count = gaussian_map.anchor_count()
    assert 0 < count < first["requested"]
    fusion.fuse_packet(packet1, sim3_identity())
    assert gaussian_map.anchor_count() == count
    assert set(gaussian_map._anchor_owner_window_id.tolist()) == {0}

    xyz_before = gaussian_map.get_xyz.detach().clone()
    scale_before = gaussian_map.get_scaling.detach().clone()
    correction = sim3_from_components(
        2.0,
        torch.eye(3),
        torch.tensor([1.0, 0.0, 0.0]),
    )
    stats = fusion.apply_owner_corrections({0: sim3_identity()}, {0: correction})
    assert stats["moved"] == count
    expected_xyz = apply_sim3(correction, xyz_before)
    distance = torch.cdist(expected_xyz, gaussian_map.get_xyz)
    nearest = distance.argmin(dim=1)
    assert float(distance.detach().min(dim=1).values.max()) < 2e-3
    torch.testing.assert_close(
        gaussian_map.get_scaling[nearest],
        2.0 * scale_before,
        atol=1e-5,
        rtol=1e-5,
    )


def test_voxel_quality_does_not_apply_latitude_weight_twice() -> None:
    poses = torch.eye(4).repeat(1, 1, 1)
    observation, feature = _observation(poses, (0,), height=10, width=20)
    observation = replace(
        observation,
        confidence=torch.ones_like(observation.confidence),
        density_sh=torch.zeros_like(observation.density_sh),
        valid_mask=torch.ones_like(observation.valid_mask),
    )
    packet = LocalGaussianWindowPacket.from_observation(
        window_id=0,
        observation=observation,
        adapter_features=feature,
        frame_ids=(0,),
        verification_size=feature.shape[-2:],
    )
    gaussian_map = PanoGaussianMap(
        config={"SphericalSelfiGlobalBackend": {"enabled": True, "rgb_sh_degree": 2}},
        device="cpu",
    )
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(1.0e-4,),
        min_confidence=0.0,
        min_opacity=0.0,
    )
    batch = fusion.packet_to_global_batch(packet, sim3_identity())
    assert len(batch) == 10 * 20
    torch.testing.assert_close(batch.quality, torch.full_like(batch.quality, 0.5))


def test_low_multiplicative_quality_alone_does_not_prune_gaussian() -> None:
    packet = _packet(0, torch.eye(4).repeat(1, 1, 1), (0,))
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.5,),
        min_confidence=0.05,
        min_opacity=0.02,
    )
    stats = fusion.fuse_packet(packet, sim3_identity())
    assert stats["anchors_after"] > 0
    gaussian_map._anchor_quality.fill_(1.0e-12)
    removed = fusion.prune_lifecycle(current_frame=1)
    assert removed == 0


def test_voxel_safety_cap_is_reported_as_saturation() -> None:
    packet = _packet(0, torch.eye(4).repeat(1, 1, 1), (0,))
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(1.0e-4,),
        min_confidence=0.0,
        min_opacity=0.0,
        max_total_gaussians=1,
    )
    stats = fusion.fuse_packet(packet, sim3_identity())
    assert stats["map_saturated"] == 1
    assert stats["anchors_before_safety_cap"] > 1
    assert stats["anchors_after"] == 1


def test_two_overlapping_windows_build_graph_and_global_map() -> None:
    frame_features = {frame_id: torch.randn(24, 6, 12) for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    packet0 = _packet(0, poses0, (0, 1), feature_by_frame=frame_features)
    packet1 = _packet(1, poses1, (1, 2), feature_by_frame=frame_features)
    root_config = {
        "SphericalSelfiGlobalBackend": {"enabled": True, "rgb_sh_degree": 2},
        "BackendOptimization": {"sh_degree": 2},
    }
    gaussian_map = PanoGaussianMap(config=root_config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map)
    backend = SphericalSelfiGlobalBackend(
        gaussian_map,
        mapper=mapper,
        config={
            "enabled": True,
            "global_graph": {
                "min_overlap_points": 8,
                "max_overlap_points": 256,
                "max_overlap_residual": 0.05,
                "allow_unaligned_fallback": False,
            },
            "loop_closure": {"exclude_recent_windows": 3, "min_matches": 8},
            "voxel_fusion": {"voxel_sizes": [0.25], "min_confidence": 0.0, "min_opacity": 0.0},
            "map_optimization": {"steps_per_window": 0, "steps_on_loop": 0, "final_steps": 0},
        },
    )
    backend.process_packet(packet0)
    result = backend.process_packet(packet1)
    assert result.aligned
    assert len(backend.graph.nodes) == 2
    assert gaussian_map.anchor_count() > 0
    updates = backend.pop_pose_updates()
    assert set(updates) == {0, 1, 2}
    assert abs(float(updates[2][0, 3]) - 0.2) < 2e-3


def _boundary_backend(
    gaussian_map: PanoGaussianMap,
    *,
    mapper=None,
    start_nodes: int = 6,
    interval_edges: int = 3,
) -> SphericalSelfiGlobalBackend:
    return SphericalSelfiGlobalBackend(
        gaussian_map,
        mapper=mapper,
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "boundary_frame",
                "min_overlap_points": 8,
                "max_overlap_points": 128,
                "min_dense_factors": 8,
                "min_match_cosine": 0.2,
                "min_match_margin": 0.0,
                "max_match_entropy": 1.0,
                "forward_backward": False,
                "allow_boundary_matching_fallback": True,
                "allow_unaligned_fallback": False,
                "optimization_start_nodes": start_nodes,
                "optimization_interval_edges": interval_edges,
                "active_nodes": 6,
            },
            "loop_closure": {
                "exclude_recent_windows": 100,
                "min_matches": 8,
            },
            "voxel_fusion": {
                "voxel_sizes": [0.25],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )


def _refined_boundary_backend(
    gaussian_map: PanoGaussianMap,
    renderer,
) -> SphericalSelfiGlobalBackend:
    return SphericalSelfiGlobalBackend(
        gaussian_map,
        renderer=renderer,
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "boundary_frame",
                "min_overlap_points": 8,
                "max_overlap_points": 128,
                "min_dense_factors": 8,
                "min_match_cosine": 0.2,
                "min_match_margin": 0.0,
                "max_match_entropy": 1.0,
                "forward_backward": False,
                "allow_boundary_matching_fallback": True,
                "allow_unaligned_fallback": False,
                "optimization_start_nodes": 100,
                "optimization_interval_edges": 100,
                "active_nodes": 6,
                "min_depth": 0.05,
                "max_depth": 20.0,
            },
            "rendered_overlap_alignment": {
                "enabled": True,
                "mode": "shared_frame_scale_only",
                "min_points": 16,
                "max_points": 64,
                "alpha_threshold": 0.05,
                "min_inlier_ratio": 0.35,
                "max_median_relative_error": 0.10,
                "max_scale_change": 2.5,
                "failure_policy": "error",
            },
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "radius_voxels": 1.0,
                "compare_existing_only": True,
                "permanent_drop": True,
                "update_existing_statistics": True,
            },
            "loop_closure": {
                "exclude_recent_windows": 100,
                "min_matches": 8,
            },
            "voxel_fusion": {
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )


def test_rendered_depth_scale_recovers_absolute_and_local_correction() -> None:
    backend = _refined_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        _SyntheticSharedDepthRenderer(local_depth=2.0, global_depth=3.0),
    )
    local = torch.full((1, 32, 64), 2.0)
    global_depth = torch.full_like(local, 3.0)
    local_valid = torch.ones_like(local, dtype=torch.bool)
    global_valid = torch.ones_like(local, dtype=torch.bool)
    sky = torch.zeros_like(local)
    # Invalid sky/alpha/hole regions contain adversarial depths and must not
    # influence the robust scale estimate.
    local_valid[:, :8] = False
    global_valid[:, :8] = False
    sky[:, :8] = 1.0
    global_depth[:, :8] = 19.0

    correction, diagnostics, _, inliers = backend._estimate_rendered_depth_scale(
        local,
        global_depth,
        local_valid=local_valid,
        global_valid=global_valid,
        local_sky_probability=sky,
        global_sky_probability=sky,
        shared_scale=0.75,
        seed=7,
    )

    assert correction == pytest.approx(2.0, rel=1.0e-5)
    assert diagnostics["absolute_scale"] == pytest.approx(1.5, rel=1.0e-5)
    assert diagnostics["shared_scale"] == pytest.approx(0.75)
    assert diagnostics["inlier_ratio"] > 0.95
    assert diagnostics["median_relative_error"] < 1.0e-5
    assert float(inliers.float().mean()) > 0.95


def test_refined_rendered_alignment_scales_complete_chunk_without_moving_shared_pose() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.15
    packet0 = _refined_packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _refined_packet(1, poses1, (1, 2), feature_by_frame=features)
    original_depth = packet1.observation.refined_depth.clone()
    original_initial_depth = packet1.observation.initial_depth.clone()
    original_voxel_size = packet1.anchor_observation.voxel_size.clone()
    original_translation = packet1.local_poses_c2w[-1, :3, 3].clone()
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _refined_boundary_backend(
        gaussian_map,
        _SyntheticSharedDepthRenderer(local_depth=1.0, global_depth=2.0),
    )

    backend.process_packet(packet0)
    shared_before = backend.graph.transform(1).clone()
    result = backend.process_packet(packet1)
    shared_after = backend.graph.transform(1)
    stored = backend._last_full_packet

    assert result.aligned
    assert result.diagnostics["alignment"]["absolute_scale"] == pytest.approx(2.0)
    assert result.diagnostics["alignment"]["chunk_scale_normalization"] == pytest.approx(2.0)
    torch.testing.assert_close(shared_after, shared_before)
    # Scaling is transactional: the caller's packet remains unchanged while
    # the admitted backend packet contains the normalized geometry.
    torch.testing.assert_close(packet1.observation.refined_depth, original_depth)
    torch.testing.assert_close(packet1.local_poses_c2w[-1, :3, 3], original_translation)
    assert stored is not None
    torch.testing.assert_close(
        stored.observation.refined_depth,
        original_depth * 2.0,
    )
    torch.testing.assert_close(
        stored.observation.initial_depth,
        original_initial_depth * 2.0,
    )
    torch.testing.assert_close(
        stored.local_poses_c2w[-1, :3, 3],
        original_translation * 2.0,
    )
    torch.testing.assert_close(
        stored.anchor_observation.voxel_size,
        original_voxel_size * 2.0,
    )
    assert result.fusion["hash_hits"] >= 0


def test_rendered_alignment_failure_leaves_packet_graph_and_map_unchanged() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    packet0 = _refined_packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _refined_packet(1, poses1, (1, 2), feature_by_frame=features)
    packet_depth = packet1.observation.refined_depth.clone()
    packet_xyz = packet1.anchor_observation.xyz.clone()
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _refined_boundary_backend(
        gaussian_map,
        _SyntheticSharedDepthRenderer(local_depth=1.0, global_depth=10.0),
    )
    backend.process_packet(packet0)
    graph_nodes = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }
    graph_edge_count = len(backend.graph.edges)
    map_xyz = gaussian_map.get_xyz.detach().clone()
    map_count = gaussian_map.anchor_count()
    window_order = list(backend.window_order)

    with pytest.raises(RuntimeError, match="rendered shared-frame scale alignment failed"):
        backend.process_packet(packet1)

    assert backend.window_order == window_order
    assert len(backend.graph.edges) == graph_edge_count
    assert set(backend.graph.nodes) == set(graph_nodes)
    for node, transform in graph_nodes.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)
    assert gaussian_map.anchor_count() == map_count
    torch.testing.assert_close(gaussian_map.get_xyz, map_xyz)
    torch.testing.assert_close(packet1.observation.refined_depth, packet_depth)
    torch.testing.assert_close(packet1.anchor_observation.xyz, packet_xyz)


def test_hash_render_failure_rolls_back_graph_owner_and_window_transaction() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.15
    packet0 = _refined_packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _refined_packet(1, poses1, (1, 2), feature_by_frame=features)
    renderer = _SyntheticSharedDepthRenderer(
        local_depth=1.0,
        global_depth=2.0,
        fail_on_call=4,
    )
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _refined_boundary_backend(gaussian_map, renderer)
    backend.process_packet(packet0)
    nodes_before = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }
    edges_before = list(backend.graph.edges)
    map_parameters_before = {
        name: parameter
        for name, parameter in gaussian_map._parameters.items()
    }
    map_xyz_before = gaussian_map.get_xyz.detach().clone()
    window_order_before = list(backend.window_order)
    anchor_nodes_before = dict(backend.window_anchor_nodes)

    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        backend.process_packet(packet1)

    assert backend.window_order == window_order_before
    assert backend.window_anchor_nodes == anchor_nodes_before
    assert len(backend.graph.edges) == len(edges_before)
    assert all(actual is expected for actual, expected in zip(backend.graph.edges, edges_before))
    assert set(backend.graph.nodes) == set(nodes_before)
    for node, transform in nodes_before.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)
    for name, parameter in map_parameters_before.items():
        assert gaussian_map._parameters[name] is parameter
    torch.testing.assert_close(gaussian_map.get_xyz, map_xyz_before)


def test_boundary_frame_graph_reuses_shared_node_and_uses_one_dense_factor_per_window() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    packet0 = _packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _packet(1, poses1, (1, 2), feature_by_frame=features)
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map)

    backend.process_packet(packet0)
    result = backend.process_packet(packet1)

    assert result.aligned
    assert set(backend.graph.nodes) == {0, 1, 2}
    assert backend.window_anchor_nodes == {0: 0, 1: 1}
    assert len(backend.graph.edges) == 2
    assert all(isinstance(factor, DenseSphericalFactorBlock) for factor in backend.graph.edges)
    assert all(factor.edge_type == "boundary_dense_spherical" for factor in backend.graph.edges)
    assert all(torch.equal(factor.factor_weight, torch.ones_like(factor.factor_weight)) for factor in backend.graph.edges)
    geometry = backend.pop_frame_geometry_updates()
    assert geometry[1].owner_window_id == geometry[1].depth_owner_window_id == 0
    torch.testing.assert_close(geometry[1].pose_c2w, apply_sim3_to_c2w(backend.graph.transform(1), torch.eye(4)))


def test_shared_frame_umeyama_ignores_descriptor_disagreement() -> None:
    positive = torch.ones(24, 6, 12)
    negative = -positive
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    packet0 = _packet(
        0,
        poses0,
        (0, 1),
        feature_by_frame={0: positive, 1: positive},
    )
    packet1 = _packet(
        1,
        poses1,
        (1, 2),
        feature_by_frame={1: negative, 2: negative},
    )
    backend = _boundary_backend(PanoGaussianMap(config={}, device="cpu"))

    measurement, diagnostics = backend._shared_frame_alignment(packet0, packet1)

    assert measurement is not None
    assert diagnostics["descriptor_gate"] is False
    assert diagnostics["weight_mode"] == "fibonacci_equal_joint_geometry_mask"
    assert diagnostics["overlap_points"] >= backend.overlap_aligner.min_points

    packet0.observation = replace(
        packet0.observation,
        confidence=torch.zeros_like(packet0.observation.confidence),
    )
    packet1.observation = replace(
        packet1.observation,
        confidence=torch.zeros_like(packet1.observation.confidence),
    )
    edge, dense_factor, shared_factor, legacy_diagnostics = backend._overlap_edge(
        packet0, packet1
    )
    assert edge is not None and dense_factor is not None and shared_factor is not None
    assert legacy_diagnostics["descriptor_gate"] is False
    assert legacy_diagnostics["weight_mode"] == "fibonacci_equal_joint_geometry_mask"
    torch.testing.assert_close(
        dense_factor.factor_weight,
        torch.ones_like(dense_factor.factor_weight),
    )


def test_boundary_alignment_rolls_back_shift_then_syncs_canonical_depth() -> None:
    shared = torch.randn(24, 6, 12)
    features = {0: shared, 1: shared, 2: shared}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    packet0 = _packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _packet(1, poses1, (1, 2), feature_by_frame=features)
    unshifted = packet1.observation.refined_depth.detach().clone()
    packet1.pre_depth_shift_depth = unshifted
    packet1.observation = packet1.observation.with_geometry(
        refined_depth=1.5 * unshifted
    )
    packet1.metadata["dense_depth_shift_applied"] = True
    backend = _boundary_backend(PanoGaussianMap(config={}, device="cpu"))
    backend.process_packet(packet0)
    actual_alignment = backend._shared_frame_alignment
    call_count = 0

    def fail_twice_then_align(source, target):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return None, {"reason": f"forced_failure_{call_count}"}
        return actual_alignment(source, target)

    backend._shared_frame_alignment = fail_twice_then_align
    result = backend.process_packet(packet1)

    assert result.aligned
    alignment = result.diagnostics["alignment"]
    assert alignment["depth_shift_rollback"] is True
    assert alignment["alignment_recovery_stage"] == "canonical_depth_retry"
    assert [value["stage"] for value in alignment["alignment_attempts"]] == [
        "ba_pose_shifted_depth",
        "ba_pose_depth_shift_rollback",
        "canonical_depth_retry",
    ]
    torch.testing.assert_close(
        packet1.observation.refined_depth[0, 0],
        packet0.observation.refined_depth[0, 1],
    )
    assert packet1.metadata["canonical_shared_depth_owner_window"] == 0
    assert backend.frame_depth_owner_window[1] == 0


def test_boundary_graph_waits_for_six_nodes_then_runs_recent_ba() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(6)}
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map, start_nodes=6, interval_edges=3)
    results = []
    for window_id in range(5):
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[0, 0, 3] = 0.1 * window_id
        poses[1, 0, 3] = 0.1 * (window_id + 1)
        if window_id == 4:
            perturbed = backend.graph.transform(4).clone()
            perturbed[0, 3] += 0.03
            backend.graph.nodes[4] = perturbed
        results.append(
            backend.process_packet(
                _packet(window_id, poses, (window_id, window_id + 1), feature_by_frame=features)
            )
        )

    assert all(result.graph is None for result in results[:4])
    assert results[4].graph is not None
    assert results[4].diagnostics["global_ba_scheduled"]
    assert len(backend.graph.nodes) == 6
    assert results[4].graph.final_objective <= results[4].graph.initial_objective
    assert 0 not in results[4].graph.optimized_node_ids


def test_boundary_loop_transaction_rolls_back_dcs_rejected_factor() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.2
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "boundary_frame",
                "min_overlap_points": 8,
                "max_overlap_points": 128,
                "min_dense_factors": 8,
                "min_match_cosine": 0.2,
                "min_match_margin": 0.0,
                "max_match_entropy": 1.0,
                "forward_backward": False,
                "allow_boundary_matching_fallback": True,
                "optimization_start_nodes": 6,
            },
            "loop_closure": {"exclude_recent_windows": 100, "min_matches": 8},
            "robust_loop": {
                "mode": "dcs",
                "dcs_phi": 1.0e-3,
                "transactional": True,
                "min_commit_dcs_scale": 0.5,
            },
            "voxel_fusion": {
                "voxel_sizes": [0.25],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )
    backend.process_packet(_packet(0, poses0, (0, 1), feature_by_frame=features))

    source = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    dense = DenseSphericalLoopMeasurement(
        source=0,
        target=1,
        source_local_pose=torch.eye(4),
        target_local_pose=torch.eye(4),
        source_bearing=source,
        target_bearing=-source,
        source_depth=torch.full((64,), 2.0),
        target_depth=torch.full((64,), 2.0),
        factor_weight=torch.ones(64),
        use_depth=False,
        edge_type="loop_dense_spherical",
        dcs_phi=1.0e-3,
    )
    predicted = sim3_inverse(backend.graph.transform(0)) @ backend.graph.transform(1)
    verification = PanoramaLoopVerification(
        accepted=True,
        factor=LoopPoseMeasurement(
            kind="sim3",
            source=0,
            target=1,
            measurement_target_to_source=predicted,
            information_diag=torch.ones(7),
            edge_type="loop",
            dcs_phi=1.0e-3,
        ),
        source_window_id=0,
        target_window_id=1,
        retrieval_score=1.0,
        yaw_shift_columns=0,
        num_matches=64,
        inlier_ratio=1.0,
        residual=0.0,
        reason="synthetic_bad_dense_loop",
        dense_factors=(dense,),
    )
    backend.loop_detector.detect = lambda packet: [verification]
    correction_calls = 0
    original_correction = backend.fusion.apply_owner_corrections

    def record_correction(old, new):
        nonlocal correction_calls
        correction_calls += 1
        return original_correction(old, new)

    backend.fusion.apply_owner_corrections = record_correction

    result = backend.process_packet(_packet(1, poses1, (1, 2), feature_by_frame=features))

    assert result.loop_accepted == 0
    assert len(backend.graph.edges) == 2
    assert verification.reason == "loop_transaction_rejected"
    assert verification.metadata["graph_transaction"]["minimum_dcs_scale"] < 0.5
    assert not backend.accepted_loop_pairs
    assert correction_calls == 0


def test_hierarchical_backend_freezes_five_window_submaps_and_keeps_six_node_local_graph() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(11)}
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "boundary_frame",
                "min_overlap_points": 8,
                "max_overlap_points": 128,
                "min_dense_factors": 8,
                "min_match_cosine": 0.2,
                "min_match_margin": 0.0,
                "max_match_entropy": 1.0,
                "forward_backward": False,
                "allow_boundary_matching_fallback": True,
                "optimization_start_nodes": 6,
                "optimization_interval_edges": 5,
                "active_nodes": 6,
            },
            "hierarchical_submaps": {
                "enabled": True,
                "windows_per_submap": 5,
                "shared_boundary_nodes": 1,
            },
            "loop_closure": {"exclude_recent_windows": 100, "min_matches": 8},
            "voxel_fusion": {
                "voxel_sizes": [0.25],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )
    results = []
    for window_id in range(10):
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[0, 0, 3] = 0.1 * window_id
        poses[1, 0, 3] = 0.1 * (window_id + 1)
        results.append(
            backend.process_packet(
                _packet(
                    window_id,
                    poses,
                    (window_id, window_id + 1),
                    feature_by_frame=features,
                )
            )
        )

    assert len(backend.graph.nodes) == 11
    assert backend.submap_graph is not None
    assert set(backend.submap_graph.nodes) == {0, 1}
    assert len(backend.submap_graph.edges) == 1
    assert backend.submap_graph.edges[0].edge_type == "submap_sequential"
    assert backend.submaps[0].frozen and backend.submaps[1].frozen
    assert backend.submaps[0].window_ids == list(range(5))
    assert backend.submaps[1].window_ids == list(range(5, 10))
    assert backend.submaps[0].boundary_node_ids == list(range(6))
    assert backend.submaps[1].boundary_node_ids == list(range(5, 11))
    for record in backend.submaps.values():
        scales = [
            float(sim3_components(backend.graph.transform(node))[0])
            for node in record.boundary_node_ids
        ]
        assert max(scales) - min(scales) < 1.0e-6
    assert backend.window_to_submap == {
        **{window_id: 0 for window_id in range(5)},
        **{window_id: 1 for window_id in range(5, 10)},
    }
    assert results[4].diagnostics["submap_frozen"]
    assert results[9].diagnostics["submap_frozen"]
    assert results[4].graph is not None and results[9].graph is not None
    assert backend.submaps[0].compressed_dense_factors == 5
    assert backend.submaps[1].compressed_dense_factors == 5
    assert not any(
        isinstance(factor, DenseSphericalFactorBlock)
        and factor.edge_type == "boundary_dense_spherical"
        for factor in backend.graph.edges
    )

    final = backend.finalize()
    assert final["hierarchical_submaps_enabled"] is True
    assert final["submap_nodes"] == 2
    assert final["compressed_dense_factors"] == 10

    window_transforms = backend._window_anchor_transforms()
    window_measurement = (
        sim3_inverse(window_transforms[0]) @ window_transforms[5]
    )
    loop = PanoramaLoopVerification(
        accepted=True,
        factor=LoopPoseMeasurement(
            kind="sim3",
            source=0,
            target=5,
            measurement_target_to_source=window_measurement,
            information_diag=torch.ones(7),
            edge_type="loop",
        ),
        source_window_id=0,
        target_window_id=5,
        retrieval_score=1.0,
        yaw_shift_columns=0,
        num_matches=64,
        inlier_ratio=1.0,
        residual=0.0,
        reason="synthetic",
    )
    converted = backend._loop_measurement_for_submaps(loop)
    assert converted is not None and converted.measurement_target_to_source is not None
    expected_submap_measurement = (
        sim3_inverse(backend.submap_graph.transform(0))
        @ backend.submap_graph.transform(1)
    )
    torch.testing.assert_close(
        converted.measurement_target_to_source,
        expected_submap_measurement,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    before = backend._window_anchor_transforms()
    moved = backend.submap_graph.transform(1).clone()
    moved[1, 3] += 0.5
    backend.submap_graph.nodes[1] = moved
    backend._apply_submap_graph_to_boundary_graph()
    after = backend._window_anchor_transforms()
    torch.testing.assert_close(before[0], after[0])
    assert abs(float(after[7][1, 3] - before[7][1, 3]) - 0.5) < 1.0e-5


def test_lazy_owner_correction_updates_transform_without_rewriting_gaussians() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 0.1
    packet = _packet(0, poses, (0, 1))
    gaussian_map = PanoGaussianMap(config={}, sh_degree=2, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.25,),
        min_confidence=0.0,
        min_opacity=0.0,
        lazy_owner_transforms=True,
    )
    identity = torch.eye(4)
    fusion.fuse_packet(packet, identity)
    base_xyz = gaussian_map.xyz.detach().clone()
    base_scaling = gaussian_map._base_scaling().detach().clone()
    reference_world = gaussian_map.get_xyz.detach().clone()
    update = sim3_exp(
        torch.tensor([0.3, -0.2, 0.1, 0.15, -0.08, 0.04, math.log(1.2)])
    )

    stats = fusion.apply_owner_corrections({0: identity}, {0: update})

    torch.testing.assert_close(gaussian_map.xyz.detach(), base_xyz)
    torch.testing.assert_close(
        gaussian_map.get_xyz.detach(),
        apply_sim3(update, reference_world),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        gaussian_map.get_scaling.detach(),
        1.2 * base_scaling,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert stats["moved"] == gaussian_map.anchor_count()
    assert stats["deduplicated"] == 0
    assert stats["lazy"] == 1


def test_lazy_loop_neighborhood_deduplicates_cross_owner_seam_only_on_commit() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 0.1
    gaussian_map = PanoGaussianMap(config={}, sh_degree=2, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.25,),
        min_confidence=0.0,
        min_opacity=0.0,
        lazy_owner_transforms=True,
    )
    fusion.fuse_packet(_packet(0, poses, (0, 1)), torch.eye(4))
    first_count = gaussian_map.anchor_count()
    fusion.fuse_packet(_packet(1, poses, (2, 3)), torch.eye(4))
    combined_count = gaussian_map.anchor_count()

    removed = fusion.deduplicate_owner_neighborhood({0, 1})

    assert first_count > 0
    assert combined_count == 2 * first_count
    assert removed == first_count
    assert gaussian_map.anchor_count() == first_count


def test_boundary_factor_ignores_confidence_and_hard_excludes_sky() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 0.1
    packet = _packet(0, poses, (0, 1))
    uv = packet.observation.source_uv.reshape(-1, 2)[:16].clone()
    bearing = erp_pixel_to_unit_ray(uv, *packet.observation.image_size)
    packet.boundary_matches = BoundaryMatchBlock(
        source_uv=uv,
        target_uv=uv.clone(),
        source_bearing=bearing,
        target_bearing=bearing.clone(),
        top1_cosine=torch.ones(16),
        top2_margin=torch.ones(16),
        normalized_entropy=torch.zeros(16),
    )
    packet.observation = replace(
        packet.observation,
        confidence=torch.rand_like(packet.observation.confidence),
    )
    packet.sky_prob[0, 0, 0, 0, 0] = 1.0
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map)
    factor, diagnostics = backend._boundary_factor(packet)

    assert factor is not None
    assert diagnostics["sky_rejected"] >= 1
    assert factor.source_depth.numel() == 15
    torch.testing.assert_close(factor.factor_weight, torch.ones_like(factor.factor_weight))


def test_global_graph_materializes_inference_factors_and_descends() -> None:
    graph = GlobalSim3FactorGraph(max_iterations=4)
    graph.add_node(0, sim3_identity())
    graph.add_node(
        1,
        sim3_from_components(1.0, torch.eye(3), torch.tensor([0.1, 0.0, 0.0])),
    )
    with torch.inference_mode():
        bearing = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
        factor = DenseSphericalFactorBlock(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            source_bearing=bearing,
            target_bearing=bearing,
            source_depth=torch.full((32,), 2.0),
            target_depth=torch.full((32,), 2.0),
            factor_weight=torch.ones(32),
        )
    graph.add_edge(factor)
    assert not graph.edges[0].source_bearing.is_inference()
    result = graph.optimize()
    assert result.accepted
    assert result.final_objective < result.initial_objective


def test_shared_umeyama_rescales_all_new_chunk_geometry() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(3)}
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 0.1
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 0.1
    poses1[1, 0, 3] = 0.15
    packet0 = _packet(0, poses0, (0, 1), feature_by_frame=features)
    packet1 = _packet(1, poses1, (1, 2), feature_by_frame=features)
    packet1.observation = packet1.observation.with_geometry(
        refined_depth=torch.ones_like(packet1.observation.refined_depth)
    )
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map)

    backend.process_packet(packet0)
    result = backend.process_packet(packet1)

    scale = float(result.diagnostics["alignment"]["chunk_scale_normalization"])
    assert abs(scale - 2.0) < 2.0e-3
    assert abs(float(packet1.local_poses_c2w[-1, 0, 3]) - 0.1) < 2.0e-3
    torch.testing.assert_close(
        packet1.observation.refined_depth,
        torch.full_like(packet1.observation.refined_depth, 2.0),
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    _, _, end_translation = sim3_components(backend.graph.transform(2))
    assert abs(float(end_translation[0]) - 0.2) < 3.0e-3
    geometry = backend.pop_frame_geometry_updates()
    assert abs(float(geometry[2].depth_scale) - 2.0) < 2.0e-3
    # The shared frame remains owned by the previous window and therefore is
    # not rescaled a second time by the new chunk's local normalization.
    assert abs(float(geometry[1].depth_scale) - 1.0) < 2.0e-3


def test_boundary_map_optimization_refines_all_poses_and_passes_group_lrs() -> None:
    packet = _packet(0, torch.eye(4).repeat(2, 1, 1), (0, 1))

    class _Mapper:
        def __init__(self) -> None:
            self.optimizer = None
            self.stats = SimpleNamespace(notes=[], n_anchors=0)
            self.settings = None
            self.extra_loss_fn = "unset"
            self.commits = 0

        def set_spherical_selfi_observation_geometry(self, *args, **kwargs) -> None:
            return None

        def prepare_spherical_selfi_window(self, frame_ids) -> int:
            return len(frame_ids)

        def optimize_spherical_selfi_window(self, *, settings, extra_loss_fn, **kwargs):
            self.settings = dict(settings)
            self.extra_loss_fn = extra_loss_fn
            return {"steps": 3.0, "window_rollback": 0.0}

        def refined_pose_c2w(self, frame_id: int):
            return torch.eye(4)

        def commit_spherical_selfi_window(self) -> None:
            self.commits += 1

        def rollback_spherical_selfi_window(self) -> None:
            raise AssertionError("map-only optimization should not roll back")

    mapper = _Mapper()
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map, mapper=mapper)
    backend.graph.add_node(0, sim3_identity())
    backend.window_anchor_nodes[0] = 0
    backend.window_order = [0]
    backend.packets[0] = packet.compact_for_memory()
    backend._optimization_packets[0] = packet

    metrics = backend._run_map_optimization(0, packet.frame_ids, 3)

    assert metrics["pose_refine_enabled"] == 1.0
    assert mapper.settings["pose_lr"] == 1.0e-3
    assert mapper.settings["pose_refine_enable"] is True
    assert mapper.settings["fixed_pose_frame_ids"] == []
    assert callable(mapper.extra_loss_fn)
    assert mapper.commits == 1


def test_separate_gaussian_groups_and_true_adam_update_scaling() -> None:
    gaussian_map = PanoGaussianMap(
        config={"SphericalSelfiGlobalBackend": {"enabled": True, "rgb_sh_degree": 2}},
        device="cpu",
    )
    gaussian_map.xyz = torch.nn.Parameter(torch.zeros(2, 3))
    gaussian_map.rotation = torch.nn.Parameter(torch.zeros(2, 4))
    gaussian_map.scaling = torch.nn.Parameter(torch.zeros(2, 3))
    gaussian_map.opacity_logit = torch.nn.Parameter(torch.zeros(2, 1))
    gaussian_map.features = torch.nn.Parameter(torch.zeros(2, 3))
    gaussian_map.sh_rest = torch.nn.Parameter(torch.zeros(2, 8, 3))
    mapper = PanoGaussianMapper(gaussian_map)
    mapper.optim_cfg.update(
        {
            "separate_gaussian_lrs": True,
            "xyz_lr": 5.0e-4,
            "feature_lr": 2.0e-3,
            "sh_rest_lr": 1.0e-4,
            "opacity_lr": 1.0e-3,
            "scaling_lr": 1.0e-4,
            "rotation_lr": 1.0e-4,
            "optimize_skybox": False,
        }
    )
    groups = mapper._map_param_groups(gaussian_enabled=True, phase="feedforward_window")
    rates = {str(group["name"]): float(group["lr"]) for group in groups}
    assert rates == {
        "xyz": 5.0e-4,
        "features": 2.0e-3,
        "sh_rest": 1.0e-4,
        "opacity": 1.0e-3,
        "scaling": 1.0e-4,
        "rotation": 1.0e-4,
    }

    optimizer = torch.optim.AdamW(
        [{"params": [gaussian_map.xyz], "lr": 1.0e-2}], weight_decay=0.0
    )
    gaussian_map.xyz.grad = torch.ones_like(gaussian_map.xyz)
    optimizer.step()
    mapper._apply_gaussian_adamw_update_scales(
        optimizer, torch.tensor([1.0, 0.1])
    )
    owner_step = gaussian_map.xyz.detach()[0].abs().mean()
    neighbor_step = gaussian_map.xyz.detach()[1].abs().mean()
    torch.testing.assert_close(neighbor_step, owner_step * 0.1, atol=1.0e-7, rtol=1.0e-5)


def test_global_graph_rolls_back_transaction_on_non_finite_factor() -> None:
    graph = GlobalSim3FactorGraph(max_iterations=3)
    graph.add_node(0, sim3_identity())
    initial = sim3_from_components(1.0, torch.eye(3), torch.tensor([0.1, 0.0, 0.0]))
    graph.add_node(1, initial)
    graph.add_edge(
        DenseSphericalFactorBlock(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            source_bearing=torch.tensor([[float("nan"), 0.0, 1.0]]),
            target_bearing=torch.tensor([[0.0, 0.0, 1.0]]),
            source_depth=torch.ones(1),
            target_depth=torch.ones(1),
            factor_weight=torch.ones(1),
        )
    )

    result = graph.optimize()

    assert not result.accepted
    assert result.reason == "non_finite_initial_objective"
    torch.testing.assert_close(graph.transform(1), initial)


def test_joint_pose_sync_rebases_scale_and_updates_both_overlap_packets() -> None:
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 2.0
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[0, 0, 3] = 2.0
    poses1[1, 0, 3] = 4.0
    packet0 = _packet(0, poses0, (0, 1))
    packet1 = _packet(1, poses1, (1, 2))

    optimized = {
        1: torch.eye(4),
        2: torch.eye(4),
    }
    optimized[1][0, 3] = 2.2
    optimized[2][0, 3] = 4.4

    class _Mapper:
        def refined_pose_c2w(self, frame_id: int):
            return optimized.get(int(frame_id))

    gaussian_map = PanoGaussianMap(
        config={"SphericalSelfiGlobalBackend": {"enabled": True, "rgb_sh_degree": 2}},
        device="cpu",
    )
    backend = SphericalSelfiGlobalBackend(
        gaussian_map,
        mapper=_Mapper(),  # type: ignore[arg-type]
        config={
            "enabled": True,
            "geometry_validation": {"enabled": True, "tolerance": 1.0e-6},
            "voxel_fusion": {"voxel_sizes": [0.2], "min_confidence": 0.0, "min_opacity": 0.0},
            "map_optimization": {"steps_per_window": 0},
        },
    )
    backend.graph.add_node(0, sim3_identity())
    backend.graph.add_node(
        1, sim3_from_components(2.0, torch.eye(3), torch.tensor([2.0, 0.0, 0.0]))
    )
    backend.packets = {0: packet0, 1: packet1}
    backend.window_order = [0, 1]
    backend.frame_windows = {0: {0}, 1: {0, 1}, 2: {1}}

    backend._synchronize_joint_optimized_window(1)

    torch.testing.assert_close(packet1.local_poses_c2w[0], torch.eye(4))
    assert abs(float(packet1.local_poses_c2w[1, 0, 3]) - 1.1) < 1.0e-6
    assert abs(float(packet0.local_poses_c2w[1, 0, 3]) - 2.2) < 1.0e-6
    for frame_id, packet, index in ((1, packet0, 1), (1, packet1, 0), (2, packet1, 1)):
        reconstructed = apply_sim3_to_c2w(
            backend.graph.transform(packet.window_id), packet.local_poses_c2w[index]
        )
        torch.testing.assert_close(reconstructed, optimized[frame_id], atol=1.0e-6, rtol=1.0e-6)
    geometry = backend.pop_frame_geometry_updates()
    assert geometry[1].owner_window_id == 1
    assert geometry[1].depth_owner_window_id == 0
    assert geometry[1].depth_scale == 1.0
    assert geometry[1].depth_scales_by_window == {0: 1.0, 1: 2.0}


def test_boundary_pose_sync_updates_nodes_and_rebases_shared_window_coordinates() -> None:
    poses0 = torch.eye(4).repeat(2, 1, 1)
    poses0[1, 0, 3] = 2.0
    poses1 = torch.eye(4).repeat(2, 1, 1)
    poses1[1, 0, 3] = 1.0
    packet0 = _packet(0, poses0, (0, 1))
    packet1 = _packet(1, poses1, (1, 2))
    optimized = {1: torch.eye(4), 2: torch.eye(4)}
    optimized[1][0, 3] = 2.2
    optimized[2][0, 3] = 4.4

    class _Mapper:
        def refined_pose_c2w(self, frame_id: int):
            return optimized.get(int(frame_id))

    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map, mapper=_Mapper())
    backend.graph.add_node(0, sim3_identity())
    backend.graph.add_node(
        1, sim3_from_components(2.0, torch.eye(3), torch.tensor([2.0, 0.0, 0.0]))
    )
    backend.graph.add_node(
        2, sim3_from_components(2.0, torch.eye(3), torch.tensor([4.0, 0.0, 0.0]))
    )
    backend.window_anchor_nodes = {0: 0, 1: 1}
    backend.boundary_node_order = [0, 1, 2]
    backend.packets = {0: packet0, 1: packet1}
    backend.window_order = [0, 1]
    backend.frame_windows = {0: {0}, 1: {0, 1}, 2: {1}}
    backend.frame_owner_window = {0: 0, 1: 0, 2: 1}
    backend.frame_depth_owner_window = {0: 0, 1: 0, 2: 1}

    backend._synchronize_joint_optimized_window(1)

    node1_scale, _, node1_translation = sim3_components(backend.graph.transform(1))
    node2_scale, _, node2_translation = sim3_components(backend.graph.transform(2))
    torch.testing.assert_close(node1_scale, torch.tensor(2.0))
    torch.testing.assert_close(node2_scale, torch.tensor(2.0))
    torch.testing.assert_close(node1_translation, torch.tensor([2.2, 0.0, 0.0]))
    torch.testing.assert_close(node2_translation, torch.tensor([4.4, 0.0, 0.0]))
    torch.testing.assert_close(packet1.local_poses_c2w[0], torch.eye(4))
    assert abs(float(packet1.local_poses_c2w[1, 0, 3]) - 1.1) < 1.0e-6
    assert abs(float(packet0.local_poses_c2w[1, 0, 3]) - 2.2) < 1.0e-6
    geometry = backend.pop_frame_geometry_updates()
    torch.testing.assert_close(geometry[1].pose_c2w[:3, 3], node1_translation)
    torch.testing.assert_close(geometry[2].pose_c2w[:3, 3], node2_translation)


def test_mapper_geometry_updates_materialize_depth_from_immutable_local_value() -> None:
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map)
    mapper.observations[5] = MapperObservation(
        frame_id=5,
        image=torch.zeros(3, 4, 8),
        pose_c2w=torch.eye(4),
        target_depth=torch.full((1, 4, 8), 2.0),
    )
    update = SimpleNamespace(
        pose_c2w=torch.eye(4),
        depth_scale=2.0,
        owner_window_id=1,
        depth_owner_window_id=1,
        depth_scales_by_window={1: 2.0},
    )
    mapper.apply_frontend_geometry_updates({5: update})
    torch.testing.assert_close(
        mapper.observations[5].target_depth, torch.full((1, 4, 8), 4.0)
    )
    update.depth_scale = 3.0
    update.depth_scales_by_window = {1: 3.0}
    mapper.apply_frontend_geometry_updates({5: update})
    torch.testing.assert_close(
        mapper.observations[5].target_depth, torch.full((1, 4, 8), 6.0)
    )
    torch.testing.assert_close(
        mapper.observations[5].target_depth_local, torch.full((1, 4, 8), 2.0)
    )


def test_overlap_pose_owner_does_not_rescale_depth_from_another_window() -> None:
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map)
    mapper.observations[3] = MapperObservation(
        frame_id=3,
        image=torch.zeros(3, 4, 8),
        pose_c2w=torch.eye(4),
        target_depth=torch.full((1, 4, 8), 2.0),
        target_depth_local=torch.full((1, 4, 8), 2.0),
        owner_window_id=0,
    )
    update = SimpleNamespace(
        pose_c2w=torch.eye(4),
        depth_scale=1.0,
        owner_window_id=1,
        depth_owner_window_id=0,
        depth_scales_by_window={0: 1.0, 1: 2.0},
    )
    mapper.apply_frontend_geometry_updates({3: update})
    assert mapper.observations[3].owner_window_id == 0
    torch.testing.assert_close(
        mapper.observations[3].target_depth, torch.full((1, 4, 8), 2.0)
    )


def test_frontend_output_contract_remains_unchanged() -> None:
    from frontend.pano_droid.interfaces import FrontendOutput

    assert tuple(FrontendOutput.__dataclass_fields__) == (
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
    )


def test_window_packet_compaction_releases_full_resolution_state() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    packet = _packet(0, poses, (0, 1))
    packet.verification_features = torch.nn.functional.interpolate(
        packet.verification_features.reshape(2, 24, 6, 12),
        size=(3, 5),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 2, 24, 3, 5)
    compact = packet.compact_for_memory()
    assert compact.observation.image_size == (3, 5)
    assert compact.adapter_features.shape[-2:] == (3, 5)
    assert compact.observation.refined_depth.device.type == "cpu"
    assert compact.observation.rgb_sh.device.type == "cpu"
    assert compact.observation.source_uv.shape == (3, 5, 2)
    assert compact.observation.source_ray.shape == (3, 5, 3)


def test_synthetic_window_runtime_emits_unchanged_outputs_and_packet(tmp_path: Path) -> None:
    config = stage2_default_config()
    config["image"] = {"height": 8, "width": 16, "head_height": 8, "head_width": 16}
    config["head"].update({"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12})
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "format": "spherical_selfi_gaussian_head_v1",
            "head": head.state_dict(),
            "adapter_sha256": "synthetic-no-checkpoint",
            "global_step": 0,
            "metrics": {},
            "best_val_psnr": None,
        },
        checkpoint,
    )
    config["stage2_checkpoint"] = {"path": str(checkpoint)}
    config["SphericalSelfiRuntime"] = {
        "enabled": True,
        "feature_device": "cpu",
        "head_device": "cpu",
        "feature_amp": False,
        "window": {"size": 3, "stride": 2, "verification_size": [4, 8]},
        "local_ba": {"enabled": False},
    }
    frontend = SphericalSelfiWindowFrontend(config)
    for frame_id in range(3):
        frontend.track(PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id))
    ready = frontend.pop_ready_outputs()
    packets = frontend.consume_local_gaussian_windows()
    diagnostics = frontend.consume_local_ba_diagnostics()
    flushed = frontend.flush()
    assert [output.frame_id for output in ready + flushed] == [0, 1, 2]
    assert all(output.inverse_depth is not None for output in ready + flushed)
    assert all(output.tracking_status == "tracked_spherical_selfi_stage2" for output in ready + flushed)
    assert len(packets) == 1
    assert packets[0].frame_ids == (0, 1, 2)
    assert packets[0].local_poses_c2w[0].equal(torch.eye(4))
    assert len(diagnostics) == 1
    assert diagnostics[0]["matcher"] == "none"
    assert diagnostics[0]["frame_ids"] == (0, 1, 2)
    assert diagnostics[0]["gt_poses_c2w"] is None
    assert frontend.consume_local_ba_diagnostics() == []


def test_synthetic_window_runtime_adapter_ba_builds_diagnostics(tmp_path: Path) -> None:
    config = stage2_default_config()
    config["image"] = {"height": 8, "width": 16, "head_height": 8, "head_width": 16}
    config["head"].update({"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12})
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2_local_ba.pt"
    torch.save(
        {
            "format": "spherical_selfi_gaussian_head_v1",
            "head": head.state_dict(),
            "adapter_sha256": "synthetic-no-checkpoint",
            "global_step": 0,
            "metrics": {},
            "best_val_psnr": None,
        },
        checkpoint,
    )
    config["stage2_checkpoint"] = {"path": str(checkpoint)}
    config["SphericalSelfiRuntime"] = {
        "enabled": True,
        "feature_device": "cpu",
        "head_device": "cpu",
        "feature_amp": False,
        "window": {"size": 3, "stride": 2, "verification_size": [4, 8]},
        "local_ba": {
            "enabled": True,
            "iterations": 1,
            "solver_mode": "standard_lm",
            "lm_max_trials": 2,
            "jacobian_mode": "analytic",
            "validate_analytic_jacobian": True,
            "pose_update_side": "right",
            "pose_dof_mode": "se3",
            "gauge_mode": "initial_baseline",
            "dense_depth_mode": "affine",
            "min_factors": 1,
            "min_affine_support": 2,
            "matching": {
                "type": "adapter",
                "num_queries": 4,
                "query_chunk_size": 2,
                "forward_backward": False,
                "min_factor_weight": 0.0,
            },
        },
    }
    frontend = SphericalSelfiWindowFrontend(config)
    for frame_id in range(3):
        frontend.track(PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id))
    frontend.pop_ready_outputs()
    packets = frontend.consume_local_gaussian_windows()
    diagnostics = frontend.consume_local_ba_diagnostics()
    assert len(packets) == 1
    assert packets[0].boundary_matches is not None
    assert packets[0].boundary_matches.count > 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["matcher"] == "adapter"
    assert diagnostics[0]["num_factors"] > 0
    assert diagnostics[0]["ba_diagnostics"] is not None
    assert diagnostics[0]["ba_diagnostics"]["reason"] != "zero_jacobian"
    assert diagnostics[0]["ba_diagnostics"]["reason"] != "analytic_jacobian_mismatch"
    assert diagnostics[0]["ba_diagnostics"]["max_factor_jacobian_norm"] > 1.0e-8
    assert diagnostics[0]["matching_metadata"]["fibonacci_seed"] == 123


def test_accepted_local_ba_pose_reaches_packet_outputs_and_world_points(tmp_path: Path) -> None:
    config = stage2_default_config()
    config["image"] = {"height": 8, "width": 16, "head_height": 8, "head_width": 16}
    config["head"].update({"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12})
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2_local_ba_writeback.pt"
    torch.save(
        {
            "format": "spherical_selfi_gaussian_head_v1",
            "head": head.state_dict(),
            "adapter_sha256": "synthetic-no-checkpoint",
            "global_step": 0,
            "metrics": {},
            "best_val_psnr": None,
        },
        checkpoint,
    )
    config["stage2_checkpoint"] = {"path": str(checkpoint)}
    config["SphericalSelfiRuntime"] = {
        "enabled": True,
        "feature_device": "cpu",
        "head_device": "cpu",
        "feature_amp": False,
        "window": {"size": 3, "stride": 2, "verification_size": [4, 8]},
        "local_ba": {"enabled": True},
    }
    frontend = SphericalSelfiWindowFrontend(config)
    captured = {}

    def accepted_ba(observation, dense_features, images, static_valid_mask=None):
        del dense_features, images, static_valid_mask
        poses = observation.poses_c2w.detach().clone()
        poses[:, 1] = poses[:, 1] @ se3_exp(
            torch.tensor([0.02, -0.01, 0.005, 0.01, -0.005, 0.003])
        )
        poses[:, 2] = poses[:, 2] @ se3_exp(
            torch.tensor([-0.01, 0.015, 0.004, -0.006, 0.004, 0.008])
        )
        depth = observation.refined_depth.detach().clone() * 1.01
        updated = observation.with_geometry(poses_c2w=poses, refined_depth=depth)
        captured["updated"] = updated
        result = SimpleNamespace(
            poses_c2w=poses,
            dense_depth=depth,
            accepted=torch.tensor([True]),
            initial_median_residual_deg=torch.tensor([2.0]),
            final_median_residual_deg=torch.tensor([1.0]),
            diagnostics=[
                {
                    "reason": "accepted",
                    "initial_objective": 2.0,
                    "final_objective": 1.0,
                    "accepted_steps": 1,
                }
            ],
        )
        return updated, None, result, 0.0, 0.0

    frontend._run_local_ba = accepted_ba
    for frame_id in range(3):
        frontend.track(PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id))
    outputs = frontend.pop_ready_outputs() + frontend.flush()
    packets = frontend.consume_local_gaussian_windows()
    assert len(packets) == 1
    packet = packets[0]
    updated = captured["updated"]
    expected_local = torch.linalg.inv(updated.poses_c2w[0, 0]) @ updated.poses_c2w[0]
    expected_local[0] = torch.eye(4)
    torch.testing.assert_close(packet.local_poses_c2w, expected_local)
    torch.testing.assert_close(packet.local_poses_c2w[0], torch.eye(4))
    by_frame = {int(output.frame_id): output for output in outputs}
    for index in range(3):
        torch.testing.assert_close(by_frame[index].pose_c2w, updated.poses_c2w[0, index])
        torch.testing.assert_close(
            by_frame[index].world_points,
            updated.centers_world()[0, index],
        )
    torch.testing.assert_close(
        by_frame[1].relative_pose,
        torch.linalg.inv(updated.poses_c2w[0, 1]) @ updated.poses_c2w[0, 0],
    )
    torch.testing.assert_close(
        by_frame[2].relative_pose,
        torch.linalg.inv(updated.poses_c2w[0, 2]) @ updated.poses_c2w[0, 1],
    )
    assert all(output.tracking_status == "tracked_spherical_selfi_stage2_ba" for output in outputs)


def test_window_scheduler_has_exact_one_frame_overlap_and_partial_flush(tmp_path: Path) -> None:
    config = stage2_default_config()
    config["image"] = {"height": 8, "width": 16, "head_height": 8, "head_width": 16}
    config["head"].update({"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12})
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2_stride3.pt"
    torch.save(
        {
            "format": "spherical_selfi_gaussian_head_v1",
            "head": head.state_dict(),
            "adapter_sha256": "synthetic-no-checkpoint",
            "global_step": 0,
            "metrics": {},
            "best_val_psnr": None,
        },
        checkpoint,
    )
    config["stage2_checkpoint"] = {"path": str(checkpoint)}
    config["SphericalSelfiRuntime"] = {
        "enabled": True,
        "feature_device": "cpu",
        "head_device": "cpu",
        "feature_amp": False,
        "window": {
            "size": 4,
            "stride": 3,
            "expected_overlap_frames": 1,
            "enforce_exact_overlap": True,
            "verification_size": [4, 8],
        },
        "local_ba": {"enabled": False},
    }
    frontend = SphericalSelfiWindowFrontend(config)
    packets = []
    for frame_id in range(9):
        frontend.track(PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id))
        frontend.pop_ready_outputs()
        packets.extend(frontend.consume_local_gaussian_windows())
    frontend.flush()
    packets.extend(frontend.consume_local_gaussian_windows())
    frontend.flush()
    assert frontend.consume_local_gaussian_windows() == []
    assert [packet.frame_ids for packet in packets] == [
        (0, 1, 2, 3),
        (3, 4, 5, 6),
        (6, 7, 8),
    ]
    assert all(torch.equal(packet.local_poses_c2w[0], torch.eye(4)) for packet in packets)
    assert set(packets[0].frame_ids) & set(packets[1].frame_ids) == {3}
    assert set(packets[1].frame_ids) & set(packets[2].frame_ids) == {6}


def test_packet_hard_sky_defines_finite_gaussian_mask() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    observation, feature = _observation(poses, (0, 1))
    sky_prob = torch.zeros_like(observation.valid_mask, dtype=torch.float32)
    sky_prob[..., :2, :] = 0.9
    packet = LocalGaussianWindowPacket.from_observation(
        window_id=0,
        observation=observation,
        adapter_features=feature,
        frame_ids=(0, 1),
        verification_size=feature.shape[-2:],
        sky_prob=sky_prob,
        sky_threshold=0.5,
    )
    assert packet.sky_mask[..., :2, :].all()
    assert not packet.finite_gaussian_mask[..., :2, :].any()
    assert torch.equal(packet.valid_mask, packet.finite_gaussian_mask)
