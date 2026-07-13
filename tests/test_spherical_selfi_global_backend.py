from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

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
from frontend.spherical_selfi.panorama_loop import circular_yaw_shift, spherical_rotation_ransac
from frontend.spherical_selfi.runtime import SphericalSelfiWindowFrontend
from frontend.spherical_selfi.window_packet import (
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
from geometry.spherical_erp import erp_pixel_to_unit_ray
from models.per_pixel_gaussian_observation import real_sh_basis
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead
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
    flushed = frontend.flush()
    assert [output.frame_id for output in ready + flushed] == [0, 1, 2]
    assert all(output.inverse_depth is not None for output in ready + flushed)
    assert all(output.tracking_status == "tracked_spherical_selfi_stage2" for output in ready + flushed)
    assert len(packets) == 1
    assert packets[0].frame_ids == (0, 1, 2)
    assert packets[0].local_poses_c2w[0].equal(torch.eye(4))


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
