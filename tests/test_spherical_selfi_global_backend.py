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
    Sim3GraphOptimizeResult,
    s2_log_tangent_coordinates,
)
from backend.pano_gs.spherical_selfi_global import (
    ChunkStrideHoldout,
    KnownPoseBridgeFrame,
    OverlapFrameGeometry,
    RenderedSharedFrame,
    SphericalSelfiGlobalBackend,
)
from backend.pano_gs.stage2_global_fusion import (
    Stage2GlobalMapFusion,
    rotate_sh_coefficients,
)
from frontend.pano_droid.spherical_ba import se3_exp
from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel
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
    ChunkStrideMatchBlock,
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
from geometry.spherical_erp import erp_pixel_to_unit_ray, sample_erp_with_wrap
from mapping.gaussian_initializer import GaussianSeedBatch
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
    height: int = 6,
    width: int = 12,
) -> LocalGaussianWindowPacket:
    observation, feature = _observation(
        poses,
        frame_ids,
        height=height,
        width=width,
        feature_by_frame=feature_by_frame,
    )
    return LocalGaussianWindowPacket.from_observation(
        window_id=window_id,
        observation=observation,
        adapter_features=feature,
        frame_ids=frame_ids,
        verification_size=feature.shape[-2:],
    )


def _chunk_stride_backend(
    *,
    min_matches: int = 256,
    skip: bool = False,
    periodic: bool = False,
    hierarchical: bool = False,
    mapper=None,
):
    return SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        mapper=mapper,
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "chunk_first_stride",
                "expected_overlap_frames": 2,
                "optimization_trigger": (
                    "periodic_and_loop" if periodic else "loop_only"
                ),
                "optimization_start_nodes": 6,
                "optimization_interval_edges": 3,
                "active_nodes": 6,
                "min_depth": 0.05,
                "max_depth": 20.0,
                "min_match_cosine": 0.2,
                "max_match_entropy": 1.0,
                "chunk_stride": {
                    "target_index": 2,
                    "holdout_stride": 5,
                    "irls_iterations": 5,
                },
                "skip_edge": {
                    "enabled": bool(skip),
                    "num_queries": max(72, int(min_matches)),
                    "forward_backward": True,
                },
            },
            "voxel_fusion": {
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
            },
            "hierarchical_submaps": {
                "enabled": bool(hierarchical),
                "windows_per_submap": 5,
                "shared_boundary_nodes": 1,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )


def _pointmap_chunk_backend(
    *,
    min_points: int = 32,
    min_points_per_frame: int = 16,
    max_points_per_frame: int = 32,
    acceptance_policy: str = "strict",
    renderer=None,
    packet_refiner=None,
    hierarchical: bool = False,
    alignment_mode: str = "two_frame_pointmap_full_sim3",
) -> SphericalSelfiGlobalBackend:
    return SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        renderer=renderer,
        pose_canonicalized_packet_refiner=packet_refiner,
        config={
            "enabled": True,
            "rendered_overlap_alignment": {
                "enabled": True,
                "mode": str(alignment_mode),
                "min_points": int(min_points),
                "max_points": int(max_points_per_frame) * 2,
                "min_points_per_frame": int(min_points_per_frame),
                "max_points_per_frame": int(max_points_per_frame),
                "holdout_stride": 5,
                "covariance_min_ratio": 1.0e-5,
                "acceptance_policy": str(acceptance_policy),
                "failure_policy": "error",
            },
            "global_graph": {
                "node_mode": "chunk_first_stride",
                "expected_overlap_frames": 2,
                "enforce_exact_overlap": True,
                "optimization_trigger": "loop_only",
                "min_depth": 0.05,
                "max_depth": 20.0,
                "min_match_cosine": 0.2,
                "max_match_entropy": 1.0,
                "min_overlap_points": 8,
                "max_overlap_residual": 1.0e-3,
                "min_overlap_inlier_ratio": 0.95,
                "max_scale_change": 2.5,
                "fibonacci_oversample_factor": 4,
                "umeyama_irls_iterations": 5,
                "chunk_stride": {
                    "target_index": 2,
                    "holdout_stride": 5,
                    "irls_iterations": 5,
                },
                "skip_edge": {"enabled": False},
            },
            "insertion_dedup": {
                "enabled": renderer is not None,
                "visible_only": True,
                "same_level_only": True,
                "compare_existing_only": True,
                "permanent_drop": True,
            },
            "voxel_fusion": {
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
            },
            "hierarchical_submaps": {
                "enabled": bool(hierarchical),
                "windows_per_submap": 5,
                "shared_boundary_nodes": 1,
                "local_camera_model": "se3_shared_scale",
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )


def _replace_packet_depth(
    packet: LocalGaussianWindowPacket,
    depth_by_view: torch.Tensor,
) -> None:
    depth = depth_by_view.to(packet.observation.refined_depth).reshape(
        1,
        len(packet.frame_ids),
        1,
        *packet.observation.image_size,
    )
    packet.observation = packet.observation.with_geometry(refined_depth=depth)


def _attach_identity_stride_matches(
    packet: LocalGaussianWindowPacket,
    *,
    target_index: int = 2,
) -> None:
    height, width = packet.observation.image_size
    row, column = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    uv = torch.stack([column, row], dim=-1).reshape(-1, 2)
    bearing = erp_pixel_to_unit_ray(uv, height, width)
    uv = torch.cat([uv, uv], dim=0)
    bearing = torch.cat([bearing, bearing], dim=0)
    count = int(uv.shape[0])
    direction = torch.cat(
        [
            torch.zeros(count // 2, dtype=torch.long),
            torch.ones(count // 2, dtype=torch.long),
        ]
    )
    packet.chunk_stride_matches = ChunkStrideMatchBlock(
        source_index=0,
        target_index=int(target_index),
        source_uv=uv,
        target_uv=uv.clone(),
        source_bearing=bearing,
        target_bearing=bearing.clone(),
        top1_cosine=torch.ones(count),
        top2_margin=torch.ones(count),
        normalized_entropy=torch.zeros(count),
        query_direction=direction,
    )
    packet.metadata.update(
        {
            "local_ba_accepted": True,
            "local_ba_final_median_residual_deg": 0.1,
            "local_ba_trust_region_touched": False,
        }
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


class _TwoViewUnionRenderer(_SyntheticSharedDepthRenderer):
    """Expose complementary anchor halves in the two dedup views."""

    def render_cameras(self, cameras, gaussians):
        package = super().render_cameras(cameras, gaussians)
        if self.calls < 5:
            return package
        visibility = package["visibility_filter"]
        count = int(visibility.shape[-1])
        visibility.zero_()
        split = count // 2
        if self.calls in {5, 6}:
            visibility[..., :split] = True
        else:
            visibility[..., split:] = True
        return package


class _FirstTwoLocalAnchorsVisibleRenderer(_SyntheticSharedDepthRenderer):
    """Mark only the first two refined anchors visible in every local view."""

    def render_cameras(self, cameras, gaussians):
        package = super().render_cameras(cameras, gaussians)
        if hasattr(gaussians, "anchor_indices"):
            visibility = package["visibility_filter"]
            visibility.zero_()
            visibility[..., : min(2, int(visibility.shape[-1]))] = True
            package["accum_visible"] = visibility.clone()
        return package


class _QueryAttributionRenderer:
    def __init__(self, responses: list[tuple[float, float]]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.seen_queries: list[torch.Tensor] = []

    def render(self, camera, gaussians, *, query_values=None):
        if query_values is None:
            raise AssertionError("query attribution render omitted query_values")
        self.seen_queries.append(query_values.detach().cpu().clone())
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        count = gaussians.anchor_count()
        accumulated = torch.ones(count, device=gaussians.xyz.device)
        answers = torch.empty(count, 2, device=gaussians.xyz.device)
        support = query_values.to(device=gaussians.xyz.device).reshape(-1, 2).amax(dim=0)
        answers[:, 0] = float(response[0]) * support[0]
        answers[:, 1] = float(response[1]) * support[1]
        return {
            "query_answers": answers,
            "accum_visible": accumulated,
            "visibility_filter": torch.ones(
                count, device=gaussians.xyz.device, dtype=torch.bool
            ),
        }


def test_spherical_selfi_global_config_uses_chunk_first_ba8_refiner_mainline() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "spherical_selfi_global_gs_slam.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    local_ba = config["SphericalSelfiRuntime"]["local_ba"]
    outlier = local_ba["outlier_refinement"]
    map_optimization = config["SphericalSelfiGlobalBackend"]["map_optimization"]
    graph = config["SphericalSelfiGlobalBackend"]["global_graph"]
    global_backend = config["SphericalSelfiGlobalBackend"]
    window = config["SphericalSelfiRuntime"]["window"]

    assert local_ba["iterations"] == 8
    assert local_ba["lm_max_trials"] == 8
    assert local_ba["lm_acceptance_eta"] == 1.0e-6
    assert local_ba["residual_worse_tolerance"] == 1.05
    assert local_ba["max_pose_update_deg"] == 30.0
    assert local_ba["max_translation_update"] == 1.0
    assert local_ba["max_logdepth_update"] == 0.70
    assert local_ba["min_factors"] == 128
    assert local_ba["pose_safe_two_stage"] is False
    assert local_ba["defer_dense_depth_affine"] is False
    assert local_ba["dense_depth_mode"] == "affine"
    assert local_ba["dense_depth_output_floor"] == 0.01
    assert outlier["enabled"] is False
    assert outlier["second_stage_iterations"] == 10
    assert outlier["angular_max_deg"] == 5.0
    assert outlier["sim3_max_relative_depth"] == 0.05
    assert outlier["min_inliers"] == 128
    assert outlier["min_inlier_ratio"] == 0.20
    assert outlier["validation_stride"] == 5
    assert outlier["validation_min_inliers"] == 32
    assert outlier["validation_residual_worse_tolerance"] == 1.0
    assert outlier["validation_sim3_worse_tolerance"] == 1.05
    assert local_ba["matching"]["num_queries"] == 1024
    assert local_ba["matching"]["factor_weight_mode"] == "descriptor_confidence"
    assert window["size"] == 4
    assert window["stride"] == 2
    assert window["expected_overlap_frames"] == 2
    assert graph["optimization_start_nodes"] == 6
    assert graph["optimization_interval_edges"] == 3
    assert graph["optimization_enabled"] is True
    assert "enforce_post_optimization_validation" not in graph
    assert graph["expected_overlap_frames"] == 2
    assert graph["node_mode"] == "chunk_first_stride"
    assert graph["optimization_trigger"] == "periodic_and_loop"
    assert graph["chunk_stride"]["target_index"] == 2
    assert graph["chunk_stride"] == {
        "target_index": 2,
        "holdout_stride": 5,
        "irls_iterations": 5,
    }
    assert "min_match_margin" not in graph
    assert graph["skip_edge"]["enabled"] is False
    assert "max_sequence_objective_ratio" not in graph["skip_edge"]
    assert "max_cumulative_scale_change" not in graph["skip_edge"]
    assert "max_translation_update" not in graph
    assert "max_rotation_update_deg" not in graph
    assert "max_log_scale_update" not in graph
    assert graph["normalize_dense_information_by_count"] is True
    assert graph["analytic_dense_linearization"] is True
    assert graph["restrict_objective_to_active_factors"] is True
    assert global_backend["hierarchical_submaps"]["enabled"] is True
    assert global_backend["hierarchical_submaps"]["windows_per_submap"] == 8
    assert global_backend["hierarchical_submaps"]["local_camera_model"] == "se3_shared_scale"
    assert global_backend["hierarchical_submaps"]["compress_frozen_dense_factors"] is True
    assert global_backend["loop_closure"]["descriptor"]["mode"] == "so3_sh_gram"
    assert global_backend["loop_closure"]["insert_pose_factor"] is True
    assert global_backend["loop_closure"]["verification"]["mode"] == "spherical_so3"
    assert global_backend["keyframe_selection"]["enabled"] is True
    assert global_backend["rendered_overlap_alignment"] == {
        "enabled": True,
        "mode": "two_frame_bridge_depth_scale",
        "min_points": 256,
        "max_points": 4096,
        "min_points_per_frame": 256,
        "max_points_per_frame": 2048,
        "alpha_threshold": 0.05,
        "min_confidence": 0.05,
        "min_inlier_ratio": 0.35,
        "max_median_relative_error": 0.10,
        "max_scale_change": 2.5,
        "irls_iterations": 5,
        "holdout_stride": 5,
        "covariance_min_ratio": 1.0e-4,
        "max_rotation_correction_deg": 10.0,
        "max_translation_correction": 1.0,
        "max_shared_rotation_error_deg": 5.0,
        "max_shared_center_error": 0.15,
        "global_map_consistency_max_relative_error": 0.15,
        "global_map_min_consistency_ratio": 0.35,
        "pose_baseline_min": 0.001,
        "post_refiner_scale_recheck": False,
        "post_refiner_scale_rerun_threshold": 0.02,
        "post_refiner_scale_max_relative_change": 0.10,
        "failure_policy": "error",
    }
    assert global_backend["loop_closure"]["exclude_recent_windows"] == 5
    assert global_backend["insertion_dedup"] == {
        "enabled": True,
        "visible_only": True,
        "same_level_only": True,
        "radius_voxels": 1.5,
        "compare_existing_only": True,
        "permanent_drop": True,
        "update_existing_statistics": True,
        "require_new_frame_support": True,
        "max_new_gaussians_per_chunk": 0,
        "coverage_coarse_cell_size": 0.64,
        "log_posthash_coverage": False,
    }
    assert global_backend["insertion_depth_gate"] == {
        "enabled": True,
        "relative_threshold": 0.15,
        "alpha_threshold": 0.05,
    }
    assert global_backend["error_gaussian_prune"] == {
        "enabled": True,
        "relative_depth_threshold": 0.15,
        "alpha_threshold": 0.05,
        "min_depth_confidence": 0.05,
        "responsibility_threshold": 0.8,
        "min_replacement_hits": 2,
        "require_query_attribution": True,
    }
    assert global_backend["post_optimization_seam_check"] == {"enabled": True}
    assert global_backend["voxel_fusion"]["coverage_aware_budget"] is True
    assert global_backend["voxel_fusion"]["max_total_gaussians"] == 3_000_000
    assert map_optimization["lazy_submap_transforms"]["enabled"] is True
    assert map_optimization["loop_neighborhood_refinement"] is True
    assert map_optimization["loop_seam_deduplication"] is True
    assert map_optimization["extra_steps_on_loop"] == 20
    assert map_optimization["pose_lr"] == 2.0e-4
    assert map_optimization["strategy"] == "gaussian_only_joint_3dgs"
    assert map_optimization["pose_refine_enable"] is False
    assert map_optimization["separate_gaussian_lrs"] is True
    assert map_optimization["scale_gaussian_parameter_updates"] is True
    assert map_optimization["steps_per_window"] == 40
    assert map_optimization["recent_window_count"] == 3
    assert map_optimization["sample_observations_per_step"] == 2
    assert map_optimization["sampler"] == "shuffled_cycle"
    assert map_optimization["photometric_only"] is True
    assert map_optimization["optimize_skybox"] is False
    assert map_optimization["pose_prior_weight"] == 0.0
    assert map_optimization["s2_loss_weight"] == 0.0
    assert map_optimization["match_depth_loss_weight"] == 0.0
    assert map_optimization["visible_neighbor_lr_scale"] == 0.0
    assert map_optimization["rgb_l1_weight"] == 0.8
    assert map_optimization["dssim_weight"] == 0.2
    assert map_optimization["max_rgb_worsening"] == 0.005
    assert "appearance" not in map_optimization
    assert "geometry" not in map_optimization
    assert "acceptance" not in map_optimization
    assert "lifecycle_prune_interval_windows" not in global_backend["voxel_fusion"]
    assert "max_stale_frames" not in global_backend["voxel_fusion"]
    assert "max_render_error" not in global_backend["voxel_fusion"]
    assert config["MapRepresentation"]["gaussian_parameterization"] == "traditional_3dgs"
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


def test_pointmap_sim3_config_is_explicit_opt_in() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "spherical_selfi_global_gs_slam_pointmap_sim3.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backend = config["SphericalSelfiGlobalBackend"]
    alignment = backend["rendered_overlap_alignment"]
    graph = backend["global_graph"]

    assert config["base_config"] == "spherical_selfi_global_gs_slam.yaml"
    assert alignment == {
        "enabled": True,
        "mode": "two_frame_pointmap_full_sim3",
        "min_points": 2048,
        "max_points": 4096,
        "min_points_per_frame": 512,
        "max_points_per_frame": 2048,
        "irls_iterations": 5,
        "acceptance_policy": "diagnostics_only",
        "failure_policy": "error",
    }
    assert graph["node_mode"] == "chunk_first_stride"
    assert graph["expected_overlap_frames"] == 2
    assert graph["enforce_exact_overlap"] is True
    assert graph["fibonacci_oversample_factor"] == 8
    assert graph["umeyama_irls_iterations"] == 5
    assert graph["allow_unaligned_fallback"] is False
    assert graph["chunk_stride"]["target_index"] == 2
    assert graph["skip_edge"]["enabled"] is False


def test_ob3d_pointmap_sim3_100_config_logs_both_ate_protocols() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "spherical_selfi_ob3d_pointmap_sim3_adapter_ba_100.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["TrajectoryEvaluation"] == {"ate_mode": "both"}


def test_hash_radius20_configs_only_sweep_the_radius_algorithmically() -> None:
    root = Path(__file__).parents[1] / "configs"
    expected = {
        "spherical_selfi_hash_radius20_r100.yaml": 1.0,
        "spherical_selfi_hash_radius20_r125.yaml": 1.25,
        "spherical_selfi_hash_radius20_r150.yaml": 1.5,
    }
    for name, radius in expected.items():
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["base_config"] == (
            "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_100.yaml"
        )
        assert config["Dataset"] == {"begin": 0, "end": 20}
        backend = config["SphericalSelfiGlobalBackend"]
        assert backend == {
            "insertion_dedup": {
                "radius_voxels": radius,
                "log_posthash_coverage": True,
            }
        }


def test_photometric_recent3_configs_only_change_iteration_count() -> None:
    root = Path(__file__).parents[1] / "configs"
    expected = {
        "spherical_selfi_ob3d_photometric_recent3_iter100.yaml": 100,
        "spherical_selfi_ob3d_photometric_recent3_iter200.yaml": 200,
    }
    for name, steps in expected.items():
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["base_config"] == (
            "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_100.yaml"
        )
        optimization = config["SphericalSelfiGlobalBackend"]["map_optimization"]
        assert optimization == {
            "steps_per_window": steps,
            "recent_window_count": 3,
            "photometric_only": True,
            "optimize_skybox": False,
        }


def test_chunk_stride_factor_uses_holdout_without_ba_quality_rejection() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    packet = _packet(0, poses, (0, 1, 2, 3))
    _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64)

    factor, measurement, holdout, diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )

    assert factor is not None, diagnostics
    assert measurement is not None
    assert holdout is not None
    assert factor.edge_type == "chunk_stride_dense_spherical"
    assert holdout.edge_type == factor.edge_type
    assert diagnostics["accepted"] is True
    assert diagnostics["quality_gating_enabled"] is False
    assert diagnostics["train_matches"] > diagnostics["holdout_matches"] > 0
    assert diagnostics["information_confidence"] > 0.0
    assert 0.0 < diagnostics["s2_information_scale"] <= 1.0
    assert 0.0 < diagnostics["depth_information_scale"] <= 1.0
    assert factor.s2_information_scale == pytest.approx(
        diagnostics["s2_information_scale"]
    )
    assert factor.depth_information_scale == pytest.approx(
        diagnostics["depth_information_scale"]
    )
    torch.testing.assert_close(measurement, sim3_identity(), atol=1.0e-4, rtol=1.0e-4)

    packet.metadata["local_ba_trust_region_touched"] = True
    admitted, _, _, admitted_diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )
    assert admitted is not None
    assert admitted_diagnostics["accepted"] is True
    assert admitted_diagnostics["local_ba_trust_region_touched"] is True


def test_chunk_stride_missing_matches_uses_canonical_ba_pose_fallback() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[2, :3, :3] = se3_exp(
        torch.tensor([0.0, 0.0, 0.0, 0.2, -0.1, 0.05])
    )[:3, :3]
    poses[2, :3, 3] = torch.tensor([1.5, -0.4, 0.2])
    packet = _packet(0, poses, (0, 1, 2, 3))
    backend = _chunk_stride_backend(min_matches=64)

    factor, measurement, holdout, diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )

    assert isinstance(factor, Sim3GraphEdge)
    assert measurement is not None
    assert holdout is None
    assert diagnostics["accepted"] is True
    assert diagnostics["fallback_used"] is True
    assert diagnostics["reason"] == "canonical_ba_pose_fallback"
    torch.testing.assert_close(measurement, poses[2], atol=1.0e-6, rtol=1.0e-6)
    assert float(factor.information_diag[-1]) == 0.0

    result = backend.process_packet(packet)
    assert result.diagnostics["boundary_factor"]["fallback_used"] is True
    assert set(backend.graph.nodes) == {0, 2}
    assert any(
        isinstance(edge, Sim3GraphEdge)
        and edge.edge_type == "chunk_stride_dense_spherical"
        and edge.metadata.get("fallback_used") is True
        for edge in backend.graph.edges
    )


def test_chunk_stride_single_direction_dense_matches_are_admitted() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    packet = _packet(0, poses, (0, 1, 2, 3))
    _attach_identity_stride_matches(packet)
    assert packet.chunk_stride_matches is not None
    packet.chunk_stride_matches = replace(
        packet.chunk_stride_matches,
        query_direction=torch.zeros_like(
            packet.chunk_stride_matches.query_direction
        ),
    )
    backend = _chunk_stride_backend(min_matches=64)

    factor, measurement, _, diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )

    assert isinstance(factor, DenseSphericalFactorBlock), diagnostics
    assert measurement is not None
    assert diagnostics["accepted"] is True
    assert diagnostics["fallback_used"] is False


def test_chunk_stride_factor_does_not_gate_on_top2_margin() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    packet = _packet(0, poses, (0, 1, 2, 3))
    _attach_identity_stride_matches(packet)
    assert packet.chunk_stride_matches is not None
    packet.chunk_stride_matches = replace(
        packet.chunk_stride_matches,
        top2_margin=torch.full_like(
            packet.chunk_stride_matches.top2_margin,
            -1.0,
        ),
    )
    backend = _chunk_stride_backend(min_matches=64)

    factor, measurement, holdout, diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )

    assert factor is not None, diagnostics
    assert measurement is not None
    assert holdout is not None
    assert diagnostics["hard_gated_chunk_stride_matches"] == (
        packet.chunk_stride_matches.count
    )


def test_chunk_stride_factor_does_not_gate_on_holdout_angular_error() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    packet = _packet(0, poses, (0, 1, 2, 3))
    _attach_identity_stride_matches(packet)
    assert packet.chunk_stride_matches is not None
    matches = packet.chunk_stride_matches
    holdout_rows = SphericalSelfiGlobalBackend._chunk_stride_holdout_mask(
        matches.query_direction,
        source_frame=0,
        target_frame=2,
        stride=5,
    )
    corrupted_target = matches.target_bearing.clone()
    corrupted_target[holdout_rows] *= -1.0
    packet.chunk_stride_matches = replace(
        matches,
        target_bearing=corrupted_target,
    )
    backend = _chunk_stride_backend(min_matches=64)

    factor, measurement, holdout, diagnostics = backend._chunk_stride_factor(
        packet,
        expected_target_index=2,
    )

    assert factor is not None, diagnostics
    assert measurement is not None
    assert holdout is not None
    assert diagnostics["accepted"] is True
    assert diagnostics["holdout_median_angular_error_deg"] > 90.0
    assert diagnostics["angular_score"] < 1.0e-3
    assert diagnostics["s2_information_scale"] < diagnostics[
        "depth_information_scale"
    ]
    assert factor.s2_information_scale == pytest.approx(
        diagnostics["s2_information_scale"]
    )


def test_chunk_first_pure_chain_inherits_parent_scale_from_canonical_packet(
    monkeypatch,
) -> None:
    packet = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
        height=32,
        width=64,
    )
    _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)
    original_factor = backend._chunk_stride_factor

    def stride_with_spurious_scale(packet, *, expected_target_index):
        factor, measurement, holdout, diagnostics = original_factor(
            packet,
            expected_target_index=expected_target_index,
        )
        assert factor is not None and measurement is not None and holdout is not None
        spurious_pose = se3_exp(
            torch.tensor([0.8, -0.4, 0.3, 0.2, -0.1, 0.15])
        )
        return (
            factor,
            sim3_from_components(
                1.75,
                spurious_pose[:3, :3],
                spurious_pose[:3, 3],
            ),
            holdout,
            diagnostics,
        )

    monkeypatch.setattr(backend, "_chunk_stride_factor", stride_with_spurious_scale)
    result = backend.process_packet(packet)

    assert result.graph is None
    scales = {
        node: float(sim3_components(transform)[0])
        for node, transform in backend.graph.nodes.items()
    }
    assert set(scales) == {0, 2}
    assert scales[2] == pytest.approx(scales[0])
    expected_next = (
        backend.graph.transform(0)
        @ packet.local_poses_c2w[2].to(backend.graph.transform(0))
    )
    torch.testing.assert_close(backend.graph.transform(2), expected_next)
    stride_edges = [
        edge
        for edge in backend.graph.edges
        if isinstance(edge, DenseSphericalFactorBlock)
        and edge.edge_type == "chunk_stride_dense_spherical"
    ]
    assert len(stride_edges) == 1
    assert stride_edges[0].use_depth is True
    assert stride_edges[0].depth_factor_weight > 0.0


def test_chunk_first_overlap_pose_difference_is_diagnostic_only() -> None:
    packet0 = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
        height=32,
        width=64,
    )
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses1[1] = se3_exp(
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, math.radians(30.0)])
    )
    packet1 = _packet(
        1,
        poses1,
        (2, 3, 4, 5),
        height=32,
        width=64,
    )
    _attach_identity_stride_matches(packet0)
    _attach_identity_stride_matches(packet1)
    backend = _chunk_stride_backend(min_matches=64, skip=False)

    backend.process_packet(packet0)
    result = backend.process_packet(packet1)

    alignment = result.diagnostics["alignment"]
    assert result.aligned is True
    assert alignment["quality_gating_enabled"] is False
    assert alignment["accepted"] is True
    assert alignment["reason"] == "accepted_without_overlap_pose_gate"
    assert alignment["raw_ba_to_canonical_rotation_error_deg"] > 29.0
    assert alignment["raw_ba_to_canonical_center_error"] > 0.9
    assert backend.window_order == [0, 1]
    assert set(backend.graph.nodes) == {0, 2, 4}
    assert backend._last_full_packet is not None
    torch.testing.assert_close(
        backend._last_full_packet.local_poses_c2w[1],
        torch.eye(4),
    )


def test_chunk_first_periodic_ba_starts_at_six_nodes_then_runs_every_three_chunks(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            window_id,
            torch.eye(4).repeat(4, 1, 1),
            (
                2 * window_id,
                2 * window_id + 1,
                2 * window_id + 2,
                2 * window_id + 3,
            ),
            height=32,
            width=64,
        )
        for window_id in range(8)
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)

    backend = _chunk_stride_backend(
        min_matches=64,
        skip=False,
        periodic=True,
    )
    monkeypatch.setattr(backend.loop_detector, "detect", lambda packet: [])
    calls: list[dict[str, object]] = []

    def record_periodic_optimize(active_node_ids=None, *, fixed_node_ids=None):
        calls.append(
            {
                "active": tuple(int(node) for node in active_node_ids),
                "fixed": tuple(sorted(int(node) for node in fixed_node_ids)),
                "scale_locked": bool(backend.graph.lock_scale_updates),
            }
        )
        return Sim3GraphOptimizeResult(
            accepted=False,
            iterations=0,
            initial_objective=0.0,
            final_objective=0.0,
            max_update_norm=0.0,
            optimized_node_ids=(),
            reason="synthetic_no_update",
            final_damping=float(backend.graph.damping),
        )

    monkeypatch.setattr(backend.graph, "optimize", record_periodic_optimize)
    results = [backend.process_packet(packet) for packet in packets]

    assert [index for index, result in enumerate(results) if result.graph] == [4, 7]
    assert calls == [
        {
            "active": (0, 2, 4, 6, 8, 10),
            "fixed": (0,),
            "scale_locked": True,
        },
        {
            "active": (6, 8, 10, 12, 14, 16),
            "fixed": (6,),
            "scale_locked": True,
        },
    ]
    for result in (results[4], results[7]):
        periodic = result.diagnostics["boundary_factor"][
            "periodic_optimization"
        ]
        assert periodic["attempted"] is True
        assert periodic["accepted"] is False
        assert periodic["scale_locked"] is True
        assert periodic["reason"] == "synthetic_no_update"
    edge_types = [edge.edge_type for edge in backend.graph.edges]
    assert edge_types.count("chunk_stride_dense_spherical") == 8
    assert "chunk_skip_dense_spherical" not in edge_types
    scales = [
        float(sim3_components(backend.graph.transform(node))[0])
        for node in sorted(backend.graph.nodes)
    ]
    assert scales == pytest.approx([scales[0]] * len(scales))


def test_periodic_graph_geometry_failures_are_diagnostic_but_state_is_hard(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            window_id,
            torch.eye(4).repeat(4, 1, 1),
            (
                2 * window_id,
                2 * window_id + 1,
                2 * window_id + 2,
                2 * window_id + 3,
            ),
            height=32,
            width=64,
        )
        for window_id in range(5)
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)

    backend = _chunk_stride_backend(
        min_matches=64,
        skip=False,
        periodic=True,
    )
    backend.post_optimization_seam_check_enabled = True
    monkeypatch.setattr(backend.loop_detector, "detect", lambda packet: [])
    moved: dict[str, object] = {}

    def accepted_optimize(active_node_ids=None, *, fixed_node_ids=None):
        node = int(active_node_ids[-1])
        transform = backend.graph.transform(node)
        scale, rotation, translation = sim3_components(transform)
        updated_rotation = (
            se3_exp(
                rotation.new_tensor([0.0, 0.0, 0.0, 0.1, -0.05, 0.02])
            )[:3, :3]
            @ rotation
        )
        updated = sim3_from_components(
            scale,
            updated_rotation,
            translation + translation.new_tensor([0.25, 0.0, 0.0]),
        )
        backend.graph.nodes[node] = updated
        moved["node"] = node
        moved["transform"] = updated.clone()
        return Sim3GraphOptimizeResult(
            accepted=True,
            iterations=1,
            initial_objective=1.0,
            final_objective=0.5,
            max_update_norm=0.25,
            optimized_node_ids=tuple(int(value) for value in active_node_ids),
            reason="synthetic_accepted",
            final_damping=float(backend.graph.damping),
        )

    monkeypatch.setattr(backend.graph, "optimize", accepted_optimize)
    monkeypatch.setattr(
        backend,
        "_overlap_seam_diagnostics",
        lambda: (_ for _ in ()).throw(
            AssertionError("chunk-first must not compare stale packet seams")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_chunk_stride_holdout_diagnostics",
        lambda *, affected_node_ids=None: {
            "enabled": True,
            "accepted": False,
            "factor_count": 1,
            "reason": "synthetic_holdout_failure",
        },
    )

    results = [backend.process_packet(packet) for packet in packets]
    result = results[-1]

    assert result.graph is not None
    assert result.graph.accepted is True
    assert result.graph.reason == "synthetic_accepted"
    assert moved
    torch.testing.assert_close(
        backend.graph.transform(int(moved["node"])),
        moved["transform"],
    )
    seam = result.diagnostics["post_optimization_seam_check"]
    holdout = result.diagnostics["chunk_stride_holdout_check"]
    assert seam["mode"] == "canonical_pose_state_consistency"
    assert seam["accepted"] is True
    assert seam["enforced"] is True
    assert seam["candidate_revision"] == seam["committed_revision"]
    moved_node = int(moved["node"])
    in_flight = backend._last_full_packet
    assert in_flight is not None
    assert in_flight.metadata["canonical_pose_revision"] == seam[
        "committed_revision"
    ]
    in_flight_pose = in_flight.global_poses(
        backend._window_anchor_transforms()[int(in_flight.window_id)].to(
            in_flight.local_poses_c2w
        )
    )[in_flight.frame_index(moved_node)]
    canonical_pose = apply_sim3_to_c2w(
        backend.graph.transform(moved_node).to(
            backend.frame_local_pose_in_owner[moved_node]
        ),
        backend.frame_local_pose_in_owner[moved_node],
    )
    torch.testing.assert_close(in_flight_pose, canonical_pose)
    assert result.fusion["pose_state_committed_revision"] == seam[
        "committed_revision"
    ]
    assert backend._pending_geometry_batch is not None
    assert backend._pending_geometry_batch.complete_snapshot is True
    assert backend._pending_geometry_batch.revision == seam[
        "committed_revision"
    ]
    assert holdout["accepted"] is False
    assert holdout["enforced"] is False


def test_legacy_overlap_seam_magnitude_is_diagnostic_only() -> None:
    backend = _boundary_backend(PanoGaussianMap(config={}, device="cpu"))
    backend.post_optimization_seam_check_enabled = True
    backend.graph.add_node(0, sim3_identity())
    rotation = se3_exp(
        torch.tensor([0.0, 0.0, 0.0, 0.0, math.radians(60.0), 0.0])
    )[:3, :3]
    backend.graph.add_node(
        1,
        sim3_from_components(1.0, rotation, torch.tensor([3.0, 0.0, 0.0])),
    )
    backend.graph.add_edge(
        CoincidentPanoramaFactor(
            source=0,
            target=1,
            source_local_pose=torch.eye(4),
            target_local_pose=torch.eye(4),
            measured_source_to_target_rotation=torch.eye(3),
            edge_type="overlap_shared_pose_consistency",
        )
    )

    diagnostics = backend._overlap_seam_diagnostics()

    assert diagnostics["max_rotation_error_deg"] > 5.0
    assert diagnostics["max_center_error"] > 0.15
    assert diagnostics["quality_gating_enabled"] is False
    assert diagnostics["accepted"] is True


def test_periodic_graph_holdout_cannot_restore_an_accepted_lm_state(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            window_id,
            torch.eye(4).repeat(4, 1, 1),
            (
                2 * window_id,
                2 * window_id + 1,
                2 * window_id + 2,
                2 * window_id + 3,
            ),
            height=32,
            width=64,
        )
        for window_id in range(5)
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)

    backend = _chunk_stride_backend(
        min_matches=64,
        skip=False,
        periodic=True,
    )
    backend.post_optimization_seam_check_enabled = True
    monkeypatch.setattr(backend.loop_detector, "detect", lambda packet: [])
    state: dict[str, object] = {}

    def accepted_optimize(active_node_ids=None, *, fixed_node_ids=None):
        node = int(active_node_ids[-1])
        before = backend.graph.transform(node).clone()
        scale, rotation, translation = sim3_components(before)
        backend.graph.nodes[node] = sim3_from_components(
            scale,
            rotation,
            translation + translation.new_tensor([0.25, 0.0, 0.0]),
        )
        state["node"] = node
        state["before"] = before
        return Sim3GraphOptimizeResult(
            accepted=True,
            iterations=1,
            initial_objective=1.0,
            final_objective=0.5,
            max_update_norm=0.25,
            optimized_node_ids=tuple(int(value) for value in active_node_ids),
            reason="synthetic_accepted",
            final_damping=float(backend.graph.damping),
        )

    monkeypatch.setattr(backend.graph, "optimize", accepted_optimize)
    monkeypatch.setattr(
        backend,
        "_chunk_stride_holdout_diagnostics",
        lambda *, affected_node_ids=None: {
            "enabled": True,
            "accepted": False,
            "quality_gating_enabled": False,
            "factor_count": 1,
            "reason": "synthetic_holdout_degradation",
        },
    )
    results = [backend.process_packet(packet) for packet in packets]
    result = results[-1]

    assert result.graph is not None
    assert result.graph.accepted is True
    assert result.graph.reason == "synthetic_accepted"
    assert not torch.allclose(
        backend.graph.transform(int(state["node"])),
        state["before"],
    )
    holdout = result.diagnostics["chunk_stride_holdout_check"]
    assert holdout["accepted"] is False
    assert holdout["enforced"] is False


def test_post_candidate_fusion_failure_restores_pose_state_transaction(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            window_id,
            torch.eye(4).repeat(4, 1, 1),
            (
                2 * window_id,
                2 * window_id + 1,
                2 * window_id + 2,
                2 * window_id + 3,
            ),
            height=32,
            width=64,
        )
        for window_id in range(5)
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(
        min_matches=64,
        skip=False,
        periodic=True,
    )
    monkeypatch.setattr(backend.loop_detector, "detect", lambda packet: [])
    for packet in packets[:4]:
        backend.process_packet(packet)
    backend.pop_frame_geometry_update_batch()

    graph_before = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }
    packet_before = {
        id(packet): (
            packet.local_poses_c2w.clone(),
            dict(packet.metadata),
        )
        for window_id in backend.window_order
        for packet in backend._packet_variants(window_id)
    }
    map_count_before = backend.map.anchor_count()
    lazy_owner_before = {
        owner: transform.clone()
        for owner, transform in backend.map._lazy_owner_current_transforms.items()
    }
    window_order_before = list(backend.window_order)
    geometry_revision_before = backend._geometry_revision
    pose_revision_before = backend._pose_state_revision

    def accepted_optimize(active_node_ids=None, *, fixed_node_ids=None):
        node = int(active_node_ids[-1])
        transform = backend.graph.transform(node)
        scale, rotation, translation = sim3_components(transform)
        backend.graph.nodes[node] = sim3_from_components(
            scale,
            rotation,
            translation + translation.new_tensor([0.2, 0.0, 0.0]),
        )
        return Sim3GraphOptimizeResult(
            accepted=True,
            iterations=1,
            initial_objective=1.0,
            final_objective=0.5,
            max_update_norm=0.2,
            optimized_node_ids=tuple(int(value) for value in active_node_ids),
            reason="synthetic_accepted",
            final_damping=float(backend.graph.damping),
        )

    monkeypatch.setattr(backend.graph, "optimize", accepted_optimize)
    monkeypatch.setattr(
        backend.fusion,
        "fuse_packet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic post-candidate fusion failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic post-candidate fusion failure",
    ):
        backend.process_packet(packets[-1])

    assert backend.window_order == window_order_before
    assert backend.map.anchor_count() == map_count_before
    assert set(backend.map._lazy_owner_current_transforms) == set(
        lazy_owner_before
    )
    for owner, transform in lazy_owner_before.items():
        torch.testing.assert_close(
            backend.map._lazy_owner_current_transforms[owner],
            transform,
        )
    assert backend._geometry_revision == geometry_revision_before
    assert backend._pose_state_revision == pose_revision_before
    for node, transform in graph_before.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)
    for window_id in backend.window_order:
        for packet in backend._packet_variants(window_id):
            local_poses, metadata = packet_before[id(packet)]
            torch.testing.assert_close(packet.local_poses_c2w, local_poses)
            assert packet.metadata == metadata
    failure = backend._last_pose_state_diagnostic
    assert failure is not None
    assert failure["transaction_committed"] is False
    assert failure["candidate_revision"] > failure["committed_revision"]


def test_chunk_stride_holdout_validation_only_checks_affected_edges(
    monkeypatch,
) -> None:
    backend = _chunk_stride_backend(min_matches=64, skip=True)
    for node in (0, 2, 4, 6):
        backend.graph.add_node(node, sim3_identity())

    def holdout(source: int, target: int, marker: float) -> ChunkStrideHoldout:
        bearing = torch.tensor([[marker, 0.0, 1.0]])
        return ChunkStrideHoldout(
            source=source,
            target=target,
            edge_type="chunk_stride_dense_spherical",
            source_bearing=bearing,
            target_bearing=bearing.clone(),
            source_depth=torch.ones(1),
            target_depth=torch.ones(1),
            initial_angular_median_deg=0.1,
            initial_relative_depth_median=0.01,
        )

    backend._chunk_stride_holdouts = {
        (0, 2, "chunk_stride_dense_spherical"): holdout(0, 2, 9.0),
        (4, 6, "chunk_stride_dense_spherical"): holdout(4, 6, 1.0),
    }

    def synthetic_errors(relative, source_bearing, *args):
        is_bad = float(source_bearing[0, 0]) > 5.0
        angular = source_bearing.new_tensor([5.0 if is_bad else 0.1])
        depth = source_bearing.new_tensor([0.5 if is_bad else 0.01])
        return angular, depth

    monkeypatch.setattr(
        backend,
        "_chunk_stride_alignment_errors",
        synthetic_errors,
    )
    affected = backend._chunk_stride_holdout_diagnostics(
        affected_node_ids={4},
    )
    all_edges = backend._chunk_stride_holdout_diagnostics()

    assert affected["factor_count"] == 1
    assert affected["accepted"] is True
    assert affected["quality_gating_enabled"] is False
    assert affected["per_edge"][0]["source"] == 4
    assert all_edges["factor_count"] == 2
    assert all_edges["accepted"] is True
    assert all_edges["quality_gating_enabled"] is False
    assert all_edges["max_relative_depth_median"] == pytest.approx(0.5)


def test_overlap_scale_uses_only_canonical_and_raw_ba_depths() -> None:
    previous = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    current = _packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
    )
    previous_depth = torch.full_like(previous.observation.refined_depth, 4.0)
    current_depth = torch.full_like(current.observation.refined_depth, 2.0)
    previous.observation = previous.observation.with_geometry(
        refined_depth=previous_depth
    )
    current.observation = current.observation.with_geometry(
        refined_depth=current_depth
    )
    backend = _chunk_stride_backend(min_matches=32)
    backend.rendered_alignment_min_points_per_frame = 32
    backend.rendered_alignment_max_points_per_frame = 64

    scale, diagnostics = backend._estimate_canonical_ba_overlap_scale(
        previous,
        current,
    )

    assert scale == pytest.approx(2.0, rel=5.0e-5)
    assert diagnostics["frame_weight"] == 0.5
    assert diagnostics["irls_iterations"] == 5
    assert diagnostics["global_render_used_for_scale"] is False
    assert diagnostics["post_refiner_scale_recheck"] is False


def test_overlap_scale_hard_rejects_low_confidence_depth_outliers() -> None:
    previous = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    current = _packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
    )
    previous_depth = torch.full_like(previous.observation.refined_depth, 4.0)
    current_depth = torch.full_like(current.observation.refined_depth, 2.0)
    previous_confidence = torch.ones_like(previous.observation.confidence)
    current_confidence = torch.ones_like(current.observation.confidence)
    for frame_id in (2, 3):
        previous_index = previous.frame_index(frame_id)
        current_index = current.frame_index(frame_id)
        previous_depth[0, previous_index, :, :2, :] = 18.0
        current_depth[0, current_index, :, :2, :] = 1.0
        previous_confidence[0, previous_index, :, :2, :] = 0.0
        current_confidence[0, current_index, :, :2, :] = 0.0
    previous.observation = replace(
        previous.observation.with_geometry(refined_depth=previous_depth),
        confidence=previous_confidence,
    )
    current.observation = replace(
        current.observation.with_geometry(refined_depth=current_depth),
        confidence=current_confidence,
    )
    backend = _chunk_stride_backend(min_matches=32)
    backend.rendered_alignment_min_points_per_frame = 32
    backend.rendered_alignment_max_points_per_frame = 64
    backend.rendered_alignment_min_confidence = 0.05

    scale, diagnostics = backend._estimate_canonical_ba_overlap_scale(
        previous,
        current,
    )

    assert scale == pytest.approx(2.0, rel=5.0e-5)
    assert diagnostics["min_confidence"] == pytest.approx(0.05)
    assert diagnostics["per_frame_confidence_rejected_pixels"] == [24, 24]
    assert diagnostics["per_frame_confidence_support_pixels"] == [48, 48]


def test_chunk_first_nodes_rigidly_publish_both_frames_in_each_segment() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[1, 0, 3] = 0.1
    poses[2, 0, 3] = 0.2
    poses[3, 0, 3] = 0.3
    packet = _packet(0, poses, (0, 1, 2, 3))
    backend = _chunk_stride_backend(min_matches=64)
    backend.graph.add_node(0, sim3_identity())
    backend.graph.add_node(
        2,
        sim3_from_components(
            1.0,
            torch.eye(3),
            torch.tensor([0.2, 0.0, 0.0]),
        ),
    )
    backend.window_order = [0]
    backend.window_anchor_nodes[0] = 0
    backend.packets[0] = packet
    for frame_id in packet.frame_ids:
        backend.frame_windows[int(frame_id)] = {0}
        backend.frame_owner_window[int(frame_id)] = 0
        backend.frame_depth_owner_window[int(frame_id)] = 0
    diagnostics = backend._register_chunk_stride_segments(
        packet,
        source_node=0,
        target_node=2,
    )
    assert diagnostics["registered_frames"] == 4
    assert backend.frame_pose_owner_node == {0: 0, 1: 0, 2: 2, 3: 2}

    backend._refresh_geometry_updates()
    initial = backend.pop_frame_geometry_update_batch()
    assert initial is not None and initial.complete_snapshot
    node = backend.graph.transform(0).clone()
    node[:3, 3] += torch.tensor([1.0, 0.0, 0.0])
    backend.graph.nodes[0] = node
    backend._refresh_geometry_updates()
    updated = backend.pop_frame_geometry_update_batch()
    assert updated is not None
    for frame_id in (0, 1):
        delta = (
            updated.updates[frame_id].pose_c2w[:3, 3]
            - initial.updates[frame_id].pose_c2w[:3, 3]
        )
        torch.testing.assert_close(delta, torch.tensor([1.0, 0.0, 0.0]))
    for frame_id in (2, 3):
        torch.testing.assert_close(
            updated.updates[frame_id].pose_c2w,
            initial.updates[frame_id].pose_c2w,
        )


def test_mapper_internal_frame_motion_ignores_geometry_quality_gates(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)
    backend.post_optimization_seam_check_enabled = True
    backend.process_packet(packets[0])
    backend.process_packet(packets[1])
    backend.pop_frame_geometry_update_batch()
    assert backend._last_full_packet is not None
    backend._optimization_packets[1] = replace(
        backend._last_full_packet,
        local_poses_c2w=backend._last_full_packet.local_poses_c2w.clone(),
        metadata=dict(backend._last_full_packet.metadata),
    )
    for window_id in backend.window_order:
        variant = backend.packets[window_id]
        refined = _refined_packet(
            window_id,
            variant.local_poses_c2w,
            variant.frame_ids,
        )
        assert refined.anchor_observation is not None
        for packet_variant in backend._packet_variants(window_id):
            packet_variant.anchor_observation = replace(
                refined.anchor_observation,
                local_poses_c2w=packet_variant.local_poses_c2w.unsqueeze(0),
            )

    proposals: dict[int, torch.Tensor] = {}
    for frame_id, owner in backend.frame_pose_owner_node.items():
        local = backend.frame_local_pose_in_owner[frame_id]
        proposals[frame_id] = apply_sim3_to_c2w(
            backend.graph.transform(owner).to(local),
            local,
        )
    proposals[3] = proposals[3].clone()
    proposals[3][:3, 3] += torch.tensor([0.20, 0.0, 0.0])

    class MapperProposal:
        optimizer = None
        stats = SimpleNamespace(n_anchors=backend.map.anchor_count())

        def refined_pose_c2w(self, frame_id: int):
            return proposals[int(frame_id)]

    backend.mapper = MapperProposal()
    revision_before = backend._geometry_revision
    objective_values = iter((1.0, 1.0, 100.0, 100.0))
    monkeypatch.setattr(
        backend.graph,
        "objective",
        lambda *, factors=None: torch.tensor(next(objective_values)),
    )
    monkeypatch.setattr(
        backend,
        "_chunk_stride_holdout_diagnostics",
        lambda *, affected_node_ids=None: {
            "enabled": True,
            "accepted": False,
            "quality_gating_enabled": False,
            "factor_count": 1,
            "reason": "synthetic_geometry_degradation",
        },
    )

    backend._synchronize_chunk_stride_optimized_window(1)

    assert backend._geometry_revision == revision_before + 1
    assert backend._pose_state_revision == backend._geometry_revision
    state = backend._last_pose_state_diagnostic
    assert state is not None
    assert state["accepted"] is True
    assert state["max_matrix_error"] <= 1.0e-5
    assert state["max_center_error"] <= 1.0e-5
    committed = backend._last_mapper_committed_state_diagnostic
    assert committed is not None
    quality = committed["geometric_quality_diagnostics"]
    assert quality["quality_gating_enabled"] is False
    assert quality["sequence_objective_ratio"] == pytest.approx(100.0)
    assert quality["skip_objective_ratio"] == pytest.approx(100.0)
    assert quality["holdout"]["accepted"] is False
    for window_id in (0, 1):
        packet = backend.packets[window_id]
        frame_index = packet.frame_index(3)
        packet_pose = apply_sim3_to_c2w(
            backend._window_anchor_transforms()[window_id].to(
                packet.local_poses_c2w
            ),
            packet.local_poses_c2w[frame_index],
        )
        torch.testing.assert_close(packet_pose, proposals[3])
        assert packet.anchor_observation is not None
        torch.testing.assert_close(
            packet.anchor_observation.local_poses_c2w[0],
            packet.local_poses_c2w,
        )
        for packet_variant in backend._packet_variants(window_id):
            assert packet_variant.anchor_observation is not None
            torch.testing.assert_close(
                packet_variant.anchor_observation.local_poses_c2w[0],
                packet_variant.local_poses_c2w,
            )


def test_mapper_internal_frame_commit_updates_all_packet_variants_and_full_batch() -> None:
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)
    backend.post_optimization_seam_check_enabled = True
    backend.process_packet(packets[0])
    backend.process_packet(packets[1])
    backend.pop_frame_geometry_update_batch()

    proposals: dict[int, torch.Tensor] = {}
    for frame_id, owner in backend.frame_pose_owner_node.items():
        local = backend.frame_local_pose_in_owner[frame_id]
        proposals[frame_id] = apply_sim3_to_c2w(
            backend.graph.transform(owner).to(local),
            local,
        )
    proposals[3] = proposals[3].clone()
    proposals[3][:3, 3] += torch.tensor([0.05, 0.0, 0.0])

    class MapperProposal:
        optimizer = None
        stats = SimpleNamespace(n_anchors=backend.map.anchor_count())

        def refined_pose_c2w(self, frame_id: int):
            return proposals[int(frame_id)]

    backend.mapper = MapperProposal()
    factor_measurements = {
        id(factor): (
            factor.source_local_pose.clone(),
            factor.target_local_pose.clone(),
        )
        for factor in backend.graph.edges
        if isinstance(factor, DenseSphericalFactorBlock)
    }
    backend._synchronize_chunk_stride_optimized_window(1)

    batch = backend.pop_frame_geometry_update_batch()
    assert batch is not None and batch.complete_snapshot is True
    assert set(batch.updates) == {0, 1, 2, 3, 4, 5}
    torch.testing.assert_close(
        batch.updates[3].pose_c2w[:3, 3],
        proposals[3][:3, 3],
    )
    for window_id in (0, 1):
        packet = backend.packets[window_id]
        frame_index = packet.frame_index(3)
        packet_pose = apply_sim3_to_c2w(
            backend._window_anchor_transforms()[window_id].to(
                packet.local_poses_c2w
            ),
            packet.local_poses_c2w[frame_index],
        )
        torch.testing.assert_close(packet_pose, proposals[3])
    for factor in backend.graph.edges:
        poses = factor_measurements.get(id(factor))
        if poses is None:
            continue
        torch.testing.assert_close(factor.source_local_pose, poses[0])
        torch.testing.assert_close(factor.target_local_pose, poses[1])


def test_mapper_recent_three_window_commit_updates_all_eight_canonical_frames() -> None:
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
        _packet(
            2,
            torch.eye(4).repeat(4, 1, 1),
            (4, 5, 6, 7),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)
    for packet in packets:
        backend.process_packet(packet)
    backend.pop_frame_geometry_update_batch()

    proposals: dict[int, torch.Tensor] = {}
    for frame_id, owner in backend.frame_pose_owner_node.items():
        local = backend.frame_local_pose_in_owner[frame_id]
        proposals[frame_id] = apply_sim3_to_c2w(
            backend.graph.transform(owner).to(local),
            local,
        )
    for frame_id, offset in ((1, 0.01), (3, 0.02), (7, 0.03)):
        proposals[frame_id] = proposals[frame_id].clone()
        proposals[frame_id][0, 3] += offset

    class MapperProposal:
        optimizer = None
        stats = SimpleNamespace(n_anchors=backend.map.anchor_count())

        def refined_pose_c2w(self, frame_id: int):
            return proposals[int(frame_id)]

    backend.mapper = MapperProposal()
    backend._synchronize_chunk_stride_optimized_window(
        2,
        optimized_frame_ids=tuple(range(8)),
    )

    batch = backend.pop_frame_geometry_update_batch()
    assert batch is not None and batch.complete_snapshot is True
    assert set(batch.updates) == set(range(8))
    for frame_id in range(8):
        torch.testing.assert_close(
            batch.updates[frame_id].pose_c2w,
            proposals[frame_id],
        )
    for window_id in (0, 1, 2):
        packet = backend.packets[window_id]
        window_transform = backend._window_anchor_transforms()[window_id]
        for frame_id in packet.frame_ids:
            packet_pose = apply_sim3_to_c2w(
                window_transform.to(packet.local_poses_c2w),
                packet.local_poses_c2w[packet.frame_index(frame_id)],
            )
            torch.testing.assert_close(packet_pose, proposals[int(frame_id)])


def test_pose_state_consistency_failure_rolls_back_graph_packets_and_revision(
    monkeypatch,
) -> None:
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)
    backend.process_packet(packets[0])
    backend.process_packet(packets[1])
    backend.pop_frame_geometry_update_batch()

    proposals: dict[int, torch.Tensor] = {}
    for frame_id, owner in backend.frame_pose_owner_node.items():
        local = backend.frame_local_pose_in_owner[frame_id]
        proposals[frame_id] = apply_sim3_to_c2w(
            backend.graph.transform(owner).to(local),
            local,
        )
    proposals[3] = proposals[3].clone()
    proposals[3][0, 3] += 0.1

    class MapperProposal:
        optimizer = None
        stats = SimpleNamespace(n_anchors=backend.map.anchor_count())

        def refined_pose_c2w(self, frame_id: int):
            return proposals[int(frame_id)]

    backend.mapper = MapperProposal()
    graph_before = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }
    local_before = {
        frame: pose.clone()
        for frame, pose in backend.frame_local_pose_in_owner.items()
    }
    packet_before = {
        id(packet): (
            packet.local_poses_c2w.clone(),
            dict(packet.metadata),
        )
        for window_id in backend.window_order
        for packet in backend._packet_variants(window_id)
    }
    geometry_revision_before = backend._geometry_revision
    pose_revision_before = backend._pose_state_revision
    original_report = backend._pose_state_consistency_report

    def reject_report(candidate, *, extra_packets=()):
        report = original_report(
            candidate,
            extra_packets=extra_packets,
        )
        return replace(
            report,
            accepted=False,
            reason="synthetic_state_mismatch",
        )

    monkeypatch.setattr(
        backend,
        "_pose_state_consistency_report",
        reject_report,
    )

    with pytest.raises(
        RuntimeError,
        match="Canonical pose-state consistency failed",
    ):
        backend._synchronize_chunk_stride_optimized_window(1)

    assert backend._geometry_revision == geometry_revision_before
    assert backend._pose_state_revision == pose_revision_before
    assert backend._pending_geometry_batch is None
    for node, transform in graph_before.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)
    for frame_id, local_pose in local_before.items():
        torch.testing.assert_close(
            backend.frame_local_pose_in_owner[frame_id],
            local_pose,
        )
    for window_id in backend.window_order:
        for packet in backend._packet_variants(window_id):
            local_poses, metadata = packet_before[id(packet)]
            torch.testing.assert_close(packet.local_poses_c2w, local_poses)
            assert packet.metadata == metadata


def test_pose_state_rotation_error_is_stable_at_float32_round_trip_scale() -> None:
    rotation = se3_exp(
        torch.tensor([0.0, 0.0, 0.0, 0.3, -0.2, 0.1])
    )[:3, :3]
    estimate = rotation.clone()
    estimate[0, 0] -= 5.0e-7

    assert float((estimate - rotation).abs().max()) <= 1.0e-5
    assert (
        SphericalSelfiGlobalBackend._pose_state_rotation_error_deg(
            rotation,
            estimate,
        )
        <= 1.0e-4
    )


def test_hierarchical_pose_candidate_refreshes_local_cache_before_check() -> None:
    backend = _chunk_stride_backend(
        min_matches=64,
        skip=False,
        hierarchical=True,
    )
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
        backend.process_packet(packet)

    assert backend.submap_graph is not None
    old_window_transforms = backend._window_anchor_transforms()
    moved = backend.graph.transform(4).clone()
    moved[1, 3] += 0.2
    backend.graph.nodes[4] = moved

    candidate, report, _ = backend._materialize_pose_state_candidate(
        affected_node_ids={4},
        affected_submap_ids={0},
        old_window_transforms=old_window_transforms,
        reason="synthetic_hierarchical_candidate",
    )

    assert candidate.affected_submap_ids == (0,)
    assert report.accepted is True
    assert report.max_submap_matrix_error <= 1.0e-5
    record = backend.submaps[0]
    reconstructed = (
        backend.submap_graph.transform(0).to(
            record.local_boundary_transforms[4]
        )
        @ record.local_boundary_transforms[4]
    )
    torch.testing.assert_close(reconstructed, backend.graph.transform(4))


def test_skip_factor_recomputes_independent_c0_to_c2_correspondences() -> None:
    shared = torch.randn(24, 32, 64)
    features = {0: shared, 4: shared}
    previous = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
        feature_by_frame=features,
        height=32,
        width=64,
    )
    current = _packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
        feature_by_frame=features,
        height=32,
        width=64,
    )
    backend = _chunk_stride_backend(min_matches=32, skip=True)
    backend.chunk_skip_forward_backward = False
    backend.graph.add_node(0, sim3_identity())
    backend.graph.add_node(4, sim3_identity())

    factor, holdout, diagnostics = backend._independent_chunk_skip_factor(
        previous,
        current,
    )

    assert factor is not None, diagnostics
    assert holdout is not None
    assert factor.edge_type == "chunk_skip_dense_spherical"
    assert (factor.source, factor.target) == (0, 4)
    assert diagnostics["independent_from_sequential_and_overlap"] is True
    assert diagnostics["accepted"] is True


def test_post_hash_incoming_budget_is_coverage_first_and_level_separated() -> None:
    packet = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    fusion = Stage2GlobalMapFusion(
        PanoGaussianMap(config={}, device="cpu"),
        voxel_sizes=(0.04, 0.08, 0.16, 0.32),
        min_confidence=0.0,
        min_opacity=0.0,
    )
    prepared = fusion.prepare_packet_batch(packet, sim3_identity())
    assert len(prepared.batch) >= 6
    prepared = prepared.index(torch.arange(6, device=prepared.batch.xyz.device))
    prepared = replace(
        prepared,
        batch=replace(
            prepared.batch,
            xyz=torch.zeros_like(prepared.batch.xyz),
            level=torch.tensor(
                [0, 1, 0, 1, 0, 1],
                device=prepared.batch.level.device,
            ),
            quality=torch.tensor(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                device=prepared.batch.quality.device,
            ),
        ),
    )

    limited, stats = fusion.limit_prepared_incoming_by_coverage(
        prepared,
        max_new_gaussians=2,
        coarse_cell_size=0.64,
    )

    assert len(limited.batch) == 2
    assert set(limited.batch.level.tolist()) == {0, 1}
    assert stats["incoming_budget_dropped"] == 4
    assert stats["incoming_budget_same_level_only"] == 1


def test_new_frame_support_and_four_view_visibility_jointly_gate_first_fusion() -> None:
    packet = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    packet.metadata["voxel_anchor_refiner_requested"] = True
    packet.metadata["voxel_anchor_refiner_pending"] = True

    def finalize_refiner(value: LocalGaussianWindowPacket):
        images = torch.zeros(
            1,
            len(value.frame_ids),
            3,
            *value.observation.image_size,
        )
        anchors = voxelize_per_pixel_gaussians(
            value.observation,
            value.adapter_features,
            images,
            VoxelAnchorConfig(
                use_resnet_error=False,
                pretrained_resnet=False,
            ),
            valid_mask=value.finite_gaussian_mask,
        ).detach_for_backend()
        assert anchors.num_anchors >= 3
        source_view_mask = torch.zeros(anchors.num_anchors, dtype=torch.long)
        # Anchor 1 is supported and visible; anchor 2 is supported but not
        # visible. Anchor 0 is visible but lacks new-frame support.
        source_view_mask[1:3] = 1
        metadata = dict(value.metadata)
        metadata.update(
            {
                "voxel_anchor_refiner_pending": False,
                "voxel_anchor_refiner_enabled": True,
                "voxel_anchor_source_view_mask": source_view_mask,
            }
        )
        return replace(
            value,
            anchor_observation=anchors,
            metadata=metadata,
        )

    _attach_identity_stride_matches(packet)
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        renderer=_FirstTwoLocalAnchorsVisibleRenderer(
            local_depth=2.0,
            global_depth=2.0,
        ),
        pose_canonicalized_packet_refiner=finalize_refiner,
        config={
            "enabled": True,
            "rendered_overlap_alignment": {
                "enabled": True,
                "mode": "two_frame_bridge_depth_scale",
                "min_points": 32,
                "max_points": 128,
                "min_points_per_frame": 32,
                "max_points_per_frame": 64,
                "post_refiner_scale_recheck": False,
            },
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "radius_voxels": 1.5,
                "compare_existing_only": True,
                "permanent_drop": True,
                "update_existing_statistics": True,
                "require_new_frame_support": True,
                "log_posthash_coverage": False,
            },
            "global_graph": {
                "node_mode": "chunk_first_stride",
                "expected_overlap_frames": 2,
                "optimization_trigger": "loop_only",
                "min_depth": 0.05,
                "max_depth": 20.0,
                "min_match_cosine": 0.2,
                "max_match_entropy": 1.0,
                "chunk_stride": {
                    "target_index": 2,
                    "min_matches": 64,
                    "holdout_stride": 5,
                },
                "skip_edge": {"enabled": False},
            },
            "loop_closure": {"exclude_recent_windows": 100},
            "voxel_fusion": {
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )

    result = backend.process_packet(packet)

    assert result.fusion["new_frame_support_requested"] >= 3
    assert result.fusion["new_frame_support_kept"] == 2
    assert result.fusion["incoming_visibility_admission_requested"] == 2
    assert result.fusion["incoming_visibility_admission_kept"] == 1
    assert result.fusion["incoming_visibility_admission_dropped"] == 1
    assert result.fusion["incoming_visibility_admission_views"] == 4
    assert result.fusion["hash_visibility_views"] == 4
    assert result.fusion["chunk_anchor_delta"] == 1
    assert backend.map.anchor_count() == 1


def test_two_view_depth_gate_classifies_hole_front_surface_behind_and_conflict() -> None:
    incoming = torch.tensor(
        [
            [1.0, 1.0],
            [0.70, 0.75],
            [1.00, 1.10],
            [1.30, 1.40],
            [0.70, 1.30],
            [1.0, 1.0],
        ]
    )
    global_depth = torch.ones_like(incoming)
    global_alpha = torch.ones_like(incoming)
    global_alpha[0] = 0.0
    support = torch.ones_like(incoming, dtype=torch.bool)
    support[5] = False

    keep, class_id, stats = (
        SphericalSelfiGlobalBackend._classify_incoming_anchor_depths(
            incoming,
            global_depth,
            global_alpha,
            support,
            relative_threshold=0.15,
            alpha_threshold=0.05,
        )
    )

    assert class_id.tolist() == [0, 1, 2, 3, 2, 4]
    assert keep.tolist() == [True, True, False, False, False, True]
    assert stats["depth_gate_hole"] == 1
    assert stats["depth_gate_front"] == 1
    assert stats["depth_gate_consistent"] == 2
    assert stats["depth_gate_behind"] == 1
    assert stats["depth_gate_no_valid"] == 1
    assert stats["depth_gate_two_view_comparable"] == 4
    assert stats["depth_gate_two_view_agreement"] == 3


def test_depth_gate_analytic_erp_projection_wraps_across_panorama_seam() -> None:
    width, height = 16, 8
    epsilon = 1.0e-4
    bearing = torch.tensor(
        [[-epsilon, 0.0, -1.0], [epsilon, 0.0, -1.0]],
        dtype=torch.float32,
    )
    pixel = bearing_to_erp_pixel(bearing, height, width, wrap=True)
    longitude = (
        torch.arange(width, dtype=torch.float32) + 0.5
    ) / float(width) * (2.0 * math.pi) - math.pi
    feature = torch.cos(longitude).view(1, 1, width).expand(1, height, width)
    sampled = sample_erp_with_wrap(feature, pixel).reshape(-1)

    assert pixel[0, 0] < 0.01
    assert pixel[1, 0] > width - 0.01
    assert torch.allclose(sampled[0], sampled[1], atol=1.0e-4)


def test_pfgs360_refined_anchor_projection_wraps_panorama_seam() -> None:
    width, height = 16, 8
    epsilon = 1.0e-4
    xyz = torch.tensor(
        [[-epsilon, 0.0, -1.0], [epsilon, 0.0, -1.0]],
        dtype=torch.float32,
    )
    pixel, valid = SphericalSelfiGlobalBackend._pfgs360_anchor_pixels(
        xyz,
        torch.eye(4),
        image_size=(height, width),
    )

    assert bool(valid.all())
    assert pixel[0, 0] < 0.01
    assert pixel[1, 0] > width - 0.01


def test_refined_anchor_prepare_can_bypass_semantic_quality_gates() -> None:
    packet = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    anchor = packet.anchor_observation
    assert anchor is not None
    packet.anchor_observation = replace(
        anchor,
        quality=torch.zeros_like(anchor.quality),
        opacity_logit=torch.full_like(anchor.opacity_logit, -12.0),
    )
    fusion = Stage2GlobalMapFusion(
        PanoGaussianMap(config={}, device="cpu"),
        voxel_sizes=(0.04, 0.08, 0.16, 0.32),
        min_confidence=0.05,
        min_opacity=0.02,
    )

    legacy = fusion.prepare_packet_batch(
        packet,
        sim3_identity(),
        apply_semantic_gates=True,
    )
    official = fusion.prepare_packet_batch(
        packet,
        sim3_identity(),
        apply_semantic_gates=False,
    )

    assert len(legacy.batch) == 0
    assert len(official.batch) == anchor.num_anchors
    assert torch.equal(official.batch.xyz, anchor.xyz)
    assert torch.equal(official.batch.scale, anchor.scaling)
    assert torch.equal(official.batch.rotation, anchor.rotation)
    assert torch.equal(official.batch.opacity, packet.anchor_observation.opacity)


def test_pfgs360_first_chunk_commits_refiner_attributes_without_reinitialization() -> None:
    packet = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    packet.metadata["voxel_anchor_source_view_mask"] = torch.full(
        (packet.anchor_observation.num_anchors,),
        (1 << 4) - 1,
        dtype=torch.long,
    )
    gaussian_map = PanoGaussianMap(
        config={"MapRepresentation": {"gaussian_parameterization": "traditional_3dgs"}},
        device="cpu",
    )
    backend = SphericalSelfiGlobalBackend.__new__(SphericalSelfiGlobalBackend)
    backend.map = gaussian_map
    backend.mapper = SimpleNamespace(
        _pending_pfgs360_anchor_admission=None,
        _pfgs360_gaussian_moments={},
    )
    backend.fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.04, 0.08, 0.16, 0.32),
        min_confidence=0.05,
        min_opacity=0.02,
    )
    expected = backend.fusion.prepare_packet_batch(
        packet,
        sim3_identity(),
        apply_semantic_gates=False,
    )

    stats = backend._update_pfgs360_refined_anchors(
        packet,
        sim3_identity(),
        event="bootstrap",
        owner_window_id=0,
        observations=(),
        new_frame_ids=packet.frame_ids,
        mono_inlier_masks={},
        optimized_poses={},
        existing_anchor_visibility={},
    )

    assert stats["candidate"] == len(expected.batch)
    assert gaussian_map.anchor_count() == stats["inserted"]
    assert torch.allclose(gaussian_map.get_xyz, expected.batch.xyz)
    assert torch.allclose(gaussian_map.get_scaling, expected.batch.scale)
    assert torch.allclose(gaussian_map.get_rotation, expected.batch.rotation)
    assert torch.allclose(gaussian_map.get_opacity, expected.batch.opacity)
    assert torch.allclose(
        gaussian_map.get_sh_coefficients,
        expected.batch.sh_coefficients,
    )


def test_depth_gate_first_window_is_passthrough_without_any_render(monkeypatch) -> None:
    packet = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    packet.metadata["voxel_anchor_source_view_mask"] = torch.ones(
        packet.anchor_observation.num_anchors, dtype=torch.long
    )
    _attach_identity_stride_matches(packet)
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        renderer=_SyntheticSharedDepthRenderer(
            local_depth=2.0, global_depth=2.0
        ),
        config={
            "enabled": True,
            "rendered_overlap_alignment": {
                "enabled": True,
                "mode": "two_frame_bridge_depth_scale",
            },
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "compare_existing_only": True,
                "permanent_drop": True,
                "require_new_frame_support": True,
                "radius_voxels": 1.5,
            },
            "insertion_depth_gate": {"enabled": True},
            "global_graph": {
                "node_mode": "chunk_first_stride",
                "expected_overlap_frames": 2,
                "optimization_trigger": "loop_only",
                "chunk_stride": {"target_index": 2, "holdout_stride": 5},
                "skip_edge": {"enabled": False},
            },
            "loop_closure": {"exclude_recent_windows": 100},
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )
    monkeypatch.setattr(
        backend,
        "_render_global_pose_frame",
        lambda *args, **kwargs: pytest.fail("first window rendered global map"),
    )
    monkeypatch.setattr(
        backend,
        "_render_refined_anchor_frame",
        lambda *args, **kwargs: pytest.fail("first window rendered incoming map"),
    )

    result = backend.process_packet(packet)

    assert result.fusion["depth_gate_first_window_passthrough"] == 1
    assert result.fusion["depth_gate_global_render_views"] == 0
    assert result.fusion["depth_gate_incoming_render_views"] == 0
    assert result.fusion["depth_gate_requested"] == result.fusion["depth_gate_kept"]


def test_depth_gate_uses_exactly_two_global_renders_and_no_incoming_render(
    monkeypatch,
) -> None:
    packet = _refined_packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
    )
    packet.metadata["voxel_anchor_source_view_mask"] = torch.full(
        (packet.anchor_observation.num_anchors,),
        (1 << 2) | (1 << 3),
        dtype=torch.long,
    )
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    gaussian_map.add_seeds(
        GaussianSeedBatch(
            xyz=torch.tensor([[0.0, 0.0, 1.0]]),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.1),
            level=torch.zeros(1, dtype=torch.long),
            frame_id=0,
        )
    )
    backend = SphericalSelfiGlobalBackend(
        gaussian_map,
        config={
            "enabled": True,
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "compare_existing_only": True,
                "permanent_drop": True,
                "require_new_frame_support": True,
                "radius_voxels": 1.5,
            },
            "insertion_depth_gate": {"enabled": True},
        },
    )
    prepared = backend.fusion.prepare_packet_batch(packet, sim3_identity())
    calls = 0

    def global_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        height, width = packet.anchor_observation.image_size
        return RenderedSharedFrame(
            depth=torch.full((1, height, width), 10.0),
            alpha=torch.ones(1, height, width),
            anchor_visibility=torch.tensor([calls == 1]),
            render_seconds=0.01,
        )

    monkeypatch.setattr(backend, "_render_global_pose_frame", global_render)
    monkeypatch.setattr(
        backend,
        "_render_refined_anchor_frame",
        lambda *args, **kwargs: pytest.fail("depth gate rendered incoming map"),
    )

    filtered, _, visible_old, stats, _, _ = (
        backend._apply_two_new_frame_depth_gate(
            packet,
            prepared,
            sim3_identity(),
            new_frame_indices=(2, 3),
        )
    )

    assert calls == 2
    assert stats["depth_gate_global_render_views"] == 2
    assert stats["depth_gate_incoming_render_views"] == 0
    assert stats["depth_gate_front"] == len(prepared.batch)
    assert len(filtered.batch) == len(prepared.batch)
    assert visible_old.tolist() == [True]


def _error_prune_fixture(
    count: int,
    renderer,
) -> tuple[SphericalSelfiGlobalBackend, LocalGaussianWindowPacket, dict[int, RenderedSharedFrame]]:
    packet = _refined_packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
    )
    packet.observation = replace(
        packet.observation,
        refined_depth=torch.full_like(packet.observation.refined_depth, 2.0),
        confidence=torch.ones_like(packet.observation.confidence),
    )
    packet.finite_gaussian_mask.fill_(True)
    packet.static_mask.fill_(True)
    packet.geometry_consistency.fill_(True)
    packet.sky_mask.fill_(False)
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    gaussian_map.add_seeds(
        GaussianSeedBatch(
            xyz=torch.stack(
                [torch.tensor([float(row), 0.0, 1.0]) for row in range(count)]
            ),
            rgb=torch.zeros(count, 3),
            confidence=torch.ones(count),
            scale=torch.full((count,), 0.1),
            level=torch.zeros(count, dtype=torch.long),
            frame_id=0,
        )
    )
    backend = SphericalSelfiGlobalBackend(
        gaussian_map,
        renderer=renderer,
        config={
            "enabled": True,
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "compare_existing_only": True,
                "permanent_drop": True,
                "require_new_frame_support": True,
                "radius_voxels": 1.5,
            },
            "insertion_depth_gate": {"enabled": True},
            "error_gaussian_prune": {"enabled": True},
        },
    )
    height, width = packet.anchor_observation.image_size
    rendered = {
        index: RenderedSharedFrame(
            depth=torch.ones(1, height, width),
            alpha=torch.ones(1, height, width),
            anchor_visibility=torch.ones(count, dtype=torch.bool),
            render_seconds=0.0,
        )
        for index in (2, 3)
    }
    return backend, packet, rendered


def test_error_prune_deletes_every_double_hit_without_window_cap() -> None:
    count = 37
    renderer = _QueryAttributionRenderer([(0.9, 0.9), (0.9, 0.9)])
    backend, packet, rendered = _error_prune_fixture(count, renderer)

    stats, deleted, _ = backend._accumulate_and_prune_error_gaussians(
        packet,
        sim3_identity(),
        new_frame_indices=(2, 3),
        global_poses=packet.global_poses(sim3_identity()),
        rendered_views=rendered,
    )

    assert deleted == count
    assert stats["error_prune_eligible"] == count
    assert stats["error_pruned"] == count
    assert stats["error_prune_deleted_fraction"] == 1.0
    assert backend.map.anchor_count() == 0
    assert renderer.calls == 2
    assert all(tuple(value.shape[-3:]) == (6, 12, 2) for value in renderer.seen_queries)
    assert bool(renderer.seen_queries[0][..., 1].all())


def test_error_prune_requires_two_hits_and_preserves_cross_window_evidence() -> None:
    renderer = _QueryAttributionRenderer(
        [(0.9, 0.9), (0.9, 0.0), (0.9, 0.9), (0.9, 0.0)]
    )
    backend, packet, rendered = _error_prune_fixture(3, renderer)
    global_poses = packet.global_poses(sim3_identity())

    first, deleted_first, _ = backend._accumulate_and_prune_error_gaussians(
        packet,
        sim3_identity(),
        new_frame_indices=(2, 3),
        global_poses=global_poses,
        rendered_views=rendered,
    )
    assert deleted_first == 0
    assert first["error_prune_new_replacement_hits"] == 3
    assert backend.map._anchor_inlier_obs.tolist() == [1, 1, 1]

    second, deleted_second, _ = backend._accumulate_and_prune_error_gaussians(
        packet,
        sim3_identity(),
        new_frame_indices=(2, 3),
        global_poses=global_poses,
        rendered_views=rendered,
    )
    assert second["error_prune_evidence_carryover"] == 3
    assert deleted_second == 3
    assert backend.map.anchor_count() == 0


def test_error_prune_never_deletes_background_behind_new_foreground() -> None:
    renderer = _QueryAttributionRenderer([(0.9, 0.9), (0.9, 0.9)])
    backend, packet, rendered = _error_prune_fixture(2, renderer)
    for value in rendered.values():
        value.depth.fill_(4.0)

    stats, deleted, _ = backend._accumulate_and_prune_error_gaussians(
        packet,
        sim3_identity(),
        new_frame_indices=(2, 3),
        global_poses=packet.global_poses(sim3_identity()),
        rendered_views=rendered,
    )

    assert deleted == 0
    assert stats["error_prune_new_replacement_hits"] == 0
    assert all(not bool(query[..., 1].any()) for query in renderer.seen_queries)
    assert backend.map.anchor_count() == 2


def test_error_prune_query_output_is_required() -> None:
    class MissingQueryRenderer:
        def __init__(self) -> None:
            self.calls = 0

        def render(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "query_answers": torch.ones(2, 2),
                    "accum_visible": torch.ones(2),
                    "visibility_filter": torch.ones(2, dtype=torch.bool),
                }
            return {"visibility_filter": torch.ones(2, dtype=torch.bool)}

    backend, packet, rendered = _error_prune_fixture(2, MissingQueryRenderer())
    evidence_before = backend.map._anchor_inlier_obs.clone()
    with pytest.raises(RuntimeError, match="query_answers and accum_visible"):
        backend._accumulate_and_prune_error_gaussians(
            packet,
            sim3_identity(),
            new_frame_indices=(2, 3),
            global_poses=packet.global_poses(sim3_identity()),
            rendered_views=rendered,
        )
    torch.testing.assert_close(backend.map._anchor_inlier_obs, evidence_before)


def test_chunk_first_hash_uses_committed_pose_revision_for_all_four_views(
    monkeypatch,
) -> None:
    packet0 = _refined_packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )
    packet1 = _refined_packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
    )
    for packet in (packet0, packet1):
        packet.metadata["voxel_anchor_refiner_requested"] = True
        packet.metadata["voxel_anchor_refiner_pending"] = True

    def finalize_refiner(packet: LocalGaussianWindowPacket):
        images = torch.zeros(
            1,
            len(packet.frame_ids),
            3,
            *packet.observation.image_size,
        )
        anchors = voxelize_per_pixel_gaussians(
            packet.observation,
            packet.adapter_features,
            images,
            VoxelAnchorConfig(
                use_resnet_error=False,
                pretrained_resnet=False,
            ),
            valid_mask=packet.finite_gaussian_mask,
        ).detach_for_backend()
        metadata = dict(packet.metadata)
        metadata["voxel_anchor_refiner_pending"] = False
        metadata["voxel_anchor_refiner_enabled"] = True
        return replace(
            packet,
            anchor_observation=anchors,
            metadata=metadata,
        )
    _attach_identity_stride_matches(packet0)
    _attach_identity_stride_matches(packet1)
    backend = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config={}, device="cpu"),
        renderer=_SyntheticSharedDepthRenderer(
            local_depth=2.0,
            global_depth=2.0,
        ),
        pose_canonicalized_packet_refiner=finalize_refiner,
        config={
            "enabled": True,
            "rendered_overlap_alignment": {
                "enabled": True,
                "mode": "two_frame_bridge_depth_scale",
                "min_points": 32,
                "max_points": 128,
                "min_points_per_frame": 32,
                "max_points_per_frame": 64,
                "post_refiner_scale_recheck": False,
            },
            "insertion_dedup": {
                "enabled": True,
                "visible_only": True,
                "same_level_only": True,
                "radius_voxels": 1.25,
                "compare_existing_only": True,
                "permanent_drop": True,
                "update_existing_statistics": True,
                "require_new_frame_support": False,
                "log_posthash_coverage": True,
            },
            "global_graph": {
                "node_mode": "chunk_first_stride",
                "expected_overlap_frames": 2,
                "optimization_trigger": "periodic_and_loop",
                "optimization_start_nodes": 3,
                "optimization_interval_edges": 1,
                "active_nodes": 3,
                "min_depth": 0.05,
                "max_depth": 20.0,
                "min_match_cosine": 0.2,
                "max_match_entropy": 1.0,
                "chunk_stride": {
                    "target_index": 2,
                    "min_matches": 64,
                    "holdout_stride": 5,
                },
                "skip_edge": {"enabled": False},
            },
            "loop_closure": {"exclude_recent_windows": 100},
            "voxel_fusion": {
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
                "min_confidence": 0.0,
                "min_opacity": 0.0,
            },
            "map_optimization": {"steps_per_window": 0, "final_steps": 0},
        },
    )

    backend.process_packet(packet0)
    rendered_global_poses: list[torch.Tensor] = []
    original_render_global = backend._render_global_pose_frame

    def record_global_pose(pose_c2w, *, image_size):
        rendered_global_poses.append(pose_c2w.detach().clone())
        return original_render_global(pose_c2w, image_size=image_size)

    def move_tail_node(active_node_ids=None, *, fixed_node_ids=None):
        node = int(active_node_ids[-1])
        transform = backend.graph.transform(node)
        scale, rotation, translation = sim3_components(transform)
        backend.graph.nodes[node] = sim3_from_components(
            scale,
            rotation,
            translation + translation.new_tensor([0.25, 0.0, 0.0]),
        )
        return Sim3GraphOptimizeResult(
            accepted=True,
            iterations=1,
            initial_objective=1.0,
            final_objective=0.5,
            max_update_norm=0.25,
            optimized_node_ids=tuple(int(value) for value in active_node_ids),
            reason="synthetic_pose_commit",
            final_damping=float(backend.graph.damping),
        )

    monkeypatch.setattr(
        backend,
        "_render_global_pose_frame",
        record_global_pose,
    )
    monkeypatch.setattr(backend.graph, "optimize", move_tail_node)
    result = backend.process_packet(packet1)

    assert result.fusion["hash_visibility_views"] == 4
    assert result.fusion["prehash_overlap_view_count"] == 2
    assert result.fusion["prehash_new_frame_view_count"] == 2
    assert result.fusion["prehash_overlap_incoming_valid_coverage"] == pytest.approx(1.0)
    assert result.fusion["prehash_new_frame_existing_valid_coverage"] == pytest.approx(1.0)
    assert result.fusion["posthash_coverage_views"] == 4
    assert result.fusion["posthash_overlap_view_count"] == 2
    assert result.fusion["posthash_new_frame_view_count"] == 2
    assert result.fusion["posthash_overlap_global_valid_coverage"] == pytest.approx(1.0)
    assert result.fusion["posthash_new_frame_global_valid_coverage"] == pytest.approx(1.0)
    assert result.fusion["posthash_overlap_valid_coverage_delta"] == pytest.approx(0.0)
    assert result.fusion["posthash_new_frame_valid_coverage_delta"] == pytest.approx(0.0)
    assert result.fusion["chunk_anchor_delta"] == (
        result.fusion["anchors_after"] - result.fusion["anchors_before"]
    )
    assert result.graph is not None and result.graph.accepted is True
    pose_state = result.diagnostics["pose_state_consistency"]
    assert pose_state["accepted"] is True
    assert result.fusion["pose_state_committed_revision"] == pose_state[
        "committed_revision"
    ]
    assert any(
        float(pose[0, 3]) == pytest.approx(0.25)
        for pose in rendered_global_poses
    )
    alignment = result.diagnostics["alignment"]
    assert alignment["global_render_used_for_scale"] is False
    assert alignment["global_render_diagnostic_only"] is True
    diagnostic = backend.consume_rendered_overlap_diagnostic()
    assert diagnostic is not None
    assert diagnostic["frame_ids"].tolist() == [2, 3]


def test_chunk_first_process_builds_only_stride_and_independent_skip_cycle() -> None:
    shared = torch.randn(24, 32, 64)
    features = {0: shared, 4: shared}
    packet0 = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
        feature_by_frame=features,
        height=32,
        width=64,
    )
    packet1 = _packet(
        1,
        torch.eye(4).repeat(4, 1, 1),
        (2, 3, 4, 5),
        feature_by_frame=features,
        height=32,
        width=64,
    )
    _attach_identity_stride_matches(packet0)
    _attach_identity_stride_matches(packet1)
    backend = _chunk_stride_backend(min_matches=64, skip=True)
    backend.chunk_skip_forward_backward = False

    first = backend.process_packet(packet0)
    second = backend.process_packet(packet1)

    assert first.graph is None
    assert set(backend.graph.nodes) == {0, 2, 4}
    assert backend.window_anchor_nodes == {0: 0, 1: 2}
    assert backend.window_end_nodes == {}
    edge_types = [edge.edge_type for edge in backend.graph.edges]
    assert edge_types.count("chunk_stride_dense_spherical") == 2
    assert edge_types.count("chunk_skip_dense_spherical") == 1
    assert not any("overlap" in edge_type for edge_type in edge_types)
    assert second.graph is not None

    backend.submaps[0] = SimpleNamespace(
        frozen=True,
        boundary_node_ids=[0, 2],
        compressed_dense_factors=0,
    )
    compressed = backend._compress_frozen_submap_factors(
        SimpleNamespace(submap_id=1, boundary_node_ids=[2, 4])
    )
    assert compressed == 1
    assert not any(
        isinstance(edge, DenseSphericalFactorBlock)
        and edge.edge_type == "chunk_stride_dense_spherical"
        and int(edge.source) == 2
        and int(edge.target) == 4
        for edge in backend.graph.edges
    )
    assert any(
        isinstance(edge, DenseSphericalFactorBlock)
        and edge.edge_type == "chunk_skip_dense_spherical"
        and int(edge.source) == 0
        and int(edge.target) == 4
        for edge in backend.graph.edges
    )
    assert (2, 4, "chunk_stride_dense_spherical") not in (
        backend._chunk_stride_holdouts
    )
    assert (0, 4, "chunk_skip_dense_spherical") in (
        backend._chunk_stride_holdouts
    )

    nodes_before_finalize = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }

    def fail_unvalidated_finalize_lm(*args, **kwargs):
        raise AssertionError("chunk-first finalize must not run an unvalidated LM")

    backend.graph.optimize = fail_unvalidated_finalize_lm
    final = backend.finalize()
    assert final["graph_reason"] == "chunk_first_stride_no_unvalidated_finalize_lm"
    assert final["graph_node_mode"] == "chunk_first_stride"
    for node, transform in nodes_before_finalize.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)


def test_pure_chain_emits_incremental_geometry_then_optimization_emits_full() -> None:
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=False)

    backend.process_packet(packets[0])
    first = backend.pop_frame_geometry_update_batch()
    assert first is not None and first.complete_snapshot is False
    assert set(first.updates) == {0, 1, 2, 3}

    backend.process_packet(packets[1])
    second = backend.pop_frame_geometry_update_batch()
    assert second is not None and second.complete_snapshot is False
    assert set(second.updates) == {4, 5}
    assert second.affected_node_ids == (4,)
    node_scales = [
        float(sim3_components(backend.graph.transform(node))[0])
        for node in sorted(backend.graph.nodes)
    ]
    assert node_scales == pytest.approx([node_scales[0]] * len(node_scales))

    backend._refresh_geometry_updates(
        complete_snapshot=True,
        affected_node_ids={0, 2, 4},
        reason="synthetic_graph_commit",
    )
    complete = backend.pop_frame_geometry_update_batch()
    assert complete is not None and complete.complete_snapshot is True
    assert set(complete.updates) == {0, 1, 2, 3, 4, 5}


def test_chunk_skip_cycle_never_runs_lm_when_graph_optimization_is_disabled() -> None:
    shared = torch.randn(24, 32, 64)
    features = {0: shared, 4: shared}
    packets = [
        _packet(
            0,
            torch.eye(4).repeat(4, 1, 1),
            (0, 1, 2, 3),
            feature_by_frame=features,
            height=32,
            width=64,
        ),
        _packet(
            1,
            torch.eye(4).repeat(4, 1, 1),
            (2, 3, 4, 5),
            feature_by_frame=features,
            height=32,
            width=64,
        ),
    ]
    for packet in packets:
        _attach_identity_stride_matches(packet)
    backend = _chunk_stride_backend(min_matches=64, skip=True)
    backend.chunk_skip_forward_backward = False
    backend.global_graph_optimization_enabled = False

    def fail_disabled_lm(*args, **kwargs):
        raise AssertionError("optimization_enabled=false must gate skip/loop LM")

    backend.graph.optimize = fail_disabled_lm
    first = backend.process_packet(packets[0])
    second = backend.process_packet(packets[1])

    assert first.graph is None and second.graph is None
    assert any(
        edge.edge_type == "chunk_skip_dense_spherical"
        for edge in backend.graph.edges
    )


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


def test_sim3_graph_lm_does_not_clamp_translation_rotation_or_scale_steps() -> None:
    tangent = torch.tensor(
        [2.5, -1.5, 0.8, 0.6, -0.35, 0.2, math.log(1.8)]
    )
    truth = sim3_exp(tangent)
    graph = GlobalSim3FactorGraph(
        max_iterations=1,
        pcg_iterations=64,
        damping=1.0e-6,
    )
    graph.add_node(0, sim3_identity())
    graph.add_node(1, sim3_identity())
    graph.add_edge(
        Sim3GraphEdge(
            source=0,
            target=1,
            measurement_target_to_source=truth,
            information_diag=torch.ones(7),
        )
    )

    result = graph.optimize()
    optimized = sim3_log(graph.transform(1))

    assert result.accepted
    assert result.max_update_norm > 1.0
    assert float(optimized[:3].norm()) > 1.0
    assert math.degrees(float(optimized[3:6].norm())) > 10.0
    assert math.exp(abs(float(optimized[6]))) > 1.25


def test_sim3_graph_optimization_can_be_disabled_without_mutating_nodes() -> None:
    tangent = torch.tensor([0.3, -0.2, 0.1, 0.03, -0.04, 0.02, math.log(1.2)])
    truth = sim3_exp(tangent)
    graph = GlobalSim3FactorGraph(
        max_iterations=10,
        pcg_iterations=32,
        optimization_enabled=False,
    )
    graph.add_node(0, sim3_identity())
    initial = sim3_exp(
        tangent + torch.tensor([0.15, 0.0, 0.0, 0.0, 0.02, 0.0, 0.1])
    )
    graph.add_node(1, initial)
    graph.add_edge(
        Sim3GraphEdge(
            source=0,
            target=1,
            measurement_target_to_source=truth,
            information_diag=torch.ones(7),
        )
    )

    result = graph.optimize()

    assert result.accepted is False
    assert result.iterations == 0
    assert result.reason == "optimization_disabled"
    assert result.final_objective == pytest.approx(result.initial_objective)
    torch.testing.assert_close(graph.transform(1), initial)


def test_no_graph_100_frame_ablation_only_disables_graph_optimization() -> None:
    root = Path(__file__).parents[1]
    base = yaml.safe_load(
        (root / "configs" / "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_100.yaml")
        .read_text(encoding="utf-8")
    )
    ablation = yaml.safe_load(
        (
            root
            / "configs"
            / "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_nograph_100.yaml"
        ).read_text(encoding="utf-8")
    )

    assert ablation["base_config"] == (
        "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_100.yaml"
    )
    assert base["Dataset"] == {"begin": 0, "end": 100, "frame_stride": 1}
    assert ablation["SphericalSelfiGlobalBackend"] == {
        "global_graph": {"optimization_enabled": False}
    }


def test_posebaseline_ablation_explicitly_uses_legacy_boundary_graph() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "spherical_selfi_ob3d_bridge_posebaseline_refiner_ba8_100.yaml"
        ).read_text(encoding="utf-8")
    )

    backend = config["SphericalSelfiGlobalBackend"]
    assert backend["global_graph"]["node_mode"] == "boundary_frame"
    assert (
        backend["rendered_overlap_alignment"]["mode"]
        == "two_frame_bridge_pose_scale"
    )


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
        s2_information_scale=0.4,
        depth_information_scale=0.7,
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


def test_dense_information_scale_survives_count_normalization() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = torch.nn.functional.normalize(
        source + torch.tensor([[0.0, 0.02, 0.0], [0.01, 0.0, 0.0]]),
        dim=-1,
    )

    def objective(information_scale: float) -> float:
        graph = GlobalSim3FactorGraph()
        graph.add_node(0, sim3_identity())
        graph.add_node(1, sim3_identity())
        graph.add_edge(
            DenseSphericalFactorBlock(
                source=0,
                target=1,
                source_local_pose=torch.eye(4),
                target_local_pose=torch.eye(4),
                source_bearing=source,
                target_bearing=target,
                source_depth=torch.ones(2),
                target_depth=torch.ones(2),
                factor_weight=torch.ones(2),
                use_depth=False,
                normalize_information_by_count=True,
                information_reference_count=64.0,
                s2_information_scale=information_scale,
            )
        )
        return float(graph.objective())

    full = objective(1.0)
    quarter = objective(0.25)
    assert quarter == pytest.approx(0.25 * full, rel=1.0e-5)


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


def test_loop_dense_matching_does_not_gate_on_top2_margin() -> None:
    source = _packet(0, torch.eye(4).view(1, 4, 4), (0,))
    target = _packet(10, torch.eye(4).view(1, 4, 4), (10,))
    source.verification_features = torch.ones_like(source.verification_features)
    target.verification_features = torch.ones_like(target.verification_features)
    detector = PanoramaLoopDetector(
        factor_queries_per_direction=32,
        min_match_cosine=0.2,
        max_match_entropy=1.01,
        forward_backward=False,
        target_area_correction=False,
    )

    matches = detector._fibonacci_matches(
        source,
        target,
        source_frame_index=0,
        target_frame_index=0,
        direction=0,
    )

    assert int(matches["count"]) > 0
    torch.testing.assert_close(
        matches["top2_margin"],
        torch.zeros_like(matches["top2_margin"]),
    )


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


def test_traditional_3dgs_parameterization_keeps_raw_parameters_unbounded() -> None:
    gaussian_map = PanoGaussianMap(
        config={
            "MapRepresentation": {
                "mode": "anchor_scaffold_panorama",
                "gaussian_parameterization": "traditional_3dgs",
            },
            "BackendOptimization": {"sh_degree": 2},
        },
        device="cpu",
    )
    gaussian_map.add_seeds(
        GaussianSeedBatch(
            xyz=torch.tensor([[1000.0, -2000.0, 3000.0]]),
            rgb=torch.tensor([[0.2, 0.4, 0.8]]),
            confidence=torch.tensor([0.25]),
            scale=torch.tensor([2.5]),
            level=torch.zeros(1, dtype=torch.long),
            frame_id=0,
        )
    )
    torch.testing.assert_close(
        gaussian_map.scaling,
        torch.full_like(gaussian_map.scaling, math.log(2.5)),
    )
    torch.testing.assert_close(
        gaussian_map.get_scaling, torch.full_like(gaussian_map.get_scaling, 2.5)
    )
    expected_dc = (torch.tensor([[0.2, 0.4, 0.8]]) - 0.5) / 0.28209479177387814
    torch.testing.assert_close(gaussian_map.features, expected_dc)
    torch.testing.assert_close(
        gaussian_map.get_sh_coefficients[:, 0], expected_dc
    )
    raw_rotation = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    gaussian_map.rotation.data.copy_(raw_rotation)
    _ = gaussian_map.get_rotation
    torch.testing.assert_close(gaussian_map.rotation, raw_rotation)
    torch.testing.assert_close(
        torch.linalg.norm(gaussian_map.get_rotation, dim=-1), torch.ones(1)
    )
    gaussian_map.features.data.fill_(12.0)
    gaussian_map.scaling.data.fill_(10.0)
    gaussian_map.opacity_logit.data.fill_(-30.0)
    assert float(gaussian_map.get_sh_coefficients.max()) == 12.0
    assert float(gaussian_map.get_scaling.min()) > 20_000.0
    assert float(gaussian_map.get_opacity.max()) < 1.0e-10
    torch.testing.assert_close(
        gaussian_map.xyz,
        torch.tensor([[1000.0, -2000.0, 3000.0]]),
    )
    rotation_before = gaussian_map.rotation.detach().clone()
    opacity_before = gaussian_map.opacity_logit.detach().clone()
    scaling_before = gaussian_map.scaling.detach().clone()
    features_before = gaussian_map.features.detach().clone()
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.5,),
        min_confidence=0.0,
        min_opacity=0.0,
    )
    fusion._write_map(fusion._batch_from_map())
    torch.testing.assert_close(gaussian_map.rotation, rotation_before)
    torch.testing.assert_close(gaussian_map.opacity_logit, opacity_before)
    torch.testing.assert_close(gaussian_map.scaling, scaling_before)
    torch.testing.assert_close(gaussian_map.features, features_before)


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


def test_global_lifecycle_prune_is_removed() -> None:
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
    count = gaussian_map.anchor_count()
    gaussian_map._anchor_quality.fill_(1.0e-12)
    with torch.no_grad():
        gaussian_map.opacity_logit.fill_(-100.0)
    gaussian_map._anchor_conf_accum.zero_()
    assert not hasattr(fusion, "prune_lifecycle")
    assert gaussian_map.anchor_count() == count


def test_stage2_writeback_preserves_error_prune_evidence() -> None:
    packet = _packet(0, torch.eye(4).repeat(1, 1, 1), (0,))
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.5,),
        min_confidence=0.0,
        min_opacity=0.0,
    )
    fusion.fuse_packet(packet, sim3_identity())
    count = gaussian_map.anchor_count()
    gaussian_map._anchor_inlier_obs.copy_(torch.arange(count, dtype=torch.int32))
    gaussian_map._anchor_outlier_obs.copy_(
        torch.arange(count, dtype=torch.int32).flip(0)
    )
    expected_inlier = gaussian_map._anchor_inlier_obs.clone()
    expected_outlier = gaussian_map._anchor_outlier_obs.clone()

    fusion._write_map(fusion._batch_from_map())

    torch.testing.assert_close(gaussian_map._anchor_inlier_obs, expected_inlier)
    torch.testing.assert_close(gaussian_map._anchor_outlier_obs, expected_outlier)
    if count >= 2:
        batch = fusion._batch_from_map().index(torch.tensor([0, 1]))
        batch.xyz[1] = batch.xyz[0]
        batch.level[:] = 0
        batch.voxel_size[:] = 0.5
        batch.grid_coord[:] = torch.floor(batch.xyz / 0.5).long()
        batch.replacement_hits[:] = torch.tensor([1, 2])
        batch.inconsistency_hits[:] = torch.tensor([3, 4])
        compacted = fusion._winner_take_global_voxel(
            batch, preserve_levels=True
        )
        assert len(compacted) == 1
        assert compacted.replacement_hits.tolist() == [3]
        assert compacted.inconsistency_hits.tolist() == [7]


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


def _two_frame_boundary_backend(
    gaussian_map: PanoGaussianMap,
    renderer=None,
    *,
    covariance_min_ratio: float = 1.0e-4,
    mode: str = "two_frame_full_sim3",
) -> SphericalSelfiGlobalBackend:
    return SphericalSelfiGlobalBackend(
        gaussian_map,
        renderer=renderer,
        config={
            "enabled": True,
            "global_graph": {
                "node_mode": "boundary_frame",
                "expected_overlap_frames": 2,
                "enforce_exact_overlap": True,
                "min_overlap_points": 8,
                "max_overlap_points": 128,
                "max_overlap_residual": 0.05,
                "min_overlap_inlier_ratio": 0.35,
                "min_dense_factors": 8,
                "min_match_cosine": 0.2,
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
                "mode": mode,
                "min_points": 16,
                "max_points": 128,
                "min_points_per_frame": 16,
                "max_points_per_frame": 64,
                "alpha_threshold": 0.05,
                "min_inlier_ratio": 0.35,
                "max_median_relative_error": 0.10,
                "max_scale_change": 2.5,
                "irls_iterations": 5,
                "holdout_stride": 5,
                "covariance_min_ratio": covariance_min_ratio,
                "max_rotation_correction_deg": 10.0,
                "max_translation_correction": 1.0,
                "max_shared_rotation_error_deg": 2.0,
                "max_shared_center_error": 0.15,
                "failure_policy": "scale_pose_then_error",
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
            "map_optimization": {
                "steps_per_window": 0,
                "final_steps": 0,
            },
        },
    )


def _known_two_frame_geometry(
    transform: torch.Tensor,
    *,
    planar: bool = False,
    add_training_outliers: bool = False,
) -> list[OverlapFrameGeometry]:
    torch.manual_seed(19)
    scale, _, _ = sim3_components(transform)
    frames: list[OverlapFrameGeometry] = []
    count = 80
    for frame_offset in range(2):
        if planar:
            side = int(math.ceil(math.sqrt(count)))
            axis = torch.linspace(-1.0, 1.0, side)
            grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
            current_points = torch.stack(
                [
                    grid_x.reshape(-1),
                    grid_y.reshape(-1),
                    torch.full((side * side,), 2.0),
                ],
                dim=-1,
            )[:count]
        else:
            current_points = torch.randn(count, 3)
            current_points[:, 2] += 3.0
        current_pose = torch.eye(4)
        current_pose[:3, 3] = torch.tensor(
            [0.25 * frame_offset, 0.04 * frame_offset, 0.0]
        )
        previous_pose = apply_sim3_to_c2w(transform, current_pose)
        previous_points = apply_sim3(transform, current_points)
        holdout = torch.arange(count) % 5 == 0
        if add_training_outliers:
            candidates = torch.nonzero(~holdout, as_tuple=False).flatten()[:6]
            previous_points[candidates] += torch.tensor([2.0, -1.5, 0.75])
        current_depth = torch.linspace(1.0, 3.0, count)
        previous_depth = current_depth * scale
        frames.append(
            OverlapFrameGeometry(
                frame_id=10 + frame_offset,
                previous_index=frame_offset,
                current_index=frame_offset,
                bearing=torch.nn.functional.normalize(
                    current_points - current_pose[:3, 3],
                    dim=-1,
                ),
                uv=torch.stack(
                    [
                        torch.arange(count, dtype=torch.float32),
                        torch.full((count,), float(frame_offset)),
                    ],
                    dim=-1,
                ),
                previous_depth=previous_depth,
                current_depth=current_depth,
                previous_points=previous_points,
                current_points=current_points,
                previous_pose=previous_pose,
                current_pose=current_pose,
                holdout_mask=holdout,
            )
        )
    return frames


def _known_pose_bridge_geometry(
    *,
    absolute_scale: float = 1.2,
    local_baseline: float = 1.0,
) -> list[KnownPoseBridgeFrame]:
    frames: list[KnownPoseBridgeFrame] = []
    count = 80
    dummy_depth = torch.ones(1, 8, 16)
    dummy_render = RenderedSharedFrame(
        depth=dummy_depth,
        alpha=torch.ones_like(dummy_depth),
        anchor_visibility=torch.ones(4, dtype=torch.bool),
        render_seconds=0.0,
    )
    for index in range(2):
        current_pose = torch.eye(4)
        current_pose[0, 3] = float(index) * local_baseline
        global_pose = torch.eye(4)
        global_pose[0, 3] = float(index) * local_baseline * absolute_scale
        current_depth = torch.linspace(1.0, 3.0, count)
        global_depth = current_depth * absolute_scale
        holdout = torch.arange(count) % 5 == 0
        frames.append(
            KnownPoseBridgeFrame(
                frame_id=2 + index,
                previous_index=2 + index,
                current_index=index,
                bearing=torch.nn.functional.normalize(
                    torch.randn(count, 3), dim=-1
                ),
                uv=torch.stack(
                    [
                        torch.arange(count, dtype=torch.float32) % 16,
                        torch.arange(count, dtype=torch.float32) % 8,
                    ],
                    dim=-1,
                ),
                global_depth=global_depth,
                current_depth=current_depth,
                source_depth_previous_owner=global_depth / 0.75,
                previous_local_pose=global_pose,
                current_local_pose=current_pose,
                known_global_pose=global_pose,
                holdout_mask=holdout,
                inlier_mask=torch.ones(count, dtype=torch.bool),
                global_render=dummy_render,
                previous_render=dummy_render,
                current_render=dummy_render,
                global_valid_image=torch.ones_like(dummy_depth, dtype=torch.bool),
                current_valid_image=torch.ones_like(dummy_depth, dtype=torch.bool),
                global_previous_consistency_image=torch.ones_like(
                    dummy_depth, dtype=torch.bool
                ),
                sky_union_image=torch.zeros_like(dummy_depth, dtype=torch.bool),
                global_previous_consistency_ratio=1.0,
            )
        )
    return frames


def test_known_pose_bridge_uses_balanced_two_frame_map_consistency_gate() -> None:
    backend = object.__new__(SphericalSelfiGlobalBackend)
    backend.rendered_alignment_global_map_min_consistency_ratio = 0.35
    backend._overlap_frame_ids = lambda previous, current: (10, 11)  # type: ignore[method-assign]
    ratios = {10: 0.34, 11: 0.36}

    def collect(previous, current, frame_id, **kwargs):
        del previous, current, kwargs
        return SimpleNamespace(
            frame_id=int(frame_id),
            global_previous_consistency_ratio=ratios[int(frame_id)],
        )

    backend._collect_known_pose_bridge_frame = collect  # type: ignore[method-assign]
    frames = backend._collect_known_pose_bridge_frames(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        torch.eye(4),
        exclude_current_target_only=True,
    )
    assert [frame.global_previous_consistency_ratio for frame in frames] == [
        0.34,
        0.36,
    ]

    ratios.update({10: 0.348, 11: 0.349})
    with pytest.raises(RuntimeError, match="Two-frame balanced"):
        backend._collect_known_pose_bridge_frames(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            torch.eye(4),
            exclude_current_target_only=True,
        )


@pytest.mark.parametrize(
    ("mode", "config_mode"),
    [
        ("depth", "two_frame_bridge_depth_scale"),
        ("pose_baseline", "two_frame_bridge_pose_scale"),
    ],
)
def test_known_pose_bridge_recovers_scale_from_only_requested_source(
    mode: str,
    config_mode: str,
) -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        mode=config_mode,
    )
    previous_owner = sim3_from_components(
        0.75, torch.eye(3), torch.zeros(3)
    )
    scale, inliers, diagnostics = backend._estimate_known_pose_bridge_scale(
        _known_pose_bridge_geometry(),
        previous_owner,
        mode=mode,
    )

    assert scale == pytest.approx(1.2, rel=1.0e-5)
    assert diagnostics["measurement_scale"] == pytest.approx(1.6, rel=1.0e-5)
    assert diagnostics["accepted"] is True
    assert all(float(mask.float().mean()) > 0.99 for mask in inliers)


def test_known_pose_depth_bridge_accepts_consistent_two_frame_scale_with_surface_noise() -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        mode="two_frame_bridge_depth_scale",
    )
    noisy_frames: list[KnownPoseBridgeFrame] = []
    for frame in _known_pose_bridge_geometry(absolute_scale=1.0):
        row = torch.arange(frame.current_depth.numel())
        multiplier = torch.where(
            row.remainder(2) == 0,
            torch.full_like(frame.current_depth, 0.85),
            torch.full_like(frame.current_depth, 1.15),
        )
        global_depth = frame.current_depth * multiplier
        noisy_frames.append(
            replace(
                frame,
                global_depth=global_depth,
                source_depth_previous_owner=global_depth,
            )
        )

    scale, inliers, diagnostics = backend._estimate_known_pose_bridge_scale(
        noisy_frames,
        sim3_identity(),
        mode="depth",
    )

    assert scale is not None
    assert diagnostics["bridge_depth_gate_passed"] is False
    assert diagnostics["bridge_depth_consensus_gate_passed"] is True
    assert diagnostics["bridge_depth_consensus_fallback_used"] is True
    assert diagnostics["bridge_frame_scale_disagreement"] < 1.0e-6
    assert all(float(mask.float().mean()) > 0.99 for mask in inliers)


def test_known_pose_bridge_canonicalization_pins_overlap_and_preserves_tail() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.0, 0.7, 1.1, 1.6])
    packet = _packet(1, poses, (2, 3, 4, 5))
    angle = torch.deg2rad(torch.tensor(5.0))
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    owner = sim3_from_components(
        1.3, rotation, torch.tensor([0.2, -0.1, 0.05])
    )
    desired_second = packet.local_poses_c2w[1].clone()
    desired_second[:3, :3] = rotation.transpose(0, 1)
    desired_second[0, 3] = 0.9
    known = (
        apply_sim3_to_c2w(owner, torch.eye(4)),
        apply_sim3_to_c2w(owner, desired_second),
    )
    original_tail_from_second = (
        torch.linalg.inv(packet.local_poses_c2w[1])
        @ packet.local_poses_c2w[3]
    )

    corrected = SphericalSelfiGlobalBackend._canonicalize_packet_from_two_known_poses(
        packet,
        owner,
        known,
    )
    global_poses = corrected.global_poses(owner)

    torch.testing.assert_close(global_poses[0], known[0], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(global_poses[1], known[1], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        torch.linalg.inv(corrected.local_poses_c2w[1])
        @ corrected.local_poses_c2w[3],
        original_tail_from_second,
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_two_frame_full_sim3_recovers_known_transform_with_training_outliers() -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu")
    )
    angle = torch.deg2rad(torch.tensor(4.0))
    rotation = torch.tensor(
        [
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ]
    )
    expected = sim3_from_components(
        1.2,
        rotation,
        torch.tensor([0.20, -0.08, 0.04]),
    )
    frames = _known_two_frame_geometry(
        expected,
        add_training_outliers=True,
    )

    estimated, inliers, diagnostics = backend._fit_two_frame_full_sim3(frames)

    assert estimated is not None, diagnostics
    assert diagnostics["full_sim3_accepted"] is True
    assert diagnostics["full_sim3_holdout_inlier_ratio"] > 0.95
    assert all(float(mask.float().mean()) > 0.85 for mask in inliers)
    error = sim3_log(sim3_inverse(expected) @ estimated)
    assert float(torch.linalg.norm(error)) < 1.0e-3


def test_planar_two_frame_geometry_rejects_full_sim3_and_uses_pose_scale_fallback() -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        covariance_min_ratio=1.0e-3,
    )
    angle = torch.deg2rad(torch.tensor(3.0))
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = sim3_from_components(
        1.15,
        rotation,
        torch.tensor([0.12, -0.03, 0.02]),
    )
    frames = _known_two_frame_geometry(
        expected,
        planar=True,
        add_training_outliers=True,
    )

    full, _, full_diagnostics = backend._fit_two_frame_full_sim3(frames)
    fallback, _, fallback_diagnostics = (
        backend._fit_two_frame_scale_pose_fallback(frames)
    )

    assert full is None
    assert full_diagnostics["full_sim3_reason"] == "covariance_degenerate"
    assert fallback is not None, fallback_diagnostics
    assert fallback_diagnostics["fallback_accepted"] is True
    error = sim3_log(sim3_inverse(expected) @ fallback)
    assert float(torch.linalg.norm(error)) < 1.0e-5


def test_two_frame_scale_pose_fallback_rejects_inconsistent_shared_poses() -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        covariance_min_ratio=1.0e-3,
    )
    expected = sim3_from_components(
        1.1,
        torch.eye(3),
        torch.tensor([0.1, 0.0, 0.0]),
    )
    frames = _known_two_frame_geometry(expected, planar=True)
    frames[1] = replace(
        frames[1],
        previous_pose=frames[1].previous_pose.clone(),
    )
    frames[1].previous_pose[:3, 3] += torch.tensor([0.4, 0.0, 0.0])

    fallback, _, diagnostics = backend._fit_two_frame_scale_pose_fallback(
        frames
    )

    assert fallback is None
    assert diagnostics["fallback_accepted"] is False
    assert diagnostics["pose_pair_translation"] > 0.15


def test_two_frame_scale_pose_accepts_pair_disagreement_when_mean_pose_passes() -> None:
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        covariance_min_ratio=1.0e-3,
    )
    expected = sim3_from_components(
        1.1,
        torch.eye(3),
        torch.tensor([0.1, 0.0, 0.0]),
    )
    frames = _known_two_frame_geometry(expected, planar=True)
    for index, angle_deg in enumerate((1.7, -1.7)):
        angle = torch.deg2rad(torch.tensor(angle_deg))
        delta = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        previous_pose = frames[index].previous_pose.clone()
        previous_pose[:3, :3] = delta @ previous_pose[:3, :3]
        frames[index] = replace(
            frames[index],
            previous_pose=previous_pose,
        )

    fallback, _, diagnostics = backend._fit_two_frame_scale_pose_fallback(
        frames
    )

    assert diagnostics["pose_pair_rotation_deg"] > 2.0
    assert max(diagnostics["fallback_shared_rotation_errors_deg"]) < 2.0
    assert fallback is not None, diagnostics
    assert diagnostics["fallback_accepted"] is True


def test_overlap2_rejection_preserves_scalar_failure_diagnostics() -> None:
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _packet(0, poses0, (0, 1, 2, 3))
    packet1 = _packet(1, poses1, (2, 3, 4, 5))
    expected = sim3_from_components(
        1.1,
        torch.eye(3),
        torch.tensor([0.1, 0.0, 0.0]),
    )
    frames = [
        replace(frame, frame_id=frame_id)
        for frame, frame_id in zip(
            _known_two_frame_geometry(expected, planar=True),
            (2, 3),
        )
    ]
    frames[1] = replace(
        frames[1],
        previous_pose=frames[1].previous_pose.clone(),
    )
    frames[1].previous_pose[:3, 3] += torch.tensor([0.4, 0.0, 0.0])
    by_frame = {frame.frame_id: frame for frame in frames}
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        covariance_min_ratio=1.0e-3,
    )
    backend.process_packet(packet0)
    backend._collect_overlap_frame_geometry = (
        lambda _previous, _current, frame_id, *, use_rendered_anchors: by_frame[
            int(frame_id)
        ]
    )

    with pytest.raises(RuntimeError, match="two-frame alignment failed"):
        backend.process_packet(packet1)

    diagnostics = backend.consume_overlap_alignment_failure()
    assert diagnostics is not None
    assert diagnostics["accepted"] is False
    assert diagnostics["fallback_accepted"] is False
    assert diagnostics["pose_pair_translation"] > 0.15
    assert backend.consume_overlap_alignment_failure() is None


def test_overlap2_scale_pose_mode_exposes_the_requested_scale_only_ablation() -> None:
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _packet(0, poses0, (0, 1, 2, 3))
    packet1 = _packet(1, poses1, (2, 3, 4, 5))
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        mode="two_frame_scale_pose",
    )

    edge, dense, pose, diagnostics = backend._two_frame_overlap_constraints(
        packet0,
        packet1,
        previous_anchor_node=0,
        current_anchor_node=2,
        use_rendered_anchors=False,
    )

    assert edge is not None, diagnostics
    assert len(dense) == 2
    assert len(pose) == 2
    assert diagnostics["mode"] == "two_frame_scale_pose"
    assert diagnostics["alignment_method"] == "scale_pose_only"
    assert diagnostics["full_sim3_reason"] == "disabled_by_mode"
    assert diagnostics["fallback_accepted"] is True


def test_overlap2_full_sim3_automatically_uses_scale_pose_fallback_on_degeneracy() -> None:
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _packet(0, poses0, (0, 1, 2, 3))
    packet1 = _packet(1, poses1, (2, 3, 4, 5))
    expected = sim3_from_components(
        1.1,
        torch.eye(3),
        torch.tensor([0.2, 0.0, 0.0]),
    )
    frames = [
        replace(frame, frame_id=frame_id)
        for frame, frame_id in zip(
            _known_two_frame_geometry(expected, planar=True),
            (2, 3),
        )
    ]
    by_frame = {frame.frame_id: frame for frame in frames}
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        covariance_min_ratio=1.0e-3,
    )
    backend._collect_overlap_frame_geometry = (
        lambda _previous, _current, frame_id, *, use_rendered_anchors: by_frame[
            int(frame_id)
        ]
    )

    edge, dense, pose, diagnostics = backend._two_frame_overlap_constraints(
        packet0,
        packet1,
        previous_anchor_node=0,
        current_anchor_node=2,
        use_rendered_anchors=False,
    )

    assert edge is not None, diagnostics
    assert len(dense) == 2
    assert len(pose) == 2
    assert diagnostics["alignment_method"] == "scale_pose_fallback"
    assert diagnostics["full_sim3_reason"] == "covariance_degenerate"
    assert diagnostics["fallback_accepted"] is True


def test_two_frame_sampling_excludes_bilateral_sky() -> None:
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _packet(0, poses0, (0, 1, 2, 3))
    packet1 = _packet(1, poses1, (2, 3, 4, 5))
    for packet, frame_id in ((packet0, 2), (packet1, 2)):
        index = packet.frame_index(frame_id)
        packet.sky_mask[0, index, :, :3, :] = True
        packet.sky_prob[0, index, :, :3, :] = 1.0
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu")
    )

    geometry = backend._collect_overlap_frame_geometry(
        packet0,
        packet1,
        2,
        use_rendered_anchors=False,
    )

    assert int(geometry.current_points.shape[0]) >= 16
    assert bool((geometry.uv[:, 1] >= 3.0).all())
    assert geometry.sky_union_image is not None
    assert bool(geometry.sky_union_image[..., :3, :].all())


def test_overlap2_boundary_graph_uses_independent_chunk_nodes_and_two_frame_factors() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(6)}
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _refined_packet(
        0, poses0, (0, 1, 2, 3), feature_by_frame=features
    )
    packet1 = _refined_packet(
        1, poses1, (2, 3, 4, 5), feature_by_frame=features
    )
    original_depth = packet1.observation.refined_depth.clone()
    original_xyz = packet1.anchor_observation.xyz.clone()
    original_voxel = packet1.anchor_observation.voxel_size.clone()
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu"),
        _TwoViewUnionRenderer(local_depth=2.0, global_depth=2.0),
    )

    backend.process_packet(packet0)
    result = backend.process_packet(packet1)

    assert result.aligned
    assert result.diagnostics["alignment"]["alignment_method"] == "full_sim3"
    assert result.diagnostics["alignment"]["full_sim3_accepted"] is True
    assert backend.window_anchor_nodes == {0: 0, 1: 2}
    assert backend.window_end_nodes == {0: 3, 1: 5}
    assert set(backend.graph.nodes) == {0, 2, 3, 5}
    edge_types = [factor.edge_type for factor in backend.graph.edges]
    assert edge_types.count("boundary_dense_spherical") == 2
    assert edge_types.count("overlap_two_frame_sim3") == 1
    assert edge_types.count("overlap_dense_spherical") == 2
    assert edge_types.count("overlap_shared_pose_consistency") == 2
    assert result.fusion["hash_visibility_views"] == 2
    assert (
        result.fusion["hash_visible_incoming"]
        == packet1.anchor_observation.num_anchors
    )
    assert result.fusion["hash_visible_existing"] == result.fusion["anchors_before"]
    torch.testing.assert_close(packet1.observation.refined_depth, original_depth)
    torch.testing.assert_close(packet1.anchor_observation.xyz, original_xyz)
    torch.testing.assert_close(packet1.anchor_observation.voxel_size, original_voxel)
    stored = backend._last_full_packet
    assert stored is not None
    torch.testing.assert_close(stored.observation.refined_depth, original_depth)
    torch.testing.assert_close(stored.anchor_observation.xyz, original_xyz)
    torch.testing.assert_close(stored.anchor_observation.voxel_size, original_voxel)
    backend.submaps[0] = SimpleNamespace(
        frozen=True,
        boundary_node_ids=[0, 3],
    )
    compressed = backend._compress_frozen_submap_factors(
        SimpleNamespace(
            submap_id=1,
            boundary_node_ids=[2, 5],
        )
    )
    assert compressed == 3
    assert not any(
        isinstance(factor, DenseSphericalFactorBlock)
        and factor.edge_type == "overlap_dense_spherical"
        for factor in backend.graph.edges
    )
    remaining_boundaries = [
        factor
        for factor in backend.graph.edges
        if isinstance(factor, DenseSphericalFactorBlock)
        and factor.edge_type == "boundary_dense_spherical"
    ]
    assert len(remaining_boundaries) == 1
    assert {
        remaining_boundaries[0].source,
        remaining_boundaries[0].target,
    } == {0, 3}


def test_full_sim3_owner_applies_nonunit_parent_scale_once_without_packet_rescale() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 0, 3] = torch.arange(4) * 0.1
    packet = _refined_packet(0, poses, (0, 1, 2, 3))
    assert packet.anchor_observation is not None
    original_xyz = packet.anchor_observation.xyz.clone()
    original_scale = packet.anchor_observation.scaling.clone()
    original_voxel = packet.anchor_observation.voxel_size.clone()
    angle = torch.deg2rad(torch.tensor(5.0))
    rotation = torch.tensor(
        [
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ]
    )
    owner = sim3_from_components(
        2.0,
        rotation,
        torch.tensor([0.3, -0.2, 0.1]),
    )
    backend = _two_frame_boundary_backend(
        PanoGaussianMap(config={}, device="cpu")
    )

    prepared = backend.fusion.prepare_packet_batch(packet, owner)

    assert prepared.source_anchor_indices is not None
    selected = prepared.source_anchor_indices
    torch.testing.assert_close(
        prepared.batch.xyz,
        apply_sim3(owner, original_xyz.index_select(0, selected)),
    )
    torch.testing.assert_close(
        prepared.batch.scale,
        2.0 * original_scale.index_select(0, selected),
    )
    torch.testing.assert_close(
        prepared.batch.voxel_size,
        2.0 * original_voxel.index_select(0, selected),
    )
    torch.testing.assert_close(packet.anchor_observation.xyz, original_xyz)
    torch.testing.assert_close(packet.anchor_observation.scaling, original_scale)
    torch.testing.assert_close(packet.anchor_observation.voxel_size, original_voxel)


def test_overlap2_failure_after_graph_insertion_rolls_back_independent_nodes() -> None:
    shared_feature = torch.randn(24, 6, 12)
    features = {frame_id: shared_feature for frame_id in range(6)}
    poses0 = torch.eye(4).repeat(4, 1, 1)
    poses1 = torch.eye(4).repeat(4, 1, 1)
    poses0[:, 0, 3] = torch.arange(4) * 0.1
    poses1[:, 0, 3] = torch.arange(2, 6) * 0.1
    packet0 = _refined_packet(
        0, poses0, (0, 1, 2, 3), feature_by_frame=features
    )
    packet1 = _refined_packet(
        1, poses1, (2, 3, 4, 5), feature_by_frame=features
    )
    renderer = _SyntheticSharedDepthRenderer(
        local_depth=2.0,
        global_depth=2.0,
        fail_on_call=7,
    )
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _two_frame_boundary_backend(gaussian_map, renderer)
    backend.process_packet(packet0)
    nodes_before = {
        node: transform.clone() for node, transform in backend.graph.nodes.items()
    }
    edges_before = list(backend.graph.edges)
    windows_before = list(backend.window_order)
    anchors_before = gaussian_map.anchor_count()

    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        backend.process_packet(packet1)

    assert backend.window_order == windows_before
    assert backend.window_anchor_nodes == {0: 0}
    assert backend.window_end_nodes == {0: 3}
    assert len(backend.graph.edges) == len(edges_before)
    assert set(backend.graph.nodes) == set(nodes_before)
    for node, transform in nodes_before.items():
        torch.testing.assert_close(backend.graph.transform(node), transform)
    assert gaussian_map.anchor_count() == anchors_before
    diagnostic = backend.consume_rendered_overlap_diagnostic()
    assert diagnostic is not None
    assert diagnostic["frame_ids"].tolist() == [2, 3]


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


def test_boundary_local_pose_fallback_keeps_chunk_connected_without_rescaling() -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[1, 0, 3] = 0.1
    poses[2, 0, 3] = 0.25
    poses[3, 0, 3] = 0.4
    packet = _packet(0, poses, (0, 1, 2, 3))

    edge = SphericalSelfiGlobalBackend._boundary_local_pose_fallback_edge(
        packet,
        {"reason": "insufficient_boundary_matches", "hard_gated_boundary_matches": 7},
    )

    assert edge.source == 0
    assert edge.target == 3
    assert edge.edge_type == "boundary_local_ba_pose_fallback"
    scale, rotation, translation = sim3_components(
        edge.measurement_target_to_source
    )
    torch.testing.assert_close(scale, torch.tensor(1.0))
    torch.testing.assert_close(rotation, poses[-1, :3, :3])
    torch.testing.assert_close(translation, poses[-1, :3, 3])
    assert edge.metadata["fallback_used"] is True
    assert edge.metadata["fallback_reason"] == "insufficient_boundary_matches"


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


def test_recent_three_window_photometric_optimization_uses_eight_unique_frames() -> None:
    packets = [
        _packet(0, torch.eye(4).repeat(4, 1, 1), (0, 1, 2, 3)),
        _packet(1, torch.eye(4).repeat(4, 1, 1), (2, 3, 4, 5)),
        _packet(2, torch.eye(4).repeat(4, 1, 1), (4, 5, 6, 7)),
    ]

    class _Mapper:
        def __init__(self) -> None:
            self.optimizer = None
            self.stats = SimpleNamespace(notes=[], n_anchors=0)
            self.frame_ids = None
            self.settings = None
            self.extra_loss_fn = "unset"
            self.commits = 0

        def prepare_spherical_selfi_window(self, frame_ids) -> int:
            return len(frame_ids)

        def optimize_spherical_selfi_window(
            self, *, frame_ids, settings, extra_loss_fn, **kwargs
        ):
            self.frame_ids = tuple(int(value) for value in frame_ids)
            self.settings = dict(settings)
            self.extra_loss_fn = extra_loss_fn
            return {"steps": 100.0, "window_rollback": 0.0}

        def commit_spherical_selfi_window(self) -> None:
            self.commits += 1

    mapper = _Mapper()
    backend = _boundary_backend(PanoGaussianMap(config={}, device="cpu"), mapper=mapper)
    backend.map_optimize_recent_windows = 3
    backend.map_optimize_photometric_only = True
    backend.map_optimize_skybox = False
    backend.window_order = [0, 1, 2]
    backend.packets = {
        packet.window_id: packet.compact_for_memory() for packet in packets
    }
    synchronized = {}

    def synchronize(window_id: int, *, optimized_frame_ids=None) -> None:
        synchronized["window_id"] = int(window_id)
        synchronized["frame_ids"] = tuple(optimized_frame_ids or ())

    backend._synchronize_joint_optimized_window = synchronize
    backend._enqueue_map_optimization(2, packets[2].frame_ids, 100)
    assert backend._pending_map_optimization == [(2, tuple(range(8)), 100)]
    backend._pending_map_optimization.clear()

    metrics = backend._run_map_optimization(2, tuple(range(8)), 100)

    assert mapper.frame_ids == tuple(range(8))
    assert mapper.settings["active_owner_window_ids"] == (0, 1, 2)
    assert mapper.settings["fixed_pose_frame_ids"] == []
    assert mapper.settings["photometric_only"] is True
    assert mapper.settings["optimize_skybox"] is False
    assert mapper.settings["pose_prior_weight"] == 0.0
    assert mapper.extra_loss_fn is None
    assert synchronized == {"window_id": 2, "frame_ids": tuple(range(8))}
    assert metrics["optimized_frame_count"] == 8.0
    assert metrics["active_owner_window_count"] == 3.0
    assert metrics["photometric_only"] == 1.0
    assert mapper.commits == 1


def test_gaussian_only_staged_backend_uses_recent_owners_without_pose_sync() -> None:
    packets = [
        _packet(0, torch.eye(4).repeat(4, 1, 1), (0, 1, 2, 3)),
        _packet(1, torch.eye(4).repeat(4, 1, 1), (2, 3, 4, 5)),
        _packet(2, torch.eye(4).repeat(4, 1, 1), (4, 5, 6, 7)),
    ]

    class _Mapper:
        def __init__(self) -> None:
            self.optimizer = None
            self.stats = SimpleNamespace(notes=[], n_anchors=0)
            self.call = None

        def prepare_spherical_selfi_window(self, frame_ids) -> int:
            return len(frame_ids)

        def optimize_spherical_selfi_staged(self, **kwargs):
            self.call = kwargs
            return {
                "steps": 40.0,
                "window_rollback": 0.0,
                "pose_unchanged": 1.0,
                "owner_transform_unchanged": 1.0,
            }

    mapper = _Mapper()
    backend = _boundary_backend(
        PanoGaussianMap(config={}, device="cpu"), mapper=mapper
    )
    backend.map_optimization_strategy = "gaussian_only_staged"
    backend.map_optimize_recent_windows = 3
    backend.map_optimize_config.update(
        {
            "sample_observations_per_step": 2,
            "appearance": {"steps": 30},
            "geometry": {"steps": 10},
        }
    )
    backend.window_order = [0, 1, 2]
    backend.packets = {
        packet.window_id: packet.compact_for_memory() for packet in packets
    }
    backend._optimization_packets[2] = packets[2]
    backend._synchronize_joint_optimized_window = lambda *args, **kwargs: pytest.fail(
        "Gaussian-only staged optimization synchronized poses"
    )

    metrics = backend._run_map_optimization(2, packets[2].frame_ids, 40)

    assert mapper.call is not None
    assert tuple(mapper.call["frame_ids"]) == tuple(range(8))
    assert tuple(mapper.call["active_owner_window_ids"]) == (0, 1, 2)
    assert mapper.call["settings"]["sample_observations_per_step"] == 2
    assert mapper.call["settings"]["appearance"]["steps"] == 30
    assert mapper.call["settings"]["geometry"]["steps"] == 10
    assert metrics["pose_refine_enabled"] == 0.0
    assert metrics["pose_unchanged"] == 1.0
    assert metrics["owner_transform_unchanged"] == 1.0


def test_gaussian_only_joint_3dgs_backend_uses_recent_owners_without_pose_sync() -> None:
    packets = [
        _packet(0, torch.eye(4).repeat(4, 1, 1), (0, 1, 2, 3)),
        _packet(1, torch.eye(4).repeat(4, 1, 1), (2, 3, 4, 5)),
        _packet(2, torch.eye(4).repeat(4, 1, 1), (4, 5, 6, 7)),
    ]

    class _Mapper:
        def __init__(self) -> None:
            self.optimizer = None
            self.stats = SimpleNamespace(notes=[], n_anchors=0)
            self.call = None

        def prepare_spherical_selfi_window(self, frame_ids) -> int:
            return len(frame_ids)

        def optimize_spherical_selfi_joint_3dgs(self, **kwargs):
            self.call = kwargs
            return {
                "steps": 40.0,
                "window_rollback": 0.0,
                "pose_unchanged": 1.0,
                "owner_transform_unchanged": 1.0,
            }

    mapper = _Mapper()
    gaussian_map = PanoGaussianMap(
        config={
            "MapRepresentation": {
                "gaussian_parameterization": "traditional_3dgs"
            }
        },
        device="cpu",
    )
    backend = _boundary_backend(gaussian_map, mapper=mapper)
    backend.map_optimization_strategy = "gaussian_only_joint_3dgs"
    backend.map_optimize_recent_windows = 3
    backend.map_optimize_config.update(
        {
            "sample_observations_per_step": 2,
            "rgb_l1_weight": 0.8,
            "dssim_weight": 0.2,
            "max_rgb_worsening": 0.005,
        }
    )
    backend.window_order = [0, 1, 2]
    backend.packets = {
        packet.window_id: packet.compact_for_memory() for packet in packets
    }
    backend._optimization_packets[2] = packets[2]
    backend._synchronize_joint_optimized_window = lambda *args, **kwargs: pytest.fail(
        "Gaussian-only joint optimization synchronized poses"
    )

    metrics = backend._run_map_optimization(2, packets[2].frame_ids, 40)

    assert mapper.call is not None
    assert tuple(mapper.call["frame_ids"]) == tuple(range(8))
    assert tuple(mapper.call["active_owner_window_ids"]) == (0, 1, 2)
    assert mapper.call["settings"]["steps"] == 40
    assert mapper.call["settings"]["sample_observations_per_step"] == 2
    assert mapper.call["settings"]["max_rgb_worsening"] == 0.005
    assert metrics["pose_refine_enabled"] == 0.0
    assert metrics["photometric_only"] == 1.0


def test_mapper_rejection_snapshots_after_durable_keyframe_registration() -> None:
    packet = _packet(
        0,
        torch.eye(4).repeat(4, 1, 1),
        (0, 1, 2, 3),
    )

    class _Mapper:
        def __init__(self) -> None:
            self.optimizer = None
            self.stats = SimpleNamespace(notes=[], n_anchors=0)
            self.keyframe_ids: list[int] = []
            self.restore_count = 0
            self.rollback_count = 0

        def set_spherical_selfi_observation_geometry(self, *args, **kwargs) -> None:
            return None

        def prepare_spherical_selfi_window(self, frame_ids) -> int:
            for frame_id in frame_ids:
                if int(frame_id) not in self.keyframe_ids:
                    self.keyframe_ids.append(int(frame_id))
            return len(frame_ids)

        def snapshot_frontend_geometry_state(self):
            return {"keyframe_ids": tuple(self.keyframe_ids)}

        def restore_frontend_geometry_state(self, state) -> None:
            # This models the real Mapper's strict topology invariant.  A
            # proposal rollback may restore geometry, but must not attempt to
            # erase keyframes durably registered by prepare().
            assert tuple(self.keyframe_ids) == state["keyframe_ids"]
            self.restore_count += 1

        def optimize_spherical_selfi_window(self, **kwargs):
            return {"steps": 1.0, "window_rollback": 0.0}

        def rollback_spherical_selfi_window(self) -> None:
            self.rollback_count += 1

        def commit_spherical_selfi_window(self) -> None:
            raise AssertionError("A rejected mapper proposal must not commit")

    mapper = _Mapper()
    backend = _chunk_stride_backend(min_matches=8, mapper=mapper)
    backend.graph.add_node(0, sim3_identity())
    backend.window_order = [0]
    backend.window_anchor_nodes[0] = 0
    backend.packets[0] = packet.compact_for_memory()
    backend._optimization_packets[0] = packet

    def reject_sync(window_id: int, *, optimized_frame_ids=None) -> None:
        raise RuntimeError(f"synthetic proposal rejection for {window_id}")

    backend._synchronize_joint_optimized_window = reject_sync
    metrics = backend._run_map_optimization(0, packet.frame_ids, 1)

    assert metrics["window_rollback"] == 1.0
    assert mapper.keyframe_ids == [0, 1, 2, 3]
    assert mapper.restore_count == 1
    assert mapper.rollback_count == 1


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


def test_recent_owner_gaussian_scope_freezes_older_rows_exactly() -> None:
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    gaussian_map.xyz = torch.nn.Parameter(torch.zeros(4, 3))
    gaussian_map.rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    )
    gaussian_map.scaling = torch.nn.Parameter(torch.zeros(4, 3))
    gaussian_map.opacity_logit = torch.nn.Parameter(torch.zeros(4, 1))
    gaussian_map.features = torch.nn.Parameter(torch.zeros(4, 3))
    gaussian_map.sh_rest = torch.nn.Parameter(torch.zeros(4, 0, 3))
    gaussian_map._anchor_owner_window_id = torch.tensor(
        [0, 1, 2, 3], dtype=torch.int32
    )
    mapper = PanoGaussianMapper(gaussian_map)
    mapper.optim_cfg["FeedForwardWindow"] = {
        "gaussian_scope": "owner_windows",
        "active_owner_window_ids": (1, 2, 3),
        "gaussian_lr_scale": 1.0,
    }

    scales = mapper._feedforward_gaussian_scales([])
    assert scales is not None
    torch.testing.assert_close(scales, torch.tensor([0.0, 1.0, 1.0, 1.0]))

    with torch.no_grad():
        gaussian_map.rotation[:, 0] = 2.0
        gaussian_map.scaling.fill_(30.0)
        gaussian_map.opacity_logit.fill_(20.0)
    frozen_rotation = gaussian_map.rotation.detach()[0].clone()
    frozen_scaling = gaussian_map.scaling.detach()[0].clone()
    frozen_opacity = gaussian_map.opacity_logit.detach()[0].clone()
    mapper._sanitize_active_gaussian_rows(scales)
    torch.testing.assert_close(gaussian_map.rotation.detach()[0], frozen_rotation)
    torch.testing.assert_close(gaussian_map.scaling.detach()[0], frozen_scaling)
    torch.testing.assert_close(gaussian_map.opacity_logit.detach()[0], frozen_opacity)
    torch.testing.assert_close(
        torch.linalg.norm(gaussian_map.rotation.detach()[1:], dim=-1),
        torch.ones(3),
    )
    torch.testing.assert_close(
        gaussian_map.scaling.detach()[1:], torch.full((3, 3), 20.0)
    )
    torch.testing.assert_close(
        gaussian_map.opacity_logit.detach()[1:], torch.full((3, 1), 12.0)
    )

    optimizer = torch.optim.AdamW(
        [{"params": [gaussian_map.xyz], "lr": 1.0e-2}], weight_decay=0.0
    )
    gaussian_map.xyz.grad = torch.ones_like(gaussian_map.xyz)
    optimizer.step()
    mapper._apply_gaussian_adamw_update_scales(optimizer, scales)
    torch.testing.assert_close(gaussian_map.xyz.detach()[0], torch.zeros(3))
    assert bool((gaussian_map.xyz.detach()[1:].abs() > 0.0).all())


def test_photometric_only_mapper_loss_disables_all_non_rgb_terms() -> None:
    mapper = PanoGaussianMapper(PanoGaussianMap(config={}, device="cpu"))
    weights = mapper._feedforward_loss_weights(photometric_only=True)

    assert weights.photometric == mapper.loss_weights.photometric
    assert weights.photometric_mode == mapper.loss_weights.photometric_mode
    assert weights.depth == 0.0
    assert weights.opacity == 0.0
    assert weights.distortion == 0.0
    assert weights.sky_alpha == 0.0


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


def test_boundary_pose_sync_retracts_graph_mapper_feedback_to_sim3() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 1.0
    packet = _packet(0, poses, (0, 1))
    optimized = {0: torch.eye(4), 1: torch.eye(4)}
    optimized[0][:3, :3] = torch.tensor(
        [[1.0, 0.03, 0.0], [0.0, 1.0, -0.02], [0.01, 0.0, 1.0]]
    )
    optimized[1][:3, :3] = torch.tensor(
        [[1.0, -0.04, 0.01], [0.02, 1.0, 0.0], [0.0, 0.03, 1.0]]
    )
    optimized[1][0, 3] = 2.0

    class _Mapper:
        def refined_pose_c2w(self, frame_id: int):
            return optimized.get(int(frame_id))

    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    backend = _boundary_backend(gaussian_map, mapper=_Mapper())
    backend.graph.add_node(0, sim3_identity())
    backend.graph.add_node(
        1, sim3_from_components(2.0, torch.eye(3), torch.tensor([2.0, 0.0, 0.0]))
    )
    # Reproduce the numerical shear that used to accumulate across repeated
    # graph -> mapper -> graph synchronization cycles.
    sheared = backend.graph.transform(0).clone()
    sheared[:3, :3] = torch.tensor(
        [[1.0, 0.02, 0.0], [0.0, 1.0, 0.01], [-0.01, 0.0, 1.0]]
    )
    backend.graph.nodes[0] = sheared
    backend.window_anchor_nodes = {0: 0}
    backend.boundary_node_order = [0, 1]
    backend.packets = {0: packet}
    backend.window_order = [0]
    backend.frame_windows = {0: {0}, 1: {0}}
    backend.frame_owner_window = {0: 0, 1: 0}
    backend.frame_depth_owner_window = {0: 0, 1: 0}

    backend._synchronize_joint_optimized_window(0)

    eye = torch.eye(3)
    for transform in backend.graph.nodes.values():
        _, rotation, _ = sim3_components(transform)
        torch.testing.assert_close(rotation.T @ rotation, eye, atol=1.0e-6, rtol=1.0e-6)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0), atol=1.0e-6, rtol=1.0e-6)
    for pose in packet.local_poses_c2w:
        rotation = pose[:3, :3]
        torch.testing.assert_close(rotation.T @ rotation, eye, atol=1.0e-6, rtol=1.0e-6)


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


def test_mapper_complete_geometry_snapshot_validates_before_atomic_commit() -> None:
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map)
    for frame_id in (5, 6):
        mapper.observations[frame_id] = MapperObservation(
            frame_id=frame_id,
            image=torch.zeros(3, 4, 8),
            pose_c2w=torch.eye(4),
            target_depth=torch.full((1, 4, 8), 2.0),
            target_depth_local=torch.full((1, 4, 8), 2.0),
            owner_window_id=0,
        )

    def update(frame_id: int, x: float, scale: float):
        pose = torch.eye(4)
        pose[0, 3] = x
        return SimpleNamespace(
            pose_c2w=pose,
            depth_scale=scale,
            owner_window_id=0,
            depth_owner_window_id=0,
            depth_scales_by_window={0: scale},
        )

    mapper.apply_frontend_geometry_snapshot(
        {5: update(5, 1.0, 2.0), 6: update(6, 2.0, 3.0)}
    )
    before_pose = mapper.refined_pose_c2w(5).clone()
    before_depth = mapper.observations[5].target_depth.clone()
    with pytest.raises(ValueError, match="Invalid complete-snapshot depth scale"):
        mapper.apply_frontend_geometry_snapshot(
            {5: update(5, 9.0, 4.0), 6: update(6, 10.0, -1.0)}
        )
    torch.testing.assert_close(mapper.refined_pose_c2w(5), before_pose)
    torch.testing.assert_close(mapper.observations[5].target_depth, before_depth)

    transaction = mapper.snapshot_frontend_geometry_state()
    mapper.apply_frontend_geometry_snapshot(
        {5: update(5, 7.0, 4.0), 6: update(6, 8.0, 5.0)}
    )
    mapper.restore_frontend_geometry_state(transaction)
    torch.testing.assert_close(mapper.refined_pose_c2w(5), before_pose)
    torch.testing.assert_close(mapper.observations[5].target_depth, before_depth)

    canonical_pose = torch.eye(4)
    canonical_pose[1, 3] = 3.0
    transaction = mapper.snapshot_frontend_geometry_state()
    assert mapper.apply_canonical_pose_state(
        {5: canonical_pose},
        revision=17,
    ) == 1
    torch.testing.assert_close(mapper.refined_pose_c2w(5), canonical_pose)
    torch.testing.assert_close(
        mapper.observations[5].pose_c2w,
        canonical_pose,
    )
    assert mapper.observations[5].pose_revision == 17
    mapper.restore_frontend_geometry_state(transaction)
    torch.testing.assert_close(mapper.refined_pose_c2w(5), before_pose)
    torch.testing.assert_close(mapper.observations[5].pose_c2w, before_pose)
    assert mapper.observations[5].pose_revision == 0


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


def test_synthetic_window_runtime_replaces_head_depth_with_aligned_pager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontend.spherical_selfi.runtime as runtime_module

    providers = []

    class FakePaGeRDepthProvider:
        def __init__(self, config, *, device) -> None:
            self.config = config
            self.device = torch.device(device)
            self.last_raw = None
            providers.append(self)

        def reset(self) -> None:
            pass

        def predict(self, images, frame_ids):
            batch, views, _, height, width = images.shape
            rows = torch.arange(height, dtype=torch.float32).view(height, 1)
            cols = torch.arange(width, dtype=torch.float32).view(1, width)
            raw = 0.5 + rows / max(1, height) + cols / max(1, width)
            raw = raw.view(1, 1, 1, height, width).repeat(
                batch, views, 1, 1, 1
            )
            self.last_raw = raw.to(self.device)
            return self.last_raw, {
                "inference_sec": 0.01,
                "cache_hits": 0,
                "cache_misses": int(frame_ids.numel()),
                "cache_hit_ratio": 0.0,
                "cache_entries": int(frame_ids.numel()),
            }

    monkeypatch.setattr(
        runtime_module,
        "PaGeRDepthProvider",
        FakePaGeRDepthProvider,
    )
    config = stage2_default_config()
    config["image"] = {
        "height": 8,
        "width": 16,
        "head_height": 8,
        "head_width": 16,
    }
    config["head"].update({"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12})
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2_pager.pt"
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
        "pager_depth": {
            "enabled": True,
            "repo_path": "/unused/pager",
            "min_valid_pixels": 8,
            "min_valid_ratio": 0.1,
        },
        "local_ba": {"enabled": False},
    }
    frontend = SphericalSelfiWindowFrontend(config)
    for frame_id in range(3):
        frontend.track(PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id))
    outputs = frontend.pop_ready_outputs() + frontend.flush()
    packets = frontend.consume_local_gaussian_windows()
    diagnostics = frontend.consume_local_ba_diagnostics()

    assert len(providers) == 1
    assert len(packets) == 1
    observation = packets[0].observation
    scales = torch.tensor(diagnostics[0]["pager_depth"]["scales"]).view(
        1, 3, 1, 1, 1
    )
    expected = providers[0].last_raw * scales
    torch.testing.assert_close(observation.initial_depth, expected)
    torch.testing.assert_close(observation.refined_depth, expected)
    torch.testing.assert_close(
        observation.depth_residual,
        torch.zeros_like(observation.depth_residual),
    )
    assert packets[0].metadata["pager_depth_enabled"] is True
    assert diagnostics[0]["pager_depth"]["cache_misses"] == 3
    by_frame = {int(output.frame_id): output for output in outputs}
    for index in range(3):
        torch.testing.assert_close(
            by_frame[index].inverse_depth,
            expected[0, index].reciprocal(),
        )


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


def test_window_scheduler_overlap2_emits_two_new_frames_and_validates_partial(
    tmp_path: Path,
) -> None:
    config = stage2_default_config()
    config["image"] = {
        "height": 8,
        "width": 16,
        "head_height": 8,
        "head_width": 16,
    }
    config["head"].update(
        {"channels": [8, 12, 16, 24], "mlp_hidden_dim": 12}
    )
    head = SphericalSelfiGaussianHead(**config["head"], renderer_config=config)
    checkpoint = tmp_path / "stage2_stride2.pt"
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
            "stride": 2,
            "expected_overlap_frames": 2,
            "enforce_exact_overlap": True,
            "verification_size": [4, 8],
        },
        "local_ba": {"enabled": False},
    }
    frontend = SphericalSelfiWindowFrontend(config)
    packets = []
    emitted: list[int] = []
    for frame_id in range(9):
        frontend.track(
            PanoFrame(torch.rand(3, 8, 16), float(frame_id), frame_id)
        )
        emitted.extend(
            int(output.frame_id) for output in frontend.pop_ready_outputs()
        )
        packets.extend(frontend.consume_local_gaussian_windows())
    emitted.extend(int(output.frame_id) for output in frontend.flush())
    packets.extend(frontend.consume_local_gaussian_windows())
    frontend.flush()

    assert [packet.frame_ids for packet in packets] == [
        (0, 1, 2, 3),
        (2, 3, 4, 5),
        (4, 5, 6, 7),
        (6, 7, 8),
    ]
    assert emitted == list(range(9))
    assert len(emitted) == len(set(emitted))
    assert all(
        torch.equal(packet.local_poses_c2w[0], torch.eye(4))
        for packet in packets
    )
    for previous, current in zip(packets, packets[1:]):
        assert (
            set(previous.frame_ids) & set(current.frame_ids)
            == set(current.frame_ids[:2])
        )


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


def test_pointmap_overlap_recovers_full_current_to_previous_sim3() -> None:
    height, width = 24, 48
    angle = 0.23
    rotation = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=torch.float32,
    )
    expected = sim3_from_components(
        1.4,
        rotation,
        torch.tensor([0.35, -0.08, 0.18]),
    )
    current_poses = torch.eye(4).repeat(4, 1, 1)
    current_poses[1, :3, 3] = torch.tensor([0.22, 0.03, -0.04])
    previous_poses = torch.eye(4).repeat(4, 1, 1)
    previous_poses[2] = apply_sim3_to_c2w(expected, current_poses[0])
    previous_poses[3] = apply_sim3_to_c2w(expected, current_poses[1])
    previous = _packet(
        0,
        previous_poses,
        (0, 1, 2, 3),
        height=height,
        width=width,
    )
    current = _packet(
        1,
        current_poses,
        (2, 3, 4, 5),
        height=height,
        width=width,
    )
    current_depth = torch.full((4, 1, height, width), 2.0)
    previous_depth = torch.full((4, 1, height, width), 2.0)
    previous_depth[2:] = 2.8
    _replace_packet_depth(previous, previous_depth)
    _replace_packet_depth(current, current_depth)
    backend = _pointmap_chunk_backend(
        min_points=64,
        min_points_per_frame=24,
        max_points_per_frame=64,
    )

    measurement, diagnostics = backend._pointmap_overlap_alignment(
        previous,
        current,
    )

    assert measurement is not None, diagnostics
    assert diagnostics["accepted"] is True
    assert diagnostics["weight_mode"] == "uniform_then_huber_irls"
    assert diagnostics["confidence_weighting"] is False
    assert diagnostics["pose_prior_used"] is False
    expected_scale, expected_rotation, expected_translation = sim3_components(expected)
    scale, recovered_rotation, recovered_translation = sim3_components(measurement)
    assert float(scale) == pytest.approx(float(expected_scale), rel=2.0e-3)
    assert torch.allclose(recovered_rotation, expected_rotation, atol=2.0e-3)
    assert torch.allclose(recovered_translation, expected_translation, atol=2.0e-3)


def test_pointmap_diagnostics_only_accepts_finite_rejected_sim3() -> None:
    height, width = 24, 48
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(
        0,
        poses,
        (0, 1, 2, 3),
        height=height,
        width=width,
    )
    current = _packet(
        1,
        poses,
        (2, 3, 4, 5),
        height=height,
        width=width,
    )
    previous_depth = torch.full((4, 1, height, width), 2.0)
    current_depth = torch.full((4, 1, height, width), 2.0)
    current_depth[:, :, :, 1::2] = 4.0
    _replace_packet_depth(previous, previous_depth)
    _replace_packet_depth(current, current_depth)

    strict_backend = _pointmap_chunk_backend(
        min_points=64,
        min_points_per_frame=24,
        max_points_per_frame=64,
    )
    strict_measurement, strict_diagnostics = (
        strict_backend._pointmap_overlap_alignment(previous, current)
    )
    assert strict_measurement is None
    assert strict_diagnostics["pointmap_sim3_quality_gate_passed"] is False

    diagnostic_backend = _pointmap_chunk_backend(
        min_points=64,
        min_points_per_frame=24,
        max_points_per_frame=64,
        acceptance_policy="diagnostics_only",
    )
    measurement, diagnostics = diagnostic_backend._pointmap_overlap_alignment(
        previous,
        current,
    )

    assert measurement is not None
    assert bool(torch.isfinite(measurement).all())
    assert diagnostics["accepted"] is True
    assert diagnostics["reason"] == "accepted_diagnostics_only"
    assert diagnostics["pointmap_sim3_accepted"] is True
    assert diagnostics["pointmap_sim3_quality_gate_passed"] is False
    assert diagnostics["pointmap_sim3_acceptance_overridden"] is True
    assert diagnostics["pointmap_sim3_acceptance_policy"] == "diagnostics_only"


def test_pointmap_sampling_jointly_filters_sky_dynamic_and_invalid_geometry() -> None:
    height, width = 24, 48
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(
        0,
        poses,
        (0, 1, 2, 3),
        height=height,
        width=width,
    )
    current = _packet(
        1,
        poses,
        (2, 3, 4, 5),
        height=height,
        width=width,
    )
    previous.sky_prob[:, 2, :, :4] = 1.0
    previous.sky_mask[:, 2, :, :4] = True
    current.static_mask[:, 0, :, 4:6] = False
    previous.geometry_consistency[:, 2, :, 6:8] = False
    current.finite_gaussian_mask[:, 0, :, 8:10] = False
    current.valid_mask[:, 0, :, 8:10] = False
    current_depth = current.observation.refined_depth.detach().clone()
    current_depth[:, 0, :, 10:12] = torch.nan
    current.observation = current.observation.with_geometry(
        refined_depth=current_depth
    )
    backend = _pointmap_chunk_backend(
        min_points=64,
        min_points_per_frame=32,
        max_points_per_frame=64,
    )

    geometry = backend._collect_pointmap_overlap_frame_geometry(
        previous,
        current,
        2,
    )

    assert int(geometry.uv.shape[0]) == 64
    sampled_previous_valid = sample_erp_with_wrap(
        geometry.previous_valid_image.float(), geometry.uv
    )[..., 0]
    sampled_current_valid = sample_erp_with_wrap(
        geometry.current_valid_image.float(), geometry.uv
    )[..., 0]
    sampled_sky = sample_erp_with_wrap(
        geometry.sky_union_image.float(), geometry.uv
    )[..., 0]
    assert bool((sampled_previous_valid >= 0.5).all())
    assert bool((sampled_current_valid >= 0.5).all())
    assert bool((sampled_sky < 0.5).all())
    assert bool(torch.isfinite(geometry.previous_points).all())
    assert bool(torch.isfinite(geometry.current_points).all())


def test_pointmap_chunk_graph_delays_node_and_uses_current_first_depth() -> None:
    height, width = 16, 32
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(
        0,
        poses,
        (0, 1, 2, 3),
        height=height,
        width=width,
    )
    current = _packet(
        1,
        poses,
        (2, 3, 4, 5),
        height=height,
        width=width,
    )
    _replace_packet_depth(
        previous,
        torch.full((4, 1, height, width), 3.0),
    )
    _replace_packet_depth(
        current,
        torch.full((4, 1, height, width), 2.0),
    )
    _attach_identity_stride_matches(previous)
    backend = _pointmap_chunk_backend(hierarchical=True)

    first_result = backend.process_packet(previous)
    assert set(backend.graph.nodes) == {0}
    assert backend.graph.edges == []
    first_sim3 = backend.pointmap_sim3_aligned_pose_snapshot()
    assert set(first_sim3) == {0, 1, 2, 3}
    torch.testing.assert_close(first_sim3[0], torch.eye(4))
    assert "q" not in first_result.diagnostics["alignment"]
    assert "c" not in first_result.diagnostics["alignment"]

    # Simulate graph/photo feedback and verify that the diagnostic frontend
    # snapshot owns independent tensors rather than aliases of graph nodes.
    backend.graph.nodes[0][0, 3] = 5.0
    torch.testing.assert_close(
        backend.pointmap_sim3_aligned_pose_snapshot()[0],
        torch.eye(4),
    )
    backend.graph.nodes[0][0, 3] = 0.0

    result = backend.process_packet(current)

    assert set(backend.graph.nodes) == {0, 2}
    assert 4 not in backend.graph.nodes
    assert backend.window_anchor_nodes == {0: 0, 1: 2}
    assert len(backend.graph.edges) == 1
    factor = backend.graph.edges[0]
    assert isinstance(factor, DenseSphericalFactorBlock)
    assert (factor.source, factor.target) == (0, 2)
    assert torch.allclose(
        factor.target_depth,
        torch.full_like(factor.target_depth, 2.0),
    )
    assert factor.metadata["cross_packet_geometry"] is True
    assert factor.metadata["matched_target_index"] == 2
    assert factor.metadata["geometry_target_index"] == 0
    assert float(sim3_components(backend.graph.transform(2))[0]) == pytest.approx(
        1.5, rel=2.0e-3
    )
    sim3_only = backend.pointmap_sim3_aligned_pose_snapshot()
    assert set(sim3_only) == {0, 1, 2, 3, 4, 5}
    assert float(sim3_only[2][0, 3]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(backend.graph.objective().detach().cpu()) < 1.0e-8
    assert backend.frame_pose_owner_node[2] == 2
    assert backend.frame_pose_owner_node[3] == 2
    assert torch.allclose(backend.frame_local_pose_in_owner[2], torch.eye(4))
    assert (
        result.diagnostics["boundary_factor"]["transferred_overlap_frames"]
        == 2
    )

    following = _packet(
        2,
        poses,
        (4, 5, 6, 7),
        height=height,
        width=width,
    )
    _replace_packet_depth(
        following,
        torch.full((4, 1, height, width), 4.0),
    )
    _attach_identity_stride_matches(current)
    backend.process_packet(following)

    assert set(backend.graph.nodes) == {0, 2, 4}
    assert 6 not in backend.graph.nodes
    assert [(edge.source, edge.target) for edge in backend.graph.edges] == [
        (0, 2),
        (2, 4),
    ]
    assert backend.frame_pose_owner_node[4] == 4
    assert backend.frame_pose_owner_node[5] == 4
    assert backend.submaps[0].boundary_node_ids == [0, 2, 4]
    assert backend.submaps[0].window_ids == [0, 1, 2]
    assert float(sim3_components(backend.graph.transform(4))[0]) == pytest.approx(
        0.75, rel=2.0e-3
    )
    geometry = backend.pop_frame_geometry_updates()
    assert geometry[4].depth_scale == pytest.approx(0.75, rel=2.0e-3)
    torch.testing.assert_close(
        geometry[4].pose_c2w,
        apply_sim3_to_c2w(
            backend.graph.transform(4),
            following.local_poses_c2w[0],
        ),
    )
    assert float(following.observation.refined_depth[0, 0].mean()) == 4.0


def test_pointmap_sim3_frontend_pose_snapshot_keeps_first_overlap_owner() -> None:
    previous_poses = torch.eye(4).repeat(4, 1, 1)
    previous_poses[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    current_poses = torch.eye(4).repeat(4, 1, 1)
    current_poses[:, 0, 3] = torch.tensor([20.0, 21.0, 22.0, 23.0])
    previous = _packet(0, previous_poses, (0, 1, 2, 3))
    current = _packet(1, current_poses, (2, 3, 4, 5))
    backend = _pointmap_chunk_backend()

    backend._record_pointmap_sim3_alignment(previous, torch.eye(4))
    backend._record_pointmap_sim3_alignment(current, torch.eye(4))

    snapshot = backend.pointmap_sim3_aligned_pose_snapshot()
    assert set(snapshot) == {0, 1, 2, 3, 4, 5}
    # Conflicting current-chunk estimates for the overlap frames do not
    # replace the canonical poses already emitted by the previous chunk.
    assert float(snapshot[2][0, 3]) == pytest.approx(2.0)
    assert float(snapshot[3][0, 3]) == pytest.approx(3.0)
    # Only genuinely new frames are appended from the current chunk.
    assert float(snapshot[4][0, 3]) == pytest.approx(2.0)
    assert float(snapshot[5][0, 3]) == pytest.approx(3.0)


def test_pointmap_alignment_finalizes_refiner_once_before_sampling(
    monkeypatch,
) -> None:
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _refined_packet(0, poses, (0, 1, 2, 3))
    current = _refined_packet(1, poses, (2, 3, 4, 5))
    for packet in (previous, current):
        _attach_identity_stride_matches(packet)
        packet.metadata["voxel_anchor_refiner_requested"] = True
        packet.metadata["voxel_anchor_refiner_pending"] = True

    events: list[tuple[str, int]] = []

    def finalize_refiner(packet: LocalGaussianWindowPacket):
        assert packet.metadata["local_ba_accepted"] is True
        events.append(("refiner", int(packet.window_id)))
        metadata = dict(packet.metadata)
        metadata["voxel_anchor_refiner_pending"] = False
        metadata["voxel_anchor_refiner_enabled"] = True
        return replace(packet, metadata=metadata)

    backend = _pointmap_chunk_backend(
        renderer=_SyntheticSharedDepthRenderer(
            local_depth=2.0,
            global_depth=2.0,
        ),
        packet_refiner=finalize_refiner,
    )
    pointmap_alignment = backend._pointmap_overlap_alignment

    def record_pointmap_alignment(
        previous_packet: LocalGaussianWindowPacket,
        current_packet: LocalGaussianWindowPacket,
    ):
        assert current_packet.metadata["voxel_anchor_refiner_pending"] is False
        events.append(("sim3", int(current_packet.window_id)))
        return pointmap_alignment(previous_packet, current_packet)

    monkeypatch.setattr(
        backend,
        "_pointmap_overlap_alignment",
        record_pointmap_alignment,
    )

    backend.process_packet(previous)
    backend.process_packet(current)

    assert events == [("refiner", 0), ("refiner", 1), ("sim3", 1)]


def test_pointmap_alignment_failure_rolls_back_node_and_owner_transfer() -> None:
    height, width = 16, 32
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(
        0,
        poses,
        (0, 1, 2, 3),
        height=height,
        width=width,
    )
    current = _packet(
        1,
        poses,
        (2, 3, 4, 5),
        height=height,
        width=width,
    )
    _replace_packet_depth(
        previous,
        torch.full((4, 1, height, width), 2.0),
    )
    _replace_packet_depth(
        current,
        torch.full((4, 1, height, width), 2.0),
    )
    _attach_identity_stride_matches(previous)
    current.sky_prob = torch.ones_like(current.sky_prob)
    current.sky_mask = torch.ones_like(current.sky_mask)
    backend = _pointmap_chunk_backend()
    backend.process_packet(previous)
    owners_before = dict(backend.frame_pose_owner_node)
    sim3_before = backend.pointmap_sim3_aligned_pose_snapshot()

    with pytest.raises(RuntimeError, match=r"point-map Sim\(3\) alignment failed"):
        backend.process_packet(current)

    assert set(backend.graph.nodes) == {0}
    assert backend.window_order == [0]
    assert backend.frame_pose_owner_node == owners_before
    sim3_after = backend.pointmap_sim3_aligned_pose_snapshot()
    assert set(sim3_after) == set(sim3_before)
    for frame_id in sim3_before:
        torch.testing.assert_close(sim3_after[frame_id], sim3_before[frame_id])
    assert backend._last_overlap_alignment_failure is not None
    assert backend._last_overlap_alignment_failure["accepted"] is False


def test_global_map_overlap_initializes_absolute_sim3_and_anchors_only_scale() -> None:
    height, width = 24, 48
    angle = 0.19
    rotation = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=torch.float32,
    )
    expected = sim3_from_components(
        1.4, rotation, torch.tensor([0.25, -0.06, 0.12])
    )
    current_poses = torch.eye(4).repeat(4, 1, 1)
    current_poses[1, :3, 3] = torch.tensor([0.2, 0.03, -0.04])
    previous_poses = torch.eye(4).repeat(4, 1, 1)
    previous_poses[2] = apply_sim3_to_c2w(expected, current_poses[0])
    previous_poses[3] = apply_sim3_to_c2w(expected, current_poses[1])
    previous = _packet(0, previous_poses, (0, 1, 2, 3), height=height, width=width)
    current = _packet(1, current_poses, (2, 3, 4, 5), height=height, width=width)
    _replace_packet_depth(previous, torch.full((4, 1, height, width), 2.8))
    _replace_packet_depth(current, torch.full((4, 1, height, width), 2.0))
    _attach_identity_stride_matches(previous)
    renderer = _SyntheticSharedDepthRenderer(local_depth=2.0, global_depth=2.8)
    backend = _pointmap_chunk_backend(
        min_points=64,
        min_points_per_frame=24,
        max_points_per_frame=64,
        acceptance_policy="diagnostics_only",
        renderer=renderer,
        alignment_mode="two_frame_global_map_full_sim3",
    )

    candidate, _, map_success, candidate_diagnostics = (
        backend._global_map_overlap_alignment(
            previous, current, torch.eye(4)
        )
    )
    assert map_success is True, candidate_diagnostics
    assert candidate is not None
    candidate_scale, candidate_rotation, candidate_translation = sim3_components(candidate)
    expected_scale, expected_rotation, expected_translation = sim3_components(expected)
    assert float(candidate_scale) == pytest.approx(float(expected_scale), rel=2.0e-4)
    assert torch.allclose(candidate_rotation, expected_rotation, atol=2.0e-4)
    assert torch.allclose(candidate_translation, expected_translation, atol=5.0e-4)
    assert candidate_diagnostics["per_frame_final_weight_sum"] == pytest.approx(
        [0.5, 0.5], abs=1.0e-6
    )
    renderer.calls = 0

    first = backend.process_packet(previous)
    assert first.diagnostics["alignment"]["reason"] == "first_window"
    assert renderer.calls == 0
    result = backend.process_packet(current)

    assert renderer.calls == 2
    diagnostics = result.diagnostics["alignment"]
    assert diagnostics["accepted"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["incoming_render_count"] == 0
    recovered = backend.graph.transform(2)
    scale, _, _ = sim3_components(recovered)
    assert float(scale) == pytest.approx(float(expected_scale), rel=2.0e-4)
    dense = [
        factor for factor in backend.graph.edges
        if isinstance(factor, DenseSphericalFactorBlock)
    ]
    anchors = [
        factor for factor in backend.graph.edges
        if isinstance(factor, Sim3GraphEdge)
        and factor.edge_type == "global_map_anchor_sim3"
    ]
    assert len(dense) == 1
    assert dense[0].use_depth is False
    assert dense[0].optimize_scale is False
    assert len(anchors) == 1
    assert (anchors[0].source, anchors[0].target) == (0, 2)
    assert torch.equal(
        anchors[0].information_diag[:6],
        torch.zeros_like(anchors[0].information_diag[:6]),
    )
    assert float(anchors[0].information_diag[6]) > 0.0


def test_global_map_overlap_falls_back_to_unchanged_pointmap_path() -> None:
    height, width = 16, 32
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(0, poses, (0, 1, 2, 3), height=height, width=width)
    current = _packet(1, poses, (2, 3, 4, 5), height=height, width=width)
    _replace_packet_depth(previous, torch.full((4, 1, height, width), 3.0))
    _replace_packet_depth(current, torch.full((4, 1, height, width), 2.0))
    _attach_identity_stride_matches(previous)
    renderer = _SyntheticSharedDepthRenderer(
        local_depth=2.0, global_depth=2.0, alpha=0.0
    )
    backend = _pointmap_chunk_backend(
        renderer=renderer,
        acceptance_policy="diagnostics_only",
        alignment_mode="two_frame_global_map_full_sim3",
    )
    backend.process_packet(previous)
    result = backend.process_packet(current)

    assert result.diagnostics["alignment"]["fallback_used"] is True
    assert result.diagnostics["alignment"]["reason"] == "fallback_pointmap_full_sim3"
    assert float(sim3_components(backend.graph.transform(2))[0]) == pytest.approx(
        1.5, rel=2.0e-3
    )
    dense = next(
        factor for factor in backend.graph.edges
        if isinstance(factor, DenseSphericalFactorBlock)
    )
    assert dense.use_depth is True
    assert dense.optimize_scale is True
    assert not any(
        isinstance(factor, Sim3GraphEdge)
        and factor.edge_type == "global_map_anchor_sim3"
        for factor in backend.graph.edges
    )


def test_dense_spherical_factor_can_strictly_disable_scale_jacobians() -> None:
    graph = GlobalSim3FactorGraph()
    graph.add_node(0, torch.eye(4))
    target = sim3_from_components(
        1.3, torch.eye(3), torch.tensor([0.2, 0.0, 0.0])
    )
    graph.add_node(1, target)
    identity = torch.eye(4)
    bearing = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
    )
    factor = DenseSphericalFactorBlock(
        source=0,
        target=1,
        source_local_pose=identity,
        target_local_pose=identity,
        source_bearing=bearing,
        target_bearing=bearing.roll(1, dims=0),
        source_depth=torch.ones(3),
        target_depth=torch.ones(3),
        factor_weight=torch.ones(3),
        use_depth=False,
        optimize_scale=False,
    )
    graph.add_edge(factor)

    endpoint_ids, blocks, _ = graph._linearize_factor(factor, {1: 0})

    assert endpoint_ids == [1]
    assert torch.equal(blocks[0][:, 6], torch.zeros_like(blocks[0][:, 6]))


def test_global_map_overlap_is_seam_safe_and_accepts_large_finite_scale() -> None:
    height, width = 24, 48
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(0, poses, (0, 1, 2, 3), height=height, width=width)
    current = _packet(1, poses, (2, 3, 4, 5), height=height, width=width)
    _replace_packet_depth(previous, torch.full((4, 1, height, width), 2.0))
    _replace_packet_depth(current, torch.full((4, 1, height, width), 2.0))
    current.static_mask.zero_()
    current.static_mask[..., :4] = True
    current.static_mask[..., -4:] = True
    renderer = _SyntheticSharedDepthRenderer(local_depth=2.0, global_depth=10.0)
    backend = _pointmap_chunk_backend(
        min_points=48,
        min_points_per_frame=24,
        max_points_per_frame=64,
        renderer=renderer,
        alignment_mode="two_frame_global_map_full_sim3",
    )

    geometry = backend._collect_global_map_overlap_frame_geometry(
        previous, current, torch.eye(4), 2
    )
    assert bool((geometry.uv[:, 0] < 4.5).any())
    assert bool((geometry.uv[:, 0] > width - 4.5).any())
    transform, _, map_success, diagnostics = backend._global_map_overlap_alignment(
        previous, current, torch.eye(4)
    )

    assert map_success is True, diagnostics
    assert transform is not None
    assert diagnostics["quality_gating_enabled"] is False
    assert diagnostics["reason"] == "accepted_finite_with_sufficient_support"
    assert float(sim3_components(transform)[0]) == pytest.approx(5.0, rel=2.0e-4)


def test_global_map_and_pointmap_failure_rolls_back_transaction() -> None:
    height, width = 16, 32
    poses = torch.eye(4).repeat(4, 1, 1)
    previous = _packet(0, poses, (0, 1, 2, 3), height=height, width=width)
    current = _packet(1, poses, (2, 3, 4, 5), height=height, width=width)
    _attach_identity_stride_matches(previous)
    current.sky_prob.fill_(1.0)
    current.sky_mask.fill_(True)
    renderer = _SyntheticSharedDepthRenderer(local_depth=2.0, global_depth=2.0)
    backend = _pointmap_chunk_backend(
        renderer=renderer,
        alignment_mode="two_frame_global_map_full_sim3",
    )
    backend.process_packet(previous)
    owners_before = dict(backend.frame_pose_owner_node)

    with pytest.raises(
        RuntimeError, match=r"global-map/point-map Sim\(3\) alignment failed"
    ):
        backend.process_packet(current)

    assert set(backend.graph.nodes) == {0}
    assert backend.window_order == [0]
    assert backend.frame_pose_owner_node == owners_before
    assert backend._last_overlap_alignment_failure is not None
    assert backend._last_overlap_alignment_failure["accepted"] is False
