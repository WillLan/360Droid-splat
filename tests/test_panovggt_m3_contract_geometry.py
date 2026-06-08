import math
from pathlib import Path

import torch
import yaml

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, seam_aware_delta
from frontend.pano_vggt.engine import normalize_panovggt_output
from frontend.pano_vggt.grid_utils import (
    feature_uv_to_image_uv,
    image_uv_to_feature_uv,
    make_feature_grid,
)
from frontend.pano_vggt.m3_config import parse_m3_sphere_config
from frontend.pano_vggt.spherical_correspondence import (
    generate_gt_spherical_correspondences,
    spherical_tangent_residual,
)
from frontend.pano_vggt.types import PanoVGGTLocalPrediction


def _eye_poses(n: int) -> torch.Tensor:
    return torch.eye(4).view(1, 4, 4).repeat(n, 1, 1)


def _yaw_c2w(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.float32,
    )
    return pose


def test_old_prediction_construction_still_defaults_m3_fields():
    pred = PanoVGGTLocalPrediction(
        poses_c2w=_eye_poses(1),
        depth=torch.ones(1, 1, 4, 8),
        confidence=torch.ones(1, 1, 4, 8),
        chunk_world_points=torch.zeros(1, 4, 8, 3),
    )
    assert pred.dense_descriptors is None
    assert pred.match_confidence is None
    assert pred.static_confidence is None
    assert pred.feature_hw is None
    assert pred.image_hw is None
    assert pred.descriptor_dim == 24


def test_m3_config_parser_defaults_and_explicit_values():
    default_cfg = parse_m3_sphere_config({})
    assert default_cfg.enabled is False
    assert default_cfg.matching_head.enabled is False
    assert default_cfg.descriptor_dim == 24
    assert default_cfg.matching_head.descriptor_dim == 24
    assert default_cfg.dense_ba.residual_mode == "tangent"
    assert default_cfg.dense_ba.mode == "local_chunk"
    assert default_cfg.dense_ba.history_keyframes == 8
    assert default_cfg.keyframe_anchor.enabled is False

    cfg = parse_m3_sphere_config(
        {
            "PanoVGGT": {
                "M3Sphere": {"enabled": True},
                "MatchingHead": {
                    "enabled": True,
                    "checkpoint": "head.pt",
                    "descriptor_dim": 32,
                    "feature_hook": "aggregator",
                },
                "KeyframeAnchor": {"enabled": True, "cell_pair_conf_threshold": 0.2},
                "DenseMatching": {"enabled": True, "search_radius": 7, "topk": 2},
                "DenseBA": {
                    "enabled": True,
                    "iters": 4,
                    "mode": "history_window",
                    "history_keyframes": 3,
                    "residual_mode": "tangent",
                },
                "InferenceWindow": {"size": 5, "overlap": 1, "temporal_radius": 3},
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.matching_head.enabled is True
    assert cfg.matching_head.checkpoint == "head.pt"
    assert cfg.matching_head.descriptor_dim == 32
    assert cfg.matching_head.feature_hook == "aggregator"
    assert cfg.dense_matching.search_radius == 7
    assert cfg.dense_matching.topk == 2
    assert cfg.dense_ba.enabled is True
    assert cfg.dense_ba.iters == 4
    assert cfg.dense_ba.mode == "history_window"
    assert cfg.dense_ba.history_keyframes == 3
    assert cfg.keyframe_anchor.enabled is True
    assert cfg.keyframe_anchor.cell_pair_conf_threshold == 0.2
    assert cfg.inference_window.size == 5

    file_cfg = yaml.safe_load(Path("configs/pano_vggt_m3_sphere_gs_slam.yaml").read_text())
    parsed_file = parse_m3_sphere_config(file_cfg)
    assert parsed_file.enabled is True
    assert parsed_file.matching_head.enabled is True
    assert parsed_file.dense_matching.enabled is True
    assert parsed_file.dense_ba.enabled is True
    assert parsed_file.dense_ba.shadow_mode is True
    assert parsed_file.keyframe_anchor.enabled is True
    assert parsed_file.matching_head.descriptor_dim == 24
    assert file_cfg["PanoVGGT"]["image_size"] is None

    shadow_cfg = yaml.safe_load(Path("configs/pano_vggt_m3_sphere_360uav_shadow.yaml").read_text())
    active_cfg = yaml.safe_load(Path("configs/pano_vggt_m3_sphere_360uav_active.yaml").read_text())
    parsed_shadow = parse_m3_sphere_config(shadow_cfg)
    parsed_active = parse_m3_sphere_config(active_cfg)
    assert shadow_cfg["Dataset"]["dataset_path"].endswith("/360uav/seqs/seq1")
    assert shadow_cfg["Dataset"]["sequence"] is None
    assert shadow_cfg["PanoVGGT"]["image_size"] == [518, 1036]
    assert shadow_cfg["PanoVGGT"]["skip_dinov2_pretrain"] is True
    assert parsed_shadow.matching_head.matching_checkpoint.endswith("/matching_head.pt")
    assert parsed_shadow.matching_head.sky_checkpoint.endswith("/sky_head.pt")
    assert parsed_shadow.matching_head.feature_hook == "aggregator"
    assert parsed_shadow.dense_matching.max_factors == 4096
    assert parsed_shadow.dense_matching.max_samples_per_edge == 512
    assert parsed_shadow.dense_ba.shadow_mode is True
    assert parsed_shadow.dense_ba.iters == 3
    assert parsed_shadow.dense_ba.min_num_factors == 128
    assert parsed_shadow.dense_ba.factor_chunk_size == 512
    assert parsed_shadow.keyframe_anchor.enabled is True
    assert shadow_cfg["Visualization"]["m3_log_every"] == 5
    assert shadow_cfg["Visualization"]["m3_max_matches"] == 80
    assert parsed_active.dense_ba.shadow_mode is False
    assert active_cfg["Results"]["save_dir"].endswith("_active_seq1")


def test_feature_image_uv_roundtrip_equal_and_unequal_grids():
    for feature_hw, image_hw in [((4, 8), (4, 8)), ((5, 7), (13, 29)), ((3, 11), (17, 19))]:
        uv = make_feature_grid(feature_hw).view(-1, 2)
        image_uv = feature_uv_to_image_uv(uv, feature_hw, image_hw)
        restored = image_uv_to_feature_uv(image_uv, feature_hw, image_hw)
        assert torch.allclose(restored, uv, atol=1e-6)


def test_grid_utils_do_not_assume_fixed_resolution():
    feature_hw = (6, 10)
    image_hw = (37, 53)
    uv = torch.tensor([[0.5, 0.5], [9.5, 5.5], [3.25, 2.75]])
    image_uv = feature_uv_to_image_uv(uv, feature_hw, image_hw)
    assert image_uv.shape == uv.shape
    assert not torch.allclose(image_uv, uv)
    assert torch.allclose(image_uv_to_feature_uv(image_uv, feature_hw, image_hw), uv, atol=1e-6)


def test_engine_missing_and_present_matching_fields_are_optional():
    images = torch.rand(2, 3, 8, 16)
    base_output = {
        "camera_poses": _eye_poses(2).unsqueeze(0),
        "depth": torch.ones(1, 2, 8, 16),
    }
    pred = normalize_panovggt_output(base_output, images)
    assert pred.dense_descriptors is None
    assert pred.match_confidence is None
    assert pred.static_confidence is None
    assert pred.image_hw == (8, 16)
    assert pred.descriptor_dim == 24

    rich_output = {
        **base_output,
        "dense_descriptors": torch.rand(1, 2, 24, 3, 5),
        "match_confidence": torch.rand(1, 2, 1, 3, 5),
        "static_confidence": torch.rand(2, 3, 5),
        "feature_hw": [3, 5],
        "image_hw": [8, 16],
        "descriptor_dim": 24,
    }
    pred_rich = normalize_panovggt_output(rich_output, images)
    assert pred_rich.dense_descriptors.shape == (2, 24, 3, 5)
    assert pred_rich.match_confidence.shape == (2, 1, 3, 5)
    assert pred_rich.static_confidence.shape == (2, 1, 3, 5)
    assert pred_rich.feature_hw == (3, 5)
    assert pred_rich.image_hw == (8, 16)
    assert pred_rich.descriptor_dim == 24


def test_identity_constant_depth_correspondence_maps_to_itself():
    depths = torch.full((2, 1, 8, 16), 2.0)
    corr = generate_gt_spherical_correspondences(
        depths,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        feature_hw=(4, 8),
        image_hw=(8, 16),
        min_baseline_deg=0.0,
    )
    assert corr.valid_mask.all()
    assert torch.allclose(corr.src_uv, corr.tgt_uv, atol=1e-4)
    assert torch.allclose(corr.src_bearing, corr.tgt_bearing, atol=1e-5)


def test_yaw_seam_correspondence_wraps_horizontally():
    depths = torch.full((2, 1, 8, 16), 2.0)
    poses = _eye_poses(2)
    poses[1] = _yaw_c2w(math.radians(120.0))
    corr = generate_gt_spherical_correspondences(
        depths,
        poses,
        torch.tensor([[0, 1]]),
        feature_hw=(8, 16),
        image_hw=(8, 16),
        min_baseline_deg=0.0,
        max_baseline_deg=180.0,
        use_wraparound=True,
    )
    assert corr.valid_mask.any()
    assert torch.all((corr.tgt_uv[..., 0] >= 0.0) & (corr.tgt_uv[..., 0] < 16.0))
    assert torch.all((corr.tgt_uv[..., 1] >= 0.0) & (corr.tgt_uv[..., 1] < 8.0))


def test_angular_baseline_is_world_parallax_not_camera_frame_delta():
    depths = torch.full((2, 1, 8, 16), 2.0)
    poses = _eye_poses(2)
    poses[1] = _yaw_c2w(math.radians(120.0))
    corr = generate_gt_spherical_correspondences(
        depths,
        poses,
        torch.tensor([[0, 1]]),
        feature_hw=(4, 8),
        image_hw=(8, 16),
        min_baseline_deg=1.0,
        max_baseline_deg=180.0,
        use_wraparound=True,
    )
    assert corr.angular_baseline.abs().max() < 1.0e-4
    assert not corr.valid_mask.any()


def test_depth_inconsistent_correspondence_is_invalid():
    depths = torch.full((2, 1, 8, 16), 2.0)
    depths[1] = 9.0
    corr = generate_gt_spherical_correspondences(
        depths,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        feature_hw=(4, 8),
        image_hw=(8, 16),
        min_baseline_deg=0.0,
    )
    assert corr.depth_consistency.logical_not().any()
    assert not corr.valid_mask.any()


def test_spherical_tangent_residual_identical_bearing_is_near_zero():
    bearing = erp_pixel_to_bearing(torch.tensor([[4.5, 3.5]]), 8, 16)
    residual = spherical_tangent_residual(bearing, bearing)
    assert residual.shape == (1, 2)
    assert residual.abs().max() < 1e-6


def test_spherical_tangent_residual_small_angle_norm_matches_angle():
    theta = 1.0e-3
    target = torch.tensor([[0.0, 0.0, 1.0]])
    predicted = torch.tensor([[math.sin(theta), 0.0, math.cos(theta)]])
    residual = spherical_tangent_residual(target, predicted)
    assert torch.allclose(residual.norm(dim=-1), torch.tensor([theta]), atol=1e-5, rtol=1e-3)


def test_tangent_residual_is_not_erp_pixel_delta():
    height, width = 8, 16
    target_pixel = torch.tensor([[15.5, 4.5]])
    predicted_pixel = torch.tensor([[0.5, 4.5]])
    target_bearing = erp_pixel_to_bearing(target_pixel, height, width)
    predicted_bearing = erp_pixel_to_bearing(predicted_pixel, height, width)
    tangent_norm = spherical_tangent_residual(target_bearing, predicted_bearing).norm(dim=-1)
    pixel_delta_norm = seam_aware_delta(target_pixel, predicted_pixel, width).norm(dim=-1)
    assert not torch.allclose(tangent_norm, pixel_delta_norm, atol=1e-3)
