from pathlib import Path
import json

from PIL import Image
import pytest
import torch
import yaml

from frontend.pano_droid.dataset import discover_erp_images
from frontend.pano_droid.interfaces import PanoFrame
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from frontend.pano_vggt import (
    FakePanoVGGTInferenceEngine,
    PanoVGGTAlignmentError,
    PanoVGGTLocalPrediction,
    PanoVGGTLongTracker,
    SubmapAligner,
    build_panovggt_frontend_from_config,
)
from frontend.pano_vggt.alignment import sample_overlap_points
from frontend.pano_vggt.engine import _ceil_size_to_multiple, _resize_prediction, normalize_panovggt_output
from scripts.run_pano_vggt_panocity_blocks import build_block_config, discover_panocity_blocks
from system.pano_droid_gs_slam import (
    PanoDroidGSSlamSystem,
    _chronological_previous_frame_ids,
    _relative_pose_from_chronological_previous,
    iter_sequence_frames,
    load_config,
)


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4), color=(32, 64, 128)).save(path)


def test_fake_panovggt_engine_outputs_local_geometry():
    engine = FakePanoVGGTInferenceEngine(image_size=(16, 32), translation_step=0.1)
    pred = engine.infer(torch.rand(3, 3, 8, 16))
    assert pred.poses_c2w.shape == (3, 4, 4)
    assert pred.depth.shape == (3, 1, 16, 32)
    assert pred.confidence.shape == (3, 1, 16, 32)
    assert pred.point_maps.shape == (3, 16, 32, 3)
    assert torch.isfinite(pred.point_maps).all()
    assert pred.poses_c2w[2, 0, 3] > pred.poses_c2w[1, 0, 3]


def test_normalize_panovggt_output_accepts_official_shapes():
    images = torch.rand(2, 3, 8, 16)
    output = {
        "camera_poses": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "depth": torch.ones(1, 2, 8, 16),
        "world_points": torch.zeros(1, 2, 8, 16, 3),
        "local_points": torch.ones(1, 2, 8, 16, 3),
    }
    pred = normalize_panovggt_output(output, images)
    assert pred.poses_c2w.shape == (2, 4, 4)
    assert pred.depth.shape == (2, 1, 8, 16)
    assert pred.point_maps.shape == (2, 8, 16, 3)


def test_normalize_panovggt_output_inverts_extrinsics_and_transforms_local_points():
    images = torch.rand(1, 3, 4, 8)
    w2c = torch.eye(4).view(1, 1, 4, 4)
    w2c[0, 0, 0, 3] = -2.0
    local = torch.zeros(1, 1, 4, 8, 3)
    local[..., 2] = 1.0
    pred = normalize_panovggt_output(
        {
            "extrinsics": w2c,
            "depth": torch.ones(1, 1, 4, 8),
            "local_points": local,
        },
        images,
    )
    assert torch.allclose(pred.poses_c2w[0, :3, 3], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(pred.chunk_world_points[0, 0, 0], torch.tensor([2.0, 0.0, 1.0]))


def test_panovggt_external_patch_multiple_resize_helpers():
    assert _ceil_size_to_multiple((512, 1024), 14) == (518, 1036)
    pred = normalize_panovggt_output(
        {
            "camera_poses": torch.eye(4).view(1, 1, 4, 4),
            "depth": torch.ones(1, 1, 518, 1036),
        },
        torch.rand(1, 3, 518, 1036),
    )
    resized = _resize_prediction(pred, (512, 1024))
    assert resized.depth.shape == (1, 1, 512, 1024)
    assert resized.confidence.shape == (1, 1, 512, 1024)
    assert resized.point_maps.shape == (1, 512, 1024, 3)


def test_panovggt_configs_use_patch_multiple_erp_size():
    for config_path in (
        Path("configs/pano_vggt_long_gs_slam.yaml"),
        Path("configs/pano_vggt_long_panocity_beijing.yaml"),
        Path("configs/pano_vggt_legacy_online_panocity_beijing.yaml"),
    ):
        cfg = yaml.safe_load(config_path.read_text()) or {}
        height = int(cfg["Dataset"]["erp_resize_height"])
        width = int(cfg["Dataset"]["erp_resize_width"])
        image_size = tuple(int(v) for v in cfg["PanoVGGT"]["image_size"])
        patch_multiple = int(cfg["PanoVGGT"].get("patch_multiple", 14))

        assert (height, width) == image_size
        assert height % patch_multiple == 0
        assert width % patch_multiple == 0
        assert (height, width) == (518, 1036)


def test_panovggt_erp_bearing_convention_matches_project_camera():
    height, width = 8, 16
    grid = pixel_grid(height, width)
    bearing = erp_pixel_to_bearing(grid, height, width)
    center = bearing[height // 2, width // 2]
    assert center[2] > 0.9
    assert abs(float(center[0])) < 0.25


def test_submap_aligner_recovers_similarity_transform():
    torch.manual_seed(1)
    source = torch.randn(256, 3)
    theta = 0.35
    c = float(torch.cos(torch.tensor(theta)))
    s = float(torch.sin(torch.tensor(theta)))
    rot = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 1.4
    trans = torch.tensor([0.3, -0.2, 0.5])
    target = scale * torch.matmul(source, rot.T) + trans
    target[:8] += 5.0
    weights = torch.ones(source.shape[0])
    weights[:8] = 0.01

    transform = SubmapAligner(align_mode="sim3", max_residual=0.05).align(source, target, weights)
    aligned = transform.apply_points(source[8:])
    assert transform.accepted
    assert abs(transform.scale - scale) < 1e-3
    assert torch.mean(torch.linalg.norm(aligned - target[8:], dim=-1)) < 1e-3


def test_submap_aligner_accepts_borderline_residual_when_threshold_relaxed():
    xs, ys = torch.meshgrid(torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij")
    source = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(16)], dim=-1)
    z_offset = torch.where((xs.reshape(-1).long() + ys.reshape(-1).long()) % 2 == 0, 0.3, -0.3)
    target = source.clone()
    target[:, 2] = z_offset

    transform = SubmapAligner(
        align_mode="sim3",
        max_residual=0.35,
        min_inlier_ratio=0.35,
        min_points=4,
    ).align(source, target, torch.ones(16))

    assert transform.accepted
    assert 0.25 < transform.residual <= 0.35


def test_sample_overlap_points_keeps_high_confidence_matches():
    source = torch.zeros(1, 4, 4, 3)
    target = torch.ones(1, 4, 4, 3)
    conf = torch.zeros(1, 4, 4)
    conf[..., 0, 0] = 1.0
    src, tgt, weights = sample_overlap_points(source, target, conf, None, max_points=1)
    assert src.shape == (1, 3)
    assert tgt.shape == (1, 3)
    assert weights.item() == 1.0


def test_panovggt_tracker_emits_stable_monotonic_outputs():
    tracker = PanoVGGTLongTracker(
        engine=FakePanoVGGTInferenceEngine(image_size=(16, 32)),
        chunk_size=3,
        overlap=1,
        emit_delay=1,
        keyframe_threshold=0.0,
        force_keyframe_interval=1,
        min_overlap_points=16,
    )
    outputs = []
    for idx in range(5):
        tracker.track(PanoFrame(image=torch.rand(3, 8, 16), timestamp=float(idx), frame_id=idx))
        outputs.extend(tracker.pop_ready_outputs())
    outputs.extend(tracker.flush())
    ids = [out.frame_id for out in outputs]
    assert ids == sorted(ids)
    assert ids == list(range(5))
    assert all(out.inverse_depth is not None for out in outputs)
    assert all(out.world_points is not None for out in outputs)
    assert all(out.valid_world_points_mask is not None for out in outputs)
    assert all(out.tracking_status.startswith("tracked_panovggt_long") for out in outputs)


def test_panovggt_tracker_raises_on_alignment_failure():
    tracker = PanoVGGTLongTracker(
        engine=FakePanoVGGTInferenceEngine(image_size=(8, 16)),
        chunk_size=1,
        overlap=0,
        emit_delay=0,
        keyframe_threshold=0.0,
        force_keyframe_interval=1,
        min_overlap_points=16,
        force_accept_alignment=False,
    )
    tracker.track(PanoFrame(image=torch.rand(3, 8, 16), timestamp=0.0, frame_id=0))
    tracker.pop_ready_outputs()
    with pytest.raises(PanoVGGTAlignmentError):
        tracker.track(PanoFrame(image=torch.rand(3, 8, 16), timestamp=1.0, frame_id=1))


def test_panovggt_tracker_can_force_accept_rejected_alignment_transform():
    tracker = PanoVGGTLongTracker(
        engine=FakePanoVGGTInferenceEngine(image_size=(4, 4)),
        device="cpu",
        chunk_size=1,
        overlap=0,
        emit_delay=0,
        min_overlap_points=4,
        max_alignment_points=4,
        max_scale_jump=1.1,
        force_accept_alignment=True,
        require_aligned_world_points=True,
        emit_unaligned=False,
    )
    yy, xx = torch.meshgrid(torch.arange(2, dtype=torch.float32), torch.arange(2, dtype=torch.float32), indexing="ij")
    source = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
    target = 2.0 * source + torch.tensor([0.25, -0.5, 0.0])
    pred = PanoVGGTLocalPrediction(
        poses_c2w=torch.eye(4).unsqueeze(0),
        depth=torch.ones(1, 1, 2, 2),
        confidence=torch.ones(1, 1, 2, 2),
        chunk_world_points=source.unsqueeze(0),
    )
    tracker.global_points_by_frame[7] = target.clone()

    transform = tracker._align_chunk(pred, (7,))

    assert transform.accepted
    assert tracker.last_alignment_debug.forced_accepted
    assert transform.scale > 1.5
    assert tracker.pose_graph.edges[-1].edge_type == "sequential_forced"


def test_panovggt_builder_reads_force_accept_alignment_flag():
    tracker = build_panovggt_frontend_from_config(
        {
            "PanoVGGT": {
                "engine": "fake",
                "image_size": [4, 4],
                "force_accept_alignment": True,
            }
        }
    )

    assert tracker.force_accept_alignment is True


def test_panovggt_builder_defaults_force_accept_alignment_enabled():
    tracker = build_panovggt_frontend_from_config(
        {
            "PanoVGGT": {
                "engine": "fake",
                "image_size": [4, 4],
            }
        }
    )

    assert tracker.force_accept_alignment is True


def test_panocity_pano_images_are_discoverable(tmp_path: Path):
    block = tmp_path / "beijing_block_001"
    _write_rgb(block / "pano_images" / "000001.png")
    files = discover_erp_images(str(block))
    assert len(files) == 1
    assert files[0].endswith("000001.png")


def test_panocity_block_config_targets_first_frames_and_wandb_run(tmp_path: Path):
    root = tmp_path / "beijing"
    block = root / "beijing_block_001"
    _write_rgb(block / "pano_images" / "000001.png")
    blocks = discover_panocity_blocks(root)
    cfg = build_block_config(
        {
            "Dataset": {"dataset_path": str(root)},
            "Results": {"save_dir": str(tmp_path / "base_out")},
            "WeightsAndBiases": {"run_name": "base_run"},
        },
        blocks[0],
        output_root=tmp_path / "out",
        frames_per_block=300,
    )
    assert cfg["Dataset"]["dataset_path"] == str(block / "pano_images")
    assert cfg["Dataset"]["end"] == 300
    assert cfg["Results"]["save_dir"].endswith("beijing_block_001")
    assert cfg["WeightsAndBiases"]["run_name"] == "base_run_beijing_block_001"


def test_panocity_gt_pose_is_attached_to_runtime_frames(tmp_path: Path):
    block = tmp_path / "beijing_block_001"
    _write_rgb(block / "pano_images" / "000001.png")
    pose = torch.eye(4)
    pose[0, 3] = 1.25
    with open(block / "poses.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "frames": [
                    {
                        "name": "000001.png",
                        "depth": "000001.png",
                        "transformation_matrix": pose.tolist(),
                    }
                ]
            },
            f,
        )
    cfg = {"Dataset": {"dataset_path": str(block / "pano_images")}}
    frame = next(iter_sequence_frames(cfg))
    assert frame.meta is not None
    assert torch.allclose(frame.meta["gt_c2w"], pose)


def test_runtime_frame_stride_preserves_source_ids_and_gt(tmp_path: Path):
    image_dir = tmp_path / "sponza" / "Non-Egocentric" / "images"
    camera_dir = tmp_path / "sponza" / "Non-Egocentric" / "cameras"
    for frame_id in range(6):
        _write_rgb(image_dir / f"{frame_id:05d}_rgb.png")
        camera_dir.mkdir(parents=True, exist_ok=True)
        (camera_dir / f"{frame_id:05d}_cam.json").write_text(
            json.dumps(
                [
                    {
                        "extrinsics": {
                            "rotation": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "translation": [float(frame_id), 0.0, 0.0],
                        }
                    }
                ]
            ),
            encoding="utf-8",
        )

    common = {
        "synthetic": False,
        "type": "ob3d",
        "dataset_path": str(tmp_path),
        "scene": "sponza",
        "split": "Non-Egocentric",
        "begin": 1,
        "end": 6,
    }
    dense = list(iter_sequence_frames({"Dataset": common}))
    sparse = list(
        iter_sequence_frames(
            {"Dataset": {**common, "frame_stride": 2}}
        )
    )

    assert [frame.frame_id for frame in dense] == [1, 2, 3, 4, 5]
    assert [frame.frame_id for frame in sparse] == [1, 3, 5]
    assert [frame.meta["source_frame_index"] for frame in sparse] == [1, 3, 5]
    assert [float(frame.meta["gt_c2w"][0, 3]) for frame in sparse] == [
        -1.0,
        -3.0,
        -5.0,
    ]


def test_runtime_frame_stride_must_be_positive(tmp_path: Path):
    image_dir = tmp_path / "images"
    _write_rgb(image_dir / "00000.png")
    config = {
        "Dataset": {
            "dataset_path": str(image_dir),
            "frame_stride": 0,
        }
    }
    with pytest.raises(ValueError, match="frame_stride must be a positive"):
        list(iter_sequence_frames(config))


def test_sparse_frame_history_uses_chronological_predecessor() -> None:
    previous = _chronological_previous_frame_ids([4, 0, 2])
    assert previous == {2: 0, 4: 2}
    poses = {frame_id: torch.eye(4) for frame_id in (0, 2, 4)}
    poses[0][0, 3] = 0.0
    poses[2][0, 3] = 0.7
    poses[4][0, 3] = 2.0

    relative = _relative_pose_from_chronological_previous(
        poses,
        previous,
        4,
        None,
    )

    assert relative is not None
    torch.testing.assert_close(relative, torch.linalg.inv(poses[4]) @ poses[2])


def test_stride2_iter200_experiment_only_changes_sampling_and_run_identity() -> None:
    root = Path(__file__).parents[1] / "configs"
    baseline = load_config(
        root / "spherical_selfi_ob3d_photometric_recent3_iter200.yaml"
    )
    sampled = load_config(
        root
        / "spherical_selfi_ob3d_photometric_recent3_iter200_stride2_50.yaml"
    )

    assert sampled["Dataset"] == {
        **baseline["Dataset"],
        "begin": 0,
        "end": 100,
        "frame_stride": 2,
    }
    assert (
        sampled["SphericalSelfiGlobalBackend"]
        == baseline["SphericalSelfiGlobalBackend"]
    )
    assert sampled["SphericalSelfiRuntime"] == baseline["SphericalSelfiRuntime"]
    assert sampled["VoxelAnchorRefiner"] == baseline["VoxelAnchorRefiner"]
    assert sampled["Results"]["save_dir"].endswith(
        "ob3d50_stride2_photometric_recent3_iter200"
    )


def test_ob3d_pointmap_sim3_config_preserves_adapter_ba_refiner_baseline() -> None:
    root = Path(__file__).parents[1] / "configs"
    baseline = load_config(
        root / "spherical_selfi_ob3d_bridge_depthscale_refiner_ba8_100.yaml"
    )
    pointmap = load_config(
        root / "spherical_selfi_ob3d_pointmap_sim3_adapter_ba_100.yaml"
    )

    assert pointmap["Dataset"] == baseline["Dataset"]
    assert pointmap["SphericalSelfiRuntime"] == baseline["SphericalSelfiRuntime"]
    assert pointmap["VoxelAnchorRefiner"] == baseline["VoxelAnchorRefiner"]
    alignment = pointmap["SphericalSelfiGlobalBackend"][
        "rendered_overlap_alignment"
    ]
    assert alignment["mode"] == "two_frame_pointmap_full_sim3"
    assert alignment["acceptance_policy"] == "diagnostics_only"
    assert alignment["min_points"] == 2048
    assert alignment["min_points_per_frame"] == 512
    graph = pointmap["SphericalSelfiGlobalBackend"]["global_graph"]
    assert graph["node_mode"] == "chunk_first_stride"
    assert graph["fibonacci_oversample_factor"] == 8
    assert graph["skip_edge"]["enabled"] is False
    assert pointmap["WeightsAndBiases"]["runtime_log_preset"] == (
        "slam_core_visuals"
    )
    assert pointmap["Results"]["save_dir"].endswith(
        "ob3d100_pointmap_sim3_adapter_ba_r1"
    )

    prepost = load_config(
        root
        / "spherical_selfi_ob3d_pointmap_sim3_adapter_ba_100_prepost.yaml"
    )
    stages = load_config(
        root
        / "spherical_selfi_ob3d_pointmap_sim3_adapter_ba_100_stages.yaml"
    )
    pfgs360_freeze = load_config(
        root
        / "spherical_selfi_ob3d_pointmap_sim3_adapter_ba_100_pfgs360_freeze.yaml"
    )
    algorithm_sections = (
        "Dataset",
        "SphericalSelfiRuntime",
        "VoxelAnchorRefiner",
        "SphericalSelfiGlobalBackend",
        "Mapping",
    )
    for section in algorithm_sections:
        assert prepost.get(section) == pointmap.get(section)
    assert prepost["WeightsAndBiases"]["runtime_log_preset"] == (
        "slam_core_visuals"
    )
    assert prepost["WeightsAndBiases"]["run_name"].endswith("prepost_r2")
    assert prepost["Results"]["save_dir"].endswith("prepost_r2")
    for section in algorithm_sections:
        assert stages.get(section) == pointmap.get(section)
    assert stages["WeightsAndBiases"]["runtime_log_preset"] == (
        "slam_core_visuals"
    )
    assert stages["WeightsAndBiases"]["run_name"].endswith("stages_r3")
    assert stages["Results"]["save_dir"].endswith("stages_r3")
    for section in algorithm_sections:
        assert pfgs360_freeze.get(section) == pointmap.get(section)
    assert pfgs360_freeze["TrajectoryEvaluation"] == {"ate_mode": "both"}
    assert pfgs360_freeze["WeightsAndBiases"]["run_name"].endswith(
        "frozen_frontend_r4"
    )
    assert pfgs360_freeze["Results"]["save_dir"].endswith(
        "frozen_frontend_r4"
    )


def test_system_runs_panovggt_long_fake_smoke(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 4, "height": 16, "width": 32},
        "Frontend": {
            "mode": "panovggt_long",
            "keyframe_threshold": 0.0,
            "force_keyframe_interval": 1,
        },
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [16, 32],
            "chunk_size": 3,
            "overlap": 1,
            "emit_delay": 1,
            "align_mode": "sim3",
            "min_overlap_points": 16,
        },
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 20,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled", "log_keyframes": False, "log_every": 3},
        "Visualization": {"save_local": True, "log_every": 3},
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=5)
    assert summary["frames"] == 5
    assert summary["keyframes"] >= 1
    assert summary["anchors"] > 0
    assert (tmp_path / "summary.json").is_file()
    assert any((tmp_path / "visualizations").glob("*_depth.png"))
    assert any((tmp_path / "visualizations").glob("*_trajectory.png"))
    assert any((tmp_path / "visualizations").glob("*_frontend_trajectory_vs_gt.png"))
    assert any((tmp_path / "visualizations").glob("*_backend_trajectory_vs_gt.png"))
    assert any((tmp_path / "visualizations").glob("*_backend_render_vs_gt.png"))
    assert any((tmp_path / "visualizations").glob("*_backend_render_depth.png"))


def test_system_runs_joint_gaussian_pose_backend_smoke(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 3, "height": 12, "width": 24},
        "Frontend": {
            "mode": "panovggt_long",
            "keyframe_threshold": 0.0,
            "force_keyframe_interval": 1,
        },
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [12, 24],
            "chunk_size": 3,
            "overlap": 1,
            "emit_delay": 1,
            "align_mode": "sim3",
            "min_overlap_points": 16,
        },
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 8,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "local_submap_steps": 1,
            "local_window_keyframes": 2,
            "sliding_window_steps": 1,
            "window_keyframes": 3,
            "final_global_steps": 1,
            "optimize_existing_gaussians": "visible_recent",
            "existing_gaussian_lr_scale": 0.1,
            "pose_prior_weight": 0.001,
            "fixed_window_frames": 1,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": False},
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=4)
    assert summary["anchors"] > 0
    assert summary["backend_optimization_steps"] > 0
    assert summary["backend_last_phase"] == "final_global"
    assert summary["backend_final_metrics"]["steps"] == 1.0


def test_system_runs_feedforward_window_backend_smoke(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 4, "height": 12, "width": 24},
        "Frontend": {
            "mode": "panovggt_long",
            "keyframe_threshold": 2.0,
            "force_keyframe_interval": 2,
        },
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [12, 24],
            "chunk_size": 3,
            "overlap": 1,
            "emit_delay": 1,
            "align_mode": "sim3",
            "min_overlap_points": 16,
        },
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 8,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "keyframe_steps": 0,
            "non_keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 1,
            "final_global_steps": 0,
            "random_window_frame_per_iter": False,
            "optimize_skybox": False,
            "FeedForwardWindow": {
                "enabled": True,
                "history_keyframes": 2,
                "optimize_non_keyframe_observations": True,
                "gaussian_scope": "selected_birth_keyframes",
                "prune": {"enabled": False},
            },
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": False},
        "Results": {"save_dir": str(tmp_path)},
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=4)

    assert summary["anchors"] > 0
    assert summary["backend_last_phase"] == "feedforward_window"
    assert summary["backend_last_window_observations"]
    assert summary["backend_last_feedforward_metrics"]["feedforward_window_size"] >= 1.0
