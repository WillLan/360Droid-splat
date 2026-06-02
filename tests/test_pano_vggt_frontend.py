from pathlib import Path
import json

from PIL import Image
import torch

from frontend.pano_droid.dataset import discover_erp_images
from frontend.pano_droid.interfaces import PanoFrame
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from frontend.pano_vggt import FakePanoVGGTInferenceEngine, PanoVGGTLongTracker, SubmapAligner
from frontend.pano_vggt.alignment import sample_overlap_points
from frontend.pano_vggt.engine import normalize_panovggt_output
from scripts.run_pano_vggt_panocity_blocks import build_block_config, discover_panocity_blocks
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem, iter_sequence_frames


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
    assert all(out.tracking_status.startswith("tracked_panovggt_long") for out in outputs)


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
