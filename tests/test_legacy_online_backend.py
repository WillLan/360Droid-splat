from pathlib import Path

import numpy as np
import torch

from backend.legacy_360gs.config import build_legacy_config
from backend.legacy_360gs.utils.erp2cubemap import ERPToCubemapTorch
from backend.legacy_360gs.online import LegacyOnlineBackendClient
from backend.legacy_360gs.viewpoint_adapter import LegacyViewpointAdapter
from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from frontend.pano_vggt import PanoVGGTLongTracker
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


def _frontend_output(frame_id: int, pose: torch.Tensor | None = None) -> FrontendOutput:
    return FrontendOutput(
        frame_id=frame_id,
        timestamp=float(frame_id),
        pose_c2w=torch.eye(4) if pose is None else pose,
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, 8, 16),
        depth_confidence=torch.ones(1, 8, 16),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=None,
        tracking_status="tracked_test",
    )


def test_legacy_viewpoint_adapter_pose_depth_and_sky_mask():
    cfg = {
        "Mapping": {
            "min_depth_confidence": 0.5,
            "sky_mask_enable": True,
            "sky_mask_top_ratio": 0.5,
            "sky_mask_min_blue": 0.4,
        }
    }
    image = torch.zeros(3, 8, 16)
    image[2, :4] = 1.0
    image[:, 4:] = 0.25
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    frame = PanoFrame(image=image, timestamp=0.0, frame_id=0, meta={})
    bundle = LegacyViewpointAdapter(cfg, use_legacy_camera=False).build(frame, _frontend_output(0, pose))

    assert bundle.depth_map.shape == (8, 16)
    assert bundle.depth_map[:4].sum() == 0.0
    assert bundle.valid_mask[4:].all()
    assert torch.allclose(bundle.pose_w2c, torch.linalg.inv(pose))
    assert torch.allclose(bundle.viewpoint.R, bundle.pose_w2c[:3, :3])
    assert torch.allclose(bundle.viewpoint.T, bundle.pose_w2c[:3, 3])


def test_legacy_viewpoint_adapter_real_camera_uses_numpy_mono_depth():
    cfg = {"LegacyOnlineBackend": {"face_w": 8}, "Mapping": {"sky_mask_enable": False}}
    frame = PanoFrame(image=torch.rand(3, 8, 16), timestamp=0.0, frame_id=0, meta={})
    bundle = LegacyViewpointAdapter(cfg, use_legacy_camera=True).build(frame, _frontend_output(0))
    assert isinstance(bundle.viewpoint.depth, torch.Tensor)
    assert isinstance(bundle.viewpoint.mono_depth, np.ndarray)
    assert bundle.viewpoint.mono_depth.shape == (8, 16)


def test_erp_to_cubemap_torch_builds_valid_cosmap():
    erp2cube = ERPToCubemapTorch(face_w=8)
    assert erp2cube.cosmap.shape == (6, 8, 8)
    assert torch.isfinite(erp2cube.cosmap).all()
    assert float(erp2cube.cosmap.min()) > 0.0

    faces = erp2cube(torch.rand(3, 16, 32))
    assert faces.shape == (6, 3, 8, 8)


def test_legacy_config_includes_rgb_boundary_threshold_default():
    cfg = build_legacy_config({})
    assert cfg["Training"]["rgb_boundary_threshold"] == 0.01


def test_legacy_fake_backend_queue_roundtrip(tmp_path: Path):
    cfg = {
        "Runtime": {"multiprocessing_start_method": "spawn"},
        "LegacyOnlineBackend": {"backend_impl": "fake"},
        "Training": {"window_size": 2},
    }
    adapter = LegacyViewpointAdapter(cfg, use_legacy_camera=False)
    frame = PanoFrame(image=torch.rand(3, 8, 16), timestamp=0.0, frame_id=0, meta={})
    bundle = adapter.build(frame, _frontend_output(0))
    client = LegacyOnlineBackendClient(cfg, save_dir=tmp_path)
    client.start()
    client.submit_init(frame_id=0, viewpoint=bundle.viewpoint, depth_map=bundle.depth_map)
    snapshots = client.stop(join_timeout_s=10.0)
    assert snapshots
    assert snapshots[-1].poses_c2w
    assert snapshots[-1].anchor_count > 0


def test_legacy_fake_backend_window_roundtrip(tmp_path: Path):
    cfg = {
        "Runtime": {"multiprocessing_start_method": "spawn"},
        "LegacyOnlineBackend": {"backend_impl": "fake"},
        "Training": {"window_size": 3},
    }
    adapter = LegacyViewpointAdapter(cfg, use_legacy_camera=False)
    client = LegacyOnlineBackendClient(cfg, save_dir=tmp_path)
    bundles = []
    for frame_id in range(3):
        frame = PanoFrame(image=torch.rand(3, 8, 16), timestamp=float(frame_id), frame_id=frame_id, meta={})
        bundles.append(adapter.build(frame, _frontend_output(frame_id)))

    client.start()
    client.submit_init(
        frame_id=bundles[0].frame_id,
        viewpoint=bundles[0].viewpoint,
        depth_map=bundles[0].depth_map,
    )
    init_snapshots = client.wait_for_frame(0, timeout_s=10.0)
    target = client.submit_window(
        [(bundle.frame_id, bundle.viewpoint, bundle.depth_map) for bundle in bundles[1:]]
    )
    window_snapshots = client.wait_for_frame(target, timeout_s=10.0)
    snapshots = init_snapshots + window_snapshots + client.stop(join_timeout_s=10.0)
    poses = {}
    for snapshot in snapshots:
        poses.update(snapshot.poses_c2w)
    assert target == 2
    assert sorted(poses) == [0, 1, 2]
    assert snapshots[-1].anchor_count > 0


def test_panovggt_backend_pose_feedback_updates_pose_cache():
    tracker = PanoVGGTLongTracker(
        engine_config={"engine": "fake", "image_size": [8, 16]},
        chunk_size=2,
        overlap=1,
        emit_delay=0,
        device="cpu",
    )
    refined = torch.eye(4)
    refined[0, 3] = 2.0
    tracker.apply_backend_pose_updates({0: refined})
    assert torch.allclose(tracker.pose_by_frame[0].cpu(), refined)


def test_legacy_online_system_fake_smoke(tmp_path: Path):
    cfg = {
        "Runtime": {"mode": "legacy_online", "multiprocessing_start_method": "spawn"},
        "Dataset": {"synthetic": True, "synthetic_length": 4, "height": 8, "width": 16},
        "LegacyOnlineBackend": {
            "backend_impl": "fake",
            "feedback_enable": True,
            "join_timeout_s": 10,
        },
        "Frontend": {
            "mode": "panovggt_long",
            "keyframe_threshold": 0.0,
            "force_keyframe_interval": 1,
        },
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [8, 16],
            "chunk_size": 2,
            "overlap": 1,
            "emit_delay": 0,
            "align_mode": "sim3",
            "min_overlap_points": 16,
        },
        "Mapping": {"seed_source": "world_points_only", "min_depth_confidence": 0.0, "sky_mask_enable": False},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": True, "log_every": 1},
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=5)
    assert summary["runtime_mode"] == "legacy_online"
    assert summary["frames"] == 5
    assert summary["keyframes"] > 0
    assert summary["anchors"] > 0
    assert summary["backend_last_tag"] in {"init", "keyframe"}
    assert any((tmp_path / "visualizations").glob("*_backend_trajectory_vs_gt.png"))
