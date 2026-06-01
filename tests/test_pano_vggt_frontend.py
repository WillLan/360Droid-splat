from pathlib import Path

import torch

from frontend.pano_droid.interfaces import PanoFrame
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from frontend.pano_vggt import FakePanoVGGTInferenceEngine, PanoVGGTLongTracker, SubmapAligner
from frontend.pano_vggt.alignment import sample_overlap_points
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


def test_fake_panovggt_engine_outputs_local_geometry():
    engine = FakePanoVGGTInferenceEngine(image_size=(16, 32), translation_step=0.1)
    pred = engine.infer(torch.rand(3, 3, 8, 16))
    assert pred.poses_c2w.shape == (3, 4, 4)
    assert pred.depth.shape == (3, 1, 16, 32)
    assert pred.confidence.shape == (3, 1, 16, 32)
    assert pred.point_maps.shape == (3, 16, 32, 3)
    assert torch.isfinite(pred.point_maps).all()
    assert pred.poses_c2w[2, 0, 3] > pred.poses_c2w[1, 0, 3]


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
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=5)
    assert summary["frames"] == 5
    assert summary["keyframes"] >= 1
    assert summary["anchors"] > 0
    assert (tmp_path / "summary.json").is_file()
