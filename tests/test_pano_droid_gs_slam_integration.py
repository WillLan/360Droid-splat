from pathlib import Path

import torch

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


def test_mapper_renders_keyframe_diagnostic():
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(
        gaussian_map,
        renderer=PFGS360Renderer(config=config, allow_fallback=True),
    )
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.2]], dtype=torch.float32),
        rgb=torch.tensor([[1.0, 0.2, 0.1], [0.1, 0.7, 1.0]], dtype=torch.float32),
        confidence=torch.ones(2),
        scale=torch.full((2,), 0.1),
        level=torch.zeros(2, dtype=torch.long),
        frame_id=7,
    )
    output = FrontendOutput(
        frame_id=7,
        timestamp=7.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=None,
        tracking_status="tracked",
    )
    image = torch.rand(3, 4, 8)

    mapper.insert_keyframe(seeds, output, image=image)
    diagnostic = mapper.render_keyframe_diagnostic(7)

    assert diagnostic is not None
    assert diagnostic.frame_id == 7
    assert diagnostic.target.shape == image.shape
    assert diagnostic.render.shape == image.shape
    assert diagnostic.depth is not None
    assert diagnostic.anchor_count == 2


def test_system_runs_synthetic_smoke(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 3, "height": 16, "width": 32},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
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
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=3)
    assert summary["frames"] == 3
    assert summary["keyframes"] >= 1
    assert summary["anchors"] > 0
    assert (tmp_path / "summary.json").is_file()


def test_system_saves_keyframe_optimized_render_and_depth(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 3, "height": 12, "width": 24},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 12,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "local_submap_steps": 1,
            "local_window_keyframes": 2,
            "sliding_window_steps": 0,
            "final_global_steps": 0,
            "fixed_window_frames": 1,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": True, "save_kf_opt": True, "kf_opt_log_every": 1},
        "Results": {"save_dir": str(tmp_path), "kf_render_format": "png"},
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=3)

    assert summary["anchors"] > 0
    assert summary["backend_optimization_steps"] > 0
    assert any((tmp_path / "kf_renders_opt").glob("kf_*.png"))
    assert any((tmp_path / "kf_depths_opt").glob("kf_*.png"))
