from pathlib import Path

import torch

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper, PanoRenderCamera
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


class _CountingRenderer:
    def __init__(self) -> None:
        self.calls = 0
        self.frame_ids: list[int] = []

    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        self.calls += 1
        fid = int(round(float(camera.c2w.detach().cpu()[0, 3]) * 10.0))
        self.frame_ids.append(fid)
        H, W = int(camera.image_height), int(camera.image_width)
        if gaussian_map.get_features.numel() == 0:
            color = torch.zeros(3, device=camera.c2w.device, dtype=camera.c2w.dtype)
        else:
            color = gaussian_map.get_features.mean(dim=0).to(device=camera.c2w.device, dtype=camera.c2w.dtype)
        render = color.view(3, 1, 1).expand(3, H, W)
        return {"render": render, "depth": render.new_ones(1, H, W)}


def _small_frontend_output(frame_id: int) -> FrontendOutput:
    pose = torch.eye(4)
    pose[0, 3] = float(frame_id) * 0.1
    return FrontendOutput(
        frame_id=frame_id,
        timestamp=float(frame_id),
        pose_c2w=pose,
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


def _small_seed_batch(frame_id: int) -> GaussianSeedBatch:
    return GaussianSeedBatch(
        xyz=torch.tensor([[0.05 * frame_id, 0.0, 1.0]], dtype=torch.float32),
        rgb=torch.tensor([[0.2 + 0.1 * frame_id, 0.4, 0.7]], dtype=torch.float32),
        confidence=torch.ones(1),
        scale=torch.full((1,), 0.1),
        level=torch.zeros(1, dtype=torch.long),
        frame_id=frame_id,
    )


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


def test_mapper_random_window_optimizes_one_sample_per_step():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 5,
            "window_keyframes": 3,
            "random_window_frame_per_iter": True,
            "sample_keyframes_per_step": 1,
            "pose_window_keyframes": 3,
            "fixed_window_frames": 1,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in range(3):
        image = torch.full((3, 4, 8), 0.25 + 0.1 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)

    metrics = mapper.optimize_after_keyframe()

    assert renderer.calls == 5
    assert metrics["window_size"] == 3.0
    assert metrics["sampled_window_size"] == 1.0
    assert metrics["trainable_pose_count"] == 2.0
    assert mapper.stats.last_phase == "sliding_window"
    assert mapper.stats.last_window_size == 3
    assert len(mapper.stats.last_sampled_keyframes) == 1
    assert mapper.stats.last_trainable_pose_count == 2


def test_skybox_renders_without_anchors():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {"enabled": True, "resolution": 8, "optimize": True},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    image = torch.zeros(3, 8, 16)
    image[1] = 0.35
    image[2] = 1.0
    assert gaussian_map.initialize_skybox_from_image(image, torch.eye(4))

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))
    pkg = renderer.render(camera, gaussian_map)

    assert pkg["render"].shape == image.shape
    assert float(pkg["render"][2].detach().mean()) > 0.5
    assert float(pkg["sky_bg_alpha"].detach().mean()) > 0.9


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
    assert summary["keyframe_decisions_path"] is None


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


def test_system_saves_final_artifacts_and_skybox(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 10, "width": 20},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 8,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
            "BootstrapOptimization": {"enabled": True, "first_keyframe_steps": 1, "save_every": 1},
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "keyframe_steps": 0,
            "non_keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 0,
            "final_global_steps": 0,
            "optimize_skybox": True,
        },
        "SkyBox": {"enabled": True, "resolution": 8, "optimize": True},
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": True, "save_kf_opt": True},
        "Results": {
            "save_dir": str(tmp_path),
            "kf_render_format": "png",
            "save_final_ply": True,
            "save_final_checkpoint": True,
            "save_final_keyframe_renders": True,
            "save_skybox_previews": True,
            "skybox_preview_height": 16,
            "skybox_preview_width": 32,
        },
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=2)

    assert summary["artifacts"]["final_ply"]
    assert Path(summary["artifacts"]["final_ply"]).is_file()
    assert Path(summary["artifacts"]["final_checkpoint"]).is_file()
    assert summary["artifacts"]["final_keyframe_render_count"] >= 1
    assert Path(summary["artifacts"]["final_skybox_erp_preview"]).is_file()
    assert Path(summary["artifacts"]["final_skybox_faces"]).is_file()
    assert any((tmp_path / "init_vis").rglob("iter_*_render.png"))
