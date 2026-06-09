from pathlib import Path

import torch

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper, PanoRenderCamera
from backend.pano_gs.losses import backend_render_loss
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem, _se3_blend_pose


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


class _DepthGateRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        render = torch.zeros(3, H, W, device=camera.c2w.device, dtype=camera.c2w.dtype)
        if W >= 3:
            render[:, :, 2] = 1.0
        depth = torch.ones(1, H, W, device=render.device, dtype=render.dtype)
        if W >= 2:
            depth[:, :, 1] = 2.0
        if W >= 3:
            depth[:, :, 2] = 0.6
        alpha = torch.ones(1, H, W, device=render.device, dtype=render.dtype)
        alpha[:, :, 0] = 0.0
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=render.device),
            "radii": torch.zeros(total, dtype=torch.int32, device=render.device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=render.device),
            "render_distort": None,
        }


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


def test_pfgs360_mapper_render_depth_gate_inserts_missing_foreground_and_inconsistent_regions():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360",
                "voxel_size": 0.1,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 20,
                "render_alpha_min": 0.2,
                "foreground_rel_threshold": 0.1,
                "render_depth_rel_threshold": 0.15,
                "photometric_error_threshold": 0.08,
                "near_grid_radius": 0,
                "reset_after_outlier_observations": 99,
                "prune_after_outlier_observations": 99,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_DepthGateRenderer())
    first = GaussianSeedBatch(
        xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        rgb=torch.zeros(1, 3),
        confidence=torch.ones(1),
        scale=torch.full((1,), 0.1),
        level=torch.zeros(1, dtype=torch.int8),
        frame_id=0,
    )
    assert mapper.insert_keyframe(first, _small_frontend_output(0), image=torch.zeros(3, 1, 4)) == 1
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]),
        rgb=torch.zeros(4, 3),
        confidence=torch.ones(4),
        scale=torch.full((4,), 0.1),
        level=torch.zeros(4, dtype=torch.int8),
        frame_id=1,
        source_flat_idx=torch.arange(4),
        source_hw=(1, 4),
        insert_enabled=torch.ones(4, dtype=torch.bool),
        insert_score=torch.tensor([1.0, 0.9, 0.8, 0.1]),
    )

    inserted = mapper.insert_keyframe(seeds, _small_frontend_output(1), image=torch.zeros(3, 1, 4))

    assert inserted == 3
    assert mapper.stats.last_suppressed_insert == 1
    assert torch.equal(mapper.last_inserted_source_flat_idx, torch.tensor([0, 1, 2]))


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


def test_mapper_frontend_graph_window_prioritizes_history_hints():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 1,
            "window_keyframes": 2,
            "use_frontend_graph_window": True,
            "random_window_frame_per_iter": False,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in (0, 4, 8, 12):
        image = torch.full((3, 4, 8), 0.2 + 0.01 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)
    mapper.set_frontend_graph_window_ids([4, 12])

    metrics = mapper.optimize_after_keyframe()

    assert metrics["window_size"] == 3.0
    assert mapper.stats.last_window_keyframes == [4, 8, 12]
    assert renderer.frame_ids == [4, 8, 12]


def test_pano_gaussian_map_saves_legacy_3dgs_ply_schema(tmp_path: Path):
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    gaussian_map.add_seeds(_small_seed_batch(0))
    ply_path = tmp_path / "point_cloud.ply"

    gaussian_map.save_ply(ply_path)

    data = ply_path.read_bytes()
    header = data.split(b"end_header")[0].decode("ascii") + "end_header"
    expected = [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 1",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        *[f"property float f_rest_{idx}" for idx in range(24)],
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    assert header.splitlines() == expected
    assert b"red" not in data.split(b"end_header")[0]
    assert ply_path.stat().st_size > len(header)


def test_pfgs360_renderer_converts_rgb_to_sh_dc_for_rasterizer():
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32),
        rgb=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]], dtype=torch.float32),
        confidence=torch.ones(2),
        scale=torch.full((2,), 0.1),
        level=torch.zeros(2, dtype=torch.long),
        frame_id=0,
    )
    gaussian_map.add_seeds(seeds)
    seen: dict[str, torch.Tensor | tuple[int, ...] | int] = {}

    def fake_rasterization(**kwargs):
        colors = kwargs["colors"]
        seen["colors_shape"] = tuple(colors.shape)
        seen["colors"] = colors.detach().clone()
        seen["sh_degree"] = int(kwargs["sh_degree"])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        device = colors.device
        dtype = colors.dtype
        render = torch.zeros(1, height, width, 4, device=device, dtype=dtype)
        rgb = colors[:, 0, :] * 0.28209479177387814 + 0.5
        render[0, :, :, :3] = rgb.mean(dim=0).view(1, 1, 3)
        render[0, :, :, 3] = 1.0
        alpha = torch.ones(1, height, width, 1, device=device, dtype=dtype)
        info = {
            "means2d": torch.zeros(1, colors.shape[0], 2, device=device, dtype=dtype),
            "radii": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
            "accum_times": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
        }
        return render, alpha, None, info

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(image_height=4, image_width=8, c2w=torch.eye(4))
    pkg = renderer._render_gsplat360(fake_rasterization, camera, gaussian_map, torch.zeros(3))

    expected_sh = ((gaussian_map.get_features.detach() - 0.5) / 0.28209479177387814).unsqueeze(1)
    assert seen["colors_shape"] == (2, 1, 3)
    assert torch.allclose(seen["colors"], expected_sh)
    assert seen["sh_degree"] == gaussian_map.active_sh_degree
    assert torch.allclose(pkg["render"], gaussian_map.get_features.mean(dim=0).view(3, 1, 1).expand(3, 4, 8))


def test_backend_feedback_se3_blend_and_hard_gate():
    source = torch.eye(4)
    target = torch.eye(4)
    target[0, 3] = 2.0
    blended = _se3_blend_pose(source, target, 0.5)
    assert torch.allclose(blended[:3, 3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)

    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 8, "width": 16},
        "Frontend": {"mode": "panovggt_long"},
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [8, 16],
            "chunk_size": 2,
            "overlap": 1,
            "emit_delay": 0,
        },
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendFeedback": {
            "enabled": True,
            "blend_alpha": 1.0,
            "reject_first_keyframe_pose_feedback": True,
            "log_decisions": True,
        },
        "Renderer": {"allow_smoke_fallback": True},
    }
    system = PanoDroidGSSlamSystem(cfg)
    assert hasattr(system.frontend, "pose_by_frame")

    for frame_id in range(2):
        system.mapper.insert_keyframe(
            _small_seed_batch(frame_id),
            _small_frontend_output(frame_id),
            image=torch.full((3, 4, 8), 0.3 + 0.1 * frame_id),
        )
        system.frontend.pose_by_frame[frame_id] = _small_frontend_output(frame_id).pose_c2w

    with torch.no_grad():
        system.mapper.pose_deltas[1].delta[0] = 1.0

    updates, decisions = system._collect_backend_feedback_updates({"steps": 1.0, "loss": 0.1})
    by_id = {int(item["frame_id"]): item for item in decisions}

    assert by_id[0]["accepted"] is False
    assert by_id[0]["reason"] == "first_keyframe_rejected"
    assert by_id[1]["accepted"] is True
    assert set(updates) == {1}
    assert torch.allclose(updates[1][0, 3], torch.tensor(1.1), atol=1e-5)
    assert system._apply_backend_feedback_updates(updates) == 1
    assert torch.allclose(system.frontend.pose_by_frame[1].cpu(), updates[1], atol=1e-5)


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


def test_skybox_optimization_mask_blocks_non_sky_gradients():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
            "optimization_mask_enable": True,
            "sky_mask_top_ratio": 0.5,
            "sky_mask_min_blue": 0.4,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    image = torch.zeros(3, 8, 16)
    image[0, 4:, :] = 1.0
    image[1, :4, :] = 0.35
    image[2, :4, :] = 1.0
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))
    pkg = renderer.render(camera, gaussian_map)
    sky_rgb = pkg["sky_bg_only"]
    assert torch.is_tensor(sky_rgb) and sky_rgb.requires_grad
    sky_rgb.retain_grad()

    sky_mask = mapper._skybox_mask_for_target(image)
    masked_pkg = mapper._apply_skybox_optimization_mask(pkg, sky_mask)
    loss, _ = backend_render_loss(masked_pkg, image)
    loss.backward()

    assert torch.allclose(masked_pkg["render"][:, 4:, :], masked_pkg["gs_only"][:, 4:, :])
    assert float(masked_pkg["render"][:, :4, :].detach().abs().sum()) > 0.0
    grad = sky_rgb.grad
    assert torch.is_tensor(grad)
    assert float(grad[:, :4, :].abs().sum()) > 0.0
    assert float(grad[:, 4:, :].abs().max()) == 0.0


def test_skybox_init_requires_sky_mask_by_default():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
            "sky_mask_top_ratio": 0.5,
            "sky_mask_min_blue": 0.4,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    image = torch.zeros(3, 8, 16)
    image[0] = 1.0

    assert not gaussian_map.initialize_skybox_from_image(image, torch.eye(4))
    assert gaussian_map._skybox_initialized is False


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
    assert (tmp_path / "point_cloud" / "init" / "point_cloud.ply").is_file()
    assert any((tmp_path / "point_cloud" / "init").glob("frame_*.ply"))
    assert Path(summary["artifacts"]["final_ply"]).is_file()
    assert Path(summary["artifacts"]["final_checkpoint"]).is_file()
    checkpoint = torch.load(summary["artifacts"]["final_checkpoint"], map_location="cpu", weights_only=False)
    assert "anchor_birth_frame" in checkpoint
    assert "anchor_outlier_obs" in checkpoint
    assert summary["artifacts"]["final_keyframe_render_count"] >= 1
    assert Path(summary["artifacts"]["final_skybox_erp_preview"]).is_file()
    assert Path(summary["artifacts"]["final_skybox_faces"]).is_file()
    assert any((tmp_path / "init_vis").rglob("iter_*_render.png"))
