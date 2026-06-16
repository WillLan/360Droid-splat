from pathlib import Path

import torch

from backend.pano_gs import NeuralScaffoldPanoMap, PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper, PanoRenderCamera
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


def _cfg(tmp_path: Path | None = None) -> dict:
    cfg = {
        "MapRepresentation": {"mode": "neural_anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat", "pfgs360_render_mode": "RGB+ED"},
        "NeuralScaffold": {
            "enabled": True,
            "scaffold_alignment": "scaffold_gs_v2",
            "feat_dim": 32,
            "hidden_dim": 32,
            "k_offsets": 10,
            "voxel_size": 0.05,
            "insert_radius_factor": 2.0,
            "opacity_mask_threshold": 0.0,
            "freeze_mlp_after_first_chunk": True,
            "save_mlp": True,
            "max_materialized_gaussians": 800000,
        },
        "BackendOptimization": {"enabled": True, "optimize_after_every_chunk": True, "first_chunk_steps": 4, "steps_per_chunk": 2},
        "Renderer": {"allow_smoke_fallback": True},
    }
    if tmp_path is not None:
        cfg["Results"] = {"save_dir": str(tmp_path)}
    return cfg


def _frontend_output(
    points: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    frame_id: int = 0,
) -> FrontendOutput:
    H, W = int(points.shape[0]), int(points.shape[1])
    return FrontendOutput(
        frame_id=frame_id,
        timestamp=float(frame_id),
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, H, W),
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=None,
        tracking_status="ok",
        world_points=points.float(),
        world_points_confidence=None,
        valid_world_points_mask=valid_mask,
    )


def _image(H: int, W: int) -> torch.Tensor:
    base = torch.linspace(0.2, 0.8, steps=H * W).view(1, H, W)
    return base.expand(3, H, W).contiguous()


def _seed_batch(
    xyz: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    insert_score: torch.Tensor | None = None,
    frame_id: int = 0,
) -> GaussianSeedBatch:
    n = int(xyz.shape[0])
    return GaussianSeedBatch(
        xyz=xyz.float(),
        rgb=torch.linspace(0.2, 0.8, steps=max(1, n)).view(n, 1).expand(n, 3).float(),
        confidence=torch.ones(n, dtype=torch.float32) if confidence is None else confidence.float(),
        scale=torch.full((n,), 0.05, dtype=torch.float32),
        level=torch.zeros(n, dtype=torch.long),
        frame_id=frame_id,
        insert_score=insert_score,
        grid_coord=torch.floor(xyz.float() / 0.05).to(torch.int32),
    )


def _force_positive_opacity(neural_map: NeuralScaffoldPanoMap) -> None:
    with torch.no_grad():
        opacity_linear = neural_map.decoder.mlp_opacity[-2]
        opacity_linear.weight.zero_()
        opacity_linear.bias.fill_(1.0)


def test_decoder_uses_scaffold_gs_input_shape():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")

    assert neural_map.decoder.input_dim == neural_map.feat_dim + 4
    assert neural_map.decoder.mlp_opacity[-1].__class__.__name__ == "Tanh"
    assert neural_map.decoder.mlp_color[-1].__class__.__name__ == "Sigmoid"


def test_neural_frontend_insertion_uses_sky_and_raster_order():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    points = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.09, 0.0, 1.0], [0.11, 0.0, 1.0]],
            [[0.3, 0.0, 1.0], [0.36, 0.0, 1.0], [0.5, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    sky = torch.zeros(1, 2, 3, dtype=torch.bool)
    sky[0, 1, 2] = True

    inserted = neural_map.insert_from_frontend_output(_frontend_output(points), _image(2, 3), sky_mask=sky)

    assert inserted == 5
    assert neural_map.anchor_count() == 5
    assert torch.allclose(neural_map.get_xyz[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[1], torch.tensor([0.09, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[2], torch.tensor([0.11, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[3], torch.tensor([0.3, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[4], torch.tensor([0.36, 0.0, 1.0]))


def test_neural_existing_anchor_radius_gate_boundaries():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0]])))

    rejected = neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.09, 0.0, 1.0]]), frame_id=1))
    accepted = neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.11, 0.0, 1.0]]), frame_id=2))

    assert rejected == 0
    assert accepted == 1
    assert neural_map.anchor_count() == 2


def test_neural_seed_scores_do_not_change_insertion_order():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    seeds = _seed_batch(
        torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]]),
        confidence=torch.tensor([0.1, 1.0, 0.2]),
        insert_score=torch.tensor([0.1, 0.9, 0.2]),
    )

    inserted = neural_map.insert_from_seed_batch(seeds)

    assert inserted == 3
    assert torch.allclose(neural_map.get_xyz[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[1], torch.tensor([0.2, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[2], torch.tensor([0.4, 0.0, 1.0]))


def test_neural_voxel_compact_keeps_first_anchor():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    seeds = _seed_batch(torch.tensor([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0], [0.06, 0.0, 1.0]]))

    inserted = neural_map.insert_from_seed_batch(seeds)

    assert inserted == 2
    assert neural_map.anchor_count() == 2
    assert torch.allclose(neural_map.get_xyz[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[1], torch.tensor([0.06, 0.0, 1.0]))


def test_neural_scaffold_materialize_outputs_finite_renderer_shapes():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0], [0.12, 0.0, 1.0]])))
    _force_positive_opacity(neural_map)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))

    materialized = neural_map.materialize(camera)

    assert materialized.get_xyz.shape == (20, 3)
    assert materialized.get_opacity.shape == (20, 1)
    assert materialized.get_features.shape == (20, 3)
    assert materialized.get_scaling.shape == (20, 3)
    assert materialized.get_rotation.shape == (20, 4)
    assert torch.isfinite(materialized.get_xyz).all()
    assert torch.isfinite(materialized.get_opacity).all()
    assert torch.isfinite(materialized.get_features).all()
    assert torch.isfinite(materialized.get_scaling).all()
    assert torch.isfinite(materialized.get_rotation).all()
    assert bool(((materialized.get_opacity >= 0.0) & (materialized.get_opacity <= 1.0)).all())
    assert bool(((materialized.get_features >= 0.0) & (materialized.get_features <= 1.0)).all())
    assert bool((materialized.get_scaling > 0.0).all())
    assert torch.allclose(torch.linalg.norm(materialized.get_rotation, dim=-1), torch.ones(20), atol=1.0e-5)


def test_neural_scaffold_optimizer_groups_freeze_mlp():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    names = {str(group["name"]) for group in neural_map.get_optimizer_param_groups()}

    assert {"anchor_xyz", "anchor_feat", "anchor_log_scale", "local_offsets", "mlp_opacity", "mlp_color", "mlp_cov"} <= names
    assert neural_map.local_offsets.requires_grad

    neural_map.freeze_mlp()
    frozen_names = {str(group["name"]) for group in neural_map.get_optimizer_param_groups()}
    assert "mlp_opacity" not in frozen_names
    assert "mlp_color" not in frozen_names
    assert "mlp_cov" not in frozen_names
    assert not any(param.requires_grad for param in neural_map.decoder.parameters())


def test_mapper_neural_first_chunk_mlp_group_then_freeze():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    mapper = PanoGaussianMapper(neural_map, renderer=PFGS360Renderer(config=_cfg(), allow_fallback=True))

    assert mapper._neural_should_train_mlp_for_chunk(0)
    first_names = {str(group["name"]) for group in mapper._map_param_groups(gaussian_enabled=True, phase="feedforward_window")}
    assert {"mlp_opacity", "mlp_color", "mlp_cov"} <= first_names

    neural_map.freeze_mlp()
    mapper._neural_first_chunk_optimized = True
    assert not mapper._neural_should_train_mlp_for_chunk(1)
    second_names = {str(group["name"]) for group in mapper._map_param_groups(gaussian_enabled=True, phase="feedforward_window")}
    assert "mlp_opacity" not in second_names
    assert "mlp_color" not in second_names
    assert "mlp_cov" not in second_names


def test_neural_scaffold_checkpoint_saves_mlp_state(tmp_path: Path):
    neural_map = NeuralScaffoldPanoMap(config=_cfg(tmp_path), device="cpu")
    ckpt = tmp_path / "final_gaussian_map.pt"

    neural_map.save_checkpoint(ckpt)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sidecar = torch.load(tmp_path / "mlp_state.pth", map_location="cpu", weights_only=False)
    assert "mlp_state" in payload
    assert {"mlp_opacity", "mlp_color", "mlp_cov"} <= set(sidecar)
    assert sidecar["input_mode"] == "anchor_feat_view_dir_ob_dist"


def test_neural_scaffold_renderer_fallback_aggregates_materialized_stats():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0], [0.12, 0.0, 1.0]])))
    _force_positive_opacity(neural_map)
    renderer = PFGS360Renderer(config=_cfg(), allow_fallback=True)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))

    pkg = renderer.render(camera, neural_map)

    assert pkg["visibility_filter"].shape == (2,)
    assert pkg["radii"].shape == (2,)
    assert pkg["n_touched"].shape == (2,)
    assert pkg["viewspace_points"].shape == (2, 2)


def test_system_map_factory_keeps_default_and_selects_neural(tmp_path: Path):
    base_cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 8, "width": 16},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Renderer": {"allow_smoke_fallback": True},
        "Results": {"save_dir": str(tmp_path / "default")},
    }
    default_system = PanoDroidGSSlamSystem({**base_cfg, "MapRepresentation": {"mode": "anchor_scaffold_panorama"}})
    assert isinstance(default_system.map, PanoGaussianMap)

    neural_cfg = {**base_cfg, "MapRepresentation": {"mode": "neural_anchor_scaffold_panorama"}, "NeuralScaffold": _cfg()["NeuralScaffold"]}
    neural_cfg["Results"] = {"save_dir": str(tmp_path / "neural")}
    neural_system = PanoDroidGSSlamSystem(neural_cfg)
    assert isinstance(neural_system.map, NeuralScaffoldPanoMap)
