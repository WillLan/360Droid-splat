from pathlib import Path

import torch

from backend.pano_gs import NeuralScaffoldPanoMap, PFGS360Renderer, PanoGaussianMap, PanoRenderCamera
from mapping.gaussian_initializer import GaussianSeedBatch
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem


def _cfg(tmp_path: Path | None = None) -> dict:
    cfg = {
        "MapRepresentation": {"mode": "neural_anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat", "pfgs360_render_mode": "RGB+ED"},
        "NeuralScaffold": {
            "enabled": True,
            "feat_dim": 24,
            "hidden_dim": 32,
            "k_offsets": 4,
            "voxel_size": 0.1,
            "insert_radius_factor": 2.0,
            "init_feat_from_rgb": True,
            "init_opacity": 0.15,
            "max_materialized_gaussians": 800000,
        },
        "Renderer": {"allow_smoke_fallback": True},
    }
    if tmp_path is not None:
        cfg["Results"] = {"save_dir": str(tmp_path)}
    return cfg


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
        scale=torch.full((n,), 0.1, dtype=torch.float32),
        level=torch.zeros(n, dtype=torch.long),
        frame_id=frame_id,
        insert_score=insert_score,
        grid_coord=torch.floor(xyz.float() / 0.1).to(torch.int32),
    )


def _force_positive_opacity(neural_map: NeuralScaffoldPanoMap) -> None:
    with torch.no_grad():
        last = neural_map.decoder.mlp_opacity[-1]
        last.weight.zero_()
        last.bias.fill_(1.0)


def test_neural_scaffold_inserts_with_radius_gate():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")

    inserted = neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0]])))
    assert inserted == 1
    assert neural_map.anchor_count() == 1

    rejected = neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.19, 0.0, 1.0]]), frame_id=1))
    assert rejected == 0
    assert neural_map.anchor_count() == 1

    accepted = neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.21, 0.0, 1.0]]), frame_id=2))
    assert accepted == 1
    assert neural_map.anchor_count() == 2


def test_neural_scaffold_batch_duplicate_keeps_best_score():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    seeds = _seed_batch(
        torch.tensor([[0.15, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]),
        insert_score=torch.tensor([0.9, 0.1, 0.2]),
    )

    inserted = neural_map.insert_from_seed_batch(seeds)

    assert inserted == 2
    assert neural_map.anchor_count() == 2
    assert torch.allclose(neural_map.get_xyz[0], torch.tensor([0.15, 0.0, 1.0]))
    assert torch.allclose(neural_map.get_xyz[1], torch.tensor([0.5, 0.0, 1.0]))


def test_neural_scaffold_materialize_outputs_finite_renderer_shapes():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0], [0.25, 0.0, 1.0]])))
    _force_positive_opacity(neural_map)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))

    materialized = neural_map.materialize(camera)

    assert materialized.get_xyz.shape == (8, 3)
    assert materialized.get_opacity.shape == (8, 1)
    assert materialized.get_features.shape == (8, 3)
    assert materialized.get_scaling.shape == (8, 3)
    assert materialized.get_rotation.shape == (8, 4)
    assert torch.isfinite(materialized.get_xyz).all()
    assert torch.isfinite(materialized.get_opacity).all()
    assert torch.isfinite(materialized.get_features).all()
    assert torch.isfinite(materialized.get_scaling).all()
    assert torch.isfinite(materialized.get_rotation).all()
    assert bool(((materialized.get_opacity >= 0.0) & (materialized.get_opacity <= 1.0)).all())
    assert bool(((materialized.get_features >= 0.0) & (materialized.get_features <= 1.0)).all())
    assert bool((materialized.get_scaling > 0.0).all())
    assert torch.allclose(torch.linalg.norm(materialized.get_rotation, dim=-1), torch.ones(8), atol=1.0e-5)


def test_neural_scaffold_optimizer_groups_include_offsets_and_mlps():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    groups = neural_map.get_optimizer_param_groups()
    names = {str(group["name"]) for group in groups}

    assert {"anchor_xyz", "anchor_feat", "anchor_log_scale", "local_offsets", "mlp_opacity", "mlp_color", "mlp_cov"} <= names
    assert neural_map.local_offsets.requires_grad


def test_neural_scaffold_renderer_fallback_aggregates_materialized_stats():
    neural_map = NeuralScaffoldPanoMap(config=_cfg(), device="cpu")
    neural_map.insert_from_seed_batch(_seed_batch(torch.tensor([[0.0, 0.0, 1.0], [0.25, 0.0, 1.0]])))
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
