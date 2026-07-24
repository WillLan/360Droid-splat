from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
import yaml
import pytest

from backend.pano_gs.adapter import PFGS360Renderer, PanoRenderCamera
from backend.pano_gs.mapper import PanoGaussianMap, PanoGaussianMapper
from backend.pano_gs.sky_sphere import (
    PanoLOGSkySphere,
    SkySphereCameraBoundaryError,
    fibonacci_sphere,
)
from models.spherical_voxel_anchor_refiner import quaternion_to_matrix
from system.pano_droid_gs_slam import (
    _SLAM_CORE_VISUAL_WANDB_KEYS,
    load_config,
)
from tools.formal_experiments import (
    _assert_dataset_policy,
    _assert_formal_mainline,
    _deep_merge_config,
    _expand_runs,
)


def _sky_config(
    *,
    count: int = 64,
    bootstrap_chunks: int = 2,
    optimize_all_chunks: bool = False,
    steps: int = 1,
) -> dict:
    return {
        "SkyBox": {"enabled": False},
        "SkySphere": {
            "enabled": True,
            "num_gaussians": count,
            "bootstrap_chunks": bootstrap_chunks,
            "optimize_all_chunks": optimize_all_chunks,
            "optimize_steps_per_chunk": steps,
            "sky_threshold": 0.6,
            "sh_degree": 3,
            "radius_scene_multiplier": 8.0,
            "radius_camera_multiplier": 16.0,
            "initialization": "fibonacci",
            "feature_lr": 2.0e-3,
            "sh_rest_lr": 1.0e-4,
            "opacity_lr": 1.0e-3,
            "scaling_lr": 1.0e-4,
            "rotation_lr": 1.0e-4,
        },
        "Training": {},
    }


def _initialize_sphere(sphere: PanoLOGSkySphere) -> None:
    image = torch.linspace(0.1, 0.9, 8 * 16).reshape(1, 8, 16).repeat(3, 1, 1)
    probability = torch.ones(1, 8, 16)
    pose = torch.eye(4)
    assert sphere.initialize(
        observations=[(image, pose, probability)],
        scene_xyz=torch.tensor([[2.0, 0.0, 0.0]]),
        camera_centers=torch.zeros(1, 3),
    )


class _DifferentiableSkyRenderer:
    def render(
        self,
        camera,
        gaussians,
        *,
        compose_background=True,
        **_,
    ):
        height, width = int(camera.image_height), int(camera.image_width)
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        if isinstance(gaussians, PanoLOGSkySphere):
            signal = (
                gaussians.features.mean(dim=0)
                + 0.01 * gaussians.sh_rest.mean(dim=(0, 1))
                + 0.01 * gaussians.scaling.mean()
                + 0.01 * gaussians.rotation.mean()
            )
            rgb = torch.sigmoid(signal).view(3, 1, 1).expand(3, height, width)
            alpha = torch.sigmoid(
                gaussians.opacity_logit.mean()
                + 0.01 * gaussians.scaling.mean()
            ).view(1, 1, 1).expand(1, height, width)
        else:
            rgb = torch.zeros(3, height, width, device=device, dtype=dtype)
            alpha = torch.zeros(1, height, width, device=device, dtype=dtype)
        return {
            "render": rgb,
            "alpha": alpha,
            "depth": torch.zeros_like(alpha),
            "query_answers": None,
        }


def test_fibonacci_sphere_is_deterministic_equal_area_and_seam_spanning() -> None:
    directions = fibonacci_sphere(1024)
    torch.testing.assert_close(
        torch.linalg.norm(directions, dim=-1),
        torch.ones(1024),
        atol=1.0e-6,
        rtol=0.0,
    )
    y_step = directions[:-1, 1] - directions[1:, 1]
    torch.testing.assert_close(
        y_step,
        torch.full_like(y_step, 2.0 / 1024.0),
        atol=2.0e-6,
        rtol=0.0,
    )
    longitude = torch.atan2(directions[:, 0], directions[:, 2])
    assert bool((longitude < -3.0).any())
    assert bool((longitude > 3.0).any())
    torch.testing.assert_close(directions, fibonacci_sphere(1024))

    sphere = PanoLOGSkySphere(
        config=_sky_config(count=1024)["SkySphere"],
        scene_config=_sky_config(count=1024),
        device="cpu",
    )
    _initialize_sphere(sphere)
    rotation = quaternion_to_matrix(sphere.get_rotation)
    torch.testing.assert_close(
        rotation[:, :, 2],
        sphere.directions,
        atol=2.0e-6,
        rtol=0.0,
    )
    scale = sphere.get_scaling
    torch.testing.assert_close(scale[:, 0], scale[:, 1])
    assert bool((scale[:, 2] < scale[:, 0]).all())


def test_sky_optimization_updates_only_sky_then_freezes_all_parameters() -> None:
    config = _sky_config()
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    gaussian_map.xyz = nn.Parameter(torch.tensor([[2.0, 0.0, 0.0]]))
    mapper = PanoGaussianMapper(
        gaussian_map,
        renderer=_DifferentiableSkyRenderer(),
    )
    image = torch.full((3, 8, 16), 0.8)
    probability = torch.full((1, 8, 16), 0.9)
    sky_mask = probability >= 0.6
    for frame_id in range(2):
        pose = torch.eye(4)
        pose[0, 3] = 0.1 * frame_id
        mapper.register_observation_values(
            frame_id=frame_id,
            image=image,
            c2w=pose,
            is_keyframe=True,
            sky_mask=sky_mask,
            sky_probability=probability,
        )

    sphere = gaussian_map.sky_sphere
    scene_before = gaussian_map.xyz.detach().clone()
    metrics_0 = mapper.optimize_sky_sphere(
        window_id=0,
        recent_frame_ids=(0, 1),
        seed=123,
    )
    xyz_after_first = sphere.get_xyz.detach().clone()
    directions_after_first = sphere.directions.detach().clone()
    sky_params_after_first = {
        name: value.detach().clone()
        for name, value in sphere.named_parameters()
    }
    assert metrics_0["sky_sphere_steps"] == 1.0
    assert not sphere.frozen
    torch.testing.assert_close(gaussian_map.xyz, scene_before)
    assert gaussian_map.anchor_count() == 1
    scene_parameter_ids = {
        id(parameter) for parameter in gaussian_map.gaussian_parameters()
    }
    assert all(
        id(parameter) not in scene_parameter_ids
        for parameter in sphere.parameters()
    )

    metrics_1 = mapper.optimize_sky_sphere(
        window_id=1,
        recent_frame_ids=(0, 1),
        seed=123,
    )
    assert metrics_1["sky_sphere_frozen"] == 1.0
    assert sphere.frozen
    torch.testing.assert_close(sphere.directions, directions_after_first)
    # Radius may only grow during bootstrap, so every center stays on the same
    # fixed direction and no learned XYZ parameter exists.
    cross = torch.cross(
        xyz_after_first - sphere.center,
        sphere.get_xyz.detach() - sphere.center,
        dim=-1,
    )
    assert float(cross.abs().max()) < 1.0e-5
    frozen_state = {
        name: value.detach().clone()
        for name, value in sphere.named_parameters()
    }
    metrics_2 = mapper.optimize_sky_sphere(
        window_id=2,
        recent_frame_ids=(0, 1),
        seed=123,
    )
    assert metrics_2["sky_sphere_steps"] == 0.0
    for name, value in sphere.named_parameters():
        torch.testing.assert_close(value, frozen_state[name])
    assert any(
        not torch.equal(sky_params_after_first[name], frozen_state[name])
        for name in frozen_state
    )


def test_sky_optimization_continues_after_radius_bootstrap_when_enabled() -> None:
    config = _sky_config(optimize_all_chunks=True)
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    gaussian_map.xyz = nn.Parameter(torch.tensor([[2.0, 0.0, 0.0]]))
    mapper = PanoGaussianMapper(
        gaussian_map,
        renderer=_DifferentiableSkyRenderer(),
    )
    image = torch.full((3, 8, 16), 0.8)
    probability = torch.full((1, 8, 16), 0.9)
    for frame_id in range(2):
        pose = torch.eye(4)
        pose[0, 3] = 0.1 * frame_id
        mapper.register_observation_values(
            frame_id=frame_id,
            image=image,
            c2w=pose,
            is_keyframe=True,
            sky_mask=probability >= 0.6,
            sky_probability=probability,
        )

    sphere = gaussian_map.sky_sphere
    center_before = sphere.center.detach().clone()
    directions_before = sphere.directions.detach().clone()
    for window_id in range(2):
        metrics = mapper.optimize_sky_sphere(
            window_id=window_id,
            recent_frame_ids=(0, 1),
            seed=123,
        )
        assert metrics["sky_sphere_steps"] == 1.0
        assert metrics["sky_sphere_frozen"] == 0.0
    assert int(sphere.chunks_completed.item()) == 2
    assert not sphere.frozen
    radius_after_bootstrap = sphere.radius.detach().clone()
    parameters_after_bootstrap = {
        name: value.detach().clone()
        for name, value in sphere.named_parameters()
    }

    with torch.no_grad():
        gaussian_map.xyz[0] = torch.tensor([1000.0, 0.0, 0.0])
    metrics = mapper.optimize_sky_sphere(
        window_id=2,
        recent_frame_ids=(0, 1),
        seed=123,
    )

    assert metrics["sky_sphere_steps"] == 1.0
    assert metrics["sky_sphere_frozen"] == 0.0
    assert int(sphere.chunks_completed.item()) == 3
    torch.testing.assert_close(sphere.radius, radius_after_bootstrap)
    torch.testing.assert_close(sphere.center, center_before)
    torch.testing.assert_close(sphere.directions, directions_before)
    assert any(
        not torch.equal(parameters_after_bootstrap[name], value)
        for name, value in sphere.named_parameters()
    )
    metadata = sphere.metadata()
    assert metadata["radius_bootstrap_complete"] is True
    assert metadata["optimize_all_chunks"] is True
    assert metadata["frozen_chunk"] is None


def test_sky_composite_preserves_scene_geometry_and_query_bitwise() -> None:
    config = _sky_config(count=32)
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(
        image_height=8,
        image_width=16,
        c2w=torch.eye(4),
    )
    depth = torch.rand(1, 8, 16)
    query = torch.rand(3, 2)
    scene_rgb = torch.full((3, 8, 16), 0.2)
    scene_alpha = torch.full((1, 8, 16), 0.4)
    package = {
        "render": scene_rgb,
        "alpha": scene_alpha,
        "depth": depth,
        "query_answers": query,
    }
    preinitialized = renderer._compose_background(
        camera,
        gaussian_map,
        package,
        enabled=True,
    )
    assert set(
        ("scene_rgb", "scene_alpha", "sky_rgb", "sky_alpha", "composite_rgb")
    ).issubset(preinitialized)
    torch.testing.assert_close(preinitialized["composite_rgb"], scene_rgb)
    _initialize_sphere(gaussian_map.sky_sphere)
    composite = renderer._compose_sky_sphere(camera, gaussian_map, package)
    assert composite["depth"] is depth
    assert composite["query_answers"] is query
    assert composite["scene_rgb"] is scene_rgb
    assert composite["scene_alpha"] is scene_alpha
    assert set(
        ("scene_rgb", "scene_alpha", "sky_rgb", "sky_alpha", "composite_rgb")
    ).issubset(composite)
    torch.testing.assert_close(
        composite["composite_rgb"],
        scene_rgb
        + (1.0 - scene_alpha)
        * composite["sky_rgb"],
    )


def test_cubemap_composite_is_opaque_and_preserves_scene_geometry() -> None:
    config = {
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
        },
        "SkySphere": {"enabled": False},
        "Training": {},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    image = torch.full((3, 8, 16), 0.7)
    assert gaussian_map.initialize_skybox_from_image(
        image,
        torch.eye(4),
        sky_mask=torch.ones(1, 8, 16, dtype=torch.bool),
    )
    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(8, 16, torch.eye(4))
    depth = torch.rand(1, 8, 16)
    query = torch.rand(3, 2)
    scene_rgb = torch.full((3, 8, 16), 0.2)
    scene_alpha = torch.full((1, 8, 16), 0.4)

    composite = renderer._compose_skybox(
        camera,
        gaussian_map,
        {
            "render": scene_rgb,
            "alpha": scene_alpha,
            "depth": depth,
            "query_answers": query,
        },
    )

    assert composite["depth"] is depth
    assert composite["query_answers"] is query
    torch.testing.assert_close(composite["sky_alpha"], torch.ones_like(scene_alpha))
    torch.testing.assert_close(
        composite["composite_rgb"],
        scene_rgb + (1.0 - scene_alpha) * composite["sky_rgb"],
    )


def test_cubemap_sky_optimizes_every_chunk_only_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
            "optimize_all_chunks": True,
            "optimize_steps_per_chunk": 1,
            "sky_threshold": 0.6,
            "lr": 1.0e-2,
        },
        "SkySphere": {"enabled": False},
        "Mapping": {
            "sky_mask_enable": True,
            "sky_mask_source": "panovggt_head",
        },
        "Training": {},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    gaussian_map._skybox_initialized = True
    mapper = PanoGaussianMapper(
        gaussian_map,
        renderer=_DifferentiableSkyRenderer(),
    )
    probability = torch.full((1, 8, 16), 0.6)
    probability[:, 0, 0] = 0.59
    for frame_id in range(2):
        mapper.register_observation_values(
            frame_id=frame_id,
            image=torch.full((3, 8, 16), 0.8),
            c2w=torch.eye(4),
            sky_mask=probability >= 0.6,
            sky_probability=probability,
        )
    monkeypatch.setattr(
        mapper,
        "_cubemap_sky_diagnostic",
        lambda observation, *, window_id: {
            "mode": "cubemap",
            "window_id": window_id,
            "frame_id": observation.frame_id,
        },
    )
    scene_before = gaussian_map.xyz.detach().clone()
    cubemap_before = gaussian_map.skybox_logits.detach().clone()

    metrics = mapper.optimize_cubemap_sky(
        window_id=0,
        recent_frame_ids=(0, 1),
        new_frame_ids=(1,),
        seed=123,
    )

    assert metrics["cubemap_sky_steps"] == 1.0
    assert metrics["cubemap_sky_reliable_pixels"] == 2.0 * (8 * 16 - 1)
    assert not torch.equal(gaussian_map.skybox_logits, cubemap_before)
    torch.testing.assert_close(gaussian_map.xyz, scene_before)

    rollback_before = gaussian_map.skybox_logits.detach().clone()
    monkeypatch.setattr(
        mapper,
        "_cubemap_sky_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic CubeMap diagnostic failure")
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="synthetic CubeMap diagnostic failure",
    ):
        mapper.optimize_cubemap_sky(
            window_id=1,
            recent_frame_ids=(0, 1),
            new_frame_ids=(1,),
            seed=123,
        )
    torch.testing.assert_close(
        gaussian_map.skybox_logits,
        rollback_before,
    )

    legacy = copy.deepcopy(config)
    legacy["SkyBox"].pop("optimize_all_chunks")
    legacy_map = PanoGaussianMap(config=legacy, device="cpu")
    legacy_map._skybox_initialized = True
    legacy_mapper = PanoGaussianMapper(
        legacy_map,
        renderer=_DifferentiableSkyRenderer(),
    )
    before = legacy_map.skybox_logits.detach().clone()
    assert (
        legacy_mapper.optimize_cubemap_sky(
            window_id=0,
            recent_frame_ids=(),
            new_frame_ids=(),
            seed=123,
        )
        == {}
    )
    torch.testing.assert_close(legacy_map.skybox_logits, before)


def test_cubemap_checkpoint_restores_faces_and_initialization_state(
    tmp_path: Path,
) -> None:
    config = {
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
        },
        "SkySphere": {"enabled": False},
        "Training": {},
    }
    source = PanoGaussianMap(config=config, device="cpu")
    assert source.initialize_skybox_from_image(
        torch.full((3, 8, 16), 0.7),
        torch.eye(4),
        sky_mask=torch.ones(1, 8, 16, dtype=torch.bool),
    )
    checkpoint = tmp_path / "cubemap.pt"
    source.save_checkpoint(checkpoint)

    restored = PanoGaussianMap(config=config, device="cpu")
    restored.load_checkpoint(checkpoint)

    assert restored._skybox_initialized is True
    torch.testing.assert_close(
        restored.skybox_logits,
        source.skybox_logits,
    )

    legacy = tmp_path / "legacy_cubemap.pt"
    torch.save({"state_dict": source.state_dict()}, legacy)
    legacy_restored = PanoGaussianMap(config=config, device="cpu")
    legacy_restored.load_checkpoint(legacy)
    assert legacy_restored._skybox_initialized is True
    torch.testing.assert_close(
        legacy_restored.skybox_logits,
        source.skybox_logits,
    )


def test_sky_sphere_checkpoint_roundtrip_and_legacy_checkpoint_loading(
    tmp_path: Path,
) -> None:
    config = _sky_config(count=32)
    source = PanoGaussianMap(config=config, device="cpu")
    _initialize_sphere(source.sky_sphere)
    source.sky_sphere.complete_chunk()
    ply = tmp_path / "sky_sphere.ply"
    assert source.save_sky_sphere_ply(ply) == str(ply)
    assert ply.is_file() and ply.stat().st_size > 0
    checkpoint = tmp_path / "with_sky.pt"
    source.save_checkpoint(checkpoint)

    restored = PanoGaussianMap(config=config, device="cpu")
    restored.load_checkpoint(checkpoint)
    assert restored.has_sky_sphere
    torch.testing.assert_close(
        restored.sky_sphere.get_xyz,
        source.sky_sphere.get_xyz,
    )
    torch.testing.assert_close(
        restored.sky_sphere.get_sh_coefficients,
        source.sky_sphere.get_sh_coefficients,
    )

    legacy_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("sky_sphere.")
    }
    legacy = tmp_path / "legacy.pt"
    torch.save({"state_dict": legacy_state}, legacy)
    old_compatible = PanoGaussianMap(config=config, device="cpu")
    old_compatible.load_checkpoint(legacy, strict=True)
    assert not old_compatible.has_sky_sphere


def test_sky_sphere_camera_warning_and_abort_thresholds() -> None:
    sphere = PanoLOGSkySphere(
        config=_sky_config(count=32)["SkySphere"],
        scene_config=_sky_config(count=32),
        device="cpu",
    )
    _initialize_sphere(sphere)
    warning_pose = torch.eye(4)
    warning_pose[0, 3] = 0.3 * sphere.radius
    ratio, warning = sphere.validate_camera(warning_pose)
    assert ratio == pytest.approx(0.3)
    assert warning is True
    abort_pose = torch.eye(4)
    abort_pose[0, 3] = 0.81 * sphere.radius
    with pytest.raises(SkySphereCameraBoundaryError):
        sphere.validate_camera(abort_pose)
    conflicting = _sky_config(count=32)
    conflicting["SkyBox"]["enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        PanoGaussianMap(config=conflicting, device="cpu")


def test_360vo_skysphere_formal_config_and_ob3d_policy_are_isolated() -> None:
    root = Path(__file__).parents[1]
    campaign_path = (
        root
        / "configs/formal/panogsslam_formal_360vo_first200_skysphere_v5.yaml"
    )
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    base = load_config(root / campaign["base_config"])
    run = _expand_runs(campaign)[0]
    resolved = _deep_merge_config(copy.deepcopy(base), run.config_overrides)
    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, run)
    assert resolved["SkyBox"]["enabled"] is False
    assert resolved["SkySphere"]["enabled"] is True
    assert resolved["SkySphere"]["num_gaussians"] == 65536
    assert resolved["SkySphere"]["optimize_all_chunks"] is True
    assert "backend/sky_sphere" in _SLAM_CORE_VISUAL_WANDB_KEYS
    assert (
        resolved["SphericalSelfiGlobalBackend"]["global_graph"][
            "sky_threshold"
        ]
        == 0.6
    )

    ob3d_campaign = yaml.safe_load(
        (
            root / "configs/formal/panogsslam_formal_ob3d_v3.yaml"
        ).read_text(encoding="utf-8")
    )
    ob3d_run = _expand_runs(ob3d_campaign)[0]
    ob3d = _deep_merge_config(copy.deepcopy(base), ob3d_run.config_overrides)
    _assert_dataset_policy(ob3d, ob3d_run)
    assert ob3d["SkyBox"]["enabled"] is False
    assert not bool(dict(ob3d.get("SkySphere", {}) or {}).get("enabled", False))


def test_360vo_cubemap_formal_config_keeps_mainline_and_sky_cleanup() -> None:
    root = Path(__file__).parents[1]
    campaign_path = (
        root
        / "configs/formal/panogsslam_formal_360vo_first200_cubemap_v7.yaml"
    )
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    base = load_config(root / campaign["base_config"])
    run = _expand_runs(campaign)[0]
    resolved = _deep_merge_config(copy.deepcopy(base), run.config_overrides)

    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, run)
    assert resolved["SkyBox"]["enabled"] is True
    assert resolved["SkyBox"]["type"] == "cubemap"
    assert resolved["SkyBox"]["resolution"] == 256
    assert resolved["SkyBox"]["optimize_all_chunks"] is True
    assert resolved["SkyBox"]["sky_threshold"] == 0.6
    assert resolved["SkySphere"]["enabled"] is False
    assert resolved["Results"]["save_skybox_previews"] is True
    cleanup = resolved["SphericalSelfiGlobalBackend"]["map_optimization"][
        "pfgs360"
    ]["sky_occluder_cleanup"]
    assert cleanup == {
        "enabled": True,
        "sky_threshold": 0.6,
        "responsibility_threshold": 0.8,
        "reset_opacity": 0.01,
    }
    assert "backend/cubemap_sky" in _SLAM_CORE_VISUAL_WANDB_KEYS


def test_360vo_cubemap_nohash_formal_config_disables_growth_hash_only() -> None:
    root = Path(__file__).parents[1]
    v7_path = (
        root
        / "configs/formal/panogsslam_formal_360vo_first200_cubemap_v7.yaml"
    )
    v8_path = (
        root
        / "configs/formal/panogsslam_formal_360vo_first200_cubemap_nohash_v8.yaml"
    )
    v7_campaign = yaml.safe_load(v7_path.read_text(encoding="utf-8"))
    v8_campaign = yaml.safe_load(v8_path.read_text(encoding="utf-8"))
    base = load_config(root / v8_campaign["base_config"])
    v7_run = _expand_runs(v7_campaign)[0]
    v8_run = _expand_runs(v8_campaign)[0]
    v7 = _deep_merge_config(copy.deepcopy(base), v7_run.config_overrides)
    v8 = _deep_merge_config(copy.deepcopy(base), v8_run.config_overrides)

    _assert_formal_mainline(v8, seed=123)
    _assert_dataset_policy(v8, v8_run)
    v8_pfgs = v8["SphericalSelfiGlobalBackend"]["map_optimization"][
        "pfgs360"
    ]
    assert v8_pfgs["growth_hash_dedup_enabled"] is False
    assert v8_pfgs["atomic_refined_anchor_replacement"] is True
    assert v8_pfgs["append_only_refined_anchors"] is True
    assert v8_pfgs["anchor_footprint"] == v7[
        "SphericalSelfiGlobalBackend"
    ]["map_optimization"]["pfgs360"]["anchor_footprint"]
    assert v8["SkyBox"] == v7["SkyBox"]
    assert v8["SkySphere"] == v7["SkySphere"]
