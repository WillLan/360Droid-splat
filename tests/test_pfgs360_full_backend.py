from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from backend.pano_gs.mapper import PanoGaussianMap, PanoGaussianMapper
from backend.pano_gs.pfgs360_full import (
    PFGS360FullBackend,
    affine_align_depth,
    sample_erp_with_wrap,
)
from backend.pano_gs.pose_param import PoseDelta
from system.pano_droid_gs_slam import (
    _SLAM_CORE_VISUAL_WANDB_KEYS,
    _resolve_cpu_threading_config,
    _requires_refiner_insertion_dedup,
    load_config,
    SlamRuntimeLogger,
)
from backend.pano_gs.spherical_selfi_global import SphericalSelfiGlobalBackend


def _config() -> dict:
    return {
        "MapRepresentation": {"gaussian_parameterization": "traditional_3dgs"},
        "BackendOptimization": {"sh_degree": 2},
        "SphericalSelfiGlobalBackend": {
            "enabled": True,
            "rgb_sh_degree": 2,
            "map_optimization": {"strategy": "pfgs360_full_50_50"},
        },
        "Training": {"pfgs360_absgrad": True, "pfgs360_distloss": True},
        "Mapping": {"sky_mask_source": "heuristic"},
    }


def test_pose_delta_rebase_preserves_photometric_residual() -> None:
    pose = PoseDelta(torch.eye(4), torch.tensor([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]))
    before_delta = pose.delta.detach().clone()
    new_base = torch.eye(4)
    new_base[1, 3] = 2.0
    pose.rebase(new_base, preserve_delta=True)
    assert torch.equal(pose.delta.detach(), before_delta)
    assert torch.equal(pose.canonical_pose(), new_base)
    assert not torch.equal(pose().detach(), new_base)


def test_erp_sampler_wraps_longitude_seam() -> None:
    image = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    pixels = torch.tensor([[[-0.25, 0.0], [7.75, 0.0], [8.25, 0.0]]])
    sampled, valid = sample_erp_with_wrap(image, pixels)
    assert bool(valid.all())
    assert torch.allclose(sampled[0, 0, 0], sampled[0, 0, 1])
    assert torch.allclose(sampled[0, 0, 2], torch.tensor(0.25), atol=1.0e-6)


def test_affine_depth_alignment_recovers_scale_and_shift() -> None:
    predicted = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 4)
    rendered = 1.5 * predicted + 0.25
    aligned, scale, shift, count = affine_align_depth(
        predicted, rendered, torch.ones_like(predicted, dtype=torch.bool), max_depth=100.0
    )
    assert count == 16
    assert abs(scale - 1.5) < 1.0e-5
    assert abs(shift - 0.25) < 1.0e-5
    assert torch.allclose(aligned, rendered, atol=1.0e-5)


def test_official_dia_validity_ignores_alpha_and_confidence_but_keeps_sky() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    observation = mapper.observations[0]
    observation.depth_confidence = torch.zeros(1, 4, 8)
    observation.sky_mask = torch.zeros(1, 4, 8, dtype=torch.bool)
    observation.sky_mask[:, 0, 0] = True
    depth = torch.ones(1, 4, 8)
    alpha = torch.zeros_like(depth)

    official = PFGS360FullBackend(
        mapper,
        {"validity_gate": "pfgs360_official_sky_only"},
    )._dia_valid_mask(observation, depth, alpha)
    legacy = PFGS360FullBackend(mapper)._dia_valid_mask(
        observation,
        depth,
        alpha,
    )

    assert int(official.sum()) == depth.numel() - 1
    assert not bool(official[0, 0, 0])
    assert not bool(legacy.any())


def test_official_no_semantic_dia_validity_keeps_indoor_sky_labels() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    observation = mapper.observations[0]
    observation.depth_confidence = torch.zeros(1, 4, 8)
    observation.sky_mask = torch.ones(1, 4, 8, dtype=torch.bool)
    depth = torch.ones(1, 4, 8)
    depth[:, 0, 0] = float("nan")
    alpha = torch.zeros_like(depth)

    valid = PFGS360FullBackend(
        mapper,
        {"validity_gate": "pfgs360_official_no_semantic_gate"},
    )._dia_valid_mask(observation, depth, alpha)

    assert int(valid.sum()) == depth.numel() - 1
    assert not bool(valid[0, 0, 0])


def test_pfgs360_voxel_growth_uses_traditional_3dgs_initialization() -> None:
    gaussian_map = PanoGaussianMap(config=_config(), device="cpu")
    grid_x, grid_y = torch.meshgrid(
        torch.arange(12), torch.arange(12), indexing="ij"
    )
    xyz = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1
    ).float() * 0.02
    rgb = torch.rand(144, 3)
    stats = gaussian_map.append_pfgs360_points(
        xyz,
        rgb,
        owner_window_id=3,
        frame_id=7,
        voxel_size=0.01,
        min_unique_voxels=100,
    )
    assert stats["inserted"] == 144
    assert gaussian_map.anchor_count() == 144
    assert torch.allclose(
        gaussian_map.get_opacity,
        torch.full((144, 1), 0.01),
        atol=1.0e-6,
    )
    assert bool(torch.isfinite(gaussian_map.scaling).all())
    assert torch.allclose(
        torch.linalg.norm(gaussian_map.get_rotation, dim=-1),
        torch.ones(144),
        atol=1.0e-5,
    )
    assert bool((gaussian_map._anchor_owner_window_id == 3).all())
    for name in gaussian_map._anchor_metadata_names():
        assert int(getattr(gaussian_map, name).shape[0]) == gaussian_map.anchor_count()


def test_pfgs360_refine_has_no_deletion_cap_and_keeps_metadata_aligned() -> None:
    gaussian_map = PanoGaussianMap(config=_config(), device="cpu")
    xyz = torch.stack(
        [torch.arange(120), torch.zeros(120), torch.ones(120)], dim=-1
    ).float() * 0.02
    gaussian_map.append_pfgs360_points(
        xyz,
        torch.rand(120, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    with torch.no_grad():
        gaussian_map.opacity_logit.fill_(-20.0)
    result = gaussian_map.pfgs360_refine_topology(
        torch.zeros(120), torch.zeros(120), cull_opacity=0.005
    )
    assert result["culled"] == 120
    assert gaussian_map.anchor_count() == 0
    for name in gaussian_map._anchor_metadata_names():
        assert int(getattr(gaussian_map, name).shape[0]) == 0


class _DifferentiableFakeRenderer:
    def render(self, camera, gaussians, *, query_values=None):
        height, width = int(camera.image_height), int(camera.image_width)
        count = gaussians.anchor_count()
        if count:
            viewspace = gaussians.xyz[:, :2]
            scalar = (
                viewspace.mean()
                + gaussians.get_scaling.mean()
                + gaussians.get_rotation.mean()
                + gaussians.get_opacity.mean()
                + gaussians.get_sh_coefficients.mean()
                + camera.c2w[:3, 3].mean()
            )
            rgb = torch.sigmoid(scalar).expand(3, height, width)
            depth = (1.0 + 0.01 * gaussians.get_xyz.mean()).expand(1, height, width)
            radii = torch.ones(count, device=gaussians.xyz.device)
            accum = torch.ones(count, device=gaussians.xyz.device)
        else:
            viewspace = gaussians.xyz[:, :2]
            rgb = torch.zeros(3, height, width)
            depth = torch.ones(1, height, width)
            radii = torch.zeros(0)
            accum = torch.zeros(0)
        answers = None
        if query_values is not None:
            answers = torch.zeros(count, int(query_values.shape[-1]), device=depth.device)
        return {
            "render": rgb,
            "depth": depth,
            "alpha": torch.ones(1, height, width, device=depth.device),
            "opacity": torch.ones(1, height, width, device=depth.device),
            "render_distort": torch.zeros(1, height, width, device=depth.device),
            "radii": radii,
            "accum_visible": accum,
            "query_answers": answers,
            "viewspace_points": viewspace,
        }


class _AttributedFakeRenderer(_DifferentiableFakeRenderer):
    def __init__(self, *, fail_query: bool = False) -> None:
        self.fail_query = bool(fail_query)

    def render(self, camera, gaussians, *, query_values=None):
        output = super().render(camera, gaussians, query_values=query_values)
        if query_values is not None:
            if self.fail_query:
                output["query_answers"] = None
                return output
            count = gaussians.anchor_count()
            answers = torch.zeros(count, 2, device=gaussians.xyz.device)
            answers[:120, 0] = 1.0
            answers[120:240, 1] = 1.0
            output["query_answers"] = answers
        return output


class _PartiallyVisibleFakeRenderer(_DifferentiableFakeRenderer):
    def render(self, camera, gaussians, *, query_values=None):
        output = super().render(camera, gaussians, query_values=query_values)
        count = gaussians.anchor_count()
        visible = torch.zeros(count, device=gaussians.xyz.device)
        visible[: count // 2] = 1.0
        output["accum_visible"] = visible
        output["radii"] = torch.ones_like(visible)
        return output


class _PartialContributorFakeRenderer(_DifferentiableFakeRenderer):
    def render(self, camera, gaussians, *, query_values=None):
        height, width = int(camera.image_height), int(camera.image_width)
        count = gaussians.anchor_count()
        contributor_count = max(0, count - 44)
        if contributor_count:
            rows = slice(0, contributor_count)
            viewspace = gaussians.xyz[:, :2]
            scalar = (
                gaussians.xyz[rows].mean()
                + gaussians.get_scaling[rows].mean()
                + gaussians.get_rotation[rows].mean()
                + gaussians.get_opacity[rows].mean()
                + gaussians.get_sh_coefficients[rows].mean()
                + camera.c2w[:3, 3].mean()
            )
            rgb = torch.sigmoid(scalar).expand(3, height, width)
            depth = (
                1.0 + 0.01 * gaussians.xyz[rows].mean()
            ).expand(1, height, width)
        else:
            viewspace = gaussians.xyz[:, :2]
            rgb = torch.zeros(3, height, width, device=gaussians.xyz.device)
            depth = torch.ones(1, height, width, device=gaussians.xyz.device)
        accumulated = torch.zeros(count, device=gaussians.xyz.device)
        accumulated[: count // 2] = 1.0
        answers = None
        if query_values is not None:
            answers = torch.zeros(
                count,
                int(query_values.shape[-1]),
                device=gaussians.xyz.device,
            )
        return {
            "render": rgb,
            "depth": depth,
            "alpha": torch.ones(1, height, width, device=gaussians.xyz.device),
            "opacity": torch.ones(1, height, width, device=gaussians.xyz.device),
            "render_distort": torch.zeros(
                1,
                height,
                width,
                device=gaussians.xyz.device,
            ),
            "radii": torch.ones(count, device=gaussians.xyz.device),
            "accum_visible": accumulated,
            "query_answers": answers,
            "viewspace_points": viewspace,
        }


class _NoAccumulatedVisibilityFakeRenderer(_DifferentiableFakeRenderer):
    def render(self, camera, gaussians, *, query_values=None):
        output = super().render(
            camera,
            gaussians,
            query_values=query_values,
        )
        output.pop("accum_visible", None)
        return output


def _registered_mapper(renderer) -> PanoGaussianMapper:
    gaussian_map = PanoGaussianMap(config=_config(), device="cpu")
    gaussian_map.configure_lazy_owner_transforms(True)
    gaussian_map.set_lazy_owner_transform(0, torch.eye(4), set_reference=True)
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)
    for frame_id in range(4):
        pose = torch.eye(4)
        pose[0, 3] = 0.01 * frame_id
        mapper.register_observation_values(
            frame_id=frame_id,
            image=torch.full((3, 4, 8), 0.2 + 0.05 * frame_id),
            c2w=pose,
            inverse_depth=torch.ones(1, 4, 8),
            depth_confidence=torch.ones(1, 4, 8),
            sky_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        )
    assert mapper.prepare_spherical_selfi_window((0, 1, 2, 3)) == 4
    return mapper


def test_full_backend_runs_camera_dia_joint_and_keeps_owner_transform() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    gaussian_map = mapper.map
    owner_before = gaussian_map.lazy_owner_transform_state()
    metrics = mapper.optimize_pfgs360_full_50_50(
        window_id=0,
        frame_ids=(0, 1, 2, 3),
        new_frame_ids=(0, 1, 2, 3),
        settings={
            "camera_steps": 2,
            "joint_steps": 2,
            "min_unique_growth_voxels": 1,
            "min_raw_growth_points": 1,
            "min_reset_gaussians": 10_000,
            "min_delete_gaussians": 10_000,
            "refine_every_joint_steps": 100,
        },
    )
    assert metrics["camera_steps"] == 2
    assert metrics["joint_steps"] == 2
    assert torch.equal(mapper.pose_deltas[0].delta, torch.zeros(6))
    assert any(
        float(mapper.pose_deltas[index].delta.detach().norm()) > 0.0
        for index in (1, 2, 3)
    )
    owner_after = gaussian_map.lazy_owner_transform_state()
    assert owner_before["enabled"] == owner_after["enabled"]
    assert torch.equal(owner_before["reference"][0], owner_after["reference"][0])
    assert torch.equal(owner_before["current"][0], owner_after["current"][0])


def test_refined_anchor_bootstrap_never_calls_raw_point_growth(monkeypatch) -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    monkeypatch.setattr(
        mapper.map,
        "append_pfgs360_points",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("refined bootstrap called raw PFGS growth")
        ),
    )
    calls = []

    def update(**kwargs):
        calls.append(kwargs["event"])
        return {"candidate": 12, "selected": 12, "inserted": 10}

    engine = PFGS360FullBackend(
        mapper,
        {"growth_source": "refined_anchor"},
        refined_anchor_update=update,
    )
    output = engine._bootstrap(engine._observations((0, 1, 2, 3)), 0)
    assert calls == ["bootstrap"]
    assert output == {"raw": 12, "unique": 12, "occupied": 0, "inserted": 10}


def test_camera_stage_changes_pose_only_and_freezes_every_gaussian_parameter() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    before = {
        name: getattr(mapper.map, name).detach().clone()
        for name in mapper.map._gaussian_parameter_names()
    }
    engine = PFGS360FullBackend(mapper)
    metrics = engine._camera_stage(engine._observations((0, 1, 2, 3)), 2, 123)
    assert metrics["camera_steps"] == 2
    for name, expected in before.items():
        assert torch.equal(getattr(mapper.map, name).detach(), expected), name
    assert torch.equal(mapper.pose_deltas[0].delta.detach(), torch.zeros(6))
    assert any(
        float(mapper.pose_deltas[index].delta.detach().norm()) > 0.0
        for index in (1, 2, 3)
    )


def test_joint_stage_updates_all_six_gaussian_groups_and_keeps_first_pose_fixed() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    before = {
        name: getattr(mapper.map, name).detach().clone()
        for name in mapper.map._gaussian_parameter_names()
    }
    engine = PFGS360FullBackend(mapper, {"refine_every_joint_steps": 0})
    metrics = engine._joint_stage(engine._observations((0, 1, 2, 3)), 2, 124)
    assert metrics["joint_steps"] == 2
    for name, expected in before.items():
        assert not torch.equal(getattr(mapper.map, name).detach(), expected), name
    assert torch.equal(mapper.pose_deltas[0].delta.detach(), torch.zeros(6))
    assert any(
        float(mapper.pose_deltas[index].delta.detach().norm()) > 0.0
        for index in (1, 2, 3)
    )


def test_joint_stage_updates_only_sampled_view_visible_gaussian_rows() -> None:
    mapper = _registered_mapper(_PartiallyVisibleFakeRenderer())
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    before = {
        name: getattr(mapper.map, name).detach().clone()
        for name in mapper.map._gaussian_parameter_names()
    }
    mapper._pfgs360_gaussian_moments = {}
    for name in mapper.map._gaussian_parameter_names():
        parameter = getattr(mapper.map, name)
        mapper._pfgs360_gaussian_moments[name] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }

    engine = PFGS360FullBackend(
        mapper,
        {
            "refine_every_joint_steps": 0,
            "gaussian_update_scope": "sampled_view_visible",
            "visibility_threshold": 0.0,
            "sampling_policy": "uniform",
        },
    )
    metrics = engine._joint_stage(engine._observations((0, 1, 2, 3)), 2, 124)

    assert metrics["joint_gaussian_scope_sampled_view_visible"] == 1.0
    assert metrics["joint_visible_gaussians_min"] == 72.0
    assert metrics["joint_visible_gaussians_max"] == 72.0
    for name, expected in before.items():
        actual = getattr(mapper.map, name).detach()
        assert not torch.equal(actual[:72], expected[:72]), name
        assert torch.equal(actual[72:], expected[72:]), name
        state = mapper._pfgs360_gaussian_moments[name]
        assert bool((state["exp_avg"][72:] == 0).all()), name
        assert bool((state["exp_avg_sq"][72:] == 0).all()), name


def test_sampled_view_visible_scope_requires_accumulated_visibility() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    engine = PFGS360FullBackend(
        mapper,
        {"gaussian_update_scope": "sampled_view_visible"},
    )
    try:
        engine._gaussian_visibility_mask({})
    except RuntimeError as error:
        assert "accum_visible" in str(error)
    else:
        raise AssertionError("Visible-only updates must require rasterizer visibility")


def test_joint_stage_updates_all_render_contributors_and_freezes_noncontributors() -> None:
    mapper = _registered_mapper(_PartialContributorFakeRenderer())
    grid_x, grid_y = torch.meshgrid(
        torch.arange(12),
        torch.arange(12),
        indexing="ij",
    )
    xyz = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.ones(144)],
        dim=-1,
    ).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    before = {
        name: getattr(mapper.map, name).detach().clone()
        for name in mapper.map._gaussian_parameter_names()
    }
    mapper._pfgs360_gaussian_moments = {}
    for name in mapper.map._gaussian_parameter_names():
        parameter = getattr(mapper.map, name)
        mapper._pfgs360_gaussian_moments[name] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }

    engine = PFGS360FullBackend(
        mapper,
        {
            "refine_every_joint_steps": 0,
            "gaussian_update_scope": "all_render_contributors",
            "sampling_policy": "uniform",
        },
    )
    metrics = engine._joint_stage(
        engine._observations((0, 1, 2, 3)),
        2,
        124,
    )

    assert metrics["joint_gaussian_scope_all_render_contributors"] == 1.0
    assert metrics["joint_visible_gaussians_min"] == 100.0
    assert metrics["joint_visible_gaussians_max"] == 100.0
    assert metrics["joint_contributing_gaussians_mean"] == 100.0
    for name, expected in before.items():
        actual = getattr(mapper.map, name).detach()
        assert not torch.equal(actual[:100], expected[:100]), name
        assert torch.equal(actual[100:], expected[100:]), name
        state = mapper._pfgs360_gaussian_moments[name]
        assert bool((state["exp_avg"][100:] == 0).all()), name
        assert bool((state["exp_avg_sq"][100:] == 0).all()), name


def test_all_render_contributors_scope_does_not_require_accumulated_visibility() -> None:
    mapper = _registered_mapper(_NoAccumulatedVisibilityFakeRenderer())
    grid_x, grid_y = torch.meshgrid(
        torch.arange(12),
        torch.arange(12),
        indexing="ij",
    )
    xyz = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.ones(144)],
        dim=-1,
    ).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    engine = PFGS360FullBackend(
        mapper,
        {
            "gaussian_update_scope": "all_render_contributors",
            "refine_every_joint_steps": 0,
        },
    )
    metrics = engine._joint_stage(
        engine._observations((0, 1, 2, 3)),
        1,
        124,
    )
    assert metrics["joint_contributing_gaussians_mean"] == 144.0


def test_refined_anchor_joint_path_never_runs_conventional_topology(monkeypatch) -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    monkeypatch.setattr(
        mapper.map,
        "pfgs360_refine_topology",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("refined-anchor path called conventional topology")
        ),
    )
    engine = PFGS360FullBackend(
        mapper,
        {
            "growth_source": "refined_anchor",
            "topology_refine_enabled": False,
            "refine_every_joint_steps": 1,
        },
        refined_anchor_update=lambda **kwargs: {},
    )
    metrics = engine._joint_stage(engine._observations((0, 1, 2, 3)), 2, 124)
    assert metrics["joint_steps"] == 2
    assert not any(key.startswith("refine_") for key in metrics)


def test_pfgs360_adam_moments_follow_append_and_prune_row_mapping() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    engine = PFGS360FullBackend(mapper)
    mapper._pfgs360_gaussian_moments = {}
    for name in mapper.map._gaussian_parameter_names():
        parameter = getattr(mapper.map, name)
        values = torch.arange(144, dtype=parameter.dtype).view(144, *([1] * (parameter.ndim - 1)))
        values = values.expand_as(parameter).clone()
        mapper._pfgs360_gaussian_moments[name] = {
            "step": torch.tensor(3.0),
            "exp_avg": values,
            "exp_avg_sq": values + 1000.0,
        }

    new_xyz = xyz + torch.tensor([10.0, 0.0, 0.0])
    inserted = mapper.map.append_pfgs360_points(
        new_xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=1,
        min_unique_voxels=100,
    )["inserted"]
    assert inserted == 144
    engine._remap_moments()
    for name, state in mapper._pfgs360_gaussian_moments.items():
        assert int(state["exp_avg"].shape[0]) == 288
        expected = torch.arange(144, dtype=state["exp_avg"].dtype).view(
            144, *([1] * (state["exp_avg"].ndim - 1))
        ).expand_as(state["exp_avg"][:144])
        assert torch.equal(state["exp_avg"][:144], expected)
        assert bool((state["exp_avg"][144:] == 0).all()), name

    prune = torch.zeros(288, dtype=torch.bool)
    prune[::2] = True
    mapper.map.prune_anchors(prune)
    engine._remap_moments()
    assert mapper.map.anchor_count() == 144
    for state in mapper._pfgs360_gaussian_moments.values():
        assert int(state["exp_avg"].shape[0]) == 144


def test_pfgs360_state_storage_uses_requested_map_device_and_rejects_unknown_mode() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    engine = PFGS360FullBackend(mapper, {"state_storage_device": "map"})
    assert engine._state_storage_device() == mapper.map.xyz.device
    snapshot = mapper.map.pfgs360_topology_snapshot(
        parameter_device=engine._state_storage_device()
    )
    assert all(
        value.device == mapper.map.xyz.device
        for value in snapshot["parameters"].values()
    )

    invalid = PFGS360FullBackend(mapper, {"state_storage_device": "elsewhere"})
    try:
        invalid._state_storage_device()
    except ValueError as error:
        assert "state_storage_device" in str(error)
    else:
        raise AssertionError("Unknown PFGS360 state storage must fail explicitly")


def test_conventional_refine_splits_large_and_duplicates_small_gaussians() -> None:
    gaussian_map = PanoGaussianMap(config=_config(), device="cpu")
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    gaussian_map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    with torch.no_grad():
        gaussian_map.scaling[0].fill_(torch.log(torch.tensor(0.02)))
        gaussian_map.scaling[1].fill_(torch.log(torch.tensor(0.005)))
    gradients = torch.zeros(144)
    gradients[:2] = 1.0
    result = gaussian_map.pfgs360_refine_topology(
        gradients,
        torch.zeros(144),
        grad_threshold=0.5,
        split_scale_threshold=0.01,
        split_samples=2,
    )
    assert result["split"] == 1
    assert result["split_children"] == 2
    assert result["duplicate"] == 1
    assert result["culled"] == 0
    assert gaussian_map.anchor_count() == 146
    for name in gaussian_map._anchor_metadata_names():
        assert int(getattr(gaussian_map, name).shape[0]) == 146


def test_pfgs360_checkpoint_contains_every_row_aligned_metadata_field(tmp_path: Path) -> None:
    gaussian_map = PanoGaussianMap(config=_config(), device="cpu")
    grid_x, grid_y = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(144)], dim=-1).float() * 0.02
    gaussian_map.append_pfgs360_points(
        xyz,
        torch.rand(144, 3),
        owner_window_id=7,
        frame_id=11,
        min_unique_voxels=100,
    )
    path = tmp_path / "map.pt"
    gaussian_map.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["pfgs360_metadata"]
    assert set(metadata) == set(gaussian_map._anchor_metadata_names())
    for name in gaussian_map._anchor_metadata_names():
        assert torch.equal(metadata[name], getattr(gaussian_map, name))


def test_invalid_complete_geometry_snapshot_does_not_partially_rebase_poses() -> None:
    mapper = _registered_mapper(_DifferentiableFakeRenderer())
    mapper.map.config["SphericalSelfiGlobalBackend"]["map_optimization"] = {
        "strategy": "pfgs360_full_50_50"
    }
    with torch.no_grad():
        mapper.pose_deltas[0].delta[3] = 0.25
    base_before = mapper.pose_deltas[0].base_c2w.detach().clone()

    class Update:
        def __init__(self, pose, owner=0, scale=1.0):
            self.pose_c2w = pose
            self.owner_window_id = owner
            self.depth_owner_window_id = owner
            self.depth_scale = scale
            self.depth_scales_by_window = {owner: scale}

    valid = torch.eye(4)
    valid[1, 3] = 3.0
    invalid = torch.full((4, 4), float("nan"))
    try:
        mapper.apply_frontend_geometry_snapshot(
            {0: Update(valid), 1: Update(invalid)}, revision=2
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid geometry snapshot must be rejected")
    assert torch.equal(mapper.pose_deltas[0].base_c2w.detach(), base_before)
    assert torch.equal(mapper.pose_deltas[0].delta.detach()[3], torch.tensor(0.25))


def test_dia_applies_official_100_threshold_without_deletion_cap() -> None:
    mapper = _registered_mapper(_AttributedFakeRenderer())
    xyz = torch.stack(
        [torch.arange(240), torch.zeros(240), torch.ones(240)], dim=-1
    ).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(240, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    with torch.no_grad():
        mapper.map.opacity_logit.fill_(0.0)
    engine = PFGS360FullBackend(
        mapper,
        {"min_reset_gaussians": 100, "min_delete_gaussians": 100},
    )
    metrics = engine._dia(engine._observations((0, 1, 2, 3)), (), 0)
    assert metrics["dia_reset_applied"] == 120
    assert metrics["dia_deleted"] == 120
    assert mapper.map.anchor_count() == 120
    assert bool((mapper.map.get_opacity <= 0.010001).all())


def test_refined_anchor_growth_runs_only_after_dia_old_map_deletion() -> None:
    mapper = _registered_mapper(_AttributedFakeRenderer())
    xyz = torch.stack(
        [torch.arange(240), torch.zeros(240), torch.ones(240)], dim=-1
    ).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(240, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    calls = []

    def update(**kwargs):
        calls.append(
            {
                "event": kwargs["event"],
                "count": mapper.map.anchor_count(),
                "visibility": {
                    key: int(value.numel())
                    for key, value in kwargs["existing_anchor_visibility"].items()
                },
            }
        )
        return {"candidate": 0, "selected": 0, "inserted": 0}

    engine = PFGS360FullBackend(
        mapper,
        {
            "growth_source": "refined_anchor",
            "validity_gate": "pfgs360_official_sky_only",
            "min_reset_gaussians": 100,
            "min_delete_gaussians": 100,
        },
        refined_anchor_update=update,
    )
    metrics = engine._dia(engine._observations((0, 1, 2, 3)), (2, 3), 1)
    assert metrics["dia_deleted"] == 120
    assert calls == [
        {
            "event": "growth",
            "count": 120,
            "visibility": {2: 120, 3: 120},
        }
    ]


def test_query_failure_rolls_back_bootstrap_pose_and_topology() -> None:
    mapper = _registered_mapper(_AttributedFakeRenderer(fail_query=True))
    deltas_before = {
        frame_id: pose.delta.detach().clone()
        for frame_id, pose in mapper.pose_deltas.items()
    }
    try:
        mapper.optimize_pfgs360_full_50_50(
            window_id=0,
            frame_ids=(0, 1, 2, 3),
            new_frame_ids=(0, 1, 2, 3),
            settings={
                "camera_steps": 1,
                "joint_steps": 1,
                "min_unique_growth_voxels": 1,
                "min_raw_growth_points": 1,
            },
        )
    except RuntimeError as error:
        assert "query_answers" in str(error)
    else:
        raise AssertionError("Missing query attribution must abort the transaction")
    assert mapper.map.anchor_count() == 0
    for frame_id, expected in deltas_before.items():
        assert torch.equal(mapper.pose_deltas[frame_id].delta.detach(), expected)


def test_refined_anchor_insertion_failure_rolls_back_dia_and_optimizer_state() -> None:
    mapper = _registered_mapper(_AttributedFakeRenderer())
    xyz = torch.stack(
        [torch.arange(240), torch.zeros(240), torch.ones(240)], dim=-1
    ).float() * 0.02
    mapper.map.append_pfgs360_points(
        xyz,
        torch.rand(240, 3),
        owner_window_id=0,
        frame_id=0,
        min_unique_voxels=100,
    )
    before = mapper.map.pfgs360_topology_snapshot()
    mapper._pfgs360_gaussian_moments = {
        "xyz": {
            "step": torch.tensor(2.0),
            "exp_avg": torch.ones_like(mapper.map.xyz),
            "exp_avg_sq": torch.full_like(mapper.map.xyz, 2.0),
        }
    }
    moments_before = {
        key: {
            field: value.detach().clone() if torch.is_tensor(value) else value
            for field, value in state.items()
        }
        for key, state in mapper._pfgs360_gaussian_moments.items()
    }

    engine = PFGS360FullBackend(
        mapper,
        {
            "growth_source": "refined_anchor",
            "validity_gate": "pfgs360_official_sky_only",
            "topology_refine_enabled": False,
            "min_reset_gaussians": 100,
            "min_delete_gaussians": 100,
        },
        refined_anchor_update=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic refined insertion failure")
        ),
    )
    try:
        engine.run(
            frame_ids=(0, 1, 2, 3),
            new_frame_ids=(2, 3),
            owner_window_id=1,
            camera_steps=0,
            joint_steps=0,
        )
    except RuntimeError as error:
        assert "synthetic refined insertion failure" in str(error)
    else:
        raise AssertionError("Refined insertion failure must abort the transaction")

    assert mapper.map.anchor_count() == 240
    assert torch.equal(mapper.map.xyz.detach().cpu(), before["parameters"]["xyz"])
    for key, state in moments_before.items():
        for field, value in state.items():
            actual = mapper._pfgs360_gaussian_moments[key][field]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value


def test_pose_gradient_clipping_skips_parameters_without_gradients() -> None:
    without_gradient = torch.nn.Parameter(torch.tensor([1.0]))
    with_gradient = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    with_gradient.grad = torch.tensor([2.0, -3.0])

    PFGS360FullBackend._clip_pose_gradients(
        [without_gradient, with_gradient],
        0.5,
    )
    torch.testing.assert_close(with_gradient.grad, torch.tensor([0.5, -0.5]))

    with_gradient.grad = None
    PFGS360FullBackend._clip_pose_gradients(
        [without_gradient, with_gradient],
        0.5,
    )


def test_formal_config_is_sphereglue_pointmap_sim3_and_strict_pfgs360() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/spherical_selfi_ob3d_pointmap_sim3_sphereglue_ba_100_pfgs360_full_50_50.yaml"
    )
    config = load_config(path)
    backend = config["SphericalSelfiGlobalBackend"]
    assert backend["rendered_overlap_alignment"]["mode"] == "two_frame_pointmap_full_sim3"
    assert backend["rendered_overlap_alignment"]["acceptance_policy"] == "diagnostics_only"
    assert backend["global_graph"]["node_mode"] == "chunk_first_stride"
    assert backend["map_optimization"]["strategy"] == "pfgs360_full_50_50"
    assert backend["map_optimization"]["camera_steps"] == 50
    assert backend["map_optimization"]["joint_steps"] == 50
    assert not backend["insertion_dedup"]["enabled"]
    assert not backend["insertion_depth_gate"]["enabled"]
    assert not backend["error_gaussian_prune"]["enabled"]
    assert config["VoxelAnchorRefiner"]["enabled"] is True
    assert config["SphericalSelfiRuntime"]["local_ba"]["matching"]["type"] == "superpoint_sphereglue"
    assert config["WeightsAndBiases"]["runtime_log_preset"] == "slam_core_visuals"
    assert config["Training"]["pfgs360_distloss"] is True
    assert config["BackendFeedback"]["enabled"] is False
    assert not _requires_refiner_insertion_dedup(backend)
    legacy_backend = {**backend, "map_optimization": {"strategy": "legacy"}}
    assert _requires_refiner_insertion_dedup(legacy_backend)


def test_refined_anchor_formal_config_keeps_pointmap_mainline_and_disables_topology() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/spherical_selfi_ob3d_pointmap_sim3_sphereglue_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )
    config = load_config(path)
    backend = config["SphericalSelfiGlobalBackend"]
    optimize = backend["map_optimization"]
    pfgs = optimize["pfgs360"]
    assert backend["rendered_overlap_alignment"]["mode"] == "two_frame_pointmap_full_sim3"
    assert backend["rendered_overlap_alignment"]["acceptance_policy"] == "diagnostics_only"
    assert backend["global_graph"]["node_mode"] == "chunk_first_stride"
    assert optimize["camera_steps"] == optimize["joint_steps"] == 50
    assert pfgs["growth_source"] == "refined_anchor"
    assert pfgs["bootstrap_source"] == "refined_anchor_all_views"
    assert pfgs["growth_frame_policy"] == "chunk_new_frames"
    assert pfgs["validity_gate"] == "pfgs360_official_sky_only"
    assert pfgs["topology_refine_enabled"] is False
    assert pfgs["state_storage_device"] == "map"
    assert "refine_every_joint_steps" not in pfgs
    assert "split_scale_threshold" not in pfgs
    assert "cull_opacity" not in pfgs
    assert backend["insertion_dedup"]["enabled"] is True
    assert backend["insertion_dedup"]["radius_voxels"] == 1.0
    assert config["Training"]["pfgs360_absgrad"] is False
    assert config["WeightsAndBiases"]["runtime_log_preset"] == "slam_core_visuals"
    assert _resolve_cpu_threading_config(config) == {
        "intraop_threads": 8,
        "interop_threads": 2,
        "native_threads": 8,
        "opencv_threads": 2,
    }
    assert "backend/pfgs360_new_anchor_admission" in _SLAM_CORE_VISUAL_WANDB_KEYS
    assert not any(
        key.startswith("backend/pfgs360_new_anchor_admission_view_")
        for key in _SLAM_CORE_VISUAL_WANDB_KEYS
    )
    runtime_backend = dict(backend)
    runtime_backend["_voxel_anchor_refiner_enabled"] = True
    validated = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config=config, device="cpu"),
        renderer=object(),
        config=runtime_backend,
    )
    assert validated.insertion_dedup_radius_voxels == 1.0


def test_recent_three_chunk_50_200_configs_have_confirmed_scope_and_learning_rates() -> None:
    config_root = Path(__file__).parents[1] / "configs"
    pointmap = load_config(
        config_root
        / "spherical_selfi_ob3d_pointmap_sim3_sphereglue_ba_100_pfgs360_refined_anchor_50_200.yaml"
    )
    pager = load_config(
        config_root
        / "spherical_selfi_ob3d_pointmap_sim3_sphereglue_pager_ba_100_pfgs360_refined_anchor_50_200.yaml"
    )

    for config in (pointmap, pager):
        backend = config["SphericalSelfiGlobalBackend"]
        optimize = backend["map_optimization"]
        pfgs = optimize["pfgs360"]
        assert backend["rendered_overlap_alignment"]["mode"] == (
            "two_frame_pointmap_full_sim3"
        )
        assert optimize["steps_per_window"] == 250
        assert optimize["recent_window_count"] == 3
        assert optimize["camera_steps"] == 50
        assert optimize["joint_steps"] == 200
        assert optimize["sample_observations_per_step"] == 1
        assert optimize["pose_lr"] == 1.0e-4
        assert optimize["joint_pose_lr"] == 2.0e-5
        assert pfgs["frame_scope"] == "recent_chunks"
        assert pfgs["sampling_policy"] == "uniform"
        assert pfgs["visibility_threshold"] == 0.0
        assert pfgs["topology_refine_enabled"] is False
        assert config["WeightsAndBiases"]["runtime_log_preset"] == (
            "slam_core_visuals"
        )
    pointmap_optimization = pointmap["SphericalSelfiGlobalBackend"][
        "map_optimization"
    ]
    assert pointmap_optimization["optimize_all_gaussians"] is False
    assert pointmap_optimization["pfgs360"]["gaussian_update_scope"] == (
        "sampled_view_visible"
    )
    pager_optimization = pager["SphericalSelfiGlobalBackend"]["map_optimization"]
    assert pager_optimization["optimize_all_gaussians"] is True
    assert pager_optimization["pfgs360"]["gaussian_update_scope"] == (
        "all_render_contributors"
    )
    assert pager["SphericalSelfiRuntime"]["pager_depth"]["enabled"] is True


def test_refined_anchor_global_map_config_is_a_strict_alignment_ablation() -> None:
    config_root = Path(__file__).parents[1] / "configs"
    pointmap = load_config(
        config_root
        / "spherical_selfi_ob3d_pointmap_sim3_sphereglue_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )
    global_map = load_config(
        config_root
        / "spherical_selfi_ob3d_global_map_sim3_sphereglue_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )

    assert pointmap["SphericalSelfiGlobalBackend"]["rendered_overlap_alignment"][
        "mode"
    ] == "two_frame_pointmap_full_sim3"
    assert global_map["SphericalSelfiGlobalBackend"]["rendered_overlap_alignment"][
        "mode"
    ] == "two_frame_global_map_full_sim3"

    for section in (
        "Dataset",
        "SphericalSelfiRuntime",
        "Runtime",
        "VoxelAnchorRefiner",
        "MapRepresentation",
        "BackendOptimization",
        "Mapping",
        "Training",
        "Renderer",
        "TrajectoryEvaluation",
    ):
        assert global_map.get(section) == pointmap.get(section)

    pointmap_backend = dict(pointmap["SphericalSelfiGlobalBackend"])
    global_backend = dict(global_map["SphericalSelfiGlobalBackend"])
    pointmap_alignment = dict(pointmap_backend.pop("rendered_overlap_alignment"))
    global_alignment = dict(global_backend.pop("rendered_overlap_alignment"))
    pointmap_alignment.pop("mode")
    global_alignment.pop("mode")
    assert global_backend == pointmap_backend
    assert global_alignment == pointmap_alignment
    assert global_map["WeightsAndBiases"]["runtime_log_preset"] == "slam_core_visuals"
    runtime_backend = dict(global_map["SphericalSelfiGlobalBackend"])
    runtime_backend["_voxel_anchor_refiner_enabled"] = True
    validated = SphericalSelfiGlobalBackend(
        PanoGaussianMap(config=global_map, device="cpu"),
        renderer=object(),
        config=runtime_backend,
    )
    assert validated.two_frame_global_map_full_sim3_enabled is True
    assert validated.map_optimization_strategy == "pfgs360_full_50_50"


def test_refined_anchor_admission_saves_two_local_views_and_one_chunk_panel(
    tmp_path: Path,
) -> None:
    logger = SlamRuntimeLogger(
        {
            "WeightsAndBiases": {"enabled": False, "mode": "disabled"},
            "Visualization": {"save_local": True},
        },
        tmp_path,
    )
    views = []
    for frame_id in (4, 5):
        mask = torch.zeros(1, 4, 8, dtype=torch.bool)
        mask[:, 1, 0] = True
        views.append(
            {
                "frame_id": frame_id,
                "image": torch.full((3, 4, 8), 0.5),
                "mono_inlier_mask": mask,
                "candidate_pixels": torch.tensor([8]),
                "dia_selected_pixels": torch.tensor([8]),
                "hash_rejected_pixels": torch.tensor([9]),
                "inserted_pixels": torch.tensor([10]),
            }
        )

    path = logger.observe_pfgs360_new_anchor_admission(
        {"window_id": 2, "image_size": (4, 8), "views": views},
        step=6,
    )

    assert path is not None and path.exists()
    directory = path.parent
    assert (directory / "window_000002_frame_000004.png").exists()
    assert (directory / "window_000002_frame_000005.png").exists()


def test_strict_pfgs360_refined_packet_does_not_require_legacy_dedup() -> None:
    backend = SphericalSelfiGlobalBackend.__new__(SphericalSelfiGlobalBackend)
    backend.voxel_anchor_refiner_enabled = True
    backend.pose_canonicalized_packet_refiner = object()
    backend.two_frame_known_pose_bridge_enabled = False
    backend.two_frame_chunk_full_sim3_enabled = True
    backend.rendered_overlap_alignment_enabled = True
    backend.insertion_dedup_enabled = False
    backend.map_optimization_strategy = "pfgs360_full_50_50"
    backend.renderer = object()
    packet = SimpleNamespace(
        metadata={
            "voxel_anchor_refiner_requested": True,
            "voxel_anchor_refiner_pending": True,
        },
        anchor_observation=object(),
    )
    assert backend._validate_refined_packet(packet) is True
