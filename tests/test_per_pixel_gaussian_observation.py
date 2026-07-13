from __future__ import annotations

import math

import torch

from backend.pano_gs.adapter import PanoRenderCamera
from geometry.spherical_erp import build_erp_ray_grid
from models.per_pixel_gaussian_observation import (
    PerPixelGaussianObservation,
    matrix_to_quaternion,
    quaternion_multiply,
    real_sh_basis,
)


def _observation(*, height: int = 4, width: int = 8, views: int = 2) -> PerPixelGaussianObservation:
    depth = torch.full((1, views, 1, height, width), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    quaternion = torch.zeros(1, views, 4, height, width)
    quaternion[:, :, 0] = 1.0
    density = torch.zeros(1, views, 4, height, width)
    density[:, :, 0] = math.log(0.2 / 0.8) / 0.28209479177387814
    rows = torch.arange(height, dtype=torch.float32) + 0.5
    cols = torch.arange(width, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return PerPixelGaussianObservation(
        initial_depth=depth,
        depth_residual=torch.zeros_like(depth),
        refined_depth=depth.clone(),
        poses_c2w=poses,
        local_quaternion=quaternion,
        log_scale_multiplier=torch.zeros(1, views, 3, height, width),
        rgb_sh=torch.zeros(1, views, 9, 3, height, width),
        density_sh=density,
        confidence=torch.full_like(depth, 0.2),
        valid_mask=torch.ones_like(depth, dtype=torch.bool),
        source_uv=torch.stack([xx, yy], dim=-1),
        source_ray=build_erp_ray_grid(height, width),
        frame_ids=torch.arange(views).view(1, views),
        max_scale_ratio=10.0,
        render_prune_fraction=0.30,
    )


def test_center_geometry_reacts_to_pose_and_depth_updates() -> None:
    observation = _observation(views=1)
    expected_camera = observation.source_ray * 2.0
    torch.testing.assert_close(observation.centers_camera()[0, 0], expected_camera)

    poses = observation.poses_c2w.clone()
    poses[0, 0, :3, 3] = torch.tensor([1.0, -2.0, 3.0])
    updated = observation.with_geometry(poses_c2w=poses, refined_depth=observation.refined_depth * 1.5)
    torch.testing.assert_close(
        updated.centers_world()[0, 0],
        observation.source_ray * 3.0 + torch.tensor([1.0, -2.0, 3.0]),
    )
    torch.testing.assert_close(updated.depth_residual, torch.ones_like(updated.depth_residual))


def test_scale_is_positive_depth_relative_and_latitude_aware() -> None:
    observation = _observation(views=1)
    scale = observation.scales()[0, 0]
    assert bool((scale > 0).all())
    equator = float(scale[0, scale.shape[1] // 2].mean())
    pole = float(scale[0, 0].mean())
    assert equator > pole
    doubled = observation.scales(depth=observation.refined_depth * 2.0)
    torch.testing.assert_close(doubled, scale.unsqueeze(0).unsqueeze(0) * 2.0)


def test_world_quaternion_composes_pose_and_local_rotation() -> None:
    observation = _observation(views=1)
    angle = math.pi / 2.0
    rotation = torch.tensor(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
    )
    poses = observation.poses_c2w.clone()
    poses[0, 0, :3, :3] = rotation
    observation = observation.with_geometry(poses_c2w=poses)
    camera = PanoRenderCamera(*observation.image_size, poses[0, 0])
    explicit = observation.materialize_batch(camera, batch_index=0, prune_fraction=0.0)
    expected = matrix_to_quaternion(rotation)
    torch.testing.assert_close(explicit.rotation, expected.view(1, 4).expand_as(explicit.rotation), atol=1e-5, rtol=1e-5)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(quaternion_multiply(expected, identity), expected)


def test_target_conditioned_sh_and_pruning_keep_canonical_dense_count() -> None:
    observation = _observation(views=2)
    # Make RGB depend on the local x direction and opacity vary by longitude.
    observation.rgb_sh[:, :, 3, 0] = 0.25
    observation.density_sh[:, :, 3] = observation.source_ray[..., 0]
    before = observation.canonical_count
    camera = PanoRenderCamera(*observation.image_size, observation.poses_c2w[0, 0])
    full = observation.materialize_batch(camera, batch_index=0, prune_fraction=0.0)
    pruned = observation.materialize_batch(camera, batch_index=0)
    assert full.xyz.shape[0] == before
    assert pruned.xyz.shape[0] == math.ceil(before * 0.7)
    assert observation.canonical_count == before
    assert float(full.features[:, 0].std()) > 0.0
    assert float(full.opacity.std()) > 0.0


def test_batched_materialization_shares_geometry_opacity_and_global_pruning() -> None:
    observation = _observation(views=2)
    observation.rgb_sh[:, :, 3, 0] = 0.25
    second_pose = observation.poses_c2w[0, 1].clone()
    second_pose[0, 3] = 0.5
    cameras = [
        PanoRenderCamera(*observation.image_size, observation.poses_c2w[0, 0]),
        PanoRenderCamera(*observation.image_size, second_pose),
    ]
    before = observation.canonical_count
    full = observation.materialize_batched(
        cameras,
        batch_index=0,
        source_indices=range(2),
        prune_fraction=0.0,
    )
    pruned = observation.materialize_batched(
        cameras,
        batch_index=0,
        source_indices=range(2),
    )
    assert full.xyz.shape == (before, 3)
    assert full.features.shape == (2, before, 3)
    assert pruned.xyz.shape[0] == math.ceil(before * 0.7)
    assert pruned.features.shape[1] == pruned.xyz.shape[0]
    torch.testing.assert_close(full.opacity, torch.full_like(full.opacity, 0.2))
    for target, camera in enumerate(cameras):
        single = observation.materialize_batch(
            camera,
            batch_index=0,
            source_indices=range(2),
            prune_fraction=0.0,
        )
        torch.testing.assert_close(full.xyz, single.xyz)
        torch.testing.assert_close(full.scaling, single.scaling)
        torch.testing.assert_close(full.rotation, single.rotation)
        torch.testing.assert_close(full.features[target], single.features)
    assert observation.canonical_count == before


def test_invalid_pixels_preserve_metadata_order_for_valid_entries() -> None:
    observation = _observation(views=1)
    observation.valid_mask[0, 0, 0, 0, 0] = False
    camera = PanoRenderCamera(*observation.image_size, observation.poses_c2w[0, 0])
    explicit = observation.materialize_batch(camera, batch_index=0, prune_fraction=0.0)
    assert explicit.xyz.shape[0] == observation.image_size[0] * observation.image_size[1] - 1
    torch.testing.assert_close(explicit.source_pixel_uv[0], observation.source_uv.reshape(-1, 2)[1])
    assert bool(torch.isfinite(explicit.xyz).all())


def test_real_sh_basis_has_expected_channel_counts() -> None:
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    assert real_sh_basis(0, direction).shape == (1, 1)
    assert real_sh_basis(1, direction).shape == (1, 4)
    assert real_sh_basis(2, direction).shape == (1, 9)
