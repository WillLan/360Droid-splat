import math

import torch

from geometry.spherical_pseudo_correspondence import (
    JointValidFibonacciSampleSet,
    SphericalCorrespondence,
    generate_spherical_pseudo_correspondence,
    sample_joint_valid_fibonacci_uv,
)


def _eye_poses(num_views: int) -> torch.Tensor:
    return torch.eye(4).view(1, 4, 4).repeat(num_views, 1, 1)


def _rotation_y(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    rot = torch.eye(4)
    rot[:3, :3] = torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.float32,
    )
    return rot


def test_identity_pose_correspondence_projects_to_self():
    height, width = 8, 16
    depth = torch.full((2, 1, height, width), 2.0)
    query = torch.tensor([[1.5, 3.5], [8.5, 4.5], [14.5, 6.5]], dtype=torch.float32)
    corr = generate_spherical_pseudo_correspondence(
        depth,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        query_uv=query,
        height=height,
        width=width,
        visibility_rel_thresh=0.01,
    )
    assert isinstance(corr, SphericalCorrespondence)
    assert corr.valid_mask.shape == (1, 3)
    assert corr.valid_mask.all()
    assert torch.allclose(corr.src_uv[0], query)
    assert torch.allclose(corr.tgt_uv[0], query, atol=1e-4)


def test_yaw_rotation_wraps_target_longitude():
    height, width = 8, 16
    depth = torch.full((2, 1, height, width), 2.0)
    poses = _eye_poses(2)
    poses[1] = _rotation_y(math.pi / 2.0)
    query = torch.tensor([[1.5, height / 2.0]], dtype=torch.float32)
    corr = generate_spherical_pseudo_correspondence(
        depth,
        poses,
        torch.tensor([[0, 1]]),
        query_uv=query,
        height=height,
        width=width,
        min_depth=0.01,
        max_depth=10.0,
    )
    expected_u = torch.remainder(query[:, 0] - width / 4.0, torch.tensor(float(width)))
    assert corr.valid_mask.all()
    assert torch.allclose(corr.tgt_uv[0, :, 0], expected_u, atol=1e-4)
    assert torch.allclose(corr.tgt_uv[0, :, 1], query[:, 1], atol=1e-4)


def test_depth_inconsistency_invalidates_visibility():
    height, width = 8, 16
    depth = torch.full((2, 1, height, width), 2.0)
    depth[1] = 9.0
    corr = generate_spherical_pseudo_correspondence(
        depth,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        query_uv=torch.tensor([[4.5, 4.5], [8.5, 4.5]], dtype=torch.float32),
        height=height,
        width=width,
        visibility_rel_thresh=0.05,
    )
    assert not corr.visibility.any()
    assert not corr.valid_mask.any()
    assert torch.all(corr.weight == 0.0)


def test_correspondence_fields_are_finite_and_keep_device_dtype_shape():
    height, width = 8, 16
    depth = torch.full((1, 2, 1, height, width), 2.0, dtype=torch.float64)
    poses = _eye_poses(2).unsqueeze(0).to(dtype=torch.float64)
    corr = generate_spherical_pseudo_correspondence(
        depth,
        poses,
        torch.tensor([[[0, 1]]]),
        query_uv=torch.tensor([[2.5, 2.5], [12.5, 5.5]], dtype=torch.float64),
        height=height,
        width=width,
    )
    assert corr.valid_mask.shape == (1, 1, 2)
    assert corr.src_view.shape == corr.valid_mask.shape
    assert corr.tgt_view.shape == corr.valid_mask.shape
    assert corr.src_uv.shape == (1, 1, 2, 2)
    assert corr.tgt_uv.shape == (1, 1, 2, 2)
    assert corr.src_ray.shape == (1, 1, 2, 3)
    assert corr.tgt_ray.shape == (1, 1, 2, 3)
    assert corr.src_uv.dtype == torch.float64
    assert corr.tgt_ray.dtype == torch.float64
    for value in (corr.src_uv, corr.tgt_uv, corr.src_ray, corr.tgt_ray, corr.weight):
        assert torch.isfinite(value).all()


def test_trailing_singleton_depth_channel_from_model_output():
    height, width = 8, 16
    depth = torch.full((1, 2, height, width, 1), 2.0)
    poses = _eye_poses(2).unsqueeze(0)
    query = torch.tensor([[3.5, 3.5], [10.5, 4.5]], dtype=torch.float32)

    corr = generate_spherical_pseudo_correspondence(
        depth,
        poses,
        torch.tensor([[[0, 1]]]),
        query_uv=query,
        height=height,
        width=width,
        visibility_rel_thresh=0.01,
    )

    assert corr.valid_mask.shape == (1, 1, 2)
    assert corr.valid_mask.all()
    assert torch.allclose(corr.src_uv[0, 0], query)
    assert torch.allclose(corr.tgt_uv[0, 0], query, atol=1e-4)


def test_fibonacci_sampling_prefers_equator_over_poles():
    height, width = 64, 128
    depth = torch.full((2, 1, height, width), 2.0)
    corr = generate_spherical_pseudo_correspondence(
        depth,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        height=height,
        width=width,
        num_query_per_pair=512,
        sampling="fibonacci_depth_filtered",
        fibonacci_oversample_factor=2,
        min_depth=0.01,
        max_depth=10.0,
        visibility_rel_thresh=0.01,
    )
    rows = corr.src_uv[0, :, 1]
    equator = ((rows >= height * 0.375) & (rows <= height * 0.625)).sum()
    poles = ((rows < height * 0.125) | (rows > height * 0.875)).sum()
    assert equator > poles
    assert corr.valid_mask.all()


def test_fibonacci_depth_filtered_sampling_prefers_finite_source_depth():
    height, width = 16, 32
    depth = torch.full((2, 1, height, width), 1000.0)
    depth[:, :, height // 2 :, :] = 2.0
    corr = generate_spherical_pseudo_correspondence(
        depth,
        _eye_poses(2),
        torch.tensor([[0, 1]]),
        height=height,
        width=width,
        num_query_per_pair=64,
        sampling="fibonacci_depth_filtered",
        fibonacci_oversample_factor=4,
        min_depth=0.01,
        max_depth=10.0,
        visibility_rel_thresh=0.01,
    )
    rows = corr.src_uv[0, :, 1]
    assert (rows >= float(height // 2)).float().mean() > 0.9
    assert corr.valid_mask.float().mean() > 0.9


def test_joint_fibonacci_uses_same_uv_and_never_fills_invalid_support():
    height, width = 17, 31
    source_depth = torch.full((1, height, width), 2.0)
    target_depth = torch.full_like(source_depth, 3.0)
    source_valid = torch.ones_like(source_depth, dtype=torch.bool)
    target_valid = torch.ones_like(source_valid)
    sky_source = torch.zeros_like(source_depth)
    sky_target = torch.zeros_like(target_depth)
    source_valid[:, : height // 2] = False
    target_valid[:, :, : width // 2] = False
    sky_target[:, height // 2 : height // 2 + 2] = 1.0
    first = sample_joint_valid_fibonacci_uv(
        source_depth,
        target_depth,
        count=128,
        oversample_factor=4,
        source_valid=source_valid,
        target_valid=target_valid,
        source_sky_probability=sky_source,
        target_sky_probability=sky_target,
        seed=19,
    )
    second = sample_joint_valid_fibonacci_uv(
        source_depth,
        target_depth,
        count=128,
        oversample_factor=4,
        source_valid=source_valid,
        target_valid=target_valid,
        source_sky_probability=sky_source,
        target_sky_probability=sky_target,
        seed=19,
    )
    assert isinstance(first, JointValidFibonacciSampleSet)
    torch.testing.assert_close(first.uv, second.uv)
    assert first.uv.shape[0] <= 128
    assert first.valid.all()
    torch.testing.assert_close(first.source_depth, torch.full_like(first.source_depth, 2.0))
    torch.testing.assert_close(first.target_depth, torch.full_like(first.target_depth, 3.0))

    empty = sample_joint_valid_fibonacci_uv(
        source_depth,
        target_depth,
        count=64,
        source_valid=torch.zeros_like(source_valid),
        target_valid=target_valid,
        seed=19,
    )
    assert empty.uv.shape == (0, 2)
    assert empty.valid.numel() == 0
