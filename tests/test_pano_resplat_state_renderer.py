from __future__ import annotations

import torch

from frontend.pano_vggt.pano_resplat_geometry import (
    erp_pixel_grid,
    erp_uv_to_bearing,
    project_world_to_erp_grid,
)
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.resplat_types import PanoGaussianState, state_to_explicit_gaussian_set


def _state(batch: int = 1, count: int = 5, dtype: torch.dtype = torch.float32) -> PanoGaussianState:
    means = torch.zeros(batch, count, 3, dtype=dtype)
    means[..., 2] = 2.0
    means[..., 0] = torch.linspace(-0.2, 0.2, steps=count, dtype=dtype).view(1, count)
    log_scales = torch.full((batch, count, 3), -3.5, dtype=dtype)
    rotations = torch.randn(batch, count, 4, dtype=dtype)
    opacity = torch.zeros(batch, count, 1, dtype=dtype)
    sh = torch.zeros(batch, count, 3, 1, dtype=dtype)
    sh[..., 0] = 0.25
    latent = torch.randn(batch, count, 8, dtype=dtype)
    source_view_ids = torch.zeros(batch, count, dtype=torch.long)
    source_uv = torch.zeros(batch, count, 2, dtype=dtype)
    source_uv[..., 0] = torch.arange(count, dtype=dtype).view(1, count)
    valid = torch.ones(batch, count, dtype=torch.bool)
    confidence = torch.ones(batch, count, 1, dtype=dtype)
    return PanoGaussianState(
        means=means,
        log_scales=log_scales,
        rotations_unnorm=rotations,
        opacity_logits=opacity,
        sh_coeffs=sh,
        latent_features=latent,
        source_view_ids=source_view_ids,
        source_uv=source_uv,
        valid_mask=valid,
        confidence=confidence,
    )


def test_state_to_explicit_gaussian_set_shapes_and_quaternion_normalization():
    state = _state(count=7)

    explicit = state_to_explicit_gaussian_set(state, 0)

    assert explicit.get_xyz.shape == (7, 3)
    assert explicit.get_scaling.shape == (7, 3)
    assert explicit.get_rotation.shape == (7, 4)
    assert explicit.get_opacity.shape == (7, 1)
    assert explicit.get_features.shape == (7, 3)
    assert explicit.get_sh_coefficients.shape == (7, 1, 3)
    assert torch.allclose(torch.linalg.norm(explicit.get_rotation, dim=-1), torch.ones(7), atol=1.0e-5)


def test_erp_bearing_is_unit_length():
    grid = erp_pixel_grid((8, 16), dtype=torch.float64)

    bearing = erp_uv_to_bearing(grid, (8, 16))

    assert bearing.dtype == grid.dtype
    assert torch.allclose(torch.linalg.norm(bearing, dim=-1), torch.ones(8, 16, dtype=torch.float64), atol=1.0e-6)


def test_project_world_to_erp_grid_range_and_invalid_mask():
    points = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.2, 0.1, 3.0],
            [0.0, 0.0, -1.0],
            [float("nan"), 0.0, 1.0],
        ]
    )
    c2w = torch.eye(4)

    projection = project_world_to_erp_grid(points, c2w, (10, 20))

    assert projection.grid.shape == (4, 2)
    assert bool((projection.grid >= -1.0).all())
    assert bool((projection.grid <= 1.0).all())
    assert projection.mask.tolist() == [True, True, False, False]


def test_soft_splat_smoke_render_preserves_shape_device_dtype():
    state = _state(batch=2, count=6, dtype=torch.float64)
    poses = torch.eye(4, dtype=torch.float64).view(1, 4, 4).repeat(2, 1, 1)
    adapter = PanoGaussianRendererAdapter(soft_max_points=32)

    output = adapter.render_state(state, poses, (12, 24), renderer_backend="soft_splat")

    assert output.color.shape == (2, 3, 12, 24)
    assert output.depth.shape == (2, 1, 12, 24)
    assert output.alpha.shape == (2, 1, 12, 24)
    assert output.color.dtype == torch.float64
    assert output.color.device == state.means.device
    assert output.extras["backend"] == ["soft_splat", "soft_splat"]
    assert torch.isfinite(output.color).all()
