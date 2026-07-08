import math

import torch

from geometry.spherical_erp import (
    build_erp_ray_grid,
    circular_pad_longitude,
    erp_pixel_to_unit_ray,
    safe_acos_dot,
    sample_erp_with_wrap,
    spherical_geodesic_distance,
    unit_ray_to_erp_pixel,
    wrap_longitude_pixel,
)


def _seam_delta(source: torch.Tensor, target: torch.Tensor, width: int) -> torch.Tensor:
    delta = target - source
    du = torch.remainder(delta[..., 0] + float(width) * 0.5, float(width)) - float(width) * 0.5
    return torch.stack([du, delta[..., 1]], dim=-1)


def test_default_erp_ray_grid_shape_and_unit_norm():
    grid = build_erp_ray_grid()
    assert grid.shape == (504, 1008, 3)
    assert torch.allclose(torch.linalg.norm(grid, dim=-1), torch.ones(504, 1008), atol=1e-5)


def test_erp_pixel_ray_round_trip_with_horizontal_wrap():
    height, width = 64, 128
    pixel = torch.tensor(
        [[0.5, 0.5], [63.5, 31.5], [127.5, 63.5], [-0.5, 32.5], [128.5, 32.5]],
        dtype=torch.float32,
    )
    ray = erp_pixel_to_unit_ray(pixel, height, width)
    restored = unit_ray_to_erp_pixel(ray, height, width)
    expected = pixel.clone()
    expected[:, 0] = wrap_longitude_pixel(expected[:, 0], width)
    delta = _seam_delta(expected, restored, width)
    assert delta.abs().max() < 1e-4
    assert torch.allclose(torch.linalg.norm(ray, dim=-1), torch.ones(ray.shape[0]), atol=1e-6)


def test_seam_points_have_small_great_circle_distance():
    height, width = 504, 1008
    pixels = torch.tensor([[0.5, height / 2.0], [width - 0.5, height / 2.0]])
    rays = erp_pixel_to_unit_ray(pixels, height, width)
    distance = spherical_geodesic_distance(rays[0], rays[1])
    assert distance < 2.0 * math.pi / float(width) + 1e-5


def test_horizontal_wrap_and_vertical_clamp_when_sampling():
    height, width = 4, 8
    columns = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width)
    rows = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width)
    feature = torch.cat([columns, rows], dim=1)
    uv = torch.tensor(
        [
            [-0.5, 1.5],
            [width - 0.5, 1.5],
            [0.5, -20.0],
            [0.5, height + 20.0],
        ],
        dtype=torch.float32,
    )
    sampled = sample_erp_with_wrap(feature, uv, mode="nearest")[0]
    assert torch.allclose(sampled[0], sampled[1])
    assert sampled[2, 1].item() == 0.0
    assert sampled[3, 1].item() == float(height - 1)


def test_great_circle_distance_same_and_orthogonal_rays():
    ray_a = torch.tensor([[1.0, 0.0, 0.0]])
    ray_b = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(safe_acos_dot(ray_a, ray_a), torch.zeros(1), atol=1e-6)
    assert torch.allclose(spherical_geodesic_distance(ray_a, ray_b), torch.tensor([math.pi / 2.0]), atol=1e-6)


def test_circular_pad_longitude_only_pads_last_dimension():
    x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    padded = circular_pad_longitude(x, 1)
    assert torch.allclose(padded, torch.tensor([[[[3.0, 1.0, 2.0, 3.0, 1.0]]]]))
