import torch

from frontend.pano_droid.spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    latitude_area_weight,
    seam_aware_delta,
    wrap_horizontal,
)


def test_erp_pixel_bearing_round_trip_y_down():
    H, W = 64, 128
    pixels = torch.tensor(
        [[0.5, 0.5], [63.5, 31.5], [127.5, 63.5], [12.25, 48.75]],
        dtype=torch.float32,
    )
    bearing = erp_pixel_to_bearing(pixels, H, W)
    restored = bearing_to_erp_pixel(bearing, H, W)
    delta = seam_aware_delta(pixels, restored, W)
    assert torch.max(delta.abs()) < 1e-4

    top = erp_pixel_to_bearing(torch.tensor([[W / 2, 0.5]]), H, W)
    bottom = erp_pixel_to_bearing(torch.tensor([[W / 2, H - 0.5]]), H, W)
    assert top[0, 1] < 0.0
    assert bottom[0, 1] > 0.0


def test_horizontal_wrap_and_seam_delta():
    u = torch.tensor([-1.0, 0.0, 127.0, 128.0, 129.0])
    assert torch.allclose(wrap_horizontal(u, 128), torch.tensor([127.0, 0.0, 127.0, 0.0, 1.0]))
    src = torch.tensor([[127.0, 5.0], [1.0, 5.0]])
    tgt = torch.tensor([[1.0, 6.0], [127.0, 4.0]])
    delta = seam_aware_delta(src, tgt, 128)
    assert torch.allclose(delta, torch.tensor([[2.0, 1.0], [-2.0, -1.0]]))


def test_latitude_area_weight_is_finite():
    weight = latitude_area_weight(32, 64)
    assert weight.shape == (1, 32, 64)
    assert torch.isfinite(weight).all()
    assert weight.min() >= 0.0
    assert torch.isclose(weight.mean(), torch.tensor(1.0), atol=1e-5)

