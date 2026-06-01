import torch

from frontend.pano_droid.spherical_ba import SphericalBA, se3_exp, spherical_ba_loss
from frontend.pano_droid.spherical_camera import pixel_grid


def test_spherical_ba_residual_and_gradient_are_finite():
    H, W = 16, 32
    pixels = pixel_grid(H, W).reshape(-1, 2)[::32].unsqueeze(0)
    inv = torch.full((1, pixels.shape[1]), 0.5, requires_grad=True)
    xi = torch.zeros(1, 6, requires_grad=True)
    out = spherical_ba_loss(
        pixels,
        inv,
        se3_exp(xi),
        height=H,
        width=W,
        target_delta=torch.zeros(1, pixels.shape[1], 2),
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert torch.isfinite(inv.grad).all()
    assert torch.isfinite(xi.grad).all()


def test_spherical_ba_smoke_optimization_and_seam_stability():
    H, W = 16, 32
    pixels = torch.tensor([[[31.5, 8.5], [0.5, 8.5], [15.5, 4.5]]])
    inv = torch.full((1, 3), 0.4)
    target_delta = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.2, -0.1]]])
    ba = SphericalBA(H, W)
    T, inv_out, losses = ba.optimize_pose_depth(
        pixels,
        inv,
        target_delta,
        torch.eye(4).unsqueeze(0),
        steps=1,
        lr=1e-3,
    )
    assert T.shape == (1, 4, 4)
    assert inv_out.shape == inv.shape
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses)).all()

