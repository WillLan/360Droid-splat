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


def test_se3_exp_uses_strict_translation_jacobian():
    theta = torch.tensor(1.57079632679)
    xi = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, float(theta)]])
    T = se3_exp(xi)
    expected = torch.stack(
        [torch.sin(theta) / theta, (1.0 - torch.cos(theta)) / theta, torch.tensor(0.0)]
    )
    assert torch.allclose(T[0, :3, 3], expected, atol=1e-5)
    assert not torch.allclose(T[0, :3, 3], xi[0, :3])


def test_spherical_ba_damping_reduces_update_size():
    H, W = 16, 32
    pixels = torch.tensor([[[8.5, 8.5], [16.5, 8.5], [24.5, 8.5]]])
    inv = torch.full((1, 3), 0.4)
    target_delta = torch.tensor([[[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]])
    ba = SphericalBA(H, W)
    T_lo, _, _ = ba.optimize_pose_depth(
        pixels,
        inv,
        target_delta,
        torch.eye(4).unsqueeze(0),
        steps=1,
        lr=1e-2,
        optimize_depth=False,
        damping=0.0,
    )
    T_hi, _, _ = ba.optimize_pose_depth(
        pixels,
        inv,
        target_delta,
        torch.eye(4).unsqueeze(0),
        steps=1,
        lr=1e-2,
        optimize_depth=False,
        damping=100.0,
    )
    eye = torch.eye(4).unsqueeze(0)
    assert (T_hi - eye).abs().sum() < (T_lo - eye).abs().sum()
