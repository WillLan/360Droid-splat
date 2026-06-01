import torch

from frontend.pano_droid.spherical_ba import SphericalBA, se3_exp, spherical_ba_loss
from frontend.pano_droid.spherical_camera import pixel_grid, seam_aware_delta
from frontend.pano_droid.projective_ops import spherical_reprojection_residual
from frontend.pano_droid.dense_ba import (
    SphericalDenseBA,
    _left_pose_jacobians,
    _solve_damped_normal_system,
)
from frontend.pano_droid.projective_ops import project_edges


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


def test_spherical_ba_pixel_residual_uses_shared_projection():
    H, W = 16, 32
    pixels = pixel_grid(H, W).reshape(-1, 2)[::41].unsqueeze(0)
    inv = torch.full((1, pixels.shape[1]), 0.5)
    target_delta = torch.full((1, pixels.shape[1], 2), 0.25)
    T = torch.eye(4).unsqueeze(0)
    out = spherical_ba_loss(
        pixels,
        inv,
        T,
        height=H,
        width=W,
        target_delta=target_delta,
        residual_mode="pixel",
    )
    residual, _, _ = spherical_reprojection_residual(
        pixels,
        inv,
        T,
        height=H,
        width=W,
        target_delta=target_delta,
        residual_mode="pixel",
    )
    assert torch.allclose(out.residual, residual, atol=1e-6)


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


def test_spherical_dense_ba_zero_residual_and_fixed_frame():
    H, W = 8, 16
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    inv = torch.full((1, 2, 1, H, W), 0.4)
    ii = torch.tensor([0])
    jj = torch.tensor([1])
    target = project_edges(poses, inv, ii, jj, height=H, width=W)
    weight = torch.ones(1, 1, 2, H, W)
    eta = torch.zeros(1, 2, 1, H, W)
    out = SphericalDenseBA()(poses, inv, target, weight, eta, ii, jj, fixed_frames=1, iters=1)
    assert out.residual.abs().max() < 1e-4
    assert torch.allclose(out.poses_c2w[:, 0], poses[:, 0], atol=1e-6)


def test_spherical_dense_ba_damping_reduces_update_norm():
    H, W = 8, 16
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    inv = torch.full((1, 2, 1, H, W), 0.4)
    ii = torch.tensor([0])
    jj = torch.tensor([1])
    target = project_edges(poses, inv, ii, jj, height=H, width=W)
    target = target.clone()
    target[..., 0] = target[..., 0] + 0.05
    weight = torch.ones(1, 1, 2, H, W)
    lo = SphericalDenseBA()(poses, inv, target, weight, torch.zeros(1, 2, 1, H, W), ii, jj, fixed_frames=1, iters=1)
    hi = SphericalDenseBA()(poses, inv, target, weight, torch.full((1, 2, 1, H, W), 100.0), ii, jj, fixed_frames=1, iters=1)
    assert hi.pose_update_norm < lo.pose_update_norm


def test_spherical_dense_ba_solver_handles_singular_system_without_svd():
    system = torch.zeros(1, 6, 6)
    rhs = torch.ones(1, 6, 1)
    solution = _solve_damped_normal_system(system, rhs, base_jitter=1e-6)
    assert solution.shape == rhs.shape
    assert torch.isfinite(solution).all()


def test_spherical_dense_ba_pose_jacobian_matches_finite_difference():
    H, W = 8, 16
    poses = torch.eye(4, dtype=torch.float64).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    poses[:, 1, 0, 3] = 0.1
    inv = torch.full((1, 2, 1, H, W), 0.4, dtype=torch.float64)
    ii = torch.tensor([0])
    jj = torch.tensor([1])
    coords, _, j_tgt, _ = _left_pose_jacobians(poses, inv, ii, jj, height=H, width=W, stride=1)
    eps = 1e-5
    xi = torch.zeros(1, 2, 6, dtype=torch.float64)
    xi[:, 1, 0] = eps
    coords_eps = project_edges(se3_exp(xi) @ poses, inv, ii, jj, height=H, width=W)
    finite = seam_aware_delta(coords, coords_eps, W) / eps
    analytic = j_tgt[..., 0]
    assert torch.allclose(analytic, finite, atol=5e-3, rtol=5e-3)
