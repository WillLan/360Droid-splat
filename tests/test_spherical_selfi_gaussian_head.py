from __future__ import annotations

import torch

from models.spherical_selfi_gaussian_head import ERPConv2d, SphericalSelfiGaussianHead


def _inputs(height: int = 16, width: int = 32, views: int = 2):
    feature = torch.randn(1, views, 24, height, width)
    rgb = torch.rand(1, views, 3, height, width)
    depth = torch.rand(1, views, 1, height, width) + 1.0
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    return feature, rgb, depth, poses


def _tiny_head(**kwargs) -> SphericalSelfiGaussianHead:
    return SphericalSelfiGaussianHead(channels=(8, 16, 24, 32), mlp_hidden_dim=16, **kwargs)


def test_head_decodes_exact_39_channel_contract_and_safe_initial_values() -> None:
    head = _tiny_head()
    feature, rgb, depth, poses = _inputs()
    output = head(feature, rgb, depth, poses)
    assert head.raw_output_channels == 39
    assert output.depth_residual.shape == (1, 2, 1, 16, 32)
    assert output.local_quaternion.shape == (1, 2, 4, 16, 32)
    assert output.log_scale_multiplier.shape == (1, 2, 3, 16, 32)
    assert output.rgb_sh.shape == (1, 2, 9, 3, 16, 32)
    assert output.density_sh.shape == (1, 2, 4, 16, 32)
    torch.testing.assert_close(output.depth_residual, torch.zeros_like(output.depth_residual))
    torch.testing.assert_close(output.local_quaternion[:, :, 0], torch.ones_like(output.local_quaternion[:, :, 0]))
    assert bool(torch.isfinite(output.scales()).all()) and bool((output.scales() > 0).all())
    torch.testing.assert_close(output.confidence, torch.full_like(output.confidence, 0.1), atol=1e-6, rtol=1e-5)


def test_head_supports_odd_runtime_shape_and_bounds_depth_residual() -> None:
    head = _tiny_head()
    with torch.no_grad():
        head.depth_head.conv.bias.fill_(10.0)
    feature, rgb, depth, poses = _inputs(15, 31, views=1)
    output = head(feature, rgb, depth, poses)
    assert output.image_size == (15, 31)
    assert bool((output.depth_residual.abs() <= depth * 0.25 + 1e-6).all())
    assert bool((output.refined_depth > 0.0).all())


def test_head_backward_is_finite_and_frozen_inputs_receive_no_required_grad() -> None:
    head = _tiny_head()
    feature, rgb, depth, poses = _inputs(16, 32, views=1)
    output = head(feature, rgb, depth, poses)
    loss = output.confidence.mean() + output.rgb_sh.square().mean() + output.depth_residual.mean()
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert feature.grad is None and rgb.grad is None and depth.grad is None


def test_horizontal_roll_equivariance_for_stride_aligned_shift() -> None:
    torch.manual_seed(7)
    head = _tiny_head().eval()
    with torch.no_grad():
        for module in (head.quaternion_head.conv, head.depth_head.conv, head.scale_head, head.rgb_sh_head, head.density_sh_head):
            module.weight.normal_(std=0.01)
    feature, rgb, depth, poses = _inputs(16, 32, views=1)
    shift = 8
    original = head(feature, rgb, depth, poses)
    rolled = head(
        torch.roll(feature, shift, dims=-1),
        torch.roll(rgb, shift, dims=-1),
        torch.roll(depth, shift, dims=-1),
        poses,
    )
    torch.testing.assert_close(
        rolled.depth_residual,
        torch.roll(original.depth_residual, shift, dims=-1),
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        rolled.density_sh,
        torch.roll(original.density_sh, shift, dims=-1),
        atol=2e-5,
        rtol=2e-4,
    )


def test_erp_convolution_has_no_artificial_longitude_edge() -> None:
    conv = ERPConv2d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
    impulse = torch.zeros(1, 1, 5, 8)
    impulse[0, 0, 2, 0] = 1.0
    output = conv(impulse)
    assert float(output[0, 0, 2, -1].detach()) == 1.0
    assert float(output[0, 0, 2, 1].detach()) == 1.0


def test_invalid_depth_is_masked_without_non_finite_outputs() -> None:
    head = _tiny_head()
    feature, rgb, depth, poses = _inputs(16, 32, views=1)
    depth[..., 0, 0] = float("nan")
    depth[..., 0, 1] = -1.0
    output = head(feature, rgb, depth, poses)
    assert not bool(output.valid_mask[..., 0, 0].any())
    assert not bool(output.valid_mask[..., 0, 1].any())
    assert bool(torch.isfinite(output.refined_depth).all())


def test_invalid_pose_masks_the_corresponding_source_view() -> None:
    head = _tiny_head()
    feature, rgb, depth, poses = _inputs(16, 32, views=2)
    poses[0, 1, 0, 0] = float("nan")
    output = head(feature, rgb, depth, poses)
    assert bool(output.valid_mask[0, 0].all())
    assert not bool(output.valid_mask[0, 1].any())
