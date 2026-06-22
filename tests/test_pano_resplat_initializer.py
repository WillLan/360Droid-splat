from __future__ import annotations

import inspect

import torch

from frontend.pano_vggt.pano_resplat_init import PanoCompactGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.resplat_types import state_to_explicit_gaussian_set


def _batch(depth_value: float = 2.0):
    torch.manual_seed(7)
    b, v, h, w = 2, 3, 16, 32
    hf, wf, c = 4, 8, 12
    images = torch.rand(b, v, 3, h, w)
    features = torch.rand(b, v, c, hf, wf)
    depths = torch.full((b, v, 1, h, w), float(depth_value))
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    poses[:, :, 0, 3] = torch.linspace(0.0, 0.2, steps=v).view(1, v)
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    return images, features, depths, poses, valid


def _initializer(max_gaussians: int = 32) -> PanoCompactGaussianInitializer:
    return PanoCompactGaussianInitializer(
        latent_downsample=1,
        gaussians_per_cell=2,
        state_dim=16,
        sh_degree=0,
        max_gaussians=max_gaussians,
        min_scale=0.002,
        max_scale=0.12,
        init_scale=0.02,
    )


def test_compact_initializer_outputs_expected_shapes():
    images, features, depths, poses, valid = _batch()
    model = _initializer(max_gaussians=25)

    state = model(images, features, depths, poses, valid)

    assert state.means.shape == (2, 25, 3)
    assert state.log_scales.shape == (2, 25, 3)
    assert state.rotations_unnorm.shape == (2, 25, 4)
    assert state.opacity_logits.shape == (2, 25, 1)
    assert state.sh_coeffs.shape == (2, 25, 3, 1)
    assert state.latent_features.shape == (2, 25, 16)
    assert state.source_view_ids.shape == (2, 25)
    assert state.source_uv.shape == (2, 25, 2)
    assert state.valid_mask.shape == (2, 25)
    assert state.confidence is not None and state.confidence.shape == (2, 25, 1)
    assert torch.isfinite(state.means).all()
    assert torch.isfinite(state.sh_coeffs).all()


def test_compact_initializer_state_materializes_and_soft_renders():
    images, features, depths, poses, valid = _batch()
    model = _initializer(max_gaussians=30)
    state = model(images, features, depths, poses, valid)

    explicit = state_to_explicit_gaussian_set(state, 0)
    renderer = PanoGaussianRendererAdapter(soft_max_points=64)
    out = renderer.render_state(state, poses[:, 0], (12, 24), renderer_backend="soft_splat")

    assert explicit.get_xyz.shape[1] == 3
    assert explicit.get_xyz.shape[0] <= 30
    assert out.color.shape == (2, 3, 12, 24)
    assert out.depth.shape == (2, 1, 12, 24)
    assert out.alpha.shape == (2, 1, 12, 24)
    assert torch.isfinite(out.color).all()


def test_changing_input_depth_changes_means():
    images, features, depths, poses, valid = _batch(depth_value=2.0)
    model = _initializer(max_gaussians=20)

    state_a = model(images, features, depths, poses, valid)
    state_b = model(images, features, depths + 1.0, poses, valid)

    assert not torch.allclose(state_a.means, state_b.means)


def test_target_image_cannot_affect_initializer_outputs():
    images, features, depths, poses, valid = _batch(depth_value=2.0)
    model = _initializer(max_gaussians=20)
    target_a = torch.zeros(2, 3, 16, 32)
    target_b = torch.ones_like(target_a)

    state_a = model(images, features, depths, poses, valid)
    _ = target_a, target_b
    state_b = model(images, features, depths, poses, valid)

    signature = inspect.signature(model.forward)
    assert "target" not in signature.parameters
    assert "target_image" not in signature.parameters
    assert torch.allclose(state_a.means, state_b.means)
    assert torch.allclose(state_a.sh_coeffs, state_b.sh_coeffs)
