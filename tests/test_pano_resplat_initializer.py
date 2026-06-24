from __future__ import annotations

import inspect

import pytest
import torch

from frontend.pano_vggt.pano_resplat_point_decoder_init import INITIALIZER_TYPE, PanoVGGTPointDecoderGaussianInitializer
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
    world = torch.zeros(b, v, h, w, 3)
    world[..., 2] = float(depth_value)
    return images, features, depths, poses, valid, world


def _initializer() -> PanoVGGTPointDecoderGaussianInitializer:
    return PanoVGGTPointDecoderGaussianInitializer(
        {
            "type": INITIALIZER_TYPE,
            "state_dim": 16,
            "sh_degree": 0,
            "patch_size": 4,
            "decoder_embed_dim": 32,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
            "init_scale": 0.02,
        }
    )


def test_legacy_initializer_modes_are_rejected():
    with pytest.raises(ValueError):
        PanoVGGTPointDecoderGaussianInitializer({"type": "panovggt_aligned"})
    with pytest.raises(ValueError):
        PanoVGGTPointDecoderGaussianInitializer({"position_mode": "dense_world_points"})


def test_point_decoder_initializer_outputs_expected_shapes():
    images, features, depths, poses, valid, world = _batch()
    model = _initializer()

    state = model(images, features, depths, poses, valid, world_points=world)

    n = int(images.shape[1] * images.shape[-2] * images.shape[-1])
    assert state.means.shape == (2, n, 3)
    assert state.log_scales.shape == (2, n, 3)
    assert state.rotations_unnorm.shape == (2, n, 4)
    assert state.opacity_logits.shape == (2, n, 1)
    assert state.sh_coeffs.shape == (2, n, 3, 1)
    assert state.latent_features.shape == (2, n, 16)
    assert state.source_view_ids.shape == (2, n)
    assert state.source_uv.shape == (2, n, 2)
    assert state.valid_mask.shape == (2, n)
    assert state.confidence is not None and state.confidence.shape == (2, n, 1)
    assert torch.isfinite(state.means).all()
    assert torch.isfinite(state.sh_coeffs).all()


def test_point_decoder_initializer_state_materializes_and_soft_renders():
    images, features, depths, poses, valid, world = _batch()
    model = _initializer()
    state = model(images, features, depths, poses, valid, world_points=world)

    explicit = state_to_explicit_gaussian_set(state, 0)
    renderer = PanoGaussianRendererAdapter(soft_max_points=64)
    out = renderer.render_state(state, poses[:, 0], (12, 24), renderer_backend="soft_splat")

    assert explicit.get_xyz.shape[1] == 3
    assert explicit.get_xyz.shape[0] == int(valid[0].sum())
    assert out.color.shape == (2, 3, 12, 24)
    assert out.depth.shape == (2, 1, 12, 24)
    assert out.alpha.shape == (2, 1, 12, 24)
    assert torch.isfinite(out.color).all()


def test_changing_input_depth_changes_means():
    images, features, depths, poses, valid, world = _batch(depth_value=2.0)
    model = _initializer()

    state_a = model(images, features, depths, poses, valid, world_points=world)
    state_b = model(images, features, depths + 1.0, poses, valid, world_points=world + torch.tensor([0.0, 0.0, 1.0]))

    assert not torch.allclose(state_a.means, state_b.means)


def test_target_image_cannot_affect_initializer_outputs():
    images, features, depths, poses, valid, world = _batch(depth_value=2.0)
    model = _initializer()
    target_a = torch.zeros(2, 3, 16, 32)
    target_b = torch.ones_like(target_a)

    state_a = model(images, features, depths, poses, valid, world_points=world)
    _ = target_a, target_b
    state_b = model(images, features, depths, poses, valid, world_points=world)

    signature = inspect.signature(model.forward)
    assert "target" not in signature.parameters
    assert "target_image" not in signature.parameters
    assert torch.allclose(state_a.means, state_b.means)
    assert torch.allclose(state_a.sh_coeffs, state_b.sh_coeffs)
