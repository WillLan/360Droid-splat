from __future__ import annotations

import torch

from frontend.pano_vggt.pano_resplat_point_decoder_init import INITIALIZER_TYPE, PanoVGGTPointDecoderGaussianInitializer
from frontend.pano_vggt.pano_resplat_refiner import PanoGaussianUpdateBlock


def _context():
    torch.manual_seed(11)
    b, v, h, w = 1, 2, 12, 24
    c, hf, wf = 8, 3, 6
    images = torch.rand(b, v, 3, h, w)
    features = torch.rand(b, v, c, hf, wf)
    depths = torch.full((b, v, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    world = torch.zeros(b, v, h, w, 3)
    world[..., 2] = 2.0
    return images, features, depths, poses, valid, world


def test_zero_init_refiner_delta_is_identity():
    images, features, depths, poses, valid, world = _context()
    initializer = PanoVGGTPointDecoderGaussianInitializer(
        {
            "type": INITIALIZER_TYPE,
            "state_dim": 16,
            "sh_degree": 0,
            "patch_size": 4,
            "decoder_embed_dim": 16,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
        }
    )
    state = initializer(images, features, depths, poses, valid, world_points=world)
    block = PanoGaussianUpdateBlock(feedback_dim=12, latent_dim=state.latent_dim, sh_dim=state.sh_dim, hidden_dim=16)
    feedback = torch.randn(state.batch_size, state.num_gaussians, 12)

    updated, metrics = block(state, feedback)

    assert torch.allclose(updated.means, state.means, atol=1.0e-7)
    assert torch.allclose(updated.log_scales, state.log_scales, atol=1.0e-7)
    assert torch.allclose(updated.rotations_unnorm, state.rotations_unnorm, atol=1.0e-7)
    assert torch.allclose(updated.opacity_logits, state.opacity_logits, atol=1.0e-7)
    assert torch.allclose(updated.sh_coeffs, state.sh_coeffs, atol=1.0e-7)
    assert torch.allclose(updated.latent_features, state.latent_features, atol=1.0e-7)
    assert max(float(value) for value in metrics.values()) < 1.0e-7
    assert torch.isfinite(updated.means).all()
