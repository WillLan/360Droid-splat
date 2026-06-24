from __future__ import annotations

import types

import pytest
import torch

from frontend.pano_vggt.pano_point_transformer import PanoKNNTransformerBlock
from frontend.pano_vggt.pano_resplat_refiner import PanoGaussianUpdateBlock
from frontend.pano_vggt.resplat_types import PanoGaussianState


def _state(batch: int = 1, points: int = 12, latent_dim: int = 8, sh_dim: int = 16) -> PanoGaussianState:
    torch.manual_seed(7)
    means = torch.randn(batch, points, 3)
    valid = torch.ones(batch, points, dtype=torch.bool)
    valid[:, -2:] = False
    return PanoGaussianState(
        means=means,
        log_scales=torch.full((batch, points, 3), -4.0),
        rotations_unnorm=torch.randn(batch, points, 4),
        opacity_logits=torch.zeros(batch, points, 1),
        sh_coeffs=torch.randn(batch, points, 3, sh_dim) * 0.01,
        latent_features=torch.randn(batch, points, latent_dim),
        source_view_ids=torch.zeros(batch, points, dtype=torch.long),
        source_uv=torch.zeros(batch, points, 2),
        valid_mask=valid,
        confidence=torch.ones(batch, points, 1),
    )


def test_resplat_style_block_uses_single_head_projected_attention_and_mlp4x():
    block = PanoKNNTransformerBlock(
        256,
        num_heads=1,
        knn=8,
        attn_proj_channels=64,
        mlp_ratio=4.0,
        knn_backend="cdist",
        max_knn_points=0,
    )

    assert block.num_heads == 1
    assert block.attn_dim == 64
    assert block.qkv.out_features == 64 * 3
    assert block.out_proj.in_features == 64
    assert block.mlp[0].out_features == 256 * 4


def test_pointops_backend_requires_single_head_attention():
    with pytest.raises(ValueError, match="num_heads=1"):
        PanoKNNTransformerBlock(
            64,
            num_heads=4,
            knn=8,
            attn_proj_channels=64,
            knn_backend="pointops",
        )


def test_update_block_runs_four_basic_blocks_and_reuses_knn_cache():
    state = _state()
    block = PanoGaussianUpdateBlock(
        feedback_dim=5,
        latent_dim=state.latent_dim,
        sh_dim=state.sh_dim,
        hidden_dim=16,
        knn=4,
        num_heads=1,
        attn_proj_channels=8,
        mlp_ratio=4.0,
        num_basic_refine_blocks=4,
        cache_knn=True,
        max_knn_points=0,
    )
    calls = 0
    originals = []
    for transformer in block.transformers:
        original = transformer._cdist_knn_indices
        originals.append(original)

        def wrapped(self, xyz, valid, _original=original):
            nonlocal calls
            calls += 1
            return _original(xyz, valid)

        transformer._cdist_knn_indices = types.MethodType(wrapped, transformer)

    feedback = torch.randn(state.batch_size, state.num_gaussians, 5)
    updated, metrics = block(state, feedback)

    assert len(block.transformers) == 4
    assert calls == 1
    assert set(metrics) >= {"mean_delta_abs", "log_scale_delta_abs", "sh_delta_abs"}
    assert torch.allclose(updated.means, state.means, atol=1.0e-7)
    assert torch.allclose(updated.sh_coeffs, state.sh_coeffs, atol=1.0e-7)


def test_invalid_points_are_not_updated_when_delta_head_is_nonzero():
    state = _state(points=10)
    block = PanoGaussianUpdateBlock(
        feedback_dim=5,
        latent_dim=state.latent_dim,
        sh_dim=state.sh_dim,
        hidden_dim=16,
        knn=4,
        num_heads=1,
        attn_proj_channels=8,
        num_basic_refine_blocks=2,
        max_knn_points=0,
    )
    with torch.no_grad():
        block.delta[-1].bias.fill_(0.5)
    feedback = torch.randn(state.batch_size, state.num_gaussians, 5)
    updated, _metrics = block(state, feedback)

    invalid = ~state.valid_mask
    valid = state.valid_mask
    assert torch.allclose(updated.means[invalid], state.means[invalid])
    assert torch.allclose(updated.log_scales[invalid], state.log_scales[invalid])
    assert not torch.allclose(updated.means[valid], state.means[valid])
