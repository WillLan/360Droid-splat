from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frontend.pano_vggt.pano_resplat_init import PanoCompactGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.resplat_types import PanoRenderOutput, PanoGaussianState, state_to_explicit_gaussian_set
from frontend.pano_vggt.train_resplat_gaussian import _sample_window, _single_output_loss, load_resplat_train_config


def _dense_batch(b: int = 1, v: int = 4, h: int = 6, w: int = 10):
    torch.manual_seed(31)
    images = torch.rand(b, v, 3, h, w)
    features = torch.rand(b, v, 7, max(1, h // 2), max(1, w // 2))
    depths = torch.full((b, v, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = torch.stack([xx.float(), yy.float(), torch.ones_like(xx).float() * 2.0], dim=-1)
    world = base.view(1, 1, h, w, 3).repeat(b, v, 1, 1, 1)
    world[:, :, :, :, 0] += torch.arange(v).view(1, v, 1, 1).float()
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    sky = torch.zeros_like(valid)
    sky[:, :, :1, :2, :] = True
    valid = valid & ~sky
    return images, features, depths, poses, valid, sky, world


def _dense_initializer() -> PanoCompactGaussianInitializer:
    return PanoCompactGaussianInitializer(
        position_mode="dense_world_points",
        latent_downsample=1,
        gaussians_per_cell=1,
        state_dim=8,
        sh_degree=0,
        max_gaussians=0,
        use_world_points_as_base=True,
        use_local_offsets=False,
    )


def test_dense_world_initializer_keeps_all_world_points_and_masks_sky():
    images, features, depths, poses, valid, _sky, world = _dense_batch()
    model = _dense_initializer()

    state = model(images, features, depths, poses, valid, world_points=world)
    explicit = state_to_explicit_gaussian_set(state, 0)

    assert state.num_gaussians == int(images.shape[1] * images.shape[-2] * images.shape[-1])
    assert torch.allclose(state.means, world.reshape(1, -1, 3))
    assert int(state.valid_mask.sum()) == int(valid.sum())
    assert explicit.get_xyz.shape[0] == int(valid[0].sum())
    assert state.confidence is not None and state.confidence.shape == (1, state.num_gaussians, 1)


def test_dense_world_initializer_means_follow_world_points_exactly():
    images, features, depths, poses, valid, _sky, world = _dense_batch()
    model = _dense_initializer()

    state_a = model(images, features, depths, poses, valid, world_points=world)
    state_b = model(images, features, depths, poses, valid, world_points=world + 3.0)

    assert torch.allclose(state_b.means - state_a.means, torch.full_like(state_a.means, 3.0), atol=1.0e-6)


def test_dense_world_state_soft_splat_smoke_renders():
    images, features, depths, poses, valid, _sky, world = _dense_batch(h=4, w=8)
    model = _dense_initializer()
    state = model(images, features, depths, poses, valid, world_points=world)
    renderer = PanoGaussianRendererAdapter(soft_max_points=128)

    out = renderer.render_state(state, poses[:, 0], (4, 8), renderer_backend="soft_splat")

    assert out.color.shape == (1, 3, 4, 8)
    assert torch.isfinite(out.color).all()


def test_random_split_samples_context_and_target_from_full_clip(monkeypatch: pytest.MonkeyPatch):
    sample = {
        "images": torch.rand(1, 6, 3, 4, 8),
        "valid_depth": torch.ones(1, 6, 1, 4, 8, dtype=torch.bool),
        "sky_mask": torch.zeros(1, 6, 1, 4, 8, dtype=torch.bool),
    }
    sample["sky_mask"][:, 2, :, :2] = True
    priors = {
        "features": torch.rand(1, 6, 4, 2, 4),
        "depth": torch.ones(1, 6, 1, 4, 8),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1),
        "world_points": torch.rand(1, 6, 4, 8, 3),
    }
    order = torch.tensor([2, 0, 5, 1, 4, 3])

    def _fake_randperm(n: int, *, device=None):
        assert n == 6
        return order.to(device=device)

    monkeypatch.setattr(torch, "randperm", _fake_randperm)
    cfg = load_resplat_train_config(None)
    cfg["Training"].update({"context_views": 4, "target_views": 2, "window_mode": "random_split"})

    context, target = _sample_window(sample, priors, cfg, step=0)

    assert context["view_indices"].tolist() == [2, 0, 5, 1]
    assert target["view_indices"].tolist() == [4, 3]
    assert context["valid_mask"].shape == (1, 4, 1, 4, 8)
    assert not bool(context["valid_mask"][0, 0, 0, 0, 0])
    assert bool(target["valid_mask"].all())


def _toy_state() -> PanoGaussianState:
    return PanoGaussianState(
        means=torch.zeros(1, 2, 3),
        log_scales=torch.full((1, 2, 3), -4.0),
        rotations_unnorm=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        opacity_logits=torch.zeros(1, 2, 1),
        sh_coeffs=torch.zeros(1, 2, 3, 1),
        latent_features=torch.zeros(1, 2, 4),
        source_view_ids=torch.zeros(1, 2, dtype=torch.long),
        source_uv=torch.zeros(1, 2, 2),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        confidence=torch.ones(1, 2, 1),
    )


def test_context_l1_metric_does_not_affect_total_loss_when_weight_zero():
    state = _toy_state()
    render = PanoRenderOutput(
        color=torch.zeros(1, 1, 3, 4, 8),
        depth=torch.ones(1, 1, 1, 4, 8),
        alpha=torch.ones(1, 1, 1, 4, 8),
        extras={},
    )
    target = {
        "images": torch.zeros(1, 1, 3, 4, 8),
        "depths": torch.ones(1, 1, 1, 4, 8),
        "valid_mask": torch.ones(1, 1, 1, 4, 8, dtype=torch.bool),
    }
    context = {"images": torch.zeros(1, 1, 3, 4, 8), "valid_mask": torch.ones(1, 1, 1, 4, 8, dtype=torch.bool)}
    cfg = load_resplat_train_config(None)
    cfg["Loss"].update(
        {
            "context_weight": 0.0,
            "opacity_reg_weight": 0.0,
            "alpha_coverage_weight": 0.0,
            "scale_reg_weight": 0.0,
            "anisotropy_reg_weight": 0.0,
            "sh_reg_weight": 0.0,
        }
    )

    loss_a, metrics_a = _single_output_loss(state, render, target, context_render=render, context=context, prev_state=None, config=cfg)
    context_b = {"images": torch.ones(1, 1, 3, 4, 8), "valid_mask": context["valid_mask"]}
    loss_b, metrics_b = _single_output_loss(state, render, target, context_render=render, context=context_b, prev_state=None, config=cfg)

    assert torch.allclose(loss_a, loss_b)
    assert metrics_b["context_l1"] > metrics_a["context_l1"]


def test_all_sky_target_excludes_rgb_and_depth_loss():
    state = _toy_state()
    render = PanoRenderOutput(
        color=torch.zeros(1, 1, 3, 4, 8),
        depth=torch.zeros(1, 1, 1, 4, 8),
        alpha=torch.zeros(1, 1, 1, 4, 8),
        extras={},
    )
    target = {
        "images": torch.ones(1, 1, 3, 4, 8),
        "depths": torch.ones(1, 1, 1, 4, 8),
        "valid_mask": torch.zeros(1, 1, 1, 4, 8, dtype=torch.bool),
    }
    cfg = load_resplat_train_config(None)

    _loss, metrics = _single_output_loss(
        state,
        render,
        target,
        context_render=None,
        context={"images": target["images"]},
        prev_state=None,
        config=cfg,
    )

    assert metrics["rgb_l1"].item() == 0.0
    assert metrics["depth_loss"].item() == 0.0
    assert metrics["target_valid_ratio"].item() == 0.0
