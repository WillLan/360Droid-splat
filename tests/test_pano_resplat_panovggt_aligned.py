from __future__ import annotations

import torch
import torch.nn.functional as F

from frontend.pano_vggt.pano_resplat_point_decoder_init import INITIALIZER_TYPE, PanoVGGTPointDecoderGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.resplat_types import state_to_explicit_gaussian_set
from frontend.pano_vggt.train_resplat_gaussian import (
    _forward_train,
    _sample_input_reconstruction,
    load_resplat_train_config,
)


def _aligned_batch(b: int = 1, v: int = 4, h: int = 8, w: int = 16):
    torch.manual_seed(123)
    images = torch.rand(b, v, 3, h, w)
    features = torch.rand(b, v, 6, max(1, h // 4), max(1, w // 4))
    depths = torch.full((b, v, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    world = torch.stack(
        [
            (xx.float() / max(w - 1, 1) - 0.5) * 2.0,
            (yy.float() / max(h - 1, 1) - 0.5),
            torch.ones_like(xx).float() * 2.0,
        ],
        dim=-1,
    ).view(1, 1, h, w, 3).repeat(b, v, 1, 1, 1)
    world[:, :, :, :, 0] += torch.arange(v).view(1, v, 1, 1).float() * 0.05
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    sky = torch.zeros_like(valid)
    sky[:, :, :, :2, :] = True
    valid = valid & ~sky
    return images, features, depths, poses, valid, sky, world


def _aligned_initializer() -> PanoVGGTPointDecoderGaussianInitializer:
    return PanoVGGTPointDecoderGaussianInitializer(
        {
            "type": INITIALIZER_TYPE,
            "state_dim": 8,
            "sh_degree": 3,
            "patch_size": 4,
            "decoder_embed_dim": 16,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
            "decoder_mlp_ratio": 2.0,
            "init_scale": 0.015,
            "use_local_offsets": True,
            "max_offset_abs": 0.05,
            "max_offset_depth_ratio": 0.02,
        }
    )


def test_point_decoder_gaussian_initializer_outputs_dense_sh3_state_and_masks_sky():
    images, features, depths, poses, valid, _sky, world = _aligned_batch()
    model = _aligned_initializer()

    state = model(images, features, depths, poses, valid, world_points=world)
    explicit = state_to_explicit_gaussian_set(state, 0)

    assert state.num_gaussians == int(images.shape[1] * images.shape[-2] * images.shape[-1])
    assert state.sh_coeffs.shape[-1] == 16
    assert torch.allclose(state.means, world.reshape(1, -1, 3), atol=1.0e-6)
    assert int(state.valid_mask.sum()) == int(valid.sum())
    assert explicit.max_sh_degree == 3
    assert explicit.active_sh_degree == 3
    assert explicit.get_xyz.shape[0] == int(valid[0].sum())


def test_point_decoder_gaussian_offset_updates_means_in_spherical_frame():
    images, features, depths, poses, valid, _sky, world = _aligned_batch()
    model = _aligned_initializer()
    state_zero = model(images, features, depths, poses, valid, world_points=world)

    with torch.no_grad():
        model.pixel_head.set_channel_bias(0, 5.0)
    state_offset = model(images, features, depths, poses, valid, world_points=world)

    delta = (state_offset.means - state_zero.means).norm(dim=-1)
    assert float(delta[state_zero.valid_mask].mean().detach()) > 1.0e-3
    assert float(delta.max().detach()) <= 0.051


def test_point_decoder_gaussian_materialize_does_not_clamp_large_scale():
    images, features, depths, poses, valid, _sky, world = _aligned_batch()
    model = _aligned_initializer()
    state = model(images, features, depths, poses, valid, world_points=world)
    state.log_scales[..., :] = 2.0

    explicit = state_to_explicit_gaussian_set(state, 0)

    assert torch.allclose(explicit.get_scaling, torch.full_like(explicit.get_scaling, torch.exp(torch.tensor(2.0))))


def test_input_reconstruction_uses_all_four_views_and_prior_poses():
    images, features, depths, poses, valid, sky, world = _aligned_batch(h=6, w=10)
    sample = {
        "images": images,
        "valid_depth": valid,
        "sky_mask": sky,
        "supervision_images": F.interpolate(images.reshape(4, 3, 6, 10), size=(12, 20), mode="bilinear", align_corners=False).view(1, 4, 3, 12, 20),
        "supervision_depths": F.interpolate(depths.reshape(4, 1, 6, 10), size=(12, 20), mode="nearest").view(1, 4, 1, 12, 20),
        "supervision_valid_depth": F.interpolate(valid.float().reshape(4, 1, 6, 10), size=(12, 20), mode="nearest").view(1, 4, 1, 12, 20).bool(),
        "supervision_sky_mask": F.interpolate(sky.float().reshape(4, 1, 6, 10), size=(12, 20), mode="nearest").view(1, 4, 1, 12, 20).bool(),
        "poses_c2w": torch.zeros_like(poses),
    }
    prior_poses = poses.clone()
    prior_poses[:, :, 0, 3] = 7.0
    priors = {"features": features, "depth": depths, "poses_c2w": prior_poses, "world_points": world}
    cfg = load_resplat_train_config(None)
    cfg["Training"].update({"view_mode": "input_reconstruction", "render_height": 12, "render_width": 20})

    context, target = _sample_input_reconstruction(sample, priors, cfg)

    assert context["images"].shape == images.shape
    assert target["images"].shape == (1, 4, 3, 12, 20)
    assert target["depths"].shape == (1, 4, 1, 12, 20)
    assert context["view_indices"].tolist() == [0, 1, 2, 3]
    assert target["view_indices"].tolist() == [0, 1, 2, 3]
    assert torch.allclose(context["poses_c2w"], prior_poses)
    assert torch.allclose(target["poses_c2w"], prior_poses)
    assert not bool(context["valid_mask"][0, 0, 0, 0, 0])
    assert not bool(target["valid_mask"][0, 0, 0, 0, 0])


def test_point_decoder_gaussian_soft_splat_forward_loss_backward():
    images, features, depths, poses, valid, sky, world = _aligned_batch(h=5, w=10)
    sample = {
        "images": images,
        "valid_depth": valid,
        "sky_mask": sky,
        "supervision_images": F.interpolate(images.reshape(4, 3, 5, 10), size=(6, 12), mode="bilinear", align_corners=False).view(1, 4, 3, 6, 12),
        "supervision_depths": F.interpolate(depths.reshape(4, 1, 5, 10), size=(6, 12), mode="nearest").view(1, 4, 1, 6, 12),
        "supervision_valid_depth": F.interpolate(valid.float().reshape(4, 1, 5, 10), size=(6, 12), mode="nearest").view(1, 4, 1, 6, 12).bool(),
        "supervision_sky_mask": F.interpolate(sky.float().reshape(4, 1, 5, 10), size=(6, 12), mode="nearest").view(1, 4, 1, 6, 12).bool(),
    }
    priors = {"features": features, "depth": depths, "poses_c2w": poses, "world_points": world}
    cfg = load_resplat_train_config(None)
    cfg["Training"].update({"view_mode": "input_reconstruction", "render_height": 6, "render_width": 12})
    cfg["Initializer"].update(
        {
            "type": INITIALIZER_TYPE,
            "state_dim": 8,
            "sh_degree": 3,
            "patch_size": 4,
            "decoder_embed_dim": 16,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
            "decoder_mlp_ratio": 2.0,
        }
    )
    cfg["Renderer"].update({"backend": "soft_splat", "allow_soft_splat_fallback": True, "soft_max_points": 256})
    cfg["Loss"].update({"lpips_weight": 0.0, "context_weight": 0.0, "delta_reg_weight": 0.0, "mean_step_reg_weight": 0.0})
    context, target = _sample_input_reconstruction(sample, priors, cfg)
    frontend = PanoReSplatFrontend(
        initializer=_aligned_initializer(),
        renderer=PanoGaussianRendererAdapter(soft_max_points=256),
        renderer_backend="soft_splat",
    )

    loss, metrics, artifacts = _forward_train(frontend, context, target, num_refine=0, stage="init", config=cfg)
    loss.backward()
    grad_sum = sum(float(p.grad.detach().abs().sum()) for p in frontend.initializer.parameters() if p.grad is not None)

    assert torch.isfinite(loss)
    assert metrics["target_valid_ratio"] > 0.0
    assert artifacts["target_renders"][-1].color.shape == (1, 4, 3, 6, 12)
    assert grad_sum > 0.0
