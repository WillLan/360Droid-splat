from __future__ import annotations

import inspect

import torch

from frontend.pano_vggt.pano_resplat_feedback import PanoRenderFeedbackEncoder
from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.pano_resplat_init import PanoCompactGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter


def _frontend() -> PanoReSplatFrontend:
    initializer = PanoCompactGaussianInitializer(
        latent_downsample=1,
        gaussians_per_cell=2,
        state_dim=16,
        max_gaussians=18,
        init_scale=0.02,
    )
    return PanoReSplatFrontend(
        initializer=initializer,
        feedback_encoder=PanoRenderFeedbackEncoder(feedback_dim=12, hidden_dim=16),
        renderer=PanoGaussianRendererAdapter(soft_max_points=64),
        renderer_backend="soft_splat",
    )


def _context(image_shift: float = 0.0) -> dict[str, torch.Tensor]:
    torch.manual_seed(23)
    b, v, h, w = 1, 2, 12, 24
    c, hf, wf = 8, 3, 6
    images = (torch.rand(b, v, 3, h, w) + float(image_shift)).clamp(0.0, 1.0)
    features = torch.rand(b, v, c, hf, wf)
    depths = torch.full((b, v, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    poses[:, 1, 0, 3] = 0.1
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    return {
        "images": images,
        "features": features,
        "depths": depths,
        "poses_c2w": poses,
        "valid_mask": valid,
    }


def _target(fill: float) -> dict[str, torch.Tensor]:
    return {
        "images": torch.full((1, 1, 3, 12, 24), float(fill)),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4),
    }


def test_feedback_encoder_has_no_target_argument():
    signature = inspect.signature(PanoRenderFeedbackEncoder.forward)
    assert "target_images" not in signature.parameters
    assert "target" not in signature.parameters


def test_different_target_images_do_not_change_refined_state():
    frontend = _frontend()
    context = _context()

    out_a = frontend(context, target=_target(0.0), num_refine=2)
    out_b = frontend(context, target=_target(1.0), num_refine=2)

    state_a = out_a["final_state"]
    state_b = out_b["final_state"]
    assert torch.allclose(state_a.means, state_b.means, atol=1.0e-7)
    assert torch.allclose(state_a.sh_coeffs, state_b.sh_coeffs, atol=1.0e-7)
    assert torch.allclose(state_a.latent_features, state_b.latent_features, atol=1.0e-7)


def test_different_context_images_change_final_state():
    frontend = _frontend()

    out_a = frontend(_context(image_shift=0.0), target=_target(0.5), num_refine=2)
    out_b = frontend(_context(image_shift=0.4), target=_target(0.5), num_refine=2)

    assert not torch.allclose(out_a["final_state"].sh_coeffs, out_b["final_state"].sh_coeffs)


def test_num_refine_shapes_and_finite_outputs():
    frontend = _frontend()
    context = _context()
    for num_refine in (0, 1, 2):
        out = frontend(context, target=_target(0.2), num_refine=num_refine)
        state = out["final_state"]
        assert state.means.shape == (1, 18, 3)
        assert state.log_scales.shape == (1, 18, 3)
        assert state.sh_coeffs.shape == (1, 18, 3, 1)
        assert len(out["states"]) == num_refine + 1
        assert torch.isfinite(state.means).all()
        assert torch.isfinite(state.log_scales).all()
        assert out["target_render"] is not None
        assert torch.isfinite(out["target_render"].color).all()
