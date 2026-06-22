from __future__ import annotations

import torch

from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.pano_resplat_init import PanoCompactGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter


def test_pano_resplat_frontend_shapes_smoke():
    torch.manual_seed(13)
    b, v, h, w = 1, 3, 12, 24
    context = {
        "images": torch.rand(b, v, 3, h, w),
        "features": torch.rand(b, v, 8, 3, 6),
        "depths": torch.full((b, v, 1, h, w), 2.0),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1),
        "valid_mask": torch.ones(b, v, 1, h, w, dtype=torch.bool),
    }
    frontend = PanoReSplatFrontend(
        initializer=PanoCompactGaussianInitializer(state_dim=8, max_gaussians=12, gaussians_per_cell=2),
        renderer=PanoGaussianRendererAdapter(soft_max_points=32),
        renderer_backend="soft_splat",
    )
    out = frontend(context, target={"poses_c2w": context["poses_c2w"][:, :1], "images": context["images"][:, :1]}, num_refine=1)

    assert out["final_state"].means.shape == (1, 12, 3)
    assert out["target_render"].color.shape == (1, 1, 3, h, w)
    assert torch.isfinite(out["final_state"].means).all()
