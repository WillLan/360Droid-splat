from __future__ import annotations

import torch

from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.pano_resplat_point_decoder_init import INITIALIZER_TYPE, PanoVGGTPointDecoderGaussianInitializer
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter


def test_pano_resplat_frontend_shapes_smoke():
    torch.manual_seed(13)
    b, v, h, w = 1, 3, 12, 24
    world = torch.zeros(b, v, h, w, 3)
    world[..., 2] = 2.0
    context = {
        "images": torch.rand(b, v, 3, h, w),
        "features": torch.rand(b, v, 8, 3, 6),
        "depths": torch.full((b, v, 1, h, w), 2.0),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1),
        "valid_mask": torch.ones(b, v, 1, h, w, dtype=torch.bool),
        "world_points": world,
    }
    frontend = PanoReSplatFrontend(
        initializer=PanoVGGTPointDecoderGaussianInitializer(
            {
                "type": INITIALIZER_TYPE,
                "state_dim": 8,
                "sh_degree": 0,
                "patch_size": 4,
                "decoder_embed_dim": 16,
                "decoder_depth": 1,
                "decoder_num_heads": 4,
            }
        ),
        renderer=PanoGaussianRendererAdapter(soft_max_points=32),
        renderer_backend="soft_splat",
    )
    out = frontend(context, target={"poses_c2w": context["poses_c2w"][:, :1], "images": context["images"][:, :1]}, num_refine=1)

    assert out["final_state"].means.shape == (1, v * h * w, 3)
    assert out["target_render"].color.shape == (1, 1, 3, h, w)
    assert torch.isfinite(out["final_state"].means).all()
