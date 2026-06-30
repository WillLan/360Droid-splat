from __future__ import annotations

from pathlib import Path

import torch

from frontend.pano_droid.adapter import build_frontend_from_config
from frontend.pano_vggt.pano_anchor_splat_decoder import PanoAnchorGaussianDecoder
from frontend.pano_vggt.pano_anchor_splat_encoder import PanoAnchorFeatureEncoder
from frontend.pano_vggt.pano_anchor_splat_frontend import PanoAnchorSplatFrontend
from frontend.pano_vggt.pano_anchor_splat_prior import PanoVGGTPriorProvider
from frontend.pano_vggt.pano_anchor_splat_types import PanoAnchorSplatConfig
from frontend.pano_vggt.pano_anchor_splat_voxel import PanoVoxelAnchorBuilder
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.resplat_types import state_to_explicit_gaussian_set
from frontend.pano_vggt.train_anchor_splat_gaussian import load_anchor_splat_train_config, train_anchor_splat_gaussian


def _small_config(**overrides) -> PanoAnchorSplatConfig:
    base = {
        "enabled": True,
        "gaussians_per_anchor": 2,
        "max_anchors": 16,
        "max_gaussians": 32,
        "anchor_dim": 32,
        "decoder_dim": 32,
        "decoder_depth": 1,
        "decoder_heads": 4,
        "refiner_dim": 32,
        "error_dim": 16,
        "error_transformer_depth": 1,
        "point_transformer_depth": 1,
        "decoder_chunk_size": 8,
        "num_global_tokens": 2,
        "raw_feature_bins": 16,
    }
    base.update(overrides)
    return PanoAnchorSplatConfig.from_dict(base)


def _context(b: int = 1, v: int = 3, h: int = 8, w: int = 16, c: int = 6) -> dict[str, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.linspace(-0.5, 0.5, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
    world = torch.stack([xx, yy, torch.ones_like(xx) * 2.0], dim=-1).view(1, 1, h, w, 3).repeat(b, v, 1, 1, 1)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    poses[:, :, 0, 3] = torch.linspace(0.0, 0.05, v).view(1, v)
    return {
        "images": torch.rand(b, v, 3, h, w),
        "features": torch.rand(b, v, c, max(1, h // 4), max(1, w // 4)),
        "depths": torch.full((b, v, 1, h, w), 2.0),
        "poses_c2w": poses,
        "world_points": world,
        "valid_mask": torch.ones(b, v, 1, h, w, dtype=torch.bool),
        "confidence": torch.ones(b, v, 1, h, w),
    }


def test_anchor_splat_config_defaults_are_sh2_and_disabled():
    cfg = PanoAnchorSplatConfig()

    assert cfg.enabled is False
    assert cfg.dtype == "bf16"
    assert cfg.sh_degree == 2
    assert cfg.min_sh_degree == 2
    assert cfg.sh_dim == 9
    assert cfg.max_anchors == 25000
    assert cfg.max_gaussians == 100000


def test_prior_provider_reads_offline_cache_and_detaches(tmp_path: Path):
    cache_path = tmp_path / "seq.pt"
    ctx = _context()
    torch.save({key: value.requires_grad_(value.is_floating_point()) if torch.is_tensor(value) else value for key, value in ctx.items()}, cache_path)
    prior = PanoVGGTPriorProvider()(dict(images=ctx["images"], cache_path=str(cache_path)))

    assert prior.features.shape[:2] == (1, 3)
    assert prior.world_points.shape[-1] == 3
    assert not prior.features.requires_grad
    assert not prior.world_points.requires_grad


def test_voxel_encoder_decoder_caps_and_outputs_sh2():
    cfg = _small_config(max_anchors=8, max_gaussians=10, gaussians_per_anchor=3)
    prior = PanoVGGTPriorProvider()(_context(h=10, w=20))
    anchors = PanoVoxelAnchorBuilder(cfg)(prior)
    tokens = PanoAnchorFeatureEncoder(cfg)(anchors)
    state = PanoAnchorGaussianDecoder(cfg)(anchors, tokens)
    explicit = state_to_explicit_gaussian_set(state)

    assert anchors.num_anchors <= cfg.effective_max_anchors
    assert tokens.shape == (1, anchors.num_anchors, cfg.anchor_dim)
    assert state.num_gaussians <= cfg.max_gaussians
    assert state.sh_coeffs.shape[-1] == 9
    assert explicit.active_sh_degree == 2
    assert explicit.max_sh_degree == 2


def test_frontend_forward_with_refiner_smoke():
    cfg = _small_config(max_anchors=6, max_gaussians=12)
    context = _context(v=2, h=6, w=12)
    frontend = PanoAnchorSplatFrontend(
        cfg,
        renderer=PanoGaussianRendererAdapter(soft_max_points=12),
        renderer_backend="soft_splat",
    )
    out = frontend(
        context,
        target={"poses_c2w": context["poses_c2w"][:, :1], "images": context["images"][:, :1]},
        num_refine=1,
    )

    assert out["final_state"].sh_coeffs.shape[-1] == 9
    assert out["final_state"].num_gaussians <= cfg.max_gaussians
    assert out["target_render"].color.shape == (1, 1, 3, 6, 12)
    assert len(out["context_renders"]) == 1
    assert torch.isfinite(out["final_state"].means).all()


def test_adapter_explicit_anchor_splat_mode_builds_frontend():
    cfg = _small_config().to_dict()
    frontend = build_frontend_from_config({"Frontend": {"mode": "pano_anchor_splat"}, "PanoAnchorSplat": cfg})

    assert isinstance(frontend, PanoAnchorSplatFrontend)


def test_anchor_splat_training_step_saves_checkpoint(tmp_path: Path):
    cfg = load_anchor_splat_train_config(None)
    cfg["Training"].update(
        {
            "steps": 1,
            "batch_size": 1,
            "frames_per_sample": 4,
            "input_frames": 2,
            "num_refine": 0,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "output_dir": str(tmp_path),
            "save_every": 1,
        }
    )
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 1, "height": 8, "width": 16})
    cfg["Model"].update({"use_synthetic_features": True, "feature_dim": 6, "feature_stride": 4})
    cfg["PanoAnchorSplat"].update(_small_config(max_anchors=6, max_gaussians=12, raw_feature_bins=8).to_dict())
    cfg["Renderer"].update({"backend": "soft_splat", "soft_max_points": 12})
    cfg["WeightsAndBiases"].update({"enabled": False, "mode": "disabled"})

    result = train_anchor_splat_gaussian(cfg)

    assert result["steps"] == 1
    assert torch.isfinite(torch.tensor(result["last_metrics"]["loss"]))
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
