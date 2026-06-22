from __future__ import annotations

from pathlib import Path

import torch

from backend.pano_gs import PFGS360Renderer, PanoRenderCamera
from frontend.pano_vggt.gaussian_head import PanoVGGTAnchorGaussianHead
from frontend.pano_vggt.matching_dataset import SyntheticOmni360TrainingDataset
from frontend.pano_vggt.train_gaussian import (
    FeedForwardGaussianModel,
    SyntheticGaussianPriorExtractor,
    _run_model_iteration,
    _select_source_target,
    _world_points_from_depth,
    load_gaussian_train_config,
    train_gaussian_head,
)
from frontend.pano_vggt.train_matching import matching_collate


def _config(tmp_path: Path) -> dict:
    cfg = load_gaussian_train_config(None)
    cfg["Training"].update(
        {
            "mode": "matching_only",
            "steps": 1,
            "batch_size": 1,
            "frames_per_sample": 3,
            "input_frames": 2,
            "num_workers": 0,
            "save_interval": 1,
            "log_interval": 100,
            "output_dir": str(tmp_path),
        }
    )
    cfg["Model"].update({"use_synthetic_features": True, "feature_dim": 8, "feature_stride": 4})
    cfg["GaussianHead"].update(
        {
            "hidden_dim": 8,
            "anchor_feat_dim": 8,
            "k_offsets": 2,
            "num_conv_blocks": 1,
            "max_anchors": 32,
            "iterations": 1,
            "refiner_hidden_dim": 16,
        }
    )
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 1, "height": 12, "width": 24})
    cfg["Renderer"].update({"backend": "soft_splat", "soft_max_points": 128})
    cfg["WeightsAndBiases"]["enabled"] = False
    cfg["Visualization"].update({"enabled": True, "interval": 1, "max_width": 96})
    return cfg


def test_anchor_gaussian_head_materializes_renderer_compatible_gaussians():
    b, n, h, w = 1, 2, 12, 24
    feature_dim = 8
    images = torch.rand(b, n, 3, h, w)
    depth = torch.full((b, n, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, n, 1, 1)
    poses[:, 1, 0, 3] = 0.1
    features = torch.rand(b, n, feature_dim, h // 4, w // 4)
    world = _world_points_from_depth(depth, poses)
    head = PanoVGGTAnchorGaussianHead(
        feature_dim,
        hidden_dim=8,
        anchor_feat_dim=8,
        k_offsets=2,
        num_conv_blocks=1,
        max_anchors=16,
    )

    pred = head(features, images, depth, poses, world_points=world)
    gaussians = pred.materialize(0)

    assert pred.num_anchors == 16
    assert gaussians.get_xyz.shape == (32, 3)
    assert gaussians.get_scaling.shape == (32, 3)
    assert gaussians.get_rotation.shape == (32, 4)
    assert gaussians.get_opacity.shape == (32, 1)
    assert gaussians.get_features.shape == (32, 3)
    assert torch.isfinite(gaussians.get_xyz).all()
    assert torch.isfinite(gaussians.get_scaling).all()
    assert torch.allclose(torch.linalg.norm(gaussians.get_rotation, dim=-1), torch.ones(32), atol=1.0e-5)

    renderer = PFGS360Renderer(config={"Training": {"pfgs360_render_mode": "RGB+ED"}}, allow_fallback=True)
    pkg = renderer.render(PanoRenderCamera(image_height=h, image_width=w, c2w=torch.eye(4)), gaussians)
    assert pkg["render"].shape == (3, h, w)
    assert pkg["depth"].shape == (1, h, w)


def test_anchor_gaussian_head_sanitizes_nonfinite_priors():
    b, n, h, w = 1, 2, 12, 24
    feature_dim = 8
    images = torch.rand(b, n, 3, h, w)
    depth = torch.full((b, n, 1, h, w), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, n, 1, 1)
    features = torch.rand(b, n, feature_dim, h // 4, w // 4)
    world = _world_points_from_depth(depth, poses)
    features[:, :, :, 0, 0] = float("nan")
    depth[:, :, :, 0, 0] = float("nan")
    world[:, :, 0, 0] = float("nan")
    head = PanoVGGTAnchorGaussianHead(
        feature_dim,
        hidden_dim=8,
        anchor_feat_dim=8,
        k_offsets=2,
        num_conv_blocks=1,
        max_anchors=16,
    )

    pred = head(features, images, depth, poses, world_points=world)
    gaussians = pred.materialize(0)

    assert torch.isfinite(pred.base_depth).all()
    assert torch.isfinite(pred.log_scales).all()
    assert torch.isfinite(pred.local_offsets).all()
    assert torch.isfinite(gaussians.get_xyz).all()
    assert torch.isfinite(gaussians.get_scaling).all()


def test_gaussian_head_smoke_training_saves_checkpoint_and_visualization(tmp_path: Path):
    cfg = _config(tmp_path)

    result = train_gaussian_head(cfg)

    assert result["steps"] == 1
    assert Path(result["checkpoint"]).is_file()
    assert (tmp_path / "visualizations" / "step_0000001_gaussian.png").is_file()
    metrics = result["last_metrics"]
    assert "final/psnr" in metrics
    assert "iter1/loss_delta_from_prev" in metrics
    assert "iter1/update_depth_delta_abs" in metrics
    assert "iter_loss_improvement" in metrics


def test_gaussian_refiner_receives_render_feedback_gradients(tmp_path: Path):
    cfg = _config(tmp_path)
    sample = matching_collate([SyntheticOmni360TrainingDataset(length=1, n_frames=3, height=12, width=24)[0]])
    prior = SyntheticGaussianPriorExtractor(feature_dim=8, feature_stride=4)
    priors = prior(sample)
    batch = _select_source_target(priors, sample, cfg)
    model = FeedForwardGaussianModel(feature_dim=8, config=cfg)

    loss, metrics, _render_history, _states = _run_model_iteration(
        model,
        features=batch["features"].float(),
        source_images=batch["source_images"].float(),
        source_depth=batch["source_depth"].float(),
        source_poses=batch["source_poses"].float(),
        source_world=batch["source_world"].float(),
        target_rgb=batch["target_rgb"].float(),
        target_depth=batch["target_depth"].float(),
        target_poses=batch["target_poses"].float(),
        renderer=None,
        config=cfg,
    )
    loss.backward()

    last_delta = model.refiner.delta[-1]
    assert isinstance(last_delta, torch.nn.Linear)
    assert last_delta.weight.grad is not None
    assert float(last_delta.weight.grad.abs().sum()) > 0.0
    assert torch.isfinite(metrics["iter_loss_improvement"])
