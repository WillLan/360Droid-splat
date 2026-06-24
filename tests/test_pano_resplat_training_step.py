from __future__ import annotations

import os
from pathlib import Path
import time

import torch

from frontend.pano_vggt.train_resplat_gaussian import _sample_input_reconstruction, _sample_window, load_resplat_train_config, train_resplat_gaussian


def _config(tmp_path: Path) -> dict:
    cfg = load_resplat_train_config(None)
    cfg["Training"].update(
        {
            "stage": "overfit",
            "steps": 1,
            "batch_size": 1,
            "frames_per_sample": 4,
            "context_views": 2,
            "target_views": 1,
            "train_min_refine": 0,
            "train_max_refine": 0,
            "num_workers": 0,
            "output_dir": str(tmp_path),
            "save_every": 1,
            "vis_every": 1,
            "debug_overfit": True,
            "log_every": 10,
        }
    )
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 1, "height": 12, "width": 24})
    cfg["Model"].update({"use_synthetic_features": True, "feature_dim": 8, "feature_stride": 4})
    cfg["Initializer"].update({"state_dim": 8, "max_gaussians": 24, "gaussians_per_cell": 2})
    cfg["Feedback"].update({"feedback_dim": 8, "hidden_dim": 8})
    cfg["Refiner"].update({"hidden_dim": 8, "knn": 4, "num_heads": 2, "max_knn_points": 64})
    cfg["Renderer"].update({"backend": "soft_splat", "soft_max_points": 64})
    cfg["WeightsAndBiases"].update({"enabled": False, "mode": "disabled"})
    return cfg


def test_resplat_training_step_saves_outputs():
    output_dir = Path("outputs/pano_resplat/test_training_step") / f"run_{os.getpid()}_{time.time_ns()}"
    result = train_resplat_gaussian(_config(output_dir), command=["pytest"])

    assert result["steps"] == 1
    assert torch.isfinite(torch.tensor(result["last_metrics"]["total_loss"]))
    assert (output_dir / "latest.pt").is_file()
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "train_metrics.csv").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "renders" / "step_000000" / "panel.png").is_file()


def test_sample_window_uses_batch_device_for_indices():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    sample = {
        "images": torch.rand(1, 4, 3, 8, 16, device=device),
        "valid_depth": torch.ones(1, 4, 1, 8, 16, dtype=torch.bool, device=device),
    }
    priors = {
        "features": torch.rand(1, 4, 4, 2, 4, device=device),
        "depth": torch.ones(1, 4, 1, 8, 16, device=device),
        "poses_c2w": torch.eye(4, device=device).view(1, 1, 4, 4).expand(1, 4, 4, 4),
        "world_points": torch.rand(1, 4, 8, 16, 3, device=device),
    }
    cfg = load_resplat_train_config(None)
    cfg["Training"].update({"context_views": 2, "target_views": 1, "debug_overfit": True})

    context, target = _sample_window(sample, priors, cfg, step=0)

    assert context["images"].device == device
    assert target["images"].device == device


def test_input_reconstruction_uses_high_resolution_supervision_target():
    cfg = load_resplat_train_config(None)
    cfg["Training"].update({"render_height": 16, "render_width": 32})
    low_images = torch.zeros(1, 4, 3, 8, 16)
    high_images = torch.ones(1, 4, 3, 16, 32)
    sample = {
        "images": low_images,
        "valid_depth": torch.ones(1, 4, 1, 8, 16, dtype=torch.bool),
        "sky_mask": torch.zeros(1, 4, 1, 8, 16, dtype=torch.bool),
        "supervision_images": high_images,
        "supervision_depths": torch.full((1, 4, 1, 16, 32), 3.0),
        "supervision_valid_depth": torch.ones(1, 4, 1, 16, 32, dtype=torch.bool),
        "supervision_sky_mask": torch.zeros(1, 4, 1, 16, 32, dtype=torch.bool),
    }
    priors = {
        "features": torch.rand(1, 4, 4, 2, 4),
        "depth": torch.ones(1, 4, 1, 8, 16),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4).expand(1, 4, 4, 4),
        "world_points": torch.rand(1, 4, 8, 16, 3),
    }

    context, target = _sample_input_reconstruction(sample, priors, cfg)

    assert tuple(context["images"].shape[-2:]) == (8, 16)
    assert tuple(target["images"].shape[-2:]) == (16, 32)
    assert torch.allclose(target["images"], high_images)
    assert torch.allclose(target["depths"], torch.full((1, 4, 1, 16, 32), 3.0))
    assert tuple(target["valid_mask"].shape[-2:]) == (16, 32)
