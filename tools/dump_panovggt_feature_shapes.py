"""Dump Stage 1 PanoVGGT feature shapes and adapter output shape."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from training.train_spherical_selfi_adapter import build_adapter, build_panovggt_wrapper, load_config


def dump_feature_shapes(config: dict, *, batch_size: int = 1, views: int | None = None, device: str | None = None) -> dict:
    """Run one dummy batch through the configured wrapper and adapter."""

    image_cfg = config.get("image", {})
    dataset_cfg = config.get("dataset", {})
    height = int(image_cfg.get("height", dataset_cfg.get("image_height", 504)))
    width = int(image_cfg.get("width", dataset_cfg.get("image_width", 1008)))
    num_views = int(views or dataset_cfg.get("views_per_sample", 4))
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wrapper = build_panovggt_wrapper(config, device=torch_device)
    adapter = build_adapter(config, device=torch_device)
    images = torch.zeros(int(batch_size), num_views, 3, height, width, device=torch_device)
    with torch.no_grad():
        output = wrapper(images)
        dense = adapter(output.stage_features)
    return {
        "image_shape": tuple(images.shape),
        "hook_names": output.hook_names,
        "stage_feature_shapes": [tuple(feature.shape) for feature in output.stage_features],
        "adapter_output_shape": tuple(dense.shape),
        "adapter_output_channel_norm_mean": float(torch.linalg.norm(dense, dim=2).mean().detach().cpu()),
        "pose_convention": output.pose_convention,
        "depth_convention": output.depth_convention,
        "feature_convention": output.feature_convention,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_spherical_selfi_adapter.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic PanoVGGT-like wrapper for local smoke.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.synthetic:
        config.setdefault("panovggt", {})["synthetic"] = True
        config.setdefault("panovggt", {})["stage_hooks"] = ["stage1", "stage2", "stage3", "stage4"]
        config.setdefault("panovggt", {})["in_channels"] = [8, 16, 24, 32]
        config.setdefault("adapter", {})["hidden_dim"] = min(int(config.get("adapter", {}).get("hidden_dim", 16)), 16)
    result = dump_feature_shapes(config, batch_size=args.batch_size, views=args.views, device=args.device)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
