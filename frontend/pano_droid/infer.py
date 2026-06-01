"""Run PanoDROID-MVP inference on an ERP image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import load_erp_image
from .model import PanoDroidModel


def infer_pair(checkpoint: str, image0_path: str, image1_path: str, output: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = payload.get("config", {}).get("Model", {})
    model = PanoDroidModel(**model_cfg)
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    model.eval()
    image0 = load_erp_image(image0_path).unsqueeze(0)
    image1 = load_erp_image(image1_path).unsqueeze(0)
    with torch.no_grad():
        pred = model(image0, image1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        spherical_flow=pred["spherical_flow"][0].cpu().numpy(),
        inverse_depth=pred["inverse_depth"][0].cpu().numpy(),
        depth_confidence=pred["depth_confidence"][0].cpu().numpy(),
        relative_pose=pred["relative_pose"][0].cpu().numpy(),
        keyframe_score=pred["keyframe_score"][0].cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--output", default="outputs/pano_droid_infer/prediction.npz")
    args = parser.parse_args()
    infer_pair(args.checkpoint, args.image0, args.image1, args.output)


if __name__ == "__main__":
    main()

