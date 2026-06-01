"""Run PanoDROID-MVP inference on an ERP image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import load_erp_image
from .graph_tracker import PanoDroidGraphTracker
from .interfaces import PanoFrame
from .model import PanoDroidModel


def infer_pair(checkpoint: str, image0_path: str, image1_path: str, output: str) -> None:
    """Run the default graph tracker on a two-frame ERP sequence."""
    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = payload.get("config", {}).get("Model", {})
    tracker = PanoDroidGraphTracker(PanoDroidModel(**model_cfg), device="cpu", window_size=2, fixed_frames=1)
    tracker.load_checkpoint(checkpoint)
    image0 = load_erp_image(image0_path)
    image1 = load_erp_image(image1_path)
    tracker.track(PanoFrame(image=image0, timestamp=0.0, frame_id=0))
    out = tracker.track(PanoFrame(image=image1, timestamp=1.0, frame_id=1))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        spherical_flow=None if out.spherical_flow is None else out.spherical_flow.cpu().numpy(),
        inverse_depth=None if out.inverse_depth is None else out.inverse_depth.cpu().numpy(),
        depth_confidence=None if out.depth_confidence is None else out.depth_confidence.cpu().numpy(),
        relative_pose=None if out.relative_pose is None else out.relative_pose.cpu().numpy(),
        pose_c2w=out.pose_c2w.cpu().numpy(),
        keyframe_score=np.array(out.keyframe_score, dtype=np.float32),
        ba_residual=np.array(out.ba_residual if out.ba_residual is not None else np.nan, dtype=np.float32),
    )


def infer_pair_legacy(checkpoint: str, image0_path: str, image1_path: str, output: str) -> None:
    """Run the old pairwise smoke-test path."""
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
    parser.add_argument("--legacy-pairwise", action="store_true")
    args = parser.parse_args()
    if args.legacy_pairwise:
        infer_pair_legacy(args.checkpoint, args.image0, args.image1, args.output)
    else:
        infer_pair(args.checkpoint, args.image0, args.image1, args.output)


if __name__ == "__main__":
    main()
