"""Lightweight visualization helpers for Stage 1 spherical adapter matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch

from data.stage1_pano_sequence_dataset import Stage1PanoSequenceDataset
from geometry.spherical_erp import erp_pixel_to_unit_ray, spherical_geodesic_distance
from geometry.spherical_pseudo_correspondence import generate_spherical_pseudo_correspondence


def angular_match_metrics(pred_uv: torch.Tensor, target_uv: torch.Tensor, *, height: int, width: int) -> dict[str, float]:
    """Compute spherical angular metrics for predicted and target ERP coords."""

    pred_ray = erp_pixel_to_unit_ray(pred_uv, height, width)
    target_ray = erp_pixel_to_unit_ray(target_uv, height, width)
    err = torch.rad2deg(spherical_geodesic_distance(pred_ray, target_ray))
    return {
        "mean_angular_error_deg": float(err.mean()) if err.numel() else 0.0,
        "median_angular_error_deg": float(err.median()) if err.numel() else 0.0,
        "pck_1deg": float((err <= 1.0).float().mean()) if err.numel() else 0.0,
        "pck_3deg": float((err <= 3.0).float().mean()) if err.numel() else 0.0,
        "pck_5deg": float((err <= 5.0).float().mean()) if err.numel() else 0.0,
    }


def save_stage1_match_preview(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    src_uv: torch.Tensor,
    pseudo_tgt_uv: torch.Tensor,
    path: str | Path,
    *,
    pred_tgt_uv: torch.Tensor | None = None,
    max_matches: int = 80,
) -> Path:
    """Save a side-by-side match preview from tensors and ERP pixel coords."""

    def to_image(tensor: torch.Tensor) -> Image.Image:
        value = tensor.detach().cpu().clamp(0.0, 1.0)
        array = (value.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        return Image.fromarray(array)

    src = to_image(source_image)
    tgt = to_image(target_image)
    canvas = Image.new("RGB", (src.width + tgt.width, max(src.height, tgt.height)), (0, 0, 0))
    canvas.paste(src, (0, 0))
    canvas.paste(tgt, (src.width, 0))
    draw = ImageDraw.Draw(canvas)
    predicted = pseudo_tgt_uv if pred_tgt_uv is None else pred_tgt_uv
    count = min(int(max_matches), int(src_uv.shape[0]), int(pseudo_tgt_uv.shape[0]), int(predicted.shape[0]))
    if count > 0:
        common_count = min(int(src_uv.shape[0]), int(pseudo_tgt_uv.shape[0]), int(predicted.shape[0]))
        keep = torch.linspace(0, common_count - 1, steps=count).round().long()
        for idx in keep.tolist():
            su = src_uv[idx].detach().cpu()
            pu = pseudo_tgt_uv[idx].detach().cpu()
            pr = predicted[idx].detach().cpu()
            p0 = (float(su[0]), float(su[1]))
            pseudo_p = (float(pu[0]) + src.width, float(pu[1]))
            pred_p = (float(pr[0]) + src.width, float(pr[1]))
            draw.line((p0, pseudo_p), fill=(255, 220, 80), width=1)
            if pred_tgt_uv is not None:
                draw.line((p0, pred_p), fill=(255, 80, 80), width=1)
            draw.ellipse((p0[0] - 2, p0[1] - 2, p0[0] + 2, p0[1] + 2), fill=(80, 220, 255))
            draw.ellipse((pseudo_p[0] - 2, pseudo_p[1] - 2, pseudo_p[0] + 2, pseudo_p[1] + 2), fill=(255, 220, 80))
            if pred_tgt_uv is not None:
                draw.ellipse((pred_p[0] - 2, pred_p[1] - 2, pred_p[0] + 2, pred_p[1] + 2), fill=(255, 80, 80))
    draw.rectangle((6, 6, 430, 28), fill=(0, 0, 0))
    draw.text((12, 10), "cyan=source yellow=pseudo target red=predicted target", fill=(255, 255, 255))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--width", type=int, default=1008)
    parser.add_argument("--max-matches", type=int, default=80)
    parser.add_argument("--predicted-uv-npy", default=None)
    parser.add_argument("--metrics-json", default=None)
    args = parser.parse_args()
    dataset = Stage1PanoSequenceDataset(args.manifest, image_height=args.height, image_width=args.width)
    sample = dataset[0]
    if sample["depths"] is not None and sample["poses_c2w"] is not None:
        corr = generate_spherical_pseudo_correspondence(
            sample["depths"],
            sample["poses_c2w"],
            sample["pair_indices"][:1],
            height=args.height,
            width=args.width,
            num_query_per_pair=max(1, int(args.max_matches)),
            sampling="grid",
        )
        valid = corr.valid_mask[0].bool()
        src_uv = corr.src_uv[0][valid]
        pseudo_uv = corr.tgt_uv[0][valid]
    else:
        h, w = int(sample["images"].shape[-2]), int(sample["images"].shape[-1])
        src_uv = torch.tensor([[w * 0.25, h * 0.5], [w * 0.5, h * 0.5], [w * 0.75, h * 0.5]], dtype=torch.float32)
        pseudo_uv = src_uv.clone()
    predicted = None
    if args.predicted_uv_npy:
        predicted = torch.as_tensor(np.load(args.predicted_uv_npy), dtype=torch.float32)
    save_stage1_match_preview(
        sample["images"][0],
        sample["images"][1],
        src_uv,
        pseudo_uv,
        args.output,
        pred_tgt_uv=predicted,
        max_matches=args.max_matches,
    )
    metrics = angular_match_metrics(predicted if predicted is not None else pseudo_uv, pseudo_uv, height=args.height, width=args.width)
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
