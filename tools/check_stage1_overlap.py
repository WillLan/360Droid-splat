"""Check Stage 1 manifest window and optional pseudo-overlap statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.stage1_pano_sequence_dataset import (
    Stage1PanoSequenceDataset,
    build_stage1_windows,
    load_stage1_manifest,
    summarize_stage1_manifest,
)
from geometry.spherical_erp import erp_pixel_to_unit_ray, spherical_geodesic_distance
from geometry.spherical_pseudo_correspondence import generate_spherical_pseudo_correspondence


def check_manifest_overlap(
    manifest: str,
    *,
    split: str = "train",
    views_per_sample: int = 4,
    max_temporal_gap: int | None = 10,
    image_height: int = 504,
    image_width: int = 1008,
    max_windows: int = 8,
    num_query_per_pair: int = 256,
) -> dict:
    records = load_stage1_manifest(manifest)
    windows = build_stage1_windows(
        records,
        split=split,
        views_per_sample=views_per_sample,
        max_temporal_gap=max_temporal_gap,
    )
    summary = summarize_stage1_manifest(records)
    summary.update(
        {
            "split_checked": split,
            "views_per_sample": int(views_per_sample),
            "valid_windows": len(windows),
            "has_trainable_windows": len(windows) > 0,
        }
    )
    pseudo = _pseudo_overlap_stats(
        manifest,
        split=split,
        views_per_sample=views_per_sample,
        max_temporal_gap=max_temporal_gap,
        image_height=image_height,
        image_width=image_width,
        max_windows=max_windows,
        num_query_per_pair=num_query_per_pair,
    )
    summary.update(pseudo)
    return summary


def _pseudo_overlap_stats(
    manifest: str,
    *,
    split: str,
    views_per_sample: int,
    max_temporal_gap: int | None,
    image_height: int,
    image_width: int,
    max_windows: int,
    num_query_per_pair: int,
) -> dict[str, Any]:
    try:
        dataset = Stage1PanoSequenceDataset(
            manifest,
            split=split,
            views_per_sample=views_per_sample,
            image_height=image_height,
            image_width=image_width,
            max_temporal_gap=max_temporal_gap,
        )
    except Exception as exc:
        return {
            "pseudo_stats_available": False,
            "pseudo_stats_reason": repr(exc),
            "mean_valid_corr_ratio": None,
            "mean_angular_pseudo_reprojection_error_deg": None,
        }
    ratios: list[float] = []
    angular_deg: list[float] = []
    checked = min(len(dataset), max(0, int(max_windows)))
    for idx in range(checked):
        sample = dataset[idx]
        if sample["depths"] is None or sample["poses_c2w"] is None:
            continue
        corr = generate_spherical_pseudo_correspondence(
            sample["depths"],
            sample["poses_c2w"],
            sample["pair_indices"],
            height=image_height,
            width=image_width,
            num_query_per_pair=int(num_query_per_pair),
            sampling="grid",
        )
        valid = corr.valid_mask.bool()
        ratios.append(float(valid.float().mean()))
        if valid.any():
            reproj_ray = erp_pixel_to_unit_ray(corr.tgt_uv, image_height, image_width).to(corr.tgt_ray)
            err = torch.rad2deg(spherical_geodesic_distance(reproj_ray, corr.tgt_ray))
            angular_deg.append(float(err[valid].mean()))
    return {
        "pseudo_stats_available": bool(ratios),
        "pseudo_windows_checked": checked,
        "pseudo_windows_with_geometry": len(ratios),
        "mean_valid_corr_ratio": None if not ratios else sum(ratios) / len(ratios),
        "mean_angular_pseudo_reprojection_error_deg": None if not angular_deg else sum(angular_deg) / len(angular_deg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--views-per-sample", type=int, default=4)
    parser.add_argument("--max-temporal-gap", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=504)
    parser.add_argument("--image-width", type=int, default=1008)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--num-query-per-pair", type=int, default=256)
    args = parser.parse_args()
    result = check_manifest_overlap(
        args.manifest,
        split=args.split,
        views_per_sample=args.views_per_sample,
        max_temporal_gap=args.max_temporal_gap,
        image_height=args.image_height,
        image_width=args.image_width,
        max_windows=args.max_windows,
        num_query_per_pair=args.num_query_per_pair,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
