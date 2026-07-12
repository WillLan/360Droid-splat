"""Evaluate frozen Stage 3 BA without Refiner training or Gaussian rendering.

This is the fast path for BA solver, gauge, and factor-selection sweeps.  It
uses the real frozen PanoVGGT, adapter, Stage 2 head, validation windows, and GT
pose/depth diagnostics, but deliberately skips the recurrent Refiner,
rasterizer, backward pass, and optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data.stage3_spherical_ba_refiner_dataset import stage3_collate
from losses.spherical_stage3_refinement_loss import aligned_pose_metrics, depth_metrics
from training.train_spherical_ba_recurrent_refiner import (
    _apply_ba,
    _build_match_cache,
    _resolve_device,
    _seed,
    _sha256_file,
    build_ba,
    build_dataset,
    load_config,
)
from training.train_spherical_selfi_gaussian_head import (
    _freeze,
    build_frozen_feature_stack,
    build_head,
    extract_frozen_inputs,
    load_stage2_checkpoint,
)


def _accumulate(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def _pose_metrics(predicted: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
    total: dict[str, float] = {}
    for batch in range(int(predicted.shape[0])):
        for key, value in aligned_pose_metrics(predicted[batch], target[batch]).items():
            total[key] = total.get(key, 0.0) + float(value) / int(predicted.shape[0])
    return {f"{prefix}/pose_{key}": value for key, value in total.items()}


def _resize_gt_depth(
    depth: torch.Tensor,
    valid: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(depth.shape[-2:]) == tuple(target_size):
        return depth, valid
    batch, views = int(depth.shape[0]), int(depth.shape[1])
    depth = torch.nn.functional.interpolate(
        depth.reshape(batch * views, 1, *depth.shape[-2:]),
        size=target_size,
        mode="nearest",
    ).reshape(batch, views, 1, *target_size)
    valid = torch.nn.functional.interpolate(
        valid.float().reshape(batch * views, 1, *valid.shape[-2:]),
        size=target_size,
        mode="nearest",
    ).reshape(batch, views, 1, *target_size) > 0.5
    return depth, valid


def evaluate(
    config: dict[str, Any],
    *,
    max_batches: int,
    start_batch: int = 0,
) -> dict[str, Any]:
    if not bool(config.get("stage3", {}).get("enabled", False)):
        raise ValueError("Stage 3 must be enabled for BA evaluation.")
    train_cfg = config["train"]
    _seed(int(train_cfg.get("seed", 1234)))
    train_device = _resolve_device(str(train_cfg.get("train_device", "auto")))
    feature_device = _resolve_device(str(train_cfg.get("feature_device", "auto")))
    wrapper, adapter, adapter_sha, _ = build_frozen_feature_stack(config, device=feature_device)
    head = _freeze(build_head(config, device=train_device))
    stage2_path = config.get("stage2_checkpoint", {}).get("path")
    stage2_sha = _sha256_file(stage2_path)
    expected_stage2_sha = config.get("stage2_checkpoint", {}).get("sha256")
    if expected_stage2_sha is not None and stage2_sha != str(expected_stage2_sha):
        raise ValueError(
            f"Stage 2 checkpoint SHA256 mismatch: expected {expected_stage2_sha}, got {stage2_sha}."
        )
    if stage2_path:
        load_stage2_checkpoint(
            stage2_path,
            head=head,
            expected_adapter_sha256=adapter_sha,
            map_location=train_device,
        )
    elif bool(config.get("stage2_checkpoint", {}).get("required", False)):
        raise ValueError("stage2_checkpoint.path is required by this Stage 3 config.")

    dataset = build_dataset(config, split="val")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage3_collate,
        pin_memory=train_device.type == "cuda",
    )
    ba = build_ba(config)
    aggregate: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for index, batch in enumerate(loader):
        if index < int(start_batch):
            continue
        if len(records) >= int(max_batches):
            break
        batch_start = time.perf_counter()
        features, images, initial_depth, poses = extract_frozen_inputs(
            wrapper,
            adapter,
            batch["images"],
            feature_device=feature_device,
            train_device=train_device,
            head_size=(int(config["image"]["head_height"]), int(config["image"]["head_width"])),
            feature_amp=bool(train_cfg.get("amp", False)),
        )
        with torch.no_grad(), torch.amp.autocast(
            device_type=train_device.type,
            dtype=torch.bfloat16,
            enabled=bool(train_cfg.get("amp", False)) and train_device.type == "cuda",
        ):
            stage2 = head(
                features,
                images,
                initial_depth,
                poses,
                frame_ids=batch["frame_ids"].to(train_device),
            )
        cache = _build_match_cache(
            features,
            stage2.refined_depth,
            config,
            step=index,
            static_valid_mask=stage2.valid_mask,
        )
        ba_observation, output = _apply_ba(stage2, ba, cache)
        gt_pose = batch["gt_poses_c2w"].to(train_device)
        gt_depth, gt_valid = _resize_gt_depth(
            batch["gt_depths"].to(train_device),
            batch["gt_valid_depth"].to(train_device),
            stage2.image_size,
        )
        metrics: dict[str, float] = {}
        metrics.update(_pose_metrics(stage2.poses_c2w, gt_pose, "initial"))
        metrics.update(_pose_metrics(ba_observation.poses_c2w, gt_pose, "ba0"))
        metrics.update(
            {
                f"initial/depth_{key}": value
                for key, value in depth_metrics(stage2.refined_depth, gt_depth, gt_valid).items()
            }
        )
        metrics.update(
            {
                f"ba0/depth_{key}": value
                for key, value in depth_metrics(ba_observation.refined_depth, gt_depth, gt_valid).items()
            }
        )
        valid = cache.valid_mask.bool()
        metrics["matching/valid_factors"] = float(valid.sum().cpu())
        if bool(valid.any()):
            for name, tensor in (
                ("top1_cosine", cache.top1_cosine),
                ("top2_margin", cache.top2_margin),
                ("entropy", cache.entropy),
                ("factor_weight", cache.factor_weight),
            ):
                if tensor is None:
                    continue
                selected = tensor[valid].detach().float()
                metrics[f"matching/{name}_mean"] = float(selected.mean().cpu())
                for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
                    metrics[f"matching/{name}_p{int(quantile * 100)}"] = float(
                        selected.quantile(quantile).cpu()
                    )
        metrics["ba0/accepted_ratio"] = float(output.accepted.float().mean().cpu())
        metrics["ba0/initial_residual_deg"] = float(output.initial_median_residual_deg.mean().cpu())
        metrics["ba0/final_residual_deg"] = float(output.final_median_residual_deg.mean().cpu())
        for key in (
            "initial_objective",
            "final_objective",
            "accepted_steps",
            "final_damping",
            "gain_ratio_mean",
            "gauge_scale_mean",
            "num_factors",
            "initial_geometry_residual_p50_deg",
            "initial_geometry_residual_p90_deg",
            "initial_parallax_p10_deg",
            "initial_parallax_p50_deg",
        ):
            values = [float(item[key]) for item in output.diagnostics if key in item]
            finite = [value for value in values if torch.isfinite(torch.tensor(value))]
            if finite:
                metrics[f"ba0/{key}"] = sum(finite) / len(finite)
        metrics["profile/batch_sec"] = time.perf_counter() - batch_start
        _accumulate(aggregate, metrics)
        records.append(
            {
                "batch_index": index,
                "frame_ids": batch["frame_ids"].tolist(),
                "metrics": metrics,
            }
        )
        print(f"evaluated BA batch {len(records)}/{max_batches} (dataset batch {index})", flush=True)

    count = len(records)
    if count == 0:
        raise RuntimeError("BA evaluation dataset produced no batches.")
    mean = {key: value / count for key, value in aggregate.items()}
    for key in list(mean):
        if key.startswith("initial/"):
            suffix = key[len("initial/") :]
            ba_key = f"ba0/{suffix}"
            if ba_key in mean:
                mean[f"delta/{suffix}"] = mean[ba_key] - mean[key]
    return {
        "format": "spherical_stage3_ba_evaluation_v1",
        "num_batches": count,
        "wall_sec": time.perf_counter() - wall_start,
        "ba_config": dict(config.get("ba", {})),
        "matching_config": dict(config.get("matching", {})),
        "mean": mean,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = evaluate(
        config,
        max_batches=max(1, int(args.max_batches)),
        start_batch=max(0, int(args.start_batch)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["mean"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
