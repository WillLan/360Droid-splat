"""Evaluate Stage 3 BA with exact GT spherical correspondences.

This diagnostic keeps the Stage 3 source queries, factor selection, and factor
weights fixed, but replaces every selected Adapter target bearing with the
continuous bearing obtained from GT depth and GT camera poses.  It compares:

1. Adapter correspondences with Stage 2 depth;
2. GT correspondences with Stage 2 depth;
3. GT correspondences with GT depth.

Each arm is evaluated with both the formal BA configuration and a wider
diagnostic trust region.  GT geometry is used only in this standalone
evaluator and never enters the training or runtime paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.stage3_spherical_ba_refiner_dataset import stage3_collate
from geometry.spherical_erp import sample_erp_with_wrap, unit_ray_to_erp_pixel
from models.spherical_selfi_stage3_ba import Stage3MatchCache
from tools.evaluate_stage3_ba import _pose_metrics, _resize_gt_depth
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


def build_gt_oracle_match_cache(
    reference: Stage3MatchCache,
    gt_depth: torch.Tensor,
    gt_valid_depth: torch.Tensor,
    gt_poses_c2w: torch.Tensor,
    *,
    min_depth: float = 0.05,
    max_depth: float = 20.0,
    depth_consistency_abs: float = 0.05,
    depth_consistency_rel: float = 0.03,
) -> Stage3MatchCache:
    """Replace selected target bearings with exact GT projections.

    Source UVs, selected factor positions, and factor weights are inherited
    from ``reference`` so that Adapter and oracle arms use the same factor
    budget and spatial support.  GT depth additionally rejects occluded or
    invalid projections.
    """

    if gt_depth.ndim != 5 or int(gt_depth.shape[2]) != 1:
        raise ValueError("gt_depth must have shape BxSx1xHxW.")
    if tuple(gt_valid_depth.shape) != tuple(gt_depth.shape):
        raise ValueError("gt_valid_depth must match gt_depth.")
    batch, views, _, height, width = (int(value) for value in gt_depth.shape)
    if tuple(gt_poses_c2w.shape) != (batch, views, 4, 4):
        raise ValueError("gt_poses_c2w must have shape BxSx4x4.")
    if reference.batch_size != batch or reference.num_views != views:
        raise ValueError("Reference cache and GT tensors must share B/S dimensions.")

    device = gt_depth.device
    dtype = torch.float32
    source_uv = reference.source_uv.to(device=device, dtype=dtype)
    source_ray = reference.source_ray.to(device=device, dtype=dtype)
    poses = gt_poses_c2w.to(device=device, dtype=dtype)
    depths = gt_depth.to(device=device, dtype=dtype)
    valid_depth = gt_valid_depth.to(device=device).bool()
    source_depth = sample_erp_with_wrap(depths, source_uv)[..., 0]
    sampled_source_valid = sample_erp_with_wrap(valid_depth.float(), source_uv)[..., 0] > 0.5
    source_valid = (
        sampled_source_valid
        & torch.isfinite(source_depth)
        & (source_depth >= float(min_depth))
        & (source_depth <= float(max_depth))
        & torch.isfinite(source_ray).all(dim=-1)
    )

    edges = reference.edges.to(device=device)
    edge_count = int(edges.shape[0])
    query_count = reference.queries_per_source
    target_uv = torch.empty(batch, edge_count, query_count, 2, device=device, dtype=dtype)
    target_ray = torch.empty(batch, edge_count, query_count, 3, device=device, dtype=dtype)
    target_valid = torch.zeros(batch, edge_count, query_count, device=device, dtype=torch.bool)
    oracle_valid = torch.zeros_like(target_valid)

    for edge_index, pair in enumerate(edges.tolist()):
        source_index, target_index = int(pair[0]), int(pair[1])
        point_source = source_depth[:, source_index, :, None] * source_ray[:, source_index]
        source_pose = poses[:, source_index]
        target_pose = poses[:, target_index]
        point_world = torch.einsum(
            "bij,bqj->bqi", source_pose[:, :3, :3], point_source
        ) + source_pose[:, None, :3, 3]
        point_target = torch.einsum(
            "bij,bqj->bqi",
            target_pose[:, :3, :3].transpose(1, 2),
            point_world - target_pose[:, None, :3, 3],
        )
        projected_depth = point_target.norm(dim=-1)
        bearing = F.normalize(point_target, dim=-1, eps=1.0e-8)
        uv = unit_ray_to_erp_pixel(bearing, height, width)
        sampled_target_depth = sample_erp_with_wrap(
            depths[:, target_index], uv
        )[..., 0]
        sampled_target_valid = sample_erp_with_wrap(
            valid_depth[:, target_index].float(), uv
        )[..., 0] > 0.5
        consistency = (
            (sampled_target_depth - projected_depth).abs()
            <= float(depth_consistency_abs)
            + float(depth_consistency_rel) * sampled_target_depth.abs().clamp_min(1.0e-6)
        )
        finite = (
            torch.isfinite(projected_depth)
            & (projected_depth > 1.0e-6)
            & torch.isfinite(bearing).all(dim=-1)
            & torch.isfinite(uv).all(dim=-1)
            & torch.isfinite(sampled_target_depth)
        )
        geometry_valid = (
            source_valid[:, source_index]
            & sampled_target_valid
            & consistency
            & finite
        )
        target_uv[:, edge_index] = uv
        target_ray[:, edge_index] = bearing
        target_valid[:, edge_index] = sampled_target_valid & finite
        oracle_valid[:, edge_index] = (
            reference.valid_mask[:, edge_index].to(device=device).bool()
            & geometry_valid
        )

    factor_weight = (
        torch.ones_like(oracle_valid, dtype=dtype)
        if reference.factor_weight is None
        else reference.factor_weight.to(device=device, dtype=dtype)
    )
    return Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=source_depth,
        source_valid=source_valid,
        edges=edges,
        target_uv=target_uv,
        target_ray=target_ray,
        top1_cosine=torch.ones_like(factor_weight),
        top2_margin=torch.ones_like(factor_weight),
        entropy=torch.zeros_like(factor_weight),
        valid_mask=oracle_valid,
        factor_weight=factor_weight,
        mutual_mask=oracle_valid.clone(),
        target_valid=target_valid,
        metadata={
            "oracle": True,
            "source_selection": "reference_cache",
            "factor_selection": "reference_cache_intersect_gt_visibility",
            "min_depth": float(min_depth),
            "max_depth": float(max_depth),
            "depth_consistency_abs": float(depth_consistency_abs),
            "depth_consistency_rel": float(depth_consistency_rel),
        },
    )


def _solver_config(
    config: dict[str, Any],
    *,
    diagnostic: bool,
    diagnostic_iterations: int,
    diagnostic_max_pose_update_deg: float,
) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    if diagnostic:
        selected["ba"].update(
            {
                "iterations": int(diagnostic_iterations),
                "max_pose_update_deg": float(diagnostic_max_pose_update_deg),
                "lm_max_trials": max(8, int(selected["ba"].get("lm_max_trials", 4))),
            }
        )
    return selected


def _ba_metrics(output: Any, prefix: str) -> dict[str, float]:
    metrics = {
        f"{prefix}/accepted_ratio": float(output.accepted.float().mean().cpu()),
        f"{prefix}/initial_residual_deg": float(output.initial_median_residual_deg.mean().cpu()),
        f"{prefix}/final_residual_deg": float(output.final_median_residual_deg.mean().cpu()),
    }
    for key in (
        "initial_objective",
        "final_objective",
        "accepted_steps",
        "final_damping",
        "gain_ratio_mean",
        "num_factors",
    ):
        values = [float(item[key]) for item in output.diagnostics if key in item]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            metrics[f"{prefix}/{key}"] = sum(finite) / len(finite)
    return metrics


def _accumulate(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def evaluate_oracle(
    config: dict[str, Any],
    *,
    max_batches: int,
    start_batch: int = 0,
    diagnostic_iterations: int = 8,
    diagnostic_max_pose_update_deg: float = 0.5,
) -> dict[str, Any]:
    """Run the six-arm Adapter/GT-depth/GT-correspondence BA audit."""

    if not bool(config.get("stage3", {}).get("enabled", False)):
        raise ValueError("Stage 3 must be enabled for BA oracle evaluation.")
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
    formal_ba = build_ba(config)
    diagnostic_config = _solver_config(
        config,
        diagnostic=True,
        diagnostic_iterations=diagnostic_iterations,
        diagnostic_max_pose_update_deg=diagnostic_max_pose_update_deg,
    )
    diagnostic_ba = build_ba(diagnostic_config)
    solvers = {"formal": formal_ba, "diagnostic": diagnostic_ba}

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
        adapter_cache = _build_match_cache(
            features,
            stage2.refined_depth,
            config,
            step=index,
            static_valid_mask=stage2.valid_mask,
        )
        gt_pose = batch["gt_poses_c2w"].to(train_device)
        gt_depth, gt_valid = _resize_gt_depth(
            batch["gt_depths"].to(train_device),
            batch["gt_valid_depth"].to(train_device),
            stage2.image_size,
        )
        oracle_cache = build_gt_oracle_match_cache(
            adapter_cache,
            gt_depth,
            gt_valid,
            gt_pose,
            min_depth=float(config["matching"].get("min_depth", 0.05)),
            max_depth=float(config["matching"].get("max_depth", 20.0)),
        )

        metrics: dict[str, float] = {}
        metrics.update(_pose_metrics(stage2.poses_c2w, gt_pose, "initial"))
        adapter_valid = adapter_cache.valid_mask.bool()
        oracle_valid = oracle_cache.valid_mask.bool()
        overlap = adapter_valid & oracle_valid
        metrics["matching/adapter_valid_factors"] = float(adapter_valid.sum().cpu())
        metrics["matching/oracle_valid_factors"] = float(oracle_valid.sum().cpu())
        if bool(overlap.any()):
            dot = (
                adapter_cache.target_ray.to(train_device)[overlap]
                * oracle_cache.target_ray[overlap]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            angular = torch.rad2deg(torch.acos(dot))
            metrics["matching/adapter_to_oracle_mean_deg"] = float(angular.mean().cpu())
            metrics["matching/adapter_to_oracle_median_deg"] = float(angular.median().cpu())
            metrics["matching/adapter_to_oracle_p90_deg"] = float(angular.quantile(0.9).cpu())

        for solver_name, solver in solvers.items():
            arms = (
                ("adapter_stage2", adapter_cache, stage2.refined_depth),
                ("gt_stage2", oracle_cache, stage2.refined_depth),
                ("gt_gtdepth", oracle_cache, gt_depth),
            )
            for arm_name, cache, solver_depth in arms:
                output = solver(stage2.poses_c2w, solver_depth, cache)
                prefix = f"{solver_name}/{arm_name}"
                metrics.update(_pose_metrics(output.poses_c2w, gt_pose, prefix))
                metrics.update(_ba_metrics(output, prefix))

        for key in list(metrics):
            if "/pose_" not in key or key.startswith("initial/"):
                continue
            suffix = key.split("/pose_", 1)[1]
            initial_key = f"initial/pose_{suffix}"
            if initial_key in metrics:
                prefix = key.rsplit("/pose_", 1)[0]
                metrics[f"{prefix}/delta_pose_{suffix}"] = metrics[key] - metrics[initial_key]
        metrics["profile/batch_sec"] = time.perf_counter() - batch_start
        _accumulate(aggregate, metrics)
        records.append(
            {
                "batch_index": index,
                "frame_ids": batch["frame_ids"].tolist(),
                "metrics": metrics,
            }
        )
        print(
            f"evaluated oracle BA batch {len(records)}/{max_batches} (dataset batch {index})",
            flush=True,
        )

    count = len(records)
    if count == 0:
        raise RuntimeError("BA oracle evaluation dataset produced no batches.")
    return {
        "format": "spherical_stage3_ba_gt_oracle_evaluation_v1",
        "num_batches": count,
        "start_batch": int(start_batch),
        "wall_sec": time.perf_counter() - wall_start,
        "formal_ba_config": dict(config.get("ba", {})),
        "diagnostic_ba_config": dict(diagnostic_config.get("ba", {})),
        "oracle_config": {
            "factor_selection": "adapter_reference_intersect_gt_visibility",
            "depth_consistency_abs": 0.05,
            "depth_consistency_rel": 0.03,
        },
        "mean": {key: value / count for key, value in aggregate.items()},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--diagnostic-iterations", type=int, default=8)
    parser.add_argument("--diagnostic-max-pose-update-deg", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = evaluate_oracle(
        config,
        max_batches=max(1, int(args.max_batches)),
        start_batch=max(0, int(args.start_batch)),
        diagnostic_iterations=max(1, int(args.diagnostic_iterations)),
        diagnostic_max_pose_update_deg=max(1.0e-6, float(args.diagnostic_max_pose_update_deg)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["mean"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
