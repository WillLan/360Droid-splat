"""Evaluate all-pair and star-graph variants of the frozen Stage 3 BA.

The four variants share one unpruned, all-directed-edge Adapter match cache per
validation batch. They differ only in graph topology and per-edge reliability
retention, which isolates graph structure from query sampling and matching
noise.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
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
from torch.utils.data import DataLoader

from data.stage3_spherical_ba_refiner_dataset import stage3_collate
from models.spherical_selfi_stage3_ba import Stage3MatchCache
from tools.evaluate_stage3_ba import _pose_metrics
from tools.evaluate_stage3_ba_oracle import _ba_metrics
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


STAR_VARIANTS: dict[str, dict[str, Any]] = {
    "a_all_top10": {"topology": "all_directed", "keep_fraction": 0.10},
    "b_star_top10": {"topology": "star_forward", "keep_fraction": 0.10},
    "c_star_top40": {"topology": "star_forward", "keep_fraction": 0.40},
    "d_star_bidir_top20": {
        "topology": "star_bidirectional",
        "keep_fraction": 0.20,
    },
}


def topology_edge_mask(edges: torch.Tensor, topology: str) -> torch.Tensor:
    """Return retained directed edges for an anchor-frame star ablation."""

    if edges.ndim != 2 or int(edges.shape[-1]) != 2:
        raise ValueError("edges must have shape Ex2.")
    source, target = edges[:, 0], edges[:, 1]
    topology = str(topology).lower()
    if topology == "all_directed":
        return source != target
    if topology == "star_forward":
        return (source == 0) & (target != 0)
    if topology == "star_bidirectional":
        return (source != target) & ((source == 0) | (target == 0))
    raise ValueError(
        "topology must be 'all_directed', 'star_forward', or "
        "'star_bidirectional'."
    )


def select_match_cache_variant(
    raw_cache: Stage3MatchCache,
    *,
    topology: str,
    keep_fraction: float,
) -> Stage3MatchCache:
    """Select a topology and top weighted factors without rematching.

    Selection follows the production matcher's per-batch/per-edge ``ceil``
    semantics. All geometry, scores, provenance, and tensor ordering are
    shared with ``raw_cache``; only ``valid_mask`` and metadata are replaced.
    """

    fraction = float(keep_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1].")
    selected = torch.zeros_like(raw_cache.valid_mask, dtype=torch.bool)
    retained_edges = topology_edge_mask(raw_cache.edges, topology)
    weights = raw_cache.factor_weight
    if weights is None:
        weights = torch.ones_like(raw_cache.valid_mask, dtype=torch.float32)
    for batch_index in range(raw_cache.batch_size):
        edge_indices = torch.nonzero(retained_edges, as_tuple=False).flatten().tolist()
        for edge_index in edge_indices:
            candidates = torch.nonzero(
                raw_cache.valid_mask[batch_index, edge_index].bool(),
                as_tuple=False,
            ).flatten()
            if int(candidates.numel()) == 0:
                continue
            keep_count = max(1, int(math.ceil(fraction * int(candidates.numel()))))
            ranking = torch.topk(
                weights[batch_index, edge_index, candidates],
                k=keep_count,
                largest=True,
                sorted=False,
            ).indices
            selected[batch_index, edge_index, candidates[ranking]] = True

    metadata = dict(raw_cache.metadata)
    metadata.update(
        {
            "star_evaluation_topology": str(topology),
            "star_evaluation_keep_fraction": fraction,
            "star_evaluation_retained_edge_count": int(retained_edges.sum().item()),
            "star_evaluation_shared_raw_cache": True,
        }
    )
    return replace(raw_cache, valid_mask=selected, metadata=metadata)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _accumulate(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def _factor_metrics(cache: Stage3MatchCache, prefix: str) -> dict[str, float]:
    valid = cache.valid_mask.bool()
    edge_has_factor = valid.any(dim=0).any(dim=-1)
    metrics = {
        f"{prefix}/valid_factors": float(valid.sum().detach().cpu()),
        f"{prefix}/active_edges": float(edge_has_factor.sum().detach().cpu()),
    }
    if bool(valid.any()):
        for name, tensor in (
            ("top1_cosine", cache.top1_cosine),
            ("top2_margin", cache.top2_margin),
            ("entropy", cache.entropy),
            ("factor_weight", cache.factor_weight),
        ):
            if tensor is not None:
                metrics[f"{prefix}/{name}_mean"] = float(
                    tensor[valid].detach().float().mean().cpu()
                )
    return metrics


def evaluate_star(
    config: dict[str, Any],
    *,
    max_batches: int,
    start_batch: int = 0,
) -> dict[str, Any]:
    """Run the four-arm topology ablation on validation data."""

    if not bool(config.get("stage3", {}).get("enabled", False)):
        raise ValueError("Stage 3 must be enabled for star BA evaluation.")
    train_cfg = config["train"]
    _seed(int(train_cfg.get("seed", 1234)))
    train_device = _resolve_device(str(train_cfg.get("train_device", "auto")))
    feature_device = _resolve_device(str(train_cfg.get("feature_device", "auto")))
    wrapper, adapter, adapter_sha, _ = build_frozen_feature_stack(
        config, device=feature_device
    )
    head = _freeze(build_head(config, device=train_device))
    stage2_path = config.get("stage2_checkpoint", {}).get("path")
    stage2_sha = _sha256_file(stage2_path)
    expected_stage2_sha = config.get("stage2_checkpoint", {}).get("sha256")
    if expected_stage2_sha is not None and stage2_sha != str(expected_stage2_sha):
        raise ValueError(
            f"Stage 2 checkpoint SHA256 mismatch: expected {expected_stage2_sha}, "
            f"got {stage2_sha}."
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
    raw_config = copy.deepcopy(config)
    raw_config["matching"]["edge_topology"] = "all_directed"
    raw_config["matching"]["reliability_keep_fraction"] = 1.0
    aggregate: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    for index, batch in enumerate(loader):
        if index < int(start_batch):
            continue
        if len(records) >= int(max_batches):
            break
        batch_start = time.perf_counter()
        _synchronize(train_device)
        feature_start = time.perf_counter()
        features, images, initial_depth, poses = extract_frozen_inputs(
            wrapper,
            adapter,
            batch["images"],
            feature_device=feature_device,
            train_device=train_device,
            head_size=(
                int(config["image"]["head_height"]),
                int(config["image"]["head_width"]),
            ),
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
        _synchronize(train_device)
        feature_sec = time.perf_counter() - feature_start

        match_start = time.perf_counter()
        raw_cache = _build_match_cache(
            features,
            stage2.refined_depth,
            raw_config,
            step=index,
            static_valid_mask=stage2.valid_mask,
        )
        _synchronize(train_device)
        matching_sec = time.perf_counter() - match_start
        gt_pose = batch["gt_poses_c2w"].to(train_device)
        metrics: dict[str, float] = {
            "shared/profile_feature_stage2_sec": feature_sec,
            "shared/profile_raw_all_edge_matching_sec": matching_sec,
            "shared/raw_valid_factors": float(raw_cache.valid_mask.sum().cpu()),
        }
        metrics.update(_pose_metrics(stage2.poses_c2w, gt_pose, "initial"))

        for variant_name, variant_cfg in STAR_VARIANTS.items():
            selected_cache = select_match_cache_variant(raw_cache, **variant_cfg)
            metrics.update(_factor_metrics(selected_cache, variant_name))
            _synchronize(train_device)
            solve_start = time.perf_counter()
            ba_observation, output = _apply_ba(stage2, ba, selected_cache)
            _synchronize(train_device)
            metrics[f"{variant_name}/profile_ba_sec"] = time.perf_counter() - solve_start
            metrics.update(_pose_metrics(ba_observation.poses_c2w, gt_pose, variant_name))
            metrics.update(_ba_metrics(output, variant_name))

        metrics["shared/profile_batch_sec"] = time.perf_counter() - batch_start
        _accumulate(aggregate, metrics)
        records.append(
            {
                "batch_index": index,
                "frame_ids": batch["frame_ids"].tolist(),
                "metrics": metrics,
            }
        )
        print(
            f"evaluated star BA batch {len(records)}/{max_batches} "
            f"(dataset batch {index})",
            flush=True,
        )

    count = len(records)
    if count == 0:
        raise RuntimeError("Star BA evaluation dataset produced no batches.")
    mean = {key: value / count for key, value in aggregate.items()}
    for variant_name in STAR_VARIANTS:
        for metric_suffix in (
            "pose_rotation_mean_deg",
            "pose_rotation_median_deg",
            "pose_rotation_p90_deg",
            "pose_ate_scale_aligned",
            "pose_rpe_rotation_mean_deg",
            "pose_rpe_translation_direction_mean_deg",
        ):
            initial_key = f"initial/{metric_suffix}"
            variant_key = f"{variant_name}/{metric_suffix}"
            if initial_key in mean and variant_key in mean:
                mean[f"delta/{variant_name}/{metric_suffix}"] = (
                    mean[variant_key] - mean[initial_key]
                )

    return {
        "format": "spherical_stage3_ba_star_evaluation_v1",
        "num_batches": count,
        "start_batch": int(start_batch),
        "wall_sec": time.perf_counter() - wall_start,
        "variants": STAR_VARIANTS,
        "shared_raw_matching": True,
        "ba_config": dict(config.get("ba", {})),
        "matching_config": dict(config.get("matching", {})),
        "mean": mean,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--start-batch", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = evaluate_star(
        config,
        max_batches=max(1, int(args.max_batches)),
        start_batch=max(0, int(args.start_batch)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["mean"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
