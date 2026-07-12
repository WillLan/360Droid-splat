"""Summarize completed Stage 3 BA ablations from their saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


SNAPSHOT_METRICS = (
    "loo_l1",
    "loo_psnr",
    "loo_ssim",
    "pose_rotation_mean_deg",
    "pose_scale_aligned_ate",
    "pose_alignment_scale",
    "pose_rpe_rotation_mean_deg",
    "pose_rpe_translation",
    "pose_translation_direction_mean_deg",
    "depth_scale_aligned_absrel",
    "depth_scale_aligned_rmse",
    "depth_scale_aligned_delta1",
    "confidence_p50",
)


def _number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_checkpoint(checkpoint: Path, *, name: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = dict(payload.get("metrics", {}))
    config = dict(payload.get("config", {}))
    ba_config = dict(config.get("ba", {}))
    row: dict[str, Any] = {
        "name": name,
        "status": "complete",
        "step": int(payload.get("global_step", 0)),
        "dense_depth_mode": ba_config.get("dense_depth_mode"),
        "gauge_mode": ba_config.get("gauge_mode"),
        "solver_mode": ba_config.get("solver_mode"),
    }
    for snapshot in ("initial", "ba0", "refine3"):
        for metric in SNAPSHOT_METRICS:
            row[f"{snapshot}_{metric}"] = _number(metrics, f"val/{snapshot}/{metric}")
    for metric in SNAPSHOT_METRICS:
        initial = row.get(f"initial_{metric}")
        ba0 = row.get(f"ba0_{metric}")
        final = row.get(f"refine3_{metric}")
        row[f"ba0_delta_{metric}"] = None if initial is None or ba0 is None else ba0 - initial
        row[f"final_delta_{metric}"] = None if initial is None or final is None else final - initial
    for metric in (
        "accepted_ratio",
        "initial_residual_deg",
        "final_residual_deg",
        "initial_objective",
        "final_objective",
        "accepted_steps",
        "final_damping",
        "gain_ratio_mean",
        "gauge_scale_mean",
        "depth_scale_mean",
        "depth_shift_mean",
    ):
        row[f"train_ba0_{metric}"] = _number(metrics, f"ba0/{metric}")
    return row


def summarize_suite(suite_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for experiment in manifest["experiments"]:
        checkpoint = Path(experiment["output_dir"]) / "checkpoints" / "latest.pt"
        if checkpoint.is_file():
            rows.append(summarize_checkpoint(checkpoint, name=str(experiment["name"])))
        else:
            rows.append({"name": str(experiment["name"]), "status": "pending"})
    return rows


def write_summary(suite_dir: Path, rows: list[dict[str, Any]]) -> None:
    (suite_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (suite_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize_suite(args.suite_dir)
    write_summary(args.suite_dir, rows)
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
