"""Isolated 100-frame local-BA matcher evaluation on the streaming frontend.

The global Gaussian backend is intentionally disabled.  Each four-frame
window is evaluated against its own GT poses before and after local BA, so the
result measures matcher/BA behavior without global graph or map optimization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch

from frontend.pano_droid.adapter import build_frontend_from_config
from losses.spherical_stage3_refinement_loss import aligned_pose_metrics
from system.pano_droid_gs_slam import iter_sequence_frames, load_config


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _mean_records(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record["metrics"]})
    output: dict[str, float] = {}
    for key in keys:
        values = [float(record["metrics"][key]) for record in records if key in record["metrics"]]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            output[key] = float(np.mean(finite, dtype=np.float64))
    return output


def _draw_metric_chart(records: list[dict[str, Any]], path: Path) -> None:
    panels = [
        ("scale_aligned_ate", "window scale-aligned ATE"),
        ("rotation_mean_deg", "window rotation mean (deg)"),
        ("rpe_translation", "window RPE translation"),
        ("rpe_rotation_mean_deg", "window RPE rotation (deg)"),
    ]
    width, panel_height = 1400, 280
    image = Image.new("RGB", (width, panel_height * len(panels)), "white")
    draw = ImageDraw.Draw(image)
    left, right = 90, width - 35
    colors = {"initial": (214, 39, 40), "refined": (31, 119, 180)}
    for panel, (metric, title) in enumerate(panels):
        top = panel * panel_height + 30
        bottom = (panel + 1) * panel_height - 45
        values = {
            prefix: [float(record["metrics"][f"{prefix}/{metric}"]) for record in records]
            for prefix in ("initial", "refined")
        }
        finite = [value for series in values.values() for value in series if math.isfinite(value)]
        upper = max(finite) if finite else 1.0
        lower = min(0.0, min(finite)) if finite else 0.0
        if upper <= lower:
            upper = lower + 1.0
        draw.line((left, top, left, bottom), fill=(80, 80, 80), width=1)
        draw.line((left, bottom, right, bottom), fill=(80, 80, 80), width=1)
        draw.text((left, top - 22), title, fill=(0, 0, 0))
        draw.text((8, top - 5), f"{upper:.4g}", fill=(0, 0, 0))
        draw.text((8, bottom - 8), f"{lower:.4g}", fill=(0, 0, 0))
        for prefix, series in values.items():
            points = []
            for index, value in enumerate(series):
                x = left + (right - left) * index / max(1, len(series) - 1)
                y = bottom - (bottom - top) * (value - lower) / (upper - lower)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=colors[prefix], width=3)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[prefix])
        draw.text((right - 210, top - 22), "initial", fill=colors["initial"])
        draw.text((right - 115, top - 22), "refined", fill=colors["refined"])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _wandb_start(config: dict[str, Any], output_dir: Path):
    cfg = dict(config.get("WeightsAndBiases", {}) or {})
    mode = str(cfg.get("mode", "online"))
    if not bool(cfg.get("enabled", False)) or mode == "disabled":
        return None, None, mode, None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Remote local-BA experiments require the wandb package") from exc
    kwargs = {
        "project": str(cfg.get("project", "360Droid-splat")),
        "entity": cfg.get("entity") or None,
        "name": cfg.get("run_name") or None,
        "group": cfg.get("group") or None,
        "tags": cfg.get("tags") or None,
        "dir": str(output_dir),
        "config": config,
    }
    try:
        return wandb.init(mode=mode, **kwargs), wandb, mode, None
    except Exception as exc:
        if mode != "online":
            raise
        return wandb.init(mode="offline", **kwargs), wandb, "offline", repr(exc)


def evaluate(config: dict[str, Any], *, max_frames: int) -> dict[str, Any]:
    if bool(config.get("SphericalSelfiGlobalBackend", {}).get("enabled", False)):
        raise ValueError("Local-BA isolation requires SphericalSelfiGlobalBackend.enabled=false")
    output_dir = Path(config.get("Results", {}).get("save_dir", "outputs/local_ba_isolated")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_enabled = bool(config.get("Visualization", {}).get("enabled", False))
    if not visualization_enabled:
        raise ValueError("Local-BA remote experiments require Visualization.enabled=true")
    run, wandb, wandb_mode, wandb_error = _wandb_start(config, output_dir)
    frontend = build_frontend_from_config(config)
    frontend.initialize({"evaluation": "local_ba_isolated"})
    records: list[dict[str, Any]] = []
    frame_count = 0
    wall_start = time.perf_counter()

    def consume() -> None:
        diagnostics = frontend.consume_local_ba_diagnostics()
        frontend.consume_local_gaussian_windows()
        for diagnostic in diagnostics:
            gt = diagnostic.get("gt_poses_c2w")
            if gt is None:
                raise RuntimeError(
                    f"Window {diagnostic['window_id']} has no GT poses; isolated BA metrics are undefined"
                )
            initial = aligned_pose_metrics(diagnostic["initial_poses_c2w"], gt)
            refined = aligned_pose_metrics(diagnostic["refined_poses_c2w"], gt)
            metrics = {
                **{f"initial/{key}": float(value) for key, value in initial.items()},
                **{f"refined/{key}": float(value) for key, value in refined.items()},
                **{
                    f"delta/{key}": float(refined[key] - initial[key])
                    for key in initial
                },
                "ba/accepted": float(bool(diagnostic["accepted"])),
                "matching/valid_factors": float(diagnostic["num_factors"]),
                "profile/matching_sec": float(diagnostic["matching_sec"]),
                "profile/ba_sec": float(diagnostic["ba_sec"]),
            }
            if diagnostic["initial_median_residual_deg"] is not None:
                metrics["ba/initial_median_residual_deg"] = float(
                    diagnostic["initial_median_residual_deg"]
                )
            if diagnostic["final_median_residual_deg"] is not None:
                metrics["ba/final_median_residual_deg"] = float(
                    diagnostic["final_median_residual_deg"]
                )
            record = {
                "window_id": int(diagnostic["window_id"]),
                "frame_ids": list(diagnostic["frame_ids"]),
                "matcher": str(diagnostic["matcher"]),
                "metrics": metrics,
                "ba_diagnostics": _json_value(diagnostic.get("ba_diagnostics")),
                "matching_metadata": _json_value(diagnostic.get("matching_metadata")),
            }
            records.append(record)
            if run is not None:
                payload = {
                    f"local_ba/{key}": value for key, value in metrics.items()
                }
                payload.update(
                    {
                        "local_ba/window_id": int(diagnostic["window_id"]),
                        "local_ba/frame_start": int(diagnostic["frame_ids"][0]),
                        "local_ba/frame_end": int(diagnostic["frame_ids"][-1]),
                    }
                )
                run.log(payload, step=int(diagnostic["window_id"]) + 1)

    try:
        for frame in iter_sequence_frames(config):
            if frame_count >= int(max_frames):
                break
            frontend.track(frame)
            ready = frontend.pop_ready_outputs()
            for output in ready:
                frontend.sky_mask_for_frame(int(output.frame_id))
            consume()
            frame_count += 1
        for output in frontend.flush():
            frontend.sky_mask_for_frame(int(output.frame_id))
        consume()
        if not records:
            raise RuntimeError("No complete or partial local windows were produced")
        mean = _mean_records(records)
        chart_path = output_dir / "visualizations" / "local_ba_window_gt_metrics.png"
        _draw_metric_chart(records, chart_path)
        result = {
            "format": "spherical_selfi_local_ba_isolated_v1",
            "frames": int(frame_count),
            "windows": int(len(records)),
            "matcher": records[0]["matcher"],
            "wall_sec": float(time.perf_counter() - wall_start),
            "mean": mean,
            "records": records,
            "visualization": str(chart_path),
            "wandb_mode": wandb_mode,
            "wandb_init_error": wandb_error,
            "wandb_run_url": None if run is None else getattr(run, "url", None),
        }
        result_path = output_dir / "local_ba_metrics.json"
        result_path.write_text(
            json.dumps(_json_value(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if run is not None:
            run.log(
                {
                    **{f"local_ba_mean/{key}": value for key, value in mean.items()},
                    "local_ba/window_gt_metrics": wandb.Image(str(chart_path)),
                },
                step=len(records) + 1,
            )
            run.summary.update(_json_value(result))
            run.finish()
        return result
    except BaseException as exc:
        if run is not None:
            run.summary.update({"failed": True, "error": repr(exc)})
            run.finish()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    result = evaluate(config, max_frames=max(1, int(args.max_frames)))
    print(json.dumps(_json_value({"mean": result["mean"], "windows": result["windows"]}), indent=2))


if __name__ == "__main__":
    main()
