"""Build Stage 1 manifests for Airsim360 Omni360-Scene DTW/NYC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

SCENES = ("DTW", "NYC")


def _frame_id_from_rgb(path: Path) -> int:
    match = re.fullmatch(r"panorama_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected Airsim RGB filename: {path.name}")
    return int(match.group(1))


def _scene_dirs(root: Path, scene: str) -> tuple[Path, Path]:
    lower = scene.lower()
    return root / scene / f"{lower}_Raw", root / scene / f"{lower}_Depth"


def _path_str(path: Path) -> str:
    return path.as_posix()


def build_airsim_records(
    root: str | Path,
    *,
    train_ratio: float = 0.9,
    scenes: tuple[str, ...] = SCENES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build DTW/NYC Stage 1 manifest records without copying source data."""

    root_path = Path(root)
    records: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "dataset_root": _path_str(root_path),
        "used_scenes": list(scenes),
        "scene_stats": {},
        "known_issues": [
            "No GT pose/trajectory file was found during Stage 1.5 audit; training uses frozen PanoVGGT init pose/depth.",
            "Airsim depth is .h5 with key 'depth'; values at 1000.0 are treated as far-plane/invalid for sanity reporting.",
        ],
    }
    for scene in scenes:
        raw_dir, depth_dir = _scene_dirs(root_path, scene)
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Missing Airsim RGB directory: {raw_dir}")
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing Airsim depth directory: {depth_dir}")
        rgb_files = sorted(raw_dir.glob("panorama_*.png"), key=_frame_id_from_rgb)
        split_cut = int(len(rgb_files) * float(train_ratio))
        depth_count = 0
        missing_depth: list[str] = []
        for idx, rgb in enumerate(rgb_files):
            frame_id = _frame_id_from_rgb(rgb)
            depth = depth_dir / f"Depth_{frame_id}.h5"
            if depth.exists():
                depth_count += 1
                depth_path: Path | None = depth
            else:
                missing_depth.append(_path_str(depth))
                depth_path = None
            split = "train" if idx < split_cut else "val"
            records.append(
                {
                    "dataset": "Airsim360-Omni360-Scene",
                    "scene_id": scene,
                    "sequence_id": f"{scene}_{split}",
                    "frame_id": frame_id,
                    "rgb_path": _path_str(rgb),
                    "depth_path": None if depth_path is None else _path_str(depth_path),
                    "pose_path": None,
                    "timestamp": float(frame_id),
                    "split": split,
                    "domain": "outdoor",
                }
            )
        audit["scene_stats"][scene] = {
            "rgb_dir": _path_str(raw_dir),
            "depth_dir": _path_str(depth_dir),
            "rgb_frames": len(rgb_files),
            "depth_frames": depth_count,
            "missing_depth": len(missing_depth),
            "train_records": split_cut,
            "val_records": len(rgb_files) - split_cut,
            "rgb_resolution": "2048x1024",
            "depth_format": "h5 key=depth, shape=1024x2048, float32",
            "pose_format": None,
        }
    return records, audit


def build_debug_records(records: list[dict[str, Any]], *, max_frames_per_scene: int = 128) -> list[dict[str, Any]]:
    """Build a small deterministic smoke manifest from train records."""

    debug: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_records = [
            record for record in records
            if record["scene_id"] == scene and record["split"] == "train"
        ][: int(max_frames_per_scene)]
        debug.extend(scene_records)
    return debug


def write_manifest(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/disk1/lanboyang/Datasets/Airsim360/Omni360-Scene")
    parser.add_argument("--output", default="data/stage1_airsim_dtw_nyc_manifest.json")
    parser.add_argument("--debug-output", default="data/stage1_airsim_dtw_nyc_debug_manifest.json")
    parser.add_argument("--audit-output", default="docs/stage1_airsim_dtw_nyc_dataset_audit.md")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--debug-max-frames-per-scene", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records, audit = build_airsim_records(args.root, train_ratio=args.train_ratio)
    debug = build_debug_records(records, max_frames_per_scene=args.debug_max_frames_per_scene)
    if not args.dry_run:
        write_manifest(args.output, records)
        write_manifest(args.debug_output, debug)
        audit_path = Path(args.audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(_audit_markdown(audit), encoding="utf-8")
    print(json.dumps({"records": len(records), "debug_records": len(debug), **audit}, indent=2))


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage 1.5 Airsim DTW/NYC Dataset Audit",
        "",
        f"- dataset root: `{audit['dataset_root']}`",
        "- used scenes: `DTW`, `NYC`",
        "- domain: `outdoor`",
        "- split: contiguous 90% train / 10% val per scene",
        "",
        "## Scene Summary",
        "",
        "| scene | RGB frames | depth frames | train | val | RGB resolution | depth format | pose |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for scene, stats in audit["scene_stats"].items():
        lines.append(
            f"| {scene} | {stats['rgb_frames']} | {stats['depth_frames']} | "
            f"{stats['train_records']} | {stats['val_records']} | "
            f"{stats['rgb_resolution']} | {stats['depth_format']} | none found |"
        )
    lines.extend(["", "## Known Issues", ""])
    for issue in audit["known_issues"]:
        lines.append(f"- {issue}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
