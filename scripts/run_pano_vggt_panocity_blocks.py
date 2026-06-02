"""Run PanoVGGT-Long SLAM over PanoCity blocks sequentially."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import yaml

from frontend.pano_droid.dataset import discover_erp_images
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem, load_config


IMAGE_DIR_NAMES = ("pano_images", "images", "rgb")


def block_image_root(block_dir: str | Path) -> Path:
    block = Path(block_dir)
    for name in IMAGE_DIR_NAMES:
        folder = block / name
        if folder.is_dir():
            try:
                discover_erp_images(str(folder))
            except FileNotFoundError:
                continue
            return folder
    discover_erp_images(str(block))
    return block


def discover_panocity_blocks(root: str | Path, *, block_glob: str = "beijing_block*") -> list[Path]:
    root_path = Path(root)
    blocks = []
    for candidate in sorted(root_path.glob(block_glob)):
        if not candidate.is_dir():
            continue
        try:
            block_image_root(candidate)
        except FileNotFoundError:
            continue
        blocks.append(candidate)
    if not blocks:
        raise FileNotFoundError(f"No PanoCity blocks with ERP images found under {root_path}")
    return blocks


def build_block_config(
    base_config: dict,
    block_dir: str | Path,
    *,
    output_root: str | Path,
    frames_per_block: int,
    run_name: str | None = None,
) -> dict:
    block = Path(block_dir)
    cfg = copy.deepcopy(base_config)
    ds_cfg = cfg.setdefault("Dataset", {})
    ds_cfg["synthetic"] = False
    ds_cfg["dataset_path"] = str(block_image_root(block))
    ds_cfg["sequence"] = None
    ds_cfg["begin"] = 0
    ds_cfg["end"] = int(frames_per_block)

    out_dir = Path(output_root) / block.name
    cfg.setdefault("Results", {})["save_dir"] = str(out_dir)

    wb_cfg = cfg.setdefault("WeightsAndBiases", {})
    base_name = run_name or wb_cfg.get("run_name") or "panovggt_long_panocity"
    wb_cfg["run_name"] = f"{base_name}_{block.name}"
    wb_cfg["group"] = str(base_name)
    tags = list(wb_cfg.get("tags") or [])
    for tag in ("panovggt_long", "panocity", block.name):
        if tag not in tags:
            tags.append(tag)
    wb_cfg["tags"] = tags
    return cfg


def _select_blocks(blocks: Iterable[Path], names: list[str] | None, max_blocks: int | None) -> list[Path]:
    selected = list(blocks)
    if names:
        wanted = set(names)
        selected = [b for b in selected if b.name in wanted]
        missing = sorted(wanted - {b.name for b in selected})
        if missing:
            raise FileNotFoundError(f"Requested blocks not found: {', '.join(missing)}")
    if max_blocks is not None:
        selected = selected[: int(max_blocks)]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pano_vggt_long_panocity_beijing.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--block-glob", default=None)
    parser.add_argument("--block", action="append", default=None, help="Run only the named block. Can repeat.")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--frames-per-block", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    ds_cfg = base_cfg.setdefault("Dataset", {})
    root = Path(args.dataset_root or ds_cfg.get("dataset_path") or "")
    if not str(root):
        raise ValueError("Dataset.dataset_path or --dataset-root is required.")
    block_glob = args.block_glob or ds_cfg.get("block_glob") or "beijing_block*"
    frames_per_block = int(args.frames_per_block or ds_cfg.get("frames_per_block") or ds_cfg.get("end") or 300)
    output_root = Path(args.output_root or base_cfg.get("Results", {}).get("save_dir", "outputs/panocity_blocks"))

    if args.wandb:
        base_cfg.setdefault("WeightsAndBiases", {})["enabled"] = True
    if args.wandb_mode is not None:
        base_cfg.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    if args.run_name:
        base_cfg.setdefault("WeightsAndBiases", {})["run_name"] = args.run_name

    blocks = _select_blocks(discover_panocity_blocks(root, block_glob=block_glob), args.block, args.max_blocks)
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "block_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for block in blocks:
        cfg = build_block_config(
            base_cfg,
            block,
            output_root=output_root,
            frames_per_block=frames_per_block,
            run_name=args.run_name,
        )
        config_path = config_dir / f"{block.name}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        record = {
            "block": block.name,
            "config": str(config_path),
            "dataset_path": cfg["Dataset"]["dataset_path"],
            "output_dir": cfg["Results"]["save_dir"],
            "frames_requested": frames_per_block,
            "status": "planned" if args.dry_run else "running",
        }
        print(json.dumps(record, indent=2), flush=True)
        if args.dry_run:
            summaries.append(record)
            continue
        try:
            summary = PanoDroidGSSlamSystem(cfg).run(max_frames=frames_per_block)
            record.update({"status": "ok", "summary": summary})
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            print(json.dumps(record, indent=2), flush=True)
            summaries.append(record)
            if args.stop_on_error:
                break
            continue
        summaries.append(record)
        print(json.dumps(record, indent=2), flush=True)

    summary_path = output_root / "summary_all_blocks.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps({"summary_all_blocks": str(summary_path), "blocks": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
