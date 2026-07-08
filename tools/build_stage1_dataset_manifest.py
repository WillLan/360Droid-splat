"""Build a Stage 1 dataset manifest from an ERP image directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_manifest_records(
    root: str | Path,
    *,
    domain: str,
    split: str = "train",
    scene_id: str | None = None,
    sequence_id: str | None = None,
) -> list[dict]:
    root_path = Path(root)
    records = []
    for idx, path in enumerate(sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)):
        rel = path.relative_to(root_path)
        parts = rel.parts
        seq = sequence_id or (parts[0] if len(parts) > 1 else root_path.name)
        scene = scene_id or root_path.name
        stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
        frame_id = int(stem_digits) if stem_digits else idx
        records.append(
            {
                "scene_id": scene,
                "sequence_id": seq,
                "frame_id": frame_id,
                "rgb_path": str(path),
                "depth_path": None,
                "pose_path": None,
                "timestamp": float(idx),
                "split": split,
                "domain": str(domain).lower(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True, choices=["indoor", "outdoor"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--sequence-id", default=None)
    args = parser.parse_args()
    records = build_manifest_records(
        args.root,
        domain=args.domain,
        split=args.split,
        scene_id=args.scene_id,
        sequence_id=args.sequence_id,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(json.dumps({"output": str(out), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
