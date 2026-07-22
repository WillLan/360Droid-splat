"""Archive metadata and safely clean historical main-worktree outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/mnt/disk1/lanboyang/Project/360Droid-splat")
METADATA_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".txt", ".log"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_inventory(root: Path) -> dict[str, Any]:
    files = 0
    size = 0
    newest = 0
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            files += 1
            size += int(stat.st_size)
            newest = max(newest, int(stat.st_mtime_ns))
    return {"files": files, "size_bytes": size, "newest_mtime_ns": newest}


def _protected_top_levels(outputs: Path, weight_manifest: dict[str, Any]) -> set[str]:
    protected: set[str] = set()
    resolved_outputs = outputs.resolve()
    for entry in weight_manifest.get("files", []):
        source = Path(str(entry["source"]))
        try:
            relative = source.resolve().relative_to(resolved_outputs)
        except ValueError:
            continue
        if relative.parts:
            protected.add(relative.parts[0])
    return protected


def _representative_images(top: Path) -> set[Path]:
    selected = {
        path
        for path in top.rglob("*.png")
        if "trajectory" in path.name.lower()
    }
    panels = sorted(top.rglob("final_all_frames/render_vs_gt/*.png"))
    if panels:
        selected.update((panels[0], panels[len(panels) // 2], panels[-1]))
    return selected


def create_archive(outputs: Path, archive: Path, weight_manifest_path: Path) -> Path:
    if not outputs.is_dir():
        raise FileNotFoundError(outputs)
    archive.mkdir(parents=True, exist_ok=True)
    weight_manifest = json.loads(weight_manifest_path.read_text(encoding="utf-8"))
    protected = _protected_top_levels(outputs, weight_manifest)
    inventory: list[dict[str, Any]] = []
    archived_files: list[dict[str, Any]] = []
    deletion_entries: list[dict[str, Any]] = []
    for top in sorted(path for path in outputs.iterdir() if path.is_dir()):
        record = {"name": top.name, **_tree_inventory(top), "protected_weight_source": top.name in protected}
        inventory.append(record)
        selected = {
            path for path in top.rglob("*")
            if path.is_file() and path.suffix.lower() in METADATA_SUFFIXES
        }
        selected.update(_representative_images(top))
        for source in sorted(selected):
            relative = source.relative_to(outputs)
            destination = archive / "metadata" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            archived_files.append(
                {
                    "source": str(source),
                    "archive": str(destination),
                    "size_bytes": int(destination.stat().st_size),
                    "sha256": _sha256(destination),
                }
            )
        deletion_entries.append(
            {
                "target": str(top.resolve()),
                "eligible_after_first_formal_run": top.name not in protected,
                "protected_until_campaign_complete": top.name in protected,
            }
        )
    payload = {
        "format": "panogsslam_preformal_output_archive_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_root": str(outputs.resolve()),
        "weight_manifest": str(weight_manifest_path.resolve()),
        "inventory": inventory,
        "archived_files": archived_files,
    }
    archive_manifest = archive / "archive_manifest.json"
    archive_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    deletion_manifest = archive / "deletion_manifest.json"
    deletion_manifest.write_text(
        json.dumps(
            {
                "outputs_root": str(outputs.resolve()),
                "archive_manifest": str(archive_manifest.resolve()),
                "entries": deletion_entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return deletion_manifest


def _active_process_paths(outputs: Path) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    current_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == current_pid:
            continue
        checked = [proc / "cwd"]
        fd_dir = proc / "fd"
        if fd_dir.is_dir():
            try:
                checked.extend(fd_dir.iterdir())
            except PermissionError:
                pass
        for link in checked:
            try:
                target = link.resolve(strict=True)
                target.relative_to(outputs)
            except (FileNotFoundError, PermissionError, ValueError, OSError):
                continue
            active.append({"pid": proc.name, "path": str(target)})
            break
    return active


def _verify_weight_copies(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in payload["files"]:
        for key in ("destination", "backup"):
            path = Path(entry[key])
            if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"Preserved weight is missing or truncated: {path}")
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Preserved weight hash mismatch: {path}")


def execute_cleanup(
    deletion_manifest_path: Path,
    *,
    formal_root: Path,
    finalize_protected: bool,
) -> list[str]:
    deletion = json.loads(deletion_manifest_path.read_text(encoding="utf-8"))
    outputs = Path(deletion["outputs_root"]).resolve()
    expected = (PROJECT_ROOT / "outputs").resolve()
    if outputs != expected:
        raise RuntimeError(f"Refusing to clean unexpected outputs root: {outputs}")
    weight_manifest_path = Path(
        json.loads(Path(deletion["archive_manifest"]).read_text(encoding="utf-8"))[
            "weight_manifest"
        ]
    )
    _verify_weight_copies(weight_manifest_path)
    complete_markers = list(formal_root.glob("runs/*/complete.marker"))
    required_completed = 34 if finalize_protected else 1
    if len(complete_markers) < required_completed:
        raise RuntimeError(
            f"Cleanup requires {required_completed} validated formal runs; found {len(complete_markers)}"
        )
    active = _active_process_paths(outputs)
    if active:
        raise RuntimeError("Active processes still reference outputs: " + json.dumps(active))
    removed: list[str] = []
    for entry in deletion["entries"]:
        eligible = bool(entry["eligible_after_first_formal_run"])
        protected = bool(entry["protected_until_campaign_complete"])
        if not eligible and not (protected and finalize_protected):
            continue
        target = Path(entry["target"]).resolve()
        if target.parent != outputs or target == outputs:
            raise RuntimeError(f"Unsafe deletion target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    result = deletion_manifest_path.parent / (
        "final_cleanup_result.json" if finalize_protected else "cleanup_result.json"
    )
    result.write_text(
        json.dumps(
            {"removed": removed, "completed_formal_runs": len(complete_markers)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "archives/preformal_outputs_20260722",
    )
    parser.add_argument(
        "--weight-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/panogsslam_formal_mainline_v1/manifest.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--formal-root", type=Path)
    parser.add_argument("--finalize-protected", action="store_true")
    args = parser.parse_args()
    deletion_manifest = args.archive / "deletion_manifest.json"
    if not args.execute:
        deletion_manifest = create_archive(args.outputs, args.archive, args.weight_manifest)
        print(deletion_manifest)
        return
    if args.formal_root is None:
        raise ValueError("--formal-root is required with --execute")
    removed = execute_cleanup(
        deletion_manifest,
        formal_root=args.formal_root,
        finalize_protected=bool(args.finalize_protected),
    )
    print(json.dumps(removed, indent=2))


if __name__ == "__main__":
    main()
