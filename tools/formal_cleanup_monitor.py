"""Bind the guarded historical-output cleanup to the v3 two-phase campaign."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.archive_preformal_outputs import execute_cleanup


def completed_runs(root: Path) -> int:
    markers = list(root.glob("runs/*/complete.marker"))
    markers.extend(root.glob("*/runs/*/complete.marker"))
    return len(markers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--ob3d-root", type=Path, required=True)
    parser.add_argument("--deletion-manifest", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=60)
    args = parser.parse_args()
    archive = args.deletion_manifest.resolve().parent
    first_result = archive / "cleanup_result.json"
    final_result = archive / "final_cleanup_result.json"
    while completed_runs(args.ob3d_root) < 1:
        time.sleep(args.poll_sec)
    if not first_result.is_file():
        execute_cleanup(
            args.deletion_manifest.resolve(),
            formal_root=args.master_root.resolve(),
            finalize_protected=False,
            required_completed=1,
        )
    while completed_runs(args.master_root) < 34 or not (
        args.master_root / "FORMAL_COMPLETE.marker"
    ).is_file():
        time.sleep(args.poll_sec)
    if not final_result.is_file():
        execute_cleanup(
            args.deletion_manifest.resolve(),
            formal_root=args.master_root.resolve(),
            finalize_protected=True,
            required_completed=34,
        )


if __name__ == "__main__":
    main()
