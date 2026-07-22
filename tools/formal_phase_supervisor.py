"""Run the OB3D-first formal campaign and gate the 360VO-200 phase."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.formal_experiments import (
    _finite_number,
    _git_commit,
    _gpu_processes,
    _read_resource_snapshot,
    _terminate_process_group,
    _write_json,
    validate_run,
)


PYTHON = "/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python"


def phase_status(formal_root: Path, *, expected_run_count: int) -> dict[str, Any]:
    campaign_path = formal_root / "campaign.json"
    if not campaign_path.is_file():
        return {"complete": 0, "failed": [], "pending": expected_run_count}
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    runs = campaign.get("runs", [])
    if len(runs) != expected_run_count or int(
        campaign.get("expected_run_count", -1)
    ) != expected_run_count:
        raise RuntimeError(
            f"Phase count mismatch: expected {expected_run_count}, found {len(runs)}"
        )
    completed = 0
    failed: list[str] = []
    for entry in runs:
        run_root = formal_root / "runs" / entry["run_id"]
        if (run_root / "failed.json").is_file():
            failed.append(str(entry["run_id"]))
        elif (run_root / "complete.marker").is_file():
            completed += 1
    return {
        "complete": completed,
        "failed": failed,
        "pending": expected_run_count - completed - len(failed),
    }


def validate_phase(formal_root: Path, *, expected_run_count: int) -> None:
    campaign = json.loads((formal_root / "campaign.json").read_text(encoding="utf-8"))
    status = phase_status(formal_root, expected_run_count=expected_run_count)
    if status["failed"] or status["complete"] != expected_run_count:
        raise RuntimeError(f"Phase is incomplete: {status}")
    errors: dict[str, list[str]] = {}
    for entry in campaign["runs"]:
        run_root = formal_root / "runs" / entry["run_id"]
        marker = json.loads((run_root / "complete.marker").read_text(encoding="utf-8"))
        attempt = run_root / f"attempt_{int(marker['attempt']):03d}"
        result = validate_run(
            attempt,
            expected_frames=int(entry["frames"]),
            expected_weights_manifest_sha256=campaign["weights_manifest_sha256"],
        )
        if not result["valid"]:
            errors[str(entry["run_id"])] = list(result["errors"])
    if errors:
        raise RuntimeError("Phase artifact validation failed: " + json.dumps(errors))


def _worker_command(
    *, formal_root: Path, repo_root: Path, worker: int, gpu: int, python: str
) -> list[str]:
    return [
        python,
        "tools/formal_experiments.py",
        "worker",
        "--formal-root",
        str(formal_root),
        "--repo-root",
        str(repo_root),
        "--worker",
        str(worker),
        "--gpu",
        str(gpu),
    ]


def run_phase(
    formal_root: Path,
    *,
    repo_root: Path,
    expected_run_count: int,
    gpus: tuple[int, int],
    python: str = PYTHON,
    retry_wait_sec: int = 60,
) -> None:
    while True:
        status = phase_status(formal_root, expected_run_count=expected_run_count)
        if status["failed"]:
            raise RuntimeError(f"Formal phase contains failed runs: {status['failed']}")
        if status["complete"] == expected_run_count:
            validate_phase(formal_root, expected_run_count=expected_run_count)
            return
        logs = formal_root / "supervisor_logs"
        logs.mkdir(parents=True, exist_ok=True)
        processes: list[tuple[subprocess.Popen[Any], Any]] = []
        for worker, gpu in enumerate(gpus):
            stream = (logs / f"worker_{worker}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(
                    formal_root=formal_root,
                    repo_root=repo_root,
                    worker=worker,
                    gpu=gpu,
                    python=python,
                ),
                cwd=repo_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, stream))
        try:
            return_codes = [process.wait() for process, _ in processes]
        except BaseException:
            for process, _ in processes:
                _terminate_process_group(process)
            raise
        finally:
            for _, stream in processes:
                stream.close()
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"Formal workers exited with codes {return_codes}")
        time.sleep(retry_wait_sec)


def _smoke_is_valid(output: Path, *, expected_frames: int) -> bool:
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        metrics = json.loads(
            (output / "final_all_frames" / "metrics.json").read_text(encoding="utf-8")
        )["metrics"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("frames", -1)) == expected_frames
        and int(metrics.get("render_count", -1)) == expected_frames
        and all(_finite_number(metrics.get(key)) for key in ("mean_psnr", "mean_ssim", "mean_lpips"))
        and len(list((output / "final_all_frames" / "render_rgb").glob("frame_*.png")))
        == expected_frames
        and (output / "checkpoints" / "final_gaussian_map.pt").is_file()
    )


def run_metric_smoke(
    formal_root: Path,
    *,
    master_root: Path,
    repo_root: Path,
    gpu: int,
    python: str = PYTHON,
    expected_frames: int = 4,
    name: str,
) -> None:
    output = master_root / "smoke" / name
    marker = output / "complete.marker"
    if marker.is_file() and _smoke_is_valid(output, expected_frames=expected_frames):
        return
    campaign = json.loads((formal_root / "campaign.json").read_text(encoding="utf-8"))
    if _git_commit(repo_root) != campaign["git_commit"]:
        raise RuntimeError("Smoke code commit does not match the prepared 360VO campaign")
    first = campaign["runs"][0]
    config = yaml.safe_load(Path(first["resolved_config"]).read_text(encoding="utf-8"))
    config["Dataset"]["end"] = expected_frames
    config["Results"]["save_dir"] = str(output)
    config["WeightsAndBiases"]["run_name"] = f"formal_{name}"
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    snapshot = _read_resource_snapshot(master_root)
    limits = campaign.get("resource_limits", {})
    if snapshot["available_memory"] < int(limits.get("min_available_memory_gib", 80)) * 1024**3:
        raise RuntimeError(f"Not enough memory for {name}")
    if _gpu_processes(gpu):
        raise RuntimeError(f"GPU {gpu} is not idle for {name}")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TORCH_HOME": str(campaign["torch_home"]),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        python,
        "-m",
        "system.pano_droid_gs_slam",
        "--config",
        str(config_path),
        "--wandb",
        "--wandb-mode",
        "online",
        "--run-name",
        config["WeightsAndBiases"]["run_name"],
    ]
    with (output / "run.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        previous = snapshot
        swap_growth = 0
        while process.poll() is None:
            time.sleep(int(limits.get("monitor_interval_sec", 15)))
            current = _read_resource_snapshot(master_root)
            grew = current["pswpin"] > previous["pswpin"] or current["pswpout"] > previous["pswpout"]
            swap_growth = swap_growth + 1 if grew else 0
            if (
                current["available_memory"]
                < int(limits.get("min_available_memory_gib", 80)) * 1024**3
                or swap_growth >= int(limits.get("consecutive_swap_growth_limit", 2))
            ):
                _terminate_process_group(process)
                raise RuntimeError(f"{name} was stopped by the resource guard")
            previous = current
    if process.wait() != 0 or not _smoke_is_valid(output, expected_frames=expected_frames):
        raise RuntimeError(f"{name} failed; see {output / 'run.log'}")
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ob3d-root", type=Path, required=True)
    parser.add_argument("--360vo-root", dest="vo_root", type=Path, required=True)
    parser.add_argument("--360vo-campaign", dest="vo_campaign", type=Path, required=True)
    parser.add_argument("--dataset-ready-marker", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--python", default=PYTHON)
    args = parser.parse_args()
    master = args.master_root.resolve()
    master.mkdir(parents=True, exist_ok=True)
    gpus = (int(args.gpus[0]), int(args.gpus[1]))
    run_metric_smoke(
        args.ob3d_root.resolve(),
        master_root=master,
        repo_root=args.repo_root.resolve(),
        gpu=gpus[0],
        python=args.python,
        name="ob3d_metric_4frame",
    )
    run_phase(
        args.ob3d_root.resolve(),
        repo_root=args.repo_root.resolve(),
        expected_run_count=24,
        gpus=gpus,
        python=args.python,
    )
    (master / "OB3D_COMPLETE.marker").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    while not args.dataset_ready_marker.is_file():
        time.sleep(60)
    if not (args.vo_root / "campaign.json").is_file():
        subprocess.run(
            [
                args.python,
                "tools/formal_experiments.py",
                "prepare",
                "--campaign",
                str(args.vo_campaign.resolve()),
                "--repo-root",
                str(args.repo_root.resolve()),
                "--formal-root",
                str(args.vo_root.resolve()),
                "--verify-weight-files",
            ],
            cwd=args.repo_root,
            check=True,
        )
    run_metric_smoke(
        args.vo_root.resolve(),
        master_root=master,
        repo_root=args.repo_root.resolve(),
        gpu=gpus[0],
        python=args.python,
        name="360vo200_metric_4frame",
    )
    run_phase(
        args.vo_root.resolve(),
        repo_root=args.repo_root.resolve(),
        expected_run_count=10,
        gpus=gpus,
        python=args.python,
    )
    (master / "FORMAL_COMPLETE.marker").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    _write_json(
        master / "phase_summary.json",
        {
            "ob3d": phase_status(args.ob3d_root, expected_run_count=24),
            "360vo200": phase_status(args.vo_root, expected_run_count=10),
        },
    )


if __name__ == "__main__":
    main()
