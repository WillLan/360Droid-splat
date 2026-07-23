"""Prepare, execute, validate, and aggregate formal PanoGS-SLAM campaigns."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system.pano_droid_gs_slam import load_config  # noqa: E402


REQUIRED_TRAJECTORY_METRICS = (
    "pfgs360_ate",
    "sim3_ate_rmse",
    "se3_ate_rmse",
    "rpe_delta_1_translation_rmse",
    "rpe_delta_1_rotation_mean_deg",
    "rpe_delta_3_translation_rmse",
    "rpe_delta_10_translation_rmse",
    "scale_drift_percent",
    "path_length_scale_ratio",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    dataset_type: str
    root: str
    scene: str
    split: str
    frames: int
    config_overrides: dict[str, Any]
    worker: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _deep_merge_config(
    destination: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively apply explicit campaign overrides to a resolved config."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_merge_config(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _campaign_root(campaign: dict[str, Any], repo_root: Path) -> Path:
    explicit = campaign.get("formal_root")
    if explicit:
        return Path(str(explicit)).resolve()
    commit = _git_commit(repo_root)[:9]
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    parent = Path(str(campaign["formal_root_parent"]))
    return parent / f"{campaign['name']}_{commit}_{date}"


def _expand_runs(campaign: dict[str, Any]) -> list[RunSpec]:
    raw_runs: list[dict[str, Any]] = []
    for dataset in campaign.get("datasets", []):
        for scene in dataset.get("scenes", []):
            for split in dataset.get("splits", []):
                dataset_name = str(dataset["name"])
                scene_name = str(scene)
                split_name = str(split)
                run_id = (
                    f"{dataset_name}__{scene_name}__{split_name}"
                    .lower()
                    .replace("-", "_")
                )
                raw_runs.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "dataset_type": str(dataset.get("type", "ob3d")),
                        "root": str(dataset["root"]),
                        "scene": scene_name,
                        "split": split_name,
                        "frames": int(dataset["frames"]),
                        "config_overrides": copy.deepcopy(
                            dict(dataset.get("config_overrides", {}) or {})
                        ),
                    }
                )
    workers = int(campaign.get("max_concurrent_workers", 2))
    if workers != 2:
        raise ValueError("The formal campaign is locked to exactly two workers")
    loads = [0 for _ in range(workers)]
    assigned: dict[str, int] = {}
    for run in sorted(raw_runs, key=lambda item: (-item["frames"], item["run_id"])):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        assigned[run["run_id"]] = worker
        loads[worker] += int(run["frames"])
    return [
        RunSpec(**run, worker=assigned[run["run_id"]])
        for run in sorted(
            raw_runs,
            key=lambda item: (
                item["dataset"] != "ob3d",
                item["run_id"],
            ),
        )
    ]


def _assert_formal_mainline(config: dict[str, Any], *, seed: int) -> None:
    runtime = config["SphericalSelfiRuntime"]
    backend = config["SphericalSelfiGlobalBackend"]
    optimization = backend["map_optimization"]
    pfgs = optimization["pfgs360"]
    expected = {
        "PaGeR enabled": runtime["pager_depth"]["enabled"] is True,
        "SphereGlue matcher": runtime["local_ba"]["matching"]["type"]
        == "superpoint_sphereglue",
        "Global-Map-Sim3": backend["rendered_overlap_alignment"]["mode"]
        == "two_frame_global_map_full_sim3",
        "diagnostics-only alignment": backend["rendered_overlap_alignment"][
            "acceptance_policy"
        ]
        == "diagnostics_only",
        "chunk-first graph": backend["global_graph"]["node_mode"]
        == "chunk_first_stride",
        "refined-anchor growth": pfgs["growth_source"] == "refined_anchor",
        "refined-anchor bootstrap": pfgs["bootstrap_source"]
        == "refined_anchor_all_views",
        "CAMERA 50": int(optimization["camera_steps"]) == 50,
        "JOINT 200": int(optimization["joint_steps"]) == 200,
        "recent three chunks": int(optimization["recent_window_count"]) == 3,
        "one sampled frame": int(optimization["sample_observations_per_step"]) == 1,
        "global Gaussian candidate set": optimization["optimize_all_gaussians"]
        is True,
        "all render-contributor Gaussian update": pfgs["gaussian_update_scope"]
        == "all_render_contributors",
        "fixed seed": int(optimization["seed"]) == int(seed),
        "core W&B preset": config["WeightsAndBiases"]["runtime_log_preset"]
        == "slam_core_visuals",
        "official image metrics": config["Results"]["final_image_metrics"]
        == "pfgs360_official",
    }
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise ValueError("Formal mainline invariant failure: " + ", ".join(failed))


def _assert_dataset_policy(config: dict[str, Any], run: RunSpec) -> None:
    runtime = config["SphericalSelfiRuntime"]
    backend = config["SphericalSelfiGlobalBackend"]
    pfgs = backend["map_optimization"]["pfgs360"]
    refiner_voxels = [float(value) for value in config["VoxelAnchorRefiner"]["voxel_sizes"]]
    fusion_voxels = [float(value) for value in backend["voxel_fusion"]["voxel_sizes"]]
    sky = runtime["sky"]
    mapping = config["Mapping"]
    skybox = config["SkyBox"]
    sky_sphere = dict(config.get("SkySphere", {}) or {})
    sky_sphere_enabled = bool(sky_sphere.get("enabled", False))
    if run.dataset == "ob3d":
        expected = {
            "OB3D Sky Head disabled": sky["enabled"] is False,
            "OB3D sky not required": sky["required"] is False,
            "OB3D mapping sky mask disabled": mapping["sky_mask_enable"] is False,
            "OB3D mapping sky source disabled": str(mapping["sky_mask_source"]).lower()
            == "none",
            "OB3D skybox disabled": skybox["enabled"] is False,
            "OB3D SkySphere disabled": sky_sphere_enabled is False,
            "OB3D DIA has no semantic gate": pfgs["validity_gate"]
            == "pfgs360_official_no_semantic_gate",
            "OB3D PFGS sky filtering disabled": pfgs["filter_sky"] is False,
            "OB3D Refiner voxel sizes": refiner_voxels
            == [0.02, 0.04, 0.08, 0.16],
            "OB3D Refiner voxel override explicit": config[
                "VoxelAnchorRefiner"
            ].get("allow_voxel_size_override")
            is True,
            "OB3D fusion voxel sizes": fusion_voxels
            == [0.02, 0.04, 0.08, 0.16],
        }
    elif run.dataset == "360vo":
        expected = {
            "360VO Sky Head enabled": sky["enabled"] is True,
            "360VO sky required": sky["required"] is True,
            "360VO sky threshold": abs(float(sky["threshold"]) - 0.6) < 1.0e-12,
            "360VO mapping sky mask enabled": mapping["sky_mask_enable"] is True,
            "360VO mapping sky source": str(mapping["sky_mask_source"]).lower()
            == "panovggt_head",
            "360VO one sky renderer enabled": bool(skybox["enabled"])
            != sky_sphere_enabled,
            "360VO SkySphere threshold": (
                not sky_sphere_enabled
                or abs(float(sky_sphere["sky_threshold"]) - 0.6) < 1.0e-12
            ),
            "360VO backend sky threshold": abs(
                float(backend["global_graph"]["sky_threshold"]) - 0.6
            )
            < 1.0e-12,
            "360VO DIA sky-only gate": pfgs["validity_gate"]
            == "pfgs360_official_sky_only",
            "360VO PFGS sky filtering": pfgs["filter_sky"] is True,
            "360VO Refiner voxel sizes": refiner_voxels
            == [0.04, 0.08, 0.16, 0.32],
            "360VO Refiner voxel override disabled": config[
                "VoxelAnchorRefiner"
            ].get("allow_voxel_size_override", False)
            is False,
            "360VO fusion voxel sizes": fusion_voxels
            == [0.04, 0.08, 0.16, 0.32],
        }
    else:
        raise ValueError(f"Unknown formal dataset policy: {run.dataset}")
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise ValueError(
            f"Formal dataset policy failure for {run.run_id}: " + ", ".join(failed)
        )


def _create_browse_links(root: Path, runs: list[RunSpec]) -> None:
    """Expose the requested paper-facing directory layout without moving runs."""

    for run in runs:
        run_root = root / "runs" / run.run_id
        if run.dataset == "ob3d":
            link = root / "ob3d" / run.scene / run.split
        else:
            link = root / "360vo" / run.scene
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            continue
        target = Path(os.path.relpath(run_root, start=link.parent))
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            link.mkdir(parents=True, exist_ok=True)
            (link / "RUN_POINTER.txt").write_text(str(run_root), encoding="utf-8")


def _verify_dataset_run(run: RunSpec) -> None:
    sequence = Path(run.root) / run.scene / run.split
    image_dir = sequence / "images"
    camera_dir = sequence / "cameras"
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ) if image_dir.is_dir() else []
    cameras = sorted(camera_dir.glob("*_cam.json")) if camera_dir.is_dir() else []
    if len(images) < run.frames or len(cameras) < run.frames:
        raise ValueError(
            f"Dataset {run.run_id} has {len(images)} images and {len(cameras)} "
            f"cameras, but {run.frames} are required"
        )


def _verify_weight_manifest(path: Path, *, verify_files: bool) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Weight manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("files", [])
    required_roles = {
        "panovggt",
        "panovggt_config",
        "sphereglue",
        "superpoint",
        "pager",
        "pager_config",
        "adapter",
        "gaussian_head",
        "voxel_anchor_refiner",
        "sky_head",
    }
    actual_roles = {str(entry.get("role")) for entry in entries}
    if not required_roles.issubset(actual_roles):
        raise ValueError(
            "Weight manifest is incomplete: "
            + ", ".join(sorted(required_roles - actual_roles))
        )
    if verify_files:
        for entry in entries:
            file_path = Path(str(entry["destination"]))
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            if int(file_path.stat().st_size) != int(entry["size_bytes"]):
                raise ValueError(f"Weight size mismatch: {file_path}")
            if _sha256(file_path) != str(entry["sha256"]):
                raise ValueError(f"Weight SHA-256 mismatch: {file_path}")
    return _sha256(path)


def _prepare_torch_home(
    campaign: dict[str, Any],
    *,
    formal_root: Path,
    weight_manifest: Path,
) -> Path | None:
    """Create a writable Torch cache without mutating the immutable weight pack."""

    configured = campaign.get("torch_home")
    if configured in (None, ""):
        return None
    torch_home = Path(str(configured))
    if not torch_home.is_absolute():
        torch_home = formal_root / torch_home
    torch_home = torch_home.resolve()
    checkpoints = torch_home / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if not os.access(checkpoints, os.W_OK):
        raise PermissionError(f"Torch checkpoint cache is not writable: {checkpoints}")

    manifest = json.loads(weight_manifest.read_text(encoding="utf-8"))
    candidates = [
        entry
        for entry in manifest.get("files", [])
        if str(entry.get("role")) == "lpips_alexnet_backbone"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Weight manifest must contain exactly one lpips_alexnet_backbone"
        )
    entry = candidates[0]
    source = Path(str(entry["destination"]))
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_size = int(entry["size_bytes"])
    expected_sha256 = str(entry["sha256"])
    if int(source.stat().st_size) != expected_size or _sha256(source) != expected_sha256:
        raise ValueError(f"Immutable LPIPS backbone failed verification: {source}")
    destination = checkpoints / source.name
    if not destination.is_file() or (
        int(destination.stat().st_size) != expected_size
        or _sha256(destination) != expected_sha256
    ):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        shutil.copy2(source, temporary)
        if int(temporary.stat().st_size) != expected_size or _sha256(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Copied LPIPS backbone failed verification: {temporary}")
        temporary.replace(destination)
    _write_json(
        formal_root / "runtime_cache" / "manifest.json",
        {
            "torch_home": str(torch_home),
            "source": str(source),
            "destination": str(destination),
            "size_bytes": expected_size,
            "sha256": expected_sha256,
        },
    )
    return torch_home


def prepare_campaign(
    campaign_path: Path,
    *,
    repo_root: Path,
    formal_root: Path | None,
    verify_weight_files: bool,
) -> Path:
    campaign = _load_yaml(campaign_path)
    root = formal_root or _campaign_root(campaign, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    weight_manifest = Path(str(campaign["weights_manifest"]))
    weight_manifest_sha256 = _verify_weight_manifest(
        weight_manifest,
        verify_files=verify_weight_files,
    )
    torch_home = _prepare_torch_home(
        campaign,
        formal_root=root,
        weight_manifest=weight_manifest,
    )
    base_path = Path(str(campaign["base_config"]))
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    base = load_config(base_path)
    seed = int(campaign.get("seed", 123))
    _assert_formal_mainline(base, seed=seed)
    commit = _git_commit(repo_root)
    runs = _expand_runs(campaign)
    expected_run_count = int(campaign.get("expected_run_count", 34))
    if expected_run_count <= 0:
        raise ValueError("expected_run_count must be positive")
    if len(runs) != expected_run_count:
        raise ValueError(
            f"Expected {expected_run_count} formal runs, got {len(runs)}"
        )
    for run in runs:
        _verify_dataset_run(run)

    expanded: list[dict[str, Any]] = []
    for run in runs:
        config = copy.deepcopy(base)
        _deep_merge_config(config, run.config_overrides)
        config["Dataset"] = {
            **dict(config.get("Dataset", {}) or {}),
            "synthetic": False,
            "type": run.dataset_type,
            "dataset_path": run.root,
            "scene": run.scene,
            "split": run.split,
            "begin": 0,
            "end": int(run.frames),
            "frame_stride": 1,
            "erp_resize_height": 504,
            "erp_resize_width": 1008,
        }
        _assert_formal_mainline(config, seed=seed)
        _assert_dataset_policy(config, run)
        run_name = f"formal_{run.run_id}_seed{seed}"
        config["WeightsAndBiases"] = {
            **dict(config.get("WeightsAndBiases", {}) or {}),
            "run_name": run_name,
            "tags": [
                str(campaign["name"]),
                "pager",
                "sphereglue",
                "global-map-sim3",
                "refined-anchor",
                "camera50-joint200",
                *(
                    ["sky-sphere"]
                    if bool(
                        dict(config.get("SkySphere", {}) or {}).get(
                            "enabled", False
                        )
                    )
                    else ["skybox"]
                ),
                run.dataset,
                run.scene,
                run.split.lower(),
            ],
        }
        attempt_dir = root / "runs" / run.run_id / "attempt_001"
        config["Results"] = {
            **dict(config.get("Results", {}) or {}),
            "save_dir": str(attempt_dir),
        }
        resolved_path = root / "resolved_configs" / f"{run.run_id}.yaml"
        _write_yaml(resolved_path, config)
        expanded.append(
            {
                **run.__dict__,
                "run_name": run_name,
                "resolved_config": str(resolved_path),
            }
        )

    copied_manifest = root / "weights_manifest.json"
    shutil.copy2(weight_manifest, copied_manifest)
    campaign_payload = {
        "name": campaign["name"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "repo_root": str(repo_root),
        "formal_root": str(root),
        "base_config": str(base_path),
        "seed": seed,
        "expected_run_count": expected_run_count,
        "weights_manifest": str(weight_manifest),
        "weights_manifest_sha256": weight_manifest_sha256,
        "torch_home": str(torch_home) if torch_home is not None else None,
        "resource_limits": campaign.get("resource_limits", {}),
        "max_concurrent_workers": 2,
        "runs": expanded,
    }
    _write_json(root / "campaign.json", campaign_payload)
    _write_yaml(root / "campaign.yaml", campaign_payload)
    _write_yaml(root / "manifest.yaml", campaign_payload)
    _create_browse_links(root, runs)
    return root


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_run(
    attempt_dir: Path,
    *,
    expected_frames: int,
    expected_weights_manifest_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    summary_path = attempt_dir / "summary.json"
    runtime_path = attempt_dir / "runtime.json"
    metrics_path = attempt_dir / "final_all_frames" / "metrics.json"
    provenance_path = attempt_dir / "run_provenance.json"
    required_files = (summary_path, runtime_path, metrics_path, provenance_path)
    for path in required_files:
        if not path.is_file():
            errors.append(f"missing {path.relative_to(attempt_dir)}")
    if errors:
        return {"valid": False, "errors": errors}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    metrics = metrics_payload.get("metrics", {})
    if int(summary.get("frames", -1)) != expected_frames:
        errors.append(f"summary frame count is {summary.get('frames')}, expected {expected_frames}")
    if int(metrics.get("render_count", -1)) != expected_frames:
        errors.append(f"render count is {metrics.get('render_count')}, expected {expected_frames}")
    if int(metrics.get("ate_count", -1)) != expected_frames:
        errors.append(f"trajectory count is {metrics.get('ate_count')}, expected {expected_frames}")
    for key in ("mean_psnr", "mean_ssim", "mean_lpips"):
        if not _finite_number(metrics.get(key)):
            errors.append(f"non-finite image metric: {key}")
    for key in REQUIRED_TRAJECTORY_METRICS:
        if not _finite_number(metrics.get(key)):
            errors.append(f"non-finite trajectory metric: {key}")
    for key in ("total_wall_sec", "seconds_per_frame", "fps"):
        if not _finite_number(runtime.get(key)):
            errors.append(f"non-finite runtime metric: {key}")

    render_dir = attempt_dir / "final_all_frames" / "render_rgb"
    if len(list(render_dir.glob("frame_*.png"))) != expected_frames:
        errors.append("plain render image count mismatch")
    trajectory_dir = attempt_dir / "final_all_frames" / "trajectory"
    for filename in (
        "predicted_c2w.json",
        "gt_c2w.json",
        "sim3_aligned_predicted_c2w.json",
        "trajectory_sim3.png",
        "metrics.json",
    ):
        if not (trajectory_dir / filename).is_file():
            errors.append(f"missing trajectory/{filename}")
    for filename in ("predicted_c2w.json", "gt_c2w.json", "sim3_aligned_predicted_c2w.json"):
        path = trajectory_dir / filename
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if len(payload.get("poses", [])) != expected_frames:
                errors.append(f"trajectory/{filename} pose count mismatch")
    checkpoint = attempt_dir / "checkpoints" / "final_gaussian_map.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        errors.append("missing or empty final Gaussian checkpoint")
    if provenance.get("weights_manifest_sha256") != expected_weights_manifest_sha256:
        errors.append("run provenance has the wrong weight manifest SHA-256")

    result = {
        "valid": not errors,
        "errors": errors,
        "expected_frames": expected_frames,
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
    }
    _write_json(attempt_dir / "validation.json", result)
    return result


def _read_resource_snapshot(path: Path) -> dict[str, int]:
    memory: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            name, value = line.split(":", 1)
            memory[name] = int(value.strip().split()[0]) * 1024
    vmstat: dict[str, int] = {}
    with Path("/proc/vmstat").open("r", encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split()
            if key in {"pswpin", "pswpout"}:
                vmstat[key] = int(value)
    disk = shutil.disk_usage(path)
    return {
        "available_memory": memory.get("MemAvailable", 0),
        "pswpin": vmstat.get("pswpin", 0),
        "pswpout": vmstat.get("pswpout", 0),
        "free_disk": int(disk.free),
        "cpu_count": int(os.cpu_count() or 1),
        "load_1m_milli": int(os.getloadavg()[0] * 1000.0),
    }


def _gpu_processes(gpu: int) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def _transient_failure(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-100_000:].lower()
    patterns = (
        "cuda out of memory",
        "cuda error",
        "cublas_status",
        "connection reset",
        "network is unreachable",
        "temporary failure",
    )
    return any(pattern in tail for pattern in patterns)


def _acquire_worker_lock(formal_root: Path, worker: int, max_workers: int) -> Path:
    lock_dir = formal_root / "worker_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for lock in lock_dir.glob("worker_*.lock"):
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink(missing_ok=True)
    active = list(lock_dir.glob("worker_*.lock"))
    if len(active) >= max_workers:
        raise RuntimeError("Formal campaign already has the maximum two active workers")
    lock = lock_dir / f"worker_{worker}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    return lock


def run_worker(formal_root: Path, *, repo_root: Path, worker: int, gpu: int) -> None:
    campaign = json.loads((formal_root / "campaign.json").read_text(encoding="utf-8"))
    if _git_commit(repo_root) != campaign["git_commit"]:
        raise RuntimeError("Worker code commit does not match the prepared campaign")
    max_workers = int(campaign.get("max_concurrent_workers", 2))
    if worker not in range(max_workers):
        raise ValueError(f"worker must be in [0, {max_workers - 1}]")
    lock = _acquire_worker_lock(formal_root, worker, max_workers)
    try:
        limits = campaign.get("resource_limits", {})
        min_memory = int(limits.get("min_available_memory_gib", 80)) * 1024**3
        min_disk = int(limits.get("min_free_disk_gib", 100)) * 1024**3
        max_cpu_fraction = float(limits.get("max_prelaunch_cpu_load_fraction", 0.8))
        interval = int(limits.get("monitor_interval_sec", 15))
        swap_limit = int(limits.get("consecutive_swap_growth_limit", 2))
        python = "/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python"
        runs = [entry for entry in campaign["runs"] if int(entry["worker"]) == worker]
        for entry in runs:
            run_root = formal_root / "runs" / entry["run_id"]
            complete = run_root / "complete.marker"
            if complete.is_file():
                continue
            base_config = _load_yaml(Path(entry["resolved_config"]))
            succeeded = False
            for attempt_number in (1, 2):
                attempt_dir = run_root / f"attempt_{attempt_number:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                snapshot = _read_resource_snapshot(formal_root)
                load_fraction = (
                    snapshot["load_1m_milli"]
                    / 1000.0
                    / max(1, snapshot["cpu_count"])
                )
                gpu_processes = _gpu_processes(gpu)
                if (
                    snapshot["available_memory"] < min_memory
                    or snapshot["free_disk"] < min_disk
                    or load_fraction > max_cpu_fraction
                    or gpu_processes
                ):
                    _write_json(
                        run_root / "paused_resource_guard.json",
                        {
                            "snapshot": snapshot,
                            "load_fraction": load_fraction,
                            "gpu_processes": gpu_processes,
                            "time": time.time(),
                        },
                    )
                    return
                config = copy.deepcopy(base_config)
                config["Results"]["save_dir"] = str(attempt_dir)
                config_path = attempt_dir / "resolved_config.yaml"
                _write_yaml(config_path, config)
                provenance = {
                    "run_id": entry["run_id"],
                    "dataset": entry["dataset"],
                    "scene": entry["scene"],
                    "split": entry["split"],
                    "expected_frames": int(entry["frames"]),
                    "worker": worker,
                    "physical_gpu": gpu,
                    "git_commit": campaign["git_commit"],
                    "weights_manifest_sha256": campaign["weights_manifest_sha256"],
                    "resolved_config": str(config_path),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_json(attempt_dir / "run_provenance.json", provenance)
                log_path = attempt_dir / "run.log"
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "8",
                        "MKL_NUM_THREADS": "8",
                        "OPENBLAS_NUM_THREADS": "8",
                        "NUMEXPR_NUM_THREADS": "8",
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                if campaign.get("torch_home"):
                    env["TORCH_HOME"] = str(campaign["torch_home"])
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
                    str(entry["run_name"]),
                ]
                previous = snapshot
                consecutive_swap_growth = 0
                resource_abort = False
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=repo_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while process.poll() is None:
                        time.sleep(interval)
                        current = _read_resource_snapshot(formal_root)
                        swap_grew = (
                            current["pswpin"] > previous["pswpin"]
                            or current["pswpout"] > previous["pswpout"]
                        )
                        consecutive_swap_growth = (
                            consecutive_swap_growth + 1 if swap_grew else 0
                        )
                        if (
                            current["available_memory"] < min_memory
                            or current["free_disk"] < min_disk
                            or consecutive_swap_growth >= swap_limit
                        ):
                            resource_abort = True
                            _write_json(
                                attempt_dir / "resource_abort.json",
                                {
                                    "snapshot": current,
                                    "previous": previous,
                                    "consecutive_swap_growth": consecutive_swap_growth,
                                },
                            )
                            _terminate_process_group(process)
                            break
                        previous = current
                return_code = int(process.wait())
                if return_code == 0 and not resource_abort:
                    validation = validate_run(
                        attempt_dir,
                        expected_frames=int(entry["frames"]),
                        expected_weights_manifest_sha256=campaign[
                            "weights_manifest_sha256"
                        ],
                    )
                    if validation["valid"]:
                        (run_root / "failed.json").unlink(missing_ok=True)
                        complete.write_text(
                            json.dumps(
                                {
                                    "attempt": attempt_number,
                                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        succeeded = True
                        aggregate_campaign(formal_root)
                        break
                if attempt_number == 1 and not resource_abort and _transient_failure(log_path):
                    continue
                break
            if not succeeded:
                _write_json(
                    run_root / "failed.json",
                    {"run_id": entry["run_id"], "time": time.time()},
                )
                aggregate_campaign(formal_root)
    finally:
        lock.unlink(missing_ok=True)


def aggregate_campaign(formal_root: Path) -> None:
    campaign = json.loads((formal_root / "campaign.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in campaign["runs"]:
        run_root = formal_root / "runs" / entry["run_id"]
        marker = run_root / "complete.marker"
        if not marker.is_file():
            status = "failed" if (run_root / "failed.json").is_file() else "pending"
            failures.append({"run_id": entry["run_id"], "status": status})
            continue
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        attempt_dir = run_root / f"attempt_{int(marker_payload['attempt']):03d}"
        summary = json.loads((attempt_dir / "summary.json").read_text(encoding="utf-8"))
        metrics_payload = json.loads(
            (attempt_dir / "final_all_frames" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        runtime = json.loads((attempt_dir / "runtime.json").read_text(encoding="utf-8"))
        metrics = metrics_payload["metrics"]
        rows.append(
            {
                "run_id": entry["run_id"],
                "dataset": entry["dataset"],
                "scene": entry["scene"],
                "split": entry["split"],
                "frames": summary["frames"],
                "anchors": summary["anchors"],
                "psnr": metrics["mean_psnr"],
                "ssim": metrics["mean_ssim"],
                "lpips": metrics["mean_lpips"],
                "pfgs360_ate": metrics["pfgs360_ate"],
                "sim3_ate_rmse": metrics["sim3_ate_rmse"],
                "se3_ate_rmse": metrics["se3_ate_rmse"],
                "scale_drift_percent": metrics["scale_drift_percent"],
                "path_length_scale_ratio": metrics["path_length_scale_ratio"],
                "runtime_sec": runtime["total_wall_sec"],
                "seconds_per_frame": runtime["seconds_per_frame"],
                "peak_gpu_memory_bytes": runtime.get("peak_gpu_memory_bytes"),
                "wandb_url": summary.get("wandb_run_url"),
                "attempt_dir": str(attempt_dir),
            }
        )
    aggregate = formal_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    _write_json(aggregate / "metrics.json", rows)
    _write_json(aggregate / "status.json", {"completed": len(rows), "incomplete": failures})
    fieldnames = list(rows[0]) if rows else [
        "run_id", "dataset", "scene", "split", "frames", "status"
    ]
    with (aggregate / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument("--formal-root", type=Path)
    prepare.add_argument("--verify-weight-files", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--formal-root", type=Path, required=True)
    worker.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    worker.add_argument("--worker", type=int, required=True)
    worker.add_argument("--gpu", type=int, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--attempt-dir", type=Path, required=True)
    validate.add_argument("--expected-frames", type=int, required=True)
    validate.add_argument("--weights-manifest-sha256", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        root = prepare_campaign(
            args.campaign.resolve(),
            repo_root=args.repo_root.resolve(),
            formal_root=args.formal_root.resolve() if args.formal_root else None,
            verify_weight_files=bool(args.verify_weight_files),
        )
        print(root)
    elif args.command == "worker":
        run_worker(
            args.formal_root.resolve(),
            repo_root=args.repo_root.resolve(),
            worker=int(args.worker),
            gpu=int(args.gpu),
        )
    elif args.command == "validate":
        result = validate_run(
            args.attempt_dir.resolve(),
            expected_frames=int(args.expected_frames),
            expected_weights_manifest_sha256=str(args.weights_manifest_sha256),
        )
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    else:
        aggregate_campaign(args.formal_root.resolve())


if __name__ == "__main__":
    main()
