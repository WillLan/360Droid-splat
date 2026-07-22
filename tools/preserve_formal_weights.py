"""Create and verify the immutable formal-mainline weight bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path("/mnt/disk1/lanboyang/Project/360Droid-splat")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(project_root: Path) -> list[dict[str, Any]]:
    env_root = Path("/mnt/disk1/lanboyang/miniconda3/envs/pfgs360")
    return [
        {
            "role": "panovggt",
            "source": Path("/mnt/disk1/lanboyang/Project/PanoVGGT/checkpoints/model.pt"),
            "relative": Path("panovggt/model.pt"),
            "config_key": "panovggt.checkpoint",
            "loader": "models.panovggt_feature_wrapper.build_external_panovggt_model",
        },
        {
            "role": "panovggt_config",
            "source": Path("/mnt/disk1/lanboyang/Project/PanoVGGT/training/config/default.yaml"),
            "relative": Path("panovggt/default.yaml"),
            "config_key": "panovggt.config_path",
            "loader": "models.panovggt_feature_wrapper.build_external_panovggt_model",
        },
        {
            "role": "sphereglue",
            "source": Path("/mnt/disk1/lanboyang/External/SphereGlue/model_weights/superpoint/model.safetensors"),
            "relative": Path("sphereglue/model.safetensors"),
            "config_key": "SphericalSelfiRuntime.local_ba.matching.sphereglue_checkpoint",
            "loader": "models.sphereglue_local_ba.SphereGlueLocalBAMatcher",
        },
        {
            "role": "superpoint",
            "source": Path("/mnt/disk1/lanboyang/External/LightGlue/weights/superpoint_v1.pth"),
            "relative": Path("sphereglue/superpoint_v1.pth"),
            "config_key": "SphericalSelfiRuntime.local_ba.matching.superpoint_checkpoint",
            "loader": "models.sphereglue_local_ba.SphereGlueLocalBAMatcher",
        },
        {
            "role": "pager",
            "source": Path("/mnt/disk1/lanboyang/External/PaGeR-checkpoints/unified/model.safetensors"),
            "relative": Path("pager/model.safetensors"),
            "config_key": "SphericalSelfiRuntime.pager_depth.checkpoint",
            "loader": "frontend.spherical_selfi.pager_depth.PaGeRDepthProvider",
        },
        {
            "role": "pager_config",
            "source": Path("/mnt/disk1/lanboyang/External/PaGeR-checkpoints/unified/config.yaml"),
            "relative": Path("pager/config.yaml"),
            "config_key": "SphericalSelfiRuntime.pager_depth.checkpoint",
            "loader": "frontend.spherical_selfi.pager_depth.PaGeRDepthProvider",
        },
        {
            "role": "adapter",
            "source": project_root / "outputs/stage1_selfi_adapter_airsim_dtw_nyc_fullres_spherical_ce_depth20_ddp2/checkpoints/best_val_angular_error.pt",
            "relative": Path("selfi/adapter.pt"),
            "config_key": "adapter_checkpoint.path",
            "loader": "training.train_spherical_selfi_gaussian_head.load_frozen_adapter",
        },
        {
            "role": "gaussian_head",
            "source": project_root / "outputs/stage2_selfi_joint_gpu03_20260712_20k_r1/stage2_selfi_omniscene_joint_fullres_s4_ddp2_effbs4_20k_gpu03_20260712_r1/checkpoints/best_val_psnr.pt",
            "relative": Path("selfi/gaussian_head.pt"),
            "config_key": "stage2_checkpoint.path",
            "loader": "models.per_pixel_gaussian_observation.load_stage2_checkpoint",
        },
        {
            "role": "voxel_anchor_refiner",
            "source": project_root / "outputs/stage3_spherical_voxel_anchor_refiner_omni360_fullres_20k_resume_83cea9d_gpu05_20260715_1510/checkpoints/best_val_psnr.pt",
            "relative": Path("selfi/voxel_anchor_refiner.pt"),
            "config_key": "VoxelAnchorRefiner.checkpoint",
            "loader": "models.spherical_voxel_anchor_refiner.load_voxel_anchor_checkpoint",
        },
        {
            "role": "sky_head",
            "source": project_root / "outputs/panovggt_m3_sphere_omni360_sky_ddp_b4_10k_dtw_nyc_20260604_0845/checkpoints/sky_head.pt",
            "relative": Path("selfi/sky_head.pt"),
            "config_key": "SphericalSelfiRuntime.sky.checkpoint",
            "loader": "models.panovggt_m3_sphere.load_matching_sky_checkpoint",
        },
        {
            "role": "lpips_alexnet_backbone",
            "source": Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
            "relative": Path("evaluation/torch_hub/checkpoints/alexnet-owt-7be5be79.pth"),
            "config_key": "Results.final_image_metrics",
            "loader": "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity",
        },
        {
            "role": "lpips_alexnet_calibration",
            "source": env_root / "lib/python3.11/site-packages/lpips/weights/v0.1/alex.pth",
            "relative": Path("evaluation/lpips/v0.1/alex.pth"),
            "config_key": "Results.final_image_metrics",
            "loader": "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity",
        },
    ]


def _copy_verified(source: Path, destination: Path) -> tuple[int, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or _sha256(destination) != source_hash:
            raise RuntimeError(f"Refusing to overwrite mismatched preserved weight: {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or _sha256(temporary) != source_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Copied weight failed verification: {source}")
        temporary.replace(destination)
    return int(source.stat().st_size), source_hash


def _git_version(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        remote = subprocess.check_output(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"path": str(path), "git_commit": commit, "origin": remote, "status": status}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": str(path), "git_commit": None, "origin": None, "status": None}


def _snapshot_pager_source(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if path.is_file() and path.suffix not in {".pt", ".pth", ".safetensors"}:
                archive.add(path, arcname=Path("PaGeR") / relative, recursive=False)


def create_bundle(project_root: Path, primary: Path, backup: Path) -> None:
    if primary.resolve() == backup.resolve():
        raise ValueError("Primary and backup weight directories must differ")
    primary_manifest = primary / "manifest.json"
    backup_manifest = backup / "manifest.json"
    if primary_manifest.is_file() and backup_manifest.is_file():
        verify_bundle(primary_manifest, verify_backup=True)
        return
    entries: list[dict[str, Any]] = []
    for spec in _sources(project_root):
        source = Path(spec["source"])
        destination = primary / spec["relative"]
        backup_destination = backup / spec["relative"]
        size, digest = _copy_verified(source, destination)
        backup_size, backup_digest = _copy_verified(destination, backup_destination)
        if size != backup_size or digest != backup_digest:
            raise RuntimeError(f"Backup verification failed for {spec['role']}")
        stat = source.stat()
        entries.append(
            {
                "role": spec["role"],
                "source": str(source),
                "destination": str(destination),
                "backup": str(backup_destination),
                "relative": str(spec["relative"]),
                "size_bytes": size,
                "sha256": digest,
                "source_mtime_ns": int(stat.st_mtime_ns),
                "config_key": spec["config_key"],
                "loader": spec["loader"],
            }
        )

    source_versions = [
        _git_version(Path("/mnt/disk1/lanboyang/Project/PanoVGGT")),
        _git_version(Path("/mnt/disk1/lanboyang/External/SphereGlue")),
        _git_version(Path("/mnt/disk1/lanboyang/External/LightGlue")),
        _git_version(Path("/mnt/disk1/lanboyang/External/PaGeR")),
    ]
    primary_source = primary / "source_versions"
    backup_source = backup / "source_versions"
    primary_source.mkdir(parents=True, exist_ok=True)
    backup_source.mkdir(parents=True, exist_ok=True)
    _snapshot_pager_source(
        Path("/mnt/disk1/lanboyang/External/PaGeR"),
        primary_source / "pager_source.tar.gz",
    )
    _copy_verified(
        primary_source / "pager_source.tar.gz",
        backup_source / "pager_source.tar.gz",
    )
    versions_payload = json.dumps(source_versions, indent=2, sort_keys=True)
    (primary_source / "versions.json").write_text(versions_payload, encoding="utf-8")
    shutil.copy2(primary_source / "versions.json", backup_source / "versions.json")
    payload = {
        "format": "panogsslam_formal_weight_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_root": str(primary),
        "backup_root": str(backup),
        "files": entries,
        "source_versions": source_versions,
    }
    for root in (primary, backup):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for root in (primary, backup):
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        root.chmod(0o555)


def verify_bundle(manifest: Path, *, verify_backup: bool) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in payload["files"]:
        paths = [Path(entry["destination"])]
        if verify_backup:
            paths.append(Path(entry["backup"]))
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"Size mismatch: {path}")
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--primary",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "artifacts/checkpoints/panogsslam_formal_mainline_v1",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "archives/formal_weight_backups/panogsslam_formal_mainline_v1",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-backup", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_bundle(args.primary / "manifest.json", verify_backup=args.verify_backup)
    else:
        create_bundle(args.project_root, args.primary, args.backup)
        verify_bundle(args.primary / "manifest.json", verify_backup=True)
    print(args.primary / "manifest.json")


if __name__ == "__main__":
    main()
