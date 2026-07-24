"""Build an atomic, fully copied 360VO subset in the OB3D/PFGS360 layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GT_HEADER = "#frame name x y z qx qy qz qw"
IMAGE_PATTERN = re.compile(r"^Frame_(\d+)_FinalColor\.png$")
A_WORLD = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
B_CAMERA = np.diag([1.0, -1.0, -1.0])
H_WORLD = np.diag([1.0, -1.0, 1.0])
M_UE_FROM_OPENGL = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_list_indices(total_count: int, target_count: int) -> list[int]:
    if target_count < 2:
        raise ValueError("target_count must be at least two")
    if total_count < target_count:
        raise ValueError(
            f"Only {total_count} valid images are available for {target_count} samples"
        )
    indices = [
        int(math.floor(index * (total_count - 1) / (target_count - 1) + 0.5))
        for index in range(target_count)
    ]
    if (
        len(indices) != target_count
        or len(set(indices)) != target_count
        or indices[0] != 0
        or indices[-1] != total_count - 1
        or indices != sorted(indices)
    ):
        raise RuntimeError("Uniform sampling failed its endpoint/uniqueness contract")
    return indices


def strided_prefix_list_indices(
    total_count: int,
    *,
    source_prefix_count: int,
    source_stride: int,
    target_count: int | None = None,
) -> list[int]:
    """Select ``0, stride, ...`` from a fixed prefix of the ordered source."""

    if source_prefix_count <= 0:
        raise ValueError("source_prefix_count must be positive")
    if source_prefix_count > total_count:
        raise ValueError(
            f"Only {total_count} valid images are available for a "
            f"{source_prefix_count}-frame prefix"
        )
    if source_stride <= 0:
        raise ValueError("source_stride must be positive")
    indices = list(range(0, source_prefix_count, source_stride))
    if target_count is not None and len(indices) != target_count:
        raise ValueError(
            f"Prefix/stride selects {len(indices)} frames, not target_count="
            f"{target_count}"
        )
    return indices


def quaternion_xyzw_to_rotation(values: tuple[float, float, float, float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def source_to_target_pose(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return H_WORLD @ rotation @ M_UE_FROM_OPENGL, H_WORLD @ translation


def target_pose_to_json_extrinsics(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_cv = A_WORLD.T @ rotation @ B_CAMERA.T
    t_cv = A_WORLD.T @ translation
    r_w2c = r_cv.T
    return r_w2c, -r_w2c @ t_cv, t_cv


def verify_round_trip(
    rotation: np.ndarray,
    translation: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> float:
    r_cv = r_w2c.T
    t_cv = -r_cv @ t_w2c
    recovered_rotation = A_WORLD @ r_cv @ B_CAMERA
    recovered_translation = A_WORLD @ t_cv
    error = max(
        float(np.max(np.abs(recovered_rotation - rotation))),
        float(np.max(np.abs(recovered_translation - translation))),
    )
    if error > 1.0e-8:
        raise RuntimeError(f"Pose round-trip error is {error}")
    return error


def read_images(sequence_dir: Path) -> list[tuple[int, Path]]:
    records: list[tuple[int, Path]] = []
    for path in (sequence_dir / "images").iterdir():
        match = IMAGE_PATTERN.match(path.name)
        if path.is_file() and match is not None:
            records.append((int(match.group(1)), path))
    records.sort(key=lambda item: item[0])
    numbers = [item[0] for item in records]
    if not records or len(numbers) != len(set(numbers)):
        raise ValueError(f"Missing or duplicate source images in {sequence_dir}")
    return records


def read_gt(sequence_dir: Path) -> dict[str, dict[str, Any]]:
    lines = (sequence_dir / "gt.txt").read_text(
        encoding="utf-8-sig", errors="strict"
    ).splitlines()
    if not lines or lines[0].strip() != GT_HEADER:
        raise ValueError(f"Unexpected GT header in {sequence_dir / 'gt.txt'}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"GT line {line_number} does not have nine columns")
        frame = int(fields[0])
        name = fields[1]
        pose_values = tuple(float(value) for value in fields[2:])
        quaternion = pose_values[3:]
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 5.0e-3:
            raise ValueError(f"GT quaternion is not normalized on line {line_number}")
        if name in records:
            raise ValueError(f"Duplicate GT image name: {name}")
        records[name] = {
            "source_frame": frame,
            "source_gt_line": line_number,
            "translation": np.asarray(pose_values[:3], dtype=np.float64),
            "rotation": quaternion_xyzw_to_rotation(quaternion),
            "pose_values": pose_values,
            "raw_line": line.strip(),
        }
    return records


def write_camera_json(
    path: Path,
    *,
    width: int,
    height: int,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> None:
    payload = [
        {
            "width": width,
            "height": height,
            "intrinsics": {"focal": float(height), "cx": width / 2.0, "cy": height / 2.0},
            "extrinsics": {"rotation": r_w2c.tolist(), "translation": t_w2c.tolist()},
        }
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_sparse_ply(path: Path, points: np.ndarray) -> None:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]
    for index, point in enumerate(points):
        ratio = index / max(1, len(points) - 1)
        lines.append(
            f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
            f"{int(round(255 * ratio))} 180 {int(round(255 * (1 - ratio)))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_indices(path: Path, indices: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in indices), encoding="utf-8")


def validate_sequence(
    output_data: Path,
    *,
    target_count: int,
    expected_first_source: int,
    expected_last_source: int,
) -> dict[str, Any]:
    images = sorted((output_data / "images").glob("*_rgb.png"))
    cameras = sorted((output_data / "cameras").glob("*_cam.json"))
    mapping = list(csv.DictReader(
        (output_data / "frame_mapping.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    if len(images) != target_count or len(cameras) != target_count or len(mapping) != target_count:
        raise RuntimeError("Converted image/camera/mapping count mismatch")
    if images[0].name != "00000_rgb.png" or images[-1].name != f"{target_count - 1:05d}_rgb.png":
        raise RuntimeError("Converted image numbering is not contiguous")
    if int(mapping[0]["source_frame"]) != expected_first_source:
        raise RuntimeError("First converted frame does not match the source first frame")
    if int(mapping[-1]["source_frame"]) != expected_last_source:
        raise RuntimeError("Last converted frame does not match the source last frame")
    poses = np.load(output_data / "groundtruth_camera_to_world.npz")["camera_to_worlds"]
    if poses.shape != (target_count, 4, 4) or not np.isfinite(poses).all():
        raise RuntimeError("Converted c2w archive is invalid")
    return {"images": len(images), "cameras": len(cameras), "poses": int(poses.shape[0])}


def convert_sequence(
    source_root: Path,
    output_root: Path,
    sequence_name: str,
    *,
    target_count: int = 200,
    source_prefix_count: int | None = None,
    source_stride: int = 1,
    expected_size: tuple[int, int] = (1920, 960),
    expected_mode: str = "RGBA",
    check_only: bool = False,
) -> dict[str, Any]:
    source_sequence = source_root / sequence_name
    output_data = output_root / sequence_name / "Egocentric"
    temporary = output_root / f".{sequence_name}-Egocentric.tmp"
    if output_data.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite existing conversion: {output_data}")
    images = read_images(source_sequence)
    gt = read_gt(source_sequence)
    image_names = {path.name for _, path in images}
    if image_names != set(gt):
        raise RuntimeError("Source images and GT do not form a one-to-one mapping")
    if source_prefix_count is None:
        selected = uniform_list_indices(len(images), target_count)
        sampling_formula = f"round(i * (N - 1) / ({target_count} - 1))"
        sampling_mode = "uniform_full_sequence"
    else:
        selected = strided_prefix_list_indices(
            len(images),
            source_prefix_count=int(source_prefix_count),
            source_stride=int(source_stride),
            target_count=target_count,
        )
        sampling_formula = (
            f"range(0, {int(source_prefix_count)}, {int(source_stride)})"
        )
        sampling_mode = "strided_source_prefix"
    for list_index in selected:
        with Image.open(images[list_index][1]) as image:
            image.verify()
            if image.size != expected_size or image.mode != expected_mode:
                raise RuntimeError(
                    f"Unexpected source image properties: {images[list_index][1]} "
                    f"size={image.size}, mode={image.mode}"
                )
    if check_only:
        return {
            "sequence": sequence_name,
            "source_count": len(images),
            "selected_count": target_count,
            "sampling_mode": sampling_mode,
            "selected_source_list_indices": selected,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = temporary / "images"
    camera_dir = temporary / "cameras"
    sparse_dir = temporary / "sparse"
    image_dir.mkdir(parents=True)
    camera_dir.mkdir()
    sparse_dir.mkdir()
    mappings: list[dict[str, Any]] = []
    original_gt: list[str] = []
    normalized_gt: list[str] = []
    c2w_poses: list[np.ndarray] = []
    sparse_points: list[np.ndarray] = []
    file_manifest: list[dict[str, Any]] = []
    max_round_trip_error = 0.0
    try:
        for new_index, list_index in enumerate(selected):
            source_frame, source_image = images[list_index]
            record = gt[source_image.name]
            if int(record["source_frame"]) != source_frame:
                raise RuntimeError("Image and GT source frame numbers differ")
            target_image = image_dir / f"{new_index:05d}_rgb.png"
            shutil.copy2(source_image, target_image)
            source_hash = sha256(source_image)
            target_hash = sha256(target_image)
            if source_hash != target_hash or source_image.stat().st_size != target_image.stat().st_size:
                raise RuntimeError(f"Image copy verification failed: {source_image}")
            file_manifest.append(
                {
                    "role": "image",
                    "source": str(source_image.resolve()),
                    "destination": str(target_image.relative_to(temporary)),
                    "size_bytes": int(target_image.stat().st_size),
                    "sha256": target_hash,
                }
            )
            rotation, translation = source_to_target_pose(record["rotation"], record["translation"])
            if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
                raise RuntimeError("Converted rotation determinant is invalid")
            r_w2c, t_w2c, t_cv = target_pose_to_json_extrinsics(rotation, translation)
            max_round_trip_error = max(
                max_round_trip_error,
                verify_round_trip(rotation, translation, r_w2c, t_w2c),
            )
            camera_name = f"{new_index:05d}_cam.json"
            write_camera_json(
                camera_dir / camera_name,
                width=expected_size[0],
                height=expected_size[1],
                r_w2c=r_w2c,
                t_w2c=t_w2c,
            )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3], pose[:3, 3] = rotation, translation
            c2w_poses.append(pose)
            sparse_points.append(t_cv)
            values = record["pose_values"]
            image_name = target_image.name
            original_gt.append(record["raw_line"])
            normalized_gt.append(
                f"{new_index} {image_name} " + " ".join(f"{value:.10g}" for value in values)
            )
            mappings.append(
                {
                    "new_index": new_index,
                    "new_image": image_name,
                    "new_camera": camera_name,
                    "source_list_index": list_index,
                    "source_frame": source_frame,
                    "source_image": source_image.name,
                    "source_gt_line": record["source_gt_line"],
                }
            )

        train_indices = list(range(0, target_count, 4))
        test_indices = list(range(2, target_count, 4))
        _write_indices(temporary / "train.txt", train_indices)
        _write_indices(temporary / "test.txt", test_indices)
        _write_indices(temporary / "selected_original_indices.txt", [images[index][0] for index in selected])
        _write_indices(temporary / "selected_source_list_indices.txt", selected)
        (temporary / "gt_selected_original.txt").write_text(
            GT_HEADER + "\n" + "\n".join(original_gt) + "\n", encoding="utf-8"
        )
        (temporary / "gt_selected.txt").write_text(
            GT_HEADER + "\n" + "\n".join(normalized_gt) + "\n", encoding="utf-8"
        )
        with (temporary / "frame_mapping.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(mappings[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(mappings)
        np.savez_compressed(
            temporary / "groundtruth_camera_to_world.npz",
            camera_to_worlds=np.stack(c2w_poses),
            image_names=np.asarray([row["new_image"] for row in mappings]),
            source_frame_numbers=np.asarray([row["source_frame"] for row in mappings], dtype=np.int64),
        )
        write_sparse_ply(sparse_dir / "sparse.ply", np.stack(sparse_points))
        generated = [
            temporary / "train.txt", temporary / "test.txt",
            temporary / "selected_original_indices.txt",
            temporary / "selected_source_list_indices.txt",
            temporary / "gt_selected_original.txt", temporary / "gt_selected.txt",
            temporary / "frame_mapping.tsv", temporary / "groundtruth_camera_to_world.npz",
            sparse_dir / "sparse.ply",
        ] + sorted(camera_dir.glob("*_cam.json"))
        for path in generated:
            file_manifest.append(
                {
                    "role": "generated",
                    "destination": str(path.relative_to(temporary)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path),
                }
            )
        metadata = {
            "format": "pfgs360_360vo_subset_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sequence": sequence_name,
            "source_sequence": str(source_sequence.resolve()),
            "source_valid_image_count": len(images),
            "selected_frame_count": target_count,
            "sampling_mode": sampling_mode,
            "sampling_formula": sampling_formula,
            "source_prefix_count": (
                None if source_prefix_count is None else int(source_prefix_count)
            ),
            "source_stride": int(source_stride),
            "sampling_interval_counts": dict(Counter(b - a for a, b in zip(selected[:-1], selected[1:]))),
            "first_selected_source_frame": images[selected[0]][0],
            "last_selected_source_frame": images[selected[-1]][0],
            "image_storage": "full copy",
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "max_pose_round_trip_error": max_round_trip_error,
        }
        (temporary / "conversion_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_path = temporary / "conversion_metadata.json"
        file_manifest.append(
            {
                "role": "generated",
                "destination": str(metadata_path.relative_to(temporary)),
                "size_bytes": int(metadata_path.stat().st_size),
                "sha256": sha256(metadata_path),
            }
        )
        (temporary / "file_manifest.json").write_text(
            json.dumps({"files": file_manifest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation = validate_sequence(
            temporary,
            target_count=target_count,
            expected_first_source=images[selected[0]][0],
            expected_last_source=images[selected[-1]][0],
        )
        output_data.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_data)
        return {**metadata, **validation, "output": str(output_data)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path("/mnt/disk1/zwh/Dataset/360VO")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/disk1/zwh/Dataset/360VO-PFGS360-200-formal-v1"),
    )
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument(
        "--source-prefix-count",
        type=int,
        default=None,
        help="Select only from the first N ordered source frames.",
    )
    parser.add_argument(
        "--source-stride",
        type=int,
        default=1,
        help="List-index stride used with --source-prefix-count.",
    )
    parser.add_argument("--sequences", nargs="*", default=[f"seq{i}" for i in range(10)])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    invalid = [name for name in args.sequences if re.fullmatch(r"seq[0-9]", name) is None]
    if invalid:
        raise ValueError(f"Invalid sequence names: {invalid}")
    results = [
        convert_sequence(
            args.source_root,
            args.output_root,
            sequence,
            target_count=args.target_count,
            source_prefix_count=args.source_prefix_count,
            source_stride=args.source_stride,
            check_only=bool(args.check_only),
        )
        for sequence in args.sequences
    ]
    if (
        not args.check_only
        and set(args.sequences) == {f"seq{i}" for i in range(10)}
        and len(args.sequences) == 10
    ):
        dataset_manifest = {
            "format": "pfgs360_360vo_formal_dataset_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(args.source_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "target_count": int(args.target_count),
            "source_prefix_count": args.source_prefix_count,
            "source_stride": int(args.source_stride),
            "sequences": results,
        }
        (args.output_root / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_root / "DATASET_READY.marker").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
