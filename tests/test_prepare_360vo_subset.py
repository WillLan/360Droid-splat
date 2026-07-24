from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_360vo_pfgs360_subset import (
    convert_sequence,
    sha256,
    strided_prefix_list_indices,
    uniform_list_indices,
)


def test_uniform_200_sampling_is_unique_and_includes_full_sequence_endpoints() -> None:
    indices = uniform_list_indices(2041, 200)

    assert len(indices) == 200
    assert len(set(indices)) == 200
    assert indices[0] == 0
    assert indices[-1] == 2040
    assert indices == sorted(indices)


def test_strided_prefix_sampling_uses_first_500_every_two_frames() -> None:
    indices = strided_prefix_list_indices(
        2041,
        source_prefix_count=500,
        source_stride=2,
        target_count=250,
    )

    assert indices == list(range(0, 500, 2))
    assert len(indices) == 250
    assert indices[0] == 0
    assert indices[-1] == 498


def test_subset_conversion_copies_images_and_preserves_pose_mapping(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    sequence = source / "seq0"
    images = sequence / "images"
    images.mkdir(parents=True)
    gt_lines = ["#frame name x y z qx qy qz qw"]
    for index in range(7):
        name = f"Frame_{10 + index * 2}_FinalColor.png"
        Image.new("RGBA", (8, 4), (index, index + 1, index + 2, 255)).save(images / name)
        gt_lines.append(f"{10 + index * 2} {name} {index} 0 0 0 0 0 1")
    (sequence / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    output = tmp_path / "converted"

    result = convert_sequence(
        source,
        output,
        "seq0",
        target_count=4,
        expected_size=(8, 4),
        expected_mode="RGBA",
    )

    target = output / "seq0/Egocentric"
    copied = sorted((target / "images").glob("*_rgb.png"))
    assert len(copied) == 4
    assert all(path.is_file() and not path.is_symlink() for path in copied)
    rows = list(csv.DictReader((target / "frame_mapping.tsv").open(), delimiter="\t"))
    assert [int(row["source_list_index"]) for row in rows] == [0, 2, 4, 6]
    assert [int(row["source_frame"]) for row in rows] == [10, 14, 18, 22]
    poses = np.load(target / "groundtruth_camera_to_world.npz")["camera_to_worlds"]
    assert poses.shape == (4, 4, 4)
    assert np.isfinite(poses).all()
    metadata = json.loads((target / "conversion_metadata.json").read_text())
    assert metadata["image_storage"] == "full copy"
    assert metadata["max_pose_round_trip_error"] <= 1.0e-8
    manifest = json.loads((target / "file_manifest.json").read_text())["files"]
    image_entries = [entry for entry in manifest if entry["role"] == "image"]
    assert len(image_entries) == 4
    for entry in image_entries:
        destination = target / entry["destination"]
        assert sha256(destination) == entry["sha256"]
        assert sha256(Path(entry["source"])) == entry["sha256"]
    assert result["first_selected_source_frame"] == 10
    assert result["last_selected_source_frame"] == 22


def test_prefix_stride_conversion_reindexes_images_and_matching_poses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    sequence = source / "seq0"
    images = sequence / "images"
    images.mkdir(parents=True)
    gt_lines = ["#frame name x y z qx qy qz qw"]
    for index in range(8):
        name = f"Frame_{100 + index}_FinalColor.png"
        Image.new("RGBA", (8, 4), (index, 0, 0, 255)).save(images / name)
        gt_lines.append(f"{100 + index} {name} {index} 0 0 0 0 0 1")
    (sequence / "gt.txt").write_text(
        "\n".join(gt_lines) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "converted"

    result = convert_sequence(
        source,
        output,
        "seq0",
        target_count=3,
        source_prefix_count=6,
        source_stride=2,
        expected_size=(8, 4),
        expected_mode="RGBA",
    )

    target = output / "seq0/Egocentric"
    rows = list(
        csv.DictReader(
            (target / "frame_mapping.tsv").open(),
            delimiter="\t",
        )
    )
    assert [int(row["new_index"]) for row in rows] == [0, 1, 2]
    assert [int(row["source_list_index"]) for row in rows] == [0, 2, 4]
    assert [int(row["source_frame"]) for row in rows] == [100, 102, 104]
    assert result["sampling_mode"] == "strided_source_prefix"
    assert result["sampling_formula"] == "range(0, 6, 2)"
    assert result["first_selected_source_frame"] == 100
    assert result["last_selected_source_frame"] == 104
