from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from frontend.pano_droid.dataset import (
    discover_rar_pano_images,
    load_rar_pano_reconstruction_c2w,
)
from system.pano_droid_gs_slam import iter_sequence_frames


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4), color=(32, 64, 96)).save(path)


def _write_reconstruction(
    sequence_dir: Path,
    shots: dict[str, dict[str, list[float]]],
) -> None:
    (sequence_dir / "reconstruction.json").write_text(
        json.dumps([{"shots": shots}]),
        encoding="utf-8",
    )


def test_rar_pano_discovery_uses_only_contiguous_numeric_frames(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "I_alley" / "images"
    _write_image(image_dir / "002.png")
    _write_image(image_dir / "001.png")
    _write_image(image_dir / "comparison_001.png")

    images = discover_rar_pano_images(str(tmp_path), sequence="I_alley")

    assert [Path(path).name for path in images] == ["001.png", "002.png"]


def test_rar_pano_opensfm_world_to_camera_is_inverted_for_pseudo_gt(
    tmp_path: Path,
) -> None:
    sequence_dir = tmp_path / "I_bridge"
    _write_image(sequence_dir / "images" / "001.png")
    _write_reconstruction(
        sequence_dir,
        {
            "001.png": {
                "rotation": [0.0, 0.0, math.pi / 2.0],
                "translation": [1.0, 2.0, 3.0],
            }
        },
    )

    poses = load_rar_pano_reconstruction_c2w(
        str(tmp_path),
        sequence="I_bridge",
    )
    pose = poses["001.png"]
    expected_rotation = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(pose[:3, :3], expected_rotation, atol=1.0e-6)
    np.testing.assert_allclose(
        pose[:3, 3],
        -expected_rotation @ np.array([1.0, 2.0, 3.0], dtype=np.float32),
        atol=1.0e-6,
    )


def test_rar_pano_runtime_attaches_opensfm_pose_only_as_evaluation_metadata(
    tmp_path: Path,
) -> None:
    sequence_dir = tmp_path / "O_car"
    for name in ("001.png", "002.png"):
        _write_image(sequence_dir / "images" / name)
    _write_reconstruction(
        sequence_dir,
        {
            "001.png": {
                "rotation": [0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
            },
            "002.png": {
                "rotation": [0.0, 0.0, 0.0],
                "translation": [1.0, 0.0, 0.0],
            },
        },
    )
    config = {
        "Dataset": {
            "type": "rar_pano",
            "dataset_path": str(tmp_path),
            "scene": "O_car",
            "split": "Full",
            "erp_resize_height": 4,
            "erp_resize_width": 8,
        }
    }

    frames = list(iter_sequence_frames(config))

    assert [frame.frame_id for frame in frames] == [0, 1]
    assert [frame.meta["source_frame_index"] for frame in frames] == [0, 1]
    assert all(
        frame.meta["gt_pose_source"] == "opensfm_reconstruction_pseudo_gt"
        for frame in frames
    )
    torch.testing.assert_close(frames[0].meta["gt_c2w"], torch.eye(4))
    torch.testing.assert_close(
        frames[1].meta["gt_c2w"][:3, 3],
        torch.tensor([-1.0, 0.0, 0.0]),
    )
    assert not hasattr(frames[0], "gt_c2w")
