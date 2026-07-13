from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from frontend.pano_droid.dataset import discover_ob3d_images, load_ob3d_camera_c2w


def test_discover_ob3d_images_sorts_by_numeric_frame_id(tmp_path: Path) -> None:
    image_dir = tmp_path / "sponza" / "Egocentric" / "images"
    image_dir.mkdir(parents=True)
    for name in ["00010_rgb.png", "00002_rgb.png", "00001_rgb.png"]:
        Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(image_dir / name)

    files = discover_ob3d_images(str(tmp_path), scene="sponza", split="Egocentric")

    assert [Path(path).name for path in files] == ["00001_rgb.png", "00002_rgb.png", "00010_rgb.png"]


def test_load_ob3d_camera_c2w_neighbor_json(tmp_path: Path) -> None:
    image_dir = tmp_path / "sponza" / "Egocentric" / "images"
    camera_dir = tmp_path / "sponza" / "Egocentric" / "cameras"
    image_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)
    image_path = image_dir / "00033_rgb.png"
    Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(image_path)
    payload = [
        {
            "extrinsics": {
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [1.0, 2.0, 3.0],
            }
        }
    ]
    with open(camera_dir / "00033_cam.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    c2w = load_ob3d_camera_c2w(image_path)

    assert c2w is not None
    assert c2w.shape == (4, 4)
    assert np.allclose(c2w[:3, 3], np.array([-1.0, -2.0, -3.0], dtype=np.float32))


def test_load_ob3d_camera_inverts_non_identity_w2c(tmp_path: Path) -> None:
    image_dir = tmp_path / "sponza" / "Non-Egocentric" / "images"
    camera_dir = tmp_path / "sponza" / "Non-Egocentric" / "cameras"
    image_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)
    image_path = image_dir / "00007_rgb.png"
    Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(image_path)
    rotation_w2c = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    translation_w2c = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    payload = [{"extrinsics": {"rotation": rotation_w2c.tolist(), "translation": translation_w2c.tolist()}}]
    (camera_dir / "00007_cam.json").write_text(json.dumps(payload), encoding="utf-8")

    c2w = load_ob3d_camera_c2w(image_path)

    assert c2w is not None
    np.testing.assert_allclose(c2w[:3, :3], rotation_w2c.T)
    np.testing.assert_allclose(c2w[:3, 3], -rotation_w2c.T @ translation_w2c)
