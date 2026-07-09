import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from data.stage1_pano_sequence_dataset import (
    Stage1PanoSequenceDataset,
    build_stage1_windows,
    load_stage1_manifest,
    summarize_stage1_manifest,
)
from tools.build_stage1_airsim_dtw_nyc_manifest import build_airsim_records, build_debug_records
from tools.build_stage1_dataset_manifest import build_manifest_records
from tools.check_stage1_overlap import check_manifest_overlap


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color).save(path)


def _write_manifest(path: Path) -> None:
    records = []
    for domain, offset in (("indoor", 0), ("outdoor", 10)):
        for idx in range(4):
            image_path = path.parent / domain / f"frame_{idx:03d}.png"
            _write_image(image_path, (idx * 20, offset, 100))
            records.append(
                {
                    "scene_id": f"{domain}_scene",
                    "sequence_id": "seq_000",
                    "frame_id": idx,
                    "rgb_path": str(image_path),
                    "depth_path": None,
                    "pose_path": None,
                    "timestamp": float(idx),
                    "split": "train",
                    "domain": domain,
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)


def test_manifest_loads_and_summarizes_indoor_outdoor(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    records = load_stage1_manifest(manifest)
    summary = summarize_stage1_manifest(records)
    assert summary["num_records"] == 8
    assert summary["domains"] == {"indoor": 4, "outdoor": 4}
    assert summary["splits"] == {"train": 8}
    windows = build_stage1_windows(records, views_per_sample=4, domains=["indoor"], split="train")
    assert len(windows) == 1
    assert [record.frame_id for record in windows[0]] == [0, 1, 2, 3]


def test_stage1_dataset_returns_v4_sample_and_pairs(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    dataset = Stage1PanoSequenceDataset(
        manifest,
        domains=["indoor"],
        views_per_sample=4,
        image_height=8,
        image_width=16,
    )
    sample = dataset[0]
    assert sample["images"].shape == (4, 3, 8, 16)
    assert sample["pair_indices"].tolist() == [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]]
    assert sample["domain"] == "indoor"
    assert sample["depths"] is None
    assert sample["poses_c2w"] is None


def test_stage1_dataset_loads_h5_depth(tmp_path: Path):
    records = []
    for idx in range(4):
        image_path = tmp_path / f"frame_{idx:03d}.png"
        _write_image(image_path, (idx * 20, 10, 100))
        depth_path = tmp_path / f"Depth_{idx}.h5"
        with h5py.File(depth_path, "w") as handle:
            handle.create_dataset("depth", data=np.full((8, 16), idx + 1, dtype=np.float32))
        records.append(
            {
                "scene_id": "airsim",
                "sequence_id": "seq_000",
                "frame_id": idx,
                "rgb_path": str(image_path),
                "depth_path": str(depth_path),
                "pose_path": None,
                "timestamp": float(idx),
                "split": "train",
                "domain": "outdoor",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    sample = Stage1PanoSequenceDataset(manifest, image_height=8, image_width=16)[0]
    assert sample["depths"].shape == (4, 1, 8, 16)
    assert sample["depths"][0].mean().item() == 1.0
    assert sample["depths"][3].mean().item() == 4.0


def test_manifest_builder_and_overlap_checker(tmp_path: Path):
    root = tmp_path / "root"
    for idx in range(4):
        _write_image(root / "seq_a" / f"erp_{idx:03d}.png", (idx, idx, idx))
    records = build_manifest_records(root, domain="outdoor", split="train", scene_id="scene_a")
    assert len(records) == 4
    manifest = tmp_path / "built.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)
    result = check_manifest_overlap(str(manifest), views_per_sample=4)
    assert result["domains"] == {"outdoor": 4}
    assert result["valid_windows"] == 1
    assert result["has_trainable_windows"] is True


def test_airsim_dtw_nyc_manifest_builder_uses_only_raw_and_h5_depth(tmp_path: Path):
    root = tmp_path / "Omni360-Scene"
    for scene, prefix in (("DTW", "dtw"), ("NYC", "nyc")):
        raw = root / scene / f"{prefix}_Raw"
        depth = root / scene / f"{prefix}_Depth"
        raw.mkdir(parents=True)
        depth.mkdir(parents=True)
        for idx in range(6):
            _write_image(raw / f"panorama_{idx}.png", (idx, 20, 30))
            with h5py.File(depth / f"Depth_{idx}.h5", "w") as handle:
                handle.create_dataset("depth", data=np.ones((8, 16), dtype=np.float32))
    records, audit = build_airsim_records(root, train_ratio=0.5)
    debug = build_debug_records(records, max_frames_per_scene=2)
    assert len(records) == 12
    assert len(debug) == 4
    assert {record["scene_id"] for record in records} == {"DTW", "NYC"}
    assert {record["domain"] for record in records} == {"outdoor"}
    assert all(record["pose_path"] is None for record in records)
    assert all(record["rgb_path"].endswith(".png") for record in records)
    assert all(record["depth_path"].endswith(".h5") for record in records)
    assert audit["scene_stats"]["DTW"]["train_records"] == 3
    assert audit["scene_stats"]["NYC"]["val_records"] == 3
