import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from frontend.pano_droid.graph_dataset import PanoCityGraphDataset
from frontend.pano_droid.train_graph import load_graph_train_config, train_graph


def _write_mock_panocity_block(root: Path, *, n_frames: int = 7, size=(32, 16)) -> None:
    block = root / "beijing_block1"
    rgb_dir = block / "pano_images"
    depth_dir = block / "panodepth_images"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    frames = []
    width, height = size
    for i in range(n_frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (i * 20) % 255
        rgb[..., 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        rgb[..., 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        depth = np.full((height, width), 1000 + i, dtype=np.uint16)
        name = f"pano_{i:07d}.png"
        depth_name = f"pano_depth_{i:07d}.png"
        Image.fromarray(rgb).save(rgb_dir / name)
        Image.fromarray(depth).save(depth_dir / depth_name)
        T = np.eye(4, dtype=float)
        T[0, 3] = float(i)
        frames.append(
            {
                "name": name,
                "depth": depth_name,
                "transformation_matrix": T.tolist(),
            }
        )
    with open(block / "beijing_Pano_block1_poses.json", "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f)


def test_panocity_graph_dataset_outputs_512x1024(tmp_path: Path):
    _write_mock_panocity_block(tmp_path, n_frames=7)
    ds = PanoCityGraphDataset(
        str(tmp_path),
        n_frames=7,
        resize=(512, 1024),
        depth_scale=0.001,
    )
    sample = ds[0]
    assert sample["images"].shape == (7, 3, 512, 1024)
    assert sample["depths"].shape == (7, 1, 512, 1024)
    assert sample["inverse_depths"].shape == (7, 1, 512, 1024)
    assert sample["poses_c2w"].shape == (7, 4, 4)
    assert sample["block_name"] == "beijing_block1"
    assert torch.isfinite(sample["images"]).all()
    assert torch.isfinite(sample["inverse_depths"]).all()


def test_train_graph_synthetic_smoke(tmp_path: Path):
    cfg = load_graph_train_config(None)
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 2, "n_frames": 3, "height": 16, "width": 32})
    cfg["Model"].update(
        {
            "feature_dim": 8,
            "context_dim": 8,
            "hidden_dim": 8,
            "encoder_base_dim": 8,
            "corr_levels": 1,
            "corr_radius": 1,
            "update_iters": 1,
        }
    )
    cfg["Graph"].update({"temporal_radius": 1, "max_edges_per_step": 2, "loss_sample_height": 8, "loss_sample_width": 16})
    cfg["Training"].update({"output_dir": str(tmp_path), "max_steps": 2, "batch_size": 1, "num_workers": 0, "iters": 1})
    result = train_graph(cfg)
    assert result["steps"] == 2
    assert Path(result["checkpoint"]).is_file()
    assert np.isfinite(result["best_loss"])
    vis_dir = tmp_path / "visualizations"
    assert (vis_dir / "step_0000001_trajectory.png").is_file()
    assert (vis_dir / "step_0000001_depth.png").is_file()
