from pathlib import Path

import torch

from frontend.pano_droid.checkpoint import load_checkpoint
from frontend.pano_droid.model import PanoDroidModel
from frontend.pano_droid.train import load_train_config, train


def test_tiny_training_saves_checkpoint(tmp_path: Path):
    cfg = load_train_config(None)
    cfg["Dataset"].update({"synthetic_length": 4, "height": 16, "width": 32})
    cfg["Model"].update({"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1})
    cfg["Training"].update({"output_dir": str(tmp_path), "max_steps": 1, "batch_size": 1})
    result = train(cfg)
    ckpt = Path(result["checkpoint"])
    assert result["steps"] == 1
    assert ckpt.is_file()
    model = PanoDroidModel(**cfg["Model"])
    payload = load_checkpoint(str(ckpt), model, map_location="cpu", strict=True)
    assert payload["step"] == 1
    pred = model(torch.rand(1, 3, 16, 32), torch.rand(1, 3, 16, 32))
    assert pred["spherical_flow"].shape == (1, 2, 16, 32)

