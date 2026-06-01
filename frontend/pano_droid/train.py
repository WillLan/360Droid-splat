"""Train PanoDROID-MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .checkpoint import save_checkpoint
from .dataset import build_dataset_from_config
from .losses import LossWeights, PanoDroidLoss
from .model import PanoDroidModel


def _default_config() -> dict:
    return {
        "Dataset": {"synthetic": True, "synthetic_length": 16, "height": 32, "width": 64},
        "Model": {"feature_dim": 32, "context_dim": 32, "hidden_dim": 48, "update_iters": 2},
        "Training": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "num_workers": 0,
            "max_steps": 4,
            "save_every": 2,
            "output_dir": "outputs/pano_droid_train",
        },
        "Loss": {},
    }


def load_train_config(path: str | None) -> dict:
    cfg = _default_config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(cfg, user_cfg)
    return cfg


def train(config: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset_from_config(config, train=True)
    tr = config.get("Training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tr.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(tr.get("num_workers", 0)),
        drop_last=False,
    )
    model = PanoDroidModel(**config.get("Model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 1e-3)))
    loss_cfg = config.get("Loss", {})
    valid_loss_keys = set(LossWeights.__dataclass_fields__)
    weights = LossWeights(**{k: float(v) for k, v in loss_cfg.items() if k in valid_loss_keys})
    criterion = PanoDroidLoss(weights)

    output_dir = Path(tr.get("output_dir", "outputs/pano_droid_train"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(tr.get("max_steps", 0))
    save_every = max(1, int(tr.get("save_every", 100)))
    step = 0
    best = float("inf")
    start = time.time()
    last_metrics = {}
    for epoch in range(int(tr.get("epochs", 1))):
        for batch in loader:
            image0 = batch["image0"].to(device)
            image1 = batch["image1"].to(device)
            pred = model(image0, image1)
            loss, metrics = criterion(batch, pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
            optimizer.step()
            step += 1
            last_metrics = {k: float(v.detach().cpu()) for k, v in metrics.items()}
            if step % save_every == 0:
                save_checkpoint(
                    str(ckpt_dir / "latest.pt"),
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=epoch,
                    config=config,
                    metrics=last_metrics,
                )
            if last_metrics["loss"] < best:
                best = last_metrics["loss"]
                save_checkpoint(
                    str(ckpt_dir / "best.pt"),
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=epoch,
                    config=config,
                    metrics=last_metrics,
                )
            if max_steps > 0 and step >= max_steps:
                break
        if max_steps > 0 and step >= max_steps:
            break
    save_checkpoint(
        str(ckpt_dir / "latest.pt"),
        model=model,
        optimizer=optimizer,
        step=step,
        epoch=max(0, int(tr.get("epochs", 1)) - 1),
        config=config,
        metrics=last_metrics,
    )
    return {
        "steps": step,
        "best_loss": best,
        "last_metrics": last_metrics,
        "checkpoint": str(ckpt_dir / "latest.pt"),
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = train(load_train_config(args.config))
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
