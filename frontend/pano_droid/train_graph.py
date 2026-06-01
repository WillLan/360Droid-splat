"""Train PanoDROID with DROID-style multi-frame graph supervision."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .checkpoint import save_checkpoint
from .graph_dataset import build_graph_dataset_from_config
from .graph_losses import (
    GraphLossWeights,
    build_temporal_edges,
    graph_supervised_loss,
    select_training_edges,
)
from .model import PanoDroidModel
from .visualization import save_graph_diagnostics


def _default_config() -> dict:
    return {
        "Dataset": {
            "synthetic": True,
            "synthetic_length": 4,
            "n_frames": 3,
            "height": 32,
            "width": 64,
        },
        "Graph": {
            "temporal_radius": 2,
            "bidirectional": True,
            "max_edges_per_step": 4,
            "loss_sample_height": 16,
            "loss_sample_width": 32,
        },
        "Model": {
            "feature_dim": 16,
            "context_dim": 16,
            "hidden_dim": 16,
            "encoder_base_dim": 16,
            "corr_levels": 2,
            "corr_radius": 1,
            "update_iters": 1,
        },
        "Training": {
            "epochs": 1,
            "batch_size": 1,
            "lr": 2.5e-4,
            "num_workers": 0,
            "max_steps": 2,
            "save_every": 1,
            "grad_clip": 2.5,
            "output_dir": "outputs/pano_droid_graph_train",
        },
        "Visualization": {
            "enabled": True,
            "every": 1,
        },
        "Loss": {},
    }


def load_graph_train_config(path: str | None) -> dict:
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


def train_graph(config: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_graph_dataset_from_config(config, train=True)
    tr = config.get("Training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tr.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(tr.get("num_workers", 0)),
        drop_last=False,
    )
    model = PanoDroidModel(**config.get("Model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 2.5e-4)))
    graph_cfg = config.get("Graph", {})
    edges = build_temporal_edges(
        int(config.get("Dataset", {}).get("n_frames", 7)),
        radius=int(graph_cfg.get("temporal_radius", 2)),
        bidirectional=bool(graph_cfg.get("bidirectional", True)),
    )
    edges = select_training_edges(edges, max_edges=int(graph_cfg.get("max_edges_per_step", 0)))
    if not edges:
        raise ValueError("Graph has no training edges.")
    loss_keys = set(GraphLossWeights.__dataclass_fields__)
    loss_cfg = config.get("Loss", {})
    weights = GraphLossWeights(**{k: float(v) for k, v in loss_cfg.items() if k in loss_keys})

    output_dir = Path(tr.get("output_dir", "outputs/pano_droid_graph_train"))
    ckpt_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "visualizations"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(tr.get("max_steps", 0))
    save_every = max(1, int(tr.get("save_every", 100)))
    vis_cfg = config.get("Visualization", {})
    vis_enabled = bool(vis_cfg.get("enabled", True))
    vis_every = max(1, int(vis_cfg.get("every", save_every)))
    step = 0
    best = float("inf")
    start = time.time()
    last_metrics: dict[str, float] = {}

    for epoch in range(int(tr.get("epochs", 1))):
        for batch in loader:
            images = batch["images"].to(device)
            pred = model.forward_graph(images, edges=edges, num_updates=int(tr.get("iters", config.get("Model", {}).get("update_iters", 1))))
            loss, metrics = graph_supervised_loss(
                {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()},
                pred,
                weights=weights,
                sample_height=int(graph_cfg.get("loss_sample_height", 32)),
                sample_width=int(graph_cfg.get("loss_sample_width", 64)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 2.5)))
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
            if vis_enabled and (step == 1 or step % vis_every == 0):
                chain_edges = [(i, i + 1) for i in range(images.shape[1] - 1)]
                model.eval()
                with torch.no_grad():
                    pred_vis = model.forward_graph(
                        images[:1],
                        edges=chain_edges,
                        num_updates=int(tr.get("iters", config.get("Model", {}).get("update_iters", 1))),
                    )
                    vis_metrics = save_graph_diagnostics(
                        {k: v[:1].detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                        {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in pred_vis.items()},
                        output_dir=vis_dir,
                        step=step,
                    )
                model.train()
                last_metrics.update(
                    {
                        "vis_trajectory_rmse": float(vis_metrics["trajectory_rmse"]),
                        "vis_depth_mae": float(vis_metrics["depth_mae"]),
                    }
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
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_graph_train_config(args.config)
    if args.max_steps is not None:
        cfg.setdefault("Training", {})["max_steps"] = int(args.max_steps)
    result = train_graph(cfg)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
