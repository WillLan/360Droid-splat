"""Train PanoDROID with DROID-style multi-frame graph supervision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": None,
            "mode": "online",
            "account": "zb2302106@buaa.edu.cn",
            "log_every": 10,
        },
        "Distributed": {
            "enabled": "auto",
            "backend": "auto",
            "find_unused_parameters": True,
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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    return int(value)


def _setup_distributed(config: dict) -> dict:
    dist_cfg = config.get("Distributed", {})
    requested = dist_cfg.get("enabled", "auto")
    env_world = _env_int("WORLD_SIZE", 1)
    if isinstance(requested, str):
        enabled = requested.lower() == "true" or (requested.lower() == "auto" and env_world > 1)
    else:
        enabled = bool(requested)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = env_world if enabled else 1

    if torch.cuda.is_available():
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if enabled and not dist.is_initialized():
        backend = str(dist_cfg.get("backend", "auto"))
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def _is_main_process(ddp: dict) -> bool:
    return int(ddp["rank"]) == 0


def _barrier(ddp: dict) -> None:
    if ddp["enabled"] and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed(ddp: dict) -> None:
    if ddp["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _forward_graph(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    edges: list[tuple[int, int]],
    num_updates: int,
) -> dict:
    if isinstance(model, DistributedDataParallel):
        return model(images, edges=edges, num_updates=num_updates)
    return model.forward_graph(images, edges=edges, num_updates=num_updates)


def _reduce_metrics(metrics: dict[str, torch.Tensor], ddp: dict) -> dict[str, float]:
    keys = sorted(metrics)
    values = torch.stack([metrics[k].detach().float() for k in keys]).to(ddp["device"])
    if ddp["enabled"] and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values / float(ddp["world_size"])
    return {k: float(v.detach().cpu()) for k, v in zip(keys, values)}


def _init_wandb(config: dict, output_dir: Path, *, enabled_on_rank: bool):
    wb_cfg = config.get("WeightsAndBiases", {})
    if not enabled_on_rank or not bool(wb_cfg.get("enabled", False)):
        return None
    mode = str(wb_cfg.get("mode") or "online")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WeightsAndBiases.enabled=true requires the 'wandb' package. "
            "Install it or disable W&B logging."
        ) from exc

    run = wandb.init(
        project=str(wb_cfg.get("project") or "360Droid-splat"),
        entity=wb_cfg.get("entity") or None,
        name=wb_cfg.get("run_name") or None,
        mode=mode,
        dir=str(output_dir),
        config=config,
        tags=wb_cfg.get("tags") or None,
    )
    return run


def _wandb_log_metrics(run, metrics: dict[str, float], *, step: int) -> None:
    if run is None:
        return
    run.log({f"train/{k}": v for k, v in metrics.items()}, step=int(step))


def _wandb_log_diagnostics(run, vis_metrics: dict, *, step: int) -> None:
    if run is None:
        return
    import wandb

    payload = {}
    for key, name in (("trajectory_png", "trajectory_3d"), ("depth_png", "depth_pred_gt_error")):
        path = vis_metrics.get(key)
        if path and Path(path).is_file():
            payload[f"diagnostics/{name}"] = wandb.Image(str(path))
    for key in ("trajectory_rmse", "depth_mae"):
        if key in vis_metrics:
            payload[f"diagnostics/{key}"] = float(vis_metrics[key])
    if payload:
        run.log(payload, step=int(step))


def train_graph(config: dict) -> dict:
    ddp = _setup_distributed(config)
    device = ddp["device"]
    is_main = _is_main_process(ddp)
    dataset = build_graph_dataset_from_config(config, train=True)
    tr = config.get("Training", {})
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=int(ddp["world_size"]),
            rank=int(ddp["rank"]),
            shuffle=True,
            drop_last=False,
        )
        if ddp["enabled"]
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(tr.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(tr.get("num_workers", 0)),
        drop_last=False,
    )
    model = PanoDroidModel(**config.get("Model", {})).to(device)
    if ddp["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[int(ddp["local_rank"])] if device.type == "cuda" else None,
            output_device=int(ddp["local_rank"]) if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("Distributed", {}).get("find_unused_parameters", True)),
        )
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
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    _barrier(ddp)
    max_steps = int(tr.get("max_steps", 0))
    save_every = max(1, int(tr.get("save_every", 100)))
    vis_cfg = config.get("Visualization", {})
    vis_enabled = bool(vis_cfg.get("enabled", True))
    vis_every = max(1, int(vis_cfg.get("every", save_every)))
    wb_cfg = config.get("WeightsAndBiases", {})
    wandb_run = _init_wandb(config, output_dir, enabled_on_rank=is_main)
    wandb_log_every = max(1, int(wb_cfg.get("log_every", 10)))
    step = 0
    best = float("inf")
    start = time.time()
    last_metrics: dict[str, float] = {}

    try:
        for epoch in range(int(tr.get("epochs", 1))):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                images = batch["images"].to(device)
                pred = _forward_graph(
                    model,
                    images,
                    edges=edges,
                    num_updates=int(tr.get("iters", config.get("Model", {}).get("update_iters", 1))),
                )
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
                last_metrics = _reduce_metrics(metrics, ddp)
                if is_main and (step == 1 or step % wandb_log_every == 0):
                    _wandb_log_metrics(wandb_run, last_metrics, step=step)
                if is_main and step % save_every == 0:
                    save_checkpoint(
                        str(ckpt_dir / "latest.pt"),
                        model=_unwrap_model(model),
                        optimizer=optimizer,
                        step=step,
                        epoch=epoch,
                        config=config,
                        metrics=last_metrics,
                    )
                if is_main and last_metrics["loss"] < best:
                    best = last_metrics["loss"]
                    save_checkpoint(
                        str(ckpt_dir / "best.pt"),
                        model=_unwrap_model(model),
                        optimizer=optimizer,
                        step=step,
                        epoch=epoch,
                        config=config,
                        metrics=last_metrics,
                    )
                if is_main and vis_enabled and (step == 1 or step % vis_every == 0):
                    chain_edges = [(i, i + 1) for i in range(images.shape[1] - 1)]
                    model.eval()
                    with torch.no_grad():
                        pred_vis = _unwrap_model(model).forward_graph(
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
                    _wandb_log_metrics(wandb_run, last_metrics, step=step)
                    _wandb_log_diagnostics(wandb_run, vis_metrics, step=step)
                if max_steps > 0 and step >= max_steps:
                    break
            if max_steps > 0 and step >= max_steps:
                break
        if is_main:
            save_checkpoint(
                str(ckpt_dir / "latest.pt"),
                model=_unwrap_model(model),
                optimizer=optimizer,
                step=step,
                epoch=max(0, int(tr.get("epochs", 1)) - 1),
                config=config,
                metrics=last_metrics,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _barrier(ddp)
        _cleanup_distributed(ddp)

    if not is_main:
        return {
            "steps": step,
            "best_loss": best,
            "last_metrics": last_metrics,
            "checkpoint": str(ckpt_dir / "latest.pt"),
            "elapsed_sec": time.time() - start,
            "edges": edges,
            "rank": int(ddp["rank"]),
            "world_size": int(ddp["world_size"]),
        }
    return {
        "steps": step,
        "best_loss": best,
        "last_metrics": last_metrics,
        "checkpoint": str(ckpt_dir / "latest.pt"),
        "elapsed_sec": time.time() - start,
        "edges": edges,
        "rank": int(ddp["rank"]),
        "world_size": int(ddp["world_size"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    cfg = load_graph_train_config(args.config)
    if args.max_steps is not None:
        cfg.setdefault("Training", {})["max_steps"] = int(args.max_steps)
    if args.wandb:
        cfg.setdefault("WeightsAndBiases", {})["enabled"] = True
    if args.wandb_mode is not None:
        cfg.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    if args.run_name is not None:
        cfg.setdefault("WeightsAndBiases", {})["run_name"] = args.run_name
    result = train_graph(cfg)
    if int(result.get("rank", 0)) == 0:
        print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
