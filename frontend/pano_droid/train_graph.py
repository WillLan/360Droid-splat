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

from .checkpoint import load_checkpoint, save_checkpoint
from .graph_dataset import build_graph_dataset_from_config
from .graph_losses import (
    GraphLossWeights,
    build_proximity_edges,
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
            "edge_strategy": "mixed",
            "temporal_radius": 2,
            "bidirectional": True,
            "max_edges_per_step": 4,
            "ba_iters_per_update": 2,
            "ba_sample_stride": 1,
            "fixed_frames": 2,
            "loss_gamma": 0.9,
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
            "scheduler": "onecycle",
            "restart_prob": 0.2,
            "resume_checkpoint": None,
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
    poses_c2w: torch.Tensor,
    init_poses_c2w: torch.Tensor | None,
    init_inverse_depth: torch.Tensor | None,
    ba_iters_per_update: int,
    fixed_frames: int,
    ba_sample_stride: int,
) -> dict:
    if isinstance(model, DistributedDataParallel):
        return model(
            images,
            edges=edges,
            num_updates=num_updates,
            poses_c2w=poses_c2w,
            init_poses_c2w=init_poses_c2w,
            init_inverse_depth=init_inverse_depth,
            ba_iters_per_update=ba_iters_per_update,
            fixed_frames=fixed_frames,
            ba_sample_stride=ba_sample_stride,
        )
    return model.forward_graph(
        images,
        edges=edges,
        num_updates=num_updates,
        poses_c2w=poses_c2w,
        init_poses_c2w=init_poses_c2w,
        init_inverse_depth=init_inverse_depth,
        ba_iters_per_update=ba_iters_per_update,
        fixed_frames=fixed_frames,
        ba_sample_stride=ba_sample_stride,
    )


def _make_graph_edges(batch: dict, graph_cfg: dict, *, step: int) -> list[tuple[int, int]]:
    n_frames = int(batch["images"].shape[1])
    radius = int(graph_cfg.get("temporal_radius", 2))
    bidirectional = bool(graph_cfg.get("bidirectional", True))
    max_edges = int(graph_cfg.get("max_edges_per_step", 0))
    strategy = str(graph_cfg.get("edge_strategy", "mixed")).lower()
    use_proximity = strategy == "proximity"
    if strategy == "mixed":
        use_proximity = bool(torch.rand((), generator=torch.Generator().manual_seed(int(step) + 1729)).item() < 0.5)
    if use_proximity and "poses_c2w" in batch:
        edges = build_proximity_edges(
            batch["poses_c2w"],
            radius=radius,
            max_edges=max(max_edges, n_frames * 2) if max_edges > 0 else 0,
            bidirectional=bidirectional,
        )
    else:
        edges = build_temporal_edges(n_frames, radius=radius, bidirectional=bidirectional)
    gen = torch.Generator().manual_seed(int(step) + 7919)
    edges = select_training_edges(edges, max_edges=max_edges, n_frames=n_frames, generator=gen)
    if not edges:
        raise ValueError("Graph has no training edges.")
    return edges


def _make_scheduler(optimizer: torch.optim.Optimizer, tr: dict, *, total_steps: int):
    name = str(tr.get("scheduler", "onecycle") or "none").lower()
    if name in ("none", "disabled", "false"):
        return None
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(tr.get("lr", 2.5e-4)),
            total_steps=max(1, int(total_steps)),
            pct_start=float(tr.get("scheduler_pct_start", 0.01)),
            cycle_momentum=False,
        )
    raise ValueError(f"Unsupported Training.scheduler: {name}")


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 2.5e-4)), weight_decay=float(tr.get("weight_decay", 1e-5)))
    graph_cfg = config.get("Graph", {})
    max_steps = int(tr.get("max_steps", 0))
    estimated_total_steps = max_steps if max_steps > 0 else max(1, int(tr.get("epochs", 1)) * len(loader))
    scheduler = _make_scheduler(optimizer, tr, total_steps=estimated_total_steps)
    resume_checkpoint = tr.get("resume_checkpoint")
    step = 0
    best = float("inf")
    if resume_checkpoint:
        payload = load_checkpoint(
            str(resume_checkpoint),
            _unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            strict=True,
        )
        step = int(payload.get("step", 0))
        metrics = payload.get("metrics") or {}
        if "loss" in metrics:
            best = float(metrics["loss"])
    loss_keys = set(GraphLossWeights.__dataclass_fields__)
    loss_cfg = config.get("Loss", {})
    weights = GraphLossWeights(**{k: float(v) for k, v in loss_cfg.items() if k in loss_keys})

    output_dir = Path(tr.get("output_dir", "outputs/pano_droid_graph_train"))
    ckpt_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "visualizations"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    _barrier(ddp)
    save_every = max(1, int(tr.get("save_every", 100)))
    vis_cfg = config.get("Visualization", {})
    vis_enabled = bool(vis_cfg.get("enabled", True))
    vis_every = max(1, int(vis_cfg.get("every", save_every)))
    wb_cfg = config.get("WeightsAndBiases", {})
    wandb_run = _init_wandb(config, output_dir, enabled_on_rank=is_main)
    wandb_log_every = max(1, int(wb_cfg.get("log_every", 10)))
    start = time.time()
    last_metrics: dict[str, float] = {}
    last_edges: list[tuple[int, int]] = []

    try:
        for epoch in range(int(tr.get("epochs", 1))):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                images = batch["images"].to(device)
                batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                edges = _make_graph_edges(batch, graph_cfg, step=step)
                last_edges = list(edges)
                optimizer.zero_grad(set_to_none=True)
                init_poses = None
                init_inv = None
                restart_prob = float(tr.get("restart_prob", 0.2))
                restart_count = 0
                metrics = {}
                pred = None
                while True:
                    pred = _forward_graph(
                        model,
                        images,
                        edges=edges,
                        num_updates=int(tr.get("iters", config.get("Model", {}).get("update_iters", 1))),
                        poses_c2w=batch_device["poses_c2w"],
                        init_poses_c2w=init_poses,
                        init_inverse_depth=init_inv,
                        ba_iters_per_update=int(graph_cfg.get("ba_iters_per_update", 2)),
                        fixed_frames=int(graph_cfg.get("fixed_frames", 2)),
                        ba_sample_stride=int(graph_cfg.get("ba_sample_stride", 1)),
                    )
                    loss, metrics = graph_supervised_loss(
                        batch_device,
                        pred,
                        weights=weights,
                        sample_height=graph_cfg.get("loss_sample_height"),
                        sample_width=graph_cfg.get("loss_sample_width"),
                        gamma=float(graph_cfg.get("loss_gamma", 0.9)),
                    )
                    loss.backward()
                    restart_count += 1
                    init_poses = pred["refined_poses_c2w"].detach()
                    init_inv = pred["refined_inverse_depth"].detach()
                    r = torch.rand((), generator=torch.Generator().manual_seed(step * 104729 + restart_count))
                    if float(r) >= restart_prob:
                        break
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 2.5)))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                last_metrics = _reduce_metrics(metrics, ddp)
                last_metrics["restart_count"] = float(restart_count)
                if is_main and (step == 1 or step % wandb_log_every == 0):
                    _wandb_log_metrics(wandb_run, last_metrics, step=step)
                if is_main and step % save_every == 0:
                    save_checkpoint(
                        str(ckpt_dir / "latest.pt"),
                        model=_unwrap_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
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
                        scheduler=scheduler,
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
                            poses_c2w=batch_device["poses_c2w"][:1],
                            ba_iters_per_update=int(graph_cfg.get("ba_iters_per_update", 2)),
                            fixed_frames=int(graph_cfg.get("fixed_frames", 2)),
                            ba_sample_stride=int(graph_cfg.get("ba_sample_stride", 1)),
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
                scheduler=scheduler,
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
            "edges": last_edges,
            "rank": int(ddp["rank"]),
            "world_size": int(ddp["world_size"]),
        }
    return {
        "steps": step,
        "best_loss": best,
        "last_metrics": last_metrics,
        "checkpoint": str(ckpt_dir / "latest.pt"),
        "elapsed_sec": time.time() - start,
        "edges": last_edges,
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
