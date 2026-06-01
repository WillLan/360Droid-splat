"""Train PanoDROID with DROID-style multi-frame graph supervision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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
from .spherical_ba import se3_exp
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
            "edge_pose_source": "init",
            "temporal_radius": 2,
            "bidirectional": True,
            "max_edges_per_step": 24,
            "ba_iters_per_update": 2,
            "ba_sample_stride": 1,
            "fixed_frames": 2,
            "loss_gamma": 0.9,
            "fullres_loss_sample_stride": 4,
            "init_mode": "droid_gt_anchor",
            "init_noise_prob": 0.2,
            "init_identity_prob": 0.05,
            "init_noise_std": 0.03,
        },
        "Model": {
            "profile": "tiny",
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
            "find_unused_parameters": False,
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


def _freeze_legacy_pairwise(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("pose_head.") or name.startswith("damping_head."):
            param.requires_grad_(False)


def _grad_diagnostics(model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    base = _unwrap_model(model)
    out: dict[str, torch.Tensor] = {}
    unused = 0
    trainable = 0
    for _, param in base.named_parameters():
        if not param.requires_grad:
            continue
        trainable += 1
        if param.grad is None:
            unused += 1
    out["grad_unused_params"] = torch.tensor(float(unused), device=device)
    out["grad_trainable_params"] = torch.tensor(float(trainable), device=device)
    for name in ("fnet", "cnet", "update_block", "delta_head", "weight_head", "graph_agg"):
        module = getattr(base, name, None)
        if module is None:
            continue
        total = torch.tensor(0.0, device=device)
        for param in module.parameters():
            if param.grad is not None:
                total = total + param.grad.detach().float().pow(2).sum()
        out[f"grad_{name}"] = total.sqrt()
    return out


def _forward_graph(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    edges: list[tuple[int, int]],
    num_updates: int,
    poses_c2w: torch.Tensor | None,
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


def _droid_anchor_initial_poses(poses_c2w: torch.Tensor, fixed_frames: int) -> torch.Tensor:
    init = poses_c2w.clone()
    frames = int(init.shape[1])
    fixed = max(1, min(int(fixed_frames), frames))
    anchor = fixed - 1
    if fixed < frames:
        init[:, fixed:] = init[:, anchor : anchor + 1].expand(-1, frames - fixed, -1, -1)
    return init


def _make_initial_poses(
    poses_c2w: torch.Tensor,
    graph_cfg: dict,
    *,
    step: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    mode = str(graph_cfg.get("init_mode", "droid_gt_anchor")).lower()
    fixed_frames = int(graph_cfg.get("fixed_frames", 2))
    r = torch.rand((), generator=torch.Generator().manual_seed(int(step) + 6151)).item()
    identity_prob = float(graph_cfg.get("init_identity_prob", 0.05))
    noise_prob = float(graph_cfg.get("init_noise_prob", 0.2))
    if mode in ("mixed", "droid_mixed"):
        if r < identity_prob:
            mode = "identity_chain"
        elif r < identity_prob + noise_prob:
            mode = "noisy_gt_anchor"
        else:
            mode = "droid_gt_anchor"
    elif mode == "droid_gt_anchor" and r < identity_prob:
        mode = "identity_chain"
    elif mode == "droid_gt_anchor" and r < identity_prob + noise_prob:
        mode = "noisy_gt_anchor"

    if mode in ("droid_gt_anchor", "gt_anchor"):
        return poses_c2w, None, mode

    B, N = poses_c2w.shape[:2]
    if mode in ("identity", "identity_chain"):
        eye = torch.eye(4, device=poses_c2w.device, dtype=poses_c2w.dtype)
        init = eye.view(1, 1, 4, 4).expand(B, N, -1, -1).clone()
        return None, init, mode

    init = _droid_anchor_initial_poses(poses_c2w, fixed_frames)
    if mode in ("noisy", "noisy_gt", "noisy_gt_anchor"):
        std = float(graph_cfg.get("init_noise_std", 0.03))
        noise = torch.randn(B, N, 6, device=poses_c2w.device, dtype=poses_c2w.dtype) * std
        noise[:, : max(1, min(fixed_frames, N))] = 0.0
        init = se3_exp(noise) @ init
        return None, init, mode

    if mode in ("constant_velocity", "constant_velocity_anchor"):
        fixed = max(1, min(fixed_frames, N))
        if fixed >= 2 and fixed < N:
            velocity = poses_c2w[:, fixed - 1, :3, 3] - poses_c2w[:, fixed - 2, :3, 3]
            for k in range(fixed, N):
                init[:, k] = init[:, fixed - 1]
                init[:, k, :3, 3] = init[:, fixed - 1, :3, 3] + velocity * float(k - fixed + 1)
        return None, init, mode

    raise ValueError(f"Unsupported Graph.init_mode: {graph_cfg.get('init_mode')}")


def _make_graph_edges(
    batch: dict,
    graph_cfg: dict,
    *,
    step: int,
    poses_for_graph: torch.Tensor | None = None,
    depths_for_graph: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    n_frames = int(batch["images"].shape[1])
    radius = int(graph_cfg.get("temporal_radius", 2))
    bidirectional = bool(graph_cfg.get("bidirectional", True))
    max_edges = int(graph_cfg.get("max_edges_per_step", 0))
    strategy = str(graph_cfg.get("edge_strategy", "mixed")).lower()
    use_proximity = strategy == "proximity"
    if strategy == "mixed":
        use_proximity = bool(torch.rand((), generator=torch.Generator().manual_seed(int(step) + 1729)).item() < 0.5)
    if use_proximity and (poses_for_graph is not None or "poses_c2w" in batch):
        graph_poses = poses_for_graph if poses_for_graph is not None else batch["poses_c2w"]
        graph_depths = depths_for_graph if depths_for_graph is not None else batch.get("depths")
        edges = build_proximity_edges(
            graph_poses,
            graph_depths,
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


def _batch_data_stats(batch: dict) -> dict[str, float]:
    images = batch["images"].detach().float()
    depths = batch["depths"].detach().float()
    poses = batch["poses_c2w"].detach().float()
    valid_depth = depths > 1e-6
    if poses.shape[1] > 1:
        trans = poses[:, 1:, :3, 3] - poses[:, :-1, :3, 3]
        pose_step = trans.norm(dim=-1).mean()
    else:
        pose_step = torch.tensor(0.0)
    valid_values = depths[valid_depth]
    return {
        "image_mean": float(images.mean().cpu()),
        "image_std": float(images.std().cpu()),
        "depth_valid_ratio": float(valid_depth.float().mean().cpu()),
        "depth_mean_valid": float(valid_values.mean().cpu()) if valid_values.numel() else 0.0,
        "depth_max_valid": float(valid_values.max().cpu()) if valid_values.numel() else 0.0,
        "pose_translation_step_mean": float(pose_step.cpu()),
        "height": float(images.shape[-2]),
        "width": float(images.shape[-1]),
    }


def _git_commit() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return commit or None


def train_graph(config: dict) -> dict:
    ddp = _setup_distributed(config)
    device = ddp["device"]
    is_main = _is_main_process(ddp)
    impl = config.setdefault("Implementation", {})
    impl.setdefault("frontend", "PanoFactorGraph")
    impl.setdefault("ba", "SphericalDenseBA")
    impl.setdefault("ba_version", "pytorch_schur_ba_v2")
    impl.setdefault("residual_mode", "erp_wrapped_pixel")
    impl.setdefault("git_commit", _git_commit())
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
    if bool(tr.get("freeze_legacy_pairwise", True)):
        _freeze_legacy_pairwise(model)
    if ddp["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[int(ddp["local_rank"])] if device.type == "cuda" else None,
            output_device=int(ddp["local_rank"]) if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("Distributed", {}).get("find_unused_parameters", False)),
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
    data_stats_written = False

    try:
        for epoch in range(int(tr.get("epochs", 1))):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                images = batch["images"].to(device)
                batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                if is_main and not data_stats_written:
                    stats = _batch_data_stats(batch_device)
                    config["DataStats"] = stats
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_dir / "data_stats.json", "w", encoding="utf-8") as f:
                        json.dump(stats, f, indent=2)
                    data_stats_written = True
                poses_arg, init_poses, init_mode = _make_initial_poses(
                    batch_device["poses_c2w"],
                    graph_cfg,
                    step=step,
                )
                graph_pose_source = str(graph_cfg.get("edge_pose_source", "init")).lower()
                if graph_pose_source in ("init", "initial", "warm_start"):
                    poses_for_edges = init_poses if init_poses is not None else poses_arg
                else:
                    poses_for_edges = batch_device["poses_c2w"]
                edges = _make_graph_edges(
                    batch_device,
                    graph_cfg,
                    step=step,
                    poses_for_graph=poses_for_edges,
                    depths_for_graph=batch_device.get("depths"),
                )
                last_edges = list(edges)
                optimizer.zero_grad(set_to_none=True)
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
                        poses_c2w=poses_arg,
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
                        fullres_sample_stride=int(graph_cfg.get("fullres_loss_sample_stride", 4)),
                        gamma=float(graph_cfg.get("loss_gamma", 0.9)),
                    )
                    loss.backward()
                    restart_count += 1
                    poses_arg = None
                    init_poses = pred["refined_poses_c2w"].detach()
                    init_inv = pred["refined_inverse_depth"].detach()
                    r = torch.rand((), generator=torch.Generator().manual_seed(step * 104729 + restart_count))
                    if float(r) >= restart_prob:
                        break
                metrics.update(_grad_diagnostics(model, device))
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 2.5)))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                last_metrics = _reduce_metrics(metrics, ddp)
                last_metrics["restart_count"] = float(restart_count)
                last_metrics["init_mode_code"] = float(
                    {
                        "droid_gt_anchor": 0,
                        "gt_anchor": 0,
                        "noisy_gt_anchor": 1,
                        "noisy_gt": 1,
                        "noisy": 1,
                        "identity_chain": 2,
                        "identity": 2,
                        "constant_velocity": 3,
                        "constant_velocity_anchor": 3,
                    }.get(init_mode, 9)
                )
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
