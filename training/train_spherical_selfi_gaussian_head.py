"""Standalone Stage 2 Selfi-style per-pixel Gaussian-head training.

PanoVGGT and the Stage 1 adapter are always frozen.  This entry point is
deliberately not connected to the SLAM frontend or backend dispatch.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from backend.pano_gs.adapter import PFGS360Renderer, PanoRenderCamera
from data.stage2_source_reconstruction_dataset import (
    Stage2SourceReconstructionDataset,
    SyntheticStage2SourceDataset,
    stage2_source_collate,
)
from losses.spherical_gaussian_render_loss import (
    Stage2GaussianLossWeights,
    spherical_dssim,
    spherical_pseudo_geometry_consistency_loss,
    spherical_psnr,
    spherical_weighted_l1,
    stage2_gaussian_render_loss,
)
from models.panovggt_feature_wrapper import build_frozen_panovggt_wrapper
from models.per_pixel_gaussian_observation import PerPixelGaussianObservation
from models.spherical_selfi_dpt_adapter import (
    LoadedSphericalSelfiAdapter,
    SphericalSelfiDPTAdapter,
    load_spherical_selfi_adapter_checkpoint,
)
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead, erp_bilinear_resize
from tools.visualize_stage2_gaussians import save_stage2_render_panel


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def default_config() -> dict[str, Any]:
    return {
        "stage2": {"enabled": False},
        "image": {"height": 64, "width": 128, "head_height": 16, "head_width": 32},
        "panovggt": {
            "synthetic": True,
            "stage_hooks": ["stage1", "stage2", "stage3", "stage4"],
            "feature_keys": [None, None, None, None],
            "token_hw": [None, None, None, None],
            "token_start_idx": [None, None, None, None],
            "in_channels": [8, 16, 24, 32],
            "use_no_grad": True,
            "pose_convention": "c2w",
            "depth_convention": "euclidean_ray_depth",
        },
        "adapter_checkpoint": {"path": None, "sha256": None, "strict": True},
        "synthetic_adapter": {"hidden_dim": 16},
        "head": {
            "feature_dim": 24,
            "channels": [32, 64, 128, 256],
            "mlp_hidden_dim": 64,
            "rgb_sh_degree": 2,
            "density_sh_degree": 1,
            "depth_residual_ratio": 0.25,
            "initial_opacity": 0.10,
            "min_depth": 1.0e-4,
            "min_scale": 1.0e-5,
            "max_scale_ratio": 0.25,
            "latitude_cos_min": 1.0e-3,
            "log_scale_clamp": 5.0,
            "render_prune_fraction": 0.30,
            "gradient_checkpointing": False,
        },
        "dataset": {
            "synthetic": True,
            "manifest": None,
            "domains": None,
            "views_per_sample": 3,
            "stride_min": 2,
            "stride_max": 6,
            "max_train_samples": None,
            "max_val_samples": None,
        },
        "renderer": {"backend": "gsplat360", "extra_gsplat360_roots": []},
        "loss": {
            "rgb_weight": 1.0,
            "depth_residual_weight": 1.0e-3,
            "dssim_weight": 0.0,
            "rendered_depth_weight": 0.0,
            "geometry_weight": 0.0,
            "geometry_num_queries": 512,
            "geometry_min_depth": 0.05,
            "geometry_max_depth": 100.0,
            "geometry_visibility_rel_thresh": 0.05,
        },
        "train": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "ddp": False,
            "ddp_backend": "nccl",
            "num_workers": 0,
            "feature_device": "auto",
            "train_device": "auto",
            "lr": 2.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_steps": 1,
            "max_steps": 2,
            "amp": False,
            "joint_multiview_backward": True,
            "diagnostics_interval": 200,
            "grad_clip": 1.0,
            "log_interval": 1,
            "val_interval": 1,
            "max_val_batches": 1,
            "save_interval": 1,
            "output_dir": "outputs/stage2_spherical_selfi_gaussian_head",
            "resume": None,
            "seed": 1234,
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "mode": "online",
            "log_every": 50,
        },
        "Visualization": {"enabled": False, "interval": 200, "save_dir": "visualizations"},
        "Validation": {"copy_diagnostics": True, "lpips": False},
        "Training": {
            "pfgs360_packed": False,
            "pfgs360_render_mode": "RGB+ED",
            "pfgs360_near_plane": 0.01,
            "pfgs360_far_plane": 1.0e5,
            "pfgs360_rasterize_mode": "antialiased",
            "pfgs360_absgrad": True,
        },
    }


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        return _deep_merge(config, yaml.safe_load(handle) or {})


def _resolve_device(value: str, *, fallback_index: int = 0) -> torch.device:
    if str(value).lower() != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda", min(int(fallback_index), torch.cuda.device_count() - 1))
    return torch.device("cpu")


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _initialize_distributed(train_cfg: dict[str, Any]) -> DistributedContext:
    """Initialize one-process-per-GPU DDP from ``torchrun`` environment variables."""

    enabled = bool(train_cfg.get("ddp", False))
    if not enabled:
        return DistributedContext()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 DDP requires CUDA and the NCCL backend.")
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError(f"Stage 2 DDP must be launched with torchrun; missing environment variables: {missing}.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(f"Stage 2 DDP requires WORLD_SIZE >= 2, got {world_size}.")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible CUDA devices."
        )
    torch.cuda.set_device(local_rank)
    backend = str(train_cfg.get("ddp_backend", "nccl"))
    if backend.lower() != "nccl":
        raise ValueError(f"Stage 2 CUDA DDP requires ddp_backend='nccl', got {backend!r}.")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def _effective_batch_size(train_cfg: dict[str, Any], distributed: DistributedContext) -> int:
    micro_batch = int(train_cfg.get("batch_size", 1))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    if micro_batch <= 0:
        raise ValueError("train.batch_size must be positive.")
    if accumulation <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive.")
    return micro_batch * accumulation * distributed.world_size


def _unwrap_head(head: nn.Module) -> SphericalSelfiGaussianHead:
    module = head.module if isinstance(head, DistributedDataParallel) else head
    if not isinstance(module, SphericalSelfiGaussianHead):
        raise TypeError(f"Expected SphericalSelfiGaussianHead, got {type(module).__name__}.")
    return module


def _distributed_mean(values: dict[str, float], distributed: DistributedContext, device: torch.device) -> dict[str, float]:
    if not distributed.enabled or not values:
        return values
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(distributed.world_size)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}


def _distributed_max(value: float, distributed: DistributedContext, device: torch.device) -> float:
    if not distributed.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _build_synthetic_frozen_stack(
    config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, SphericalSelfiDPTAdapter, str]:
    # Reuse the Stage 1 synthetic PanoVGGT contract rather than adding fake
    # features to the real inference path.
    from training.train_spherical_selfi_adapter import build_adapter, build_panovggt_wrapper

    stage1_config = {
        "image": {
            "height": int(config["image"]["height"]),
            "width": int(config["image"]["width"]),
        },
        "panovggt": dict(config["panovggt"]),
        "adapter": {
            "hidden_dim": int(config.get("synthetic_adapter", {}).get("hidden_dim", 16)),
            "out_dim": 24,
            "use_circular_padding": True,
            "norm_output": True,
        },
    }
    wrapper = _freeze(build_panovggt_wrapper(stage1_config, device=device))
    adapter = _freeze(build_adapter(stage1_config, device=device))
    return wrapper, adapter, "synthetic-no-checkpoint"


def build_frozen_feature_stack(
    config: dict[str, Any], *, device: torch.device
) -> tuple[nn.Module, SphericalSelfiDPTAdapter, str, dict[str, Any]]:
    pano_cfg = config.get("panovggt", {})
    if bool(pano_cfg.get("synthetic", False)):
        wrapper, adapter, sha = _build_synthetic_frozen_stack(config, device)
        return wrapper, adapter, sha, {"synthetic": True}
    wrapper = _freeze(build_frozen_panovggt_wrapper(pano_cfg, device=device))
    checkpoint_cfg = config.get("adapter_checkpoint", {})
    path = checkpoint_cfg.get("path")
    if not path:
        raise ValueError("adapter_checkpoint.path is required for non-synthetic Stage 2 training.")
    loaded: LoadedSphericalSelfiAdapter = load_spherical_selfi_adapter_checkpoint(
        path,
        expected_sha256=checkpoint_cfg.get("sha256"),
        expected_image_size=(int(config["image"]["height"]), int(config["image"]["width"])),
        expected_stage_hooks=list(pano_cfg.get("stage_hooks", [])),
        expected_token_hw=list(pano_cfg.get("token_hw", [])),
        expected_token_start_idx=list(pano_cfg.get("token_start_idx", [])),
        expected_pose_convention=str(pano_cfg.get("pose_convention", "c2w")),
        expected_depth_convention=str(pano_cfg.get("depth_convention", "euclidean_ray_depth")),
        device=device,
    )
    return wrapper, loaded.module, loaded.sha256, loaded.metadata


def build_head(config: dict[str, Any], *, device: torch.device) -> SphericalSelfiGaussianHead:
    kwargs = dict(config.get("head", {}))
    return SphericalSelfiGaussianHead(**kwargs, renderer_config=config).to(device)


def build_dataset(config: dict[str, Any], *, split: str):
    dataset_cfg = config.get("dataset", {})
    views = int(dataset_cfg.get("views_per_sample", 4))
    if bool(dataset_cfg.get("synthetic", False)):
        maximum = dataset_cfg.get("max_train_samples" if split == "train" else "max_val_samples")
        return SyntheticStage2SourceDataset(
            length=int(maximum or 2),
            views_per_sample=views,
            height=int(config["image"]["height"]),
            width=int(config["image"]["width"]),
        )
    maximum = dataset_cfg.get("max_train_samples" if split == "train" else "max_val_samples")
    return Stage2SourceReconstructionDataset(
        dataset_cfg.get("manifest"),
        split=split,
        domains=dataset_cfg.get("domains"),
        views_per_sample=views,
        stride_min=int(dataset_cfg.get("stride_min", 2)),
        stride_max=int(dataset_cfg.get("stride_max", 6)),
        image_height=int(config["image"]["height"]),
        image_width=int(config["image"]["width"]),
        seed=int(config.get("train", {}).get("seed", 1234)),
        max_samples=maximum,
    )


def build_renderer(config: dict[str, Any]):
    renderer_cfg = config.get("renderer", {})
    backend = str(renderer_cfg.get("backend", "gsplat360")).lower()
    if backend != "gsplat360":
        raise ValueError(f"Unsupported Stage 2 renderer backend: {backend!r}.")
    return PFGS360Renderer(
        config=config,
        extra_gsplat360_roots=list(renderer_cfg.get("extra_gsplat360_roots", [])),
        allow_fallback=False,
    )


def _canonical_depth(depth: torch.Tensor, *, batch: int, views: int) -> torch.Tensor:
    if depth.ndim == 5 and int(depth.shape[-1]) == 1:
        depth = depth.permute(0, 1, 4, 2, 3)
    elif depth.ndim == 4 and tuple(depth.shape[:2]) == (batch, views):
        depth = depth.unsqueeze(2)
    if depth.ndim != 5 or tuple(depth.shape[:3]) != (batch, views, 1):
        raise ValueError(f"PanoVGGT depth must normalize to BxSx1xHxW, got {tuple(depth.shape)}.")
    return depth


def extract_frozen_inputs(
    wrapper: nn.Module,
    adapter: SphericalSelfiDPTAdapter,
    images: torch.Tensor,
    *,
    feature_device: torch.device,
    train_device: torch.device,
    head_size: tuple[int, int],
    feature_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-resolution frozen stack then explicitly resize Stage 2 inputs."""

    wrapper.eval()
    adapter.eval()
    full_images = images.to(feature_device, non_blocking=True).float()
    with torch.no_grad(), torch.amp.autocast(
        device_type=feature_device.type,
        dtype=torch.bfloat16,
        enabled=bool(feature_amp) and feature_device.type == "cuda",
    ):
        pano_output = wrapper(full_images)
        if pano_output.pose_convention != "c2w" or pano_output.depth_convention != "euclidean_ray_depth":
            raise ValueError(
                "Stage 2 requires c2w poses and Euclidean ERP ray depth; got "
                f"{pano_output.pose_convention!r}/{pano_output.depth_convention!r}."
            )
        if pano_output.init_depth is None or pano_output.init_poses is None:
            raise RuntimeError("PanoVGGT must provide initial depth and poses for Stage 2.")
        dense = adapter(pano_output.stage_features)
        batch, views = int(full_images.shape[0]), int(full_images.shape[1])
        depth = _canonical_depth(pano_output.init_depth, batch=batch, views=views)
        poses = pano_output.init_poses
        if tuple(poses.shape) != (batch, views, 4, 4):
            raise ValueError(f"PanoVGGT poses must be BxSx4x4, got {tuple(poses.shape)}.")
        height, width = (int(value) for value in head_size)
        if tuple(dense.shape[-2:]) != (height, width):
            dense = erp_bilinear_resize(
                dense.reshape(batch * views, int(dense.shape[2]), *dense.shape[-2:]),
                (height, width),
            ).reshape(batch, views, int(dense.shape[2]), height, width)
        rgb = erp_bilinear_resize(
            full_images.reshape(batch * views, 3, *full_images.shape[-2:]),
            (height, width),
        ).reshape(batch, views, 3, height, width)
        depth = erp_bilinear_resize(
            depth.reshape(batch * views, 1, *depth.shape[-2:]),
            (height, width),
        ).reshape(batch, views, 1, height, width)
    return (
        dense.detach().to(train_device, non_blocking=True),
        rgb.detach().to(train_device, non_blocking=True),
        depth.detach().to(train_device, non_blocking=True),
        poses.detach().to(train_device, non_blocking=True),
    )


def render_observation_views(
    renderer: Any,
    observation: PerPixelGaussianObservation,
    *,
    batch_index: int,
    source_mode: str = "all",
) -> list[dict[str, Any]]:
    return [
        render_observation_target(
            renderer,
            observation,
            batch_index=batch_index,
            target_view=target_view,
            source_mode=source_mode,
        )
        for target_view in range(observation.num_source_views)
    ]


def render_observation_target(
    renderer: Any,
    observation: PerPixelGaussianObservation,
    *,
    batch_index: int,
    target_view: int,
    source_mode: str = "all",
) -> dict[str, Any]:
    height, width = observation.image_size
    views = observation.num_source_views
    target = int(target_view)
    if target < 0 or target >= views:
        raise IndexError(f"target_view {target} is outside [0, {views}).")
    if source_mode == "all":
        source_indices: Iterable[int] = range(views)
    elif source_mode == "self":
        source_indices = [target]
    elif source_mode == "leave_one_out":
        source_indices = [index for index in range(views) if index != target]
    else:
        raise ValueError(f"Unknown source_mode: {source_mode!r}.")
    camera = PanoRenderCamera(
        image_height=height,
        image_width=width,
        c2w=observation.poses_c2w[int(batch_index), target].float(),
    )
    materialize_start = time.perf_counter()
    explicit = observation.materialize_batch(
        camera,
        batch_index=int(batch_index),
        source_indices=source_indices,
    )
    materialize_sec = float(time.perf_counter() - materialize_start)
    package = dict(renderer.render(camera, explicit))
    package["stage2_materialize_sec"] = materialize_sec
    package["stage2_materialized_gaussians"] = float(explicit.xyz.shape[0])
    return package


def _copy_metrics(
    packages: list[dict[str, Any]],
    targets: torch.Tensor,
    *,
    lpips_model: nn.Module | None = None,
) -> dict[str, float]:
    l1, psnr, ssim = [], [], []
    for view, package in enumerate(packages):
        rendered = package["render"]
        target = targets[view].to(rendered)
        l1.append(spherical_weighted_l1(rendered, target))
        psnr.append(spherical_psnr(rendered, target))
        ssim.append(1.0 - 2.0 * spherical_dssim(rendered, target))
    result = {
        "l1": float(torch.stack(l1).mean().detach().cpu()),
        "psnr": float(torch.stack(psnr).mean().detach().cpu()),
        "ssim": float(torch.stack(ssim).mean().detach().cpu()),
    }
    if lpips_model is not None:
        rendered_batch = torch.stack([package["render"] for package in packages])
        target_batch = targets.to(rendered_batch)
        result["lpips"] = float(
            lpips_model(rendered_batch * 2.0 - 1.0, target_batch * 2.0 - 1.0).mean().detach().cpu()
        )
    return result


@torch.no_grad()
def _observation_diagnostics(observation: PerPixelGaussianObservation, *, batch_index: int) -> dict[str, float]:
    batch = int(batch_index)
    valid = observation.valid_mask[batch].bool()
    residual = (observation.depth_residual[batch] / observation.initial_depth[batch].clamp_min(1.0e-4))[valid]
    scales = observation.scales()[batch]
    scale = scales[valid.expand_as(scales)]
    confidence = observation.confidence[batch][valid]
    result: dict[str, float] = {"canonical_gaussians": float(valid.sum().detach().cpu())}
    for name, tensor in (("relative_depth_residual", residual), ("scale", scale), ("confidence", confidence)):
        if tensor.numel() == 0:
            continue
        tensor = tensor.detach().float()
        for quantile in (0.1, 0.5, 0.9):
            result[f"{name}_p{int(quantile * 100)}"] = float(tensor.quantile(quantile).cpu())
    # Confidence is density SH evaluated along each source ray, so it equals
    # the unpruned opacity when that source camera renders its own observation.
    # Avoid four full target-conditioned materializations for this diagnostic.
    for source in range(observation.num_source_views):
        source_valid = valid[source]
        source_confidence = observation.confidence[batch, source][source_valid]
        result[f"source_{source}_mean_opacity"] = (
            float(source_confidence.mean().detach().cpu()) if source_confidence.numel() else 0.0
        )
    return result


def _is_interval_step(step: int, interval: int) -> bool:
    if int(interval) <= 0:
        raise ValueError("interval must be positive.")
    return int(step) > 0 and int(step) % int(interval) == 0


def _loss_weights(config: dict[str, Any]) -> Stage2GaussianLossWeights:
    cfg = config.get("loss", {})
    return Stage2GaussianLossWeights(
        rgb=float(cfg.get("rgb_weight", 1.0)),
        depth_residual=float(cfg.get("depth_residual_weight", 1.0e-3)),
        dssim=float(cfg.get("dssim_weight", 0.0)),
        rendered_depth=float(cfg.get("rendered_depth_weight", 0.0)),
        geometry=float(cfg.get("geometry_weight", 0.0)),
    )


def _scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int):
    warmup = max(0, int(warmup_steps))
    total = max(1, int(max_steps))

    def multiplier(step: int) -> float:
        current = int(step)
        if warmup > 0 and current < warmup:
            return float(current + 1) / float(warmup)
        progress = float(current - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_stage2_checkpoint(
    path: str | Path,
    *,
    head: nn.Module,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    metrics: dict[str, float],
    adapter_sha256: str,
    best_val_psnr: float | None,
) -> Path:
    head_module = _unwrap_head(head)
    for name, parameter in head_module.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"Non-finite Stage 2 parameter {name!r}; refusing to checkpoint.")
    non_finite_metrics = [key for key, value in metrics.items() if not math.isfinite(float(value))]
    if non_finite_metrics:
        raise RuntimeError(f"Non-finite Stage 2 metrics {non_finite_metrics}; refusing to checkpoint.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spherical_selfi_gaussian_head_v1",
            "head": head_module.state_dict(),
            "head_config": head_module.head_config(),
            "training_config": config,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(step),
            "metrics": dict(metrics),
            "adapter_sha256": str(adapter_sha256),
            "panovggt_config": dict(config.get("panovggt", {})),
            "best_val_psnr": best_val_psnr,
        },
        output,
    )
    return output


def load_stage2_checkpoint(
    path: str | Path,
    *,
    head: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_adapter_sha256: str | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[int, dict[str, float], float | None]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "spherical_selfi_gaussian_head_v1":
        raise ValueError(f"Unsupported Stage 2 checkpoint: {path}.")
    if expected_adapter_sha256 is not None and payload.get("adapter_sha256") != expected_adapter_sha256:
        raise ValueError("Stage 2 checkpoint adapter SHA256 does not match the frozen adapter.")
    _unwrap_head(head).load_state_dict(payload["head"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("global_step", 0)), dict(payload.get("metrics", {})), payload.get("best_val_psnr")


def _init_wandb(config: dict[str, Any], output_dir: Path):
    cfg = config.get("WeightsAndBiases", {})
    if not bool(cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WeightsAndBiases.enabled=true requires wandb.") from exc
    return wandb.init(
        project=str(cfg.get("project", "360Droid-splat")),
        entity=cfg.get("entity"),
        name=cfg.get("run_name"),
        mode=str(cfg.get("mode", "online")),
        dir=str(output_dir),
        config=config,
        tags=cfg.get("tags"),
    )


def _optional_lpips(config: dict[str, Any], device: torch.device) -> nn.Module | None:
    if not bool(config.get("Validation", {}).get("lpips", False)):
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Validation.lpips=true requires the optional lpips package.") from exc
    return _freeze(lpips.LPIPS(net="alex").to(device))


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_spherical_selfi_gaussian_head(config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("stage2", {}).get("enabled", False)):
        raise RuntimeError("Stage 2 is config-gated; set stage2.enabled=true to run this trainer.")
    train_cfg = config.get("train", {})
    distributed = _initialize_distributed(train_cfg)
    _seed_everything(int(train_cfg.get("seed", 1234)) + distributed.rank)
    if distributed.enabled:
        feature_device = torch.device("cuda", distributed.local_rank)
        train_device = feature_device
    else:
        feature_device = _resolve_device(str(train_cfg.get("feature_device", "auto")), fallback_index=0)
        train_device = _resolve_device(str(train_cfg.get("train_device", "auto")), fallback_index=1)
    if train_device.type != "cuda":
        raise RuntimeError("Stage 2 rendering and training require a CUDA train_device with gsplat360.")
    if train_device.index is not None:
        torch.cuda.set_device(train_device)
    # PyTorch 2.11 rejects reset_peak_memory_stats on a CUDA device whose
    # allocator has not been initialized yet.
    torch.empty(1, device=train_device)
    torch.cuda.reset_peak_memory_stats(train_device)
    wrapper, adapter, adapter_sha, adapter_metadata = build_frozen_feature_stack(config, device=feature_device)
    head_module = build_head(config, device=train_device)
    head: nn.Module
    if distributed.enabled:
        head = DistributedDataParallel(
            head_module,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
        )
    else:
        head = head_module
    renderer = build_renderer(config)
    lpips_model = _optional_lpips(config, train_device) if distributed.is_main else None
    train_dataset = build_dataset(config, split="train")
    try:
        val_dataset = build_dataset(config, split="val") if distributed.is_main else None
    except ValueError:
        val_dataset = None
    train_sampler = None
    if distributed.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=int(train_cfg.get("seed", 1234)),
            drop_last=False,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage2_source_collate,
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage2_source_collate,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 2.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(train_cfg.get("max_steps", 1))
    accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    effective_batch_size = _effective_batch_size(train_cfg, distributed)
    scheduler = _scheduler(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        max_steps=max_steps,
    )
    output_dir = Path(train_cfg.get("output_dir", "outputs/stage2_spherical_selfi_gaussian_head"))
    checkpoint_dir = output_dir / "checkpoints"
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.enabled:
        dist.barrier()
    wandb_run = _init_wandb(config, output_dir) if distributed.is_main else None
    step, latest_metrics, best_val_psnr = 0, {}, None
    if train_cfg.get("resume"):
        step, latest_metrics, best_val_psnr = load_stage2_checkpoint(
            train_cfg["resume"],
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_adapter_sha256=adapter_sha,
            map_location=train_device,
        )
    head_size = (int(config["image"]["head_height"]), int(config["image"]["head_width"]))
    amp_enabled = bool(train_cfg.get("amp", True)) and train_device.type == "cuda"
    weights = _loss_weights(config)
    vis_cfg = config.get("Visualization", {})
    first_val_metrics: dict[str, float] | None = None

    def make_inputs(raw_images: torch.Tensor):
        return extract_frozen_inputs(
            wrapper,
            adapter,
            raw_images,
            feature_device=feature_device,
            train_device=train_device,
            head_size=head_size,
            feature_amp=amp_enabled,
        )

    def predict_observation(
        dense: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        poses: torch.Tensor,
        frame_ids: torch.Tensor,
        *,
        predictor: nn.Module | None = None,
    ) -> PerPixelGaussianObservation:
        with torch.amp.autocast(device_type=train_device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            return (head if predictor is None else predictor)(
                dense,
                rgb,
                depth,
                poses,
                frame_ids=frame_ids.to(train_device),
            )

    def make_observation(raw_images: torch.Tensor, frame_ids: torch.Tensor) -> tuple[PerPixelGaussianObservation, torch.Tensor]:
        dense, rgb, depth, poses = make_inputs(raw_images)
        return predict_observation(dense, rgb, depth, poses, frame_ids, predictor=_unwrap_head(head)), rgb

    @torch.no_grad()
    def validate(current_step: int) -> dict[str, float]:
        nonlocal first_val_metrics
        if val_loader is None:
            return {}
        head.eval()
        sums: dict[str, float] = {}
        count = 0
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= int(train_cfg.get("max_val_batches", 1)):
                break
            observation, rgb = make_observation(batch["images"], batch["frame_ids"])
            for item in range(observation.batch_size):
                modes = ["all"]
                if bool(config.get("Validation", {}).get("copy_diagnostics", True)):
                    modes += ["self", "leave_one_out"]
                for mode in modes:
                    packages = render_observation_views(renderer, observation, batch_index=item, source_mode=mode)
                    metrics = _copy_metrics(packages, rgb[item], lpips_model=lpips_model)
                    prefix = {"all": "all_source", "self": "self_only", "leave_one_out": "leave_one_out"}[mode]
                    for key, value in metrics.items():
                        sums[f"{prefix}_{key}"] = sums.get(f"{prefix}_{key}", 0.0) + value
                diagnostics = _observation_diagnostics(observation, batch_index=item)
                for key, value in diagnostics.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
        head.train()
        metrics = {key: value / max(1, count) for key, value in sums.items()}
        if first_val_metrics is None:
            first_val_metrics = dict(metrics)
            metrics["copy_degeneracy_flag"] = 0.0
        else:
            all_delta = metrics.get("all_source_psnr", 0.0) - first_val_metrics.get("all_source_psnr", 0.0)
            loo_delta = metrics.get("leave_one_out_psnr", 0.0) - first_val_metrics.get("leave_one_out_psnr", 0.0)
            metrics["copy_degeneracy_flag"] = float(all_delta > 0.5 and loo_delta < 0.1)
        return metrics

    head.train()
    epoch = 0
    start_time = time.perf_counter()
    accumulated_microbatches = 0
    optimizer.zero_grad(set_to_none=True)
    total_loss_value = 0.0
    aggregate: dict[str, torch.Tensor] = {}
    first_package: dict[str, Any] | None = None
    first_rgb: torch.Tensor | None = None
    first_depth_residual: torch.Tensor | None = None
    first_confidence: torch.Tensor | None = None
    last_observation: PerPixelGaussianObservation | None = None
    if not bool(train_cfg.get("joint_multiview_backward", True)):
        raise ValueError(
            "Stage 2 requires train.joint_multiview_backward=true so one Head forward is shared by all "
            "source-view reconstruction renders."
        )
    while step < max_steps:
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in loader:
            dense, rgb, depth, poses = make_inputs(batch["images"])
            final_microbatch = accumulated_microbatches + 1 == accumulation_steps
            batch_size = int(rgb.shape[0])
            divisor = float(batch_size * accumulation_steps)
            sync_context = nullcontext()
            if isinstance(head, DistributedDataParallel) and not final_microbatch:
                sync_context = head.no_sync()
            with sync_context:
                # Predict the complete source window once. All target-view
                # renders branch from this shared observation and contribute
                # to one joint backward pass.
                observation = predict_observation(
                    dense,
                    rgb,
                    depth,
                    poses,
                    batch["frame_ids"],
                )
                scaled_loss = observation.depth_residual.sum() * 0.0
                for item in range(batch_size):
                    with torch.amp.autocast(
                        device_type=train_device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        packages = render_observation_views(
                            renderer,
                            observation,
                            batch_index=item,
                            source_mode="all",
                        )
                        if first_package is None:
                            package = packages[0]
                            first_package = {
                                "render": package["render"].detach(),
                                "stage2_materialize_sec": package.get("stage2_materialize_sec", 0.0),
                                "profile_renderer_rasterize_sec": package.get(
                                    "profile_renderer_rasterize_sec", 0.0
                                ),
                                "profile_renderer_total_sec": package.get(
                                    "profile_renderer_total_sec", 0.0
                                ),
                                "stage2_materialized_gaussians": package.get(
                                    "stage2_materialized_gaussians", 0.0
                                ),
                            }
                            first_rgb = rgb[item, 0].detach()
                            first_depth_residual = observation.depth_residual[item, 0].detach()
                            first_confidence = observation.confidence[item, 0].detach()
                        geometry = None
                        if float(weights.geometry) != 0.0:
                            loss_cfg = config.get("loss", {})
                            geometry = spherical_pseudo_geometry_consistency_loss(
                                observation,
                                batch_index=item,
                                num_query_per_pair=int(loss_cfg.get("geometry_num_queries", 512)),
                                min_depth=float(loss_cfg.get("geometry_min_depth", 0.05)),
                                max_depth=float(loss_cfg.get("geometry_max_depth", 100.0)),
                                visibility_rel_thresh=float(
                                    loss_cfg.get("geometry_visibility_rel_thresh", 0.05)
                                ),
                            )
                        terms = stage2_gaussian_render_loss(
                            packages,
                            rgb[item],
                            observation,
                            batch_index=item,
                            target_depths=observation.initial_depth[item],
                            geometry_loss=geometry,
                            weights=weights,
                        )
                        scaled_loss = scaled_loss + terms["loss"] / divisor
                    for key, value in terms.items():
                        aggregate[key] = (
                            aggregate.get(key, torch.zeros_like(value.detach())) + value.detach() / divisor
                        )
                if not bool(torch.isfinite(scaled_loss)):
                    raise RuntimeError(f"Non-finite Stage 2 loss at step {step + 1}.")
                scaled_loss.backward()
            last_observation = observation
            total_loss_value += float(scaled_loss.detach().float().cpu())
            accumulated_microbatches += 1
            if not final_microbatch:
                continue
            if last_observation is None:
                raise RuntimeError("Stage 2 batch produced no source target views.")
            for name, parameter in head.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"Non-finite Stage 2 gradient {name!r} at step {step + 1}.")
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"Non-finite Stage 2 gradient norm at step {step + 1}.")
            optimizer.step()
            scheduler.step()
            step += 1
            local_metrics = {key: float(value.detach().float().cpu()) for key, value in aggregate.items()}
            local_metrics["loss"] = total_loss_value
            if first_package:
                for key in (
                    "stage2_materialize_sec",
                    "profile_renderer_rasterize_sec",
                    "profile_renderer_total_sec",
                    "stage2_materialized_gaussians",
                ):
                    local_metrics[key] = float(first_package.get(key, 0.0))
            diagnostics_interval = int(train_cfg.get("diagnostics_interval", 200))
            if _is_interval_step(step, diagnostics_interval):
                local_metrics.update(_observation_diagnostics(last_observation, batch_index=0))
            local_metrics["grad_norm"] = float(grad_norm.detach().float().cpu())
            local_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            local_metrics["elapsed_sec"] = float(time.perf_counter() - start_time)
            latest_metrics = _distributed_mean(local_metrics, distributed, train_device)
            latest_metrics["effective_batch_size"] = float(effective_batch_size)
            if train_device.type == "cuda":
                rank_peak = float(torch.cuda.max_memory_allocated(train_device) / (1024**3))
                latest_metrics["peak_memory_gib"] = _distributed_max(rank_peak, distributed, train_device)
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
            total_loss_value = 0.0
            aggregate = {}

            if distributed.is_main and (step == 1 or step % int(train_cfg.get("log_interval", 50)) == 0):
                print(yaml.safe_dump({"step": step, "metrics": latest_metrics}, sort_keys=False).strip())
            if wandb_run is not None and (
                step == 1 or step % int(config["WeightsAndBiases"].get("log_every", 50)) == 0
            ):
                wandb_run.log({f"train/{key}": value for key, value in latest_metrics.items()}, step=step)
            if (
                distributed.is_main
                and bool(vis_cfg.get("enabled", False))
                and first_package is not None
                and first_rgb is not None
                and first_depth_residual is not None
                and first_confidence is not None
                and _is_interval_step(step, int(vis_cfg.get("interval", 200)))
            ):
                path = save_stage2_render_panel(
                    first_rgb,
                    first_package["render"],
                    first_depth_residual,
                    first_confidence,
                    output_dir / str(vis_cfg.get("save_dir", "visualizations")) / f"step_{step:06d}.png",
                    title=f"step {step}",
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({"diagnostics/stage2_gaussian_render": wandb.Image(str(path))}, step=step)
            first_package = None
            first_rgb = None
            first_depth_residual = None
            first_confidence = None
            last_observation = None

            should_validate = step % int(train_cfg.get("val_interval", 1000)) == 0 or step == max_steps
            if distributed.enabled and should_validate:
                dist.barrier()
            if distributed.is_main and val_loader is not None and should_validate:
                val_metrics = validate(step)
                if val_metrics:
                    print(yaml.safe_dump({"step": step, "validation": val_metrics}, sort_keys=False).strip())
                    if wandb_run is not None:
                        wandb_run.log({f"val/{key}": value for key, value in val_metrics.items()}, step=step)
                    if val_metrics.get("copy_degeneracy_flag", 0.0) > 0.5:
                        print(
                            "WARNING: all-source PSNR improved without meaningful leave-one-out improvement; "
                            "this run currently exhibits source-copy degeneration."
                        )
                    score = val_metrics.get("all_source_psnr")
                    if score is not None and (best_val_psnr is None or score > best_val_psnr):
                        best_val_psnr = float(score)
                        save_stage2_checkpoint(
                            checkpoint_dir / "best_val_psnr.pt",
                            head=head,
                            config=config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            metrics={**latest_metrics, **{f"val/{key}": value for key, value in val_metrics.items()}},
                            adapter_sha256=adapter_sha,
                            best_val_psnr=best_val_psnr,
                        )
            if distributed.enabled and should_validate:
                dist.barrier()
            should_save = step % int(train_cfg.get("save_interval", 1000)) == 0 or step == max_steps
            if distributed.is_main and should_save:
                save_stage2_checkpoint(
                    checkpoint_dir / "latest.pt",
                    head=head,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=latest_metrics,
                    adapter_sha256=adapter_sha,
                    best_val_psnr=best_val_psnr,
                )
            if distributed.enabled and should_save:
                dist.barrier()
            if step >= max_steps:
                break
        epoch += 1
    if distributed.is_main:
        save_stage2_checkpoint(
            checkpoint_dir / "latest.pt",
            head=head,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            metrics=latest_metrics,
            adapter_sha256=adapter_sha,
            best_val_psnr=best_val_psnr,
        )
    if distributed.enabled:
        dist.barrier()
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "step": step,
        "metrics": latest_metrics,
        "checkpoint": str(checkpoint_dir / "latest.pt"),
        "adapter_sha256": adapter_sha,
        "adapter_metadata": adapter_metadata,
        "feature_device": str(feature_device),
        "train_device": str(train_device),
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "effective_batch_size": effective_batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--head-height", type=int, default=None)
    parser.add_argument("--head-width", type=int, default=None)
    parser.add_argument("--views-per-sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config["train"]["resume"] = args.resume
    if args.max_steps is not None:
        config["train"]["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.manifest is not None:
        config["dataset"]["manifest"] = args.manifest
    if args.head_height is not None:
        config["image"]["head_height"] = int(args.head_height)
    if args.head_width is not None:
        config["image"]["head_width"] = int(args.head_width)
    if args.views_per_sample is not None:
        config["dataset"]["views_per_sample"] = int(args.views_per_sample)
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.ddp:
        config["train"]["ddp"] = True
    if args.feature_device is not None:
        config["train"]["feature_device"] = args.feature_device
    if args.train_device is not None:
        config["train"]["train_device"] = args.train_device
    if args.wandb_mode is not None:
        config["WeightsAndBiases"]["enabled"] = args.wandb_mode != "disabled"
        if args.wandb_mode != "disabled":
            config["WeightsAndBiases"]["mode"] = args.wandb_mode
    result = train_spherical_selfi_gaussian_head(config)
    if int(result.get("rank", 0)) == 0:
        print(yaml.safe_dump(result, sort_keys=False).strip())
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
