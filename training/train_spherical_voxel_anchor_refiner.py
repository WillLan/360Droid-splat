"""Train the config-gated simplified Stage-3 voxel-anchor refiner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from backend.pano_gs.adapter import PFGS360Renderer
from data.stage3_spherical_ba_refiner_dataset import stage3_collate
from losses.spherical_gaussian_render_loss import (
    spherical_dssim,
    spherical_psnr,
    spherical_weighted_l1,
)
from losses.spherical_voxel_anchor_refinement_loss import (
    VoxelAnchorLossWeights,
    spherical_voxel_anchor_loss,
)
from models.per_pixel_gaussian_observation import SH_C0, PerPixelGaussianObservation
from models.spherical_recurrent_gaussian_refiner import EncodedTargetReference
from models.spherical_selfi_stage3_ba import BlockSparseSphericalBA, Stage3MatchCache
from models.spherical_voxel_anchor_refiner import (
    VoxelAnchorConfig,
    VoxelAnchorObservation,
    VoxelAnchorRefinerOutput,
    VoxelAnchorRenderGroup,
    VoxelAnchorStage3Model,
    render_voxel_anchor_group,
    voxelize_per_pixel_gaussians,
)
from tools.visualize_stage3_refinement import save_stage3_snapshot_panel
from training.train_spherical_ba_recurrent_refiner import (
    _apply_ba,
    _build_match_cache,
    _detach_reference,
    _resolve_device,
    _seed,
    _sha256_file,
    build_ba,
    build_dataset,
    load_config as load_legacy_stage3_config,
)
from training.train_spherical_selfi_gaussian_head import (
    _deep_merge,
    _freeze,
    _init_wandb,
    _initialize_distributed,
    _scheduler,
    build_frozen_feature_stack,
    build_head,
    extract_frozen_inputs,
    load_stage2_checkpoint,
)


def default_config() -> dict[str, Any]:
    config = load_legacy_stage3_config(None)
    return _deep_merge(
        config,
        {
            "stage3": {"enabled": True},
            "VoxelAnchorRefiner": {
                "enabled": True,
                "checkpoint": None,
                "iterations": 3,
                "hidden_dim": 32,
                "adapter_dim": 24,
                "depth_boundaries": [5.0, 20.0, 40.0],
                "voxel_sizes": [0.04, 0.08, 0.16, 0.32],
                "alpha_threshold": 0.05,
                "depth_abs_threshold": 0.10,
                "depth_rel_threshold": 0.05,
                "tangent_scale_floor_ratio": 1.0 / 3.0,
                "normal_scale_floor_ratio": 0.05,
                "use_resnet_error": False,
                "pretrained_resnet": False,
            },
            "loss": {
                "stage_weights": [0.64, 0.80, 1.0],
                "dssim": 0.0,
                "depth": 0.05,
                "alpha_hole": 0.05,
                "update_regularization": 1.0e-4,
            },
            "train": {
                "output_dir": "outputs/stage3_spherical_voxel_anchor_refiner",
            },
        },
    )


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    base = raw.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = source.parent / base_path
        config = _deep_merge(config, load_legacy_stage3_config(base_path))
    return _deep_merge(config, raw)


class VoxelAnchorTrainableModel(VoxelAnchorStage3Model):
    def forward(
        self,
        observation: VoxelAnchorObservation,
        render_group: VoxelAnchorRenderGroup,
        reference: EncodedTargetReference,
        target_valid: torch.Tensor,
        *,
        iteration_index: int,
        hidden: torch.Tensor | None,
    ) -> VoxelAnchorRefinerOutput:
        return self.forward_step(
            observation,
            render_group,
            reference,
            target_valid,
            iteration_index=iteration_index,
            hidden=hidden,
        )


class SyntheticVoxelAnchorRenderer:
    """Small differentiable renderer used only by synthetic unit/smoke tests."""

    def render_group(self, observation: VoxelAnchorObservation) -> VoxelAnchorRenderGroup:
        batch, views = observation.batch_size, observation.num_views
        height, width = observation.image_size
        rendered = observation.xyz.new_zeros(batch, views, 3, height, width)
        depth = observation.xyz.new_zeros(batch, views, 1, height, width)
        alpha = observation.xyz.new_zeros(batch, views, 1, height, width)
        visibility = torch.zeros(
            batch,
            views,
            observation.num_anchors,
            device=observation.xyz.device,
            dtype=torch.bool,
        )
        for batch_index in range(batch):
            indices = observation.indices_for_batch(batch_index)
            if int(indices.numel()) == 0:
                continue
            opacity = observation.opacity.index_select(0, indices)
            weight = opacity / opacity.sum().clamp_min(1.0e-6)
            dc = observation.sh_coefficients.index_select(0, indices)[:, 0]
            color = (0.5 + float(SH_C0) * (weight * dc).sum(dim=0)).clamp(0.0, 1.0)
            rendered[batch_index] = color.view(1, 3, 1, 1)
            alpha[batch_index] = opacity.mean().clamp(0.0, 1.0)
            xyz = observation.xyz.index_select(0, indices)
            for view_index in range(views):
                camera = observation.local_poses_c2w[batch_index, view_index, :3, 3].to(xyz)
                mean_depth = (weight[:, 0] * torch.linalg.norm(xyz - camera, dim=-1)).sum()
                depth[batch_index, view_index] = mean_depth
            visibility[batch_index, :, indices] = True
        return VoxelAnchorRenderGroup(
            rendered=rendered,
            depth=depth,
            alpha=alpha,
            anchor_visibility=visibility,
            profiles={"materialized_gaussians": float(observation.num_anchors) / max(1, batch)},
        )


def build_renderer(config: dict[str, Any]):
    renderer_cfg = dict(config.get("renderer", {}) or {})
    backend = str(renderer_cfg.get("backend", "gsplat360")).lower()
    if backend == "synthetic":
        if not bool(config.get("dataset", {}).get("synthetic", False)):
            raise ValueError("The synthetic renderer is forbidden for real voxel-anchor training")
        return SyntheticVoxelAnchorRenderer()
    if backend != "gsplat360":
        raise ValueError(f"Unsupported voxel-anchor renderer backend: {backend!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Real voxel-anchor training requires CUDA gsplat360")
    return PFGS360Renderer(
        config=config,
        extra_gsplat360_roots=list(renderer_cfg.get("extra_gsplat360_roots", []) or []),
        allow_fallback=False,
    )


def render_group(renderer: Any, observation: VoxelAnchorObservation) -> VoxelAnchorRenderGroup:
    if isinstance(renderer, SyntheticVoxelAnchorRenderer):
        return renderer.render_group(observation)
    return render_voxel_anchor_group(renderer, observation)


def _unwrap(model: nn.Module) -> VoxelAnchorTrainableModel:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(module, VoxelAnchorTrainableModel):
        raise TypeError("Expected VoxelAnchorTrainableModel")
    return module


def _loss_weights(config: dict[str, Any]) -> VoxelAnchorLossWeights:
    raw = dict(config.get("loss", {}) or {})
    return VoxelAnchorLossWeights(
        dssim=float(raw.get("dssim", 0.0)),
        depth=float(raw.get("depth", 0.05)),
        alpha_hole=float(raw.get("alpha_hole", 0.05)),
        update_regularization=float(raw.get("update_regularization", 1.0e-4)),
    )


@dataclass
class VoxelAnchorWindowResult:
    final: VoxelAnchorObservation
    ba0_observation: PerPixelGaussianObservation
    snapshots: dict[str, VoxelAnchorObservation]
    rendered_snapshots: dict[str, torch.Tensor]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ValidationVisualization:
    """Detached CPU tensors retained from one validation window for logging."""

    target: torch.Tensor
    rendered_snapshots: dict[str, torch.Tensor]


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if device.index is None:
        torch.cuda.empty_cache()
        return
    with torch.cuda.device(device.index):
        torch.cuda.empty_cache()


def _detach_validation_visualization(
    result: VoxelAnchorWindowResult,
    images: torch.Tensor,
) -> ValidationVisualization:
    return ValidationVisualization(
        target=images.detach().float().cpu(),
        rendered_snapshots={
            name: rendered.detach().float().cpu()
            for name, rendered in result.rendered_snapshots.items()
        },
    )


def run_voxel_anchor_window(
    model: nn.Module,
    ba: BlockSparseSphericalBA,
    renderer: Any,
    stage2: PerPixelGaussianObservation,
    adapter_features: torch.Tensor,
    images: torch.Tensor,
    cache: Stage3MatchCache,
    config: dict[str, Any],
    *,
    backward: bool,
    accumulation_scale: float = 1.0,
) -> VoxelAnchorWindowResult:
    module = _unwrap(model)
    ba0, ba_output = _apply_ba(stage2, ba, cache)
    target_valid = ba0.valid_mask.bool()
    current = voxelize_per_pixel_gaussians(
        ba0,
        adapter_features,
        images,
        module.config,
        valid_mask=target_valid,
    )
    reference = _detach_reference(module.encode_references(images))
    stage_weights = [float(value) for value in config["loss"].get("stage_weights", [0.64, 0.8, 1.0])]
    if len(stage_weights) != 3:
        raise ValueError("loss.stage_weights must contain exactly three values")
    weights = _loss_weights(config)
    hidden = None
    snapshots = {"voxelized": current.detach_parameters()}
    initial_render = render_group(renderer, current)
    rendered_snapshots: dict[str, torch.Tensor] = {
        "voxelized": initial_render.rendered.detach()
    }
    metrics: dict[str, float] = {
        "ba0/accepted_ratio": float(ba_output.accepted.float().mean().detach().cpu()),
        "anchors/initial": float(current.num_anchors),
        "anchors/members": float(current.member_count.sum().detach().cpu()),
    }
    for iteration in range(3):
        feedback = initial_render if iteration == 0 else render_group(renderer, current)
        output = model(
            current,
            feedback,
            reference,
            target_valid,
            iteration_index=iteration,
            hidden=hidden,
        )
        current, hidden = output.observation, output.hidden
        rendered = render_group(renderer, current)
        loss, parts = spherical_voxel_anchor_loss(
            rendered.rendered,
            images,
            rendered.depth,
            rendered.alpha,
            ba0.refined_depth,
            target_valid,
            output.normalized_update_energy,
            weights=weights,
        )
        if backward:
            scaled = loss * stage_weights[iteration] * float(accumulation_scale)
            if scaled.requires_grad:
                scaled.backward()
        name = f"refine{iteration + 1}"
        snapshots[name] = current.detach_parameters()
        rendered_snapshots[name] = rendered.rendered.detach()
        for key, value in parts.items():
            metrics[f"stage{iteration + 1}/{key}"] = float(value.detach().float().cpu())
        metrics[f"stage{iteration + 1}/coverage_hole"] = float(
            ((1.0 - rendered.alpha).clamp_min(0.0) * target_valid).sum().detach().cpu()
            / target_valid.sum().clamp_min(1).detach().cpu()
        )
        if iteration < 2:
            current = current.detach_parameters()
            hidden = hidden.detach()
    metrics["anchors/final"] = float(current.num_anchors)
    return VoxelAnchorWindowResult(current, ba0, snapshots, rendered_snapshots, metrics)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, float],
    adapter_sha256: str,
    stage2_checkpoint_sha256: str | None,
) -> Path:
    module = _unwrap(model)
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite voxel-anchor parameter {name!r}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spherical_voxel_anchor_refiner_v1",
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_config": config,
            "global_step": int(step),
            "metrics": dict(metrics),
            "adapter_sha256": str(adapter_sha256),
            "stage2_checkpoint_sha256": stage2_checkpoint_sha256,
        },
        output,
    )
    return output


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "spherical_voxel_anchor_refiner_v1":
        raise ValueError(f"Unsupported voxel-anchor checkpoint: {path}")
    _unwrap(model).load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("global_step", 0)), dict(payload.get("metrics", {}))


@torch.no_grad()
def _render_metrics(result: VoxelAnchorWindowResult, images: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in ("voxelized", "refine3"):
        rendered = result.rendered_snapshots[name]
        l1_values, psnr_values, ssim_values = [], [], []
        for batch_index in range(int(images.shape[0])):
            for view_index in range(int(images.shape[1])):
                prediction = rendered[batch_index, view_index]
                target = images[batch_index, view_index]
                l1_values.append(spherical_weighted_l1(prediction, target))
                psnr_values.append(spherical_psnr(prediction, target))
                ssim_values.append(1.0 - 2.0 * spherical_dssim(prediction, target))
        prefix = "initial" if name == "voxelized" else "final"
        metrics[f"{prefix}_l1"] = float(torch.stack(l1_values).mean().cpu())
        metrics[f"{prefix}_psnr"] = float(torch.stack(psnr_values).mean().cpu())
        metrics[f"{prefix}_ssim"] = float(torch.stack(ssim_values).mean().cpu())
    metrics["psnr_delta"] = metrics["final_psnr"] - metrics["initial_psnr"]
    metrics["ssim_delta"] = metrics["final_ssim"] - metrics["initial_ssim"]
    metrics["hole_rate"] = result.metrics["stage3/coverage_hole"]
    metrics["anchor_count"] = result.metrics["anchors/final"]
    return metrics


@torch.no_grad()
def _validate(
    model: VoxelAnchorTrainableModel,
    wrapper: nn.Module,
    adapter: nn.Module,
    head: nn.Module,
    ba: BlockSparseSphericalBA,
    renderer: Any,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    feature_device: torch.device,
    train_device: torch.device,
    step: int,
) -> tuple[dict[str, float], ValidationVisualization | None]:
    was_training = model.training
    model.eval()
    aggregate: dict[str, float] = {}
    count = 0
    first_visualization: ValidationVisualization | None = None
    try:
        for batch in loader:
            features, images, initial_depth, poses = extract_frozen_inputs(
                wrapper,
                adapter,
                batch["images"],
                feature_device=feature_device,
                train_device=train_device,
                head_size=(int(config["image"]["head_height"]), int(config["image"]["head_width"])),
                feature_amp=bool(config["train"].get("amp", False)),
            )
            stage2 = head(
                features,
                images,
                initial_depth,
                poses,
                frame_ids=batch["frame_ids"].to(train_device),
            )
            cache = _build_match_cache(
                features,
                stage2.refined_depth,
                config,
                step=step + count,
                static_valid_mask=stage2.valid_mask,
            )
            with torch.amp.autocast(
                device_type=train_device.type,
                dtype=torch.bfloat16,
                enabled=bool(config["train"].get("amp", False)) and train_device.type == "cuda",
            ):
                result = run_voxel_anchor_window(
                    model,
                    ba,
                    renderer,
                    stage2,
                    features,
                    images,
                    cache,
                    config,
                    backward=False,
                )
            values = _render_metrics(result, images)
            values.update(
                {
                    "stage3_rgb_l1": result.metrics["stage3/rgb_l1"],
                    "stage3_relative_ba0_depth": result.metrics["stage3/relative_ba0_depth"],
                }
            )
            for key, value in values.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
            if first_visualization is None:
                first_visualization = _detach_validation_visualization(result, images)
            count += 1
            del result, cache, stage2, features, images, initial_depth, poses
            _empty_cuda_cache(train_device)
            if count >= int(config["train"].get("max_val_batches", 8)):
                break
    finally:
        model.train(was_training)
        _empty_cuda_cache(train_device)
    if count == 0:
        return {}, first_visualization
    return ({f"val/{key}": value / count for key, value in aggregate.items()}, first_visualization)


def train(config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("stage3", {}).get("enabled", False)):
        raise ValueError("Stage 3 is config-gated; set stage3.enabled=true")
    if not bool(config.get("VoxelAnchorRefiner", {}).get("enabled", False)):
        raise ValueError("Set VoxelAnchorRefiner.enabled=true for anchor-refiner training")
    if not bool(config.get("dataset", {}).get("synthetic", False)):
        if not bool(config.get("Visualization", {}).get("enabled", False)):
            raise ValueError("Real voxel-anchor training requires Visualization.enabled=true")
        if not bool(config.get("WeightsAndBiases", {}).get("enabled", False)):
            raise ValueError("Real voxel-anchor training requires WeightsAndBiases.enabled=true")

    train_cfg = config["train"]
    distributed = _initialize_distributed(train_cfg)
    _seed(int(train_cfg.get("seed", 1234)) + distributed.rank)
    train_device = _resolve_device(str(train_cfg.get("train_device", "auto")), local_rank=distributed.local_rank)
    feature_device = _resolve_device(str(train_cfg.get("feature_device", "auto")), local_rank=distributed.local_rank)
    output_dir = Path(train_cfg["output_dir"])
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    wrapper, adapter, adapter_sha, _ = build_frozen_feature_stack(config, device=feature_device)
    head = _freeze(build_head(config, device=train_device))
    stage2_path = config.get("stage2_checkpoint", {}).get("path")
    stage2_sha = _sha256_file(stage2_path)
    if stage2_path:
        load_stage2_checkpoint(stage2_path, head=head, expected_adapter_sha256=adapter_sha, map_location=train_device)
    elif bool(config.get("stage2_checkpoint", {}).get("required", False)):
        raise ValueError("stage2_checkpoint.path is required")

    anchor_config = VoxelAnchorConfig.from_mapping(config["VoxelAnchorRefiner"])
    trainable: nn.Module = VoxelAnchorTrainableModel(anchor_config).to(train_device)
    if distributed.enabled:
        trainable = DistributedDataParallel(
            trainable,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
        )
    parameters = [parameter for parameter in trainable.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        warmup_steps=int(train_cfg["warmup_steps"]),
        max_steps=int(train_cfg["max_steps"]),
    )
    start_step = 0
    if train_cfg.get("resume"):
        start_step, _ = load_checkpoint(
            train_cfg["resume"], model=trainable, optimizer=optimizer, scheduler=scheduler
        )
    ba = build_ba(config)
    renderer = build_renderer(config)
    dataset = build_dataset(config, split="train")
    sampler = DistributedSampler(dataset, shuffle=True) if distributed.enabled else None
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage3_collate,
        pin_memory=train_device.type == "cuda",
    )
    val_loader = None
    if distributed.is_main:
        val_dataset = build_dataset(config, split="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=stage3_collate,
            pin_memory=train_device.type == "cuda",
        )
    wandb_run = _init_wandb(config, output_dir) if distributed.is_main else None
    accumulation = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    max_steps = int(train_cfg.get("max_steps", 20_000))
    step, micro, epoch = start_step, 0, 0
    best_psnr = -float("inf")
    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for batch in loader:
            features, images, initial_depth, poses = extract_frozen_inputs(
                wrapper,
                adapter,
                batch["images"],
                feature_device=feature_device,
                train_device=train_device,
                head_size=(int(config["image"]["head_height"]), int(config["image"]["head_width"])),
                feature_amp=bool(train_cfg.get("amp", False)),
            )
            with torch.no_grad():
                stage2 = head(
                    features,
                    images,
                    initial_depth,
                    poses,
                    frame_ids=batch["frame_ids"].to(train_device),
                )
            cache = _build_match_cache(
                features,
                stage2.refined_depth,
                config,
                step=step + micro,
                static_valid_mask=stage2.valid_mask,
            )
            with torch.amp.autocast(
                device_type=train_device.type,
                dtype=torch.bfloat16,
                enabled=bool(train_cfg.get("amp", False)) and train_device.type == "cuda",
            ):
                result = run_voxel_anchor_window(
                    trainable,
                    ba,
                    renderer,
                    stage2,
                    features,
                    images,
                    cache,
                    config,
                    backward=True,
                    accumulation_scale=1.0 / accumulation,
                )
            micro += 1
            if micro % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            metrics = dict(result.metrics)
            metrics.update(_render_metrics(result, images))
            metrics.update({"train/step": float(step), "train/lr": float(optimizer.param_groups[0]["lr"])})
            if train_device.type == "cuda":
                metrics["memory/max_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(train_device) / (1024.0 * 1024.0)
                )
            last_metrics = metrics
            if distributed.is_main and step % max(1, int(train_cfg.get("log_interval", 1))) == 0:
                print(yaml.safe_dump({"step": step, "metrics": metrics}, sort_keys=False).strip(), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            diagnostic_interval = max(1, int(train_cfg.get("diagnostics_interval", 200)))
            if distributed.is_main and step % diagnostic_interval == 0 and bool(config.get("Visualization", {}).get("enabled", False)):
                visual_path = save_stage3_snapshot_panel(
                    output_dir / str(config["Visualization"].get("save_dir", "visualizations")) / f"step_{step:07d}.png",
                    target=images,
                    rendered_snapshots=result.rendered_snapshots,
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({"diagnostics/voxel_anchor_refiner": wandb.Image(str(visual_path))}, step=step)
            val_interval = max(1, int(train_cfg.get("val_interval", 1000)))
            if step % val_interval == 0:
                # Validation is a separate full-resolution workload. Drop the
                # completed training window before either rank enters the
                # barrier so its tensors cannot overlap the validation peak.
                del result, cache, stage2, features, images, initial_depth, poses
                _empty_cuda_cache(train_device)
                if distributed.enabled:
                    dist.barrier()
                if distributed.is_main and val_loader is not None:
                    val_metrics, val_visualization = _validate(
                        _unwrap(trainable),
                        wrapper,
                        adapter,
                        head,
                        ba,
                        renderer,
                        val_loader,
                        config,
                        feature_device=feature_device,
                        train_device=train_device,
                        step=step,
                    )
                    metrics.update(val_metrics)
                    last_metrics = metrics
                    if wandb_run is not None and val_metrics:
                        wandb_run.log(val_metrics, step=step)
                    if (
                        val_visualization is not None
                        and bool(config.get("Visualization", {}).get("enabled", False))
                    ):
                        val_path = save_stage3_snapshot_panel(
                            output_dir
                            / str(config["Visualization"].get("save_dir", "visualizations"))
                            / f"val_{step:07d}.png",
                            target=val_visualization.target,
                            rendered_snapshots=val_visualization.rendered_snapshots,
                        )
                        if wandb_run is not None:
                            import wandb

                            wandb_run.log(
                                {"validation/voxel_anchor_refiner": wandb.Image(str(val_path))},
                                step=step,
                            )
                    del val_visualization
                    _empty_cuda_cache(train_device)
                    final_psnr = val_metrics.get("val/final_psnr")
                    if final_psnr is not None and final_psnr > best_psnr:
                        best_psnr = final_psnr
                        save_checkpoint(
                            output_dir / "checkpoints" / "best_val_psnr.pt",
                            model=trainable,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            step=step,
                            metrics=metrics,
                            adapter_sha256=adapter_sha,
                            stage2_checkpoint_sha256=stage2_sha,
                        )
                if distributed.enabled:
                    dist.barrier()
            if distributed.is_main and step % max(1, int(train_cfg.get("save_interval", 1000))) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / "latest.pt",
                    model=trainable,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    step=step,
                    metrics=metrics,
                    adapter_sha256=adapter_sha,
                    stage2_checkpoint_sha256=stage2_sha,
                )
            if step >= max_steps:
                break
        epoch += 1

    if distributed.is_main:
        save_checkpoint(
            output_dir / "checkpoints" / "latest.pt",
            model=trainable,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            metrics=last_metrics,
            adapter_sha256=adapter_sha,
            stage2_checkpoint_sha256=stage2_sha,
        )
    if wandb_run is not None:
        wandb_run.finish()
    if distributed.enabled:
        dist.barrier()
        dist.destroy_process_group()
    return {
        "step": step,
        "metrics": last_metrics,
        "checkpoint": str(output_dir / "checkpoints" / "latest.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
