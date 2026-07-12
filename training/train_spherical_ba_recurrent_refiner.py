"""Standalone Stage 3 spherical BA and recurrent Gaussian-refiner training."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

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
from data.stage3_spherical_ba_refiner_dataset import (
    Stage3Omni360Dataset,
    SyntheticStage3Dataset,
    stage3_collate,
)
from losses.spherical_gaussian_render_loss import spherical_dssim, spherical_psnr, spherical_weighted_l1
from losses.spherical_stage3_refinement_loss import (
    Stage3LossWeights,
    aligned_pose_metrics,
    build_ba_support_map,
    depth_metrics,
    stage3_loss,
)
from models.per_pixel_gaussian_observation import SH_C0, PerPixelGaussianObservation
from models.spherical_recurrent_gaussian_refiner import (
    EncodedTargetReference,
    ReSplatErrorEncoder,
    SphericalErrorRouter,
    SphericalRecurrentGaussianRefiner,
    Stage3RefinementResult,
    Stage3RefinerOutput,
    scatter_materialized_visibility,
)
from models.spherical_selfi_stage3_ba import (
    BlockSparseSphericalBA,
    Stage3BAOutput,
    Stage3MatchCache,
    build_stage3_match_cache,
)
from training.train_spherical_selfi_gaussian_head import (
    DistributedContext,
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
from tools.visualize_stage3_refinement import save_stage3_snapshot_panel


def default_config() -> dict[str, Any]:
    return {
        "stage3": {"enabled": False},
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
        "stage2_checkpoint": {"path": None, "sha256": None, "required": False},
        "synthetic_adapter": {"hidden_dim": 16},
        "head": {
            "feature_dim": 24,
            "channels": [8, 16, 24, 32],
            "mlp_hidden_dim": 16,
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
            "root": None,
            "pose_root": None,
            "scenes": ["DTW", "NYC"],
            "views_per_sample": 4,
            "stride_min": 2,
            "stride_max": 6,
            "validation_fraction": 0.1,
            "depth_format": "euclidean_range",
            "depth_scale": 1.0,
            "depth_invalid_value": 1000.0,
            "pose_coordinate_system": "ue_airsim",
            "pose_translation_scale": 0.01,
            "max_train_samples": 2,
            "max_val_samples": 1,
        },
        "matching": {
            "num_queries": 2048,
            "min_depth": 0.05,
            "max_depth": 20.0,
            "temperature": 0.07,
            "query_chunk_size": 32,
            "fibonacci_oversample_factor": 8,
            "use_spherical_area_correction": True,
        },
        "ba": {
            "outer_schedule": [True, False, False],
            "iterations": 3,
            "damping": 1.0e-4,
            "huber_delta_deg": 0.5,
            "pose_prior_weight": 1.0e-3,
            "depth_prior_weight": 1.0e-2,
            "max_pose_update_deg": 5.0,
            "max_translation_update": 0.05,
            "max_logdepth_update": 0.35,
            "factor_chunk_size": 2048,
            "min_factors": 256,
            "residual_worse_tolerance": 1.05,
            "min_affine_support": 64,
            "solver_mode": "standard_lm",
            "dense_depth_mode": "none",
            "gauge_mode": "initial_baseline",
            "lm_max_trials": 4,
            "lm_acceptance_eta": 1.0e-4,
            "lm_damping_min": 1.0e-8,
            "lm_damping_max": 1.0e8,
            "lm_diagonal_floor": 1.0e-6,
        },
        "refiner": {
            "adapter_dim": 24,
            "hidden_dim": 32,
            "use_resnet_error": False,
            "pretrained_resnet": False,
            "alpha_threshold": 0.05,
            "depth_abs_threshold": 0.10,
            "depth_rel_threshold": 0.05,
            "profile_synchronize_cuda": False,
        },
        "loss": {
            "stage_weights": [0.64, 0.80, 1.0],
            "dssim": 0.2,
            "geometry": 0.05,
            "depth_anchor": 1.0e-3,
            "update_regularization": 1.0e-4,
        },
        "renderer": {"backend": "synthetic", "extra_gsplat360_roots": []},
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
            "grad_clip": 1.0,
            "diagnostics_interval": 200,
            "log_interval": 1,
            "val_interval": 1,
            "max_val_batches": 1,
            "save_interval": 1,
            "output_dir": "outputs/stage3_spherical_ba_recurrent_refiner",
            "resume": None,
            "seed": 1234,
        },
        "WeightsAndBiases": {"enabled": False, "project": "360Droid-splat", "mode": "online", "log_every": 50},
        "Visualization": {"enabled": False, "interval": 200, "save_dir": "visualizations"},
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


def _resolve_device(value: str, *, local_rank: int = 0) -> torch.device:
    if str(value).lower() != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda", min(int(local_rank), torch.cuda.device_count() - 1))
    return torch.device("cpu")


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_dataset(config: dict[str, Any], *, split: str):
    cfg = config["dataset"]
    maximum = cfg.get("max_train_samples" if split == "train" else "max_val_samples")
    if bool(cfg.get("synthetic", False)):
        return SyntheticStage3Dataset(
            length=int(maximum or 2),
            views=int(cfg.get("views_per_sample", 4)),
            height=int(config["image"]["height"]),
            width=int(config["image"]["width"]),
        )
    return Stage3Omni360Dataset(
        cfg["root"],
        pose_root=cfg["pose_root"],
        scenes=cfg.get("scenes", ["DTW", "NYC"]),
        split=split,
        views_per_sample=int(cfg.get("views_per_sample", 4)),
        stride_min=int(cfg.get("stride_min", 2)),
        stride_max=int(cfg.get("stride_max", 6)),
        resize=(int(config["image"]["height"]), int(config["image"]["width"])),
        validation_fraction=float(cfg.get("validation_fraction", 0.1)),
        depth_format=str(cfg.get("depth_format", "euclidean_range")),
        depth_scale=float(cfg.get("depth_scale", 1.0)),
        depth_invalid_value=cfg.get("depth_invalid_value", 1000.0),
        pose_coordinate_system=str(cfg.get("pose_coordinate_system", "ue_airsim")),
        pose_translation_scale=float(cfg.get("pose_translation_scale", 0.01)),
        seed=int(config["train"].get("seed", 1234)),
        max_clips=maximum,
    )


class SyntheticDifferentiableRenderer:
    """Tiny differentiable contract renderer used only by explicit synthetic tests."""

    def render_group(self, observation: PerPixelGaussianObservation) -> "RenderGroup":
        batch, views, height, width = (
            observation.batch_size,
            observation.num_source_views,
            *observation.image_size,
        )
        dc = observation.rgb_sh[:, :, 0]
        color = (0.5 + float(SH_C0) * dc).clamp(0.0, 1.0)
        opacity = observation.confidence
        rendered, depths, alphas = [], [], []
        visibility = torch.zeros(batch, views, views, 1, height, width, device=color.device, dtype=torch.bool)
        for target in range(views):
            sources = [source for source in range(views) if source != target]
            weights = opacity[:, sources].clamp_min(1.0e-4)
            rendered.append((color[:, sources] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-4))
            depths.append((observation.refined_depth[:, sources] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-4))
            alphas.append(weights.mean(dim=1).clamp(0.0, 1.0))
            visibility[:, target, sources] = observation.valid_mask[:, sources]
        return RenderGroup(
            rendered=torch.stack(rendered, dim=1),
            depth=torch.stack(depths, dim=1),
            alpha=torch.stack(alphas, dim=1),
            source_visibility=visibility,
            profiles={"materialized_gaussians": float(observation.canonical_count)},
        )


@dataclass
class RenderGroup:
    rendered: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    source_visibility: torch.Tensor
    profiles: dict[str, float]


def render_leave_one_out_group(renderer: Any, observation: PerPixelGaussianObservation) -> RenderGroup:
    if isinstance(renderer, SyntheticDifferentiableRenderer):
        return renderer.render_group(observation)
    batch, views = observation.batch_size, observation.num_source_views
    height, width = observation.image_size
    rendered, depth, alpha = [], [], []
    visibility = torch.zeros(
        batch, views, views, 1, height, width,
        device=observation.refined_depth.device,
        dtype=torch.bool,
    )
    total_gaussians = 0.0
    profile_sums: dict[str, float] = {}
    for batch_idx in range(batch):
        rendered_batch, depth_batch, alpha_batch = [], [], []
        for target in range(views):
            camera = PanoRenderCamera(height, width, observation.poses_c2w[batch_idx, target].float())
            sources = [source for source in range(views) if source != target]
            explicit = observation.materialize_batch(camera, batch_index=batch_idx, source_indices=sources)
            package = renderer.render(camera, explicit)
            rendered_batch.append(package["render"])
            depth_batch.append(package["depth"])
            alpha_batch.append(package["alpha"])
            visibility[batch_idx, target] = scatter_materialized_visibility(
                explicit,
                package["visibility_filter"],
                frame_ids=observation.frame_ids[batch_idx],
                height=height,
                width=width,
            )
            total_gaussians += float(explicit.xyz.shape[0])
            for key, value in package.items():
                if str(key).startswith("profile_renderer_") and isinstance(value, (float, int)):
                    profile_sums[str(key)] = profile_sums.get(str(key), 0.0) + float(value)
        rendered.append(torch.stack(rendered_batch, dim=0))
        depth.append(torch.stack(depth_batch, dim=0))
        alpha.append(torch.stack(alpha_batch, dim=0))
    return RenderGroup(
        rendered=torch.stack(rendered, dim=0),
        depth=torch.stack(depth, dim=0),
        alpha=torch.stack(alpha, dim=0),
        source_visibility=visibility,
        profiles={
            "materialized_gaussians": total_gaussians / max(1, batch * views),
            **{key: value / max(1, batch * views) for key, value in profile_sums.items()},
        },
    )


class Stage3TrainableModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        cfg = config["refiner"]
        self.error_encoder = ReSplatErrorEncoder(
            use_resnet=bool(cfg.get("use_resnet_error", True)),
            pretrained_resnet=bool(cfg.get("pretrained_resnet", True)),
        )
        self.error_router = SphericalErrorRouter()
        self.refiner = SphericalRecurrentGaussianRefiner(
            adapter_dim=int(cfg.get("adapter_dim", 24)),
            hidden_dim=int(cfg.get("hidden_dim", 32)),
        )
        self.alpha_threshold = float(cfg.get("alpha_threshold", 0.05))
        self.depth_abs_threshold = float(cfg.get("depth_abs_threshold", 0.10))
        self.depth_rel_threshold = float(cfg.get("depth_rel_threshold", 0.05))
        self.profile_synchronize_cuda = bool(cfg.get("profile_synchronize_cuda", False))

    def _sync(self, device: torch.device) -> None:
        if self.profile_synchronize_cuda and device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.no_grad()
    def encode_references(self, images: torch.Tensor) -> EncodedTargetReference:
        batch, views, channels, height, width = images.shape
        return self.error_encoder.encode_reference(images.reshape(batch * views, channels, height, width))

    def forward(
        self,
        observation: PerPixelGaussianObservation,
        stage2_observation: PerPixelGaussianObservation,
        adapter_features: torch.Tensor,
        images: torch.Tensor,
        render_group: RenderGroup,
        reference: EncodedTargetReference,
        *,
        iteration_index: int,
        hidden: torch.Tensor | None,
    ) -> Stage3RefinerOutput:
        batch, views = observation.batch_size, observation.num_source_views
        self._sync(observation.refined_depth.device)
        start = time.perf_counter()
        error = self.error_encoder(
            render_group.rendered.reshape(batch * views, 3, *observation.image_size),
            reference,
        )
        self._sync(observation.refined_depth.device)
        error_sec = time.perf_counter() - start
        start = time.perf_counter()
        error = error.reshape(batch, views, 32, *error.shape[-2:])
        routed = self.error_router(
            observation,
            error,
            render_group.depth,
            render_group.alpha,
            target_source_visibility=render_group.source_visibility,
            alpha_threshold=self.alpha_threshold,
            depth_abs_threshold=self.depth_abs_threshold,
            depth_rel_threshold=self.depth_rel_threshold,
        )
        self._sync(observation.refined_depth.device)
        routing_sec = time.perf_counter() - start
        start = time.perf_counter()
        output = self.refiner(
            observation,
            stage2_observation,
            adapter_features,
            images,
            routed,
            iteration_index=iteration_index,
            hidden=hidden,
        )
        self._sync(observation.refined_depth.device)
        output.profile = {
            "error_encoder_sec": float(error_sec),
            "error_routing_sec": float(routing_sec),
            "refiner_sec": float(time.perf_counter() - start),
        }
        return output


def _unwrap(model: nn.Module) -> Stage3TrainableModel:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(module, Stage3TrainableModel):
        raise TypeError("Expected Stage3TrainableModel.")
    return module


def build_ba(config: dict[str, Any]) -> BlockSparseSphericalBA:
    kwargs = dict(config.get("ba", {}))
    kwargs.pop("outer_schedule", None)
    return BlockSparseSphericalBA(**kwargs)


def _ba_outer_schedule(config: dict[str, Any]) -> tuple[bool, bool, bool]:
    raw = config.get("ba", {}).get("outer_schedule", [True, True, True])
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("ba.outer_schedule must contain exactly three booleans for BA0/BA1/BA2.")
    schedule = tuple(bool(value) for value in raw)
    if not schedule[0]:
        raise ValueError("Stage 3 currently requires BA0 before Refine1.")
    return schedule


def build_renderer(config: dict[str, Any]):
    cfg = config["renderer"]
    backend = str(cfg.get("backend", "gsplat360")).lower()
    if backend == "synthetic":
        if not bool(config["dataset"].get("synthetic", False)):
            raise ValueError("The synthetic renderer is forbidden for real Stage 3 training.")
        return SyntheticDifferentiableRenderer()
    if backend != "gsplat360":
        raise ValueError(f"Unsupported Stage 3 renderer backend: {backend!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Real Stage 3 training requires the CUDA gsplat360 renderer.")
    return PFGS360Renderer(
        config=config,
        extra_gsplat360_roots=list(cfg.get("extra_gsplat360_roots", [])),
        allow_fallback=False,
    )


def _apply_ba(
    observation: PerPixelGaussianObservation,
    ba: BlockSparseSphericalBA,
    cache: Stage3MatchCache,
) -> tuple[PerPixelGaussianObservation, Stage3BAOutput]:
    start = time.perf_counter()
    with torch.amp.autocast(device_type=observation.refined_depth.device.type, enabled=False):
        output = ba(observation.poses_c2w, observation.refined_depth, cache)
    elapsed = time.perf_counter() - start
    for diagnostic in output.diagnostics:
        diagnostic["solver_wall_sec"] = float(elapsed / max(1, len(output.diagnostics)))
    return observation.with_geometry(poses_c2w=output.poses_c2w, refined_depth=output.dense_depth), output


def _detach_reference(reference: EncodedTargetReference) -> EncodedTargetReference:
    return EncodedTargetReference(
        image=reference.image.detach(),
        features=None if reference.features is None else tuple(feature.detach() for feature in reference.features),
    )


def _loss_weights(config: dict[str, Any]) -> Stage3LossWeights:
    cfg = config["loss"]
    return Stage3LossWeights(
        dssim=float(cfg.get("dssim", 0.2)),
        geometry=float(cfg.get("geometry", 0.05)),
        depth_anchor=float(cfg.get("depth_anchor", 1.0e-3)),
        update_regularization=float(cfg.get("update_regularization", 1.0e-4)),
    )


def save_stage3_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, float],
    adapter_sha256: str,
    stage2_checkpoint: str | None,
    stage2_checkpoint_sha256: str | None = None,
) -> Path:
    module = _unwrap(model)
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite Stage 3 parameter {name!r}; refusing to checkpoint.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spherical_ba_recurrent_gaussian_refiner_v1",
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_config": config,
            "global_step": int(step),
            "metrics": dict(metrics),
            "adapter_sha256": str(adapter_sha256),
            "stage2_checkpoint": stage2_checkpoint,
            "stage2_checkpoint_sha256": stage2_checkpoint_sha256,
        },
        output,
    )
    return output


def load_stage3_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_adapter_sha256: str | None = None,
    expected_stage2_sha256: str | None = None,
) -> tuple[int, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "spherical_ba_recurrent_gaussian_refiner_v1":
        raise ValueError(f"Unsupported Stage 3 checkpoint: {path}.")
    if expected_adapter_sha256 is not None and payload.get("adapter_sha256") != expected_adapter_sha256:
        raise ValueError("Stage 3 checkpoint adapter SHA256 mismatch.")
    if expected_stage2_sha256 is not None and payload.get("stage2_checkpoint_sha256") != expected_stage2_sha256:
        raise ValueError("Stage 3 checkpoint Stage 2 SHA256 mismatch.")
    _unwrap(model).load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("global_step", 0)), dict(payload.get("metrics", {}))


def _sha256_file(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_match_cache(
    features: torch.Tensor,
    depth: torch.Tensor,
    config: dict[str, Any],
    *,
    step: int,
    static_valid_mask: torch.Tensor | None = None,
) -> Stage3MatchCache:
    cfg = config["matching"]
    generator = torch.Generator(device=features.device).manual_seed(int(config["train"].get("seed", 1234)) + int(step))
    return build_stage3_match_cache(
        features,
        depth,
        num_queries=int(cfg.get("num_queries", 2048)),
        min_depth=float(cfg.get("min_depth", 0.05)),
        max_depth=float(cfg.get("max_depth", 20.0)),
        temperature=float(cfg.get("temperature", 0.07)),
        query_chunk_size=int(cfg.get("query_chunk_size", 32)),
        fibonacci_oversample_factor=int(cfg.get("fibonacci_oversample_factor", 8)),
        use_spherical_area_correction=bool(cfg.get("use_spherical_area_correction", True)),
        forward_backward=bool(cfg.get("forward_backward", True)),
        fb_tolerance_deg=float(cfg.get("fb_tolerance_deg", 1.0)),
        min_factor_weight=float(cfg.get("min_factor_weight", 0.01)),
        static_valid_mask=static_valid_mask,
        generator=generator,
    )


def _scalar_metrics(parts: dict[str, torch.Tensor], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value.detach().float().cpu()) for key, value in parts.items()}


def _train_microbatch(
    model: nn.Module,
    ba: BlockSparseSphericalBA,
    renderer: Any,
    stage2_observation: PerPixelGaussianObservation,
    adapter_features: torch.Tensor,
    images: torch.Tensor,
    cache: Stage3MatchCache,
    reference: EncodedTargetReference,
    config: dict[str, Any],
    *,
    accumulation_scale: float,
) -> tuple[PerPixelGaussianObservation, dict[str, float], dict[str, PerPixelGaussianObservation]]:
    module = _unwrap(model)
    support = build_ba_support_map(cache, height=stage2_observation.image_size[0], width=stage2_observation.image_size[1])
    stage_weights = [float(value) for value in config["loss"].get("stage_weights", [0.64, 0.8, 1.0])]
    weights = _loss_weights(config)
    ba_schedule = _ba_outer_schedule(config)
    snapshots: dict[str, PerPixelGaussianObservation] = {"initial": stage2_observation.detach_parameters()}
    match_valid = cache.valid_mask.bool()
    metrics: dict[str, float] = {"matching/valid_factors": float(match_valid.sum().cpu())}
    if match_valid.any():
        metrics.update(
            {
                "matching/top1_cosine_mean": float(cache.top1_cosine[match_valid].mean().cpu()),
                "matching/top2_margin_mean": float(cache.top2_margin[match_valid].mean().cpu()),
                "matching/entropy_mean": float(cache.entropy[match_valid].mean().cpu()),
            }
        )

    current, ba0 = _apply_ba(stage2_observation, ba, cache)
    snapshots["ba0"] = current.detach_parameters()
    feedback = render_leave_one_out_group(renderer, current)
    hidden: torch.Tensor | None = None
    ba_metrics: list[tuple[int, Stage3BAOutput]] = [(0, ba0)]
    for iteration in range(3):
        sync_context = model.no_sync() if isinstance(model, DistributedDataParallel) and iteration < 2 else nullcontext()
        with sync_context:
            ba_base_depth = current.refined_depth
            refined = model(
                current,
                stage2_observation,
                adapter_features,
                images,
                feedback,
                reference,
                iteration_index=iteration,
                hidden=hidden,
            )
            if refined.profile:
                for key, value in refined.profile.items():
                    metrics[f"profile/stage{iteration + 1}_{key}"] = float(value)
            snapshots[f"refine{iteration + 1}"] = refined.observation.detach_parameters()
            hidden = refined.hidden
            next_ba_index = iteration + 1
            if iteration < 2 and ba_schedule[next_ba_index]:
                current, ba_output = _apply_ba(refined.observation, ba, cache)
                ba_metrics.append((next_ba_index, ba_output))
                snapshots[f"ba{next_ba_index}"] = current.detach_parameters()
            else:
                current = refined.observation
                ba_output = None
            render = render_leave_one_out_group(renderer, current)
            for key, value in render.profiles.items():
                metrics[f"profile/stage{iteration + 1}_{key}"] = float(value)
            loss, parts = stage3_loss(
                render.rendered,
                images,
                current,
                cache,
                ba_base_depth,
                support,
                refined.normalized_update_energy,
                anchor_prediction=refined.observation.refined_depth,
                anchor_valid_mask=refined.observation.valid_mask,
                weights=weights,
            )
            scaled = loss * stage_weights[iteration] * float(accumulation_scale)
            scaled.backward()
        metrics.update(_scalar_metrics(parts, f"stage{iteration + 1}"))
        if iteration < 2:
            current = current.detach_parameters()
            hidden = hidden.detach()
            feedback = RenderGroup(
                rendered=render.rendered.detach(),
                depth=render.depth.detach(),
                alpha=render.alpha.detach(),
                source_visibility=render.source_visibility.detach(),
                profiles=render.profiles,
            )
    for ba_index, selected_ba in ba_metrics:
        metrics[f"ba{ba_index}/accepted_ratio"] = float(selected_ba.accepted.float().mean().cpu())
        metrics[f"ba{ba_index}/initial_residual_deg"] = float(selected_ba.initial_median_residual_deg.mean().cpu())
        metrics[f"ba{ba_index}/final_residual_deg"] = float(selected_ba.final_median_residual_deg.mean().cpu())
        metrics[f"ba{ba_index}/depth_scale_mean"] = float(selected_ba.depth_scale.float().mean().cpu())
        metrics[f"ba{ba_index}/depth_scale_min"] = float(selected_ba.depth_scale.float().min().cpu())
        metrics[f"ba{ba_index}/depth_scale_max"] = float(selected_ba.depth_scale.float().max().cpu())
        metrics[f"ba{ba_index}/depth_shift_mean"] = float(selected_ba.depth_shift.float().mean().cpu())
        metrics[f"ba{ba_index}/depth_shift_abs_max"] = float(selected_ba.depth_shift.float().abs().max().cpu())
        metrics[f"profile/ba{ba_index}_solver_sec"] = float(
            sum(float(value.get("solver_wall_sec", 0.0)) for value in selected_ba.diagnostics)
        )
        for key in (
            "initial_objective",
            "final_objective",
            "accepted_steps",
            "final_damping",
            "gain_ratio_mean",
            "gauge_scale_mean",
            "gauge_scale_min",
            "gauge_scale_max",
        ):
            values = [float(item[key]) for item in selected_ba.diagnostics if key in item]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                metrics[f"ba{ba_index}/{key}"] = float(sum(finite) / len(finite))
    metrics["render/materialized_gaussians"] = float(feedback.profiles.get("materialized_gaussians", 0.0))
    return current, metrics, snapshots


@torch.no_grad()
def _diagnostic_metrics(
    snapshots: dict[str, PerPixelGaussianObservation],
    renderer: Any,
    images: torch.Tensor,
    gt_poses: torch.Tensor,
    gt_depths: torch.Tensor,
    gt_valid: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    result: dict[str, float] = {}
    rendered_snapshots: dict[str, torch.Tensor] = {}
    target_size = next(iter(snapshots.values())).image_size
    if tuple(gt_depths.shape[-2:]) != tuple(target_size):
        batch, views = int(gt_depths.shape[0]), int(gt_depths.shape[1])
        gt_depths = torch.nn.functional.interpolate(
            gt_depths.reshape(batch * views, 1, *gt_depths.shape[-2:]),
            size=target_size,
            mode="nearest",
        ).reshape(batch, views, 1, *target_size)
        gt_valid = torch.nn.functional.interpolate(
            gt_valid.float().reshape(batch * views, 1, *gt_valid.shape[-2:]),
            size=target_size,
            mode="nearest",
        ).reshape(batch, views, 1, *target_size) > 0.5
    for name, observation in snapshots.items():
        render = render_leave_one_out_group(renderer, observation)
        rendered_snapshots[name] = render.rendered.detach()
        l1_values, psnr_values, ssim_values = [], [], []
        for batch in range(observation.batch_size):
            for view in range(observation.num_source_views):
                l1_values.append(spherical_weighted_l1(render.rendered[batch, view], images[batch, view]))
                psnr_values.append(spherical_psnr(render.rendered[batch, view], images[batch, view]))
                ssim_values.append(1.0 - 2.0 * spherical_dssim(render.rendered[batch, view], images[batch, view]))
        result[f"{name}/loo_l1"] = float(torch.stack(l1_values).mean().cpu())
        result[f"{name}/loo_psnr"] = float(torch.stack(psnr_values).mean().cpu())
        result[f"{name}/loo_ssim"] = float(torch.stack(ssim_values).mean().cpu())
        for batch in range(observation.batch_size):
            for key, value in aligned_pose_metrics(observation.poses_c2w[batch], gt_poses[batch]).items():
                result.setdefault(f"{name}/pose_{key}", 0.0)
                result[f"{name}/pose_{key}"] += value / observation.batch_size
        result.update({f"{name}/depth_{key}": value for key, value in depth_metrics(observation.refined_depth, gt_depths, gt_valid).items()})
        confidence = observation.confidence[observation.valid_mask.bool()].detach().float()
        if confidence.numel():
            result[f"{name}/confidence_p10"] = float(confidence.quantile(0.1).cpu())
            result[f"{name}/confidence_p50"] = float(confidence.quantile(0.5).cpu())
            result[f"{name}/confidence_p90"] = float(confidence.quantile(0.9).cpu())
    for index in range(1, 4):
        ba_name = f"ba{index - 1}"
        refine_name = f"refine{index}"
        predecessor_name = ba_name if ba_name in snapshots else ("initial" if index == 1 else f"refine{index - 1}")
        if predecessor_name in snapshots and refine_name in snapshots:
            if not torch.equal(snapshots[predecessor_name].poses_c2w, snapshots[refine_name].poses_c2w):
                raise AssertionError("Refiner changed camera poses; only BA may update pose.")
            before, after = snapshots[predecessor_name], snapshots[refine_name]
            valid = before.valid_mask.bool()
            depth_delta = ((after.refined_depth - before.refined_depth) / before.refined_depth.clamp_min(1.0e-6))[valid]
            scale_delta = (after.log_scale_multiplier - before.log_scale_multiplier)[valid.expand_as(before.log_scale_multiplier)]
            before_q = before.local_quaternion.permute(0, 1, 3, 4, 2)
            after_q = after.local_quaternion.permute(0, 1, 3, 4, 2)
            rotation_dot = (before_q * after_q).sum(dim=-1).abs().clamp(0.0, 1.0)
            rotation_deg = torch.rad2deg(2.0 * torch.acos(rotation_dot))[valid[:, :, 0]]
            rgb_delta = after.rgb_sh - before.rgb_sh
            density_delta = after.density_sh - before.density_sh
            opacity_delta = (after.confidence - before.confidence)[valid]
            prefix = f"refine{index}/update"
            result[f"{prefix}_depth_abs_mean"] = float(depth_delta.abs().mean().cpu()) if depth_delta.numel() else 0.0
            result[f"{prefix}_rotation_deg_mean"] = float(rotation_deg.mean().cpu()) if rotation_deg.numel() else 0.0
            result[f"{prefix}_log_scale_abs_mean"] = float(scale_delta.abs().mean().cpu()) if scale_delta.numel() else 0.0
            result[f"{prefix}_rgb_dc_abs_mean"] = float(rgb_delta[:, :, 0].abs().mean().cpu())
            result[f"{prefix}_rgb_ac_abs_mean"] = float(rgb_delta[:, :, 1:].abs().mean().cpu())
            result[f"{prefix}_density_dc_abs_mean"] = float(density_delta[:, :, 0].abs().mean().cpu())
            result[f"{prefix}_density_ac_abs_mean"] = float(density_delta[:, :, 1:].abs().mean().cpu())
            if depth_delta.numel() > 1 and opacity_delta.numel() == depth_delta.numel():
                centered_depth = depth_delta.abs() - depth_delta.abs().mean()
                centered_opacity = opacity_delta.abs() - opacity_delta.abs().mean()
                denominator = centered_depth.norm() * centered_opacity.norm()
                correlation = (centered_depth * centered_opacity).sum() / denominator.clamp_min(1.0e-8)
                result[f"{prefix}_depth_opacity_correlation"] = float(correlation.cpu())
            for source in range(after.num_source_views):
                source_valid = after.valid_mask[:, source].bool()
                confidence = after.confidence[:, source][source_valid]
                result[f"{refine_name}/source_{source}_mean_opacity"] = (
                    float(confidence.mean().cpu()) if confidence.numel() else 0.0
                )
    return result, rendered_snapshots


def _inference_snapshots(
    model: Stage3TrainableModel,
    ba: BlockSparseSphericalBA,
    renderer: Any,
    stage2: PerPixelGaussianObservation,
    features: torch.Tensor,
    images: torch.Tensor,
    cache: Stage3MatchCache,
    config: dict[str, Any],
) -> Stage3RefinementResult:
    snapshots: dict[str, PerPixelGaussianObservation] = {"initial": stage2.detach_parameters()}
    current = stage2
    hidden: torch.Tensor | None = None
    ba_outputs: list[Stage3BAOutput] = []
    reference = _detach_reference(model.encode_references(images))
    ba_schedule = _ba_outer_schedule(config)
    for iteration in range(3):
        if ba_schedule[iteration]:
            with torch.enable_grad():
                current, ba_output = _apply_ba(current, ba, cache)
            ba_outputs.append(ba_output)
            current = current.detach_parameters()
            snapshots[f"ba{iteration}"] = current
        feedback = render_leave_one_out_group(renderer, current)
        with torch.no_grad():
            refined = model(
                current,
                stage2,
                features,
                images,
                feedback,
                reference,
                iteration_index=iteration,
                hidden=hidden,
            )
        current = refined.observation.detach_parameters()
        hidden = refined.hidden.detach()
        snapshots[f"refine{iteration + 1}"] = current
    return Stage3RefinementResult(
        final_observation=current,
        initial_observation=stage2,
        snapshot_observations=snapshots,
        ba_outputs=tuple(ba_outputs),
        match_cache=cache,
        diagnostics={},
    )


def _validate(
    model: Stage3TrainableModel,
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
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None, torch.Tensor | None]:
    model.eval()
    aggregate: dict[str, float] = {}
    count = 0
    first_renders: dict[str, torch.Tensor] | None = None
    first_target: torch.Tensor | None = None
    for batch in loader:
        features, images, initial_depth, poses = extract_frozen_inputs(
            wrapper,
            adapter,  # type: ignore[arg-type]
            batch["images"],
            feature_device=feature_device,
            train_device=train_device,
            head_size=(int(config["image"]["head_height"]), int(config["image"]["head_width"])),
            feature_amp=bool(config["train"].get("amp", False)),
        )
        with torch.no_grad(), torch.amp.autocast(
            device_type=train_device.type,
            dtype=torch.bfloat16,
            enabled=bool(config["train"].get("amp", False)) and train_device.type == "cuda",
        ):
            stage2 = head(features, images, initial_depth, poses, frame_ids=batch["frame_ids"].to(train_device))
        cache = _build_match_cache(
            features,
            stage2.refined_depth,
            config,
            step=step + count,
            static_valid_mask=stage2.valid_mask,
        )
        refinement = _inference_snapshots(model, ba, renderer, stage2, features, images, cache, config)
        metrics, renders = _diagnostic_metrics(
            refinement.snapshot_observations,
            renderer,
            images,
            batch["gt_poses_c2w"].to(train_device),
            batch["gt_depths"].to(train_device),
            batch["gt_valid_depth"].to(train_device),
        )
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
        if first_renders is None:
            first_renders, first_target = renders, images.detach()
        count += 1
        if count >= int(config["train"].get("max_val_batches", 8)):
            break
    model.train()
    if count == 0:
        return {}, first_renders, first_target
    return {f"val/{key}": value / count for key, value in aggregate.items()}, first_renders, first_target


def train(config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("stage3", {}).get("enabled", False)):
        raise ValueError("Stage 3 is config-gated; set stage3.enabled=true.")
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
    expected_stage2_sha = config.get("stage2_checkpoint", {}).get("sha256")
    if expected_stage2_sha is not None and stage2_sha != str(expected_stage2_sha):
        raise ValueError(
            f"Stage 2 checkpoint SHA256 mismatch: expected {expected_stage2_sha}, got {stage2_sha}."
        )
    if stage2_path:
        load_stage2_checkpoint(stage2_path, head=head, expected_adapter_sha256=adapter_sha, map_location=train_device)
    elif bool(config.get("stage2_checkpoint", {}).get("required", False)):
        raise ValueError("stage2_checkpoint.path is required by this Stage 3 config.")

    trainable: nn.Module = Stage3TrainableModel(config).to(train_device)
    if distributed.enabled:
        trainable = DistributedDataParallel(trainable, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = _scheduler(optimizer, warmup_steps=int(train_cfg["warmup_steps"]), max_steps=int(train_cfg["max_steps"]))
    start_step = 0
    if train_cfg.get("resume"):
        start_step, _ = load_stage3_checkpoint(
            train_cfg["resume"],
            model=trainable,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_adapter_sha256=adapter_sha,
            expected_stage2_sha256=stage2_sha,
        )
    ba = build_ba(config)
    renderer = build_renderer(config)
    train_dataset = build_dataset(config, split="train")
    sampler = DistributedSampler(train_dataset, shuffle=True) if distributed.enabled else None
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage3_collate,
        pin_memory=train_device.type == "cuda",
    )
    val_loader: DataLoader | None = None
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
    step, micro = start_step, 0
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, float] = {}
    best_val_psnr = -float("inf")
    best_val_pose_ate = float("inf")
    epoch = 0
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        for batch in loader:
            images_full = batch["images"]
            features, images, initial_depth, poses = extract_frozen_inputs(
                wrapper,
                adapter,
                images_full,
                feature_device=feature_device,
                train_device=train_device,
                head_size=(int(config["image"]["head_height"]), int(config["image"]["head_width"])),
                feature_amp=bool(train_cfg.get("amp", False)),
            )
            frame_ids = batch["frame_ids"].to(train_device)
            with torch.no_grad(), torch.amp.autocast(
                device_type=train_device.type,
                dtype=torch.bfloat16,
                enabled=bool(train_cfg.get("amp", False)) and train_device.type == "cuda",
            ):
                stage2 = head(features, images, initial_depth, poses, frame_ids=frame_ids)
            matching_start = time.perf_counter()
            cache = _build_match_cache(
                features,
                stage2.refined_depth,
                config,
                step=step + micro,
                static_valid_mask=stage2.valid_mask,
            )
            if bool(config.get("refiner", {}).get("profile_synchronize_cuda", False)) and train_device.type == "cuda":
                torch.cuda.synchronize(train_device)
            matching_sec = time.perf_counter() - matching_start
            reference = _detach_reference(_unwrap(trainable).encode_references(images))
            with torch.amp.autocast(
                device_type=train_device.type,
                dtype=torch.bfloat16,
                enabled=bool(train_cfg.get("amp", False)) and train_device.type == "cuda",
            ):
                final, metrics, snapshots = _train_microbatch(
                    trainable,
                    ba,
                    renderer,
                    stage2,
                    features,
                    images,
                    cache,
                    reference,
                    config,
                    accumulation_scale=1.0 / accumulation,
                )
            metrics["profile/matching_sec"] = float(matching_sec)
            micro += 1
            if micro % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["train/step"] = float(step)
            diagnostics_interval = int(train_cfg.get("diagnostics_interval", 200))
            if step % max(1, diagnostics_interval) == 0:
                diagnostic_values, rendered_snapshots = _diagnostic_metrics(
                    snapshots,
                    renderer,
                    images,
                    batch["gt_poses_c2w"].to(train_device),
                    batch["gt_depths"].to(train_device),
                    batch["gt_valid_depth"].to(train_device),
                )
                metrics.update(diagnostic_values)
                if distributed.is_main and bool(config.get("Visualization", {}).get("enabled", False)):
                    visual_dir = output_dir / str(config["Visualization"].get("save_dir", "visualizations"))
                    visual_path = save_stage3_snapshot_panel(
                        visual_dir / f"step_{step:07d}.png",
                        target=images,
                        rendered_snapshots=rendered_snapshots,
                    )
                    if wandb_run is not None:
                        import wandb

                        wandb_run.log({"diagnostics/stage3_ba_refiner": wandb.Image(str(visual_path))}, step=step)
            last_metrics = metrics
            if distributed.is_main and step % max(1, int(train_cfg.get("log_interval", 1))) == 0:
                print(yaml.safe_dump({"step": step, "metrics": metrics}, sort_keys=False).strip(), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            val_interval = max(1, int(train_cfg.get("val_interval", 1000)))
            if step % val_interval == 0:
                if distributed.enabled:
                    dist.barrier()
                if distributed.is_main and val_loader is not None:
                    val_metrics, val_renders, val_target = _validate(
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
                    if val_metrics:
                        metrics["val/final_loo_psnr_delta"] = (
                            val_metrics.get("val/refine3/loo_psnr", 0.0)
                            - val_metrics.get("val/initial/loo_psnr", 0.0)
                        )
                        metrics["val/final_pose_ate_delta"] = (
                            val_metrics.get("val/refine3/pose_scale_aligned_ate", 0.0)
                            - val_metrics.get("val/initial/pose_scale_aligned_ate", 0.0)
                        )
                        if metrics["val/final_loo_psnr_delta"] > 0.0 and metrics["val/final_pose_ate_delta"] > 0.0:
                            print(
                                "WARNING: validation rendering improved while pose ATE worsened; "
                                "do not claim geometric refinement without the remaining geometry metrics.",
                                flush=True,
                            )
                        if val_metrics.get("val/refine3/confidence_p50", 1.0) < 0.01:
                            print("WARNING: final median confidence is below 0.01 (opacity collapse).", flush=True)
                    last_metrics = metrics
                    if wandb_run is not None and val_metrics:
                        wandb_run.log({key: value for key, value in metrics.items() if key.startswith("val/")}, step=step)
                    if val_renders is not None and val_target is not None and bool(config.get("Visualization", {}).get("enabled", False)):
                        val_path = save_stage3_snapshot_panel(
                            output_dir / str(config["Visualization"].get("save_dir", "visualizations")) / f"val_{step:07d}.png",
                            target=val_target,
                            rendered_snapshots=val_renders,
                        )
                        if wandb_run is not None:
                            import wandb

                            wandb_run.log({"validation/stage3_ba_refiner": wandb.Image(str(val_path))}, step=step)
                    final_psnr = val_metrics.get("val/refine3/loo_psnr")
                    final_ate = val_metrics.get("val/refine3/pose_scale_aligned_ate")
                    if final_psnr is not None and final_psnr > best_val_psnr:
                        best_val_psnr = final_psnr
                        save_stage3_checkpoint(
                            output_dir / "checkpoints" / "best_val_loo_psnr.pt",
                            model=trainable,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            step=step,
                            metrics=metrics,
                            adapter_sha256=adapter_sha,
                            stage2_checkpoint=stage2_path,
                            stage2_checkpoint_sha256=stage2_sha,
                        )
                    if final_ate is not None and final_ate < best_val_pose_ate:
                        best_val_pose_ate = final_ate
                        save_stage3_checkpoint(
                            output_dir / "checkpoints" / "best_val_pose_ate.pt",
                            model=trainable,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            step=step,
                            metrics=metrics,
                            adapter_sha256=adapter_sha,
                            stage2_checkpoint=stage2_path,
                            stage2_checkpoint_sha256=stage2_sha,
                        )
                if distributed.enabled:
                    dist.barrier()
            if distributed.is_main and step % max(1, int(train_cfg.get("save_interval", 1000))) == 0:
                save_stage3_checkpoint(
                    output_dir / "checkpoints" / "latest.pt",
                    model=trainable,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    step=step,
                    metrics=metrics,
                    adapter_sha256=adapter_sha,
                    stage2_checkpoint=stage2_path,
                    stage2_checkpoint_sha256=stage2_sha,
                )
            if step >= max_steps:
                break
        epoch += 1

    if distributed.is_main:
        save_stage3_checkpoint(
            output_dir / "checkpoints" / "latest.pt",
            model=trainable,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            metrics=last_metrics,
            adapter_sha256=adapter_sha,
            stage2_checkpoint=stage2_path,
            stage2_checkpoint_sha256=stage2_sha,
        )
    if wandb_run is not None:
        wandb_run.finish()
    if distributed.enabled:
        dist.barrier()
        dist.destroy_process_group()
    return {"step": step, "metrics": last_metrics, "checkpoint": str(output_dir / "checkpoints" / "latest.pt")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
