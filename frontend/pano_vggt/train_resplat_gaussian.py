"""Train Pano-ReSplat feed-forward Gaussian initializer/refiner."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.pano_droid.spherical_camera import latitude_area_weight
from frontend.pano_vggt.matching_dataset import build_matching_dataset_from_config, validate_training_sample
from frontend.pano_vggt.pano_resplat_feedback import PanoRenderFeedbackEncoder
from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.pano_resplat_point_decoder_init import INITIALIZER_TYPE, PanoVGGTPointDecoderGaussianInitializer
from frontend.pano_vggt.pano_resplat_refiner import PanoGaussianUpdateBlock, PanoGaussianUpdateLimits
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.resplat_types import PanoGaussianState, PanoRenderOutput, state_to_explicit_gaussian_set
from frontend.pano_vggt.train_gaussian import _build_prior_extractor
from frontend.pano_vggt.train_matching import _merge_config, matching_collate


def _default_config() -> dict[str, Any]:
    return {
        "Training": {
            "stage": "overfit",
            "steps": 5,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "frames_per_sample": 4,
            "context_views": 3,
            "target_views": 1,
            "window_mode": "fixed",
            "view_mode": "split",
            "render_height": None,
            "render_width": None,
            "train_min_refine": 0,
            "train_max_refine": 0,
            "eval_refine": 0,
            "amp": False,
            "seed": 1234,
            "output_dir": "outputs/pano_resplat/smoke_softsplat",
            "save_every": 100,
            "eval_every": 100,
            "vis_every": 100,
            "log_every": 1,
            "grad_clip": 1.0,
            "debug_overfit": False,
        },
        "Model": {
            "use_synthetic_features": True,
            "feature_dim": 16,
            "feature_stride": 4,
            "panovggt_repo": None,
            "panovggt_config": None,
            "panovggt_checkpoint": None,
            "class_path": None,
            "feature_hook": None,
            "feature_key": None,
            "image_size": None,
            "patch_size": 14,
            "patch_multiple": 14,
            "amp": True,
            "input_batch_dim": True,
            "strict_checkpoint": False,
            "skip_dinov2_pretrain": False,
        },
        "Initializer": {
            "type": INITIALIZER_TYPE,
            "state_dim": 32,
            "sh_degree": 0,
            "patch_size": 14,
            "decoder_embed_dim": 64,
            "decoder_depth": 2,
            "decoder_num_heads": 4,
            "decoder_mlp_ratio": 4.0,
            "init_scale": 0.02,
            "use_local_offsets": True,
            "max_offset_abs": 0.05,
            "max_offset_depth_ratio": 0.02,
        },
        "Feedback": {
            "type": "legacy",
            "feedback_dim": 32,
            "hidden_dim": 64,
            "error_dim": 64,
            "mv_down_factor": 4,
            "mv_attn_blocks": 1,
            "mv_num_heads": 4,
            "feature_backbone": "resnet18",
            "enable_group_correction": False,
            "group_rotation_deg": 1.0,
            "group_translation": 0.03,
            "group_translation_scale_ratio": 0.005,
        },
        "Refiner": {
            "hidden_dim": 64,
            "knn": 8,
            "num_heads": 4,
            "attn_proj_channels": None,
            "mlp_ratio": 2.0,
            "num_basic_refine_blocks": 1,
            "knn_backend": "cdist",
            "strict_knn_backend": False,
            "cache_knn": False,
            "detach_feedback": True,
            "gradient_checkpoint": False,
            "max_knn_points": 1024,
            "chunk_size": None,
            "limits": {
                "mean": 0.02,
                "log_scale": 0.05,
                "rotation": 0.05,
                "opacity": 0.25,
                "sh": 0.10,
                "latent": 0.10,
                "min_scale": 1.0e-5,
                "max_scale": 0.50,
            },
        },
        "Dataset": {
            "synthetic": True,
            "synthetic_variant": "complete",
            "synthetic_length": 4,
            "height": 32,
            "width": 64,
            "depth_format": "euclidean_range",
            "pose_coordinate_system": "ue_airsim",
            "validation_fraction": 0.0,
            "validation_split": "tail",
            "pair_sampling": "all",
            "pairs_per_sample": None,
            "class_map": {"sky_ids": [1], "classes": {"sky": 1}},
        },
        "Renderer": {
            "backend": "soft_splat",
            "allow_soft_splat_fallback": True,
            "soft_sigma_px": 1.25,
            "soft_max_points": 4096,
            "extra_gsplat360_roots": [],
        },
        "TrainingRender": {
            "panorama_render_mode": "pfgs360_gsplat",
            "pfgs360_render_mode": "RGB+ED",
            "pfgs360_rasterize_mode": "antialiased",
            "pfgs360_packed": False,
            "pfgs360_near_plane": 0.01,
            "pfgs360_far_plane": 100000.0,
            "pfgs360_absgrad": True,
            "pfgs360_distloss": False,
        },
        "Loss": {
            "rgb_l1_weight": 1.0,
            "dssim_weight": 0.1,
            "lpips_weight": 0.0,
            "depth_weight": 0.05,
            "context_weight": 0.0,
            "opacity_reg_weight": 0.001,
            "alpha_coverage_weight": 0.01,
            "scale_reg_weight": 0.001,
            "anisotropy_reg_weight": 0.001,
            "delta_reg_weight": 0.01,
            "mean_step_reg_weight": 0.01,
            "sh_reg_weight": 0.0005,
            "intermediate_weight": 0.5,
            "type": "default",
        },
        "Optimizer": {
            "lr": 2.0e-4,
            "initializer_lr": 5.0e-5,
            "refiner_lr": 2.0e-4,
            "weight_decay": 0.01,
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": "pano_resplat_gaussian",
            "mode": "disabled",
            "log_every": 10,
            "tags": ["pano-resplat", "gaussian"],
        },
        "Validation": {"enabled": False, "max_batches": 1},
        "Checks": {
            "target_leakage_check": True,
            "renderer_gradient_check": True,
            "nan_check": True,
        },
    }


def load_resplat_train_config(path: str | None) -> dict[str, Any]:
    config = _default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _merge_config(config, user)


class _Logger:
    def __init__(self, output_dir: Path, *, enabled: bool = True, rank: int = 0) -> None:
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.stdout_path = output_dir / "logs" / "stdout.log"
        self.stderr_path = output_dir / "logs" / "stderr.log"
        if not self.enabled:
            return
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_path.write_text("", encoding="utf-8")
        self.stderr_path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        print(message, flush=True)
        with self.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def error(self, message: str) -> None:
        if not self.enabled:
            return
        print(message, file=sys.stderr, flush=True)
        with self.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def _init_wandb(config: dict[str, Any], output_dir: Path):
    wb = config.get("WeightsAndBiases", {})
    mode = str(wb.get("mode", "disabled")).lower()
    if not bool(wb.get("enabled", False)) or mode == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:
        warnings.warn(f"wandb unavailable, disabling logging: {exc}")
        return None
    return wandb.init(
        project=wb.get("project", "360Droid-splat"),
        entity=wb.get("entity"),
        name=wb.get("run_name", "pano_resplat_gaussian"),
        mode=mode,
        tags=list(wb.get("tags", [])),
        config=config,
        dir=str(output_dir),
    )


def _build_lpips_model(config: dict[str, Any], device: torch.device, logger: _Logger | None = None) -> nn.Module | None:
    weight = float(config.get("Loss", {}).get("lpips_weight", 0.0))
    if weight <= 0.0:
        return None
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="vgg").to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model
    except Exception as exc:
        message = f"LPIPS requested but unavailable; disabling lpips loss: {exc}"
        warnings.warn(message)
        if logger is not None:
            logger.error(message)
        config.setdefault("Loss", {})["lpips_weight"] = 0.0
        return None


def _device_from_arg(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _distributed_state() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def _init_distributed_if_needed(state: dict[str, int | bool]) -> None:
    if not bool(state["distributed"]):
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(state["local_rank"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier_if_distributed(state: dict[str, int | bool]) -> None:
    if bool(state["distributed"]) and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _unwrap_frontend(frontend: PanoReSplatFrontend) -> PanoReSplatFrontend:
    if isinstance(frontend, DistributedDataParallel):
        module = frontend.module
    else:
        module = frontend
    if not isinstance(module, PanoReSplatFrontend):
        raise TypeError(f"Expected PanoReSplatFrontend, got {type(module)!r}")
    return module


def _wrap_trainable_modules_for_ddp(frontend: PanoReSplatFrontend, state: dict[str, int | bool]) -> PanoReSplatFrontend:
    if not bool(state["distributed"]):
        return frontend
    device_ids = [int(state["local_rank"])] if torch.cuda.is_available() else None
    for name in ("initializer", "feedback_encoder", "update_block"):
        module = getattr(frontend, name)
        has_trainable = any(param.requires_grad for param in module.parameters())
        if has_trainable:
            setattr(
                frontend,
                name,
                DistributedDataParallel(
                    module,
                    device_ids=device_ids,
                    output_device=int(state["local_rank"]) if torch.cuda.is_available() else None,
                    find_unused_parameters=False,
                ),
            )
    return frontend


def _build_frontend(config: dict[str, Any], *, device: torch.device) -> PanoReSplatFrontend:
    init_cfg = dict(config.get("Initializer", {}))
    initializer = PanoVGGTPointDecoderGaussianInitializer(init_cfg)
    feedback_cfg = config.get("Feedback", {})
    feedback = PanoRenderFeedbackEncoder(
        feedback_dim=int(feedback_cfg.get("feedback_dim", 32)),
        hidden_dim=int(feedback_cfg.get("hidden_dim", 64)),
        feedback_type=str(feedback_cfg.get("type", feedback_cfg.get("feedback_type", "legacy"))),
        error_dim=int(feedback_cfg.get("error_dim", 64)),
        mv_down_factor=int(feedback_cfg.get("mv_down_factor", 4)),
        mv_attn_blocks=int(feedback_cfg.get("mv_attn_blocks", 1)),
        mv_num_heads=int(feedback_cfg.get("mv_num_heads", 4)),
        feature_backbone=str(feedback_cfg.get("feature_backbone", "resnet18")),
        enable_group_correction=bool(feedback_cfg.get("enable_group_correction", False)),
        group_rotation_deg=float(feedback_cfg.get("group_rotation_deg", 1.0)),
        group_translation=float(feedback_cfg.get("group_translation", 0.03)),
        group_translation_scale_ratio=float(feedback_cfg.get("group_translation_scale_ratio", 0.005)),
    )
    ref_cfg = config.get("Refiner", {})
    limits = PanoGaussianUpdateLimits(**dict(ref_cfg.get("limits", {})))
    update = PanoGaussianUpdateBlock(
        feedback_dim=int(feedback_cfg.get("feedback_dim", 32)),
        latent_dim=int(init_cfg.get("state_dim", 64)),
        sh_dim=(int(init_cfg.get("sh_degree", 0)) + 1) ** 2,
        hidden_dim=int(ref_cfg.get("hidden_dim", 64)),
        knn=int(ref_cfg.get("knn", 8)),
        num_heads=int(ref_cfg.get("num_heads", 4)),
        limits=limits,
        max_knn_points=int(ref_cfg.get("max_knn_points", 1024)),
        chunk_size=ref_cfg.get("chunk_size"),
        attn_proj_channels=ref_cfg.get("attn_proj_channels"),
        mlp_ratio=float(ref_cfg.get("mlp_ratio", 2.0)),
        num_basic_refine_blocks=int(ref_cfg.get("num_basic_refine_blocks", 1)),
        knn_backend=str(ref_cfg.get("knn_backend", "cdist")),
        strict_knn_backend=bool(ref_cfg.get("strict_knn_backend", False)),
        cache_knn=bool(ref_cfg.get("cache_knn", False)),
        detach_feedback=bool(ref_cfg.get("detach_feedback", True)),
        gradient_checkpoint=bool(ref_cfg.get("gradient_checkpoint", False)),
    )
    render_cfg = config.get("Renderer", {})
    render_config = {"Training": dict(config.get("TrainingRender", {})), "Renderer": dict(render_cfg)}
    renderer = PanoGaussianRendererAdapter(
        config=render_config,
        extra_gsplat360_roots=list(render_cfg.get("extra_gsplat360_roots", [])),
        allow_soft_splat_fallback=bool(render_cfg.get("allow_soft_splat_fallback", True)),
        soft_sigma_px=float(render_cfg.get("soft_sigma_px", 1.25)),
        soft_max_points=int(render_cfg.get("soft_max_points", 4096)),
    )
    frontend = PanoReSplatFrontend(
        initializer=initializer,
        feedback_encoder=feedback,
        update_block=update,
        renderer=renderer,
        renderer_backend=str(render_cfg.get("backend", "soft_splat")),
    )
    return frontend.to(device)


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for param in module.parameters():
        if isinstance(param, UninitializedParameter):
            param.requires_grad = bool(value)
        else:
            param.requires_grad_(bool(value))


def _set_stage_trainability(frontend: PanoReSplatFrontend, stage: str, *, overfit_trains_refiner: bool = False) -> None:
    _set_requires_grad(frontend, False)
    if stage in {"init", "overfit", "joint"}:
        _set_requires_grad(frontend.initializer, True)
    if stage in {"refine", "joint"} or (stage == "overfit" and overfit_trains_refiner):
        _set_requires_grad(frontend.feedback_encoder, True)
        _set_requires_grad(frontend.update_block, True)
        feature_extractor = getattr(frontend.feedback_encoder, "feature_extractor", None)
        if isinstance(feature_extractor, nn.Module):
            _set_requires_grad(feature_extractor, False)
            feature_extractor.eval()


def _optimizer(frontend: PanoReSplatFrontend, config: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    opt_cfg = config.get("Optimizer", {})
    wd = float(opt_cfg.get("weight_decay", 0.01))
    if stage == "joint":
        groups = [
            {"params": [p for p in frontend.initializer.parameters() if p.requires_grad], "lr": float(opt_cfg.get("initializer_lr", 5.0e-5)), "name": "initializer"},
            {"params": [p for p in list(frontend.feedback_encoder.parameters()) + list(frontend.update_block.parameters()) if p.requires_grad], "lr": float(opt_cfg.get("refiner_lr", 2.0e-4)), "name": "refiner"},
        ]
        return torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=wd)
    params = [p for p in frontend.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(opt_cfg.get("lr", 2.0e-4)), weight_decay=wd)


def _load_compatible_state(module: nn.Module, state: dict[str, Any], *, strict: bool = False) -> list[str]:
    current = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        key_str = str(key).removeprefix("module.")
        if key_str not in current:
            skipped.append(key_str)
            continue
        if isinstance(value, UninitializedParameter) or isinstance(current[key_str], UninitializedParameter):
            skipped.append(key_str)
            continue
        if torch.is_tensor(value) and torch.is_tensor(current[key_str]) and tuple(value.shape) != tuple(current[key_str].shape):
            skipped.append(key_str)
            continue
        compatible[key_str] = value
    module.load_state_dict(compatible, strict=strict)
    return skipped


def _incompatible_state_keys(module: nn.Module, state: dict[str, Any]) -> list[str]:
    current = module.state_dict()
    skipped = []
    for key, value in state.items():
        key_str = str(key).removeprefix("module.")
        if key_str not in current:
            skipped.append(key_str)
            continue
        if isinstance(value, UninitializedParameter) or isinstance(current[key_str], UninitializedParameter):
            skipped.append(key_str)
            continue
        if torch.is_tensor(value) and torch.is_tensor(current[key_str]) and tuple(value.shape) != tuple(current[key_str].shape):
            skipped.append(key_str)
    return skipped


def _torch_load_checkpoint(path: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}")
    return payload


def _load_checkpoint(frontend: PanoReSplatFrontend, path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _torch_load_checkpoint(path)
    frontend = _unwrap_frontend(frontend)
    expected_type = getattr(_unwrap_module(frontend.initializer), "initializer_type", INITIALIZER_TYPE)
    found_type = payload.get("initializer_type")
    if found_type != expected_type:
        raise ValueError(
            f"Checkpoint {path} has initializer_type={found_type!r}; expected {expected_type!r}. "
            "Old lightweight initializer checkpoints are intentionally incompatible."
        )
    if int(payload.get("state_dim", getattr(frontend.initializer, "state_dim", -1))) != int(getattr(frontend.initializer, "state_dim", -2)):
        raise ValueError("Checkpoint state_dim does not match current Initializer.state_dim.")
    if int(payload.get("sh_degree", getattr(frontend.initializer, "sh_degree", -1))) != int(getattr(frontend.initializer, "sh_degree", -2)):
        raise ValueError("Checkpoint sh_degree does not match current Initializer.sh_degree.")
    skipped: dict[str, list[str]] = {}
    if "initializer" in payload:
        skipped["initializer"] = _load_compatible_state(_unwrap_module(frontend.initializer), payload["initializer"], strict=False)
    if "feedback_encoder" in payload:
        skipped["feedback_encoder"] = _load_compatible_state(_unwrap_module(frontend.feedback_encoder), payload["feedback_encoder"], strict=False)
    if "update_block" in payload:
        update_block = _unwrap_module(frontend.update_block)
        incompatible = _incompatible_state_keys(update_block, payload["update_block"])
        if incompatible:
            skipped["update_block"] = sorted({str(key).removeprefix("module.") for key in payload["update_block"].keys()})
        else:
            skipped["update_block"] = _load_compatible_state(update_block, payload["update_block"], strict=False)
    payload["_skipped_incompatible_keys"] = {key: value for key, value in skipped.items() if value}
    return payload


def _checkpoint_payload(
    frontend: PanoReSplatFrontend,
    *,
    config: dict[str, Any],
    step: int,
    stage: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    frontend = _unwrap_frontend(frontend)
    initializer = _unwrap_module(frontend.initializer)
    return {
        "format": "pano_resplat_gaussian_v1",
        "initializer_type": getattr(initializer, "initializer_type", INITIALIZER_TYPE),
        "initializer_config": getattr(initializer, "initializer_config", dict(config.get("Initializer", {}))),
        "panovggt_checkpoint": config.get("Model", {}).get("panovggt_checkpoint"),
        "state_dim": int(getattr(initializer, "state_dim", config.get("Initializer", {}).get("state_dim", 0))),
        "sh_degree": int(getattr(initializer, "sh_degree", config.get("Initializer", {}).get("sh_degree", 0))),
        "sh_dim": int(getattr(initializer, "sh_dim", (int(config.get("Initializer", {}).get("sh_degree", 0)) + 1) ** 2)),
        "stage": stage,
        "step": int(step),
        "initializer": initializer.state_dict(),
        "feedback_encoder": _unwrap_module(frontend.feedback_encoder).state_dict(),
        "update_block": _unwrap_module(frontend.update_block).state_dict(),
        "config": config,
        "metrics": metrics,
    }


def _save_checkpoint(path: Path, frontend: PanoReSplatFrontend, config: dict[str, Any], step: int, stage: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(frontend, config=config, step=step, stage=stage, metrics=metrics), path)


def _select_optional_frames(values: Any, indices: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(values):
        return None
    return values.index_select(1, indices)


def _supervision_tensor(sample: dict[str, Any], supervision_key: str, fallback_key: str) -> torch.Tensor | None:
    value = sample.get(supervision_key)
    if torch.is_tensor(value):
        return value
    fallback = sample.get(fallback_key)
    return fallback if torch.is_tensor(fallback) else None


def _target_supervision(
    sample: dict[str, Any],
    priors: dict[str, torch.Tensor],
    *,
    indices: torch.Tensor | None,
    render_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    images = sample.get("supervision_images")
    has_native_supervision = torch.is_tensor(images)
    if not has_native_supervision:
        images = sample.get("images")
    if images is None:
        raise ValueError("Training target requires sample images.")
    input_images = sample.get("images")
    if (
        not has_native_supervision
        and torch.is_tensor(input_images)
        and tuple(int(x) for x in input_images.shape[-2:]) != tuple(int(x) for x in render_hw)
    ):
        raise ValueError(
            "render_hw differs from input image size but sample['supervision_images'] is missing. "
            "Enable Dataset.use_native_supervision_gt with supervision_resize_height/width so loss is "
            "computed against native high-resolution GT instead of upsampled model input."
        )
    depths = _supervision_tensor(sample, "supervision_depths", "depths")
    valid_depth = _supervision_tensor(sample, "supervision_valid_depth", "valid_depth")
    sky_mask = _supervision_tensor(sample, "supervision_sky_mask", "sky_mask")
    poses = priors["poses_c2w"]
    if indices is not None:
        images = images.index_select(1, indices)
        depths = _select_optional_frames(depths, indices)
        valid_depth = _select_optional_frames(valid_depth, indices)
        sky_mask = _select_optional_frames(sky_mask, indices)
        poses = poses.index_select(1, indices)
    if depths is None:
        depths = priors["depth"] if indices is None else priors["depth"].index_select(1, indices)
    target: dict[str, torch.Tensor] = {
        "images": _resize_5d(images.float(), render_hw, mode="bilinear"),
        "poses_c2w": poses.float(),
    }
    target_depths = _resize_5d(depths.float(), render_hw, mode="nearest") if depths is not None else None
    if target_depths is not None:
        target["depths"] = target_depths
    target_sky = _resize_5d(sky_mask, render_hw, mode="nearest", is_mask=True)
    target_valid = _compose_valid_mask(valid_depth=valid_depth, sky_mask=sky_mask)
    target_valid = _resize_5d(target_valid, render_hw, mode="nearest", is_mask=True) if target_valid is not None else None
    if target_sky is not None:
        target["sky_mask"] = target_sky
    if target_valid is not None:
        target["valid_mask"] = target_valid
    return target


def _as_5d_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.ndim != 5 or int(mask.shape[2]) != 1:
        raise ValueError(f"{name} must have shape BxVxHxW or BxVx1xHxW, got {tuple(mask.shape)}")
    return mask.bool()


def _compose_valid_mask(
    *,
    valid_depth: torch.Tensor | None,
    sky_mask: torch.Tensor | None,
    world_points: torch.Tensor | None = None,
) -> torch.Tensor | None:
    valid = None if valid_depth is None else _as_5d_mask(valid_depth, name="valid_depth")
    if sky_mask is not None:
        non_sky = ~_as_5d_mask(sky_mask, name="sky_mask")
        valid = non_sky if valid is None else valid & non_sky
    if world_points is not None:
        finite_world = torch.isfinite(world_points).all(dim=-1).unsqueeze(2)
        valid = finite_world if valid is None else valid & finite_world
    return valid


def _is_input_reconstruction(config: dict[str, Any]) -> bool:
    tr = config.get("Training", {})
    init_cfg = config.get("Initializer", {})
    mode = str(tr.get("view_mode", "split")).lower()
    position_mode = str(init_cfg.get("position_mode", "")).lower()
    init_type = str(init_cfg.get("type", "")).lower()
    return (
        mode in {"input_reconstruction", "all_views", "input_views"}
        or position_mode == "panovggt_aligned"
        or init_type in {"panovggt_aligned", "pano_vggt_aligned", INITIALIZER_TYPE}
    )


def _render_hw(config: dict[str, Any], fallback_hw: tuple[int, int]) -> tuple[int, int]:
    tr = config.get("Training", {})
    height = tr.get("render_height")
    width = tr.get("render_width")
    if height is None or width is None:
        renderer = config.get("Renderer", {})
        height = renderer.get("render_height", height)
        width = renderer.get("render_width", width)
    if height is None or width is None:
        return int(fallback_hw[0]), int(fallback_hw[1])
    return int(height), int(width)


def _resize_5d(values: torch.Tensor | None, image_hw: tuple[int, int], *, mode: str, is_mask: bool = False) -> torch.Tensor | None:
    if not torch.is_tensor(values):
        return None
    tensor = values
    squeeze_channel = False
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(2)
        squeeze_channel = True
    if tensor.ndim != 5:
        raise ValueError(f"Expected BxVxCxHxW or BxVxHxW tensor, got {tuple(values.shape)}")
    if tuple(int(x) for x in tensor.shape[-2:]) == tuple(int(x) for x in image_hw):
        return values
    b, v, c, h, w = [int(x) for x in tensor.shape]
    flat = tensor.reshape(b * v, c, h, w)
    if is_mask:
        resized = F.interpolate(flat.float(), size=image_hw, mode="nearest") > 0.5
    else:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        resized = F.interpolate(flat.float(), size=image_hw, mode=mode, **kwargs).to(dtype=values.dtype)
    resized = resized.reshape(b, v, c, int(image_hw[0]), int(image_hw[1]))
    if squeeze_channel:
        resized = resized.squeeze(2)
    return resized


def _sample_input_reconstruction(
    sample: dict[str, Any],
    priors: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Use every input frame as both Gaussian source and supervised render view."""

    n = int(sample["images"].shape[1])
    device = sample["images"].device
    view_idx = torch.arange(n, device=device)
    context_valid = _compose_valid_mask(
        valid_depth=sample.get("valid_depth"),
        sky_mask=sample.get("sky_mask"),
        world_points=priors.get("world_points"),
    )
    context = {
        "images": sample["images"].float(),
        "features": priors["features"].float(),
        "depths": priors["depth"].float(),
        "poses_c2w": priors["poses_c2w"].float(),
        "world_points": priors["world_points"].float(),
        "view_indices": view_idx,
    }
    if torch.is_tensor(priors.get("tokens")):
        context["tokens"] = priors["tokens"].float()
    if torch.is_tensor(priors.get("token_hw")):
        context["token_hw"] = priors["token_hw"]
    if torch.is_tensor(sample.get("sky_mask")):
        context["sky_mask"] = sample["sky_mask"]
    if context_valid is not None:
        context["valid_mask"] = context_valid

    render_hw = _render_hw(config, tuple(int(x) for x in sample["images"].shape[-2:]))
    target = _target_supervision(sample, priors, indices=None, render_hw=render_hw)
    target["view_indices"] = view_idx
    return context, target


def _sample_window(
    sample: dict[str, Any],
    priors: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tr = config.get("Training", {})
    b, n = int(sample["images"].shape[0]), int(sample["images"].shape[1])
    index_device = sample["images"].device
    vc = min(max(1, int(tr.get("context_views", 3))), max(1, n - 1))
    vt = min(max(1, int(tr.get("target_views", 1))), max(1, n - vc))
    mode = str(tr.get("window_mode", "fixed")).lower()
    if bool(tr.get("debug_overfit", False)) or mode == "fixed" or (mode != "random_split" and n <= vc + vt):
        context_idx = torch.arange(vc, device=index_device)
        target_idx = torch.arange(vc, min(vc + vt, n), device=index_device)
        if target_idx.numel() < vt:
            target_idx = torch.full((vt,), n - 1, dtype=torch.long, device=index_device)
    elif mode == "random_split":
        order = torch.randperm(n, device=index_device)
        context_idx = order[:vc]
        target_idx = order[vc : vc + vt]
    else:
        max_start = max(0, n - vc - vt)
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        context_idx = torch.arange(start, start + vc, device=index_device)
        target_idx = torch.arange(start + vc, start + vc + vt, device=index_device)
    context = {
        "images": sample["images"].index_select(1, context_idx).float(),
        "features": priors["features"].index_select(1, context_idx).float(),
        "depths": priors["depth"].index_select(1, context_idx).float(),
        "poses_c2w": priors["poses_c2w"].index_select(1, context_idx).float(),
        "world_points": priors["world_points"].index_select(1, context_idx).float(),
        "view_indices": context_idx,
    }
    if torch.is_tensor(priors.get("tokens")):
        context["tokens"] = priors["tokens"].index_select(1, context_idx).float()
    if torch.is_tensor(priors.get("token_hw")):
        context["token_hw"] = priors["token_hw"]
    context_sky = _select_optional_frames(sample.get("sky_mask"), context_idx)
    if context_sky is not None:
        context["sky_mask"] = context_sky
    context_valid = _compose_valid_mask(
        valid_depth=_select_optional_frames(sample.get("valid_depth"), context_idx),
        sky_mask=context_sky,
        world_points=context["world_points"],
    )
    if context_valid is not None:
        context["valid_mask"] = context_valid
    render_hw = _render_hw(config, tuple(int(x) for x in sample["images"].shape[-2:]))
    target = _target_supervision(sample, priors, indices=target_idx, render_hw=render_hw)
    target["view_indices"] = target_idx
    return context, target


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _render_views(
    frontend: PanoReSplatFrontend,
    state: PanoGaussianState,
    poses: torch.Tensor,
    image_hw: tuple[int, int],
) -> PanoRenderOutput:
    if poses.ndim == 3:
        return frontend.renderer.render_state(state, poses, image_hw, renderer_backend=frontend.renderer_backend)
    colors, depths, alphas, packages, backends = [], [], [], [], []
    for idx in range(int(poses.shape[1])):
        out = frontend.renderer.render_state(state, poses[:, idx], image_hw, renderer_backend=frontend.renderer_backend)
        colors.append(out.color)
        depths.append(out.depth)
        alphas.append(out.alpha)
        packages.append(out.extras.get("packages"))
        backends.append(out.extras.get("backend"))
    return PanoRenderOutput(
        color=torch.stack(colors, dim=1),
        depth=torch.stack(depths, dim=1),
        alpha=torch.stack(alphas, dim=1),
        extras={"packages": packages, "backend": backends},
    )


def _broadcast_mask(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    out = _as_5d_mask(mask, name="loss_mask").to(device=values.device)
    while out.ndim < values.ndim:
        out = out.unsqueeze(-1)
    if out.shape[-2:] != values.shape[-2:]:
        flat = out.float().reshape(-1, 1, int(out.shape[-2]), int(out.shape[-1]))
        flat = F.interpolate(flat, size=tuple(int(x) for x in values.shape[-2:]), mode="nearest")
        out = flat.reshape(*out.shape[:-2], *values.shape[-2:]) > 0.5
    return out.expand_as(values).bool()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, *, default: float = 0.0) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_b = _broadcast_mask(mask, values)
    if mask_b is None or not bool(mask_b.any()):
        return values.new_tensor(float(default))
    return values[mask_b].mean()


def _gaussian_masked_mean(values: torch.Tensor, mask: torch.Tensor | None, *, default: float = 0.0) -> torch.Tensor:
    if mask is None:
        return values.mean()
    valid = mask.to(device=values.device, dtype=torch.bool)
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(values)
    if not bool(valid.any()):
        return values.new_tensor(float(default))
    return values[valid].mean()


def _latitude_weighted_image_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError("latitude weighted image loss expects BxVx3xHxW tensors")
    err = (pred - target).abs().mean(dim=2, keepdim=True)
    h, w = int(pred.shape[-2]), int(pred.shape[-1])
    lat = latitude_area_weight(h, w, device=pred.device, dtype=pred.dtype, normalize=False).view(1, 1, 1, h, w)
    if mask is None:
        weight = lat.expand_as(err)
    else:
        valid = _as_5d_mask(mask, name="loss_mask").to(device=pred.device)
        if valid.shape[-2:] != pred.shape[-2:]:
            flat = valid.float().reshape(-1, 1, int(valid.shape[-2]), int(valid.shape[-1]))
            flat = F.interpolate(flat, size=pred.shape[-2:], mode="nearest")
            valid = flat.reshape(*valid.shape[:-2], *pred.shape[-2:]) > 0.5
        weight = lat * valid.to(dtype=pred.dtype)
    denom = weight.sum().clamp_min(torch.finfo(pred.dtype).eps)
    return (err * weight).sum() / denom


def _clamp_state_min_scale(state: PanoGaussianState, min_scale: float | None) -> PanoGaussianState:
    if min_scale is None or float(min_scale) <= 0.0:
        return state
    min_log_scale = math.log(max(float(min_scale), 1.0e-12))
    return PanoGaussianState(
        means=state.means,
        log_scales=torch.nan_to_num(state.log_scales, nan=float(min_log_scale), posinf=0.0, neginf=float(min_log_scale)).clamp_min(float(min_log_scale)),
        rotations_unnorm=state.rotations_unnorm,
        opacity_logits=state.opacity_logits,
        sh_coeffs=state.sh_coeffs,
        latent_features=state.latent_features,
        source_view_ids=state.source_view_ids,
        source_uv=state.source_uv,
        valid_mask=state.valid_mask,
        confidence=state.confidence,
    )


def _configured_min_scale(config: dict[str, Any]) -> float | None:
    loss_cfg = config.get("Loss", {})
    if "min_scale" in loss_cfg:
        return float(loss_cfg["min_scale"])
    ref_limits = config.get("Refiner", {}).get("limits", {})
    if "min_scale" in ref_limits:
        return float(ref_limits["min_scale"])
    return None


def _ssim_dssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.reshape(-1, 3, pred.shape[-2], pred.shape[-1])
    target_f = target.reshape(-1, 3, target.shape[-2], target.shape[-1])
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred_f, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target_f, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred_f * pred_f, 3, stride=1, padding=1) - mu_x.square()
    sigma_y = F.avg_pool2d(target_f * target_f, 3, stride=1, padding=1) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred_f * target_f, 3, stride=1, padding=1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)).clamp_min(1.0e-6)
    return ((1.0 - ssim.clamp(-1.0, 1.0)) * 0.5).mean()


def _masked_ssim_dssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return _ssim_dssim(pred, target)
    mask_b = _broadcast_mask(mask, pred)
    if mask_b is None or not bool(mask_b.any()):
        return pred.new_tensor(0.0)
    mask_f = mask_b.to(dtype=pred.dtype)
    return _ssim_dssim(pred * mask_f, target * mask_f)


def _psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mse = _masked_mean((pred - target).square(), mask).clamp_min(1.0e-8)
    return -10.0 * torch.log10(mse)


def _gaussian_stats(state: PanoGaussianState) -> dict[str, torch.Tensor]:
    explicit = state_to_explicit_gaussian_set(state, 0)
    scale = explicit.get_scaling
    opacity = explicit.get_opacity
    rotation_norm = torch.linalg.norm(explicit.get_rotation, dim=-1) if explicit.get_rotation.numel() else state.means.new_zeros(1)
    return {
        "scale_min": scale.min().detach() if scale.numel() else state.means.new_tensor(0.0),
        "scale_max": scale.max().detach() if scale.numel() else state.means.new_tensor(0.0),
        "scale_mean": scale.mean().detach() if scale.numel() else state.means.new_tensor(0.0),
        "opacity_min": opacity.min().detach() if opacity.numel() else state.means.new_tensor(0.0),
        "opacity_max": opacity.max().detach() if opacity.numel() else state.means.new_tensor(0.0),
        "opacity_mean": opacity.mean().detach() if opacity.numel() else state.means.new_tensor(0.0),
        "means_norm": state.means.norm(dim=-1).mean().detach(),
        "rotation_norm": rotation_norm.mean().detach(),
        "gaussian_count": state.means.new_tensor(float(explicit.get_xyz.shape[0])),
    }


def _lpips_resize_hw(loss_cfg: dict[str, Any], image_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = image_hw
    resize_h = loss_cfg.get("lpips_resize_height", loss_cfg.get("lpips_height"))
    resize_w = loss_cfg.get("lpips_resize_width", loss_cfg.get("lpips_width"))
    max_side = loss_cfg.get("lpips_max_side")
    if resize_h is None and resize_w is None and max_side is None:
        return height, width
    if max_side is not None and (resize_h is None or resize_w is None):
        max_side_i = max(1, int(max_side))
        scale = min(1.0, float(max_side_i) / float(max(height, width)))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    if resize_h is None:
        resize_w_i = max(1, int(resize_w))
        resize_h_i = max(1, int(round(height * resize_w_i / max(1, width))))
        return resize_h_i, resize_w_i
    if resize_w is None:
        resize_h_i = max(1, int(resize_h))
        resize_w_i = max(1, int(round(width * resize_h_i / max(1, height))))
        return resize_h_i, resize_w_i
    return max(1, int(resize_h)), max(1, int(resize_w))


def _prepare_lpips_inputs(
    pred: torch.Tensor,
    gt: torch.Tensor,
    target_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = pred.shape[-2:]
    pred_f = pred.reshape(-1, 3, height, width).clamp(0.0, 1.0)
    gt_f = gt.reshape(-1, 3, height, width).clamp(0.0, 1.0)
    mask_f = None
    if torch.is_tensor(target_mask):
        mask_5d = _as_5d_mask(target_mask, name="lpips_mask").to(device=pred.device)
        if mask_5d.shape[-2:] != (height, width):
            flat = mask_5d.float().reshape(-1, 1, int(mask_5d.shape[-2]), int(mask_5d.shape[-1]))
            flat = F.interpolate(flat, size=(height, width), mode="nearest")
            mask_5d = flat.reshape(*mask_5d.shape[:-2], height, width) > 0.5
        mask_f = mask_5d.reshape(-1, 1, height, width).to(dtype=pred.dtype)
    lpips_h, lpips_w = _lpips_resize_hw(loss_cfg, (height, width))
    if (lpips_h, lpips_w) != (height, width):
        pred_f = F.interpolate(pred_f, size=(lpips_h, lpips_w), mode="bilinear", align_corners=False)
        gt_f = F.interpolate(gt_f, size=(lpips_h, lpips_w), mode="bilinear", align_corners=False)
        if mask_f is not None:
            mask_f = F.interpolate(mask_f, size=(lpips_h, lpips_w), mode="nearest")
    if mask_f is not None:
        # Avoid feeding large all-zero invalid regions into LPIPS' feature normalization.
        # Invalid pixels contribute no difference because pred is filled with GT there.
        pred_f = pred_f * mask_f + gt_f.detach() * (1.0 - mask_f)
    return pred_f * 2.0 - 1.0, gt_f * 2.0 - 1.0


def _single_output_loss(
    state: PanoGaussianState,
    target_render: PanoRenderOutput,
    target: dict[str, torch.Tensor],
    *,
    context_render: PanoRenderOutput | None,
    context: dict[str, torch.Tensor],
    prev_state: PanoGaussianState | None,
    config: dict[str, Any],
    lpips_model: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = config.get("Loss", {})
    loss_type = str(loss_cfg.get("type", loss_cfg.get("loss_type", "default"))).lower()
    pred = target_render.color
    gt = target["images"].to(pred).clamp(0.0, 1.0)
    target_mask = target.get("valid_mask") if torch.is_tensor(target.get("valid_mask")) else None
    latitude_l1 = _latitude_weighted_image_l1(pred, gt, target_mask)
    rgb_l1 = latitude_l1 if loss_type in {"image_lpips_latitude", "resplat_image_lpips_latitude"} else _masked_mean((pred - gt).abs(), target_mask)
    dssim = _masked_ssim_dssim(pred, gt, target_mask)
    lpips_loss = pred.new_tensor(0.0)
    if lpips_model is not None:
        pred_f, gt_f = _prepare_lpips_inputs(pred, gt, target_mask, loss_cfg)
        lpips_loss = lpips_model(pred_f, gt_f).mean()
    depth_loss = pred.new_tensor(0.0)
    if torch.is_tensor(target.get("depths")) and torch.is_tensor(target_render.depth):
        td = target["depths"].to(pred)
        rd = target_render.depth.to(pred)
        mask = torch.isfinite(td) & (td > 0.0)
        if torch.is_tensor(target.get("valid_mask")):
            mask = mask & target["valid_mask"].to(device=pred.device).bool()
        if bool(mask.any()):
            depth_loss = ((rd - td).abs() / td.abs().clamp_min(1.0))[mask].mean()
    context_l1 = pred.new_tensor(0.0)
    if context_render is not None:
        context_gt = context["images"].to(pred).clamp(0.0, 1.0)
        context_mask = context.get("valid_mask") if torch.is_tensor(context.get("valid_mask")) else None
        context_l1 = _masked_mean((context_render.color - context_gt).abs(), context_mask)
    scale = state.log_scales.exp()
    opacity = torch.sigmoid(state.opacity_logits)
    alpha_coverage = _masked_mean(target_render.alpha, target_mask, default=1.0)
    gaussian_valid = state.valid_mask
    opacity_reg = _gaussian_masked_mean(opacity, gaussian_valid)
    scale_reg = _gaussian_masked_mean(scale, gaussian_valid)
    anisotropy_values = scale.max(dim=-1).values / scale.min(dim=-1).values.clamp_min(1.0e-6) - 1.0
    anisotropy_reg = _gaussian_masked_mean(anisotropy_values, gaussian_valid)
    sh_reg = _gaussian_masked_mean(state.sh_coeffs.abs(), gaussian_valid)
    delta_reg = pred.new_tensor(0.0)
    mean_step = pred.new_tensor(0.0)
    if prev_state is not None:
        valid_delta = state.valid_mask & prev_state.valid_mask
        delta_reg = (
            _gaussian_masked_mean((state.log_scales - prev_state.log_scales).abs(), valid_delta)
            + _gaussian_masked_mean((state.opacity_logits - prev_state.opacity_logits).abs(), valid_delta)
            + _gaussian_masked_mean((state.sh_coeffs - prev_state.sh_coeffs).abs(), valid_delta)
        )
        mean_step = _gaussian_masked_mean((state.means - prev_state.means).norm(dim=-1), valid_delta)
    if loss_type in {"image_lpips_latitude", "resplat_image_lpips_latitude"}:
        loss = float(loss_cfg.get("rgb_l1_weight", loss_cfg.get("image_weight", 1.0))) * latitude_l1 + float(loss_cfg.get("lpips_weight", 0.5)) * lpips_loss
    else:
        loss = (
            float(loss_cfg.get("rgb_l1_weight", 1.0)) * rgb_l1
            + float(loss_cfg.get("dssim_weight", 0.1)) * dssim
            + float(loss_cfg.get("lpips_weight", 0.0)) * lpips_loss
            + float(loss_cfg.get("depth_weight", 0.05)) * depth_loss
            + float(loss_cfg.get("context_weight", 0.0)) * context_l1
            + float(loss_cfg.get("opacity_reg_weight", 0.001)) * opacity_reg
            + float(loss_cfg.get("alpha_coverage_weight", 0.01)) * (1.0 - alpha_coverage)
            + float(loss_cfg.get("scale_reg_weight", 0.001)) * scale_reg
            + float(loss_cfg.get("anisotropy_reg_weight", 0.001)) * anisotropy_reg
            + float(loss_cfg.get("delta_reg_weight", 0.01)) * delta_reg
            + float(loss_cfg.get("mean_step_reg_weight", 0.01)) * mean_step
            + float(loss_cfg.get("sh_reg_weight", 0.0005)) * sh_reg
        )
    return loss, {
        "rgb_l1": rgb_l1.detach(),
        "latitude_image_l1": latitude_l1.detach(),
        "dssim": dssim.detach(),
        "lpips": lpips_loss.detach(),
        "depth_loss": depth_loss.detach(),
        "context_l1": context_l1.detach(),
        "alpha_coverage": alpha_coverage.detach(),
        "opacity_reg": opacity_reg.detach(),
        "scale_reg": scale_reg.detach(),
        "anisotropy_reg": anisotropy_reg.detach(),
        "delta_reg": delta_reg.detach(),
        "mean_step_reg": mean_step.detach(),
        "sh_reg": sh_reg.detach(),
        "psnr": _psnr(pred.detach(), gt.detach(), target_mask).detach(),
        "target_l1": rgb_l1.detach(),
        "target_valid_ratio": (
            target["valid_mask"].to(device=pred.device, dtype=pred.dtype).mean().detach()
            if torch.is_tensor(target.get("valid_mask"))
            else pred.new_tensor(1.0)
        ),
    }


def _forward_train(
    frontend: PanoReSplatFrontend,
    context: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    num_refine: int,
    stage: str,
    config: dict[str, Any],
    lpips_model: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    image_hw = tuple(int(x) for x in target["images"].shape[-2:])
    if stage == "refine":
        with torch.no_grad():
            state = frontend.initializer(
                context["images"],
                context["features"],
                context["depths"],
                context["poses_c2w"],
                context.get("valid_mask"),
                world_points=context.get("world_points"),
                tokens=context.get("tokens"),
                token_hw=context.get("token_hw"),
            )
    else:
        state = frontend.initializer(
            context["images"],
            context["features"],
            context["depths"],
            context["poses_c2w"],
            context.get("valid_mask"),
            world_points=context.get("world_points"),
            tokens=context.get("tokens"),
            token_hw=context.get("token_hw"),
        )
    state = _clamp_state_min_scale(state, _configured_min_scale(config))
    states = [state]
    context_renders: list[PanoRenderOutput | None] = []
    update_metrics: list[dict[str, torch.Tensor]] = []
    feedback_metrics: list[dict[str, torch.Tensor]] = []
    for _ in range(int(num_refine)):
        context_render = _render_views(frontend, state, context["poses_c2w"], tuple(int(x) for x in context["images"].shape[-2:]))
        context_renders.append(context_render)
        feedback, state_for_update, feedback_debug = frontend.feedback_encoder.refine_state_and_feedback(
            state,
            context["images"],
            context["poses_c2w"],
            context_render,
            context_depth=context.get("depths"),
            context_valid_mask=context.get("valid_mask"),
        )
        state_for_update = _clamp_state_min_scale(state_for_update, _configured_min_scale(config))
        state, metrics = frontend.update_block(state_for_update, feedback)
        state = _clamp_state_min_scale(state, _configured_min_scale(config))
        states.append(state)
        update_metrics.append(metrics)
        feedback_metrics.append(feedback_debug)
    final_context_render = _render_views(frontend, states[-1], context["poses_c2w"], tuple(int(x) for x in context["images"].shape[-2:]))
    target_renders = [_render_views(frontend, s, target["poses_c2w"], image_hw) for s in states]
    losses, metrics_by_iter = [], []
    for idx, (s, target_render) in enumerate(zip(states, target_renders)):
        context_render_i = final_context_render if idx == len(states) - 1 else (context_renders[idx] if idx < len(context_renders) else None)
        prev = None if idx == 0 else states[idx - 1]
        loss_i, m_i = _single_output_loss(
            s,
            target_render,
            target,
            context_render=context_render_i,
            context=context,
            prev_state=prev,
            config=config,
            lpips_model=lpips_model,
        )
        losses.append(loss_i)
        metrics_by_iter.append(m_i)
    inter_w = float(config.get("Loss", {}).get("intermediate_weight", 0.5))
    total = target_renders[-1].color.new_tensor(0.0)
    for idx, loss_i in enumerate(losses):
        total = total + (inter_w ** (len(losses) - 1 - idx)) * loss_i
    metrics: dict[str, torch.Tensor] = {"total_loss": total.detach()}
    for idx, m_i in enumerate(metrics_by_iter):
        metrics[f"iter{idx}/loss"] = losses[idx].detach()
        metrics[f"iter{idx}/psnr"] = m_i["psnr"]
        metrics[f"iter{idx}/target_l1"] = m_i["target_l1"]
        metrics[f"iter{idx}/context_l1"] = m_i["context_l1"]
    metrics.update(metrics_by_iter[-1])
    metrics.update({f"final/{k}": v for k, v in metrics_by_iter[-1].items()})
    if len(metrics_by_iter) > 1:
        metrics["refinement_improvement"] = losses[0].detach() - losses[-1].detach()
        metrics["iter_final/loss"] = losses[-1].detach()
        metrics["iter_final/psnr"] = metrics_by_iter[-1]["psnr"]
    metrics.update(_gaussian_stats(states[-1]))
    for idx, update in enumerate(update_metrics):
        for key, value in update.items():
            metrics[f"update{idx}/{key}"] = value.detach()
    for idx, feedback_debug in enumerate(feedback_metrics):
        for key, value in feedback_debug.items():
            metrics[f"feedback{idx}/{key}"] = value.detach() if torch.is_tensor(value) else target_renders[-1].color.new_tensor(float(value))
    artifacts = {
        "states": states,
        "target_renders": target_renders,
        "context_render": final_context_render,
    }
    return total, metrics, artifacts


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        out[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return out


def _check_finite(metrics: dict[str, torch.Tensor], artifacts: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Non-finite metric {key}: {value}")
    for state in artifacts.get("states", []):
        for name in ("means", "log_scales", "rotations_unnorm", "opacity_logits", "sh_coeffs", "latent_features"):
            value = getattr(state, name)
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Non-finite state tensor {name}")


def _write_csv(path: Path, row: dict[str, float], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = ["step"] + sorted(k for k in row if k != "step")
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if not exists or not append:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in keys})


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if t.ndim == 3 and int(t.shape[0]) == 3:
        t = t.permute(1, 2, 0)
    arr = (t.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_to_image(mask: torch.Tensor) -> Image.Image:
    m = mask.detach().float().cpu()
    if m.ndim == 3:
        m = m[0]
    arr = (m.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def _save_visualization(output_dir: Path, step_name: str, target: dict[str, torch.Tensor], artifacts: dict[str, Any]) -> Path:
    render_dir = output_dir / "renders" / step_name
    render_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = target["images"][0, 0]
    renders: list[PanoRenderOutput] = artifacts["target_renders"]
    final = renders[-1].color[0, 0] if renders[-1].color.ndim == 5 else renders[-1].color[0]
    initial = renders[0].color[0, 0] if renders[0].color.ndim == 5 else renders[0].color[0]
    target_mask = None
    if torch.is_tensor(target.get("valid_mask")):
        target_mask = target["valid_mask"][0, 0].detach().float().cpu()
        if target_mask.ndim == 3:
            target_mask = target_mask[0]
    err = (final.detach().float().cpu() - target_rgb.detach().float().cpu()).abs().mean(dim=0)
    if target_mask is not None:
        err = err * target_mask
    err = err / err.max().clamp_min(1.0e-6)
    err_img = torch.stack([err, 1.0 - err, torch.zeros_like(err)], dim=0)
    panels = [
        ("gt_target", _tensor_to_image(target_rgb)),
        ("iter0_render", _tensor_to_image(initial)),
        ("rendered_target", _tensor_to_image(final)),
        ("error_map", _tensor_to_image(err_img)),
    ]
    if target_mask is not None:
        panels.insert(1, ("masked_target", _tensor_to_image(target_rgb.detach().float().cpu() * target_mask.unsqueeze(0))))
        panels.append(("target_valid_mask", _mask_to_image(target_mask)))
    if torch.is_tensor(target.get("sky_mask")):
        sky = target["sky_mask"][0, 0].detach().float().cpu()
        panels.append(("target_sky_mask", _mask_to_image(sky)))
    context = artifacts.get("context_render")
    if isinstance(context, PanoRenderOutput):
        ctx = context.color[0, 0] if context.color.ndim == 5 else context.color[0]
        panels.insert(0, ("rendered_context", _tensor_to_image(ctx)))
    for name, image in panels:
        image.save(render_dir / f"{name}.png")
    width = sum(img.width for _, img in panels)
    height = max(img.height for _, img in panels) + 24
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in panels:
        canvas.paste(image, (x, 24))
        draw.text((x + 4, 4), label, fill=(255, 255, 255))
        x += image.width
    panel_path = render_dir / "panel.png"
    canvas.save(panel_path)
    return panel_path


def _depth_to_image(depth: torch.Tensor, mask: torch.Tensor | None = None) -> Image.Image:
    d = depth.detach().float().cpu()
    if d.ndim == 3:
        d = d[0]
    valid = torch.isfinite(d) & (d > 0.0)
    if mask is not None:
        m = mask.detach().float().cpu()
        if m.ndim == 3:
            m = m[0]
        valid = valid & (m > 0.5)
    if bool(valid.any()):
        vals = d[valid]
        lo = torch.quantile(vals, 0.02)
        hi = torch.quantile(vals, 0.98).clamp_min(lo + 1.0e-6)
        norm = ((d - lo) / (hi - lo)).clamp(0.0, 1.0)
    else:
        norm = torch.zeros_like(d)
    if mask is not None:
        norm = norm * valid.float()
    arr = (norm.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def _save_aligned_visualization(output_dir: Path, step_name: str, target: dict[str, torch.Tensor], artifacts: dict[str, Any]) -> Path:
    render_dir = output_dir / "renders" / step_name
    render_dir.mkdir(parents=True, exist_ok=True)
    renders: list[PanoRenderOutput] = artifacts["target_renders"]
    final = renders[-1].color
    if final.ndim != 5:
        final = final.unsqueeze(1)
    gt = target["images"]
    depth = target.get("depths")
    mask = target.get("valid_mask")
    views = min(4, int(gt.shape[1]), int(final.shape[1]))
    gt_images = [_tensor_to_image(gt[0, idx]) for idx in range(views)]
    depth_images = []
    render_images = [_tensor_to_image(final[0, idx]) for idx in range(views)]
    for idx in range(views):
        depth_i = depth[0, idx] if torch.is_tensor(depth) else torch.zeros(1, *gt.shape[-2:], device=gt.device, dtype=gt.dtype)
        mask_i = mask[0, idx] if torch.is_tensor(mask) else None
        depth_images.append(_depth_to_image(depth_i, mask_i))
    for idx, image in enumerate(gt_images):
        image.save(render_dir / f"gt_view_{idx}.png")
    for idx, image in enumerate(depth_images):
        image.save(render_dir / f"depth_view_{idx}.png")
    for idx, image in enumerate(render_images):
        image.save(render_dir / f"render_view_{idx}.png")
    rows = [gt_images, depth_images, render_images]
    tile_w = max(img.width for row in rows for img in row)
    tile_h = max(img.height for row in rows for img in row)
    canvas = Image.new("RGB", (tile_w * views, tile_h * 3), (0, 0, 0))
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            if image.size != (tile_w, tile_h):
                image = image.resize((tile_w, tile_h), Image.BILINEAR)
            canvas.paste(image, (col_idx * tile_w, row_idx * tile_h))
    panel_path = render_dir / "panel.png"
    canvas.save(panel_path)
    return panel_path


def _log_wandb_image(wandb_run: Any, key: str, path: Path, *, step: int, logger: _Logger) -> None:
    if wandb_run is None:
        return
    try:
        import wandb

        wandb_run.log({key: wandb.Image(str(path))}, step=step)
    except Exception as exc:
        logger.error(f"Failed to log W&B image {key}: {exc}")


def _save_run_metadata(output_dir: Path, config: dict[str, Any], command: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    try:
        status = subprocess.run(["git", "status", "--short", "--branch"], cwd=Path.cwd(), capture_output=True, text=True, check=False)
        (output_dir / "git_status.txt").write_text(status.stdout + status.stderr, encoding="utf-8")
    except Exception as exc:
        (output_dir / "git_status.txt").write_text(f"git status unavailable: {exc}\n", encoding="utf-8")


def _count_params(module: nn.Module) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for param in module.parameters():
        count = 0 if isinstance(param, UninitializedParameter) else int(param.numel())
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    return int(trainable), int(frozen)


def _nonfinite_grad_report(module: nn.Module, *, limit: int = 16) -> str:
    rows: list[str] = []
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all()):
            continue
        bad = (~finite).float().mean().item()
        nan = torch.isnan(grad).float().mean().item()
        inf = torch.isinf(grad).float().mean().item()
        finite_abs = grad[finite].abs()
        finite_max = finite_abs.max().item() if finite_abs.numel() else float("nan")
        rows.append(f"{name}: bad={bad:.6f} nan={nan:.6f} inf={inf:.6f} finite_abs_max={finite_max:.6g}")
        if len(rows) >= limit:
            break
    return "; ".join(rows) if rows else "no non-finite parameter gradients found"


def _write_report(
    output_dir: Path,
    *,
    command: list[str],
    renderer: str,
    stage: str,
    trainable: int,
    frozen: int,
    first_metrics: dict[str, float],
    last_metrics: dict[str, float],
    passed: bool,
    diagnosis: str = "",
) -> None:
    report = [
        f"# Pano-ReSplat {stage} Report",
        "",
        f"1. command: `{' '.join(command)}`",
        f"2. renderer: `{renderer}`",
        "3. data split: train",
        f"4. trainable parameter count: {trainable}",
        f"5. frozen parameter count: {frozen}",
        f"6. Gaussian count: {last_metrics.get('gaussian_count', 0.0):.0f}",
        f"7. initial loss / final loss: {first_metrics.get('total_loss', float('nan')):.6f} / {last_metrics.get('total_loss', float('nan')):.6f}",
        f"8. initial PSNR / final PSNR: {first_metrics.get('psnr', float('nan')):.3f} / {last_metrics.get('psnr', float('nan')):.3f}",
        f"9. refinement iter0 vs iter_final: {last_metrics.get('iter0/loss', float('nan')):.6f} vs {last_metrics.get('iter_final/loss', last_metrics.get('total_loss', float('nan'))):.6f}",
        f"10. acceptance passed: {bool(passed)}",
        f"11. failures and diagnosis: {diagnosis or 'none'}",
        "12. next step: run gsplat360 training once renderer preflight succeeds.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def _stage_best_name(stage: str) -> str:
    return {"init": "best_init.pt", "refine": "best_refine.pt", "joint": "best_joint.pt"}.get(stage, "best.pt")


def _num_refine_for_step(config: dict[str, Any], stage: str) -> int:
    tr = config.get("Training", {})
    if stage == "init":
        return 0
    lo = int(tr.get("train_min_refine", 0))
    hi = int(tr.get("train_max_refine", lo))
    if hi <= lo:
        return max(0, lo)
    return int(torch.randint(lo, hi + 1, (1,)).item())


def _amp_enabled(config: dict[str, Any], device: torch.device) -> bool:
    return bool(config.get("Training", {}).get("amp", False)) and device.type == "cuda"


def _autocast_context(config: dict[str, Any], device: torch.device):
    if not _amp_enabled(config, device):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _target_leakage_check(frontend: PanoReSplatFrontend, context: dict[str, torch.Tensor]) -> None:
    target_a = {"images": torch.zeros_like(context["images"][:, :1]), "poses_c2w": context["poses_c2w"][:, :1]}
    target_b = {"images": torch.ones_like(context["images"][:, :1]), "poses_c2w": context["poses_c2w"][:, :1]}
    with torch.no_grad():
        out_a = frontend(context, target=target_a, num_refine=2)
        out_b = frontend(context, target=target_b, num_refine=2)
    if not torch.allclose(out_a["final_state"].means, out_b["final_state"].means, atol=1.0e-6):
        raise RuntimeError("Target leakage check failed: final_state.means changed with target image.")
    if not torch.allclose(out_a["final_state"].sh_coeffs, out_b["final_state"].sh_coeffs, atol=1.0e-6):
        raise RuntimeError("Target leakage check failed: final_state.sh_coeffs changed with target image.")


def _renderer_gradient_check(
    frontend: PanoReSplatFrontend,
    context: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    stage: str,
    config: dict[str, Any],
    lpips_model: nn.Module | None = None,
) -> None:
    frontend.zero_grad(set_to_none=True)
    device = target["images"].device
    with _autocast_context(config, device):
        loss, _metrics, _artifacts = _forward_train(
            frontend,
            context,
            target,
            num_refine=_num_refine_for_step(config, stage),
            stage=stage,
            config=config,
            lpips_model=lpips_model,
        )
    loss.backward()
    total_grad = 0.0
    for param in frontend.parameters():
        if param.requires_grad and param.grad is not None:
            total_grad += float(param.grad.detach().abs().sum().cpu())
    frontend.zero_grad(set_to_none=True)
    if total_grad <= 0.0:
        raise RuntimeError("Renderer gradient check failed: no nonzero gradients on trainable parameters.")


def _validation_loader(config: dict[str, Any], *, logger: _Logger) -> DataLoader | None:
    if not bool(config.get("Validation", {}).get("enabled", False)):
        return None
    try:
        dataset = build_matching_dataset_from_config(config, split="val")
    except Exception as exc:
        logger.error(f"Validation dataset unavailable: {exc}")
        return None
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except Exception:
        length = 1
    if int(length) <= 0:
        logger.error("Validation dataset is empty; disabling validation.")
        return None
    return DataLoader(
        dataset,
        batch_size=int(config.get("Training", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("Training", {}).get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )


def _run_validation(
    frontend: PanoReSplatFrontend,
    prior_extractor: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    device: torch.device,
    stage: str,
    step: int,
    output_dir: Path,
    wandb_run: Any,
    logger: _Logger,
    lpips_model: nn.Module | None = None,
    write_outputs: bool = True,
) -> dict[str, float]:
    was_training = frontend.training
    frontend.eval()
    prior_extractor.eval()
    max_batches = max(1, int(config.get("Validation", {}).get("max_batches", 1)))
    input_reconstruction = _is_input_reconstruction(config)
    rows: list[dict[str, float]] = []
    first_panel: Path | None = None
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            sample = _to_device_batch(raw_batch, device)
            validate_training_sample(sample, "matching_only", allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            priors = prior_extractor(sample)
            if input_reconstruction:
                context, target = _sample_input_reconstruction(sample, priors, config)
            else:
                context, target = _sample_window(sample, priors, config, step=step)
            with _autocast_context(config, device):
                loss, metrics_t, artifacts = _forward_train(
                    frontend,
                    context,
                    target,
                    num_refine=max(0, int(config.get("Training", {}).get("eval_refine", 0))),
                    stage=stage,
                    config=config,
                    lpips_model=lpips_model,
                )
            metrics = _float_metrics({"total_loss": loss.detach(), **metrics_t})
            rows.append(metrics)
            if write_outputs and first_panel is None:
                tag = f"val_step_{step:06d}"
                if input_reconstruction:
                    first_panel = _save_aligned_visualization(output_dir, tag, target, artifacts)
                else:
                    first_panel = _save_visualization(output_dir, tag, target, artifacts)
    if was_training:
        frontend.train()
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    averaged = {"step": float(step)}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        if vals:
            averaged[key] = float(sum(vals) / len(vals))
    if write_outputs:
        _write_csv(output_dir / "val_metrics.csv", averaged)
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in averaged.items()}, step=step)
        if first_panel is not None:
            _log_wandb_image(
                wandb_run,
                "renders/val_panel" if input_reconstruction else f"renders/val_step_{step:06d}",
                first_panel,
                step=step,
                logger=logger,
            )
        logger.log(yaml.safe_dump({"step": step, "validation": averaged}, sort_keys=False).strip())
    return averaged


def train_resplat_gaussian(config: dict[str, Any], *, command: list[str] | None = None, checkpoint: str | None = None, resume: str | None = None) -> dict[str, Any]:
    ddp_state = _distributed_state()
    _init_distributed_if_needed(ddp_state)
    is_main = bool(ddp_state["is_main"])
    tr = config.get("Training", {})
    stage = str(tr.get("stage", "overfit")).lower()
    if stage not in {"overfit", "init", "refine", "joint"}:
        raise ValueError(f"Unsupported stage: {stage}")
    torch.manual_seed(int(tr.get("seed", 1234)) + int(ddp_state["rank"]))
    if bool(ddp_state["distributed"]):
        device = torch.device("cuda", int(ddp_state["local_rank"])) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = _device_from_arg(tr.get("device"))
    output_dir = Path(tr.get("output_dir", "outputs/pano_resplat/run"))
    if is_main:
        _save_run_metadata(output_dir, config, command or sys.argv)
    _barrier_if_distributed(ddp_state)
    logger = _Logger(output_dir, enabled=is_main, rank=int(ddp_state["rank"]))
    logger.log(
        yaml.safe_dump(
            {
                "stage": stage,
                "device": str(device),
                "renderer": config.get("Renderer", {}).get("backend"),
                "distributed": bool(ddp_state["distributed"]),
                "world_size": int(ddp_state["world_size"]),
            },
            sort_keys=False,
        ).strip()
    )
    input_reconstruction = _is_input_reconstruction(config)

    dataset = build_matching_dataset_from_config(config, split="train")
    if bool(tr.get("debug_overfit", False)) or stage == "overfit":
        dataset = Subset(dataset, [0])
    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=int(ddp_state["world_size"]),
            rank=int(ddp_state["rank"]),
            shuffle=not (bool(tr.get("debug_overfit", False)) or stage == "overfit"),
            drop_last=False,
        )
        if bool(ddp_state["distributed"])
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(tr.get("batch_size", 1)),
        shuffle=train_sampler is None and not (bool(tr.get("debug_overfit", False)) or stage == "overfit"),
        sampler=train_sampler,
        num_workers=int(tr.get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )
    val_loader = _validation_loader(config, logger=logger)
    prior_extractor = _build_prior_extractor(config, device=device)
    prior_extractor.eval()
    for p in prior_extractor.parameters():
        p.requires_grad_(False)

    frontend = _build_frontend(config, device=device)
    lpips_model = _build_lpips_model(config, device, logger)
    load_path = resume or checkpoint
    if load_path:
        _load_checkpoint(frontend, load_path)
    overfit_trains_refiner = stage == "overfit" and int(tr.get("train_max_refine", tr.get("train_min_refine", 0))) > 0
    _set_stage_trainability(frontend, stage, overfit_trains_refiner=overfit_trains_refiner)
    if stage == "refine":
        _set_requires_grad(frontend.initializer, False)
    frontend = _wrap_trainable_modules_for_ddp(frontend, ddp_state)
    optimizer = _optimizer(frontend, config, stage)
    trainable, frozen = _count_params(frontend)
    logger.log(yaml.safe_dump({"trainable_params": trainable, "frozen_params": frozen}, sort_keys=False).strip())

    wandb_run = _init_wandb(config, output_dir) if is_main else None
    max_steps = int(tr.get("steps", 1))
    accum = max(1, int(tr.get("gradient_accumulation_steps", 1)))
    amp_enabled = _amp_enabled(config, device)
    scaler = _grad_scaler(amp_enabled)
    save_every = max(1, int(tr.get("save_every", 100)))
    vis_every = max(1, int(tr.get("vis_every", 100)))
    eval_every = max(1, int(tr.get("eval_every", 100)))
    log_every = max(1, int(tr.get("log_every", 1)))
    best = float("inf")
    first_metrics: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    first_artifacts: dict[str, Any] | None = None
    first_target: dict[str, torch.Tensor] | None = None
    checked = False
    step = 0
    micro_step = 0
    epoch = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for raw_batch in loader:
            sample = _to_device_batch(raw_batch, device)
            validate_training_sample(sample, "matching_only", allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            with torch.no_grad():
                priors = prior_extractor(sample)
            if input_reconstruction:
                context, target = _sample_input_reconstruction(sample, priors, config)
            else:
                context, target = _sample_window(sample, priors, config, step=step)
            if not checked:
                if bool(config.get("Checks", {}).get("target_leakage_check", True)):
                    if input_reconstruction and _num_refine_for_step(config, stage) <= 0:
                        logger.log("Skipping target leakage check for init-only input reconstruction training.")
                    else:
                        _target_leakage_check(frontend, context)
                if bool(config.get("Checks", {}).get("renderer_gradient_check", True)):
                    _renderer_gradient_check(frontend, context, target, stage=stage, config=config, lpips_model=lpips_model)
                checked = True
            num_refine = _num_refine_for_step(config, stage)
            with _autocast_context(config, device):
                loss, metrics_t, artifacts = _forward_train(
                    frontend,
                    context,
                    target,
                    num_refine=num_refine,
                    stage=stage,
                    config=config,
                    lpips_model=lpips_model,
                )
            _check_finite(metrics_t, artifacts)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at optimizer step {step}: {loss}")
            scaler.scale(loss / float(accum)).backward()
            micro_step += 1
            if micro_step % accum != 0:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(frontend.parameters(), float(tr.get("grad_clip", 1.0)), error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                report = _nonfinite_grad_report(frontend)
                raise RuntimeError(f"Non-finite grad norm at optimizer step {step}: {grad_norm}; {report}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = _float_metrics({"step": float(step), **metrics_t, "grad_norm": grad_norm.detach()})
            last_metrics = metrics
            if not first_metrics:
                first_metrics = dict(metrics)
                first_artifacts = artifacts
                first_target = target
                if is_main:
                    if input_reconstruction:
                        panel = _save_aligned_visualization(output_dir, "step_000000", target, artifacts)
                        _log_wandb_image(wandb_run, "renders/train_panel", panel, step=step, logger=logger)
                    else:
                        panel = _save_visualization(output_dir, "step_000000", target, artifacts)
                        _log_wandb_image(wandb_run, "renders/step_000000", panel, step=step, logger=logger)
            if is_main:
                _write_csv(output_dir / "train_metrics.csv", metrics)
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            if step == 1 or step % log_every == 0:
                logger.log(yaml.safe_dump({"step": step, "metrics": metrics}, sort_keys=False).strip())
            if is_main and (step % vis_every == 0 or step == max_steps):
                tag = f"step_{step:06d}"
                if input_reconstruction:
                    panel = _save_aligned_visualization(output_dir, tag, target, artifacts)
                    _log_wandb_image(wandb_run, "renders/train_panel", panel, step=step, logger=logger)
                else:
                    panel = _save_visualization(output_dir, tag, target, artifacts)
                    _log_wandb_image(wandb_run, f"renders/{tag}", panel, step=step, logger=logger)
            if val_loader is not None and (step % eval_every == 0 or step == max_steps):
                _run_validation(
                    frontend,
                    prior_extractor,
                    val_loader,
                    config,
                    device=device,
                    stage=stage,
                    step=step,
                    output_dir=output_dir,
                    wandb_run=wandb_run,
                    logger=logger,
                    lpips_model=lpips_model,
                    write_outputs=is_main,
                )
            if is_main and metrics["total_loss"] < best:
                best = metrics["total_loss"]
                _save_checkpoint(output_dir / _stage_best_name(stage), frontend, config, step, stage, metrics)
                _save_checkpoint(output_dir / "best.pt", frontend, config, step, stage, metrics)
            if is_main and (step % save_every == 0 or step == max_steps):
                _save_checkpoint(output_dir / "latest.pt", frontend, config, step, stage, metrics)
            if step >= max_steps:
                break
    if is_main and first_artifacts is not None and first_target is not None:
        if input_reconstruction:
            panel = _save_aligned_visualization(output_dir, "final", first_target if step == 0 else target, artifacts)
            _log_wandb_image(wandb_run, "renders/train_panel", panel, step=step, logger=logger)
        else:
            panel = _save_visualization(output_dir, "final", first_target if step == 0 else target, artifacts)
            _log_wandb_image(wandb_run, "renders/final", panel, step=step, logger=logger)
    metrics_json = {
        "steps": step,
        "best_loss": best,
        "first_metrics": first_metrics,
        "last_metrics": last_metrics,
        "elapsed_sec": time.time() - start,
        "checkpoint": str(output_dir / "latest.pt"),
    }
    if is_main:
        _save_checkpoint(output_dir / "latest.pt", frontend, config, step, stage, last_metrics)
        (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
        if not (output_dir / "val_metrics.csv").exists():
            (output_dir / "val_metrics.csv").write_text("step,total_loss,psnr,rgb_l1\n", encoding="utf-8")
        _write_report(
            output_dir,
            command=command or sys.argv,
            renderer=str(config.get("Renderer", {}).get("backend", "soft_splat")),
            stage=stage,
            trainable=trainable,
            frozen=frozen,
            first_metrics=first_metrics,
            last_metrics=last_metrics,
            passed=math.isfinite(last_metrics.get("total_loss", float("nan"))),
        )
    if wandb_run is not None:
        wandb_run.finish()
    _barrier_if_distributed(ddp_state)
    _cleanup_distributed()
    return metrics_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=None, choices=["init", "refine", "joint", "overfit"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--renderer", default=None, choices=["soft_splat", "gsplat360"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--vis-every", type=int, default=None)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_resplat_train_config(args.config)
    if args.stage is not None:
        config.setdefault("Training", {})["stage"] = args.stage
    if args.steps is not None:
        config.setdefault("Training", {})["steps"] = int(args.steps)
    if args.batch_size is not None:
        config.setdefault("Training", {})["batch_size"] = int(args.batch_size)
    if args.renderer is not None:
        config.setdefault("Renderer", {})["backend"] = args.renderer
        if args.renderer == "gsplat360":
            config.setdefault("Renderer", {})["allow_soft_splat_fallback"] = False
    if args.output_dir is not None:
        config.setdefault("Training", {})["output_dir"] = args.output_dir
    if args.device is not None:
        config.setdefault("Training", {})["device"] = args.device
    if args.amp:
        config.setdefault("Training", {})["amp"] = True
    if args.num_workers is not None:
        config.setdefault("Training", {})["num_workers"] = int(args.num_workers)
    if args.wandb_mode is not None:
        config.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
        config.setdefault("WeightsAndBiases", {})["enabled"] = args.wandb_mode != "disabled"
    if args.save_every is not None:
        config.setdefault("Training", {})["save_every"] = int(args.save_every)
    if args.eval_every is not None:
        config.setdefault("Training", {})["eval_every"] = int(args.eval_every)
    if args.vis_every is not None:
        config.setdefault("Training", {})["vis_every"] = int(args.vis_every)
    if args.max_train_scenes is not None:
        config.setdefault("Dataset", {})["max_clips"] = int(args.max_train_scenes)
    if args.debug_overfit:
        config.setdefault("Training", {})["debug_overfit"] = True
        config.setdefault("Training", {})["window_mode"] = "fixed"
        config.setdefault("Dataset", {})["max_clips"] = 1
    if args.seed is not None:
        config.setdefault("Training", {})["seed"] = int(args.seed)
    try:
        result = train_resplat_gaussian(config, command=sys.argv, checkpoint=args.checkpoint, resume=args.resume)
    except Exception as exc:
        out = Path(config.get("Training", {}).get("output_dir", "outputs/pano_resplat/run"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.log").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
