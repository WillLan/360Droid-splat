"""Train staged PanoVGGT-M3-Sphere matching and sky heads."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing

from .grid_utils import feature_uv_to_image_uv
from .matching_dataset import build_matching_dataset_from_config, normalize_training_mode, validate_training_sample
from .matching_head import PanoVGGTMatchingSkyHead
from .matching_losses import PanoVGGTMatchingLossWeights, PanoVGGTMatchingSkyLoss, sample_feature_values
from .spherical_correspondence import generate_gt_spherical_correspondences
from .spherical_correspondence import spherical_tangent_residual


def _default_config() -> dict[str, Any]:
    return {
        "Training": {
            "mode": "matching_only",
            "steps": 2,
            "batch_size": 1,
            "frames_per_sample": 3,
            "num_workers": 0,
            "amp": False,
            "seed": 1234,
            "log_interval": 1,
            "save_interval": 1,
            "val_interval": 1,
            "output_dir": "outputs/panovggt_m3_sphere_omni360",
            "grad_clip": 1.0,
            "resume_checkpoint": None,
        },
        "Model": {
            "use_synthetic_features": True,
            "freeze_panovggt": True,
            "feature_hook": None,
            "feature_dim": 16,
            "feature_stride": 4,
            "panovggt_repo": None,
            "panovggt_checkpoint": None,
            "class_path": None,
        },
        "Heads": {
            "descriptor_dim": 24,
            "hidden_dim": 32,
            "num_conv_blocks": 2,
            "train_matching": True,
            "train_sky": False,
            "train_static": True,
        },
        "Dataset": {
            "synthetic": True,
            "synthetic_variant": "complete",
            "height": 32,
            "width": 64,
            "class_map": {"sky_ids": [1], "classes": {"sky": 1}},
        },
        "Pairs": {
            "samples_per_edge": 128,
            "min_baseline_deg": 0.0,
            "max_baseline_deg": 60.0,
            "sky_filter": True,
        },
        "Optimizer": {
            "type": "adamw",
            "lr": 2.0e-4,
            "weight_decay": 0.05,
            "scheduler": "none",
        },
        "Loss": {
            "nce_weight": 1.0,
            "confidence_weight": 0.1,
            "spherical_regression_weight": 0.2,
            "static_weight": 0.05,
            "sky_weight": 0.2,
            "sky_dice_weight": 0.5,
            "smoothness_weight": 0.01,
            "temperature": 0.07,
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": None,
            "mode": "online",
            "log_every": 10,
        },
        "Validation": {
            "enabled": False,
            "batch_size": 1,
            "num_workers": 0,
            "max_batches": 1,
        },
        "Visualization": {
            "enabled": False,
            "interval": 100,
            "max_matches": 40,
            "save_dir": "visualizations",
        },
    }


def _merge_config(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


def load_matching_train_config(path: str | None) -> dict[str, Any]:
    """Load a staged PanoVGGT-M3-Sphere training config."""

    config = _default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _merge_config(config, user)


def matching_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples while preserving optional ``None`` fields."""

    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        if all(torch.is_tensor(value) for value in values):
            out[key] = torch.stack(values, dim=0)
        elif all(value is None for value in values):
            out[key] = None
        elif all(isinstance(value, bool) for value in values):
            out[key] = torch.tensor(values, dtype=torch.bool)
        else:
            out[key] = values
    return out


class FrozenSyntheticFeatureExtractor(nn.Module):
    """Frozen deterministic feature extractor for explicit synthetic tests."""

    def __init__(self, feature_dim: int = 16, feature_stride: int = 4) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.feature_stride = max(1, int(feature_stride))
        self.proj = nn.Conv2d(3, self.feature_dim, kernel_size=1, bias=False)
        with torch.no_grad():
            weight = torch.linspace(-0.5, 0.5, steps=self.feature_dim * 3).view(self.feature_dim, 3, 1, 1)
            self.proj.weight.copy_(weight)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract frozen features with shape ``B x N x C x Hf x Wf``."""

        if images.ndim != 5:
            raise ValueError(f"images must have shape BxNx3xHxW, got {tuple(images.shape)}")
        b, n = int(images.shape[0]), int(images.shape[1])
        flat = images.reshape(b * n, 3, images.shape[-2], images.shape[-1])
        pooled = F.avg_pool2d(flat, kernel_size=self.feature_stride, stride=self.feature_stride, ceil_mode=False)
        feat = self.proj(pooled)
        return feat.view(b, n, self.feature_dim, feat.shape[-2], feat.shape[-1])


class ExternalPanoVGGTFeatureExtractor(nn.Module):
    """Frozen external PanoVGGT feature extractor using an explicit hook."""

    def __init__(self, model_cfg: dict[str, Any], *, device: torch.device) -> None:
        super().__init__()
        feature_hook = model_cfg.get("feature_hook")
        if not feature_hook:
            raise ValueError("Model.feature_hook is required for real PanoVGGT feature extraction.")
        from .engine import ExternalPanoVGGTInferenceEngine

        image_size_raw = model_cfg.get("image_size", None)
        image_size = None
        if image_size_raw is not None:
            image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        self.engine = ExternalPanoVGGTInferenceEngine(
            repo_path=model_cfg.get("panovggt_repo"),
            config_path=model_cfg.get("panovggt_config") or model_cfg.get("config_path"),
            checkpoint=model_cfg.get("panovggt_checkpoint"),
            class_path=model_cfg.get("class_path"),
            model_kwargs=dict(model_cfg.get("model_kwargs", {})),
            image_size=image_size,
            device=device,
            amp=bool(model_cfg.get("amp", True)),
            input_batch_dim=bool(model_cfg.get("input_batch_dim", True)),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            skip_dinov2_pretrain=bool(model_cfg.get("skip_dinov2_pretrain", False)),
            patch_multiple=int(model_cfg.get("patch_multiple", 14)),
        )
        self.model = self.engine.model
        for param in self.model.parameters():
            param.requires_grad_(False)
        modules = dict(self.model.named_modules())
        if str(feature_hook) not in modules:
            raise ValueError(f"Model.feature_hook={feature_hook!r} was not found in external PanoVGGT modules.")
        self._feature: torch.Tensor | None = None
        self._input_hw: tuple[int, int] | None = None
        self._patch_size = int(model_cfg.get("patch_size") or getattr(self.model, "patch_size", self.engine.patch_multiple))

        def hook_fn(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            self._input_hw = self._infer_input_hw(inputs)
            self._feature = self._normalize_hook_output(output)

        modules[str(feature_hook)].register_forward_hook(hook_fn)

    @staticmethod
    def _infer_input_hw(inputs: tuple[Any, ...]) -> tuple[int, int] | None:
        for value in inputs:
            if torch.is_tensor(value) and value.ndim >= 4:
                return int(value.shape[-2]), int(value.shape[-1])
        return None

    def _tokens_to_feature(self, tokens: torch.Tensor, patch_start_idx: int = 0) -> torch.Tensor:
        if tokens.ndim == 5:
            if tokens.shape[2] < tokens.shape[-1]:
                return tokens
            return tokens.permute(0, 1, 4, 2, 3).contiguous()
        if tokens.ndim != 4:
            raise ValueError(f"Captured PanoVGGT tokens must be 4D or 5D, got {tuple(tokens.shape)}")
        if int(patch_start_idx) > 0:
            tokens = tokens[:, :, int(patch_start_idx) :, :]
        if self._input_hw is None:
            raise RuntimeError("Cannot reshape PanoVGGT token features because hook input image size was not captured.")
        h_in, w_in = self._input_hw
        patch = max(1, int(self._patch_size))
        height_f = max(1, h_in // patch)
        width_f = max(1, w_in // patch)
        expected = height_f * width_f
        if int(tokens.shape[2]) != expected:
            side = int(round(math.sqrt(float(tokens.shape[2]))))
            if side * side == int(tokens.shape[2]):
                height_f, width_f = side, side
            else:
                raise ValueError(
                    "Captured PanoVGGT token count cannot be reshaped to a feature grid: "
                    f"tokens={int(tokens.shape[2])}, input_hw={self._input_hw}, patch_size={patch}."
                )
        return tokens.reshape(tokens.shape[0], tokens.shape[1], height_f, width_f, tokens.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()

    def _normalize_hook_output(self, output: Any) -> torch.Tensor:
        patch_start_idx = 0
        feature = output
        if isinstance(output, tuple) and len(output) >= 2:
            if isinstance(output[1], int):
                patch_start_idx = int(output[1])
            feature = output[0]
        if isinstance(feature, (list, tuple)):
            tensors = [item for item in feature if torch.is_tensor(item)]
            if not tensors:
                raise TypeError("Feature hook returned a list/tuple without tensors.")
            feature = tensors[-1]
        if not torch.is_tensor(feature):
            raise TypeError(f"Feature hook returned unsupported output type {type(output)!r}.")
        if feature.ndim in (4, 5):
            return self._tokens_to_feature(feature, patch_start_idx=patch_start_idx)
        raise ValueError(f"Captured PanoVGGT feature must be token/grid tensor, got {tuple(feature.shape)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen external model and return the captured feature tensor."""

        if images.ndim != 5:
            raise ValueError(f"images must have shape BxNx3xHxW, got {tuple(images.shape)}")
        features: list[torch.Tensor] = []
        batch_size = int(images.shape[0])
        with torch.no_grad():
            for b in range(batch_size):
                self._feature = None
                self._input_hw = None
                _ = self.engine.infer(images[b])
                if self._feature is None:
                    raise RuntimeError("External PanoVGGT feature hook did not capture any tensor.")
                feature = self._feature
                if feature.ndim != 5:
                    raise ValueError(f"Captured PanoVGGT feature must be BxNxCxHxW, got {tuple(feature.shape)}")
                if int(feature.shape[0]) != 1:
                    raise ValueError(f"External PanoVGGT per-sample hook must return B=1, got {tuple(feature.shape)}")
                features.append(feature[0])
        return torch.stack(features, dim=0)


def _build_feature_extractor(config: dict[str, Any], *, device: torch.device) -> nn.Module:
    model_cfg = config.get("Model", {})
    if bool(model_cfg.get("use_synthetic_features", False)):
        return FrozenSyntheticFeatureExtractor(
            feature_dim=int(model_cfg.get("feature_dim", 16)),
            feature_stride=int(model_cfg.get("feature_stride", 4)),
        ).to(device)
    return ExternalPanoVGGTFeatureExtractor(model_cfg, device=device).to(device)


def _mode_head_flags(mode: str, heads_cfg: dict[str, Any]) -> dict[str, bool]:
    training_mode = normalize_training_mode(mode)
    if training_mode == "sky_only":
        return {"train_matching": False, "train_static": False, "train_sky": True}
    if training_mode == "matching_only":
        return {"train_matching": True, "train_static": bool(heads_cfg.get("train_static", True)), "train_sky": False}
    return {"train_matching": True, "train_static": bool(heads_cfg.get("train_static", True)), "train_sky": True}


def _set_trainable_for_mode(wrapper: PanoVGGTMatchingSkyHead, mode: str) -> None:
    training_mode = normalize_training_mode(mode)
    for param in wrapper.parameters():
        param.requires_grad_(False)
    if training_mode in ("matching_only", "head_joint_calibration"):
        for param in wrapper.matching_head.parameters():
            param.requires_grad_(True)
    if training_mode in ("sky_only", "head_joint_calibration"):
        for param in wrapper.sky_head.parameters():
            param.requires_grad_(True)


def _loss_weights_from_config(config: dict[str, Any]) -> PanoVGGTMatchingLossWeights:
    raw = config.get("Loss", {})
    return PanoVGGTMatchingLossWeights(
        nce=float(raw.get("nce_weight", 1.0)),
        confidence=float(raw.get("confidence_weight", 0.1)),
        spherical=float(raw.get("spherical_regression_weight", 0.2)),
        static=float(raw.get("static_weight", 0.05)),
        sky=float(raw.get("sky_weight", 0.2)),
        sky_dice=float(raw.get("sky_dice_weight", 0.5)),
        smoothness=float(raw.get("smoothness_weight", 0.01)),
        temperature=float(raw.get("temperature", 0.07)),
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _resize_non_sky_mask(sample_sky_mask: torch.Tensor | None, feature_hw: tuple[int, int]) -> torch.Tensor | None:
    if sample_sky_mask is None:
        return None
    mask = (~sample_sky_mask.bool()).float()
    flat = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    flat = F.interpolate(flat, size=feature_hw, mode="nearest")
    return flat.view(mask.shape[0], mask.shape[1], 1, feature_hw[0], feature_hw[1])


def save_sky_head_checkpoint(
    path: str | Path,
    *,
    wrapper: PanoVGGTMatchingSkyHead,
    config: dict[str, Any],
    global_step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a standalone sky-head checkpoint."""

    payload = {
        "format": "panovggt_m3_sphere_sky_head_v1",
        "sky_mask_head": wrapper.sky_head.state_dict(),
        "head_config": wrapper.head_config(),
        "class_map": dict(config.get("Dataset", {}).get("class_map", {})),
        "feature_hook": config.get("Model", {}).get("feature_hook"),
        "training_config": config,
        "global_step": int(global_step),
        "metrics": metrics or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_matching_head_checkpoint(
    path: str | Path,
    *,
    wrapper: PanoVGGTMatchingSkyHead,
    config: dict[str, Any],
    global_step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a standalone matching-head checkpoint."""

    payload = {
        "format": "panovggt_m3_sphere_matching_head_v1",
        "matching_head": wrapper.matching_head.state_dict(),
        "descriptor_dim": int(wrapper.descriptor_dim),
        "head_config": wrapper.head_config(),
        "feature_hook": config.get("Model", {}).get("feature_hook"),
        "training_config": config,
        "global_step": int(global_step),
        "metrics": metrics or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_combined_head_bundle(
    path: str | Path,
    *,
    wrapper: PanoVGGTMatchingSkyHead,
    config: dict[str, Any],
) -> None:
    """Save a combined bundle for later inference adapter loading."""

    payload = {
        "format": "panovggt_m3_sphere_matching_sky_bundle_v1",
        "matching_head": wrapper.matching_head.state_dict(),
        "sky_mask_head": wrapper.sky_head.state_dict(),
        "descriptor_dim": int(wrapper.descriptor_dim),
        "class_map": dict(config.get("Dataset", {}).get("class_map", {})),
        "head_config": wrapper.head_config(),
        "feature_hook": config.get("Model", {}).get("feature_hook"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_head_checkpoint(path: str | Path, wrapper: PanoVGGTMatchingSkyHead, *, strict: bool = True) -> dict[str, Any]:
    """Load a sky, matching, or combined head checkpoint into ``wrapper``."""

    payload = torch.load(path, map_location="cpu")
    fmt = payload.get("format")
    if fmt == "panovggt_m3_sphere_sky_head_v1":
        wrapper.sky_head.load_state_dict(payload["sky_mask_head"], strict=strict)
    elif fmt == "panovggt_m3_sphere_matching_head_v1":
        wrapper.matching_head.load_state_dict(payload["matching_head"], strict=strict)
    elif fmt == "panovggt_m3_sphere_matching_sky_bundle_v1":
        wrapper.matching_head.load_state_dict(payload["matching_head"], strict=strict)
        wrapper.sky_head.load_state_dict(payload["sky_mask_head"], strict=strict)
    else:
        raise ValueError(f"Unsupported PanoVGGT-M3-Sphere head checkpoint format: {fmt!r}")
    return payload


def _init_wandb(config: dict[str, Any], output_dir: Path):
    wb_cfg = config.get("WeightsAndBiases", {})
    if not bool(wb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WeightsAndBiases.enabled=true requires wandb.") from exc
    mode = str(wb_cfg.get("mode", "online"))
    try:
        return wandb.init(
            project=str(wb_cfg.get("project", "360Droid-splat")),
            entity=wb_cfg.get("entity"),
            name=wb_cfg.get("run_name"),
            mode=mode,
            dir=str(output_dir),
            config=config,
            tags=wb_cfg.get("tags"),
        )
    except Exception as exc:
        if mode == "offline":
            raise
        print(f"W&B online init failed, falling back to offline mode: {exc}")
        return wandb.init(
            project=str(wb_cfg.get("project", "360Droid-splat")),
            entity=wb_cfg.get("entity"),
            name=wb_cfg.get("run_name"),
            mode="offline",
            dir=str(output_dir),
            config=config,
            tags=wb_cfg.get("tags"),
        )


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        else:
            out[key] = float(value)
    return out


def _build_head_and_optimizer(
    config: dict[str, Any],
    *,
    feature_dim: int,
    mode: str,
    device: torch.device,
) -> tuple[PanoVGGTMatchingSkyHead, torch.optim.Optimizer, list[torch.nn.Parameter]]:
    heads_cfg = config.get("Heads", {})
    flags = _mode_head_flags(mode, heads_cfg)
    wrapper = PanoVGGTMatchingSkyHead(
        int(feature_dim),
        descriptor_dim=int(heads_cfg.get("descriptor_dim", 24)),
        hidden_dim=int(heads_cfg.get("hidden_dim", 128)),
        num_conv_blocks=int(heads_cfg.get("num_conv_blocks", 2)),
        feature_key=heads_cfg.get("feature_key"),
        **flags,
    ).to(device)
    resume = config.get("Training", {}).get("resume_checkpoint")
    if resume:
        load_head_checkpoint(resume, wrapper, strict=False)
    _set_trainable_for_mode(wrapper, mode)
    trainable = [param for param in wrapper.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters for mode {mode}.")
    opt_cfg = config.get("Optimizer", {})
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg.get("lr", 2.0e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    return wrapper, optimizer, trainable


def _descriptor_recall_metrics(
    dense_descriptors: torch.Tensor,
    corr: Any,
    *,
    image_hw: tuple[int, int],
    search_radius: int = 2,
    max_correspondences: int = 256,
) -> dict[str, torch.Tensor]:
    descriptors = dense_descriptors[0] if dense_descriptors.ndim == 5 else dense_descriptors
    valid = corr.valid_mask.reshape(-1).bool()
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        zero = descriptors.sum() * 0.0
        return {
            "match_recall_0_1deg": zero.detach(),
            "match_recall_0_5deg": zero.detach(),
            "match_recall_1deg": zero.detach(),
            "match_angular_error_deg": zero.detach(),
        }
    if valid_idx.numel() > int(max_correspondences):
        keep = torch.linspace(0, valid_idx.numel() - 1, steps=int(max_correspondences), device=valid_idx.device).round().long()
        valid_idx = valid_idx[keep]
    src_idx = corr.src_indices.reshape(-1).long()[valid_idx]
    tgt_idx = corr.tgt_indices.reshape(-1).long()[valid_idx]
    src_uv = corr.src_uv.reshape(-1, 2)[valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    tgt_uv = corr.tgt_uv.reshape(-1, 2)[valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    gt_bearing = corr.tgt_bearing.reshape(-1, 3)[valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    src = sample_feature_values(descriptors, src_idx, src_uv)
    src = F.normalize(src, dim=-1, eps=1.0e-6)
    height_f, width_f = int(descriptors.shape[-2]), int(descriptors.shape[-1])
    offsets = [(float(dx), float(dy)) for dy in range(-int(search_radius), int(search_radius) + 1) for dx in range(-int(search_radius), int(search_radius) + 1)]
    offset_t = torch.tensor(offsets, device=descriptors.device, dtype=descriptors.dtype)
    candidates = tgt_uv.unsqueeze(1) + offset_t.view(1, -1, 2)
    candidates = candidates.clone()
    candidates[..., 0] = torch.remainder(candidates[..., 0], float(width_f))
    candidates[..., 1] = candidates[..., 1].clamp(0.5, float(height_f) - 0.5)
    target_maps = descriptors[tgt_idx]
    norm_x = 2.0 * (candidates[..., 0] - 0.5) / max(width_f - 1, 1) - 1.0
    norm_y = 2.0 * (candidates[..., 1] - 0.5) / max(height_f - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(candidates.shape[0], candidates.shape[1], 1, 2)
    sampled = F.grid_sample(target_maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled[:, :, :, 0].transpose(1, 2)
    sampled = F.normalize(sampled, dim=-1, eps=1.0e-6)
    best = (sampled * src.unsqueeze(1)).sum(dim=-1).argmax(dim=-1)
    pred_uv = candidates[torch.arange(candidates.shape[0], device=descriptors.device), best]
    pred_image_uv = feature_uv_to_image_uv(pred_uv, (height_f, width_f), image_hw)
    pred_bearing = erp_pixel_to_bearing(pred_image_uv, int(image_hw[0]), int(image_hw[1])).to(descriptors)
    residual_deg = torch.rad2deg(spherical_tangent_residual(gt_bearing, pred_bearing).norm(dim=-1))
    return {
        "match_recall_0_1deg": (residual_deg <= 0.1).float().mean().detach(),
        "match_recall_0_5deg": (residual_deg <= 0.5).float().mean().detach(),
        "match_recall_1deg": (residual_deg <= 1.0).float().mean().detach(),
        "match_angular_error_deg": residual_deg.mean().detach(),
    }


def _compute_matching_batch_loss(
    *,
    outputs: dict[str, torch.Tensor],
    sample: dict[str, Any],
    features: torch.Tensor,
    loss_fn: PanoVGGTMatchingSkyLoss,
    config: dict[str, Any],
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[Any]]:
    feature_hw = (int(features.shape[-2]), int(features.shape[-1]))
    pairs_cfg = config.get("Pairs", {})
    total = features.sum() * 0.0
    metrics_accum: dict[str, torch.Tensor] = {}
    correspondences: list[Any] = []
    non_sky = None
    if bool(pairs_cfg.get("sky_filter", True)) and torch.is_tensor(sample.get("sky_mask")):
        non_sky = _resize_non_sky_mask(sample["sky_mask"], feature_hw)
    batch_size = int(features.shape[0])
    for b in range(batch_size):
        pair_indices = sample["pair_indices"][b] if sample["pair_indices"].ndim == 3 else sample["pair_indices"]
        corr = generate_gt_spherical_correspondences(
            sample["depths"][b],
            sample["poses_c2w"][b],
            pair_indices,
            feature_hw,
            image_hw,
            samples_per_edge=pairs_cfg.get("samples_per_edge"),
            min_baseline_deg=float(pairs_cfg.get("min_baseline_deg", 0.0)),
            max_baseline_deg=float(pairs_cfg.get("max_baseline_deg", 60.0)),
        )
        correspondences.append(corr)
        out_b = {key: value[b : b + 1] for key, value in outputs.items() if torch.is_tensor(value)}
        mask_b = non_sky[b : b + 1] if non_sky is not None else None
        loss_b, metrics_b = loss_fn.matching_only(out_b, corr, image_hw=image_hw, non_sky_mask=mask_b)
        recall_b = _descriptor_recall_metrics(out_b["dense_descriptors"], corr, image_hw=image_hw)
        metrics_b.update(recall_b)
        total = total + loss_b / float(batch_size)
        for key, value in metrics_b.items():
            metrics_accum[key] = metrics_accum.get(key, torch.zeros_like(value)) + value.detach() / float(batch_size)
    return total, metrics_accum, correspondences


def _image_from_tensor(image: torch.Tensor) -> Image.Image:
    arr = image.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 3 and int(arr.shape[0]) == 3:
        arr = arr.permute(1, 2, 0)
    data = (arr.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(data, mode="RGB")


def _mask_overlay(image: Image.Image, mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    m = mask.detach().float().cpu()
    if m.ndim == 3:
        m = m[0]
    if tuple(m.shape[-2:]) != (image.height, image.width):
        resized = F.interpolate(m.view(1, 1, int(m.shape[-2]), int(m.shape[-1])), size=(image.height, image.width), mode="bilinear", align_corners=False)
        m = resized[0, 0]
    alpha = m.clamp(0.0, 1.0).numpy()[..., None] * 0.45
    overlay = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    data = base * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(data.clip(0, 255).astype(np.uint8), mode="RGB")


def _save_sky_visualization(
    *,
    sample: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    path: Path,
) -> Path | None:
    if "sky_prob" not in outputs or not torch.is_tensor(sample.get("sky_mask")):
        return None
    image = _image_from_tensor(sample["images"][0, 0])
    gt = _mask_overlay(image, sample["sky_mask"][0, 0, 0], (0, 200, 255))
    pred = _mask_overlay(image, outputs["sky_prob"][0, 0, 0], (255, 80, 60))
    canvas = Image.new("RGB", (image.width * 3, image.height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    canvas.paste(gt, (image.width, 0))
    canvas.paste(pred, (image.width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, label in enumerate(("rgb", "sky_gt", "sky_pred")):
        draw.rectangle((idx * image.width + 6, 6, idx * image.width + 92, 28), fill=(0, 0, 0))
        draw.text((idx * image.width + 12, 10), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _predict_match_uv(
    dense_descriptors: torch.Tensor,
    corr: Any,
    *,
    max_matches: int,
    search_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    descriptors = dense_descriptors[0] if dense_descriptors.ndim == 5 else dense_descriptors
    valid_idx = torch.nonzero(corr.valid_mask.reshape(-1).bool(), as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return None
    all_src_idx = corr.src_indices.reshape(-1).long()
    all_tgt_idx = corr.tgt_indices.reshape(-1).long()
    first_src = all_src_idx[valid_idx[0]]
    first_tgt = all_tgt_idx[valid_idx[0]]
    same_pair = (all_src_idx[valid_idx] == first_src) & (all_tgt_idx[valid_idx] == first_tgt)
    valid_idx = valid_idx[same_pair]
    if valid_idx.numel() > int(max_matches):
        keep = torch.linspace(0, valid_idx.numel() - 1, steps=int(max_matches), device=valid_idx.device).round().long()
        valid_idx = valid_idx[keep]
    src_idx = all_src_idx[valid_idx]
    tgt_idx = all_tgt_idx[valid_idx]
    src_uv = corr.src_uv.reshape(-1, 2)[valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    tgt_uv = corr.tgt_uv.reshape(-1, 2)[valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    src = F.normalize(sample_feature_values(descriptors, src_idx, src_uv), dim=-1, eps=1.0e-6)
    height_f, width_f = int(descriptors.shape[-2]), int(descriptors.shape[-1])
    offsets = [(float(dx), float(dy)) for dy in range(-int(search_radius), int(search_radius) + 1) for dx in range(-int(search_radius), int(search_radius) + 1)]
    offset_t = torch.tensor(offsets, device=descriptors.device, dtype=descriptors.dtype)
    candidates = tgt_uv.unsqueeze(1) + offset_t.view(1, -1, 2)
    candidates = candidates.clone()
    candidates[..., 0] = torch.remainder(candidates[..., 0], float(width_f))
    candidates[..., 1] = candidates[..., 1].clamp(0.5, float(height_f) - 0.5)
    target_maps = descriptors[tgt_idx]
    norm_x = 2.0 * (candidates[..., 0] - 0.5) / max(width_f - 1, 1) - 1.0
    norm_y = 2.0 * (candidates[..., 1] - 0.5) / max(height_f - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(candidates.shape[0], candidates.shape[1], 1, 2)
    sampled = F.grid_sample(target_maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = F.normalize(sampled[:, :, :, 0].transpose(1, 2), dim=-1, eps=1.0e-6)
    best = (sampled * src.unsqueeze(1)).sum(dim=-1).argmax(dim=-1)
    pred_uv = candidates[torch.arange(candidates.shape[0], device=descriptors.device), best]
    return src_idx.detach(), tgt_idx.detach(), src_uv.detach(), tgt_uv.detach(), pred_uv.detach()


def _save_matching_visualization(
    *,
    sample: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    corr: Any | None,
    image_hw: tuple[int, int],
    path: Path,
    max_matches: int,
) -> Path | None:
    if corr is None or "dense_descriptors" not in outputs:
        return None
    prediction = _predict_match_uv(outputs["dense_descriptors"], corr, max_matches=max_matches)
    if prediction is None:
        return None
    src_idx, tgt_idx, src_uv, tgt_uv, pred_uv = prediction
    src_frame = int(src_idx[0].detach().cpu()) if src_idx.numel() else 0
    tgt_frame = int(tgt_idx[0].detach().cpu()) if tgt_idx.numel() else 0
    image_src = _image_from_tensor(sample["images"][0, src_frame])
    image_tgt = _image_from_tensor(sample["images"][0, tgt_frame])
    canvas = Image.new("RGB", (image_src.width + image_tgt.width, max(image_src.height, image_tgt.height)), (0, 0, 0))
    canvas.paste(image_src, (0, 0))
    canvas.paste(image_tgt, (image_src.width, 0))
    draw = ImageDraw.Draw(canvas)
    feature_hw = tuple(int(v) for v in outputs["dense_descriptors"].shape[-2:])
    src_img_uv = feature_uv_to_image_uv(src_uv.cpu(), feature_hw, image_hw)
    tgt_img_uv = feature_uv_to_image_uv(tgt_uv.cpu(), feature_hw, image_hw)
    pred_img_uv = feature_uv_to_image_uv(pred_uv.cpu(), feature_hw, image_hw)
    for s, gt, pred in zip(src_img_uv, tgt_img_uv, pred_img_uv):
        sx, sy = float(s[0]), float(s[1])
        gx, gy = float(gt[0]) + image_src.width, float(gt[1])
        px, py = float(pred[0]) + image_src.width, float(pred[1])
        draw.line((sx, sy, px, py), fill=(255, 180, 0), width=1)
        draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(80, 180, 255))
        draw.ellipse((gx - 3, gy - 3, gx + 3, gy + 3), outline=(0, 255, 80), width=2)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 60, 60))
    draw.rectangle((6, 6, 205, 28), fill=(0, 0, 0))
    draw.text((12, 10), "blue=src green=gt red=pred", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _log_visualization_paths(wandb_run: Any, paths: dict[str, Path | None], *, step: int) -> None:
    if wandb_run is None:
        return
    import wandb

    payload = {}
    for key, path in paths.items():
        if path is not None and path.exists():
            payload[f"visualization/{key}"] = wandb.Image(str(path))
    if payload:
        wandb_run.log(payload, step=step)


def _maybe_write_visualizations(
    *,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    mode: str,
    sample: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    correspondences: list[Any] | None,
    image_hw: tuple[int, int],
    wandb_run: Any,
) -> None:
    vis_cfg = config.get("Visualization", {})
    if not bool(vis_cfg.get("enabled", False)):
        return
    interval = max(1, int(vis_cfg.get("interval", 100)))
    if step != 1 and step % interval != 0:
        return
    vis_dir = output_dir / str(vis_cfg.get("save_dir", "visualizations"))
    paths: dict[str, Path | None] = {}
    if mode in ("sky_only", "head_joint_calibration"):
        paths["sky_segmentation"] = _save_sky_visualization(
            sample=sample,
            outputs=outputs,
            path=vis_dir / f"step_{step:07d}_sky.png",
        )
    if mode in ("matching_only", "head_joint_calibration"):
        paths["matching"] = _save_matching_visualization(
            sample=sample,
            outputs=outputs,
            corr=correspondences[0] if correspondences else None,
            image_hw=image_hw,
            path=vis_dir / f"step_{step:07d}_matching.png",
            max_matches=int(vis_cfg.get("max_matches", 40)),
        )
    _log_visualization_paths(wandb_run, paths, step=step)


def _run_validation(
    *,
    config: dict[str, Any],
    wrapper: PanoVGGTMatchingSkyHead,
    feature_extractor: nn.Module,
    loss_fn: PanoVGGTMatchingSkyLoss,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    output_dir: Path,
    step: int,
    wandb_run: Any,
) -> dict[str, float]:
    val_cfg = config.get("Validation", {})
    if not bool(val_cfg.get("enabled", False)):
        return {}
    max_batches = max(1, int(val_cfg.get("max_batches", 1)))
    was_training = wrapper.training
    wrapper.eval()
    accum: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            sample = _to_device(batch, device)
            validate_training_sample(sample, mode, allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            images = sample["images"].float()
            features = feature_extractor(images)
            outputs = wrapper(features)
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            correspondences: list[Any] | None = None
            if mode == "sky_only":
                loss, metrics_t = loss_fn.sky_only(outputs, sample)
            elif mode == "matching_only":
                loss, metrics_t, correspondences = _compute_matching_batch_loss(
                    outputs=outputs,
                    sample=sample,
                    features=features,
                    loss_fn=loss_fn,
                    config=config,
                    image_hw=image_hw,
                )
            else:
                match_loss, metrics_t, correspondences = _compute_matching_batch_loss(
                    outputs=outputs,
                    sample=sample,
                    features=features,
                    loss_fn=loss_fn,
                    config=config,
                    image_hw=image_hw,
                )
                sky_loss, sky_metrics = loss_fn.sky_only(outputs, sample)
                loss = match_loss + sky_loss
                metrics_t.update({f"sky_{key}": value for key, value in sky_metrics.items()})
            metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value)
            count += 1
            if count == 1:
                _maybe_write_visualizations(
                    config=config,
                    output_dir=output_dir,
                    step=step,
                    mode=mode,
                    sample=sample,
                    outputs=outputs,
                    correspondences=correspondences,
                    image_hw=image_hw,
                    wandb_run=wandb_run,
                )
            if count >= max_batches:
                break
    if was_training:
        wrapper.train()
    metrics = {f"val/{key}": value / max(count, 1) for key, value in accum.items()}
    if wandb_run is not None and metrics:
        wandb_run.log(metrics, step=step)
    return metrics


def train_matching(config: dict[str, Any]) -> dict[str, Any]:
    """Run staged PanoVGGT-M3-Sphere head training."""

    torch.manual_seed(int(config.get("Training", {}).get("seed", 1234)))
    mode = normalize_training_mode(str(config.get("Training", {}).get("mode", "matching_only")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_matching_dataset_from_config(config)
    tr_cfg = config.get("Training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tr_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )
    val_cfg = config.get("Validation", {})
    val_loader = None
    if bool(val_cfg.get("enabled", False)):
        val_loader = DataLoader(
            dataset,
            batch_size=int(val_cfg.get("batch_size", tr_cfg.get("batch_size", 1))),
            shuffle=False,
            num_workers=int(val_cfg.get("num_workers", 0)),
            collate_fn=matching_collate,
            drop_last=False,
        )
    feature_extractor = _build_feature_extractor(config, device=device)
    feature_extractor.eval()
    wrapper: PanoVGGTMatchingSkyHead | None = None
    optimizer: torch.optim.Optimizer | None = None
    trainable: list[torch.nn.Parameter] = []
    loss_fn = PanoVGGTMatchingSkyLoss(_loss_weights_from_config(config))
    output_dir = Path(tr_cfg.get("output_dir", "outputs/panovggt_m3_sphere_omni360"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(config, output_dir)
    max_steps = int(tr_cfg.get("steps", tr_cfg.get("max_steps", 1)))
    save_interval = max(1, int(tr_cfg.get("save_interval", 1000)))
    log_interval = max(1, int(tr_cfg.get("log_interval", 50)))
    val_interval = max(1, int(tr_cfg.get("val_interval", val_cfg.get("interval", 1000))))
    step = 0
    best = float("inf")
    latest_metrics: dict[str, float] = {}
    start = time.time()

    while step < max_steps:
        for batch in loader:
            sample = _to_device(batch, device)
            validate_training_sample(sample, mode, allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            images = sample["images"].float()
            with torch.no_grad():
                features = feature_extractor(images)
            if wrapper is None or optimizer is None:
                configured_dim = config.get("Model", {}).get("feature_dim") or config.get("Heads", {}).get("feature_dim")
                feature_dim = int(configured_dim) if configured_dim is not None else int(features.shape[2])
                if int(features.shape[2]) != feature_dim:
                    raise ValueError(f"Configured feature_dim={feature_dim} does not match extracted feature dim={int(features.shape[2])}.")
                wrapper, optimizer, trainable = _build_head_and_optimizer(
                    config,
                    feature_dim=feature_dim,
                    mode=mode,
                    device=device,
                )
                wrapper.train()
            outputs = wrapper(features)
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            optimizer.zero_grad(set_to_none=True)
            correspondences: list[Any] | None = None
            if mode == "sky_only":
                loss, metrics_t = loss_fn.sky_only(outputs, sample)
            else:
                loss, metrics_t, correspondences = _compute_matching_batch_loss(
                    outputs=outputs,
                    sample=sample,
                    features=features,
                    loss_fn=loss_fn,
                    config=config,
                    image_hw=image_hw,
                )
                if mode == "head_joint_calibration":
                    sky_loss, sky_metrics = loss_fn.sky_only(outputs, sample)
                    loss = loss + sky_loss
                    metrics_t.update({f"sky_{key}": value for key, value in sky_metrics.items()})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(tr_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            step += 1
            latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
            if wandb_run is not None and (step == 1 or step % int(config.get("WeightsAndBiases", {}).get("log_every", 10)) == 0):
                wandb_run.log({f"train/{key}": value for key, value in latest_metrics.items()}, step=step)
            if step == 1 or step % log_interval == 0:
                print(yaml.safe_dump({"step": step, "mode": mode, "metrics": latest_metrics}, sort_keys=False).strip())
            _maybe_write_visualizations(
                config=config,
                output_dir=output_dir,
                step=step,
                mode=mode,
                sample=sample,
                outputs=outputs,
                correspondences=correspondences,
                image_hw=image_hw,
                wandb_run=wandb_run,
            )
            if val_loader is not None and (step == 1 or step % val_interval == 0):
                val_metrics = _run_validation(
                    config=config,
                    wrapper=wrapper,
                    feature_extractor=feature_extractor,
                    loss_fn=loss_fn,
                    loader=val_loader,
                    mode=mode,
                    device=device,
                    output_dir=output_dir,
                    step=step,
                    wandb_run=wandb_run,
                )
                if val_metrics:
                    latest_metrics.update(val_metrics)
            if step % save_interval == 0 or step == max_steps:
                if mode == "sky_only":
                    save_sky_head_checkpoint(ckpt_dir / "sky_head.pt", wrapper=wrapper, config=config, global_step=step, metrics=latest_metrics)
                elif mode == "matching_only":
                    save_matching_head_checkpoint(ckpt_dir / "matching_head.pt", wrapper=wrapper, config=config, global_step=step, metrics=latest_metrics)
                else:
                    save_combined_head_bundle(ckpt_dir / "matching_sky_bundle.pt", wrapper=wrapper, config=config)
            if latest_metrics["loss"] < best:
                best = latest_metrics["loss"]
            if step >= max_steps:
                break
    if wrapper is None:
        raise RuntimeError("Training finished without initializing PanoVGGT-M3-Sphere heads.")
    if mode == "sky_only":
        checkpoint = ckpt_dir / "sky_head.pt"
        save_sky_head_checkpoint(checkpoint, wrapper=wrapper, config=config, global_step=step, metrics=latest_metrics)
    elif mode == "matching_only":
        checkpoint = ckpt_dir / "matching_head.pt"
        save_matching_head_checkpoint(checkpoint, wrapper=wrapper, config=config, global_step=step, metrics=latest_metrics)
    else:
        checkpoint = ckpt_dir / "matching_sky_bundle.pt"
        save_combined_head_bundle(checkpoint, wrapper=wrapper, config=config)
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "mode": mode,
        "steps": step,
        "best_loss": best,
        "last_metrics": latest_metrics,
        "checkpoint": str(checkpoint),
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    """Command-line entry point for staged head training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default=None, choices=["sky_only", "matching_only", "head_joint_calibration"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()
    config = load_matching_train_config(args.config)
    if args.mode is not None:
        config.setdefault("Training", {})["mode"] = args.mode
    if args.steps is not None:
        config.setdefault("Training", {})["steps"] = int(args.steps)
    if args.output_dir is not None:
        config.setdefault("Training", {})["output_dir"] = args.output_dir
    if args.run_name is not None:
        config.setdefault("WeightsAndBiases", {})["run_name"] = args.run_name
    if args.wandb_mode is not None:
        if args.wandb_mode == "disabled":
            config.setdefault("WeightsAndBiases", {})["enabled"] = False
        else:
            config.setdefault("WeightsAndBiases", {})["enabled"] = True
            config.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    if args.max_clips is not None:
        config.setdefault("Dataset", {})["max_clips"] = int(args.max_clips)
    result = train_matching(config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
