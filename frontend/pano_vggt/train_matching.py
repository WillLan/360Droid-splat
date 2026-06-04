"""Train staged PanoVGGT-M3-Sphere matching and sky heads."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .matching_dataset import build_matching_dataset_from_config, normalize_training_mode, validate_training_sample
from .matching_head import PanoVGGTMatchingSkyHead
from .matching_losses import PanoVGGTMatchingLossWeights, PanoVGGTMatchingSkyLoss
from .spherical_correspondence import generate_gt_spherical_correspondences


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
        repo = model_cfg.get("panovggt_repo")
        if repo:
            repo_path = str(Path(repo).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        class_path = model_cfg.get("class_path")
        if not class_path:
            raise ValueError("Model.class_path is required for real PanoVGGT feature extraction.")
        module_name, attr_name = str(class_path).rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), attr_name)
        self.model = cls()
        checkpoint = model_cfg.get("panovggt_checkpoint")
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu")
            state = payload
            for key in ("state_dict", "model", "model_state_dict", "net"):
                if isinstance(payload, dict) and key in payload:
                    state = payload[key]
                    break
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported PanoVGGT checkpoint payload: {checkpoint}")
            self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=False)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        modules = dict(self.model.named_modules())
        if str(feature_hook) not in modules:
            raise ValueError(f"Model.feature_hook={feature_hook!r} was not found in external PanoVGGT modules.")
        self._feature: torch.Tensor | None = None

        def hook_fn(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output):
                self._feature = output
            elif isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
                self._feature = output[0]
            else:
                raise TypeError(f"Feature hook {feature_hook!r} returned unsupported output type {type(output)!r}.")

        modules[str(feature_hook)].register_forward_hook(hook_fn)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen external model and return the captured feature tensor."""

        self._feature = None
        with torch.no_grad():
            _ = self.model(images)
        if self._feature is None:
            raise RuntimeError("External PanoVGGT feature hook did not capture any tensor.")
        feature = self._feature
        if feature.ndim != 5:
            raise ValueError(f"Captured PanoVGGT feature must be BxNxCxHxW, got {tuple(feature.shape)}")
        return feature


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
    return wandb.init(
        project=str(wb_cfg.get("project", "360Droid-splat")),
        entity=wb_cfg.get("entity"),
        name=wb_cfg.get("run_name"),
        mode=str(wb_cfg.get("mode", "online")),
        dir=str(output_dir),
        config=config,
    )


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        else:
            out[key] = float(value)
    return out


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
    feature_extractor = _build_feature_extractor(config, device=device)
    feature_extractor.eval()
    heads_cfg = config.get("Heads", {})
    flags = _mode_head_flags(mode, heads_cfg)
    feature_dim = int(config.get("Model", {}).get("feature_dim") or heads_cfg.get("feature_dim") or 16)
    wrapper = PanoVGGTMatchingSkyHead(
        feature_dim,
        descriptor_dim=int(heads_cfg.get("descriptor_dim", 24)),
        hidden_dim=int(heads_cfg.get("hidden_dim", 128)),
        num_conv_blocks=int(heads_cfg.get("num_conv_blocks", 2)),
        feature_key=heads_cfg.get("feature_key"),
        **flags,
    ).to(device)
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
    loss_fn = PanoVGGTMatchingSkyLoss(_loss_weights_from_config(config))
    output_dir = Path(tr_cfg.get("output_dir", "outputs/panovggt_m3_sphere_omni360"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(config, output_dir)
    max_steps = int(tr_cfg.get("steps", tr_cfg.get("max_steps", 1)))
    save_interval = max(1, int(tr_cfg.get("save_interval", 1000)))
    log_interval = max(1, int(tr_cfg.get("log_interval", 50)))
    pairs_cfg = config.get("Pairs", {})
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
            outputs = wrapper(features)
            feature_hw = (int(features.shape[-2]), int(features.shape[-1]))
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            optimizer.zero_grad(set_to_none=True)
            if mode == "sky_only":
                loss, metrics_t = loss_fn.sky_only(outputs, sample)
            else:
                total = features.sum() * 0.0
                metrics_accum: dict[str, torch.Tensor] = {}
                non_sky = None
                if bool(pairs_cfg.get("sky_filter", True)) and torch.is_tensor(sample.get("sky_mask")):
                    non_sky = _resize_non_sky_mask(sample["sky_mask"], feature_hw)
                batch_size = int(images.shape[0])
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
                    out_b = {key: value[b : b + 1] for key, value in outputs.items()}
                    mask_b = non_sky[b : b + 1] if non_sky is not None else None
                    loss_b, metrics_b = loss_fn.matching_only(out_b, corr, image_hw=image_hw, non_sky_mask=mask_b)
                    total = total + loss_b / float(batch_size)
                    for key, value in metrics_b.items():
                        metrics_accum[key] = metrics_accum.get(key, torch.zeros_like(value)) + value.detach() / float(batch_size)
                loss = total
                metrics_t = metrics_accum
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
    args = parser.parse_args()
    config = load_matching_train_config(args.config)
    if args.mode is not None:
        config.setdefault("Training", {})["mode"] = args.mode
    if args.steps is not None:
        config.setdefault("Training", {})["steps"] = int(args.steps)
    if args.output_dir is not None:
        config.setdefault("Training", {})["output_dir"] = args.output_dir
    result = train_matching(config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
