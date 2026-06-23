"""Fine-tune PanoVGGT geometry heads on explicit ERP depth/pose supervision."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.pano_vggt.engine import (
    ExternalPanoVGGTInferenceEngine,
    _ceil_size_to_multiple,
    _resize_images,
    _resize_prediction,
    normalize_panovggt_output,
)
from frontend.pano_vggt.matching_dataset import build_matching_dataset_from_config, validate_training_sample
from frontend.pano_vggt.panovggt_geometry_losses import (
    PanoVGGTGeometryLossWeights,
    build_erp_local_points,
    local_points_to_world,
    panovggt_geometry_loss,
    weights_from_config,
)
from frontend.pano_vggt.train_matching import _float_metrics, _init_wandb, _merge_config, matching_collate


def _default_config() -> dict[str, Any]:
    return {
        "Training": {
            "steps": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "frames_per_sample": 4,
            "num_workers": 0,
            "amp": False,
            "seed": 1234,
            "output_dir": "outputs/panovggt_geometry_finetune_smoke",
            "log_every": 1,
            "save_every": 100,
            "val_every": 100,
            "vis_every": 100,
            "grad_clip": 1.0,
            "device": None,
            "resume": None,
        },
        "Model": {
            "use_synthetic_model": True,
            "hidden_dim": 16,
            "panovggt_repo": None,
            "panovggt_config": None,
            "panovggt_checkpoint": None,
            "class_path": None,
            "model_kwargs": {},
            "image_size": None,
            "patch_multiple": 14,
            "amp": True,
            "input_batch_dim": True,
            "strict_checkpoint": False,
            "skip_dinov2_pretrain": False,
            "trainable_modules": [
                "point_decoder",
                "point_head",
                "global_points_decoder",
                "global_point_head",
                "camera_decoder",
                "camera_head",
            ],
            "strict_trainable_modules": True,
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
        "Loss": {
            "local_point_weight": 1.0,
            "global_point_weight": 0.5,
            "depth_weight": 0.2,
            "pose_rot_weight": 0.1,
            "pose_trans_weight": 0.1,
            "smooth_weight": 0.02,
        },
        "Optimizer": {
            "lr": 1.0e-4,
            "weight_decay": 0.01,
        },
        "Validation": {
            "enabled": False,
            "max_batches": 1,
            "batch_size": 1,
            "num_workers": 0,
        },
        "Visualization": {
            "enabled": False,
            "max_views": 4,
            "save_dir": "visualizations",
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": "panovggt_geometry_finetune",
            "mode": "disabled",
            "log_every": 10,
            "tags": ["panovggt", "geometry-finetune"],
        },
    }


def load_geometry_train_config(path: str | None) -> dict[str, Any]:
    config = _default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _merge_config(config, user)


def _device_from_config(config: dict[str, Any]) -> torch.device:
    value = config.get("Training", {}).get("device")
    if value:
        return torch.device(str(value))
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


class TinyPanoVGGTGeometryModel(nn.Module):
    """Small differentiable stand-in with PanoVGGT-like module names for tests."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.aggregator = nn.Conv2d(3, hidden, kernel_size=3, padding=1)
        self.point_decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.point_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.global_points_decoder = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.global_point_head = nn.Conv2d(hidden, 3, kernel_size=1)
        self.camera_decoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.camera_head = nn.Linear(hidden, 3)
        with torch.no_grad():
            self.point_head.bias.fill_(math.log(2.0))
            self.global_point_head.weight.zero_()
            self.global_point_head.bias.zero_()
            self.camera_head.weight.zero_()
            self.camera_head.bias.zero_()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"images must have shape BxVx3xHxW, got {tuple(images.shape)}")
        b, v, _, h, w = [int(x) for x in images.shape]
        flat = images.reshape(b * v, 3, h, w)
        feat = self.aggregator(flat)
        point_feat = self.point_decoder(feat)
        depth = self.point_head(point_feat).clamp(-6.0, 6.0).exp().view(b, v, 1, h, w).clamp_min(1.0e-6)
        local = build_erp_local_points(depth)
        camera_feat = self.camera_decoder(feat).view(b, v, -1)
        trans = 0.1 * torch.tanh(self.camera_head(camera_feat))
        eye = torch.eye(4, device=images.device, dtype=images.dtype).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        poses = eye.clone()
        poses[:, :, :3, 3] = trans
        world = local_points_to_world(local, poses)
        global_delta = 0.05 * torch.tanh(
            self.global_point_head(self.global_points_decoder(feat)).view(b, v, 3, h, w).permute(0, 1, 3, 4, 2)
        )
        global_points = world + global_delta
        return {
            "depth": depth,
            "local_points": local,
            "camera_poses": poses,
            "world_points": world,
            "global_points": global_points,
        }


class DifferentiablePanoVGGTGeometryModel(nn.Module):
    """Differentiable PanoVGGT wrapper for geometry fine-tuning."""

    def __init__(self, model_cfg: dict[str, Any], *, device: torch.device) -> None:
        super().__init__()
        self.device_ref = torch.device(device)
        self.use_synthetic_model = bool(model_cfg.get("use_synthetic_model", False))
        self.image_size = None
        image_size_raw = model_cfg.get("image_size")
        if image_size_raw is not None:
            self.image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        self.amp = bool(model_cfg.get("amp", True))
        self.input_batch_dim = bool(model_cfg.get("input_batch_dim", True))
        self.patch_multiple = int(model_cfg.get("patch_multiple", 14))
        self.engine: ExternalPanoVGGTInferenceEngine | None = None
        if self.use_synthetic_model:
            self.model = TinyPanoVGGTGeometryModel(hidden_dim=int(model_cfg.get("hidden_dim", 16))).to(self.device_ref)
        else:
            self.engine = ExternalPanoVGGTInferenceEngine(
                repo_path=model_cfg.get("panovggt_repo"),
                config_path=model_cfg.get("panovggt_config") or model_cfg.get("config_path"),
                checkpoint=model_cfg.get("panovggt_checkpoint"),
                class_path=model_cfg.get("class_path"),
                model_kwargs=dict(model_cfg.get("model_kwargs", {})),
                image_size=self.image_size,
                device=self.device_ref,
                amp=self.amp,
                input_batch_dim=self.input_batch_dim,
                strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
                skip_dinov2_pretrain=bool(model_cfg.get("skip_dinov2_pretrain", False)),
                patch_multiple=self.patch_multiple,
            )
            self.model = self.engine.model
        self.trainable_modules = set_trainable_modules(
            self.model,
            list(model_cfg.get("trainable_modules", [])),
            strict=bool(model_cfg.get("strict_trainable_modules", True)),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.float().to(self.device_ref)
        if self.use_synthetic_model:
            return self.model(_resize_images(images, self.image_size))
        if self.engine is None:
            raise RuntimeError("External PanoVGGT engine was not constructed.")
        images = _resize_images(images, self.image_size)
        outputs = [self._forward_external_one(images[idx]) for idx in range(int(images.shape[0]))]
        return _stack_prediction_dicts(outputs)

    def _forward_external_one(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.engine is None:
            raise RuntimeError("External PanoVGGT engine was not constructed.")
        target_size = tuple(int(v) for v in images.shape[-2:])
        model_size = _ceil_size_to_multiple(target_size, self.patch_multiple)
        model_images = _resize_images(images, model_size)
        model_input = model_images.unsqueeze(0) if self.input_batch_dim else model_images
        if self.amp and self.device_ref.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.engine._call_model(model_input)
        else:
            output = self.engine._call_model(model_input)
        pred = normalize_panovggt_output(output, model_images)
        pred = _resize_prediction(pred, target_size)
        return {
            "depth": pred.depth,
            "local_points": pred.local_points,
            "camera_poses": pred.poses_c2w,
            "world_points": pred.chunk_world_points,
            "global_points": pred.global_points,
        }

    @property
    def unwrapped_model(self) -> nn.Module:
        return self.model


def set_trainable_modules(model: nn.Module, module_names: list[str], *, strict: bool = True) -> list[str]:
    """Freeze all parameters, then unfreeze the requested module names."""

    for param in model.parameters():
        param.requires_grad_(False)
    modules = dict(model.named_modules())
    trainable: list[str] = []
    missing: list[str] = []
    for name in module_names:
        module = modules.get(str(name))
        if module is None:
            missing.append(str(name))
            continue
        any_param = False
        for param in module.parameters():
            param.requires_grad_(True)
            any_param = True
        if any_param:
            trainable.append(str(name))
    if missing and strict:
        raise ValueError(f"Requested trainable PanoVGGT modules were not found: {missing}")
    return trainable


def _stack_prediction_dicts(outputs: list[dict[str, torch.Tensor | None]]) -> dict[str, torch.Tensor | None]:
    if not outputs:
        raise ValueError("No predictions to stack.")
    result: dict[str, torch.Tensor | None] = {}
    for key in outputs[0]:
        values = [out.get(key) for out in outputs]
        if all(torch.is_tensor(value) for value in values):
            result[key] = torch.stack([value for value in values if torch.is_tensor(value)], dim=0)
        elif all(value is None for value in values):
            result[key] = None
        else:
            raise ValueError(f"Cannot stack prediction field {key!r}: mixed tensor/None values.")
    return result


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _validation_loader(config: dict[str, Any]) -> DataLoader | None:
    val_cfg = config.get("Validation", {})
    if not bool(val_cfg.get("enabled", False)):
        return None
    dataset = build_matching_dataset_from_config(config, split="val")
    return DataLoader(
        dataset,
        batch_size=int(val_cfg.get("batch_size", config.get("Training", {}).get("batch_size", 1))),
        shuffle=False,
        num_workers=int(val_cfg.get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )


def _trainable_param_groups(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable PanoVGGT parameters were selected.")
    opt_cfg = config.get("Optimizer", {})
    return torch.optim.AdamW(
        params,
        lr=float(opt_cfg.get("lr", 1.0e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
    )


def _count_params(model: nn.Module) -> tuple[int, int]:
    trainable, frozen = 0, 0
    for param in model.parameters():
        count = int(param.numel())
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    return trainable, frozen


def _check_required_sample(sample: dict[str, Any], config: dict[str, Any]) -> None:
    validate_training_sample(
        sample,
        "matching_only",
        allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)),
    )
    for key in ("depths", "valid_depth", "poses_c2w"):
        if not torch.is_tensor(sample.get(key)):
            raise ValueError(f"Geometry fine-tuning requires sample[{key!r}].")


@torch.no_grad()
def _run_validation(
    model: DifferentiablePanoVGGTGeometryModel,
    loader: DataLoader,
    weights: PanoVGGTGeometryLossWeights,
    config: dict[str, Any],
    *,
    device: torch.device,
    output_dir: Path,
    step: int,
    wandb_run: Any = None,
) -> dict[str, float]:
    model.eval()
    max_batches = max(1, int(config.get("Validation", {}).get("max_batches", 1)))
    sums: dict[str, float] = {}
    count = 0
    first_sample: dict[str, Any] | None = None
    first_pred: dict[str, torch.Tensor | None] | None = None
    for batch_idx, raw in enumerate(loader):
        if batch_idx >= max_batches:
            break
        sample = _to_device(raw, device)
        _check_required_sample(sample, config)
        pred = model(sample["images"])
        loss, metrics_t = panovggt_geometry_loss(pred, sample, weights)
        metrics = _float_metrics({"total_loss": loss.detach(), **metrics_t})
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if first_sample is None:
            first_sample = sample
            first_pred = pred
    out = {f"val/{key}": value / max(count, 1) for key, value in sums.items()}
    if bool(config.get("Visualization", {}).get("enabled", False)) and first_sample is not None and first_pred is not None:
        panel = _save_visualization(output_dir, first_sample, first_pred, step=step, split="val", config=config)
        if wandb_run is not None:
            import wandb

            wandb_run.log({"geometry/val_panel": wandb.Image(str(panel))}, step=step)
    model.train()
    return out


def save_geometry_checkpoint(
    path: str | Path,
    *,
    model: DifferentiablePanoVGGTGeometryModel,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    payload = {
        "format": "panovggt_geometry_finetune_v1",
        "model_state_dict": model.unwrapped_model.state_dict(),
        "trainable_modules": list(model.trainable_modules),
        "config": config,
        "global_step": int(step),
        "metrics": metrics or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_resume(path: str | None, model: DifferentiablePanoVGGTGeometryModel) -> int:
    if not path:
        return 0
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload.get("state_dict", payload.get("model")))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported geometry checkpoint payload: {path}")
    model.unwrapped_model.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=False)
    return int(payload.get("global_step", payload.get("step", 0)))


def _depth_to_pil(depth: torch.Tensor, mask: torch.Tensor | None = None) -> Image.Image:
    d = depth.detach().float().cpu()
    if d.ndim == 3:
        d = d[0]
    valid = torch.isfinite(d) & (d > 0.0)
    if mask is not None:
        m = mask.detach().cpu().bool()
        if m.ndim == 3:
            m = m[0]
        valid = valid & m
    if bool(valid.any()):
        lo = torch.quantile(d[valid], 0.02)
        hi = torch.quantile(d[valid], 0.98)
        norm = ((d - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    else:
        norm = torch.zeros_like(d)
    arr = (norm.numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="L").convert("RGB")


def _rgb_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().float().cpu().clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _make_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    if not images:
        raise ValueError("No images for panel.")
    width, height = images[0].size
    label_h = 18
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img.resize((width, height)), (idx * width, label_h))
        draw.text((idx * width + 4, 2), label, fill=(0, 0, 0))
    return canvas


def _save_visualization(
    output_dir: Path,
    sample: dict[str, Any],
    pred: dict[str, torch.Tensor | None],
    *,
    step: int,
    split: str,
    config: dict[str, Any],
) -> Path:
    vis_cfg = config.get("Visualization", {})
    max_views = min(int(vis_cfg.get("max_views", 4)), int(sample["images"].shape[1]))
    target_hw = tuple(int(x) for x in sample["depths"].shape[-2:])
    pred_depth = pred["depth"]
    if not torch.is_tensor(pred_depth):
        raise ValueError("Visualization requires pred['depth'].")
    if tuple(pred_depth.shape[-2:]) != target_hw:
        b, v = int(pred_depth.shape[0]), int(pred_depth.shape[1])
        flat = pred_depth.reshape(b * v, 1, pred_depth.shape[-2], pred_depth.shape[-1])
        flat = F.interpolate(flat.float(), size=target_hw, mode="bilinear", align_corners=False)
        pred_depth = flat.view(b, v, 1, target_hw[0], target_hw[1])
    valid = sample.get("valid_depth")
    if torch.is_tensor(valid) and torch.is_tensor(sample.get("sky_mask")):
        valid = valid & ~sample["sky_mask"].bool()
    views: list[Image.Image] = []
    labels: list[str] = []
    for row_name, source in (("rgb", sample["images"]), ("gt_depth", sample["depths"]), ("pred_depth", pred_depth)):
        row_images: list[Image.Image] = []
        row_labels: list[str] = []
        for idx in range(max_views):
            if row_name == "rgb":
                row_images.append(_rgb_to_pil(source[0, idx]))
            elif row_name == "gt_depth":
                row_images.append(_depth_to_pil(source[0, idx], valid[0, idx] if torch.is_tensor(valid) else None))
            else:
                row_images.append(_depth_to_pil(source[0, idx], valid[0, idx] if torch.is_tensor(valid) else None))
            row_labels.append(f"{row_name}_{idx}")
        row = _make_panel(row_images, row_labels)
        views.append(row)
        labels.append(row_name)
    panel_w = max(img.size[0] for img in views)
    panel_h = sum(img.size[1] for img in views)
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    y = 0
    for img in views:
        panel.paste(img, (0, y))
        y += img.size[1]
    save_dir = output_dir / str(vis_cfg.get("save_dir", "visualizations")) / split
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"step_{int(step):06d}.png"
    panel.save(path)
    return path


def train_panovggt_geometry(config: dict[str, Any], *, command: list[str] | None = None) -> dict[str, Any]:
    tr = config.get("Training", {})
    torch.manual_seed(int(tr.get("seed", 1234)))
    device = _device_from_config(config)
    output_dir = Path(tr.get("output_dir", "outputs/panovggt_geometry_finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if command is not None:
        (output_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")

    dataset = build_matching_dataset_from_config(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(tr.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(tr.get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )
    val_loader = _validation_loader(config)
    model = DifferentiablePanoVGGTGeometryModel(config.get("Model", {}), device=device).to(device)
    resume_step = _load_resume(tr.get("resume"), model)
    optimizer = _trainable_param_groups(model, config)
    weights = weights_from_config(config)
    wandb_run = _init_wandb(config, output_dir)
    trainable, frozen = _count_params(model.unwrapped_model)
    print(yaml.safe_dump({"device": str(device), "trainable_params": trainable, "frozen_params": frozen, "trainable_modules": model.trainable_modules}, sort_keys=False).strip(), flush=True)

    max_steps = int(tr.get("steps", 1))
    accum = max(1, int(tr.get("gradient_accumulation_steps", 1)))
    log_every = max(1, int(tr.get("log_every", 1)))
    save_every = max(1, int(tr.get("save_every", 100)))
    val_every = max(1, int(tr.get("val_every", 100)))
    vis_every = max(1, int(tr.get("vis_every", 100)))
    grad_clip = float(tr.get("grad_clip", 1.0))
    step = int(resume_step)
    micro_step = 0
    best = float("inf")
    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    start = time.time()
    model.train()
    while step < max_steps:
        for raw in loader:
            sample = _to_device(raw, device)
            _check_required_sample(sample, config)
            pred = model(sample["images"])
            loss, metrics_t = panovggt_geometry_loss(pred, sample, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite geometry loss at optimizer step {step}: {loss}")
            (loss / float(accum)).backward()
            micro_step += 1
            if micro_step % accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite grad norm at optimizer step {step}: {grad_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = _float_metrics({"step": float(step), **metrics_t, "grad_norm": grad_norm.detach()})
            last_metrics = metrics
            if step % log_every == 0:
                print("step: " + str(step), flush=True)
                print(yaml.safe_dump({"metrics": metrics}, sort_keys=False).strip(), flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)
            if bool(config.get("Visualization", {}).get("enabled", False)) and step % vis_every == 0:
                panel = _save_visualization(output_dir, sample, pred, step=step, split="train", config=config)
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({"geometry/train_panel": wandb.Image(str(panel))}, step=step)
            val_metrics: dict[str, float] = {}
            if val_loader is not None and step % val_every == 0:
                val_metrics = _run_validation(
                    model,
                    val_loader,
                    weights,
                    config,
                    device=device,
                    output_dir=output_dir,
                    step=step,
                    wandb_run=wandb_run,
                )
                print(yaml.safe_dump({"validation": val_metrics}, sort_keys=False).strip(), flush=True)
                if wandb_run is not None:
                    wandb_run.log(val_metrics, step=step)
            score = val_metrics.get("val/total_loss", metrics.get("total_loss", float("inf")))
            if score < best:
                best = score
                save_geometry_checkpoint(output_dir / "best_geometry.pt", model=model, config=config, step=step, metrics={**metrics, **val_metrics})
            if step % save_every == 0:
                save_geometry_checkpoint(output_dir / f"checkpoint_{step:06d}.pt", model=model, config=config, step=step, metrics={**metrics, **val_metrics})
            save_geometry_checkpoint(output_dir / "last_geometry.pt", model=model, config=config, step=step, metrics={**metrics, **val_metrics})
            if step >= max_steps:
                break
    elapsed = time.time() - start
    result = {
        "output_dir": str(output_dir),
        "steps": step,
        "micro_steps": micro_step,
        "optimizer_steps": step - int(resume_step),
        "best_loss": best,
        "elapsed_sec": elapsed,
        "last_metrics": last_metrics,
    }
    (output_dir / "summary.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    config = load_geometry_train_config(args.config)
    train_panovggt_geometry(config, command=sys.argv if argv is None else argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
