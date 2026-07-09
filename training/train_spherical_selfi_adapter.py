"""Train the Stage 1 spherical Selfi DPT adapter.

This is a standalone training entry point. It does not modify or call the SLAM
runtime path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
from importlib import import_module
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from data.stage1_pano_sequence_dataset import Stage1PanoSequenceDataset, stage1_collate
from geometry.spherical_pseudo_correspondence import generate_spherical_pseudo_correspondence
from losses.spherical_selfi_alignment_loss import (
    SphericalSelfiAlignmentLoss,
    SphericalSelfiAlignmentLossConfig,
)
from models.panovggt_feature_wrapper import PanoVGGTFeatureWrapper
from models.spherical_selfi_dpt_adapter import SphericalSelfiDPTAdapter
from tools.visualize_spherical_adapter_matches import save_stage1_match_preview


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def default_config() -> dict[str, Any]:
    return {
        "image": {"height": 64, "width": 128},
        "panovggt": {
            "synthetic": True,
            "repo_path": None,
            "config_path": None,
            "checkpoint": None,
            "class_path": None,
            "model_kwargs": {},
            "stage_hooks": ["stage1", "stage2", "stage3", "stage4"],
            "feature_keys": [None, None, None, None],
            "token_hw": [None, None, None, None],
            "token_start_idx": [None, None, None, None],
            "in_channels": [8, 16, 24, 32],
            "use_no_grad": True,
            "allow_dataset_geometry_fallback": False,
            "strict_checkpoint": False,
            "skip_dinov2_pretrain": False,
        },
        "adapter": {
            "hidden_dim": 16,
            "out_dim": 24,
            "use_circular_padding": True,
            "norm_output": True,
        },
        "dataset": {
            "manifest": "data/stage1_dataset_manifest.json",
            "views_per_sample": 4,
            "domains": ["indoor", "outdoor"],
            "image_height": 64,
            "image_width": 128,
            "pair_mode": "adjacent_and_skip",
            "max_temporal_gap": 10,
        },
        "correspondence": {
            "num_query_per_pair": 128,
            "sampling": "cosine_latitude_weighted",
            "min_depth": 0.05,
            "max_depth": 100.0,
            "visibility_rel_thresh": 0.05,
        },
        "matching": {
            "mode": "global_fullres_spherical_ce",
            "loss_stride": 1,
            "local_window_radius": 16,
            "temperature": 0.07,
            "max_queries": 128,
            "soft_label_sigma_deg": 2.0,
            "ce_query_chunk_size": 32,
            "use_spherical_area_correction": True,
        },
        "loss": {"erp_aux_weight": 0.0, "spherical_match_weight": 1.0, "expected_geodesic_weight": 0.0},
        "train": {
            "batch_size": 1,
            "num_workers": 0,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "amp": False,
            "max_steps": 2,
            "log_interval": 1,
            "save_interval": 1,
            "val_interval": 1000,
            "max_val_batches": 8,
            "output_dir": "outputs/stage1_spherical_selfi_adapter",
            "grad_clip": 1.0,
            "seed": 1234,
        },
        "WeightsAndBiases": {"enabled": False, "project": "360Droid-splat", "mode": "online", "log_every": 50},
        "Visualization": {"enabled": False, "interval": 1000, "save_dir": "visualizations", "max_matches": 80},
    }


def load_config(path: str | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _deep_merge(config, user)


class _SyntheticStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = int(stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return self.conv(x)


class SyntheticStage1PanoVGGT(nn.Module):
    """Explicit synthetic frozen feature source for smoke tests only."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.stage1 = _SyntheticStage(3, channels[0], 1)
        self.stage2 = _SyntheticStage(channels[0], channels[1], 2)
        self.stage3 = _SyntheticStage(channels[1], channels[2], 2)
        self.stage4 = _SyntheticStage(channels[2], channels[3], 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w)
        f1 = self.stage1(flat)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        _ = self.stage4(f3)
        depth = torch.full((b, v, 1, h, w), 2.0, device=images.device, dtype=images.dtype)
        poses = torch.eye(4, device=images.device, dtype=images.dtype).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        return {"depth": depth, "poses_c2w": poses}


def _import_attr(path: str) -> Any:
    module_name, attr = path.rsplit(".", 1)
    return getattr(import_module(module_name), attr)


@contextmanager
def _maybe_skip_dinov2_pretrain(enabled: bool):
    if not enabled:
        yield
        return
    try:
        aggregator_mod = import_module("panovggt.models.aggregator")
        aggregator_cls = getattr(aggregator_mod, "Aggregator", None)
        original = getattr(aggregator_cls, "_try_load_dinov2", None) if aggregator_cls is not None else None
        if aggregator_cls is None or original is None:
            yield
            return

        def _skip(self, hub_name, url, patch_embed_key):
            return None

        aggregator_cls._try_load_dinov2 = _skip
        try:
            yield
        finally:
            aggregator_cls._try_load_dinov2 = original
    except ModuleNotFoundError:
        yield


def _instantiate_panovggt_model(cls: type, cfg: dict[str, Any]) -> nn.Module:
    kwargs = dict(cfg.get("model_kwargs", {}))
    config_path = cfg.get("config_path")
    if config_path is not None and cls.__name__ == "PanoVGGTModel":
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError("PanoVGGT config_path loading requires omegaconf.") from exc
        official = OmegaConf.load(str(config_path))
        OmegaConf.resolve(official)
        mc = official.model
        aggregator = OmegaConf.to_container(mc.aggregator, resolve=True)
        kwargs = {
            "img_size": int(official.img_size),
            "patch_size": int(official.patch_size),
            "embed_dim": int(official.embed_dim),
            "enable_camera": bool(mc.enable_camera),
            "enable_depth": bool(mc.enable_depth),
            "enable_point": bool(mc.enable_point),
            "aggregator": aggregator,
            **kwargs,
        }
    signature = inspect.signature(cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return cls(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**filtered)


def _build_external_model(cfg: dict[str, Any]) -> nn.Module:
    repo_path = cfg.get("repo_path")
    if repo_path:
        repo = str(Path(repo_path).expanduser().resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    class_path = cfg.get("class_path")
    if not class_path:
        raise ValueError("panovggt.class_path is required when panovggt.synthetic=false.")
    cls = _import_attr(str(class_path))
    with _maybe_skip_dinov2_pretrain(bool(cfg.get("skip_dinov2_pretrain", False))):
        model = _instantiate_panovggt_model(cls, cfg)
    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if isinstance(payload, dict) and key in payload:
                state = payload[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported PanoVGGT checkpoint payload: {checkpoint}")
        model.load_state_dict(
            {str(k).removeprefix("module."): v for k, v in state.items()},
            strict=bool(cfg.get("strict_checkpoint", False)),
        )
    return model


def build_panovggt_wrapper(config: dict[str, Any], *, device: torch.device) -> PanoVGGTFeatureWrapper:
    cfg = config.get("panovggt", {})
    channels = [int(value) for value in cfg.get("in_channels", [8, 16, 24, 32])]
    if bool(cfg.get("synthetic", False)):
        model = SyntheticStage1PanoVGGT(channels)
    else:
        model = _build_external_model(cfg)
    wrapper = PanoVGGTFeatureWrapper(
        model,
        stage_hooks=list(cfg.get("stage_hooks", [])),
        feature_keys=list(cfg.get("feature_keys", [None, None, None, None])),
        token_hw=list(cfg.get("token_hw", [None, None, None, None])),
        token_start_idx=list(cfg.get("token_start_idx", [None, None, None, None])),
        use_no_grad=bool(cfg.get("use_no_grad", True)),
    )
    return wrapper.to(device)


def build_adapter(config: dict[str, Any], *, device: torch.device) -> SphericalSelfiDPTAdapter:
    pano_cfg = config.get("panovggt", {})
    adapter_cfg = config.get("adapter", {})
    image_cfg = config.get("image", {})
    adapter = SphericalSelfiDPTAdapter(
        list(pano_cfg.get("in_channels", [8, 16, 24, 32])),
        hidden_dim=int(adapter_cfg.get("hidden_dim", 128)),
        out_dim=int(adapter_cfg.get("out_dim", 24)),
        image_height=int(image_cfg.get("height", config.get("dataset", {}).get("image_height", 504))),
        image_width=int(image_cfg.get("width", config.get("dataset", {}).get("image_width", 1008))),
        use_circular_padding=bool(adapter_cfg.get("use_circular_padding", True)),
        norm_output=bool(adapter_cfg.get("norm_output", True)),
        token_hw=list(pano_cfg.get("token_hw", [None, None, None, None])),
        reassemble_sizes=adapter_cfg.get("reassemble_sizes"),
        fusion_output_size=adapter_cfg.get("fusion_output_size"),
    )
    return adapter.to(device)


def build_dataset(config: dict[str, Any], *, split: str) -> Stage1PanoSequenceDataset:
    cfg = config.get("dataset", {})
    return Stage1PanoSequenceDataset(
        cfg.get("manifest"),
        split=split,
        domains=list(cfg.get("domains", ["indoor", "outdoor"])),
        views_per_sample=int(cfg.get("views_per_sample", 4)),
        image_height=int(cfg.get("image_height", config.get("image", {}).get("height", 504))),
        image_width=int(cfg.get("image_width", config.get("image", {}).get("width", 1008))),
        pair_mode=str(cfg.get("pair_mode", "adjacent_and_skip")),
        max_temporal_gap=cfg.get("max_temporal_gap", 10),
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _init_wandb(config: dict[str, Any], output_dir: Path):
    wb_cfg = config.get("WeightsAndBiases", {})
    if not bool(wb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WeightsAndBiases.enabled=true requires wandb.") from exc
    mode = str(wb_cfg.get("mode", "online"))
    kwargs = {
        "project": str(wb_cfg.get("project", "360Droid-splat")),
        "entity": wb_cfg.get("entity"),
        "name": wb_cfg.get("run_name"),
        "mode": mode,
        "dir": str(output_dir),
        "config": config,
        "tags": wb_cfg.get("tags"),
    }
    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        if mode == "online":
            raise RuntimeError("W&B online init failed; refusing to continue an online-required training run.") from exc
        raise


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _assert_adapter_finite(adapter: SphericalSelfiDPTAdapter, *, context: str) -> None:
    for name, param in adapter.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite adapter parameter {name!r} detected {context}.")


def _assert_adapter_grads_finite(adapter: SphericalSelfiDPTAdapter, *, context: str) -> None:
    for name, param in adapter.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Non-finite adapter gradient {name!r} detected {context}.")


def _assert_metrics_finite(metrics: dict[str, float], *, context: str) -> None:
    bad = [key for key, value in metrics.items() if not np.isfinite(float(value))]
    if bad:
        raise RuntimeError(f"Non-finite metrics {bad} detected {context}; refusing to continue.")


def _save_checkpoint(
    path: Path,
    adapter: SphericalSelfiDPTAdapter,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    best_val_angular_error: float | None = None,
) -> None:
    _assert_adapter_finite(adapter, context=f"before saving {path}")
    _assert_metrics_finite(metrics, context=f"before saving {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spherical_selfi_adapter_v1",
            "adapter": adapter.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "adapter_config": {
                "in_channels": adapter.in_channels,
                "hidden_dim": adapter.hidden_dim,
                "out_dim": adapter.out_dim,
                "image_height": adapter.image_height,
                "image_width": adapter.image_width,
                "use_circular_padding": adapter.use_circular_padding,
                "norm_output": adapter.norm_output,
                "reassemble_sizes": adapter.reassemble_sizes,
                "fusion_output_size": adapter.fusion_output_size,
            },
            "training_config": config,
            "global_step": int(step),
            "metrics": metrics,
            "best_val_angular_error": best_val_angular_error,
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    adapter: SphericalSelfiDPTAdapter,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device,
) -> tuple[int, dict[str, float], float | None]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "adapter" not in payload:
        raise ValueError(f"Unsupported Stage 1 adapter checkpoint: {path}")
    adapter.load_state_dict(payload["adapter"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    metrics = payload.get("metrics", {})
    return int(payload.get("global_step", 0)), dict(metrics), payload.get("best_val_angular_error")


def _valid_ratio(corr) -> torch.Tensor:
    valid = corr.valid_mask.float()
    return valid.mean() if valid.numel() else torch.tensor(0.0, device=valid.device)


def _make_correspondence(config: dict[str, Any], batch: dict[str, Any], pano_out, images: torch.Tensor):
    init_depth = pano_out.init_depth
    init_poses = pano_out.init_poses
    if init_depth is None or init_poses is None:
        if not bool(config.get("panovggt", {}).get("allow_dataset_geometry_fallback", False)):
            raise RuntimeError("PanoVGGT wrapper did not return init depth/poses for pseudo correspondence.")
        init_depth = batch.get("depths")
        init_poses = batch.get("poses_c2w")
    if init_depth is None or init_poses is None:
        raise RuntimeError("Stage 1 training requires init depth and c2w poses.")
    return generate_spherical_pseudo_correspondence(
        init_depth,
        init_poses,
        batch["pair_indices"],
        height=int(images.shape[-2]),
        width=int(images.shape[-1]),
        **dict(config.get("correspondence", {})),
    )


def _save_match_visualization(
    *,
    output_dir: Path,
    save_dir: str,
    split: str,
    step: int,
    images: torch.Tensor,
    matches: dict[str, torch.Tensor],
    max_matches: int,
) -> Path | None:
    if matches["src_uv"].numel() == 0:
        return None
    flat_images = images.detach().cpu().reshape(images.shape[0] * images.shape[1], *images.shape[2:])
    flat_src = matches["flat_src"].detach().cpu().long()
    flat_tgt = matches["flat_tgt"].detach().cpu().long()
    first_src, first_tgt = int(flat_src[0]), int(flat_tgt[0])
    same_pair = (flat_src == first_src) & (flat_tgt == first_tgt)
    src_uv = matches["src_uv"].detach().cpu()[same_pair]
    tgt_uv = matches["tgt_uv"].detach().cpu()[same_pair]
    pred_uv = matches["pred_uv"].detach().cpu()[same_pair]
    vis_dir = output_dir / str(save_dir) / split
    path = vis_dir / f"matches_step_{int(step):06d}.png"
    return save_stage1_match_preview(
        flat_images[first_src],
        flat_images[first_tgt],
        src_uv,
        tgt_uv,
        path,
        pred_tgt_uv=pred_uv,
        max_matches=int(max_matches),
    )


def train_spherical_selfi_adapter(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config.get("train", {}).get("seed", 1234)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, split="train")
    try:
        val_dataset = build_dataset(config, split="val")
    except ValueError:
        val_dataset = None
    train_cfg = config.get("train", {})
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage1_collate,
        drop_last=False,
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage1_collate,
        drop_last=False,
    )
    wrapper = build_panovggt_wrapper(config, device=device)
    adapter = build_adapter(config, device=device)
    loss_cfg = SphericalSelfiAlignmentLossConfig(
        mode=str(config.get("matching", {}).get("mode", "global_fullres_spherical_ce")),
        loss_stride=int(config.get("matching", {}).get("loss_stride", 1)),
        local_window_radius=int(config.get("matching", {}).get("local_window_radius", 16)),
        temperature=float(config.get("matching", {}).get("temperature", 0.07)),
        max_queries=config.get("matching", {}).get("max_queries", 512),
        erp_aux_weight=float(config.get("loss", {}).get("erp_aux_weight", 0.01)),
        soft_label_sigma_deg=float(config.get("matching", {}).get("soft_label_sigma_deg", 2.0)),
        expected_geodesic_weight=float(config.get("loss", {}).get("expected_geodesic_weight", 0.0)),
        ce_query_chunk_size=int(config.get("matching", {}).get("ce_query_chunk_size", 32)),
        use_spherical_area_correction=bool(config.get("matching", {}).get("use_spherical_area_correction", True)),
    )
    criterion = SphericalSelfiAlignmentLoss(loss_cfg)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    output_dir = Path(train_cfg.get("output_dir", "outputs/stage1_spherical_selfi_adapter"))
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(config, output_dir)
    max_steps = int(train_cfg.get("max_steps", 1))
    save_interval = max(1, int(train_cfg.get("save_interval", 1000)))
    log_interval = max(1, int(train_cfg.get("log_interval", 50)))
    val_interval = max(1, int(train_cfg.get("val_interval", 1000)))
    max_val_batches = max(1, int(train_cfg.get("max_val_batches", 8)))
    resume = train_cfg.get("resume")
    step = 0
    latest_metrics: dict[str, float] = {}
    best_val_angular_error: float | None = None
    if resume:
        step, latest_metrics, best_val_angular_error = _load_checkpoint(
            resume,
            adapter,
            optimizer,
            device=device,
        )
    start = time.time()
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    vis_cfg = config.get("Visualization", {})
    vis_enabled = bool(vis_cfg.get("enabled", False))
    vis_interval = max(1, int(vis_cfg.get("interval", val_interval)))
    vis_max_matches = int(vis_cfg.get("max_matches", 80))
    vis_save_dir = str(vis_cfg.get("save_dir", "visualizations"))
    adapter.train()
    wrapper.eval()

    def run_validation(current_step: int) -> dict[str, float]:
        if val_loader is None:
            return {}
        adapter.eval()
        sums: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(val_loader):
                if batch_idx >= max_val_batches:
                    break
                batch = _to_device(raw_batch, device)
                images = batch["images"].float()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    pano_out = wrapper(images)
                    corr = _make_correspondence(config, batch, pano_out, images)
                    dense = adapter(pano_out.stage_features)
                    _, metrics_t = criterion(dense, corr)
                metrics = _float_metrics(metrics_t)
                metrics["valid_corr_ratio"] = float(_valid_ratio(corr).detach().cpu())
                _assert_metrics_finite(metrics, context=f"during validation at step {current_step}")
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                count += 1
        adapter.train()
        if count == 0:
            return {}
        metrics = {key: value / float(count) for key, value in sums.items()}
        if wandb_run is not None:
            wandb_run.log(_prefixed("val", metrics), step=current_step)
        return metrics

    while step < max_steps:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            images = batch["images"].float()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                with torch.no_grad():
                    pano_out = wrapper(images)
                corr = _make_correspondence(config, batch, pano_out, images)
                dense = adapter(pano_out.stage_features)
                need_matches = vis_enabled and (step == 0 or (step + 1) % vis_interval == 0)
                if need_matches:
                    loss, metrics_t, matches = criterion(dense, corr, return_matches=True)
                else:
                    loss, metrics_t = criterion(dense, corr)
                    matches = None
            loss = float(config.get("loss", {}).get("spherical_match_weight", 1.0)) * loss
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite training loss detected before backward at step {step + 1}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_adapter_grads_finite(adapter, context=f"before gradient clipping at step {step + 1}")
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite adapter gradient norm detected at step {step + 1}.")
            _assert_adapter_grads_finite(adapter, context=f"after gradient clipping at step {step + 1}")
            optimizer.step()
            _assert_adapter_finite(adapter, context=f"after optimizer step {step + 1}")
            step += 1
            latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
            latest_metrics["valid_corr_ratio"] = float(_valid_ratio(corr).detach().cpu())
            _assert_metrics_finite(latest_metrics, context=f"after training step {step}")
            if matches is not None:
                path = _save_match_visualization(
                    output_dir=output_dir,
                    save_dir=vis_save_dir,
                    split="train",
                    step=step,
                    images=images,
                    matches=matches,
                    max_matches=vis_max_matches,
                )
                if path is not None and wandb_run is not None:
                    try:
                        import wandb
                        wandb_run.log({"diagnostics/stage1_matches_train": wandb.Image(str(path))}, step=step)
                    except ImportError:
                        pass
            if wandb_run is not None and (step == 1 or step % int(config.get("WeightsAndBiases", {}).get("log_every", 50)) == 0):
                wandb_run.log(_prefixed("train", latest_metrics), step=step)
            if step == 1 or step % log_interval == 0:
                print(yaml.safe_dump({"step": step, "metrics": latest_metrics}, sort_keys=False).strip())
            if val_loader is not None and (step % val_interval == 0 or step == max_steps):
                val_metrics = run_validation(step)
                if val_metrics:
                    print(yaml.safe_dump({"step": step, "val_metrics": val_metrics}, sort_keys=False).strip())
                    val_angular = val_metrics.get("mean_angular_deg")
                    if val_angular is not None and (
                        best_val_angular_error is None or float(val_angular) < float(best_val_angular_error)
                    ):
                        best_val_angular_error = float(val_angular)
                        _save_checkpoint(
                            ckpt_dir / "best_val_angular_error.pt",
                            adapter,
                            config,
                            step,
                            {**latest_metrics, **_prefixed("val", val_metrics)},
                            optimizer,
                            best_val_angular_error,
                        )
            if step % save_interval == 0 or step == max_steps:
                _save_checkpoint(ckpt_dir / "latest.pt", adapter, config, step, latest_metrics, optimizer, best_val_angular_error)
                _save_checkpoint(ckpt_dir / "adapter_latest.pt", adapter, config, step, latest_metrics, optimizer, best_val_angular_error)
                if step % save_interval == 0:
                    _save_checkpoint(
                        ckpt_dir / f"step_{step:06d}.pt",
                        adapter,
                        config,
                        step,
                        latest_metrics,
                        optimizer,
                        best_val_angular_error,
                    )
            if step >= max_steps:
                break
    _save_checkpoint(ckpt_dir / "latest.pt", adapter, config, step, latest_metrics, optimizer, best_val_angular_error)
    _save_checkpoint(ckpt_dir / "adapter_latest.pt", adapter, config, step, latest_metrics, optimizer, best_val_angular_error)
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "steps": step,
        "last_metrics": latest_metrics,
        "checkpoint": str(ckpt_dir / "latest.pt"),
        "best_val_angular_error": best_val_angular_error,
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_spherical_selfi_adapter.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-interval", "--log_interval", type=int, default=None)
    parser.add_argument("--save-interval", "--save_interval", type=int, default=None)
    parser.add_argument("--val-interval", "--val_interval", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.manifest is not None:
        config.setdefault("dataset", {})["manifest"] = args.manifest
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        config.setdefault("train", {})["output_dir"] = args.output_dir
    if args.log_interval is not None:
        config.setdefault("train", {})["log_interval"] = int(args.log_interval)
    if args.save_interval is not None:
        config.setdefault("train", {})["save_interval"] = int(args.save_interval)
    if args.val_interval is not None:
        config.setdefault("train", {})["val_interval"] = int(args.val_interval)
    if args.resume is not None:
        config.setdefault("train", {})["resume"] = args.resume
    if args.wandb_mode is not None:
        if args.wandb_mode == "disabled":
            config.setdefault("WeightsAndBiases", {})["enabled"] = False
        else:
            config.setdefault("WeightsAndBiases", {})["enabled"] = True
            config.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    result = train_spherical_selfi_adapter(config)
    print(yaml.safe_dump(result, sort_keys=False).strip())


if __name__ == "__main__":
    main()
