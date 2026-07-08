"""Train the Stage 1 spherical Selfi DPT adapter.

This is a standalone training entry point. It does not modify or call the SLAM
runtime path.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
            "checkpoint": None,
            "class_path": None,
            "model_kwargs": {},
            "stage_hooks": ["stage1", "stage2", "stage3", "stage4"],
            "feature_keys": [None, None, None, None],
            "token_hw": [None, None, None, None],
            "in_channels": [8, 16, 24, 32],
            "use_no_grad": True,
            "allow_dataset_geometry_fallback": False,
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
            "mode": "global_lowres",
            "loss_stride": 4,
            "local_window_radius": 16,
            "temperature": 0.07,
            "max_queries": 128,
        },
        "loss": {"erp_aux_weight": 0.01, "spherical_match_weight": 1.0},
        "train": {
            "batch_size": 1,
            "num_workers": 0,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "amp": False,
            "max_steps": 2,
            "log_interval": 1,
            "save_interval": 1,
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


def _build_external_model(cfg: dict[str, Any]) -> nn.Module:
    class_path = cfg.get("class_path")
    if not class_path:
        raise ValueError("panovggt.class_path is required when panovggt.synthetic=false.")
    cls = _import_attr(str(class_path))
    model = cls(**dict(cfg.get("model_kwargs", {})))
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
        model.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=False)
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
        if mode == "offline":
            raise
        print(f"W&B online init failed, falling back to offline mode: {exc}")
        kwargs["mode"] = "offline"
        return wandb.init(**kwargs)


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def _save_checkpoint(path: Path, adapter: SphericalSelfiDPTAdapter, config: dict[str, Any], step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spherical_selfi_adapter_v1",
            "adapter": adapter.state_dict(),
            "adapter_config": {
                "in_channels": adapter.in_channels,
                "hidden_dim": adapter.hidden_dim,
                "out_dim": adapter.out_dim,
                "image_height": adapter.image_height,
                "image_width": adapter.image_width,
                "use_circular_padding": adapter.use_circular_padding,
                "norm_output": adapter.norm_output,
            },
            "training_config": config,
            "global_step": int(step),
            "metrics": metrics,
        },
        path,
    )


def train_spherical_selfi_adapter(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config.get("train", {}).get("seed", 1234)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, split="train")
    train_cfg = config.get("train", {})
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage1_collate,
        drop_last=False,
    )
    wrapper = build_panovggt_wrapper(config, device=device)
    adapter = build_adapter(config, device=device)
    loss_cfg = SphericalSelfiAlignmentLossConfig(
        mode=str(config.get("matching", {}).get("mode", "global_lowres")),
        loss_stride=int(config.get("matching", {}).get("loss_stride", 4)),
        local_window_radius=int(config.get("matching", {}).get("local_window_radius", 16)),
        temperature=float(config.get("matching", {}).get("temperature", 0.07)),
        max_queries=config.get("matching", {}).get("max_queries", 512),
        erp_aux_weight=float(config.get("loss", {}).get("erp_aux_weight", 0.01)),
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
    step = 0
    latest_metrics: dict[str, float] = {}
    start = time.time()
    adapter.train()
    wrapper.eval()
    while step < max_steps:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            images = batch["images"].float()
            with torch.no_grad():
                pano_out = wrapper(images)
            init_depth = pano_out.init_depth
            init_poses = pano_out.init_poses
            if init_depth is None or init_poses is None:
                if not bool(config.get("panovggt", {}).get("allow_dataset_geometry_fallback", False)):
                    raise RuntimeError("PanoVGGT wrapper did not return init depth/poses for pseudo correspondence.")
                init_depth = batch.get("depths")
                init_poses = batch.get("poses_c2w")
            if init_depth is None or init_poses is None:
                raise RuntimeError("Stage 1C training requires init depth and c2w poses.")
            corr = generate_spherical_pseudo_correspondence(
                init_depth,
                init_poses,
                batch["pair_indices"],
                height=int(images.shape[-2]),
                width=int(images.shape[-1]),
                **dict(config.get("correspondence", {})),
            )
            dense = adapter(pano_out.stage_features)
            loss, metrics_t = criterion(dense, corr)
            loss = float(config.get("loss", {}).get("spherical_match_weight", 1.0)) * loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            step += 1
            latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
            if wandb_run is not None and (step == 1 or step % int(config.get("WeightsAndBiases", {}).get("log_every", 50)) == 0):
                wandb_run.log({f"train/{key}": value for key, value in latest_metrics.items()}, step=step)
            if step == 1 or step % log_interval == 0:
                print(yaml.safe_dump({"step": step, "metrics": latest_metrics}, sort_keys=False).strip())
            if step % save_interval == 0 or step == max_steps:
                _save_checkpoint(ckpt_dir / "adapter_latest.pt", adapter, config, step, latest_metrics)
            if step >= max_steps:
                break
    _save_checkpoint(ckpt_dir / "adapter_latest.pt", adapter, config, step, latest_metrics)
    if wandb_run is not None:
        wandb_run.finish()
    return {"steps": step, "last_metrics": latest_metrics, "checkpoint": str(ckpt_dir / "adapter_latest.pt"), "elapsed_sec": time.time() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_spherical_selfi_adapter.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        config.setdefault("train", {})["output_dir"] = args.output_dir
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
