"""Train the PanoAnchorSplat local-window Gaussian frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from .matching_dataset import build_matching_dataset_from_config, validate_training_sample
from .pano_anchor_splat_frontend import PanoAnchorSplatFrontend
from .pano_anchor_splat_types import PanoAnchorSplatConfig
from .pano_resplat_renderer import PanoGaussianRendererAdapter
from .train_gaussian import _build_prior_extractor
from .train_matching import _merge_config, matching_collate


def _default_config() -> dict[str, Any]:
    anchor_cfg = PanoAnchorSplatConfig(enabled=True).to_dict()
    return {
        "Training": {
            "steps": 2,
            "batch_size": 1,
            "frames_per_sample": 4,
            "input_frames": 3,
            "target_frame": "last",
            "num_refine": 0,
            "num_workers": 0,
            "amp": True,
            "seed": 1234,
            "grad_clip": 1.0,
            "grad_accum_steps": anchor_cfg["grad_accum_steps"],
            "output_dir": "outputs/pano_anchor_splat/smoke",
            "log_every": 1,
            "save_every": 100,
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
        "PanoAnchorSplat": anchor_cfg,
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
            "soft_max_points": 2048,
            "extra_gsplat360_roots": [],
        },
        "Loss": {
            "rgb_l1_weight": 1.0,
            "depth_rel_l1_weight": 0.05,
            "alpha_weight": 0.01,
            "scale_reg_weight": 0.001,
        },
        "Optimizer": {
            "lr": 2.0e-4,
            "weight_decay": 0.01,
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": "pano_anchor_splat",
            "mode": "online",
            "tags": ["pano-anchor-splat", "feedforward"],
            "log_every": 10,
        },
    }


def load_anchor_splat_train_config(path: str | None) -> dict[str, Any]:
    config = _default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _merge_config(config, user)


def _wandb_init(config: dict[str, Any], output_dir: Path) -> Any:
    wb_cfg = config.get("WeightsAndBiases", {})
    if not bool(wb_cfg.get("enabled", False)) or str(wb_cfg.get("mode", "online")).lower() == "disabled":
        return None
    import wandb

    return wandb.init(
        project=wb_cfg.get("project", "360Droid-splat"),
        entity=wb_cfg.get("entity"),
        name=wb_cfg.get("run_name", "pano_anchor_splat"),
        mode=wb_cfg.get("mode", "online"),
        tags=list(wb_cfg.get("tags", [])),
        config=config,
        dir=str(output_dir),
    )


def _select_window(sample: dict[str, Any], priors: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tr_cfg = config.get("Training", {})
    n = int(sample["images"].shape[1])
    input_frames = min(max(1, int(tr_cfg.get("input_frames", n - 1))), max(1, n - 1))
    target_raw = tr_cfg.get("target_frame", "last")
    target_idx = n - 1 if str(target_raw).lower() == "last" else int(target_raw)
    target_idx = min(max(target_idx, input_frames), n - 1)
    valid = sample.get("valid_depth", sample.get("valid_mask"))
    sky = sample.get("sky_mask")
    context = {
        "images": sample["images"][:, :input_frames].float(),
        "features": priors["features"][:, :input_frames].float(),
        "depths": priors["depth"][:, :input_frames].float(),
        "poses_c2w": priors["poses_c2w"][:, :input_frames].float(),
        "world_points": priors["world_points"][:, :input_frames].float(),
        "valid_mask": valid[:, :input_frames].bool() if torch.is_tensor(valid) else torch.ones_like(priors["depth"][:, :input_frames], dtype=torch.bool),
    }
    if torch.is_tensor(sky):
        context["sky_mask"] = sky[:, :input_frames].bool()
    target = {
        "images": sample["images"][:, target_idx : target_idx + 1].float(),
        "depths": priors["depth"][:, target_idx : target_idx + 1].float(),
        "poses_c2w": priors["poses_c2w"][:, target_idx : target_idx + 1].float(),
    }
    return context, target


def _render_loss(out: dict[str, Any], target: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    render = out["target_render"]
    if render is None:
        raise RuntimeError("PanoAnchorSplat training requires target_render.")
    rgb_pred = render.color
    rgb_target = target["images"].to(device=rgb_pred.device, dtype=rgb_pred.dtype)
    if rgb_pred.ndim == 4:
        rgb_pred = rgb_pred.unsqueeze(1)
    rgb_l1 = (rgb_pred - rgb_target).abs().mean()
    depth_rel = rgb_l1.new_tensor(0.0)
    if "depths" in target and torch.is_tensor(render.depth):
        depth_pred = render.depth if render.depth.ndim == 5 else render.depth.unsqueeze(1)
        depth_target = target["depths"].to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid = torch.isfinite(depth_target) & (depth_target > 0.0)
        if bool(valid.any()):
            depth_rel = ((depth_pred - depth_target).abs() / depth_target.abs().clamp_min(1.0))[valid].mean()
    final_state = out["final_state"]
    scale_reg = final_state.log_scales.exp().mean()
    alpha_mean = render.alpha.mean() if torch.is_tensor(render.alpha) else rgb_l1.new_tensor(0.0)
    weights = config.get("Loss", {})
    loss = (
        float(weights.get("rgb_l1_weight", 1.0)) * rgb_l1
        + float(weights.get("depth_rel_l1_weight", 0.05)) * depth_rel
        + float(weights.get("scale_reg_weight", 0.001)) * scale_reg
        + float(weights.get("alpha_weight", 0.01)) * (1.0 - alpha_mean)
    )
    mse = (rgb_pred - rgb_target).square().mean().clamp_min(1.0e-8)
    return loss, {
        "rgb_l1": rgb_l1.detach(),
        "depth_rel_l1": depth_rel.detach(),
        "scale_mean": scale_reg.detach(),
        "alpha_mean": alpha_mean.detach(),
        "psnr": (-10.0 * torch.log10(mse)).detach(),
        "anchor_count": out["anchors"].valid_mask.sum(dim=1).float().mean().detach(),
        "gaussian_count": out["final_state"].valid_mask.sum(dim=1).float().mean().detach(),
    }


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def _save_checkpoint(path: Path, frontend: PanoAnchorSplatFrontend, config: dict[str, Any], step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pano_anchor_splat_frontend_v1",
            "state_dict": frontend.state_dict(),
            "config": config,
            "global_step": int(step),
            "metrics": metrics,
        },
        path,
    )


def train_anchor_splat_gaussian(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config.get("Training", {}).get("seed", 1234)))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    dataset = build_matching_dataset_from_config(config, split="train")
    tr_cfg = config.get("Training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tr_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 0)),
        collate_fn=matching_collate,
        drop_last=False,
    )
    prior_extractor = _build_prior_extractor(config, device=device)
    prior_extractor.eval()
    renderer_cfg = dict(config.get("Renderer", {}))
    renderer = PanoGaussianRendererAdapter(
        config={"Training": dict(config.get("TrainingRender", {})), "Renderer": renderer_cfg},
        extra_gsplat360_roots=list(renderer_cfg.get("extra_gsplat360_roots", [])),
        allow_soft_splat_fallback=bool(renderer_cfg.get("allow_soft_splat_fallback", True)),
        soft_sigma_px=float(renderer_cfg.get("soft_sigma_px", 1.25)),
        soft_max_points=int(renderer_cfg.get("soft_max_points", 2048)),
    )
    frontend = PanoAnchorSplatFrontend(
        PanoAnchorSplatConfig.from_dict(config.get("PanoAnchorSplat", {})),
        renderer=renderer,
        renderer_backend=str(renderer_cfg.get("backend", "soft_splat")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in frontend.parameters() if param.requires_grad],
        lr=float(config.get("Optimizer", {}).get("lr", 2.0e-4)),
        weight_decay=float(config.get("Optimizer", {}).get("weight_decay", 0.01)),
    )
    output_dir = Path(tr_cfg.get("output_dir", "outputs/pano_anchor_splat/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _wandb_init(config, output_dir)
    max_steps = int(tr_cfg.get("steps", 1))
    grad_accum = max(1, int(tr_cfg.get("grad_accum_steps", config.get("PanoAnchorSplat", {}).get("grad_accum_steps", 4))))
    num_refine = max(0, int(tr_cfg.get("num_refine", 0)))
    amp_enabled = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    dtype = PanoAnchorSplatConfig.from_dict(config.get("PanoAnchorSplat", {})).torch_dtype
    step = 0
    micro = 0
    best = float("inf")
    latest_metrics: dict[str, float] = {}
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for raw_batch in loader:
            sample = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw_batch.items()}
            validate_training_sample(sample, "matching_only", allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            with torch.no_grad():
                priors = prior_extractor(sample)
            context, target = _select_window(sample, priors, config)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                out = frontend(context, target=target, num_refine=num_refine)
                loss, metrics_t = _render_loss(out, target, config)
                loss_for_backward = loss / float(grad_accum)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
                print(yaml.safe_dump({"skipped_batch": "nonfinite_loss", "metrics": latest_metrics}, sort_keys=False).strip())
                continue
            loss_for_backward.backward()
            micro += 1
            if micro % grad_accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                frontend.parameters(),
                float(tr_cfg.get("grad_clip", 1.0)),
                error_if_nonfinite=False,
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t, "grad_norm": grad_norm.detach()})
            best = min(best, latest_metrics["loss"])
            if step == 1 or step % int(tr_cfg.get("log_every", 1)) == 0:
                print(yaml.safe_dump({"step": step, "metrics": latest_metrics}, sort_keys=False).strip())
            if wandb_run is not None and (step == 1 or step % int(config.get("WeightsAndBiases", {}).get("log_every", 10)) == 0):
                wandb_run.log({f"train/{key}": value for key, value in latest_metrics.items()}, step=step)
            if step % int(tr_cfg.get("save_every", 100)) == 0:
                _save_checkpoint(output_dir / "latest.pt", frontend, config, step, latest_metrics)
            if step >= max_steps:
                break
    _save_checkpoint(output_dir / "latest.pt", frontend, config, step, latest_metrics)
    _save_checkpoint(output_dir / "best.pt", frontend, config, step, latest_metrics)
    (output_dir / "metrics.json").write_text(json.dumps({"best_loss": best, "last_metrics": latest_metrics}, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "steps": int(step),
        "best_loss": float(best),
        "last_metrics": latest_metrics,
        "checkpoint": str(output_dir / "latest.pt"),
        "elapsed_sec": float(time.time() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--renderer", default=None, choices=["gsplat360", "soft_splat"])
    parser.add_argument("--num-refine", type=int, default=None)
    args = parser.parse_args()
    config = load_anchor_splat_train_config(args.config)
    if args.steps is not None:
        config.setdefault("Training", {})["steps"] = int(args.steps)
    if args.batch_size is not None:
        config.setdefault("Training", {})["batch_size"] = int(args.batch_size)
    if args.output_dir is not None:
        config.setdefault("Training", {})["output_dir"] = args.output_dir
    if args.renderer is not None:
        config.setdefault("Renderer", {})["backend"] = args.renderer
    if args.num_refine is not None:
        config.setdefault("Training", {})["num_refine"] = int(args.num_refine)
    result = train_anchor_splat_gaussian(config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
