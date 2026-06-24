"""Train feed-forward PanoVGGT anchor/scaffold Gaussian prediction heads."""

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

from backend.pano_gs import PFGS360Renderer, PanoRenderCamera
from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing, pixel_grid

from .gaussian_head import (
    AnchorGaussianPrediction,
    IterativeGaussianRefiner,
    PanoVGGTAnchorGaussianHead,
    merge_feedback,
    sample_render_feedback,
)
from .matching_adapter import normalize_pano_feature
from .matching_dataset import build_matching_dataset_from_config, validate_training_sample
from .train_matching import FrozenSyntheticFeatureExtractor, _init_wandb, _merge_config, matching_collate


def _default_config() -> dict[str, Any]:
    return {
        "Training": {
            "mode": "matching_only",
            "steps": 2,
            "batch_size": 1,
            "frames_per_sample": 4,
            "input_frames": 3,
            "target_frame": "last",
            "num_workers": 0,
            "amp": False,
            "seed": 1234,
            "log_interval": 1,
            "save_interval": 1,
            "output_dir": "outputs/panovggt_m3_sphere_omni360_gaussian",
            "grad_clip": 1.0,
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
        "GaussianHead": {
            "hidden_dim": 32,
            "anchor_feat_dim": 32,
            "k_offsets": 2,
            "num_conv_blocks": 2,
            "anchor_stride": 1,
            "max_anchors": 256,
            "min_scale": 0.002,
            "max_scale": 0.12,
            "init_scale": 0.02,
            "depth_delta_limit": 0.35,
            "iterations": 2,
            "refiner_hidden_dim": 64,
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
            "allow_smoke_fallback": False,
            "extra_gsplat360_roots": [],
            "soft_sigma_px": 1.25,
            "soft_max_points": 4096,
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
            "rgb_weight": 1.0,
            "depth_weight": 0.05,
            "alpha_weight": 0.01,
            "scale_reg_weight": 0.001,
            "offset_reg_weight": 0.001,
            "depth_delta_reg_weight": 0.01,
            "intermediate_weight": 0.25,
        },
        "Optimizer": {
            "lr": 2.0e-4,
            "weight_decay": 0.05,
        },
        "WeightsAndBiases": {
            "enabled": False,
            "project": "360Droid-splat",
            "entity": None,
            "run_name": "panovggt_m3_sphere_gaussian_head",
            "mode": "online",
            "log_every": 10,
            "tags": ["panovggt-gaussian-head", "feedforward"],
        },
        "Visualization": {
            "enabled": False,
            "interval": 100,
            "save_dir": "visualizations",
            "max_width": 256,
        },
        "Validation": {
            "enabled": False,
            "batch_size": 1,
            "num_workers": 0,
            "max_batches": 1,
        },
    }


def load_gaussian_train_config(path: str | None) -> dict[str, Any]:
    config = _default_config()
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    return _merge_config(config, user)


def _image_hw(images: torch.Tensor) -> tuple[int, int]:
    return int(images.shape[-2]), int(images.shape[-1])


def _safe_log_scales(pred: AnchorGaussianPrediction) -> torch.Tensor:
    return pred.log_scales.clamp(math.log(float(pred.min_scale)), math.log(float(pred.max_scale)))


def _world_points_from_depth(depth: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    """Build world points with shape ``B x N x H x W x 3`` from ERP range depth."""

    if depth.ndim != 5 or poses_c2w.ndim != 4:
        raise ValueError("depth must be BxNx1xHxW and poses_c2w must be BxNx4x4.")
    b, n, _, h, w = [int(v) for v in depth.shape]
    grid = pixel_grid(h, w, device=depth.device, dtype=depth.dtype)
    bearing = erp_pixel_to_bearing(grid, h, w).to(device=depth.device, dtype=depth.dtype)
    local = bearing.view(1, 1, h, w, 3) * depth[:, :, 0].unsqueeze(-1)
    rot = poses_c2w[:, :, :3, :3].to(device=depth.device, dtype=depth.dtype)
    trans = poses_c2w[:, :, :3, 3].to(device=depth.device, dtype=depth.dtype)
    return torch.einsum("bnij,bnhwj->bnhwi", rot, local) + trans.view(b, n, 1, 1, 3)


class SyntheticGaussianPriorExtractor(nn.Module):
    """Synthetic feature/prior extractor for local tests and smoke training."""

    def __init__(self, *, feature_dim: int, feature_stride: int) -> None:
        super().__init__()
        self.extractor = FrozenSyntheticFeatureExtractor(feature_dim=feature_dim, feature_stride=feature_stride)

    def forward(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = sample["images"].float()
        features = self.extractor(images)
        depth = sample["depths"].float()
        poses = sample["poses_c2w"].float()
        world = _world_points_from_depth(depth, poses)
        b, v, c, hf, wf = [int(x) for x in features.shape]
        tokens = features.permute(0, 1, 3, 4, 2).reshape(b, v, hf * wf, c).contiguous()
        return {
            "features": features,
            "tokens": tokens,
            "token_hw": torch.tensor([hf, wf], device=features.device, dtype=torch.long),
            "depth": depth,
            "poses_c2w": poses,
            "world_points": world,
        }


class ExternalPanoVGGTGaussianPriorExtractor(nn.Module):
    """Frozen PanoVGGT feature and geometry prior extractor for Gaussian training."""

    def __init__(self, model_cfg: dict[str, Any], *, device: torch.device) -> None:
        super().__init__()
        feature_hook = model_cfg.get("feature_hook")
        if not feature_hook:
            raise ValueError("Model.feature_hook is required for real Gaussian-head training.")
        from .engine import ExternalPanoVGGTInferenceEngine

        image_size_raw = model_cfg.get("image_size")
        image_size = None if image_size_raw is None else (int(image_size_raw[0]), int(image_size_raw[1]))
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
        self.feature_hook = str(feature_hook)
        self.feature_key = model_cfg.get("feature_key")
        self.patch_size = int(model_cfg.get("patch_size") or getattr(self.model, "patch_size", self.engine.patch_multiple))
        self._feature: torch.Tensor | None = None
        modules[self.feature_hook].register_forward_hook(self._hook)

    @staticmethod
    def _infer_input_hw(inputs: tuple[Any, ...]) -> tuple[int, int] | None:
        for value in inputs:
            if torch.is_tensor(value) and value.ndim >= 4:
                return int(value.shape[-2]), int(value.shape[-1])
        return None

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        self._feature = normalize_pano_feature(
            output,
            input_hw=self._infer_input_hw(inputs),
            patch_size=self.patch_size,
            feature_key=self.feature_key,
        )

    def forward(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = sample["images"].float()
        if images.ndim != 5:
            raise ValueError(f"images must have shape BxNx3xHxW, got {tuple(images.shape)}")
        features = []
        depths = []
        poses = []
        worlds = []
        with torch.no_grad():
            for batch_idx in range(int(images.shape[0])):
                self._feature = None
                pred = self.engine.infer(images[batch_idx])
                if self._feature is None:
                    raise RuntimeError("External PanoVGGT feature hook did not capture any tensor.")
                feat = self._feature
                if feat.ndim == 5:
                    if int(feat.shape[0]) != 1:
                        raise ValueError(f"Expected hook feature B=1, got {tuple(feat.shape)}")
                    feat = feat[0]
                if feat.ndim != 4:
                    raise ValueError(f"Expected feature as NxCxHfxWf, got {tuple(feat.shape)}")
                features.append(torch.nan_to_num(feat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0))
                depths.append(torch.nan_to_num(pred.depth.detach().float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6))
                poses.append(pred.poses_c2w.detach().float())
                worlds.append(pred.chunk_world_points.detach().float())
        feature_batch = torch.stack(features, dim=0)
        b, v, c, hf, wf = [int(x) for x in feature_batch.shape]
        tokens = feature_batch.permute(0, 1, 3, 4, 2).reshape(b, v, hf * wf, c).contiguous()
        return {
            "features": feature_batch,
            "tokens": tokens,
            "token_hw": torch.tensor([hf, wf], device=feature_batch.device, dtype=torch.long),
            "depth": torch.stack(depths, dim=0),
            "poses_c2w": torch.stack(poses, dim=0),
            "world_points": torch.stack(worlds, dim=0),
        }


def _build_prior_extractor(config: dict[str, Any], *, device: torch.device) -> nn.Module:
    model_cfg = config.get("Model", {})
    if bool(model_cfg.get("use_synthetic_features", False)):
        return SyntheticGaussianPriorExtractor(
            feature_dim=int(model_cfg.get("feature_dim", 16)),
            feature_stride=int(model_cfg.get("feature_stride", 4)),
        ).to(device)
    return ExternalPanoVGGTGaussianPriorExtractor(model_cfg, device=device).to(device)


class FeedForwardGaussianModel(nn.Module):
    """Initial Gaussian prediction plus recurrent render-feedback refinement."""

    def __init__(self, *, feature_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        cfg = config.get("GaussianHead", {})
        self.iterations = max(0, int(cfg.get("iterations", 2)))
        self.head = PanoVGGTAnchorGaussianHead(
            feature_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            anchor_feat_dim=int(cfg.get("anchor_feat_dim", 64)),
            k_offsets=int(cfg.get("k_offsets", 4)),
            num_conv_blocks=int(cfg.get("num_conv_blocks", 2)),
            anchor_stride=int(cfg.get("anchor_stride", 1)),
            max_anchors=int(cfg.get("max_anchors", 4096)),
            min_scale=float(cfg.get("min_scale", 0.002)),
            max_scale=float(cfg.get("max_scale", 0.12)),
            init_scale=float(cfg.get("init_scale", 0.02)),
            depth_delta_limit=float(cfg.get("depth_delta_limit", 0.35)),
        )
        self.refiner = IterativeGaussianRefiner(
            anchor_feat_dim=int(cfg.get("anchor_feat_dim", 64)),
            hidden_dim=int(cfg.get("refiner_hidden_dim", cfg.get("hidden_dim", 128))),
            k_offsets=int(cfg.get("k_offsets", 4)),
        )

    def initial_prediction(
        self,
        *,
        features: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        poses_c2w: torch.Tensor,
        world_points: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> AnchorGaussianPrediction:
        return self.head(
            features,
            images,
            depth,
            poses_c2w,
            world_points=world_points,
            valid_mask=valid_mask,
        )

    def refine(self, pred: AnchorGaussianPrediction, feedback: dict[str, torch.Tensor]) -> AnchorGaussianPrediction:
        return self.refiner(pred, feedback)

    def checkpoint_payload(self, *, config: dict[str, Any], global_step: int, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "format": "panovggt_anchor_gaussian_head_v1",
            "gaussian_head": self.head.state_dict(),
            "gaussian_refiner": self.refiner.state_dict(),
            "head_config": self.head.head_config(),
            "iterations": int(self.iterations),
            "training_config": config,
            "global_step": int(global_step),
            "metrics": metrics,
        }


def _soft_splat_render(
    gaussians,
    camera: PanoRenderCamera,
    *,
    sigma_px: float = 1.25,
    max_points: int = 4096,
) -> dict[str, torch.Tensor]:
    xyz = gaussians.get_xyz
    h, w = int(camera.image_height), int(camera.image_width)
    if int(xyz.shape[0]) > int(max_points):
        idx = torch.linspace(0, int(xyz.shape[0]) - 1, steps=int(max_points), device=xyz.device).round().long()
        xyz = xyz.index_select(0, idx)
        color = gaussians.get_features.index_select(0, idx)
        opacity = gaussians.get_opacity.index_select(0, idx)
        scale = gaussians.get_scaling.index_select(0, idx)
    else:
        color = gaussians.get_features
        opacity = gaussians.get_opacity
        scale = gaussians.get_scaling
    if int(xyz.shape[0]) == 0:
        render = torch.zeros(3, h, w, device=camera.c2w.device, dtype=camera.c2w.dtype)
        alpha = torch.zeros(1, h, w, device=render.device, dtype=render.dtype)
        return {"render": render, "depth": alpha.clone(), "alpha": alpha, "opacity": alpha}
    device = xyz.device
    dtype = xyz.dtype
    c2w = camera.c2w.to(device=device, dtype=dtype)
    w2c = torch.linalg.inv(c2w)
    cam = (w2c @ torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device, dtype=dtype)], dim=-1).T).T[:, :3]
    bearing = F.normalize(cam, dim=-1, eps=1.0e-6)
    uv = bearing_to_erp_pixel(bearing, h, w)
    depth = torch.linalg.norm(cam, dim=-1).clamp_min(1.0e-6)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = torch.remainder(xs.reshape(1, h, w) - uv[:, 0].view(-1, 1, 1) + float(w) * 0.5, float(w)) - float(w) * 0.5
    dy = ys.reshape(1, h, w) - uv[:, 1].view(-1, 1, 1)
    sigma = (
        float(sigma_px)
        + (scale.mean(dim=-1) / depth).clamp(0.0, 0.10) * float(max(h, w))
    ).view(-1, 1, 1)
    weight = torch.exp(-0.5 * (dx.square() + dy.square()) / sigma.square().clamp_min(1.0e-6))
    weight = weight * opacity.view(-1, 1, 1).clamp(0.0, 1.0)
    denom = weight.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
    rgb = (weight.unsqueeze(1) * color.view(-1, 3, 1, 1)).sum(dim=0) / denom
    depth_img = (weight * depth.view(-1, 1, 1)).sum(dim=0, keepdim=True) / denom
    alpha = weight.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    return {"render": rgb.clamp(0.0, 1.0), "depth": depth_img, "alpha": alpha, "opacity": alpha}


def _render_prediction_batch(
    pred: AnchorGaussianPrediction,
    *,
    target_poses: torch.Tensor,
    image_hw: tuple[int, int],
    renderer: PFGS360Renderer | None,
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    backend = str(config.get("Renderer", {}).get("backend", "gsplat360")).lower()
    out = []
    render_config = {
        "Training": dict(config.get("TrainingRender", {})),
        "Renderer": dict(config.get("Renderer", {})),
    }
    for batch_idx in range(pred.batch_size):
        gaussians = pred.materialize(batch_idx, config=render_config)
        camera = PanoRenderCamera(
            image_height=int(image_hw[0]),
            image_width=int(image_hw[1]),
            c2w=target_poses[batch_idx].to(gaussians.get_xyz),
        )
        if backend == "soft_splat":
            pkg = _soft_splat_render(
                gaussians,
                camera,
                sigma_px=float(config.get("Renderer", {}).get("soft_sigma_px", 1.25)),
                max_points=int(config.get("Renderer", {}).get("soft_max_points", 4096)),
            )
        elif backend == "gsplat360":
            if renderer is None:
                raise RuntimeError("gsplat360 renderer is not initialized.")
            pkg = renderer.render(camera, gaussians)
        else:
            raise ValueError(f"Unsupported Renderer.backend={backend!r}; expected gsplat360 or soft_splat.")
        out.append(pkg)
    return out


def _loss_for_renders(
    renders: list[dict[str, torch.Tensor]],
    *,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor | None,
    pred: AnchorGaussianPrediction,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = config.get("Loss", {})
    losses = []
    rgb_losses = []
    depth_losses = []
    psnrs = []
    alpha_means = []
    for batch_idx, pkg in enumerate(renders):
        render = torch.nan_to_num(pkg["render"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target = torch.nan_to_num(target_rgb[batch_idx].to(render), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        rgb = (render - target).abs().mean()
        depth_loss = render.sum() * 0.0
        if target_depth is not None and torch.is_tensor(pkg.get("depth")):
            td = torch.nan_to_num(target_depth[batch_idx].to(render), nan=0.0, posinf=0.0, neginf=0.0)
            rd = torch.nan_to_num(pkg["depth"].to(render), nan=0.0, posinf=0.0, neginf=0.0)
            alpha = pkg.get("alpha")
            mask = torch.isfinite(td) & (td > 0.0)
            if torch.is_tensor(alpha):
                mask = mask & (torch.nan_to_num(alpha.to(render), nan=0.0, posinf=1.0, neginf=0.0) > 0.01)
            if bool(mask.any()):
                depth_loss = (((rd - td).abs() / td.abs().clamp_min(1.0))[mask]).mean()
        mse = (render - target).square().mean().clamp_min(1.0e-8)
        psnr = -10.0 * torch.log10(mse)
        alpha = pkg.get("alpha")
        alpha_mean = render.new_tensor(0.0) if not torch.is_tensor(alpha) else torch.nan_to_num(alpha.to(render), nan=0.0, posinf=1.0, neginf=0.0).mean()
        losses.append(float(weights.get("rgb_weight", 1.0)) * rgb + float(weights.get("depth_weight", 0.05)) * depth_loss)
        rgb_losses.append(rgb.detach())
        depth_losses.append(depth_loss.detach())
        psnrs.append(psnr.detach())
        alpha_means.append(alpha_mean.detach())
    base = torch.stack(losses).mean()
    scale_reg = _safe_log_scales(pred).exp().mean()
    safe_offsets = torch.nan_to_num(pred.local_offsets, nan=0.0, posinf=float(pred.max_scale), neginf=-float(pred.max_scale))
    offset_reg = (safe_offsets.square().sum(dim=-1) + 1.0e-12).sqrt().mean()
    depth_reg = torch.nan_to_num(pred.log_depth_delta, nan=0.0, posinf=float(pred.depth_delta_limit), neginf=-float(pred.depth_delta_limit)).abs().mean()
    opacity_reg = torch.sigmoid(torch.nan_to_num(pred.opacity_logit, nan=0.0, posinf=0.0, neginf=0.0)).mean()
    loss = (
        base
        + float(weights.get("scale_reg_weight", 0.001)) * scale_reg
        + float(weights.get("offset_reg_weight", 0.001)) * offset_reg
        + float(weights.get("depth_delta_reg_weight", 0.01)) * depth_reg
        + float(weights.get("alpha_weight", 0.01)) * (1.0 - opacity_reg)
    )
    return loss, {
        "rgb_l1": torch.stack(rgb_losses).mean(),
        "depth_rel_l1": torch.stack(depth_losses).mean(),
        "psnr": torch.stack(psnrs).mean(),
        "alpha_mean": torch.stack(alpha_means).mean(),
        "scale_mean": scale_reg.detach(),
        "offset_mean": offset_reg.detach(),
        "depth_delta_abs": depth_reg.detach(),
        "opacity_mean": opacity_reg.detach(),
    }


def _run_model_iteration(
    model: FeedForwardGaussianModel,
    *,
    features: torch.Tensor,
    source_images: torch.Tensor,
    source_depth: torch.Tensor,
    source_poses: torch.Tensor,
    source_world: torch.Tensor,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor,
    target_poses: torch.Tensor,
    renderer: PFGS360Renderer | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[list[dict[str, torch.Tensor]]], list[AnchorGaussianPrediction]]:
    pred = model.initial_prediction(
        features=features,
        images=source_images,
        depth=source_depth,
        poses_c2w=source_poses,
        world_points=source_world,
    )
    states = [pred]
    render_history: list[list[dict[str, torch.Tensor]]] = []
    losses = []
    metrics_by_iter = []
    image_hw = _image_hw(target_rgb)
    for iter_idx in range(model.iterations + 1):
        renders = _render_prediction_batch(pred, target_poses=target_poses, image_hw=image_hw, renderer=renderer, config=config)
        render_history.append(renders)
        loss_i, metrics_i = _loss_for_renders(
            renders,
            target_rgb=target_rgb,
            target_depth=target_depth,
            pred=pred,
            config=config,
        )
        losses.append(loss_i)
        metrics_by_iter.append(metrics_i)
        if iter_idx >= model.iterations:
            break
        feedback_parts = []
        for batch_idx, pkg in enumerate(renders):
            feedback_parts.append(
                sample_render_feedback(
                    pred,
                    render_rgb=pkg["render"],
                    render_depth=pkg.get("depth"),
                    render_alpha=pkg.get("alpha"),
                    target_rgb=target_rgb[batch_idx],
                    target_depth=target_depth[batch_idx],
                    target_pose_c2w=target_poses[batch_idx],
                    batch_index=batch_idx,
                )
            )
        pred = model.refine(pred, merge_feedback(feedback_parts))
        states.append(pred)
    intermediate_weight = float(config.get("Loss", {}).get("intermediate_weight", 0.25))
    final_loss = losses[-1]
    if len(losses) > 1 and intermediate_weight > 0.0:
        final_loss = final_loss + intermediate_weight * torch.stack(losses[:-1]).mean()
    metrics: dict[str, torch.Tensor] = {}
    for iter_idx, (loss_i, metrics_i) in enumerate(zip(losses, metrics_by_iter)):
        metrics[f"iter{iter_idx}/loss_raw"] = loss_i.detach()
        metrics.update({f"iter{iter_idx}/{key}": value for key, value in metrics_i.items()})
        if iter_idx > 0:
            metrics[f"iter{iter_idx}/loss_delta_from_prev"] = (losses[iter_idx - 1] - loss_i).detach()
            prev_state = states[iter_idx - 1]
            cur_state = states[iter_idx]
            metrics[f"iter{iter_idx}/update_depth_delta_abs"] = torch.nan_to_num(cur_state.log_depth_delta - prev_state.log_depth_delta, nan=0.0).abs().mean().detach()
            metrics[f"iter{iter_idx}/update_offset_abs"] = torch.nan_to_num(cur_state.local_offsets - prev_state.local_offsets, nan=0.0).abs().mean().detach()
            metrics[f"iter{iter_idx}/update_log_scale_abs"] = torch.nan_to_num(cur_state.log_scales - prev_state.log_scales, nan=0.0).abs().mean().detach()
            metrics[f"iter{iter_idx}/update_opacity_logit_abs"] = torch.nan_to_num(cur_state.opacity_logit - prev_state.opacity_logit, nan=0.0).abs().mean().detach()
            metrics[f"iter{iter_idx}/update_color_logit_abs"] = torch.nan_to_num(cur_state.color_logit - prev_state.color_logit, nan=0.0).abs().mean().detach()
    metrics.update({f"final/{key}": value for key, value in metrics_by_iter[-1].items()})
    metrics["loss_initial"] = losses[0].detach()
    metrics["loss_final_raw"] = losses[-1].detach()
    metrics["iter_loss_improvement"] = (losses[0] - losses[-1]).detach()
    metrics["anchor_count"] = torch.tensor(float(states[-1].num_anchors), device=final_loss.device)
    metrics["gaussian_count"] = torch.tensor(float(states[-1].num_anchors * states[-1].k_offsets), device=final_loss.device)
    return final_loss, metrics, render_history, states


def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def _select_source_target(priors: dict[str, torch.Tensor], sample: dict[str, Any], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    tr_cfg = config.get("Training", {})
    input_frames = max(1, int(tr_cfg.get("input_frames", int(sample["images"].shape[1]) - 1)))
    n = int(sample["images"].shape[1])
    input_frames = min(input_frames, n - 1)
    target_raw = tr_cfg.get("target_frame", "last")
    target_idx = n - 1 if str(target_raw).lower() == "last" else int(target_raw)
    if target_idx < input_frames:
        target_idx = input_frames
    target_idx = min(target_idx, n - 1)
    return {
        "features": priors["features"][:, :input_frames],
        "source_images": sample["images"][:, :input_frames].float(),
        "source_depth": priors["depth"][:, :input_frames],
        "source_poses": priors["poses_c2w"][:, :input_frames],
        "source_world": priors["world_points"][:, :input_frames],
        "target_rgb": sample["images"][:, target_idx].float(),
        "target_depth": priors["depth"][:, target_idx],
        "target_poses": priors["poses_c2w"][:, target_idx],
    }


def _batch_has_finite_camera_state(batch: dict[str, torch.Tensor]) -> bool:
    pose_keys = ("source_poses", "target_poses")
    depth_keys = ("source_depth", "target_depth")
    for key in pose_keys:
        value = batch.get(key)
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            return False
    for key in depth_keys:
        value = batch.get(key)
        if torch.is_tensor(value) and not bool((torch.isfinite(value) & (value > 0.0)).any()):
            return False
    return True


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    img = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if img.ndim == 3 and int(img.shape[0]) == 3:
        img = img.permute(1, 2, 0)
    arr = (img.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_visualization(
    *,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    batch: dict[str, torch.Tensor],
    render_history: list[list[dict[str, torch.Tensor]]],
    wandb_run: Any,
) -> Path | None:
    vis_cfg = config.get("Visualization", {})
    if not bool(vis_cfg.get("enabled", False)):
        return None
    interval = max(1, int(vis_cfg.get("interval", 100)))
    if step != 1 and step % interval != 0:
        return None
    source = _tensor_to_image(batch["source_images"][0, 0])
    target = _tensor_to_image(batch["target_rgb"][0])
    initial = _tensor_to_image(render_history[0][0]["render"])
    final = _tensor_to_image(render_history[-1][0]["render"])
    err = (render_history[-1][0]["render"].detach().float().cpu() - batch["target_rgb"][0].detach().float().cpu()).abs().mean(dim=0)
    err = err / err.max().clamp_min(1.0e-6)
    err_rgb = torch.stack([err, 1.0 - err, torch.zeros_like(err)], dim=0)
    error_img = _tensor_to_image(err_rgb)
    max_width = int(vis_cfg.get("max_width", 256))
    panels = [source, target, initial, final, error_img]
    if panels[0].width > max_width:
        scale = max_width / float(panels[0].width)
        panels = [img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale)))) for img in panels]
    width = sum(img.width for img in panels)
    height = max(img.height for img in panels) + 26
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img in zip(("source", "target", "render0", "render_final", "error"), panels):
        canvas.paste(img, (x, 26))
        draw.text((x + 6, 6), label, fill=(255, 255, 255))
        x += img.width
    vis_dir = output_dir / str(vis_cfg.get("save_dir", "visualizations"))
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / f"step_{int(step):07d}_gaussian.png"
    canvas.save(path)
    if wandb_run is not None:
        import wandb

        wandb_run.log({"visualization/gaussian_panel": wandb.Image(str(path))}, step=step)
    return path


def _build_model(config: dict[str, Any], feature_dim: int, *, device: torch.device) -> tuple[FeedForwardGaussianModel, torch.optim.Optimizer]:
    model = FeedForwardGaussianModel(feature_dim=feature_dim, config=config).to(device)
    opt_cfg = config.get("Optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 2.0e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    return model, optimizer


def _save_checkpoint(path: Path, *, model: FeedForwardGaussianModel, config: dict[str, Any], step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint_payload(config=config, global_step=step, metrics=metrics), path)


def train_gaussian_head(config: dict[str, Any]) -> dict[str, Any]:
    """Train the standalone feed-forward Gaussian head."""

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
    renderer = None
    if str(config.get("Renderer", {}).get("backend", "gsplat360")).lower() == "gsplat360":
        renderer = PFGS360Renderer(
            config={"Training": dict(config.get("TrainingRender", {})), "Renderer": dict(config.get("Renderer", {}))},
            extra_gsplat360_roots=list(config.get("Renderer", {}).get("extra_gsplat360_roots", [])),
            allow_fallback=bool(config.get("Renderer", {}).get("allow_smoke_fallback", False)),
        )
    output_dir = Path(tr_cfg.get("output_dir", "outputs/panovggt_m3_sphere_omni360_gaussian"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    wandb_run = _init_wandb(config, output_dir)
    model: FeedForwardGaussianModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    step = 0
    best = float("inf")
    latest_metrics: dict[str, float] = {}
    start = time.time()
    max_steps = int(tr_cfg.get("steps", 1))
    save_interval = max(1, int(tr_cfg.get("save_interval", 1000)))
    log_interval = max(1, int(tr_cfg.get("log_interval", 50)))
    wb_log_every = max(1, int(config.get("WeightsAndBiases", {}).get("log_every", 10)))

    while step < max_steps:
        for raw_batch in loader:
            sample = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw_batch.items()}
            validate_training_sample(sample, "matching_only", allow_fallback_mode=bool(config.get("Dataset", {}).get("allow_fallback_mode", False)))
            with torch.no_grad():
                priors = prior_extractor(sample)
            batch = _select_source_target(priors, sample, config)
            if model is None or optimizer is None:
                configured_dim = config.get("Model", {}).get("feature_dim") or int(batch["features"].shape[2])
                feature_dim = int(configured_dim)
                if int(batch["features"].shape[2]) != feature_dim:
                    raise ValueError(f"Configured feature_dim={feature_dim} does not match extracted dim={int(batch['features'].shape[2])}.")
                model, optimizer = _build_model(config, feature_dim, device=device)
                model.train()
            if not _batch_has_finite_camera_state(batch):
                print(yaml.safe_dump({"skipped_batch": "nonfinite_camera_state", "step": step}, sort_keys=False).strip())
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, metrics_t, render_history, _states = _run_model_iteration(
                model,
                features=batch["features"].float(),
                source_images=batch["source_images"].float(),
                source_depth=batch["source_depth"].float(),
                source_poses=batch["source_poses"].float(),
                source_world=batch["source_world"].float(),
                target_rgb=batch["target_rgb"].float(),
                target_depth=batch["target_depth"].float(),
                target_poses=batch["target_poses"].float(),
                renderer=renderer,
                config=config,
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t})
                print(yaml.safe_dump({"skipped_batch": "nonfinite_loss", "step": step, "metrics": latest_metrics}, sort_keys=False).strip())
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg.get("grad_clip", 1.0)), error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t, "grad_norm": grad_norm.detach()})
                print(yaml.safe_dump({"skipped_batch": "nonfinite_grad", "step": step, "metrics": latest_metrics}, sort_keys=False).strip())
                continue
            optimizer.step()
            step += 1
            latest_metrics = _float_metrics({"loss": loss.detach(), **metrics_t, "grad_norm": grad_norm.detach()})
            if wandb_run is not None and (step == 1 or step % wb_log_every == 0):
                wandb_run.log({f"train/{key}": value for key, value in latest_metrics.items()}, step=step)
            if step == 1 or step % log_interval == 0:
                print(yaml.safe_dump({"step": step, "metrics": latest_metrics}, sort_keys=False).strip())
            _save_visualization(
                config=config,
                output_dir=output_dir,
                step=step,
                batch=batch,
                render_history=render_history,
                wandb_run=wandb_run,
            )
            if latest_metrics["loss"] < best:
                best = latest_metrics["loss"]
            if model is not None and (step % save_interval == 0 or step == max_steps):
                _save_checkpoint(ckpt_dir / "gaussian_head.pt", model=model, config=config, step=step, metrics=latest_metrics)
            if step >= max_steps:
                break
    if model is None:
        raise RuntimeError("Training finished without initializing the Gaussian model.")
    checkpoint = ckpt_dir / "gaussian_head.pt"
    _save_checkpoint(checkpoint, model=model, config=config, step=step, metrics=latest_metrics)
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "steps": int(step),
        "best_loss": float(best),
        "last_metrics": latest_metrics,
        "checkpoint": str(checkpoint),
        "elapsed_sec": float(time.time() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--renderer", default=None, choices=["gsplat360", "soft_splat"])
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()
    config = load_gaussian_train_config(args.config)
    if args.steps is not None:
        config.setdefault("Training", {})["steps"] = int(args.steps)
    if args.batch_size is not None:
        config.setdefault("Training", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config.setdefault("Training", {})["num_workers"] = int(args.num_workers)
    if args.output_dir is not None:
        config.setdefault("Training", {})["output_dir"] = args.output_dir
    if args.renderer is not None:
        config.setdefault("Renderer", {})["backend"] = args.renderer
    if args.max_clips is not None:
        config.setdefault("Dataset", {})["max_clips"] = int(args.max_clips)
    if args.wandb_mode is not None:
        if args.wandb_mode == "disabled":
            config.setdefault("WeightsAndBiases", {})["enabled"] = False
        else:
            config.setdefault("WeightsAndBiases", {})["enabled"] = True
            config.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    result = train_gaussian_head(config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
