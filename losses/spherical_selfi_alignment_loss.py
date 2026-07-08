"""Selfi-style spherical feature alignment loss for Stage 1C."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from geometry.spherical_erp import (
    erp_pixel_to_unit_ray,
    sample_erp_with_wrap,
    spherical_geodesic_distance,
    unit_ray_to_erp_pixel,
    wrap_longitude_pixel,
)
from geometry.spherical_pseudo_correspondence import SphericalCorrespondence


@dataclass(frozen=True)
class SphericalSelfiAlignmentLossConfig:
    """Configuration for Stage 1C spherical Selfi alignment."""

    mode: str = "global_lowres"
    loss_stride: int = 4
    local_window_radius: int = 16
    temperature: float = 0.07
    max_queries: int | None = 512
    erp_aux_weight: float = 0.01
    eps: float = 1.0e-6


def _ensure_batched_corr(corr: SphericalCorrespondence) -> dict[str, torch.Tensor]:
    valid = corr.valid_mask
    if valid.ndim == 2:
        return {
            "src_view": corr.src_view.unsqueeze(0),
            "tgt_view": corr.tgt_view.unsqueeze(0),
            "src_uv": corr.src_uv.unsqueeze(0),
            "tgt_uv": corr.tgt_uv.unsqueeze(0),
            "tgt_ray": corr.tgt_ray.unsqueeze(0),
            "valid": corr.valid_mask.unsqueeze(0),
            "weight": corr.weight.unsqueeze(0),
        }
    if valid.ndim == 3:
        return {
            "src_view": corr.src_view,
            "tgt_view": corr.tgt_view,
            "src_uv": corr.src_uv,
            "tgt_uv": corr.tgt_uv,
            "tgt_ray": corr.tgt_ray,
            "valid": corr.valid_mask,
            "weight": corr.weight,
        }
    raise ValueError(f"corr.valid_mask must have shape ExS or BxExS, got {tuple(valid.shape)}.")


def _flatten_valid(
    features: torch.Tensor,
    corr: SphericalCorrespondence,
    max_queries: int | None,
) -> dict[str, torch.Tensor]:
    if features.ndim != 5:
        raise ValueError(f"adapter_features must have shape BxVxCxHxW, got {tuple(features.shape)}.")
    batch_size, num_views = int(features.shape[0]), int(features.shape[1])
    data = _ensure_batched_corr(corr)
    if int(data["valid"].shape[0]) != batch_size:
        if int(data["valid"].shape[0]) == 1 and batch_size > 1:
            data = {
                key: value.expand(batch_size, *value.shape[1:]) if value.ndim >= 1 else value
                for key, value in data.items()
            }
        else:
            raise ValueError(
                f"Correspondence batch size {int(data['valid'].shape[0])} does not match features batch size {batch_size}."
            )
    valid = data["valid"].reshape(-1).bool()
    if not valid.any():
        empty_long = torch.empty(0, device=features.device, dtype=torch.long)
        empty_float = torch.empty(0, device=features.device, dtype=features.dtype)
        return {
            "flat_src": empty_long,
            "flat_tgt": empty_long,
            "src_uv": empty_float.view(0, 2),
            "tgt_uv": empty_float.view(0, 2),
            "tgt_ray": empty_float.view(0, 3),
            "weight": empty_float,
        }
    b_ids = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1).expand_as(data["valid"])
    flat_src = (b_ids * num_views + data["src_view"].long()).reshape(-1)[valid]
    flat_tgt = (b_ids * num_views + data["tgt_view"].long()).reshape(-1)[valid]
    if int(flat_src.max()) >= batch_size * num_views or int(flat_tgt.max()) >= batch_size * num_views:
        raise ValueError("Correspondence view indices exceed adapter feature view count.")
    out = {
        "flat_src": flat_src,
        "flat_tgt": flat_tgt,
        "src_uv": data["src_uv"].to(device=features.device, dtype=features.dtype).reshape(-1, 2)[valid],
        "tgt_uv": data["tgt_uv"].to(device=features.device, dtype=features.dtype).reshape(-1, 2)[valid],
        "tgt_ray": data["tgt_ray"].to(device=features.device, dtype=features.dtype).reshape(-1, 3)[valid],
        "weight": data["weight"].to(device=features.device, dtype=features.dtype).reshape(-1)[valid],
    }
    if max_queries is not None and out["flat_src"].numel() > int(max_queries):
        keep = torch.linspace(
            0,
            out["flat_src"].numel() - 1,
            steps=int(max_queries),
            device=features.device,
        ).round().long()
        out = {key: value[keep] for key, value in out.items()}
    return out


def _feature_grid_uv(height: int, width: int, image_hw: tuple[int, int], *, device, dtype) -> torch.Tensor:
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    scale_x = float(image_w) / float(width)
    scale_y = float(image_h) / float(height)
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx * scale_x, yy * scale_y], dim=-1).reshape(-1, 2)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    w = weight.clamp_min(0.0)
    if w.numel() == 0:
        return value.sum() * 0.0
    return (value * w).sum() / w.sum().clamp_min(eps)


def _seam_aware_pixel_l1(pred_uv: torch.Tensor, tgt_uv: torch.Tensor, width: int) -> torch.Tensor:
    du = torch.remainder(pred_uv[:, 0] - tgt_uv[:, 0] + float(width) * 0.5, float(width)) - float(width) * 0.5
    dv = pred_uv[:, 1] - tgt_uv[:, 1]
    return torch.stack([du / float(width), dv / max(float(width), 1.0)], dim=-1).abs().sum(dim=-1)


class SphericalSelfiAlignmentLoss(nn.Module):
    """Compute spherical Selfi alignment loss from adapter features."""

    def __init__(self, config: SphericalSelfiAlignmentLossConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = config or SphericalSelfiAlignmentLossConfig(**kwargs)

    def forward(
        self,
        adapter_features: torch.Tensor,
        correspondences: SphericalCorrespondence,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = F.normalize(adapter_features, dim=2, eps=self.config.eps)
        flat = features.reshape(features.shape[0] * features.shape[1], features.shape[2], features.shape[3], features.shape[4])
        height, width = int(features.shape[-2]), int(features.shape[-1])
        entries = _flatten_valid(features, correspondences, self.config.max_queries)
        if entries["flat_src"].numel() == 0:
            zero = features.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "spherical": zero.detach(),
                "erp_aux": zero.detach(),
                "num_queries": torch.tensor(0.0, device=features.device),
                "mean_angular_deg": torch.tensor(0.0, device=features.device),
            }
        src_values = sample_erp_with_wrap(flat[entries["flat_src"]], entries["src_uv"])
        src_values = F.normalize(src_values, dim=-1, eps=self.config.eps)
        if self.config.mode == "global_lowres":
            pred_ray, pred_uv = self._global_lowres_prediction(flat, src_values, entries, image_hw=(height, width))
        elif self.config.mode == "local_fullres":
            pred_ray, pred_uv = self._local_fullres_prediction(flat, src_values, entries, image_hw=(height, width))
        else:
            raise ValueError(f"Unsupported spherical Selfi matching mode: {self.config.mode!r}.")

        angular = spherical_geodesic_distance(pred_ray, entries["tgt_ray"], eps=self.config.eps)
        spherical = _weighted_mean(angular, entries["weight"], self.config.eps)
        erp_aux = _weighted_mean(_seam_aware_pixel_l1(pred_uv, entries["tgt_uv"], width), entries["weight"], self.config.eps)
        loss = spherical + float(self.config.erp_aux_weight) * erp_aux
        return loss, {
            "loss": loss.detach(),
            "spherical": spherical.detach(),
            "erp_aux": erp_aux.detach(),
            "num_queries": torch.tensor(float(entries["flat_src"].numel()), device=features.device),
            "mean_angular_deg": torch.rad2deg(angular.detach()).mean(),
        }

    def _global_lowres_prediction(
        self,
        flat_features: torch.Tensor,
        src_values: torch.Tensor,
        entries: dict[str, torch.Tensor],
        *,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride = max(1, int(self.config.loss_stride))
        target_h = max(1, int(math.ceil(float(image_hw[0]) / float(stride))))
        target_w = max(1, int(math.ceil(float(image_hw[1]) / float(stride))))
        target = F.interpolate(flat_features, size=(target_h, target_w), mode="bilinear", align_corners=False)
        target = F.normalize(target, dim=1, eps=self.config.eps)
        selected = target[entries["flat_tgt"]].flatten(2)
        scores = torch.einsum("nc,nck->nk", src_values, selected) / max(float(self.config.temperature), self.config.eps)
        prob = torch.softmax(scores, dim=-1)
        uv = _feature_grid_uv(target_h, target_w, image_hw, device=flat_features.device, dtype=flat_features.dtype)
        rays = erp_pixel_to_unit_ray(uv, image_hw[0], image_hw[1]).to(device=flat_features.device, dtype=flat_features.dtype)
        pred_ray = F.normalize(prob @ rays, dim=-1, eps=self.config.eps)
        pred_uv = unit_ray_to_erp_pixel(pred_ray, image_hw[0], image_hw[1])
        return pred_ray, pred_uv

    def _local_fullres_prediction(
        self,
        flat_features: torch.Tensor,
        src_values: torch.Tensor,
        entries: dict[str, torch.Tensor],
        *,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius = max(0, int(self.config.local_window_radius))
        offsets = [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        offset_t = torch.tensor(offsets, device=flat_features.device, dtype=flat_features.dtype)
        candidates = entries["tgt_uv"].unsqueeze(1) + offset_t.view(1, -1, 2)
        candidates = candidates.clone()
        candidates[..., 0] = wrap_longitude_pixel(candidates[..., 0], image_hw[1])
        candidates[..., 1] = candidates[..., 1].clamp(0.5, float(image_hw[0]) - 0.5)
        target_maps = flat_features[entries["flat_tgt"]]
        sampled = sample_erp_with_wrap(target_maps, candidates)
        sampled = F.normalize(sampled, dim=-1, eps=self.config.eps)
        scores = torch.einsum("nkc,nc->nk", sampled, src_values) / max(float(self.config.temperature), self.config.eps)
        prob = torch.softmax(scores, dim=-1)
        rays = erp_pixel_to_unit_ray(candidates, image_hw[0], image_hw[1]).to(device=flat_features.device, dtype=flat_features.dtype)
        pred_ray = F.normalize((prob.unsqueeze(-1) * rays).sum(dim=1), dim=-1, eps=self.config.eps)
        pred_uv = unit_ray_to_erp_pixel(pred_ray, image_hw[0], image_hw[1])
        return pred_ray, pred_uv


def compute_spherical_selfi_alignment_loss(
    adapter_features: torch.Tensor,
    correspondences: SphericalCorrespondence,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Functional convenience wrapper for ``SphericalSelfiAlignmentLoss``."""

    return SphericalSelfiAlignmentLoss(**kwargs)(adapter_features, correspondences)
