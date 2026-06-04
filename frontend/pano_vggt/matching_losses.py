"""Losses for staged PanoVGGT-M3-Sphere head training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing

from .grid_utils import feature_uv_to_image_uv
from .spherical_correspondence import SphericalCorrespondenceBatch, spherical_tangent_residual


@dataclass(frozen=True)
class PanoVGGTMatchingLossWeights:
    """Weights for PanoVGGT-M3-Sphere staged losses."""

    nce: float = 1.0
    confidence: float = 0.1
    spherical: float = 0.2
    sky: float = 0.2
    sky_dice: float = 0.5
    smoothness: float = 0.01
    temperature: float = 0.07


def _as_single_sample_maps(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim == 5:
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"{name} loss helpers currently expect B=1, got {tuple(tensor.shape)}.")
        return tensor[0]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"{name} must have shape BxNxCxHxW or NxCxHxW, got {tuple(tensor.shape)}")


def _flatten_correspondence(corr: SphericalCorrespondenceBatch) -> dict[str, torch.Tensor]:
    return {
        "src_idx": corr.src_indices.reshape(-1).long(),
        "tgt_idx": corr.tgt_indices.reshape(-1).long(),
        "src_uv": corr.src_uv.reshape(-1, 2),
        "tgt_uv": corr.tgt_uv.reshape(-1, 2),
        "tgt_bearing": corr.tgt_bearing.reshape(-1, 3),
        "valid": corr.valid_mask.reshape(-1).bool(),
    }


def _filter_flat_with_non_sky(
    flat: dict[str, torch.Tensor],
    non_sky_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if non_sky_mask is None:
        return flat
    non_sky = _as_single_sample_maps(non_sky_mask.float(), name="non_sky_mask")
    src_non_sky = sample_feature_values(non_sky, flat["src_idx"], flat["src_uv"])[:, 0] > 0.5
    tgt_non_sky = sample_feature_values(non_sky, flat["tgt_idx"], flat["tgt_uv"])[:, 0] > 0.5
    out = dict(flat)
    out["valid"] = flat["valid"] & src_non_sky & tgt_non_sky
    return out


def _select_valid(flat: dict[str, torch.Tensor], max_correspondences: int | None) -> torch.Tensor:
    valid_idx = torch.nonzero(flat["valid"], as_tuple=False).flatten()
    if max_correspondences is not None and valid_idx.numel() > int(max_correspondences):
        keep = torch.linspace(
            0,
            valid_idx.numel() - 1,
            steps=int(max_correspondences),
            device=valid_idx.device,
        ).round().long()
        valid_idx = valid_idx[keep]
    return valid_idx


def _uv_to_grid(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    norm_x = 2.0 * (uv[..., 0] - 0.5) / max(width - 1, 1) - 1.0
    norm_y = 2.0 * (uv[..., 1] - 0.5) / max(height - 1, 1) - 1.0
    return torch.stack([norm_x, norm_y], dim=-1)


def sample_feature_values(maps: torch.Tensor, frame_indices: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``maps`` at feature-grid ``uv`` coordinates."""

    if maps.ndim != 4:
        raise ValueError(f"maps must have shape NxCxHxW, got {tuple(maps.shape)}")
    if uv.ndim != 2 or int(uv.shape[-1]) != 2:
        raise ValueError(f"uv must have shape Px2, got {tuple(uv.shape)}")
    height, width = int(maps.shape[-2]), int(maps.shape[-1])
    selected = maps[frame_indices.long()]
    grid = _uv_to_grid(uv.to(device=maps.device, dtype=maps.dtype), height, width).view(-1, 1, 1, 2)
    sampled = F.grid_sample(selected, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, 0, 0]


def _zero_loss(reference: torch.Tensor, **stats: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = reference.device
    loss = reference.sum() * 0.0
    return loss, {key: torch.tensor(float(value), device=device) for key, value in stats.items()}


def symmetric_info_nce_loss(
    dense_descriptors: torch.Tensor,
    correspondences: SphericalCorrespondenceBatch,
    *,
    non_sky_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
    max_correspondences: int = 1024,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute bidirectional M3-style InfoNCE on dense descriptor positives."""

    descriptors = _as_single_sample_maps(dense_descriptors, name="dense_descriptors")
    flat = _filter_flat_with_non_sky(_flatten_correspondence(correspondences), non_sky_mask)
    valid_idx = _select_valid(flat, max_correspondences)
    if valid_idx.numel() < 2:
        return _zero_loss(dense_descriptors, n_pos=float(valid_idx.numel()), n_neg=0.0, mean_pos_sim=0.0, mean_neg_sim=0.0)

    src = sample_feature_values(descriptors, flat["src_idx"][valid_idx], flat["src_uv"][valid_idx])
    tgt = sample_feature_values(descriptors, flat["tgt_idx"][valid_idx], flat["tgt_uv"][valid_idx])
    src = F.normalize(src, dim=-1, eps=1.0e-6)
    tgt = F.normalize(tgt, dim=-1, eps=1.0e-6)
    logits = src @ tgt.transpose(0, 1) / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
    with torch.no_grad():
        pos = logits.diag() * float(temperature)
        offdiag = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg = logits[offdiag] * float(temperature)
    return loss, {
        "n_pos": torch.tensor(float(valid_idx.numel()), device=logits.device),
        "n_neg": torch.tensor(float(neg.numel()), device=logits.device),
        "mean_pos_sim": pos.mean().detach(),
        "mean_neg_sim": neg.mean().detach() if neg.numel() else torch.tensor(0.0, device=logits.device),
    }


def confidence_calibration_loss(
    match_confidence: torch.Tensor,
    correspondences: SphericalCorrespondenceBatch,
    *,
    non_sky_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calibrate match confidence against valid non-sky correspondence labels."""

    confidence = _as_single_sample_maps(match_confidence, name="match_confidence")
    flat = _filter_flat_with_non_sky(_flatten_correspondence(correspondences), non_sky_mask)
    target = flat["valid"].float()
    src_conf = sample_feature_values(confidence, flat["src_idx"], flat["src_uv"])[:, 0]
    tgt_conf = sample_feature_values(confidence, flat["tgt_idx"], flat["tgt_uv"])[:, 0]
    pred = (src_conf * tgt_conf).clamp(1.0e-5, 1.0 - 1.0e-5)
    loss = F.binary_cross_entropy(pred, target.to(pred))
    return loss, {
        "confidence_target_ratio": target.to(pred).mean().detach(),
        "confidence_mean": pred.mean().detach(),
    }


def spherical_match_regression_loss(
    dense_descriptors: torch.Tensor,
    correspondences: SphericalCorrespondenceBatch,
    *,
    image_hw: tuple[int, int],
    non_sky_mask: torch.Tensor | None = None,
    search_radius: int = 2,
    max_correspondences: int = 512,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress local descriptor soft matches with an S2 tangent residual."""

    descriptors = _as_single_sample_maps(dense_descriptors, name="dense_descriptors")
    flat = _filter_flat_with_non_sky(_flatten_correspondence(correspondences), non_sky_mask)
    valid_idx = _select_valid(flat, max_correspondences)
    if valid_idx.numel() == 0:
        return _zero_loss(dense_descriptors, spherical_residual_deg=0.0, spherical_n=0.0)

    height_f, width_f = int(descriptors.shape[-2]), int(descriptors.shape[-1])
    src = sample_feature_values(descriptors, flat["src_idx"][valid_idx], flat["src_uv"][valid_idx])
    src = F.normalize(src, dim=-1, eps=1.0e-6)
    tgt_idx = flat["tgt_idx"][valid_idx]
    tgt_uv = flat["tgt_uv"][valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    offsets = []
    radius = int(search_radius)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((float(dx), float(dy)))
    offset_t = torch.tensor(offsets, device=descriptors.device, dtype=descriptors.dtype)
    candidates = tgt_uv.unsqueeze(1) + offset_t.view(1, -1, 2)
    candidates = candidates.clone()
    candidates[..., 0] = torch.remainder(candidates[..., 0], float(width_f))
    candidates[..., 1] = candidates[..., 1].clamp(0.5, float(height_f) - 0.5)
    target_maps = descriptors[tgt_idx.long()]
    grid = _uv_to_grid(candidates, height_f, width_f).view(candidates.shape[0], candidates.shape[1], 1, 2)
    sampled = F.grid_sample(target_maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled[:, :, :, 0].transpose(1, 2)
    sampled = F.normalize(sampled, dim=-1, eps=1.0e-6)
    scores = (sampled * src.unsqueeze(1)).sum(dim=-1)
    prob = torch.softmax(scores, dim=-1)
    expected_uv = (prob.unsqueeze(-1) * candidates).sum(dim=1)

    feature_hw = (height_f, width_f)
    pred_image_uv = feature_uv_to_image_uv(expected_uv, feature_hw, image_hw)
    pred_bearing = erp_pixel_to_bearing(pred_image_uv, int(image_hw[0]), int(image_hw[1])).to(descriptors)
    gt_bearing = flat["tgt_bearing"][valid_idx].to(device=descriptors.device, dtype=descriptors.dtype)
    residual = spherical_tangent_residual(gt_bearing, pred_bearing)
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=float(huber_delta))
    return loss, {
        "spherical_residual_deg": torch.rad2deg(residual.norm(dim=-1)).mean().detach(),
        "spherical_n": torch.tensor(float(valid_idx.numel()), device=descriptors.device),
    }


def sky_bce_dice_loss(
    sky_logits: torch.Tensor,
    sky_mask_gt: torch.Tensor,
    *,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute BCE+Dice sky-mask loss and common segmentation metrics."""

    logits = sky_logits
    target = sky_mask_gt.float().to(device=logits.device)
    if target.ndim == 4 and logits.ndim == 5:
        target = target.unsqueeze(0)
    if target.shape[-2:] != logits.shape[-2:]:
        flat = target.reshape(-1, 1, target.shape[-2], target.shape[-1])
        flat = F.interpolate(flat, size=logits.shape[-2:], mode="nearest")
        target = flat.view(*target.shape[:-2], logits.shape[-2], logits.shape[-1])
    if target.shape != logits.shape:
        raise ValueError(f"sky target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1.0e-6) / (denom + 1.0e-6)).mean()
    pred_mask = prob >= 0.5
    gt_mask = target >= 0.5
    pixel_acc = (pred_mask == gt_mask).float().mean()
    tp = (pred_mask & gt_mask).float().sum()
    fp = (pred_mask & ~gt_mask).float().sum()
    fn = (~pred_mask & gt_mask).float().sum()
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    loss = bce + float(dice_weight) * dice
    return loss, {
        "sky_bce": bce.detach(),
        "sky_dice": dice.detach(),
        "sky_iou": iou.detach(),
        "sky_pixel_acc": pixel_acc.detach(),
        "sky_precision": precision.detach(),
        "sky_recall": recall.detach(),
    }


def descriptor_smoothness_loss(dense_descriptors: torch.Tensor) -> torch.Tensor:
    """Lightweight descriptor smoothness regularizer for staged training."""

    dx = dense_descriptors[..., :, 1:] - dense_descriptors[..., :, :-1]
    dy = dense_descriptors[..., 1:, :] - dense_descriptors[..., :-1, :]
    return torch.sqrt(dx.square() + 1.0e-6).mean() + torch.sqrt(dy.square() + 1.0e-6).mean()


class PanoVGGTMatchingSkyLoss:
    """Compute staged losses for sky-only, matching-only, or calibration mode."""

    def __init__(self, weights: PanoVGGTMatchingLossWeights | None = None) -> None:
        self.weights = weights or PanoVGGTMatchingLossWeights()

    def sky_only(self, outputs: dict[str, torch.Tensor], sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute sky-only loss."""

        loss, metrics = sky_bce_dice_loss(
            outputs["sky_logits"],
            sample["sky_mask"],
            dice_weight=self.weights.sky_dice,
        )
        return self.weights.sky * loss, {"loss": (self.weights.sky * loss).detach(), **metrics}

    def matching_only(
        self,
        outputs: dict[str, torch.Tensor],
        correspondences: SphericalCorrespondenceBatch,
        *,
        image_hw: tuple[int, int],
        non_sky_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute descriptor and confidence losses without sky-head supervision."""

        total = outputs["dense_descriptors"].sum() * 0.0
        metrics: dict[str, torch.Tensor] = {}
        nce, nce_stats = symmetric_info_nce_loss(
            outputs["dense_descriptors"],
            correspondences,
            non_sky_mask=non_sky_mask,
            temperature=self.weights.temperature,
        )
        sph, sph_stats = spherical_match_regression_loss(
            outputs["dense_descriptors"],
            correspondences,
            image_hw=image_hw,
            non_sky_mask=non_sky_mask,
        )
        conf, conf_stats = confidence_calibration_loss(
            outputs["match_confidence"],
            correspondences,
            non_sky_mask=non_sky_mask,
        )
        smooth = descriptor_smoothness_loss(outputs["dense_descriptors"])
        total = (
            total
            + self.weights.nce * nce
            + self.weights.spherical * sph
            + self.weights.confidence * conf
            + self.weights.smoothness * smooth
        )
        metrics.update(
            {
                "loss": total.detach(),
                "nce": nce.detach(),
                "spherical": sph.detach(),
                "confidence": conf.detach(),
                "smoothness": smooth.detach(),
            }
        )
        metrics.update(nce_stats)
        metrics.update(sph_stats)
        metrics.update(conf_stats)
        return total, metrics
