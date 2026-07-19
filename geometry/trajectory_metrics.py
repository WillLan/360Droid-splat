"""Trajectory metrics for c2w camera poses with monocular scale diagnostics."""

from __future__ import annotations

import math

import torch


def _rotation_angle_deg(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(skew, dim=-1)
    return torch.rad2deg(torch.atan2(sine, cosine))


def _align_centers(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_mean = predicted.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = predicted - source_mean
    target_centered = target - target_mean
    covariance = source_centered.transpose(0, 1) @ target_centered / max(1, predicted.shape[0])
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=predicted.device, dtype=predicted.dtype)
    if float(torch.linalg.det(vh.transpose(0, 1) @ u.transpose(0, 1))) < 0.0:
        correction[-1, -1] = -1.0
    rotation = vh.transpose(0, 1) @ correction @ u.transpose(0, 1)
    if allow_scale:
        variance = source_centered.square().sum(dim=-1).mean().clamp_min(1.0e-12)
        scale = (singular * correction.diagonal()).sum() / variance
    else:
        scale = predicted.new_tensor(1.0)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (predicted @ rotation.transpose(0, 1)) + translation
    return aligned, scale, rotation, translation


def _align_rotations(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the global SO(3) gauge minimizing chordal orientation error."""

    covariance = (
        target @ predicted.transpose(1, 2)
    ).sum(dim=0)
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(
        3, device=predicted.device, dtype=predicted.dtype
    )
    if float(torch.linalg.det(u @ vh)) < 0.0:
        correction[-1, -1] = -1.0
    return u @ correction @ vh


def c2w_trajectory_metrics(
    predicted_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    deltas: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float]:
    """Compute ATE, RPE, and scale drift for matching c2w trajectories.

    RPE translation uses the single global Sim(3) scale fitted for ATE.  The
    reported scale-drift terms therefore measure *local* scale inconsistency,
    not the unavoidable monocular global scale gauge.
    """

    if (
        predicted_c2w.shape != target_c2w.shape
        or predicted_c2w.ndim != 3
        or predicted_c2w.shape[-2:] != (4, 4)
    ):
        raise ValueError("Trajectory metrics expect matching Nx4x4 c2w tensors")
    count = int(predicted_c2w.shape[0])
    if count < 2:
        raise ValueError("Trajectory metrics require at least two poses")
    predicted = predicted_c2w.detach().double()
    target = target_c2w.detach().double()
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(target).all()):
        raise ValueError("Trajectory poses must be finite")

    pred_center = predicted[:, :3, 3]
    target_center = target[:, :3, 3]
    sim3_center, scale, alignment_rotation, _ = _align_centers(
        pred_center, target_center, allow_scale=True
    )
    se3_center, _, _, _ = _align_centers(pred_center, target_center, allow_scale=False)
    sim3_error = (sim3_center - target_center).norm(dim=-1)
    se3_error = (se3_center - target_center).norm(dim=-1)
    aligned_rotation = alignment_rotation.unsqueeze(0) @ predicted[:, :3, :3]
    rotation_ape = _rotation_angle_deg(
        target[:, :3, :3].transpose(1, 2) @ aligned_rotation
    )
    orientation_alignment = _align_rotations(
        predicted[:, :3, :3],
        target[:, :3, :3],
    )
    orientation_aligned_rotation = (
        orientation_alignment.unsqueeze(0) @ predicted[:, :3, :3]
    )
    so3_rotation_ape = _rotation_angle_deg(
        target[:, :3, :3].transpose(1, 2)
        @ orientation_aligned_rotation
    )

    metrics = {
        "alignment_scale": float(scale.cpu()),
        "sim3_ate_rmse": float(sim3_error.square().mean().sqrt().cpu()),
        "sim3_ate_mean": float(sim3_error.mean().cpu()),
        "sim3_ate_median": float(sim3_error.median().cpu()),
        "sim3_ate_max": float(sim3_error.max().cpu()),
        "se3_ate_rmse": float(se3_error.square().mean().sqrt().cpu()),
        "rotation_ape_mean_deg": float(rotation_ape.mean().cpu()),
        "rotation_ape_median_deg": float(rotation_ape.median().cpu()),
        "so3_aligned_rotation_ape_mean_deg": float(
            so3_rotation_ape.mean().cpu()
        ),
        "so3_aligned_rotation_ape_median_deg": float(
            so3_rotation_ape.median().cpu()
        ),
    }

    for delta in sorted({int(value) for value in deltas if 0 < int(value) < count}):
        pred_relative = torch.linalg.inv(predicted[:-delta]) @ predicted[delta:]
        target_relative = torch.linalg.inv(target[:-delta]) @ target[delta:]
        rotation_error = _rotation_angle_deg(
            target_relative[:, :3, :3].transpose(1, 2) @ pred_relative[:, :3, :3]
        )
        pred_translation = scale * pred_relative[:, :3, 3]
        target_translation = target_relative[:, :3, 3]
        translation_error = (pred_translation - target_translation).norm(dim=-1)
        pred_length = pred_translation.norm(dim=-1)
        target_length = target_translation.norm(dim=-1)
        valid_scale = (pred_length > 1.0e-10) & (target_length > 1.0e-10)
        prefix = f"rpe_delta_{delta}"
        metrics[f"{prefix}_rotation_mean_deg"] = float(rotation_error.mean().cpu())
        metrics[f"{prefix}_rotation_rmse_deg"] = float(
            rotation_error.square().mean().sqrt().cpu()
        )
        metrics[f"{prefix}_translation_rmse"] = float(
            translation_error.square().mean().sqrt().cpu()
        )
        if bool(valid_scale.any()):
            log_scale_error = (
                pred_length[valid_scale] / target_length[valid_scale]
            ).log().abs()
            metrics[f"{prefix}_log_scale_error_mean"] = float(log_scale_error.mean().cpu())
            metrics[f"{prefix}_log_scale_error_p90"] = float(
                log_scale_error.quantile(0.9).cpu()
            )

    pred_step = scale * (pred_center[1:] - pred_center[:-1]).norm(dim=-1)
    target_step = (target_center[1:] - target_center[:-1]).norm(dim=-1)
    target_path = target_step.sum().clamp_min(1.0e-12)
    path_ratio = pred_step.sum() / target_path
    metrics["path_length_scale_ratio"] = float(path_ratio.cpu())
    metrics["scale_drift_percent"] = float((100.0 * (path_ratio - 1.0).abs()).cpu())
    return metrics
