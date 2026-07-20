from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from frontend.pano_droid.spherical_ba import so3_exp
from geometry.sim3 import apply_sim3_to_c2w, sim3_from_components
from geometry.trajectory_metrics import (
    c2w_trajectory_metrics,
    pfgs360_normalized_trajectory_alignment,
)


def _trajectory(count: int = 12) -> torch.Tensor:
    poses = torch.eye(4).repeat(count, 1, 1)
    for index in range(count):
        poses[index, :3, :3] = so3_exp(torch.tensor([0.0, 0.02 * index, 0.0]))
        poses[index, :3, 3] = torch.tensor(
            [0.25 * index, 0.03 * math.sin(index), 0.08 * index]
        )
    return poses


def _pfgs360_official_numpy_reference(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Independent transcription of the official PFGS360 evaluation order."""

    predicted_normalized = predicted - predicted.mean(axis=0, keepdims=True)
    target_normalized = target - target.mean(axis=0, keepdims=True)
    predicted_normalized /= np.linalg.norm(predicted_normalized)
    target_normalized /= np.linalg.norm(target_normalized)

    # scipy.linalg.orthogonal_procrustes(target, predicted) returns this
    # singular-value sum as its second result.  PFGS360 discards the rotation.
    procrustes_scale = np.linalg.svd(
        target_normalized.T @ predicted_normalized,
        compute_uv=False,
    ).sum()
    predicted_pre_scaled = predicted_normalized * procrustes_scale

    model_mean = target_normalized.mean(axis=0)
    data_mean = predicted_pre_scaled.mean(axis=0)
    model_centered = target_normalized - model_mean
    data_centered = predicted_pre_scaled - data_mean
    covariance = model_centered.T @ data_centered / len(predicted)
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    variance = np.square(data_centered).sum() / len(predicted)
    sim3_scale = np.trace(np.diag(singular) @ correction) / variance
    translation = model_mean - sim3_scale * (rotation @ data_mean)
    aligned = (
        sim3_scale * (rotation @ predicted_pre_scaled.T)
        + translation[:, None]
    ).T
    errors = np.linalg.norm(aligned - target_normalized, axis=1)
    ate = float(np.sqrt(np.square(errors).mean()))
    return aligned, target_normalized, errors, ate


def test_trajectory_metrics_remove_one_global_sim3_gauge() -> None:
    target = _trajectory()
    rotation = so3_exp(torch.tensor([0.1, -0.2, 0.05]))
    transform = sim3_from_components(2.0, rotation, torch.tensor([3.0, -1.0, 0.5]))
    predicted = apply_sim3_to_c2w(
        transform.unsqueeze(0).expand(len(target), -1, -1), target
    )
    metrics = c2w_trajectory_metrics(predicted, target)
    assert abs(metrics["alignment_scale"] - 0.5) < 1.0e-5
    assert metrics["sim3_ate_rmse"] < 1.0e-5
    assert metrics["rpe_delta_1_translation_rmse"] < 1.0e-5
    assert metrics["rpe_delta_1_rotation_mean_deg"] < 1.0e-2
    assert metrics["so3_aligned_rotation_ape_mean_deg"] < 1.0e-3
    assert metrics["scale_drift_percent"] < 1.0e-4
    assert metrics["se3_ate_rmse"] > 0.1


def test_trajectory_metrics_detect_local_scale_drift() -> None:
    target = _trajectory()
    predicted = target.clone()
    factor = torch.linspace(1.0, 1.8, len(predicted))
    predicted[:, :3, 3] *= factor[:, None]
    metrics = c2w_trajectory_metrics(predicted, target)
    assert metrics["rpe_delta_1_translation_rmse"] > 0.01
    assert metrics["rpe_delta_1_log_scale_error_mean"] > 0.01
    assert metrics["scale_drift_percent"] > 1.0


def test_orientation_alignment_is_independent_from_center_alignment() -> None:
    target = _trajectory()
    predicted = target.clone()
    orientation_gauge = so3_exp(torch.tensor([0.15, -0.08, 0.04]))
    predicted[:, :3, :3] = orientation_gauge @ target[:, :3, :3]

    metrics = c2w_trajectory_metrics(predicted, target)

    assert metrics["sim3_ate_rmse"] < 1.0e-6
    assert metrics["rotation_ape_mean_deg"] > 5.0
    assert metrics["so3_aligned_rotation_ape_mean_deg"] < 1.0e-3


def test_pfgs360_ate_matches_official_normalized_protocol() -> None:
    target = _trajectory()[:, :3, 3].double()
    predicted = target.clone()
    predicted[:, 0] += torch.linspace(0.0, 0.8, len(predicted), dtype=torch.double)
    predicted[:, 1] += 0.15 * torch.sin(
        torch.linspace(0.0, 2.5, len(predicted), dtype=torch.double)
    )

    aligned, normalized_gt, errors, metrics = (
        pfgs360_normalized_trajectory_alignment(predicted, target)
    )
    reference = _pfgs360_official_numpy_reference(
        predicted.numpy(),
        target.numpy(),
    )

    assert np.allclose(aligned.numpy(), reference[0], atol=1.0e-10)
    assert np.allclose(normalized_gt.numpy(), reference[1], atol=1.0e-10)
    assert np.allclose(errors.numpy(), reference[2], atol=1.0e-10)
    assert metrics["pfgs360_ate"] == pytest.approx(reference[3], abs=1.0e-12)


def test_pfgs360_ate_is_dimensionless_and_explicitly_enabled() -> None:
    target = _trajectory().double()
    predicted = target.clone()
    predicted[:, :3, 3] *= torch.linspace(
        1.0,
        1.6,
        len(predicted),
        dtype=torch.double,
    )[:, None]

    standard_metrics = c2w_trajectory_metrics(predicted, target)
    combined_metrics = c2w_trajectory_metrics(
        predicted,
        target,
        include_pfgs360=True,
    )
    scaled_predicted = predicted.clone()
    scaled_target = target.clone()
    scaled_predicted[:, :3, 3] = 7.0 * predicted[:, :3, 3] + torch.tensor(
        [4.0, -3.0, 2.0],
        dtype=torch.double,
    )
    scaled_target[:, :3, 3] = 3.0 * target[:, :3, 3] + torch.tensor(
        [-2.0, 5.0, 1.0],
        dtype=torch.double,
    )
    scaled_metrics = c2w_trajectory_metrics(
        scaled_predicted,
        scaled_target,
        include_pfgs360=True,
    )

    assert "pfgs360_ate" not in standard_metrics
    assert combined_metrics["pfgs360_ate"] > 0.0
    assert scaled_metrics["pfgs360_ate"] == pytest.approx(
        combined_metrics["pfgs360_ate"],
        abs=1.0e-10,
    )


def test_pfgs360_ate_rejects_zero_motion_trajectory() -> None:
    centers = torch.ones(4, 3)
    with pytest.raises(ValueError, match="non-zero motion"):
        pfgs360_normalized_trajectory_alignment(centers, centers)
