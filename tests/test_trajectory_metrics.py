from __future__ import annotations

import math

import torch

from frontend.pano_droid.spherical_ba import so3_exp
from geometry.sim3 import apply_sim3_to_c2w, sim3_from_components
from geometry.trajectory_metrics import c2w_trajectory_metrics


def _trajectory(count: int = 12) -> torch.Tensor:
    poses = torch.eye(4).repeat(count, 1, 1)
    for index in range(count):
        poses[index, :3, :3] = so3_exp(torch.tensor([0.0, 0.02 * index, 0.0]))
        poses[index, :3, 3] = torch.tensor(
            [0.25 * index, 0.03 * math.sin(index), 0.08 * index]
        )
    return poses


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
