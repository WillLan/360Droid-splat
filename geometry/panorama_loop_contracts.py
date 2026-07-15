"""Backend-neutral contracts for panoramic loop-closure measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass
class LoopPoseMeasurement:
    """Compact verified pose measurement before factor-graph materialization."""

    kind: Literal["sim3", "coincident"]
    source: int
    target: int
    edge_type: str
    measurement_target_to_source: torch.Tensor | None = None
    information_diag: torch.Tensor | None = None
    source_local_pose: torch.Tensor | None = None
    target_local_pose: torch.Tensor | None = None
    measured_source_to_target_rotation: torch.Tensor | None = None
    center_weight: float = 1.0
    rotation_weight: float = 1.0
    robust_delta: float = 2.5
    dcs_phi: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "sim3":
            if self.measurement_target_to_source is None or tuple(
                self.measurement_target_to_source.shape
            ) != (4, 4):
                raise ValueError("Sim(3) loop measurement requires a 4x4 transform")
            if self.information_diag is None or int(self.information_diag.numel()) != 7:
                raise ValueError("Sim(3) loop measurement requires seven information entries")
        elif self.kind == "coincident":
            tensors = (
                self.source_local_pose,
                self.target_local_pose,
                self.measured_source_to_target_rotation,
            )
            if any(value is None for value in tensors):
                raise ValueError("Coincident loop measurement requires local poses and rotation")
            assert self.source_local_pose is not None
            assert self.target_local_pose is not None
            assert self.measured_source_to_target_rotation is not None
            if tuple(self.source_local_pose.shape) != (4, 4) or tuple(
                self.target_local_pose.shape
            ) != (4, 4):
                raise ValueError("Coincident loop local poses must be 4x4")
            if tuple(self.measured_source_to_target_rotation.shape) != (3, 3):
                raise ValueError("Coincident loop rotation must be 3x3")
        else:
            raise ValueError(f"Unsupported loop pose measurement kind: {self.kind!r}")


@dataclass
class DenseSphericalLoopMeasurement:
    """Backend-neutral dense S²/depth correspondence block."""

    source: int
    target: int
    source_local_pose: torch.Tensor
    target_local_pose: torch.Tensor
    source_bearing: torch.Tensor
    target_bearing: torch.Tensor
    source_depth: torch.Tensor
    target_depth: torch.Tensor
    factor_weight: torch.Tensor
    depth_factor_weight: float = 0.1
    s2_huber_delta_deg: float = 1.0
    use_depth: bool = True
    robust_delta: float = float("inf")
    edge_type: str = "loop_dense_spherical"
    dcs_phi: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = int(self.source_depth.numel())
        if count < 1:
            raise ValueError("Dense loop measurement requires at least one correspondence")
        if tuple(self.source_bearing.shape) != (count, 3) or tuple(
            self.target_bearing.shape
        ) != (count, 3):
            raise ValueError("Dense loop bearings must both have shape Nx3")
        if int(self.target_depth.numel()) != count or int(self.factor_weight.numel()) != count:
            raise ValueError("Dense loop depths and weights must share correspondence count")


@dataclass
class PanoramaLoopVerification:
    """Complete retrieval/verification result without a graph dependency."""

    accepted: bool
    factor: LoopPoseMeasurement | None
    source_window_id: int
    target_window_id: int
    retrieval_score: float
    yaw_shift_columns: int
    num_matches: int
    inlier_ratio: float
    residual: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_factors: tuple[DenseSphericalLoopMeasurement, ...] = ()
