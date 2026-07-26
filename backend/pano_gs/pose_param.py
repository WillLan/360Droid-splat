"""Pose parameterization helpers for backend refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from frontend.pano_droid.spherical_ba import se3_exp, skew
from geometry.sim3 import canonicalize_c2w, so3_log


def ensure_homogeneous(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {tuple(T.shape)}")
    return T


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`se3_exp` for finite SE(3) transforms."""

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log(rotation)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta.square()
    matrix = skew(omega)
    eye = torch.eye(3, device=transform.device, dtype=transform.dtype)
    eye = eye.expand(*transform.shape[:-2], 3, 3)
    small = theta < 1.0e-4
    coefficient = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0,
        (
            1.0
            - 0.5
            * theta
            * torch.sin(theta)
            / (1.0 - torch.cos(theta)).clamp_min(1.0e-8)
        )
        / theta2.clamp_min(1.0e-8),
    )
    inverse_v = eye - 0.5 * matrix + coefficient[..., None] * (matrix @ matrix)
    rho = torch.einsum("...ij,...j->...i", inverse_v, translation)
    return torch.cat((rho, omega), dim=-1)


@dataclass
class PoseRefinementState:
    base_c2w: torch.Tensor
    refined_c2w: torch.Tensor
    delta: torch.Tensor


class PoseDelta(nn.Module):
    """Small SE(3) pose delta module.

    The delta is left-multiplied in camera-to-world space:
    ``c2w_refined = exp(delta) @ c2w_base``.
    """

    def __init__(self, base_c2w: torch.Tensor, init_delta: torch.Tensor | None = None) -> None:
        super().__init__()
        base = canonicalize_c2w(
            ensure_homogeneous(base_c2w.detach().clone().float())
        )
        self.register_buffer("base_c2w", base)
        if init_delta is None:
            init_delta = torch.zeros(6, dtype=base.dtype)
        self.delta = nn.Parameter(init_delta.detach().clone().to(dtype=base.dtype).view(6))

    def forward(self) -> torch.Tensor:
        # Compose the six-DoF residual in float64, then return the canonical
        # pose dtype expected by the renderer.  A float32 log/exp composition
        # can lose millimetres when a near-pi residual is rebased in a
        # large-coordinate scene, even though the final 4x4 pose is float32.
        # The operation is only on one 4x4 pose and has negligible cost beside
        # rasterization while retaining gradients to the float32 parameter.
        refined = (
            se3_exp(self.delta.to(dtype=torch.float64))
            @ self.base_c2w.to(dtype=torch.float64)
        )
        return refined.to(dtype=self.base_c2w.dtype)

    def rebase(
        self,
        base_c2w: torch.Tensor,
        *,
        preserve_delta: bool = True,
    ) -> None:
        """Move the canonical graph base without leaking pose residuals into it.

        PointMap-Sim3 owns the canonical pose while PFGS360 optimizes a local
        photometric SE(3) residual.  A graph correction therefore updates only
        ``base_c2w``; the learned residual remains attached to the frame.
        """

        base = canonicalize_c2w(
            ensure_homogeneous(base_c2w.detach().clone().to(self.base_c2w))
        )
        with torch.no_grad():
            self.base_c2w.copy_(base)
            if not preserve_delta:
                self.delta.zero_()

    def rebase_preserving_effective_pose(self, base_c2w: torch.Tensor) -> None:
        """Change the graph-owned base while preserving the rendered pose."""

        effective = self.forward().detach()
        base = canonicalize_c2w(
            ensure_homogeneous(base_c2w.detach().clone().to(self.base_c2w))
        )
        relative = (
            effective.to(dtype=torch.float64)
            @ torch.linalg.inv(base.to(dtype=torch.float64))
        )
        delta = se3_log(relative)
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Pose rebase produced a non-finite SE(3) residual")
        with torch.no_grad():
            self.base_c2w.copy_(base)
            self.delta.copy_(delta.to(self.delta))

    def canonical_pose(self) -> torch.Tensor:
        return self.base_c2w.detach().clone()

    def state(self) -> PoseRefinementState:
        refined = self.forward()
        return PoseRefinementState(
            base_c2w=self.base_c2w.detach().clone(),
            refined_c2w=refined.detach().clone(),
            delta=self.delta.detach().clone(),
        )


def make_pose_optimizer(
    pose_delta: PoseDelta,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW([pose_delta.delta], lr=float(lr), weight_decay=float(weight_decay))
