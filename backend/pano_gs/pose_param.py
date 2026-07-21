"""Pose parameterization helpers for backend refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from frontend.pano_droid.spherical_ba import se3_exp
from geometry.sim3 import canonicalize_c2w


def ensure_homogeneous(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {tuple(T.shape)}")
    return T


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
        return se3_exp(self.delta) @ self.base_c2w

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
