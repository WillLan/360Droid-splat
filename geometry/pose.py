"""Named camera-pose convention helpers.

The project stores camera poses as camera-to-world (c2w) SE(3) matrices:
``X_world = R_c2w @ X_camera + t_c2w``. These helpers make convention
boundaries explicit and avoid ambiguous ad-hoc inversions in callers.
"""

from __future__ import annotations

import torch


def _require_homogeneous_pose(pose: torch.Tensor, *, name: str = "pose") -> None:
    if pose.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must end in 4x4, got {tuple(pose.shape)}")
    if not torch.is_floating_point(pose):
        raise TypeError(f"{name} must be floating point")


def invert_c2w(pose_c2w: torch.Tensor) -> torch.Tensor:
    """Return the world-to-camera inverse of a c2w SE(3) pose."""

    _require_homogeneous_pose(pose_c2w, name="pose_c2w")
    rotation = pose_c2w[..., :3, :3]
    translation = pose_c2w[..., :3, 3]
    rotation_t = rotation.transpose(-1, -2)
    out = torch.zeros_like(pose_c2w)
    out[..., :3, :3] = rotation_t
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_t, translation)
    out[..., 3, 3] = 1.0
    return out


def relative_c2w(source_c2w: torch.Tensor, target_c2w: torch.Tensor) -> torch.Tensor:
    """Map source-camera coordinates into target-camera coordinates."""

    _require_homogeneous_pose(source_c2w, name="source_c2w")
    _require_homogeneous_pose(target_c2w, name="target_c2w")
    return invert_c2w(target_c2w) @ source_c2w


def convert_pose_convention(pose: torch.Tensor, source_convention: str) -> torch.Tensor:
    """Convert an SE(3) pose from ``c2w`` or ``w2c`` into project c2w."""

    convention = str(source_convention).strip().lower()
    if convention not in {"c2w", "w2c"}:
        raise ValueError(f"pose_convention must be 'c2w' or 'w2c', got {source_convention!r}")
    _require_homogeneous_pose(pose)
    return pose if convention == "c2w" else invert_c2w(pose)
