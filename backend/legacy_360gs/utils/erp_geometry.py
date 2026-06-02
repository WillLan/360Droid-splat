"""
Shared ERP pixel (u, v) <-> unit bearing helpers.

Single source of truth: CUDA ``erp_math.h`` / ``erp_project``:

Camera: +X right, +Y down, +Z forward.

PFGS360/Nerfstudio-style camera axes are represented explicitly as:
+X right, +Y up, +Z backward.  The conversion is a 180-degree flip around X:
``[x, y, z] -> [x, -y, -z]``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


SLAM_TO_PFGS360_AXES = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def erp_uv_to_bearing_numpy(u, v, W: int, H: int):
    """
    Convert ERP pixel coordinates to unit bearings (NumPy).

    u, v: float arrays, broadcastable to the same shape (e.g. (H, W)).
    W, H: image width and height.

    Returns:
        dx, dy, dz: unit direction components, same broadcast shape as u/v.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    Wf = float(W)
    Hf = float(H)
    lam = 2.0 * math.pi * (u / Wf - 0.5)
    phi = math.pi * (v / Hf - 0.5)
    dx = np.cos(phi) * np.sin(lam)
    dy = np.sin(phi)
    dz = np.cos(phi) * np.cos(lam)
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    n = np.maximum(n, 1e-12)
    return dx / n, dy / n, dz / n


def erp_bearing_to_uv_numpy(bearing, W: int, H: int):
    """Project SLAM-convention unit bearings back to ERP pixel coordinates."""
    b = np.asarray(bearing, dtype=np.float64)
    x = b[..., 0]
    y = b[..., 1]
    z = b[..., 2]
    n = np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    x = x / n
    y = y / n
    z = z / n
    lam = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1.0, 1.0))
    u = float(W) * (lam / (2.0 * math.pi) + 0.5)
    v = float(H) * (phi / math.pi + 0.5)
    return u, v


def slam_bearing_to_pfgs360_numpy(bearing):
    """Convert +Y-down/+Z-forward SLAM bearings to PFGS360/OpenGL axes."""
    b = np.asarray(bearing, dtype=np.float64)
    return b @ SLAM_TO_PFGS360_AXES.T


def pfgs360_bearing_to_slam_numpy(bearing):
    """Convert PFGS360/OpenGL bearings to +Y-down/+Z-forward SLAM axes."""
    b = np.asarray(bearing, dtype=np.float64)
    return b @ SLAM_TO_PFGS360_AXES.T


def erp_dense_pixel_center_bearings(H: int, W: int):
    """
    Unit bearing at each pixel center: u = c + 0.5, v = r + 0.5.

    Returns:
        dx, dy, dz: each (H, W) float64
    """
    u = (np.arange(W, dtype=np.float64) + 0.5)[np.newaxis, :]  # (1, W)
    v = (np.arange(H, dtype=np.float64) + 0.5)[:, np.newaxis]  # (H, 1)
    return erp_uv_to_bearing_numpy(u, v, W, H)


def erp_uv_to_bearing_torch(uv: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """
    Convert ERP pixel coordinates to unit bearings (Torch).

    uv: (N, 2), columns [u, v]; subpixel values allowed (matches SphereGlue).

    Returns:
        (N, 3) unit bearings.
    """
    u = uv[:, 0]
    v = uv[:, 1]
    Wf = float(W)
    Hf = float(H)
    lam = 2.0 * math.pi * (u/ Wf - 0.5)
    phi = math.pi * (v / Hf - 0.5)
    x = torch.cos(phi) * torch.sin(lam)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(lam)
    b = torch.stack([x, y, z], dim=-1)
    b = b / torch.clamp(torch.linalg.norm(b, dim=-1, keepdim=True), min=1e-12)
    return b


def erp_bearing_to_uv_torch(bearing: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Project SLAM-convention unit bearings back to ERP pixel coordinates."""
    b = bearing / torch.clamp(torch.linalg.norm(bearing, dim=-1, keepdim=True), min=1e-12)
    lam = torch.atan2(b[..., 0], b[..., 2])
    phi = torch.asin(b[..., 1].clamp(-1.0, 1.0))
    u = float(W) * (lam / (2.0 * math.pi) + 0.5)
    v = float(H) * (phi / math.pi + 0.5)
    return torch.stack([u, v], dim=-1)


def slam_bearing_to_pfgs360_torch(bearing: torch.Tensor) -> torch.Tensor:
    """Convert +Y-down/+Z-forward SLAM bearings to PFGS360/OpenGL axes."""
    return bearing * bearing.new_tensor([1.0, -1.0, -1.0])


def pfgs360_bearing_to_slam_torch(bearing: torch.Tensor) -> torch.Tensor:
    """Convert PFGS360/OpenGL bearings to +Y-down/+Z-forward SLAM axes."""
    return bearing * bearing.new_tensor([1.0, -1.0, -1.0])
