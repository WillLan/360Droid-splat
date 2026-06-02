"""
Differentiable ERP (Equirectangular Projection) 鈫?Cubemap conversion.

Face order: [F, R, B, L, U, D]
Coordinate convention: +Z forward, +X right, +Y down (OpenCV / 3DGS camera space)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cubemap rotation matrices (3脳3, float32 numpy)
# CUBEMAP_RS_NP[i] maps a direction from face-i local space to body/world space:
#   d_world = CUBEMAP_RS_NP[i] @ d_face_local
# Face local space: +Z = optical axis, +X = right, +Y = down
# ---------------------------------------------------------------------------
CUBEMAP_RS_NP = np.array(
    [
        # 0 鈥?Front  (look +Z) : identity
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # 1 鈥?Right  (look +X) : R_y(+90掳)
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        # 2 鈥?Back   (look -Z) : R_y(180掳)
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        # 3 鈥?Left   (look -X) : R_y(-90掳)
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        # 4 鈥?Up     (look -Y) : R_x(+90掳)
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        # 5 鈥?Down   (look +Y) : R_x(-90掳)
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    ],
    dtype=np.float32,
)

FACE_LABELS = ["F", "R", "B", "L", "U", "D"]

# Horizontal-only faces (skip up/down for PnP by default)
HORIZONTAL_FACES = [0, 1, 2, 3]


def get_face_intrinsics(face_w: int):
    """Return (fx, fy, cx, cy) for a square cubemap face with 90掳 FoV."""
    f = face_w / 2.0
    c = face_w / 2.0 - 0.5
    return f, f, c, c


def get_face_fov() -> float:
    """Return field-of-view (radians) for one cubemap face (90掳)."""
    return math.pi / 2.0


class ERPToCubemapTorch(nn.Module):
    """
    Differentiable ERP 鈫?Cubemap converter using precomputed F.grid_sample grids.

    Usage::

        erp2cube = ERPToCubemapTorch(face_w=256).to("cuda")
        # erp: (C, H, W)  or  (B, C, H, W),  W = 2*H
        faces = erp2cube(erp)  # (6, C, fw, fw)  or  (B, 6, C, fw, fw)
    """

    def __init__(self, face_w: int = 256):
        super().__init__()
        self.face_w = face_w

        grids, cosmaps = [], []
        for R in CUBEMAP_RS_NP:
            grid, cosmap = self._make_face_grid(torch.tensor(R, dtype=torch.float32), face_w)
            grids.append(grid)
            cosmaps.append(cosmap)

        # (6, face_w, face_w, 2)  鈥?sampling coordinates for F.grid_sample
        self.register_buffer("grids", torch.stack(grids, dim=0))
        # (6, face_w, face_w)  鈥?cos(angle from optical axis), for R鈫抁 depth conversion
        self.register_buffer("cosmap", torch.stack(cosmaps, dim=0))

    # ------------------------------------------------------------------
    @staticmethod
    def _make_face_grid(R: torch.Tensor, face_w: int):
        """
        Compute the F.grid_sample sampling grid and cosmap for one face.

        Args:
            R:      3脳3 rotation (face local 鈫?body/world).
            face_w: face size in pixels.

        Returns:
            grid:   (face_w, face_w, 2) in [-1, 1], ready for F.grid_sample.
            cosmap: (face_w, face_w) cos of angle from face optical axis.
        """
        u = torch.linspace(-1.0, 1.0, face_w)
        v = torch.linspace(-1.0, 1.0, face_w)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (fw, fw)

        ones = torch.ones_like(uu)
        xyz_local = torch.stack([uu, vv, ones], dim=-1)  # (fw, fw, 3)

        # cosmap in face-local space: 1 / 鈥朳u, v, 1]鈥?        norm_local = xyz_local.norm(dim=-1).clamp(min=1e-8)
        cosmap = 1.0 / norm_local  # (fw, fw)

        # Rotate to body space
        xyz_body = (R @ xyz_local.reshape(-1, 3).T).T.reshape(face_w, face_w, 3)

        # Normalise to unit sphere
        norm = xyz_body.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_n = xyz_body / norm

        dx, dy, dz = xyz_n[..., 0], xyz_n[..., 1], xyz_n[..., 2]

        # Spherical 鈫?ERP pixel (grid_sample coords in [-1, 1])
        # longitude 蠁 鈭?[-蟺, 蟺]  鈫?grid_x 鈭?[-1, 1]
        # latitude  胃 鈭?[-蟺/2, 蟺/2] (positive = down in Y-down coords)
        #   top of ERP (v=-1) corresponds to dy = -1 (up)
        phi = torch.atan2(dx, dz)
        theta = torch.asin(dy.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        grid_x = phi / math.pi          # [-1, 1]
        grid_y = theta / (math.pi / 2)  # [-1, 1]  (-1 = top/up, +1 = bottom/down)

        grid = torch.stack([grid_x, grid_y], dim=-1)  # (fw, fw, 2)
        return grid, cosmap

    # ------------------------------------------------------------------
    def forward(self, erp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            erp: (C, H, W) or (B, C, H, W) ERP image.  W should equal 2*H.

        Returns:
            faces: (6, C, fw, fw) or (B, 6, C, fw, fw)
        """
        single = erp.dim() == 3
        if single:
            erp = erp.unsqueeze(0)

        B, C, H, W = erp.shape
        fw = self.face_w

        # Expand grids: (6, fw, fw, 2) 鈫?(B*6, fw, fw, 2)
        grids = (
            self.grids.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * 6, fw, fw, 2)
        )
        # Expand ERP: (B, C, H, W) 鈫?(B*6, C, H, W)
        erp_exp = erp.unsqueeze(1).expand(-1, 6, -1, -1, -1).reshape(B * 6, C, H, W)

        faces = F.grid_sample(
            erp_exp, grids, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B*6, C, fw, fw)

        faces = faces.reshape(B, 6, C, fw, fw)
        if single:
            faces = faces.squeeze(0)
        return faces
