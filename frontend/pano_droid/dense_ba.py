"""PyTorch spherical dense BA used by the graph frontend.

This is a correctness-first ERP version of DROID's dense BA contract.  It builds
per-frame damped normal equations for pose updates and diagonal per-pixel inverse
depth updates from the shared spherical projection residual.  The implementation
keeps the public shape and optimization semantics close to DROID while leaving a
clear replacement point for a future CUDA/Schur backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .projective_ops import project_edges
from .spherical_ba import se3_exp
from .spherical_camera import pixel_grid, seam_aware_delta


@dataclass
class SphericalDenseBAOutput:
    poses_c2w: torch.Tensor
    inverse_depth: torch.Tensor
    residual: torch.Tensor
    valid_mask: torch.Tensor
    pose_update_norm: torch.Tensor
    depth_update_norm: torch.Tensor
    normal_condition: torch.Tensor


def _ensure_weight2(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 5:
        raise ValueError(f"Expected weight as BxExCxHxW, got {tuple(weight.shape)}")
    if weight.shape[2] == 2:
        return weight
    if weight.shape[2] == 1:
        return weight.expand(-1, -1, 2, -1, -1)
    raise ValueError(f"Expected 1 or 2 weight channels, got {weight.shape[2]}")


class SphericalDenseBA:
    """Low-resolution ERP dense bundle adjustment with DROID-style inputs."""

    def __init__(
        self,
        *,
        pose_eps: float = 1e-3,
        depth_eps: float = 1e-3,
        lm: float = 1e-4,
        max_pose_step: float = 0.05,
        max_depth_step: float = 0.10,
        min_inverse_depth: float = 1e-6,
    ) -> None:
        self.pose_eps = float(pose_eps)
        self.depth_eps = float(depth_eps)
        self.lm = float(lm)
        self.max_pose_step = float(max_pose_step)
        self.max_depth_step = float(max_depth_step)
        self.min_inverse_depth = float(min_inverse_depth)

    def __call__(
        self,
        poses_c2w: torch.Tensor,
        inverse_depth: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        eta: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        fixed_frames: int = 2,
        iters: int = 2,
        sample_stride: int = 1,
    ) -> SphericalDenseBAOutput:
        B, N, _, H, W = inverse_depth.shape
        stride = max(1, int(sample_stride))
        pixels = None if stride == 1 else pixel_grid(H, W, device=target.device, dtype=target.dtype)[::stride, ::stride]
        target_s = target[:, :, ::stride, ::stride]
        weight_s = _ensure_weight2(weight)[:, :, :, ::stride, ::stride].clamp(1e-4, 1.0)
        eta_s = eta[:, :, :, ::stride, ::stride].clamp_min(0.0)
        fixed = max(1, min(int(fixed_frames), N))
        pose_mask = torch.ones(B, N, 1, device=poses_c2w.device, dtype=poses_c2w.dtype)
        pose_mask[:, :fixed] = 0.0

        cur_poses = poses_c2w
        cur_inv = inverse_depth.clamp_min(self.min_inverse_depth)
        valid_mask = target_s.new_ones(B, int(ii.numel()), target_s.shape[-3], target_s.shape[-2])
        pose_norm = target_s.new_tensor(0.0)
        depth_norm = target_s.new_tensor(0.0)
        cond_accum = target_s.new_tensor(0.0)
        cond_count = 0

        for _ in range(max(0, int(iters))):
            coords = project_edges(cur_poses, cur_inv, ii, jj, height=H, width=W, pixels=pixels)
            residual = seam_aware_delta(coords, target_s, W)
            valid_mask = torch.isfinite(residual).all(dim=-1) & torch.isfinite(weight_s).all(dim=2)
            residual = torch.where(valid_mask.unsqueeze(-1), residual, torch.zeros_like(residual))
            w_hw2 = weight_s.permute(0, 1, 3, 4, 2)

            H_pose = residual.new_zeros(B, N, 6, 6)
            b_pose = residual.new_zeros(B, N, 6)
            for frame_idx in range(N):
                if frame_idx < fixed:
                    continue
                for dof in range(6):
                    xi = residual.new_zeros(B, N, 6)
                    xi[:, frame_idx, dof] = self.pose_eps
                    poses_eps = se3_exp(xi) @ cur_poses
                    coords_eps = project_edges(poses_eps, cur_inv, ii, jj, height=H, width=W, pixels=pixels)
                    res_eps = seam_aware_delta(coords_eps, target_s, W)
                    jac = (res_eps - residual) / self.pose_eps
                    touched = (ii == frame_idx) | (jj == frame_idx)
                    if not bool(touched.any()):
                        continue
                    jac = jac[:, touched]
                    res_t = residual[:, touched]
                    w_t = w_hw2[:, touched]
                    valid_t = valid_mask[:, touched].unsqueeze(-1).to(residual.dtype)
                    jw = jac * w_t * valid_t
                    b_pose[:, frame_idx, dof] = (jw * res_t).sum(dim=(1, 2, 3, 4))
                    for dof2 in range(dof, 6):
                        if dof2 == dof:
                            jac2 = jac
                        else:
                            xi2 = residual.new_zeros(B, N, 6)
                            xi2[:, frame_idx, dof2] = self.pose_eps
                            poses_eps2 = se3_exp(xi2) @ cur_poses
                            coords_eps2 = project_edges(poses_eps2, cur_inv, ii, jj, height=H, width=W, pixels=pixels)
                            res_eps2 = seam_aware_delta(coords_eps2, target_s, W)
                            jac2 = ((res_eps2 - residual) / self.pose_eps)[:, touched]
                        h_val = (jw * jac2).sum(dim=(1, 2, 3, 4))
                        H_pose[:, frame_idx, dof, dof2] = h_val
                        H_pose[:, frame_idx, dof2, dof] = h_val

            eta_frame = eta_s.mean(dim=(-1, -2, -3)).view(B, N)
            eye6 = torch.eye(6, device=target.device, dtype=target.dtype).view(1, 1, 6, 6)
            H_pose = H_pose + eye6 * (self.lm + eta_frame.view(B, N, 1, 1))
            try:
                dx = -torch.linalg.solve(H_pose.reshape(B * N, 6, 6), b_pose.reshape(B * N, 6, 1))
                dx = dx.reshape(B, N, 6)
            except RuntimeError:
                dx = -torch.linalg.pinv(H_pose.reshape(B * N, 6, 6)) @ b_pose.reshape(B * N, 6, 1)
                dx = dx.reshape(B, N, 6)
            dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            dx = dx.clamp(-self.max_pose_step, self.max_pose_step) * pose_mask
            cur_poses = se3_exp(dx) @ cur_poses
            pose_norm = dx.norm(dim=-1).mean()

            eig = torch.linalg.eigvalsh(H_pose.detach().float()).clamp_min(1e-12)
            cond_accum = cond_accum + (eig[..., -1] / eig[..., 0]).mean().to(cond_accum)
            cond_count += 1

            coords = project_edges(cur_poses, cur_inv, ii, jj, height=H, width=W, pixels=pixels)
            residual = seam_aware_delta(coords, target_s, W)
            inv_eps = (cur_inv * torch.exp(torch.full_like(cur_inv, self.depth_eps))).clamp_min(self.min_inverse_depth)
            coords_z = project_edges(cur_poses, inv_eps, ii, jj, height=H, width=W, pixels=pixels)
            res_z = seam_aware_delta(coords_z, target_s, W)
            jz = (res_z - residual) / self.depth_eps
            valid_mask = torch.isfinite(residual).all(dim=-1) & torch.isfinite(jz).all(dim=-1)
            valid_f = valid_mask.unsqueeze(-1).to(residual.dtype)
            h_depth_edge = (w_hw2 * jz * jz * valid_f).sum(dim=-1).unsqueeze(2)
            b_depth_edge = (w_hw2 * jz * residual * valid_f).sum(dim=-1).unsqueeze(2)
            depth_h = residual.new_zeros(B, N, 1, h_depth_edge.shape[-2], h_depth_edge.shape[-1])
            depth_b = torch.zeros_like(depth_h)
            depth_h.index_add_(1, ii, h_depth_edge)
            depth_b.index_add_(1, ii, b_depth_edge)
            depth_h = depth_h + self.lm + eta_s
            dz = (-depth_b / depth_h.clamp_min(1e-6)).clamp(-self.max_depth_step, self.max_depth_step)
            if stride != 1:
                dz = torch.nn.functional.interpolate(
                    dz.reshape(B * N, 1, dz.shape[-2], dz.shape[-1]),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=True,
                ).view(B, N, 1, H, W)
            cur_inv = (cur_inv * torch.exp(torch.nan_to_num(dz, nan=0.0))).clamp_min(self.min_inverse_depth)
            depth_norm = dz.abs().mean()

        coords_full = project_edges(cur_poses, cur_inv, ii, jj, height=H, width=W)
        residual_full = seam_aware_delta(coords_full, target, W)
        valid_full = torch.isfinite(residual_full).all(dim=-1)
        cond = cond_accum / max(cond_count, 1)
        return SphericalDenseBAOutput(
            poses_c2w=cur_poses,
            inverse_depth=cur_inv,
            residual=residual_full,
            valid_mask=valid_full,
            pose_update_norm=pose_norm.detach(),
            depth_update_norm=depth_norm.detach(),
            normal_condition=cond.detach(),
        )
