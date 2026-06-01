"""PyTorch spherical dense BA used by the graph frontend.

This is the correctness-first ERP version of DROID's dense BA contract.  It
builds a pose-depth normal equation from shared spherical projection residuals,
eliminates per-pixel inverse-depth variables with a Schur complement, then
applies SE(3) pose retractions and log inverse-depth updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .projective_ops import project_edges
from .spherical_ba import se3_exp, skew
from .spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing, pixel_grid, seam_aware_delta


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


def _left_pose_jacobians(
    poses_c2w: torch.Tensor,
    inverse_depth: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    *,
    height: int,
    width: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project graph edges and return ERP pixel Jacobians.

    Jacobians are for left SE(3) perturbations ``exp(dx) @ T_c2w`` with
    ``dx=[tx,ty,tz,rx,ry,rz]`` and for log inverse-depth increments.
    """
    B = int(poses_c2w.shape[0])
    E = int(ii.numel())
    pixels = pixel_grid(height, width, device=poses_c2w.device, dtype=poses_c2w.dtype)[
        ::stride, ::stride
    ]
    Hs, Ws = int(pixels.shape[0]), int(pixels.shape[1])
    p = pixels.view(1, 1, Hs, Ws, 2).expand(B, E, -1, -1, -1)
    bearing_i = erp_pixel_to_bearing(p.reshape(B * E, Hs * Ws, 2), height, width).view(
        B, E, Hs, Ws, 3
    )
    inv_src = inverse_depth[:, ii, 0, ::stride, ::stride].clamp_min(1e-6)
    xyz_i = bearing_i / inv_src.unsqueeze(-1)

    Ti = poses_c2w[:, ii]
    Tj = poses_c2w[:, jj]
    Ri = Ti[..., :3, :3]
    Rj = Tj[..., :3, :3]
    ti = Ti[..., :3, 3]
    tj = Tj[..., :3, 3]
    world = torch.einsum("beij,behwj->behwi", Ri, xyz_i) + ti.view(B, E, 1, 1, 3)
    rjt = Rj.transpose(-1, -2)
    xyz_j = torch.einsum("beij,behwj->behwi", rjt, world - tj.view(B, E, 1, 1, 3))
    bearing_j = F.normalize(xyz_j, dim=-1, eps=1e-12)
    coords = bearing_to_erp_pixel(bearing_j.reshape(B * E, Hs * Ws, 3), height, width).view(
        B, E, Hs, Ws, 2
    )

    x = xyz_j[..., 0]
    y = xyz_j[..., 1]
    z = xyz_j[..., 2]
    q = (x * x + z * z).clamp_min(1e-8)
    r2 = (q + y * y).clamp_min(1e-8)
    sqrt_q = torch.sqrt(q).clamp_min(1e-8)
    su = float(width) / (2.0 * math.pi)
    sv = float(height) / math.pi
    j_pix = xyz_j.new_zeros(B, E, Hs, Ws, 2, 3)
    j_pix[..., 0, 0] = su * z / q
    j_pix[..., 0, 2] = -su * x / q
    j_pix[..., 1, 0] = sv * (-x * y) / (r2 * sqrt_q)
    j_pix[..., 1, 1] = sv * sqrt_q / r2
    j_pix[..., 1, 2] = sv * (-z * y) / (r2 * sqrt_q)

    eye = torch.eye(3, device=poses_c2w.device, dtype=poses_c2w.dtype).view(1, 1, 1, 1, 3, 3)
    rjt_px = rjt.view(B, E, 1, 1, 3, 3)
    world_skew = skew(world)
    src_motion = torch.cat(
        [
            rjt_px.expand(-1, -1, Hs, Ws, -1, -1),
            -torch.matmul(rjt_px, world_skew),
        ],
        dim=-1,
    )
    tgt_motion = torch.cat(
        [
            -rjt_px.expand(-1, -1, Hs, Ws, -1, -1),
            torch.matmul(rjt_px, world_skew),
        ],
        dim=-1,
    )
    ji = torch.einsum("behwca,behwak->behwck", j_pix, src_motion)
    jjac = torch.einsum("behwca,behwak->behwck", j_pix, tgt_motion)

    d_world_dlog = -torch.einsum("beij,behwj->behwi", Ri, xyz_i)
    d_cam_dlog = torch.einsum("beij,behwj->behwi", rjt, d_world_dlog)
    jz = torch.einsum("behwca,behwa->behwc", j_pix, d_cam_dlog)
    _ = eye  # keeps the intended coordinate convention visible for readers.
    return coords, ji, jjac, jz


class SphericalDenseBA:
    """Low-resolution ERP dense bundle adjustment with DROID-style inputs."""

    def __init__(
        self,
        *,
        lm: float = 1e-4,
        max_pose_step: float = 0.05,
        max_depth_step: float = 0.10,
        min_inverse_depth: float = 1e-6,
    ) -> None:
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
        E = int(ii.numel())
        stride = max(1, int(sample_stride))
        target_s = target[:, :, ::stride, ::stride]
        weight_s = _ensure_weight2(weight)[:, :, :, ::stride, ::stride].clamp(1e-4, 1.0)
        eta_s = eta[:, :, :, ::stride, ::stride].clamp_min(0.0)
        Hs, Ws = int(target_s.shape[-3]), int(target_s.shape[-2])
        fixed = max(1, min(int(fixed_frames), N))
        pose_mask = torch.ones(B, N, 1, device=poses_c2w.device, dtype=poses_c2w.dtype)
        pose_mask[:, :fixed] = 0.0

        cur_poses = poses_c2w
        cur_inv = inverse_depth.clamp_min(self.min_inverse_depth)
        valid_mask = target_s.new_ones(B, E, Hs, Ws, dtype=torch.bool)
        pose_norm = target_s.new_tensor(0.0)
        depth_norm = target_s.new_tensor(0.0)
        cond_accum = target_s.new_tensor(0.0)
        cond_count = 0
        eye6 = torch.eye(6, device=target.device, dtype=target.dtype)

        for _ in range(max(0, int(iters))):
            coords, j_src, j_tgt, jz = _left_pose_jacobians(
                cur_poses,
                cur_inv,
                ii,
                jj,
                height=H,
                width=W,
                stride=stride,
            )
            residual = seam_aware_delta(coords, target_s, W)
            valid_mask = (
                torch.isfinite(residual).all(dim=-1)
                & torch.isfinite(weight_s).all(dim=2)
                & torch.isfinite(jz).all(dim=-1)
            )
            residual = torch.where(valid_mask.unsqueeze(-1), residual, torch.zeros_like(residual))
            w = weight_s.permute(0, 1, 3, 4, 2) * valid_mask.unsqueeze(-1).to(target.dtype)

            Hpp = residual.new_zeros(B, N, N, 6, 6)
            bp = residual.new_zeros(B, N, 6)
            C = residual.new_zeros(B, N, 1, Hs, Ws)
            bz = residual.new_zeros(B, N, 1, Hs, Ws)
            Epd = residual.new_zeros(B, N, N, 6, Hs, Ws)

            for edge_idx in range(E):
                src = int(ii[edge_idx])
                dst = int(jj[edge_idx])
                js = j_src[:, edge_idx]
                jt = j_tgt[:, edge_idx]
                j_depth = jz[:, edge_idx]
                res = residual[:, edge_idx]
                we = w[:, edge_idx]

                C[:, src, 0] = C[:, src, 0] + (we * j_depth * j_depth).sum(dim=-1)
                bz[:, src, 0] = bz[:, src, 0] + (we * j_depth * res).sum(dim=-1)
                pose_terms = ((src, js), (dst, jt))
                for a_frame, ja in pose_terms:
                    if a_frame >= fixed:
                        bp[:, a_frame] = bp[:, a_frame] + torch.einsum(
                            "bhwc,bhwc,bhwck->bk", we, res, ja
                        )
                        Epd[:, a_frame, src] = Epd[:, a_frame, src] + torch.einsum(
                            "bhwc,bhwc,bhwck->bkhw", we, j_depth, ja
                        )
                    for b_frame, jb in pose_terms:
                        if a_frame >= fixed and b_frame >= fixed:
                            Hpp[:, a_frame, b_frame] = Hpp[:, a_frame, b_frame] + torch.einsum(
                                "bhwc,bhwck,bhwcl->bkl", we, ja, jb
                            )

            eta_frame = eta_s.mean(dim=(-1, -2, -3)).view(B, N)
            for frame_idx in range(N):
                Hpp[:, frame_idx, frame_idx] = Hpp[:, frame_idx, frame_idx] + eye6 * (
                    self.lm + eta_frame[:, frame_idx].view(B, 1, 1)
                )
                if frame_idx < fixed:
                    Hpp[:, frame_idx] = 0.0
                    Hpp[:, :, frame_idx] = 0.0
                    Hpp[:, frame_idx, frame_idx] = eye6
                    bp[:, frame_idx] = 0.0
                    Epd[:, frame_idx] = 0.0

            C = C + self.lm + eta_s
            Cinv = C.clamp_min(1e-6).reciprocal()
            S = Hpp.clone()
            y = bp.clone()
            for depth_frame in range(N):
                c = Cinv[:, depth_frame, 0]
                bzd = bz[:, depth_frame, 0]
                e = Epd[:, :, depth_frame]
                S = S - torch.einsum("bprhw,bhw,bqshw->bpqrs", e, c, e)
                y = y - torch.einsum("bprhw,bhw,bhw->bpr", e, c, bzd)

            S_flat = S.permute(0, 1, 3, 2, 4).reshape(B, N * 6, N * 6)
            y_flat = y.reshape(B, N * 6, 1)
            try:
                dx = -torch.linalg.solve(S_flat, y_flat).reshape(B, N, 6)
            except RuntimeError:
                dx = (-torch.linalg.pinv(S_flat) @ y_flat).reshape(B, N, 6)
            dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            dx = dx.clamp(-self.max_pose_step, self.max_pose_step) * pose_mask
            cur_poses = se3_exp(dx) @ cur_poses
            pose_norm = dx.norm(dim=-1).mean()

            pose_term = torch.einsum("bpdrhw,bpr->bdhw", Epd, dx)
            dz = (-(bz[:, :, 0] + pose_term) * Cinv[:, :, 0]).clamp(
                -self.max_depth_step, self.max_depth_step
            )
            if stride != 1:
                dz = F.interpolate(
                    dz.reshape(B * N, 1, Hs, Ws),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=True,
                ).view(B, N, H, W)
            cur_inv = (cur_inv * torch.exp(torch.nan_to_num(dz.unsqueeze(2), nan=0.0))).clamp_min(
                self.min_inverse_depth
            )
            depth_norm = dz.abs().mean()

            eig = torch.linalg.eigvalsh(S_flat.detach().float()).clamp_min(1e-12)
            cond_accum = cond_accum + (eig[..., -1] / eig[..., 0]).mean().to(cond_accum)
            cond_count += 1

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
