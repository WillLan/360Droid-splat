"""Spherical tangent dense BA for PanoVGGT-M3 dense correspondences."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_ba import se3_exp

from .factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from .spherical_correspondence import spherical_tangent_residual


@dataclass
class SphericalTangentDenseBAOutput:
    poses_c2w: torch.Tensor
    inverse_depth: torch.Tensor
    log_inv_depth: torch.Tensor
    residual_angular: torch.Tensor
    valid_mask: torch.Tensor
    mean_angular_residual_deg: float
    median_angular_residual_deg: float
    pose_update_norm: dict[str, float]
    depth_update_norm: dict[str, float]
    failed: bool
    debug: dict[str, Any]


@dataclass
class _FlatFactors:
    src: torch.Tensor
    tgt: torch.Tensor
    src_y: torch.Tensor
    src_x: torch.Tensor
    unique_depth: torch.Tensor
    src_bearing: torch.Tensor
    tgt_bearing: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


class SphericalTangentDenseBA:
    """Correctness-first LM/GN solver using an S2 tangent residual."""

    def __init__(
        self,
        *,
        min_inverse_depth: float = 1.0e-6,
        factor_chunk_size: int = 2048,
        huber_delta_deg: float = 0.5,
    ) -> None:
        self.min_inverse_depth = float(min_inverse_depth)
        self.factor_chunk_size = max(1, int(factor_chunk_size))
        self.huber_delta = math.radians(float(huber_delta_deg))

    def __call__(
        self,
        poses_c2w: torch.Tensor,
        log_inv_depth: torch.Tensor,
        factors: DenseSphereFactorGraph | list[DenseSphereFactor] | tuple[DenseSphereFactor, ...],
        fixed_frames: int,
        iters: int = 6,
        damping: float = 1.0e-4,
        optimize_pose: bool = True,
        optimize_depth: bool = True,
        pose_prior_weight: float = 1.0e-3,
        depth_prior_weight: float = 1.0e-2,
        max_pose_update_deg: float = 5.0,
        max_logdepth_update: float = 0.35,
        line_search: bool = False,
    ) -> SphericalTangentDenseBAOutput:
        """Optimize local chunk poses and log inverse depth with tangent residuals."""

        poses0 = poses_c2w.detach().float()
        log0 = _normalize_log_inv_depth(log_inv_depth).detach().float()
        if poses0.ndim != 3 or poses0.shape[-2:] != (4, 4):
            raise ValueError(f"poses_c2w must have shape Nx4x4, got {tuple(poses0.shape)}.")
        if int(poses0.shape[0]) != int(log0.shape[0]):
            raise ValueError("poses_c2w and log_inv_depth must contain the same number of frames.")

        flat = _flatten_factors(factors, feature_hw=tuple(int(v) for v in log0.shape[-2:]), device=poses0.device, dtype=poses0.dtype)
        if flat.src.numel() == 0:
            return self._failed_output(poses0, log0, flat, reason="empty_or_invalid_factors")

        fixed = max(1, min(int(fixed_frames), int(poses0.shape[0])))
        free_frames = [idx for idx in range(int(poses0.shape[0])) if idx >= fixed]
        optimize_pose = bool(optimize_pose) and len(free_frames) > 0
        optimize_depth = bool(optimize_depth) and int(flat.unique_depth.max().item()) >= 0
        pose_dim = len(free_frames) * 6 if optimize_pose else 0
        depth_dim = int(flat.unique_depth.max().item()) + 1 if optimize_depth else 0
        total_dim = pose_dim + depth_dim
        if total_dim == 0:
            residual = self._residual_for_state(poses0, log0, flat)
            return self._success_output(poses0, log0, flat, residual, initial_residual=residual, debug={"iters": 0, "reason": "no_free_variables"})

        cur_poses = poses0.clone()
        cur_log = log0.clone()
        initial_residual = self._residual_for_state(cur_poses, cur_log, flat).detach()
        failed_reason = None
        pose_update = poses0.new_zeros(len(free_frames), 6)
        depth_update = poses0.new_zeros(depth_dim)

        for iter_idx in range(max(0, int(iters))):
            x0 = poses0.new_zeros(total_dim, requires_grad=True)

            def residual_fn(delta: torch.Tensor) -> torch.Tensor:
                return self._weighted_residual_vector(
                    delta,
                    cur_poses,
                    cur_log,
                    flat,
                    free_frames=free_frames,
                    pose_dim=pose_dim,
                    depth_dim=depth_dim,
                    pose_prior_weight=float(pose_prior_weight),
                    depth_prior_weight=float(depth_prior_weight),
                    optimize_pose=optimize_pose,
                    optimize_depth=optimize_depth,
                )

            try:
                residual0 = residual_fn(x0)
                if not torch.isfinite(residual0).all():
                    failed_reason = "non_finite_residual"
                    break
                jac = torch.autograd.functional.jacobian(residual_fn, x0, vectorize=False)
                jac = jac.reshape(residual0.numel(), total_dim)
            except RuntimeError as exc:
                failed_reason = f"autograd_failure:{exc}"
                break
            if not torch.isfinite(jac).all():
                failed_reason = "non_finite_jacobian"
                break

            hess = jac.T @ jac
            grad = jac.T @ residual0.detach().reshape(-1)
            step = _solve_lm_step(hess, grad, damping=float(damping))
            if step is None:
                failed_reason = "linear_solve_failure"
                break
            step = self._clamp_step(
                step,
                pose_dim=pose_dim,
                depth_dim=depth_dim,
                max_pose_update_deg=float(max_pose_update_deg),
                max_logdepth_update=float(max_logdepth_update),
            )
            if not torch.isfinite(step).all():
                failed_reason = "non_finite_update"
                break

            if optimize_depth and depth_dim:
                dz = step[pose_dim : pose_dim + depth_dim]
                total_dz = (depth_update + dz.detach()).clamp(
                    -float(max_logdepth_update),
                    float(max_logdepth_update),
                )
                step = step.clone()
                step[pose_dim : pose_dim + depth_dim] = total_dz - depth_update

            step_scale, next_poses, next_log = self._line_search_state(
                cur_poses,
                cur_log,
                step,
                flat,
                free_frames=free_frames,
                pose_dim=pose_dim,
                depth_dim=depth_dim,
                optimize_pose=optimize_pose,
                optimize_depth=optimize_depth,
                enabled=bool(line_search),
            )
            if step_scale <= 0.0:
                break
            step = step * float(step_scale)
            if optimize_pose and pose_dim:
                pose_delta = step[:pose_dim].view(len(free_frames), 6)
                pose_update = pose_update + pose_delta.detach()
            if optimize_depth and depth_dim:
                dz = step[pose_dim : pose_dim + depth_dim]
                depth_update = (depth_update + dz.detach()).clamp(
                    -float(max_logdepth_update),
                    float(max_logdepth_update),
                )
            cur_poses = next_poses
            cur_log = next_log

            if float(step.abs().max().detach().cpu()) < 1.0e-7:
                break
            _ = iter_idx

        if failed_reason is not None:
            return self._failed_output(poses0, log0, flat, reason=failed_reason, initial_residual=initial_residual)

        final_residual = self._residual_for_state(cur_poses, cur_log, flat).detach()
        if not torch.isfinite(cur_poses).all() or not torch.isfinite(cur_log).all() or not torch.isfinite(final_residual).all():
            return self._failed_output(poses0, log0, flat, reason="non_finite_output", initial_residual=initial_residual)

        return self._success_output(
            cur_poses.detach(),
            cur_log.detach(),
            flat,
            final_residual,
            initial_residual=initial_residual,
            pose_update=pose_update.detach(),
            depth_update=depth_update.detach(),
            debug={"iters": int(iters), "fixed_frames": fixed, "num_variables": total_dim},
        )

    def _line_search_state(
        self,
        poses: torch.Tensor,
        log_inv_depth: torch.Tensor,
        step: torch.Tensor,
        flat: _FlatFactors,
        *,
        free_frames: list[int],
        pose_dim: int,
        depth_dim: int,
        optimize_pose: bool,
        optimize_depth: bool,
        enabled: bool,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        scales = (1.0, 0.5, 0.25, 0.125, 0.0625) if enabled else (1.0,)
        base = self._residual_for_state(poses, log_inv_depth, flat).detach()
        base_mean = _mean_rad(base)
        best_scale = 0.0
        best_poses = poses
        best_log = log_inv_depth
        best_mean = base_mean
        for scale in scales:
            cur_step = step * float(scale)
            cur_poses = poses
            if optimize_pose and pose_dim:
                cur_poses = _apply_free_pose_delta(
                    poses,
                    cur_step[:pose_dim].view(len(free_frames), 6),
                    free_frames,
                )
            cur_log = log_inv_depth
            if optimize_depth and depth_dim:
                cur_log = _apply_unique_depth_delta(log_inv_depth, cur_step[pose_dim : pose_dim + depth_dim], flat)
            residual = self._residual_for_state(cur_poses, cur_log, flat).detach()
            if not torch.isfinite(residual).all():
                continue
            mean = _mean_rad(residual)
            if not enabled or mean <= base_mean + 1.0e-12:
                return float(scale), cur_poses.detach(), cur_log.detach()
            if mean < best_mean:
                best_scale = float(scale)
                best_poses = cur_poses.detach()
                best_log = cur_log.detach()
                best_mean = mean
        return best_scale, best_poses, best_log

    def _weighted_residual_vector(
        self,
        delta: torch.Tensor,
        poses: torch.Tensor,
        log_inv_depth: torch.Tensor,
        flat: _FlatFactors,
        *,
        free_frames: list[int],
        pose_dim: int,
        depth_dim: int,
        pose_prior_weight: float,
        depth_prior_weight: float,
        optimize_pose: bool,
        optimize_depth: bool,
    ) -> torch.Tensor:
        cur_poses = poses
        if optimize_pose and pose_dim:
            pose_delta = delta[:pose_dim].view(len(free_frames), 6)
            cur_poses = _apply_free_pose_delta(poses, pose_delta, free_frames)
        cur_log = log_inv_depth
        if optimize_depth and depth_dim:
            cur_log = _apply_unique_depth_delta(log_inv_depth, delta[pose_dim : pose_dim + depth_dim], flat)

        residual = self._residual_for_state(cur_poses, cur_log, flat)
        angular = residual.norm(dim=-1).detach().clamp_min(1.0e-12)
        huber = torch.where(angular <= self.huber_delta, torch.ones_like(angular), self.huber_delta / angular)
        scale = (flat.weight * huber).clamp_min(0.0).sqrt().unsqueeze(-1)
        parts = [(residual * scale).reshape(-1)]
        if optimize_pose and pose_dim and pose_prior_weight > 0.0:
            parts.append(delta[:pose_dim] * math.sqrt(float(pose_prior_weight)))
        if optimize_depth and depth_dim and depth_prior_weight > 0.0:
            parts.append(delta[pose_dim : pose_dim + depth_dim] * math.sqrt(float(depth_prior_weight)))
        return torch.cat(parts, dim=0)

    def _residual_for_state(self, poses: torch.Tensor, log_inv_depth: torch.Tensor, flat: _FlatFactors) -> torch.Tensor:
        parts = []
        for start in range(0, int(flat.src.numel()), self.factor_chunk_size):
            stop = min(start + self.factor_chunk_size, int(flat.src.numel()))
            idx = slice(start, stop)
            src = flat.src[idx]
            tgt = flat.tgt[idx]
            log_rho = log_inv_depth[src, 0, flat.src_y[idx], flat.src_x[idx]]
            src_xyz = flat.src_bearing[idx] / torch.exp(log_rho).clamp_min(self.min_inverse_depth).unsqueeze(-1)
            src_pose = poses[src]
            tgt_pose = poses[tgt]
            world = torch.einsum("pij,pj->pi", src_pose[:, :3, :3], src_xyz) + src_pose[:, :3, 3]
            tgt_cam = torch.einsum("pij,pj->pi", tgt_pose[:, :3, :3].transpose(-1, -2), world - tgt_pose[:, :3, 3])
            pred_bearing = F.normalize(tgt_cam, dim=-1, eps=1.0e-12)
            parts.append(spherical_tangent_residual(flat.tgt_bearing[idx], pred_bearing))
        if not parts:
            return poses.new_zeros(0, 2)
        return torch.cat(parts, dim=0)

    def _clamp_step(
        self,
        step: torch.Tensor,
        *,
        pose_dim: int,
        depth_dim: int,
        max_pose_update_deg: float,
        max_logdepth_update: float,
    ) -> torch.Tensor:
        out = step.clone()
        if pose_dim:
            pose = out[:pose_dim].view(-1, 6)
            max_rot = math.radians(float(max_pose_update_deg))
            rot_norm = pose[:, 3:].norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            rot_scale = (max_rot / rot_norm).clamp_max(1.0)
            pose[:, 3:] = pose[:, 3:] * rot_scale
            out[:pose_dim] = pose.reshape(-1)
        if depth_dim:
            out[pose_dim : pose_dim + depth_dim] = out[pose_dim : pose_dim + depth_dim].clamp(
                -float(max_logdepth_update),
                float(max_logdepth_update),
            )
        return out

    def _failed_output(
        self,
        poses: torch.Tensor,
        log_inv_depth: torch.Tensor,
        flat: _FlatFactors,
        *,
        reason: str,
        initial_residual: torch.Tensor | None = None,
    ) -> SphericalTangentDenseBAOutput:
        residual = initial_residual if initial_residual is not None else poses.new_zeros(flat.src.numel(), 2)
        return SphericalTangentDenseBAOutput(
            poses_c2w=poses.detach(),
            inverse_depth=torch.exp(log_inv_depth.detach()).clamp_min(self.min_inverse_depth),
            log_inv_depth=log_inv_depth.detach(),
            residual_angular=residual.detach().norm(dim=-1) if residual.numel() else poses.new_zeros(0),
            valid_mask=flat.valid_mask.detach(),
            mean_angular_residual_deg=_mean_deg(residual),
            median_angular_residual_deg=_median_deg(residual),
            pose_update_norm={"mean": 0.0, "max": 0.0},
            depth_update_norm={"mean": 0.0, "max": 0.0},
            failed=True,
            debug={"fallback_reason": reason, "num_factors": int(flat.src.numel())},
        )

    def _success_output(
        self,
        poses: torch.Tensor,
        log_inv_depth: torch.Tensor,
        flat: _FlatFactors,
        residual: torch.Tensor,
        *,
        initial_residual: torch.Tensor,
        pose_update: torch.Tensor | None = None,
        depth_update: torch.Tensor | None = None,
        debug: dict[str, Any] | None = None,
    ) -> SphericalTangentDenseBAOutput:
        pose_norm = pose_update.norm(dim=-1) if pose_update is not None and pose_update.numel() else poses.new_zeros(0)
        trans_norm = pose_update[:, :3].norm(dim=-1) if pose_update is not None and pose_update.numel() else poses.new_zeros(0)
        rot_norm = pose_update[:, 3:].norm(dim=-1) if pose_update is not None and pose_update.numel() else poses.new_zeros(0)
        depth_abs = depth_update.abs() if depth_update is not None and depth_update.numel() else poses.new_zeros(0)
        return SphericalTangentDenseBAOutput(
            poses_c2w=poses.detach(),
            inverse_depth=torch.exp(log_inv_depth.detach()).clamp_min(self.min_inverse_depth),
            log_inv_depth=log_inv_depth.detach(),
            residual_angular=residual.detach().norm(dim=-1),
            valid_mask=flat.valid_mask.detach(),
            mean_angular_residual_deg=_mean_deg(residual),
            median_angular_residual_deg=_median_deg(residual),
            pose_update_norm={
                "mean": float(pose_norm.mean().detach().cpu()) if pose_norm.numel() else 0.0,
                "max": float(pose_norm.max().detach().cpu()) if pose_norm.numel() else 0.0,
                "trans_mean": float(trans_norm.mean().detach().cpu()) if trans_norm.numel() else 0.0,
                "trans_max": float(trans_norm.max().detach().cpu()) if trans_norm.numel() else 0.0,
                "rot_mean_deg": float(torch.rad2deg(rot_norm).mean().detach().cpu()) if rot_norm.numel() else 0.0,
                "rot_max_deg": float(torch.rad2deg(rot_norm).max().detach().cpu()) if rot_norm.numel() else 0.0,
            },
            depth_update_norm={
                "mean": float(depth_abs.mean().detach().cpu()) if depth_abs.numel() else 0.0,
                "max": float(depth_abs.max().detach().cpu()) if depth_abs.numel() else 0.0,
            },
            failed=False,
            debug={
                **(debug or {}),
                "initial_mean_angular_residual_deg": _mean_deg(initial_residual),
                "num_factors": int(flat.src.numel()),
            },
        )


def _normalize_log_inv_depth(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"log_inv_depth must have shape Nx1xHxW, got {tuple(value.shape)}.")
    return value


def _factor_list(factors: DenseSphereFactorGraph | list[DenseSphereFactor] | tuple[DenseSphereFactor, ...]) -> list[DenseSphereFactor]:
    if isinstance(factors, DenseSphereFactorGraph):
        return list(factors.factors)
    return list(factors)


def _flatten_factors(
    factors: DenseSphereFactorGraph | list[DenseSphereFactor] | tuple[DenseSphereFactor, ...],
    *,
    feature_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> _FlatFactors:
    src_parts = []
    tgt_parts = []
    uv_parts = []
    src_bearing_parts = []
    tgt_bearing_parts = []
    weight_parts = []
    valid_parts = []
    for factor in _factor_list(factors):
        valid = factor.valid_mask.to(device=device).bool()
        factor_weight = factor.weight.to(device=device)
        valid = valid & torch.isfinite(factor_weight) & (factor_weight > 0.0)
        valid_count = int(valid.sum().detach().cpu())
        if valid.numel() == 0 or valid_count <= 0:
            continue
        src_parts.append(torch.full((valid_count,), int(factor.src), device=device, dtype=torch.long))
        tgt_parts.append(torch.full((valid_count,), int(factor.tgt), device=device, dtype=torch.long))
        uv_parts.append(factor.src_uv.to(device=device, dtype=dtype)[valid])
        src_bearing_parts.append(factor.src_bearing.to(device=device, dtype=dtype)[valid])
        tgt_bearing_parts.append(factor.tgt_bearing.to(device=device, dtype=dtype)[valid])
        weight_parts.append(factor_weight.to(dtype=dtype)[valid].clamp(0.0, 1.0))
        valid_parts.append(valid[valid])
    if not src_parts:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_float = torch.empty(0, device=device, dtype=dtype)
        empty_bool = torch.empty(0, device=device, dtype=torch.bool)
        return _FlatFactors(
            src=empty_long,
            tgt=empty_long,
            src_y=empty_long,
            src_x=empty_long,
            unique_depth=empty_long,
            src_bearing=empty_float.reshape(0, 3),
            tgt_bearing=empty_float.reshape(0, 3),
            weight=empty_float,
            valid_mask=empty_bool,
        )
    src = torch.cat(src_parts, dim=0)
    tgt = torch.cat(tgt_parts, dim=0)
    uv = torch.cat(uv_parts, dim=0)
    height, width = int(feature_hw[0]), int(feature_hw[1])
    src_x = uv[:, 0].floor().long().clamp(0, width - 1)
    src_y = uv[:, 1].floor().long().clamp(0, height - 1)
    key = src * (height * width) + src_y * width + src_x
    _, unique_depth = torch.unique(key, sorted=True, return_inverse=True)
    return _FlatFactors(
        src=src,
        tgt=tgt,
        src_y=src_y,
        src_x=src_x,
        unique_depth=unique_depth.long(),
        src_bearing=torch.cat(src_bearing_parts, dim=0),
        tgt_bearing=torch.cat(tgt_bearing_parts, dim=0),
        weight=torch.cat(weight_parts, dim=0),
        valid_mask=torch.ones_like(src, dtype=torch.bool),
    )


def _apply_free_pose_delta(poses: torch.Tensor, delta: torch.Tensor, free_frames: list[int]) -> torch.Tensor:
    if not free_frames:
        return poses
    out = poses.clone()
    updated = se3_exp(delta) @ poses[torch.tensor(free_frames, device=poses.device, dtype=torch.long)]
    out[torch.tensor(free_frames, device=poses.device, dtype=torch.long)] = updated
    return out


def _apply_unique_depth_delta(log_inv_depth: torch.Tensor, delta: torch.Tensor, flat: _FlatFactors) -> torch.Tensor:
    if delta.numel() == 0:
        return log_inv_depth
    out = log_inv_depth.clone()
    per_factor = delta[flat.unique_depth]
    out[flat.src, 0, flat.src_y, flat.src_x] = out[flat.src, 0, flat.src_y, flat.src_x] + per_factor
    return out


def _solve_lm_step(hess: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor | None:
    hess = torch.nan_to_num(hess.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    hess = 0.5 * (hess + hess.T)
    diag = torch.diagonal(hess).abs().clamp_min(1.0)
    eye = torch.eye(hess.shape[0], device=hess.device, dtype=hess.dtype)
    for power in range(8):
        lam = float(damping) * (10.0**power)
        system = hess + eye * (lam * diag)
        try:
            step = torch.linalg.solve(system, -grad)
        except RuntimeError:
            continue
        if torch.isfinite(step).all():
            return step
    return None


def _mean_deg(residual: torch.Tensor) -> float:
    if residual.numel() == 0:
        return 0.0
    angular = residual.detach().norm(dim=-1)
    finite = angular[torch.isfinite(angular)]
    if finite.numel() == 0:
        return 0.0
    return float(torch.rad2deg(finite).mean().cpu())


def _mean_rad(residual: torch.Tensor) -> float:
    if residual.numel() == 0:
        return 0.0
    angular = residual.detach().norm(dim=-1)
    finite = angular[torch.isfinite(angular)]
    if finite.numel() == 0:
        return float("inf")
    return float(finite.mean().cpu())


def _median_deg(residual: torch.Tensor) -> float:
    if residual.numel() == 0:
        return 0.0
    angular = residual.detach().norm(dim=-1)
    finite = angular[torch.isfinite(angular)]
    if finite.numel() == 0:
        return 0.0
    return float(torch.rad2deg(finite).median().cpu())
