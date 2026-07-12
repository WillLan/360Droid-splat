"""Stage 3 adapter matching and block-sparse spherical bundle adjustment.

The matcher deliberately mirrors Stage 1's full-resolution spherical CE
prediction rule.  The BA backend keeps one scalar inverse-depth variable per
source query and eliminates those variables with a Schur complement, avoiding
the dense global Jacobian used by the older correctness-first solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_ba import skew, so3_exp
from frontend.pano_vggt.spherical_correspondence import spherical_tangent_residual
from geometry.sim3 import sim3_log
from geometry.spherical_erp import (
    build_erp_ray_grid,
    erp_pixel_to_unit_ray,
    sample_erp_with_wrap,
)
from geometry.spherical_pseudo_correspondence import _sample_depth_filtered_fibonacci_uv


@dataclass
class Stage3MatchCache:
    """Pose-independent full-resolution adapter matches for one batch."""

    source_uv: torch.Tensor  # B,S,Q,2
    source_ray: torch.Tensor  # B,S,Q,3
    source_depth: torch.Tensor  # B,S,Q
    source_valid: torch.Tensor  # B,S,Q
    edges: torch.Tensor  # E,2
    target_uv: torch.Tensor  # B,E,Q,2
    target_ray: torch.Tensor  # B,E,Q,3
    top1_cosine: torch.Tensor  # B,E,Q
    top2_margin: torch.Tensor  # B,E,Q
    entropy: torch.Tensor  # B,E,Q
    valid_mask: torch.Tensor  # B,E,Q
    factor_weight: torch.Tensor | None = None  # B,E,Q, confidence used by BA
    mutual_mask: torch.Tensor | None = None  # B,E,Q
    target_valid: torch.Tensor | None = None  # B,E,Q
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.source_uv.shape[0])

    @property
    def num_views(self) -> int:
        return int(self.source_uv.shape[1])

    @property
    def queries_per_source(self) -> int:
        return int(self.source_uv.shape[2])

    @property
    def num_factors(self) -> int:
        return int(self.valid_mask.sum().detach().cpu())


def all_directed_pairs(num_views: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    pairs = [(src, tgt) for src in range(int(num_views)) for tgt in range(int(num_views)) if src != tgt]
    return torch.tensor(pairs, dtype=torch.long, device=device)


@torch.no_grad()
def build_stage3_match_cache(
    adapter_features: torch.Tensor,
    depth: torch.Tensor,
    *,
    num_queries: int = 2048,
    min_depth: float = 0.05,
    max_depth: float = 20.0,
    temperature: float = 0.07,
    query_chunk_size: int = 32,
    fibonacci_oversample_factor: int = 8,
    use_spherical_area_correction: bool = True,
    forward_backward: bool = False,
    fb_tolerance_deg: float = 1.0,
    min_factor_weight: float = 0.01,
    reliability_keep_fraction: float = 1.0,
    static_valid_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    query_uv: torch.Tensor | None = None,
) -> Stage3MatchCache:
    """Match every sampled source descriptor against every target ERP pixel."""

    if adapter_features.ndim != 5:
        raise ValueError("adapter_features must have shape BxSxCxHxW.")
    if depth.ndim != 5 or int(depth.shape[2]) != 1:
        raise ValueError("depth must have shape BxSx1xHxW.")
    batch, views, channels, height, width = (int(value) for value in adapter_features.shape)
    if tuple(depth.shape) != (batch, views, 1, height, width):
        raise ValueError("adapter_features and depth must share B/S/H/W dimensions.")
    if static_valid_mask is not None and tuple(static_valid_mask.shape) != tuple(depth.shape):
        raise ValueError("static_valid_mask must have shape BxSx1xHxW matching depth.")
    if views < 2 or channels <= 0:
        raise ValueError("Stage 3 matching requires at least two views and non-empty descriptors.")

    device = adapter_features.device
    features = F.normalize(adapter_features.float(), dim=2, eps=1.0e-8)
    flat_depth = depth.detach().float().reshape(batch * views, 1, height, width)
    count = max(1, min(int(num_queries), height * width))
    if query_uv is None:
        sampled_uv = _sample_depth_filtered_fibonacci_uv(
            flat_depth,
            height=height,
            width=width,
            count=count,
            min_depth=float(min_depth),
            max_depth=float(max_depth),
            oversample_factor=int(fibonacci_oversample_factor),
            dtype=torch.float32,
            generator=generator,
        ).reshape(batch, views, count, 2)
    else:
        sampled_uv = query_uv.to(device=device, dtype=torch.float32)
        if tuple(sampled_uv.shape) != (batch, views, count, 2):
            raise ValueError(f"query_uv must have shape {(batch, views, count, 2)}.")

    source_depth = sample_erp_with_wrap(depth.float(), sampled_uv)[..., 0]
    source_valid = (
        torch.isfinite(source_depth)
        & (source_depth >= float(min_depth))
        & (source_depth <= float(max_depth))
    )
    if static_valid_mask is not None:
        sampled_source_static = sample_erp_with_wrap(
            static_valid_mask.float(), sampled_uv
        )[..., 0]
        source_valid &= sampled_source_static > 0.5
    source_ray = erp_pixel_to_unit_ray(sampled_uv, height, width).float()
    source_feature = sample_erp_with_wrap(features, sampled_uv)
    source_feature = F.normalize(source_feature.float(), dim=-1, eps=1.0e-8)

    edges = all_directed_pairs(views, device=device)
    edge_count = int(edges.shape[0])
    target_uv = torch.empty(batch, edge_count, count, 2, device=device, dtype=torch.float32)
    target_ray = torch.empty(batch, edge_count, count, 3, device=device, dtype=torch.float32)
    top1_cosine = torch.empty(batch, edge_count, count, device=device, dtype=torch.float32)
    top2_margin = torch.empty_like(top1_cosine)
    entropy = torch.empty_like(top1_cosine)
    valid = torch.empty(batch, edge_count, count, device=device, dtype=torch.bool)
    mutual = torch.empty_like(valid)
    target_depth_valid = torch.empty_like(valid)
    factor_weight = torch.empty(batch, edge_count, count, device=device, dtype=torch.float32)

    rows = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    cos_latitude = torch.cos(math.pi * (rows / float(height) - 0.5)).clamp_min(1.0e-8)
    log_area = cos_latitude.log().view(height, 1).expand(height, width).reshape(1, -1)
    if not bool(use_spherical_area_correction):
        log_area.zero_()
    uv_grid = torch.stack(
        torch.meshgrid(
            torch.arange(width, device=device, dtype=torch.float32) + 0.5,
            torch.arange(height, device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        ),
        dim=-1,
    ).permute(1, 0, 2).reshape(-1, 2)
    ray_grid = build_erp_ray_grid(height, width, device=device, dtype=torch.float32).reshape(-1, 3)
    tau = max(float(temperature), 1.0e-8)
    chunk_size = max(1, int(query_chunk_size))
    keep_fraction = float(reliability_keep_fraction)
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("reliability_keep_fraction must be in (0, 1].")

    for batch_idx in range(batch):
        for edge_idx, pair in enumerate(edges.tolist()):
            src, tgt = int(pair[0]), int(pair[1])
            target = features[batch_idx, tgt].flatten(1)
            source = source_feature[batch_idx, src]
            for start in range(0, count, chunk_size):
                stop = min(count, start + chunk_size)
                cosine = source[start:stop] @ target
                logits = cosine / tau + log_area
                probability = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(logits, k=min(2, int(logits.shape[-1])), dim=-1)
                best = indices[:, 0]
                target_uv[batch_idx, edge_idx, start:stop] = uv_grid[best]
                target_ray[batch_idx, edge_idx, start:stop] = ray_grid[best]
                top1_cosine[batch_idx, edge_idx, start:stop] = cosine.gather(1, best[:, None])[:, 0]
                if int(values.shape[1]) > 1:
                    top2_margin[batch_idx, edge_idx, start:stop] = values[:, 0] - values[:, 1]
                else:
                    top2_margin[batch_idx, edge_idx, start:stop] = values[:, 0]
                entropy[batch_idx, edge_idx, start:stop] = -(
                    probability * probability.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                if bool(forward_backward):
                    matched = target[:, best].transpose(0, 1)
                    reverse_logits = matched @ features[batch_idx, src].flatten(1)
                    reverse_logits = reverse_logits / tau + log_area
                    reverse_best = reverse_logits.argmax(dim=-1)
                    reverse_ray = ray_grid[reverse_best]
                    reverse_angle = torch.atan2(
                        torch.cross(
                            reverse_ray,
                            source_ray[batch_idx, src, start:stop],
                            dim=-1,
                        ).norm(dim=-1),
                        (reverse_ray * source_ray[batch_idx, src, start:stop]).sum(dim=-1).clamp(-1.0, 1.0),
                    )
                    mutual[batch_idx, edge_idx, start:stop] = reverse_angle <= math.radians(float(fb_tolerance_deg))
                else:
                    mutual[batch_idx, edge_idx, start:stop] = True
            sampled_target_depth = sample_erp_with_wrap(
                depth[batch_idx : batch_idx + 1, tgt],
                target_uv[batch_idx, edge_idx].unsqueeze(0),
            )[0, ..., 0]
            target_depth_valid[batch_idx, edge_idx] = (
                torch.isfinite(sampled_target_depth)
                & (sampled_target_depth >= float(min_depth))
                & (sampled_target_depth <= float(max_depth))
            )
            if static_valid_mask is not None:
                sampled_target_static = sample_erp_with_wrap(
                    static_valid_mask[batch_idx : batch_idx + 1, tgt].float(),
                    target_uv[batch_idx, edge_idx].unsqueeze(0),
                )[0, ..., 0]
                target_depth_valid[batch_idx, edge_idx] &= sampled_target_static > 0.5
            normalized_entropy = entropy[batch_idx, edge_idx] / max(math.log(max(2, height * width)), 1.0e-8)
            cosine_score = ((top1_cosine[batch_idx, edge_idx] + 1.0) * 0.5).clamp(0.0, 1.0)
            margin_score = torch.sigmoid(top2_margin[batch_idx, edge_idx])
            entropy_score = (1.0 - normalized_entropy).clamp(0.0, 1.0)
            factor_weight[batch_idx, edge_idx] = cosine_score * margin_score * entropy_score
            valid[batch_idx, edge_idx] = (
                source_valid[batch_idx, src]
                & target_depth_valid[batch_idx, edge_idx]
                & mutual[batch_idx, edge_idx]
                & (factor_weight[batch_idx, edge_idx] >= float(min_factor_weight))
            )
            if keep_fraction < 1.0:
                candidates = torch.nonzero(valid[batch_idx, edge_idx], as_tuple=False).flatten()
                if int(candidates.numel()) > 0:
                    keep_count = max(1, int(math.ceil(keep_fraction * int(candidates.numel()))))
                    ranked = torch.topk(
                        factor_weight[batch_idx, edge_idx, candidates],
                        k=keep_count,
                        largest=True,
                        sorted=False,
                    ).indices
                    reliability_mask = torch.zeros_like(valid[batch_idx, edge_idx])
                    reliability_mask[candidates[ranked]] = True
                    valid[batch_idx, edge_idx] &= reliability_mask

    return Stage3MatchCache(
        source_uv=sampled_uv,
        source_ray=source_ray,
        source_depth=source_depth,
        source_valid=source_valid,
        edges=edges,
        target_uv=target_uv,
        target_ray=target_ray,
        top1_cosine=top1_cosine,
        top2_margin=top2_margin,
        entropy=entropy,
        valid_mask=valid,
        factor_weight=factor_weight,
        mutual_mask=mutual,
        target_valid=target_depth_valid,
        metadata={
            "temperature": tau,
            "query_chunk_size": chunk_size,
            "min_depth": float(min_depth),
            "max_depth": float(max_depth),
            "use_spherical_area_correction": bool(use_spherical_area_correction),
            "forward_backward": bool(forward_backward),
            "fb_tolerance_deg": float(fb_tolerance_deg),
            "min_factor_weight": float(min_factor_weight),
            "reliability_keep_fraction": keep_fraction,
            "static_validity_filter": static_valid_mask is not None,
        },
    )


@dataclass
class Stage3BAOutput:
    poses_c2w: torch.Tensor
    dense_depth: torch.Tensor
    sparse_depth: torch.Tensor
    depth_scale: torch.Tensor
    depth_shift: torch.Tensor
    accepted: torch.Tensor
    initial_median_residual_deg: torch.Tensor
    final_median_residual_deg: torch.Tensor
    diagnostics: list[dict[str, Any]]


def _factor_residual_from_local_delta(
    delta: torch.Tensor,
    src_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    src_ray: torch.Tensor,
    tgt_ray: torch.Tensor,
    log_inv_depth: torch.Tensor,
) -> torch.Tensor:
    src_updated = _se3_exp_out_of_place(delta[:6]) @ src_pose
    tgt_updated = _se3_exp_out_of_place(delta[6:12]) @ tgt_pose
    depth = torch.exp(-(log_inv_depth + delta[12]))
    point_source = depth * src_ray
    point_world = src_updated[:3, :3] @ point_source + src_updated[:3, 3]
    point_target = tgt_updated[:3, :3].transpose(0, 1) @ (point_world - tgt_updated[:3, 3])
    predicted = F.normalize(point_target, dim=0, eps=1.0e-8)
    return spherical_tangent_residual(tgt_ray, predicted)


def _se3_exp_out_of_place(xi: torch.Tensor) -> torch.Tensor:
    """A vmap-compatible form of the project's strict SE(3) exponential."""

    rho, omega = xi[..., :3], xi[..., 3:]
    rotation = so3_exp(omega)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta.square()
    matrix = skew(omega)
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    small = theta < 1.0e-4
    a = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-8),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1.0e-8),
    )
    jacobian = eye + a[..., None] * matrix + b[..., None] * (matrix @ matrix)
    translation = torch.einsum("...ij,...j->...i", jacobian, rho)
    top = torch.cat([rotation, translation[..., None]], dim=-1)
    bottom = torch.cat([torch.zeros_like(xi[..., :3]), torch.ones_like(xi[..., :1])], dim=-1)
    return torch.cat([top, bottom[..., None, :]], dim=-2)


def _weighted_affine_fit(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    median_depth: float,
    iterations: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(source) & torch.isfinite(target) & torch.isfinite(weight) & (weight > 0)
    if int(valid.sum()) < 2:
        one = source.new_tensor(1.0)
        return one, source.new_tensor(0.0)
    x, y, w = source[valid].float(), target[valid].float(), weight[valid].float()
    design = torch.stack([x, torch.ones_like(x)], dim=-1)
    solution = torch.stack([x.new_tensor(1.0), x.new_tensor(0.0)])
    delta = max(0.01, 0.05 * float(median_depth))
    eye = torch.eye(2, device=x.device, dtype=x.dtype) * 1.0e-6
    for _ in range(max(1, int(iterations))):
        residual = design @ solution - y
        robust = torch.where(residual.abs() <= delta, torch.ones_like(residual), delta / residual.abs().clamp_min(1.0e-8))
        total_weight = (w * robust).clamp_min(0.0)
        normal = design.transpose(0, 1) @ (total_weight[:, None] * design) + eye
        rhs = design.transpose(0, 1) @ (total_weight * y)
        solution = torch.linalg.solve(normal, rhs)
    scale = solution[0].clamp(0.5, 2.0)
    shift_limit = 0.25 * float(median_depth)
    shift = solution[1].clamp(-shift_limit, shift_limit)
    return scale, shift


class BlockSparseSphericalBA:
    """LM/GN spherical BA with diagonal sparse-depth Schur elimination."""

    def __init__(
        self,
        *,
        iterations: int = 3,
        damping: float = 1.0e-4,
        huber_delta_deg: float = 0.5,
        pose_prior_weight: float = 1.0e-3,
        depth_prior_weight: float = 1.0e-2,
        max_pose_update_deg: float = 5.0,
        max_translation_update: float = 0.05,
        max_logdepth_update: float = 0.35,
        factor_chunk_size: int = 2048,
        min_factors: int = 256,
        residual_worse_tolerance: float = 1.05,
        min_affine_support: int = 64,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
        solver_mode: str = "backtracking_gn",
        dense_depth_mode: str = "affine",
        gauge_mode: str = "none",
        lm_max_trials: int = 4,
        lm_acceptance_eta: float = 1.0e-4,
        lm_damping_min: float = 1.0e-8,
        lm_damping_max: float = 1.0e8,
        lm_diagonal_floor: float = 1.0e-6,
        max_initial_residual_deg: float | None = None,
        min_parallax_deg: float = 0.0,
    ) -> None:
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.huber_delta = math.radians(float(huber_delta_deg))
        self.pose_prior_weight = float(pose_prior_weight)
        self.depth_prior_weight = float(depth_prior_weight)
        self.max_pose_update = math.radians(float(max_pose_update_deg))
        self.max_translation_update = float(max_translation_update)
        self.max_logdepth_update = float(max_logdepth_update)
        self.factor_chunk_size = max(1, int(factor_chunk_size))
        self.min_factors = max(1, int(min_factors))
        self.residual_worse_tolerance = float(residual_worse_tolerance)
        self.min_affine_support = max(2, int(min_affine_support))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.solver_mode = str(solver_mode).lower()
        if self.solver_mode not in {"backtracking_gn", "standard_lm"}:
            raise ValueError("solver_mode must be 'backtracking_gn' or 'standard_lm'.")
        self.dense_depth_mode = str(dense_depth_mode).lower()
        if self.dense_depth_mode not in {"affine", "none"}:
            raise ValueError("dense_depth_mode must be 'affine' or 'none'.")
        self.gauge_mode = str(gauge_mode).lower()
        if self.gauge_mode not in {"none", "initial_baseline"}:
            raise ValueError("gauge_mode must be 'none' or 'initial_baseline'.")
        self.lm_max_trials = max(1, int(lm_max_trials))
        self.lm_acceptance_eta = float(lm_acceptance_eta)
        self.lm_damping_min = max(0.0, float(lm_damping_min))
        self.lm_damping_max = max(self.lm_damping_min, float(lm_damping_max))
        self.lm_diagonal_floor = max(1.0e-12, float(lm_diagonal_floor))
        self.max_initial_residual = (
            None
            if max_initial_residual_deg is None
            else math.radians(max(0.0, float(max_initial_residual_deg)))
        )
        self.min_parallax = math.radians(max(0.0, float(min_parallax_deg)))

    def __call__(
        self,
        poses_c2w: torch.Tensor,
        dense_depth: torch.Tensor,
        cache: Stage3MatchCache,
    ) -> Stage3BAOutput:
        if poses_c2w.ndim != 4 or poses_c2w.shape[-2:] != (4, 4):
            raise ValueError("poses_c2w must have shape BxSx4x4.")
        if dense_depth.ndim != 5 or int(dense_depth.shape[2]) != 1:
            raise ValueError("dense_depth must have shape BxSx1xHxW.")
        batch, views = int(poses_c2w.shape[0]), int(poses_c2w.shape[1])
        if cache.batch_size != batch or cache.num_views != views:
            raise ValueError("BA inputs and match cache must share batch/view dimensions.")

        output_poses: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        shifts: list[torch.Tensor] = []
        sparse_values: list[torch.Tensor] = []
        accepted_values: list[torch.Tensor] = []
        initial_residuals: list[torch.Tensor] = []
        final_residuals: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        for batch_idx in range(batch):
            result = self._solve_one(
                poses_c2w[batch_idx].detach().float(),
                dense_depth[batch_idx].detach().float(),
                cache,
                batch_idx=batch_idx,
            )
            output_poses.append(result[0])
            sparse_values.append(result[1])
            scales.append(result[2])
            shifts.append(result[3])
            accepted_values.append(torch.tensor(result[4], device=dense_depth.device, dtype=torch.bool))
            initial_residuals.append(dense_depth.new_tensor(result[5]))
            final_residuals.append(dense_depth.new_tensor(result[6]))
            diagnostics.append(result[7])

        pose_tensor = torch.stack(output_poses, dim=0).to(device=poses_c2w.device, dtype=poses_c2w.dtype).detach()
        scale_tensor = torch.stack(scales, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype).detach()
        shift_tensor = torch.stack(shifts, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype).detach()
        valid_geometry = torch.isfinite(dense_depth) & (dense_depth >= self.min_depth) & (dense_depth <= self.max_depth)
        accepted_tensor = torch.stack(accepted_values)
        if self.dense_depth_mode == "affine":
            affine = scale_tensor[:, :, None, None, None] * dense_depth + shift_tensor[:, :, None, None, None]
            affine = affine.clamp(self.min_depth, self.max_depth)
            use_affine = accepted_tensor[:, None, None, None, None] & valid_geometry
            output_depth = torch.where(use_affine, affine, dense_depth)
        else:
            output_depth = dense_depth
        return Stage3BAOutput(
            poses_c2w=pose_tensor,
            dense_depth=output_depth,
            sparse_depth=torch.stack(sparse_values, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype),
            depth_scale=scale_tensor,
            depth_shift=shift_tensor,
            accepted=accepted_tensor,
            initial_median_residual_deg=torch.stack(initial_residuals),
            final_median_residual_deg=torch.stack(final_residuals),
            diagnostics=diagnostics,
        )

    def _solve_one(
        self,
        poses: torch.Tensor,
        depth_map: torch.Tensor,
        cache: Stage3MatchCache,
        *,
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, float, float, dict[str, Any]]:
        device = poses.device
        views, query_count = cache.num_views, cache.queries_per_source
        source_depth = sample_erp_with_wrap(depth_map, cache.source_uv[batch_idx])[..., 0].reshape(-1)
        source_depth = source_depth.clamp(self.min_depth, self.max_depth)
        log0 = -source_depth.log()
        edges = cache.edges.to(device=device)
        src = edges[:, 0, None].expand(-1, query_count).reshape(-1)
        tgt = edges[:, 1, None].expand(-1, query_count).reshape(-1)
        query = torch.arange(query_count, device=device).view(1, -1).expand(int(edges.shape[0]), -1).reshape(-1)
        depth_index = src * query_count + query
        src_ray = cache.source_ray[batch_idx].reshape(-1, 3)[depth_index].to(device=device, dtype=torch.float32)
        tgt_ray = cache.target_ray[batch_idx].reshape(-1, 3).to(device=device, dtype=torch.float32)
        valid = cache.valid_mask[batch_idx].reshape(-1).to(device=device)
        if cache.factor_weight is None:
            factor_weight = torch.ones_like(valid, dtype=torch.float32)
        else:
            factor_weight = cache.factor_weight[batch_idx].reshape(-1).to(device=device, dtype=torch.float32)
        valid &= torch.isfinite(src_ray).all(dim=-1) & torch.isfinite(tgt_ray).all(dim=-1)
        valid &= torch.isfinite(factor_weight) & (factor_weight > 0.0)
        keep = torch.nonzero(valid, as_tuple=False).flatten()
        if int(keep.numel()) < self.min_factors:
            identity_scale = torch.ones(views, device=device)
            zero_shift = torch.zeros(views, device=device)
            return poses, source_depth.reshape(views, query_count), identity_scale, zero_shift, False, float("inf"), float("inf"), {
                "reason": "insufficient_factors", "num_factors": int(keep.numel())
            }
        src, tgt, depth_index = src[keep], tgt[keep], depth_index[keep]
        src_ray, tgt_ray = src_ray[keep], tgt_ray[keep]
        factor_weight = factor_weight[keep].clamp(0.0, 1.0)
        initial_predicted = self._predicted_bearing(poses, log0, src, tgt, depth_index, src_ray)
        antipodal_ok = (initial_predicted * tgt_ray).sum(dim=-1) > -0.99
        if not antipodal_ok.all():
            src, tgt, depth_index = src[antipodal_ok], tgt[antipodal_ok], depth_index[antipodal_ok]
            src_ray, tgt_ray = src_ray[antipodal_ok], tgt_ray[antipodal_ok]
            factor_weight = factor_weight[antipodal_ok]
        if int(src.numel()) < self.min_factors:
            identity_scale = torch.ones(views, device=device)
            zero_shift = torch.zeros(views, device=device)
            return poses, source_depth.reshape(views, query_count), identity_scale, zero_shift, False, float("inf"), float("inf"), {
                "reason": "insufficient_non_antipodal_factors", "num_factors": int(src.numel())
            }

        initial_predicted = self._predicted_bearing(poses, log0, src, tgt, depth_index, src_ray)
        initial_geometry_residual = torch.atan2(
            torch.cross(initial_predicted, tgt_ray, dim=-1).norm(dim=-1),
            (initial_predicted * tgt_ray).sum(dim=-1).clamp(-1.0, 1.0),
        )
        source_world_ray = torch.einsum("nij,nj->ni", poses[src, :3, :3], src_ray)
        target_world_ray = torch.einsum("nij,nj->ni", poses[tgt, :3, :3], initial_predicted)
        initial_parallax = torch.atan2(
            torch.cross(source_world_ray, target_world_ray, dim=-1).norm(dim=-1),
            (source_world_ray * target_world_ray).sum(dim=-1).clamp(-1.0, 1.0),
        )
        geometry_keep = torch.ones_like(initial_geometry_residual, dtype=torch.bool)
        if self.max_initial_residual is not None:
            geometry_keep &= initial_geometry_residual <= self.max_initial_residual
        if self.min_parallax > 0.0:
            geometry_keep &= initial_parallax >= self.min_parallax
        initial_geometry_residual = initial_geometry_residual[geometry_keep]
        initial_parallax = initial_parallax[geometry_keep]
        src, tgt, depth_index = src[geometry_keep], tgt[geometry_keep], depth_index[geometry_keep]
        src_ray, tgt_ray = src_ray[geometry_keep], tgt_ray[geometry_keep]
        factor_weight = factor_weight[geometry_keep]
        if int(src.numel()) < self.min_factors:
            identity_scale = torch.ones(views, device=device)
            zero_shift = torch.zeros(views, device=device)
            return poses, source_depth.reshape(views, query_count), identity_scale, zero_shift, False, float("inf"), float("inf"), {
                "reason": "insufficient_geometry_gated_factors",
                "num_factors": int(src.numel()),
            }

        cur_pose = poses.clone()
        cur_log = log0.clone()
        initial_angle = self._angular_residual(cur_pose, cur_log, src, tgt, depth_index, src_ray, tgt_ray)
        initial_median = float(torch.rad2deg(initial_angle).median().detach().cpu())
        pose_dim = max(0, (views - 1) * 6)
        depth_dim = views * query_count
        failed_reason: str | None = None
        current_damping = min(
            self.lm_damping_max,
            max(self.damping, self.lm_damping_min),
        )
        accepted_steps = 0
        gain_ratios: list[float] = []
        gauge_scales: list[float] = []
        lm_nu = 2.0
        initial_objective = self._objective(
            cur_pose,
            cur_log,
            poses,
            log0,
            src,
            tgt,
            depth_index,
            src_ray,
            tgt_ray,
            factor_weight,
        )
        current_objective = initial_objective

        for _ in range(max(0, self.iterations)):
            hpp = torch.zeros(pose_dim, pose_dim, device=device, dtype=torch.float32)
            gp = torch.zeros(pose_dim, device=device, dtype=torch.float32)
            if pose_dim and self.pose_prior_weight > 0.0:
                prior_zero = torch.zeros(pose_dim, device=device, dtype=torch.float32)

                def pose_prior_from_delta(delta: torch.Tensor) -> torch.Tensor:
                    updated = [cur_pose[0]]
                    for frame_idx in range(1, views):
                        step = delta[(frame_idx - 1) * 6 : frame_idx * 6]
                        updated.append(_se3_exp_out_of_place(step) @ cur_pose[frame_idx])
                    return self._pose_prior_vector(torch.stack(updated, dim=0), poses)

                prior_residual = pose_prior_from_delta(prior_zero)
                prior_jacobian = torch.func.jacrev(pose_prior_from_delta)(prior_zero)
                hpp += self.pose_prior_weight * (prior_jacobian.T @ prior_jacobian)
                gp += self.pose_prior_weight * (prior_jacobian.T @ prior_residual)
            hpd = torch.zeros(depth_dim, pose_dim, device=device, dtype=torch.float32)
            hdd = torch.full((depth_dim,), self.depth_prior_weight, device=device, dtype=torch.float32)
            gd = self.depth_prior_weight * (cur_log - log0)

            for start in range(0, int(src.numel()), self.factor_chunk_size):
                stop = min(int(src.numel()), start + self.factor_chunk_size)
                chunk_src, chunk_tgt = src[start:stop], tgt[start:stop]
                chunk_depth = depth_index[start:stop]
                local_zero = torch.zeros(stop - start, 13, device=device, dtype=torch.float32)

                def single(delta, sp, tp, sr, tr, ld):
                    return _factor_residual_from_local_delta(delta, sp, tp, sr, tr, ld)

                jacobian_fn = torch.func.jacrev(single, argnums=0)
                residual = torch.func.vmap(single)(
                    local_zero,
                    cur_pose[chunk_src],
                    cur_pose[chunk_tgt],
                    src_ray[start:stop],
                    tgt_ray[start:stop],
                    cur_log[chunk_depth],
                )
                jacobian = torch.func.vmap(jacobian_fn)(
                    local_zero,
                    cur_pose[chunk_src],
                    cur_pose[chunk_tgt],
                    src_ray[start:stop],
                    tgt_ray[start:stop],
                    cur_log[chunk_depth],
                )
                norm = residual.norm(dim=-1)
                robust = torch.where(norm <= self.huber_delta, torch.ones_like(norm), self.huber_delta / norm.clamp_min(1.0e-8))
                sqrt_weight = (robust * factor_weight[start:stop]).clamp_min(0.0).sqrt()
                residual = residual * sqrt_weight[:, None]
                jacobian = jacobian * sqrt_weight[:, None, None]
                jp = torch.zeros(stop - start, 2, pose_dim, device=device, dtype=torch.float32)
                for frame in range(1, views):
                    frame_slice = slice((frame - 1) * 6, frame * 6)
                    src_mask = chunk_src == frame
                    tgt_mask = chunk_tgt == frame
                    if src_mask.any():
                        jp[src_mask, :, frame_slice] += jacobian[src_mask, :, :6]
                    if tgt_mask.any():
                        jp[tgt_mask, :, frame_slice] += jacobian[tgt_mask, :, 6:12]
                jd = jacobian[:, :, 12]
                hpp += torch.einsum("nrp,nrq->pq", jp, jp)
                gp += torch.einsum("nrp,nr->p", jp, residual)
                hpd.index_add_(0, chunk_depth, torch.einsum("nrp,nr->np", jp, jd))
                hdd.index_add_(0, chunk_depth, jd.square().sum(dim=-1))
                gd.index_add_(0, chunk_depth, (jd * residual).sum(dim=-1))

            if not all(torch.isfinite(value).all() for value in (hpp, gp, hpd, hdd, gd)):
                failed_reason = "non_finite_normal_equations"
                break
            pose_diagonal = (
                hpp.diagonal().clamp_min(self.lm_diagonal_floor)
                if pose_dim
                else torch.zeros(0, device=device, dtype=torch.float32)
            )
            depth_diagonal = hdd.clamp_min(self.lm_diagonal_floor)
            gauge_jacobian = self._baseline_gauge_jacobian(cur_pose, poses)

            def solve_step(damping: float, *, diagonal: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
                if pose_dim:
                    if diagonal:
                        damped_hpp = hpp + torch.diag(pose_diagonal * float(damping))
                    else:
                        damped_hpp = hpp + torch.eye(pose_dim, device=device) * float(damping)
                else:
                    damped_hpp = hpp
                damped_hdd = hdd + (
                    depth_diagonal * float(damping)
                    if diagonal
                    else torch.full_like(hdd, float(damping))
                )
                inv_hdd = damped_hdd.clamp_min(1.0e-12).reciprocal()
                schur = damped_hpp - hpd.transpose(0, 1) @ (inv_hdd[:, None] * hpd)
                rhs = gp - hpd.transpose(0, 1) @ (inv_hdd * gd)
                try:
                    if pose_dim and gauge_jacobian is not None:
                        kkt = torch.zeros(
                            pose_dim + 1,
                            pose_dim + 1,
                            device=device,
                            dtype=torch.float32,
                        )
                        kkt[:pose_dim, :pose_dim] = schur
                        kkt[:pose_dim, pose_dim] = gauge_jacobian
                        kkt[pose_dim, :pose_dim] = gauge_jacobian
                        kkt_rhs = torch.cat([-rhs, rhs.new_zeros(1)], dim=0)
                        pose_step = torch.linalg.solve(kkt, kkt_rhs)[:pose_dim]
                    else:
                        pose_step = -torch.linalg.solve(schur, rhs) if pose_dim else torch.zeros(0, device=device)
                except RuntimeError:
                    return None
                depth_step = -(gd + hpd @ pose_step) * inv_hdd
                if not torch.isfinite(depth_step).all() or not torch.isfinite(pose_step).all():
                    return None
                pose_step = pose_step.reshape(max(0, views - 1), 6)
                if pose_step.numel():
                    translation_norm = pose_step[:, :3].norm(dim=-1).clamp_min(1.0e-8)
                    rotation_norm = pose_step[:, 3:].norm(dim=-1).clamp_min(1.0e-8)
                    pose_step[:, :3] *= torch.minimum(
                        torch.ones_like(translation_norm),
                        translation_norm.new_tensor(self.max_translation_update) / translation_norm,
                    )[:, None]
                    pose_step[:, 3:] *= torch.minimum(
                        torch.ones_like(rotation_norm),
                        rotation_norm.new_tensor(self.max_pose_update) / rotation_norm,
                    )[:, None]
                return pose_step, depth_step

            def build_trial(
                pose_step: torch.Tensor,
                depth_step: torch.Tensor,
                *,
                step_scale: float,
            ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
                trial_pose = cur_pose.clone()
                for frame in range(1, views):
                    trial_pose[frame] = _se3_exp_out_of_place(
                        float(step_scale) * pose_step[frame - 1]
                    ) @ cur_pose[frame]
                trial_log = (
                    cur_log + float(step_scale) * depth_step
                ).clamp(log0 - self.max_logdepth_update, log0 + self.max_logdepth_update)
                trial_pose, trial_log, gauge_scale, gauge_ok = self._apply_scale_gauge(
                    trial_pose,
                    trial_log,
                    poses,
                )
                if not gauge_ok:
                    return None
                trial_predicted = self._predicted_bearing(
                    trial_pose, trial_log, src, tgt, depth_index, src_ray
                )
                if not bool(torch.isfinite(trial_predicted).all()):
                    return None
                if bool(((trial_predicted * tgt_ray).sum(dim=-1) <= -0.99).any()):
                    return None
                return trial_pose, trial_log, gauge_scale

            accepted_step = False
            if self.solver_mode == "standard_lm":
                for _trial in range(self.lm_max_trials):
                    solved = solve_step(current_damping, diagonal=True)
                    if solved is None:
                        current_damping = min(self.lm_damping_max, current_damping * lm_nu)
                        lm_nu *= 2.0
                        continue
                    delta_pose, delta_depth = solved
                    candidate = build_trial(delta_pose, delta_depth, step_scale=1.0)
                    if candidate is None:
                        current_damping = min(self.lm_damping_max, current_damping * lm_nu)
                        lm_nu *= 2.0
                        continue
                    trial_pose, trial_log, gauge_scale = candidate
                    trial_objective = self._objective(
                        trial_pose,
                        trial_log,
                        poses,
                        log0,
                        src,
                        tgt,
                        depth_index,
                        src_ray,
                        tgt_ray,
                        factor_weight,
                    )
                    relative_pose = trial_pose[1:] @ torch.linalg.inv(cur_pose[1:])
                    # Gauge projection changes world translations after the raw
                    # LM step.  The gain-ratio model must therefore use the
                    # effective left-multiplicative tangent step, not the
                    # pre-projection proposal returned by the linear solve.
                    flat_pose = sim3_log(relative_pose)[..., :6].reshape(-1)
                    effective_depth = trial_log - cur_log
                    quadratic = (
                        (flat_pose @ (hpp @ flat_pose) if pose_dim else hpp.new_tensor(0.0))
                        + 2.0 * (effective_depth @ (hpd @ flat_pose) if pose_dim else hpp.new_tensor(0.0))
                        + (hdd * effective_depth.square()).sum()
                    )
                    predicted_reduction = -(
                        (gp @ flat_pose if pose_dim else gp.new_tensor(0.0))
                        + gd @ effective_depth
                        + 0.5 * quadratic
                    )
                    actual_reduction = current_objective - trial_objective
                    rho = actual_reduction / predicted_reduction.clamp_min(1.0e-12)
                    if (
                        bool(torch.isfinite(trial_objective))
                        and float(predicted_reduction) > 0.0
                        and float(rho) > self.lm_acceptance_eta
                    ):
                        cur_pose = trial_pose
                        cur_log = trial_log
                        current_objective = trial_objective
                        gain_ratios.append(float(rho.detach().cpu()))
                        gauge_scales.append(float(gauge_scale))
                        damping_factor = max(1.0 / 3.0, 1.0 - (2.0 * float(rho) - 1.0) ** 3)
                        current_damping = max(
                            self.lm_damping_min,
                            min(self.lm_damping_max, current_damping * damping_factor),
                        )
                        lm_nu = 2.0
                        accepted_steps += 1
                        accepted_step = True
                        break
                    current_damping = min(self.lm_damping_max, current_damping * lm_nu)
                    lm_nu *= 2.0
            else:
                solved = solve_step(current_damping, diagonal=False)
                if solved is None:
                    failed_reason = "linear_solve_failure"
                    break
                delta_pose, delta_depth = solved
                for step_scale in (1.0, 0.5, 0.25, 0.125):
                    candidate = build_trial(delta_pose, delta_depth, step_scale=step_scale)
                    if candidate is None:
                        continue
                    trial_pose, trial_log, gauge_scale = candidate
                    trial_objective = self._objective(
                        trial_pose,
                        trial_log,
                        poses,
                        log0,
                        src,
                        tgt,
                        depth_index,
                        src_ray,
                        tgt_ray,
                        factor_weight,
                    )
                    if bool(torch.isfinite(trial_objective)) and float(trial_objective) < float(current_objective) - 1.0e-10:
                        cur_pose = trial_pose
                        cur_log = trial_log
                        current_objective = trial_objective
                        gauge_scales.append(float(gauge_scale))
                        current_damping = max(self.lm_damping_min, current_damping * 0.5)
                        accepted_steps += 1
                        accepted_step = True
                        break
                if not accepted_step:
                    current_damping = min(self.lm_damping_max, current_damping * 10.0)
            if not accepted_step and self.solver_mode == "standard_lm":
                break

        final_angle = self._angular_residual(cur_pose, cur_log, src, tgt, depth_index, src_ray, tgt_ray)
        final_median = float(torch.rad2deg(final_angle).median().detach().cpu())
        objective_improved = float(current_objective) <= float(initial_objective) + 1.0e-10
        residual_limit = max(1.0e-6, initial_median * self.residual_worse_tolerance)
        residual_acceptable = final_median <= residual_limit
        accepted = (
            failed_reason is None
            and math.isfinite(final_median)
            and objective_improved
            and residual_acceptable
            and (accepted_steps > 0 or self.iterations == 0 or initial_median <= 1.0e-6)
        )
        if not accepted:
            cur_pose = poses
            cur_log = log0
        optimized_depth = torch.exp(-cur_log).reshape(views, query_count)
        original_depth = source_depth.reshape(views, query_count)
        scales = torch.ones(views, device=device, dtype=torch.float32)
        shifts = torch.zeros(views, device=device, dtype=torch.float32)
        if accepted and self.dense_depth_mode == "affine":
            for frame in range(views):
                mask = cache.source_valid[batch_idx, frame].to(device=device)
                support = int(mask.sum())
                if support < self.min_affine_support:
                    continue
                median_depth = float(original_depth[frame, mask].median().detach().cpu())
                scales[frame], shifts[frame] = _weighted_affine_fit(
                    original_depth[frame, mask],
                    optimized_depth[frame, mask],
                    self._query_weights(
                        src,
                        depth_index,
                        factor_weight,
                        frame=frame,
                        query_count=query_count,
                    )[mask],
                    median_depth=median_depth,
                )
        return cur_pose, optimized_depth, scales, shifts, accepted, initial_median, final_median, {
            "reason": "accepted" if accepted else (
                failed_reason
                or ("residual_worse" if not residual_acceptable else "objective_not_improved")
            ),
            "num_factors": int(src.numel()),
            "initial_geometry_residual_p50_deg": float(
                torch.rad2deg(initial_geometry_residual).median().detach().cpu()
            ),
            "initial_geometry_residual_p90_deg": float(
                torch.rad2deg(initial_geometry_residual).quantile(0.9).detach().cpu()
            ),
            "initial_parallax_p10_deg": float(torch.rad2deg(initial_parallax).quantile(0.1).detach().cpu()),
            "initial_parallax_p50_deg": float(torch.rad2deg(initial_parallax).median().detach().cpu()),
            "num_depth_variables": depth_dim,
            "initial_median_residual_deg": initial_median,
            "final_median_residual_deg": final_median,
            "residual_acceptable": bool(residual_acceptable),
            "initial_objective": float(initial_objective.detach().cpu()),
            "final_objective": float(current_objective.detach().cpu()),
            "accepted_steps": int(accepted_steps),
            "final_damping": float(current_damping),
            "solver_mode": self.solver_mode,
            "dense_depth_mode": self.dense_depth_mode,
            "gauge_mode": self.gauge_mode,
            "gain_ratio_mean": float(sum(gain_ratios) / len(gain_ratios)) if gain_ratios else float("nan"),
            "gauge_scale_mean": float(sum(gauge_scales) / len(gauge_scales)) if gauge_scales else 1.0,
            "gauge_scale_min": min(gauge_scales) if gauge_scales else 1.0,
            "gauge_scale_max": max(gauge_scales) if gauge_scales else 1.0,
        }

    def _objective(
        self,
        poses: torch.Tensor,
        log_depth: torch.Tensor,
        initial_poses: torch.Tensor,
        initial_log_depth: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
        depth_index: torch.Tensor,
        src_ray: torch.Tensor,
        tgt_ray: torch.Tensor,
        factor_weight: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self._predicted_bearing(poses, log_depth, src, tgt, depth_index, src_ray)
        residual = spherical_tangent_residual(tgt_ray, predicted)
        norm = residual.norm(dim=-1)
        delta = norm.new_tensor(max(self.huber_delta, 1.0e-8))
        robust = torch.where(
            norm <= delta,
            0.5 * norm.square(),
            delta * (norm - 0.5 * delta),
        )
        objective = (factor_weight * robust).sum()
        if self.pose_prior_weight > 0.0:
            pose_prior = self._pose_prior_vector(poses, initial_poses)
            objective = objective + 0.5 * self.pose_prior_weight * pose_prior.square().sum()
        if self.depth_prior_weight > 0.0:
            depth_prior = log_depth - initial_log_depth
            objective = objective + 0.5 * self.depth_prior_weight * depth_prior.square().sum()
        return objective

    @staticmethod
    def _query_weights(
        src: torch.Tensor,
        depth_index: torch.Tensor,
        factor_weight: torch.Tensor,
        *,
        frame: int,
        query_count: int,
    ) -> torch.Tensor:
        selected = src == int(frame)
        output = factor_weight.new_zeros(query_count)
        count = factor_weight.new_zeros(query_count)
        if bool(selected.any()):
            local_index = depth_index[selected] - int(frame) * int(query_count)
            output.index_add_(0, local_index, factor_weight[selected])
            count.index_add_(0, local_index, torch.ones_like(factor_weight[selected]))
        return torch.where(count > 0, output / count.clamp_min(1.0), torch.ones_like(output)).clamp_min(1.0e-6)

    def _apply_scale_gauge(
        self,
        poses: torch.Tensor,
        log_depth: torch.Tensor,
        initial_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        """Fix the monocular scale gauge to the initial longest baseline.

        Scaling every camera center about the fixed first center and every
        sparse depth by the same positive scalar leaves all bearing factors
        unchanged.  This projection therefore selects a stable representative
        of the gauge orbit without injecting GT metric information.
        """

        if self.gauge_mode == "none" or int(poses.shape[0]) <= 1:
            return poses, log_depth, 1.0, True
        reference_center = initial_poses[0, :3, 3]
        initial_offsets = initial_poses[:, :3, 3] - reference_center
        reference_index = int(initial_offsets.norm(dim=-1).argmax())
        initial_length = initial_offsets[reference_index].norm()
        current_center = poses[0, :3, 3]
        current_length = (poses[reference_index, :3, 3] - current_center).norm()
        if (
            not bool(torch.isfinite(initial_length))
            or not bool(torch.isfinite(current_length))
            or float(initial_length) <= 1.0e-8
            or float(current_length) <= 1.0e-8
        ):
            return poses, log_depth, 1.0, False
        scale = initial_length / current_length
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            return poses, log_depth, 1.0, False
        normalized_pose = poses.clone()
        normalized_pose[:, :3, 3] = current_center + scale * (poses[:, :3, 3] - current_center)
        normalized_pose[0] = initial_poses[0]
        normalized_log_depth = log_depth - torch.log(scale)
        return normalized_pose, normalized_log_depth, float(scale.detach().cpu()), True

    def _baseline_gauge_jacobian(
        self,
        poses: torch.Tensor,
        initial_poses: torch.Tensor,
    ) -> torch.Tensor | None:
        """Linearized hard constraint for the selected baseline length.

        For a left SE(3) update ``[rho, omega]``, a camera center changes as
        ``dC = rho + omega x C``.  The selected baseline derivative is thus
        ``u^T dC = u^T rho + (C x u)^T omega``.  Adding this row to the Schur
        pose system removes the monocular scale null direction before solving;
        the exact gauge retraction then only corrects second-order drift.
        """

        views = int(poses.shape[0])
        if self.gauge_mode == "none" or views <= 1:
            return None
        reference_center = initial_poses[0, :3, 3]
        initial_offsets = initial_poses[:, :3, 3] - reference_center
        reference_index = int(initial_offsets.norm(dim=-1).argmax())
        if reference_index == 0:
            return None
        baseline = poses[reference_index, :3, 3] - poses[0, :3, 3]
        length = baseline.norm()
        if not bool(torch.isfinite(length)) or float(length) <= 1.0e-8:
            return None
        direction = baseline / length
        row = poses.new_zeros((views - 1) * 6, dtype=torch.float32)
        frame_slice = slice((reference_index - 1) * 6, reference_index * 6)
        center = poses[reference_index, :3, 3]
        row[frame_slice.start : frame_slice.start + 3] = direction
        row[frame_slice.start + 3 : frame_slice.stop] = torch.cross(center, direction, dim=0)
        if not bool(torch.isfinite(row).all()) or float(row.norm()) <= 1.0e-8:
            return None
        return row

    @staticmethod
    def _angular_residual(
        poses: torch.Tensor,
        log_depth: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
        depth_index: torch.Tensor,
        src_ray: torch.Tensor,
        tgt_ray: torch.Tensor,
    ) -> torch.Tensor:
        predicted = BlockSparseSphericalBA._predicted_bearing(poses, log_depth, src, tgt, depth_index, src_ray)
        dot = (tgt_ray * predicted).sum(dim=-1).clamp(-1.0, 1.0)
        cross = torch.cross(tgt_ray, predicted, dim=-1).norm(dim=-1)
        return torch.atan2(cross, dot)

    @staticmethod
    def _predicted_bearing(
        poses: torch.Tensor,
        log_depth: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
        depth_index: torch.Tensor,
        src_ray: torch.Tensor,
    ) -> torch.Tensor:
        depth = torch.exp(-log_depth[depth_index])
        point_source = depth[:, None] * src_ray
        point_world = torch.einsum("nij,nj->ni", poses[src, :3, :3], point_source) + poses[src, :3, 3]
        point_target = torch.einsum(
            "nij,nj->ni",
            poses[tgt, :3, :3].transpose(1, 2),
            point_world - poses[tgt, :3, 3],
        )
        return F.normalize(point_target, dim=-1, eps=1.0e-8)

    @staticmethod
    def _pose_prior_vector(current: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        if int(current.shape[0]) <= 1:
            return current.new_zeros(0)
        relative_rotation = current[1:, :3, :3] @ initial[1:, :3, :3].transpose(1, 2)
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7))
        vee = torch.stack(
            [
                relative_rotation[:, 2, 1] - relative_rotation[:, 1, 2],
                relative_rotation[:, 0, 2] - relative_rotation[:, 2, 0],
                relative_rotation[:, 1, 0] - relative_rotation[:, 0, 1],
            ],
            dim=-1,
        )
        scale = theta / (2.0 * torch.sin(theta)).clamp_min(1.0e-7)
        omega = scale[:, None] * vee
        small = theta < 1.0e-4
        omega = torch.where(small[:, None], 0.5 * vee, omega)
        translation = current[1:, :3, 3] - initial[1:, :3, 3]
        return torch.cat([translation, omega], dim=-1).reshape(-1)
