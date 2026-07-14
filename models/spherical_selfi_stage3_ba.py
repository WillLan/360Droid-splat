"""Stage 3 adapter matching and block-sparse spherical bundle adjustment.

The matcher deliberately mirrors Stage 1's full-resolution spherical CE
prediction rule.  The BA backend keeps one scalar inverse-depth variable per
source query and eliminates those variables with a Schur complement, avoiding
the dense global Jacobian used by the older correctness-first solver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_ba import skew, so3_exp
from frontend.pano_droid.spherical_camera import tangent_basis
from frontend.pano_vggt.spherical_correspondence import spherical_tangent_residual
from geometry.sim3 import sim3_log
from geometry.spherical_erp import (
    build_erp_ray_grid,
    erp_pixel_to_unit_ray,
    sample_erp_with_wrap,
    unit_ray_to_erp_pixel,
)
from geometry.spherical_pseudo_correspondence import sample_depth_filtered_fibonacci_uv


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

    def detached_clone(self) -> "Stage3MatchCache":
        """Materialize ordinary detached tensors outside inference mode."""

        def clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().clone()

        return Stage3MatchCache(
            source_uv=clone(self.source_uv),
            source_ray=clone(self.source_ray),
            source_depth=clone(self.source_depth),
            source_valid=clone(self.source_valid),
            edges=clone(self.edges),
            target_uv=clone(self.target_uv),
            target_ray=clone(self.target_ray),
            top1_cosine=clone(self.top1_cosine),
            top2_margin=clone(self.top2_margin),
            entropy=clone(self.entropy),
            valid_mask=clone(self.valid_mask),
            factor_weight=clone(self.factor_weight),
            mutual_mask=clone(self.mutual_mask),
            target_valid=clone(self.target_valid),
            metadata=copy.deepcopy(self.metadata),
        )


def all_directed_pairs(num_views: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    pairs = [(src, tgt) for src in range(int(num_views)) for tgt in range(int(num_views)) if src != tgt]
    return torch.tensor(pairs, dtype=torch.long, device=device)


def directed_pairs_for_topology(
    num_views: int,
    topology: str = "all_directed",
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the configured pose-independent directed matching graph."""

    views = int(num_views)
    value = str(topology).lower()
    if value == "all_directed":
        return all_directed_pairs(views, device=device)
    if value == "star_forward":
        if views < 2:
            raise ValueError("star_forward matching requires at least two views.")
        return torch.tensor(
            [(0, target) for target in range(1, views)],
            dtype=torch.long,
            device=device,
        )
    raise ValueError("edge_topology must be 'all_directed' or 'star_forward'.")


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
    distinctiveness_exclusion_deg: float = 0.0,
    subpixel_refine_radius: int = 0,
    edge_topology: str = "all_directed",
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
        sampled_uv = sample_depth_filtered_fibonacci_uv(
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

    edges = directed_pairs_for_topology(views, edge_topology, device=device)
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
    exclusion_deg = float(distinctiveness_exclusion_deg)
    if not 0.0 <= exclusion_deg < 180.0:
        raise ValueError("distinctiveness_exclusion_deg must be in [0, 180).")
    exclusion_cosine = math.cos(math.radians(exclusion_deg))
    refine_radius = int(subpixel_refine_radius)
    if refine_radius < 0 or refine_radius > 4:
        raise ValueError("subpixel_refine_radius must be in [0, 4].")
    if refine_radius > 0:
        offset_y, offset_x = torch.meshgrid(
            torch.arange(-refine_radius, refine_radius + 1, device=device),
            torch.arange(-refine_radius, refine_radius + 1, device=device),
            indexing="ij",
        )
        local_offset_y = offset_y.reshape(1, -1)
        local_offset_x = offset_x.reshape(1, -1)

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
                if refine_radius > 0:
                    best_row = torch.div(best, width, rounding_mode="floor")[:, None]
                    best_col = (best % width)[:, None]
                    raw_row = best_row + local_offset_y
                    local_valid = (raw_row >= 0) & (raw_row < height)
                    local_row = raw_row.clamp(0, height - 1)
                    local_col = (best_col + local_offset_x).remainder(width)
                    local_index = local_row * width + local_col
                    local_logits = logits.gather(1, local_index).masked_fill(
                        ~local_valid,
                        -torch.inf,
                    )
                    local_weight = torch.softmax(local_logits, dim=-1)
                    refined_ray = F.normalize(
                        (local_weight[..., None] * ray_grid[local_index]).sum(dim=1),
                        dim=-1,
                        eps=1.0e-8,
                    )
                    target_ray[batch_idx, edge_idx, start:stop] = refined_ray
                    target_uv[batch_idx, edge_idx, start:stop] = unit_ray_to_erp_pixel(
                        refined_ray,
                        height,
                        width,
                    )
                else:
                    target_uv[batch_idx, edge_idx, start:stop] = uv_grid[best]
                    target_ray[batch_idx, edge_idx, start:stop] = ray_grid[best]
                top1_cosine[batch_idx, edge_idx, start:stop] = cosine.gather(1, best[:, None])[:, 0]
                if int(values.shape[1]) > 1:
                    if exclusion_deg > 0.0:
                        best_ray = ray_grid[best]
                        inside_peak = (best_ray @ ray_grid.T) >= exclusion_cosine
                        independent_second = logits.masked_fill(inside_peak, -torch.inf).amax(dim=-1)
                        independent_second = torch.where(
                            torch.isfinite(independent_second),
                            independent_second,
                            values[:, 1],
                        )
                        top2_margin[batch_idx, edge_idx, start:stop] = (
                            values[:, 0] - independent_second
                        )
                    else:
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
            "distinctiveness_exclusion_deg": exclusion_deg,
            "subpixel_refine_radius": refine_radius,
            "edge_topology": str(edge_topology).lower(),
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
    depth_affine_accepted: torch.Tensor
    depth_affine_identity_error: torch.Tensor
    depth_affine_fit_error: torch.Tensor
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
    pose_update_side: str = "left",
) -> torch.Tensor:
    src_step = _se3_exp_out_of_place(delta[:6])
    tgt_step = _se3_exp_out_of_place(delta[6:12])
    if pose_update_side == "right":
        src_updated = src_pose @ src_step
        tgt_updated = tgt_pose @ tgt_step
    else:
        src_updated = src_step @ src_pose
        tgt_updated = tgt_step @ tgt_pose
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


def _so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) logarithm used by the right-local pose prior."""

    vee = 0.5 * torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = vee.norm(dim=-1)
    cos_theta = ((rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(
        sin_theta > 1.0e-7,
        theta / sin_theta.clamp_min(1.0e-7),
        1.0 + theta.square() / 6.0,
    )
    return scale[..., None] * vee


def _so3_right_jacobian_inverse(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Inverse SO(3) right Jacobian for ``Log(R Exp(delta))``."""

    theta = rotation_vector.norm(dim=-1, keepdim=True)
    theta2 = theta.square()
    matrix = skew(rotation_vector)
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype).expand(
        *rotation_vector.shape[:-1], 3, 3
    )
    small = theta < 1.0e-4
    coefficient = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0,
        1.0 / theta2.clamp_min(1.0e-12)
        - (1.0 + torch.cos(theta))
        / (2.0 * theta * torch.sin(theta)).clamp_min(1.0e-12),
    )
    return eye + 0.5 * matrix + coefficient[..., None] * (matrix @ matrix)


def _spherical_log_residual_and_jacobian(
    target_bearing: torch.Tensor,
    predicted_bearing: torch.Tensor,
    *,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Log_target(predicted)`` and its analytic derivative w.r.t. a unit prediction."""

    target = F.normalize(target_bearing, dim=-1, eps=1.0e-12)
    predicted = F.normalize(predicted_bearing, dim=-1, eps=1.0e-12)
    dot = (target * predicted).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)
    tangent = predicted - dot[..., None] * target
    scale = theta / sin_theta
    log_vector = scale[..., None] * tangent
    basis = tangent_basis(target, eps=1.0e-12)
    residual = torch.einsum("...ij,...i->...j", basis, log_vector)

    projector = (
        torch.eye(3, device=target.device, dtype=target.dtype).expand(*target.shape[:-1], 3, 3)
        - target[..., :, None] * target[..., None, :]
    )
    exact_derivative = (
        -sin_theta.reciprocal().square()
        + theta * dot / sin_theta.pow(3)
    )
    near_zero = theta < 1.0e-4
    derivative_scale = torch.where(
        near_zero,
        dot.new_full(dot.shape, -1.0 / 3.0),
        exact_derivative,
    )
    derivative_log = (
        scale[..., None, None] * projector
        + tangent[..., :, None]
        * (derivative_scale[..., None] * target)[..., None, :]
    )
    jacobian = torch.einsum("...ji,...jk->...ik", basis, derivative_log)
    return residual, jacobian


def _factor_residual_and_analytic_jacobian(
    src_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    src_ray: torch.Tensor,
    tgt_ray: torch.Tensor,
    log_inv_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic right-local SE(3)/log-depth Jacobian for one or more factors."""

    depth = torch.exp(-log_inv_depth)
    point_source = depth[..., None] * src_ray
    src_rotation = src_pose[..., :3, :3]
    tgt_rotation = tgt_pose[..., :3, :3]
    point_world = torch.einsum("...ij,...j->...i", src_rotation, point_source) + src_pose[..., :3, 3]
    point_target = torch.einsum(
        "...ij,...j->...i",
        tgt_rotation.transpose(-1, -2),
        point_world - tgt_pose[..., :3, 3],
    )
    point_norm = point_target.norm(dim=-1).clamp_min(1.0e-8)
    predicted = point_target / point_norm[..., None]
    residual, residual_wrt_bearing = _spherical_log_residual_and_jacobian(tgt_ray, predicted)

    eye = torch.eye(3, device=point_target.device, dtype=point_target.dtype).expand(
        *point_target.shape[:-1], 3, 3
    )
    bearing_wrt_point = (
        eye - predicted[..., :, None] * predicted[..., None, :]
    ) / point_norm[..., None, None]
    residual_wrt_point = residual_wrt_bearing @ bearing_wrt_point
    target_from_source = tgt_rotation.transpose(-1, -2) @ src_rotation
    source_point_jacobian = torch.cat(
        [eye, -skew(point_source)],
        dim=-1,
    )
    source_jacobian = target_from_source @ source_point_jacobian
    target_jacobian = torch.cat([-eye, skew(point_target)], dim=-1)
    depth_jacobian = -torch.einsum("...ij,...j->...i", target_from_source, point_source)
    geometry_jacobian = torch.cat(
        [source_jacobian, target_jacobian, depth_jacobian[..., None]],
        dim=-1,
    )
    return residual, residual_wrt_point @ geometry_jacobian


def _right_pose_prior_residual_and_jacobian(
    current: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose prior and block Jacobian for right-local c2w increments."""

    if int(current.shape[0]) <= 1:
        zero = current.new_zeros(0)
        return zero, current.new_zeros((0, 0))
    current_free = current[1:]
    initial_free = initial[1:]
    initial_rotation_t = initial_free[:, :3, :3].transpose(1, 2)
    relative_rotation = initial_rotation_t @ current_free[:, :3, :3]
    rotation_residual = _so3_log_vector(relative_rotation)
    translation_residual = torch.einsum(
        "nij,nj->ni",
        initial_rotation_t,
        current_free[:, :3, 3] - initial_free[:, :3, 3],
    )
    residual = torch.cat([translation_residual, rotation_residual], dim=-1).reshape(-1)
    blocks = current.new_zeros((int(current_free.shape[0]), 6, 6))
    blocks[:, :3, :3] = initial_rotation_t @ current_free[:, :3, :3]
    blocks[:, 3:, 3:] = _so3_right_jacobian_inverse(rotation_residual)
    count = int(current_free.shape[0])
    jacobian = current.new_zeros((count * 6, count * 6))
    for index in range(count):
        frame_slice = slice(index * 6, (index + 1) * 6)
        jacobian[frame_slice, frame_slice] = blocks[index]
    return residual, jacobian


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


def _weighted_huber_error(
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    absolute = residual.abs()
    delta_tensor = absolute.new_tensor(max(float(delta), 1.0e-8))
    robust = torch.where(
        absolute <= delta_tensor,
        0.5 * absolute.square(),
        delta_tensor * (absolute - 0.5 * delta_tensor),
    )
    valid_weight = weight.clamp_min(0.0)
    return (valid_weight * robust).sum() / valid_weight.sum().clamp_min(1.0e-8)


def _weighted_affine_fit_with_acceptance(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    median_depth: float,
    min_support: int,
    min_relative_improvement: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(source) & torch.isfinite(target) & torch.isfinite(weight) & (weight > 0)
    one = source.new_tensor(1.0)
    zero = source.new_tensor(0.0)
    inf = source.new_tensor(float("inf"))
    if int(valid.sum()) < int(min_support):
        return one, zero, False, inf, inf
    x, y, w = source[valid], target[valid], weight[valid]
    scale, shift = _weighted_affine_fit(
        x,
        y,
        w,
        median_depth=float(median_depth),
    )
    delta = max(0.01, 0.05 * float(median_depth))
    identity_error = _weighted_huber_error(x - y, w, delta=delta)
    fit_error = _weighted_huber_error(scale * x + shift - y, w, delta=delta)
    finite = bool(torch.isfinite(scale)) and bool(torch.isfinite(shift))
    finite = finite and bool(torch.isfinite(identity_error)) and bool(torch.isfinite(fit_error))
    required = identity_error * (1.0 - max(0.0, float(min_relative_improvement)))
    accepted = finite and float(identity_error) > 0.0 and float(fit_error) < float(required)
    if not accepted:
        return one, zero, False, identity_error, fit_error
    return scale, shift, True, identity_error, fit_error


def _solve_diagonal_depth_schur(
    hpp: torch.Tensor,
    gp: torch.Tensor,
    hpd: torch.Tensor,
    hdd: torch.Tensor,
    gd: torch.Tensor,
    *,
    constraint: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a pose/depth normal equation with diagonal depth blocks.

    The block convention is ``[[Hpp, Hpd.T], [Hpd, diag(Hdd)]]`` and the
    returned step solves ``H step = -gradient``.  An optional pose-space row
    imposes the hard linear constraint ``constraint @ pose_step = 0``.
    """

    pose_dim = int(gp.numel())
    depth_dim = int(gd.numel())
    if tuple(hpp.shape) != (pose_dim, pose_dim):
        raise ValueError("hpp shape must match the pose gradient")
    if tuple(hpd.shape) != (depth_dim, pose_dim):
        raise ValueError("hpd shape must be depth_dim x pose_dim")
    if tuple(hdd.shape) != (depth_dim,):
        raise ValueError("hdd shape must match the depth gradient")
    if not all(torch.isfinite(value).all() for value in (hpp, gp, hpd, hdd, gd)):
        raise ValueError("Schur system contains non-finite values")
    if bool((hdd <= 0.0).any()):
        raise ValueError("Schur depth diagonal must be strictly positive")

    inverse_hdd = hdd.reciprocal()
    schur = hpp - hpd.transpose(0, 1) @ (inverse_hdd[:, None] * hpd)
    rhs = gp - hpd.transpose(0, 1) @ (inverse_hdd * gd)
    if pose_dim:
        if constraint is not None and float(constraint.norm()) > 1.0e-10:
            if tuple(constraint.shape) != (pose_dim,):
                raise ValueError("constraint shape must match the pose dimension")
            kkt = torch.zeros(
                pose_dim + 1,
                pose_dim + 1,
                device=hpp.device,
                dtype=hpp.dtype,
            )
            kkt[:pose_dim, :pose_dim] = schur
            kkt[:pose_dim, pose_dim] = constraint
            kkt[pose_dim, :pose_dim] = constraint
            kkt_rhs = torch.cat([-rhs, rhs.new_zeros(1)], dim=0)
            pose_step = torch.linalg.solve(kkt, kkt_rhs)[:pose_dim]
        else:
            pose_step = -torch.linalg.solve(schur, rhs)
    else:
        pose_step = gp.new_zeros(0)
    depth_step = -(gd + hpd @ pose_step) * inverse_hdd
    return pose_step, depth_step


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
        pose_update_side: str = "left",
        pose_dof_mode: str = "se3",
        min_initial_median_residual_deg: float = 0.0,
        jacobian_mode: str = "autodiff_reference",
        validate_analytic_jacobian: bool = False,
        analytic_jacobian_atol: float = 1.0e-5,
        analytic_jacobian_rtol: float = 1.0e-4,
        gradient_tolerance: float = 1.0e-8,
        step_tolerance: float = 1.0e-8,
        relative_objective_tolerance: float = 1.0e-6,
        affine_min_relative_improvement: float = 1.0e-3,
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
        self.pose_update_side = str(pose_update_side).lower()
        if self.pose_update_side not in {"left", "right"}:
            raise ValueError("pose_update_side must be 'left' or 'right'.")
        self.pose_dof_mode = str(pose_dof_mode).lower()
        if self.pose_dof_mode not in {
            "se3",
            "rotation_only",
            "translation_only",
            "rotation_then_translation",
        }:
            raise ValueError(
                "pose_dof_mode must be 'se3', 'rotation_only', 'translation_only', "
                "or 'rotation_then_translation'."
            )
        if self.pose_dof_mode != "se3" and self.pose_update_side != "right":
            raise ValueError(
                f"pose_dof_mode='{self.pose_dof_mode}' requires pose_update_side='right' "
                "for an unambiguous camera-local reduced-DOF update."
            )
        if self.pose_dof_mode == "rotation_then_translation" and self.dense_depth_mode != "none":
            raise ValueError(
                "pose_dof_mode='rotation_then_translation' requires dense_depth_mode='none'."
            )
        self.min_initial_median_residual_deg = max(
            0.0, float(min_initial_median_residual_deg)
        )
        self.jacobian_mode = str(jacobian_mode).lower()
        if self.jacobian_mode not in {"analytic", "autodiff_reference"}:
            raise ValueError("jacobian_mode must be 'analytic' or 'autodiff_reference'.")
        if self.jacobian_mode == "analytic" and self.pose_update_side != "right":
            raise ValueError("jacobian_mode='analytic' requires pose_update_side='right'.")
        self.validate_analytic_jacobian = bool(validate_analytic_jacobian)
        self.analytic_jacobian_atol = max(0.0, float(analytic_jacobian_atol))
        self.analytic_jacobian_rtol = max(0.0, float(analytic_jacobian_rtol))
        self.gradient_tolerance = max(0.0, float(gradient_tolerance))
        self.step_tolerance = max(0.0, float(step_tolerance))
        self.relative_objective_tolerance = max(0.0, float(relative_objective_tolerance))
        self.affine_min_relative_improvement = max(
            0.0, float(affine_min_relative_improvement)
        )

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
        if self.pose_dof_mode == "rotation_then_translation":
            return self._run_rotation_then_translation(
                poses_c2w,
                dense_depth,
                cache,
            )

        output_poses: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        shifts: list[torch.Tensor] = []
        affine_accepted_values: list[torch.Tensor] = []
        affine_identity_errors: list[torch.Tensor] = []
        affine_fit_errors: list[torch.Tensor] = []
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
            affine_accepted_values.append(
                torch.as_tensor(
                    result[7].get("depth_affine_accepted", [False] * views),
                    device=dense_depth.device,
                    dtype=torch.bool,
                )
            )
            affine_identity_errors.append(
                torch.as_tensor(
                    result[7].get("depth_affine_identity_error", [float("inf")] * views),
                    device=dense_depth.device,
                    dtype=dense_depth.dtype,
                )
            )
            affine_fit_errors.append(
                torch.as_tensor(
                    result[7].get("depth_affine_fit_error", [float("inf")] * views),
                    device=dense_depth.device,
                    dtype=dense_depth.dtype,
                )
            )

        pose_tensor = torch.stack(output_poses, dim=0).to(device=poses_c2w.device, dtype=poses_c2w.dtype).detach()
        scale_tensor = torch.stack(scales, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype).detach()
        shift_tensor = torch.stack(shifts, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype).detach()
        valid_geometry = torch.isfinite(dense_depth) & (dense_depth >= self.min_depth) & (dense_depth <= self.max_depth)
        accepted_tensor = torch.stack(accepted_values)
        affine_accepted_tensor = torch.stack(affine_accepted_values)
        affine_identity_tensor = torch.stack(affine_identity_errors)
        affine_fit_tensor = torch.stack(affine_fit_errors)
        if self.dense_depth_mode == "affine":
            affine = scale_tensor[:, :, None, None, None] * dense_depth + shift_tensor[:, :, None, None, None]
            affine = affine.clamp(self.min_depth, self.max_depth)
            use_affine = affine_accepted_tensor[:, :, None, None, None] & valid_geometry
            output_depth = torch.where(use_affine, affine, dense_depth)
        else:
            output_depth = dense_depth
        return Stage3BAOutput(
            poses_c2w=pose_tensor,
            dense_depth=output_depth,
            sparse_depth=torch.stack(sparse_values, dim=0).to(device=dense_depth.device, dtype=dense_depth.dtype),
            depth_scale=scale_tensor,
            depth_shift=shift_tensor,
            depth_affine_accepted=affine_accepted_tensor,
            depth_affine_identity_error=affine_identity_tensor,
            depth_affine_fit_error=affine_fit_tensor,
            accepted=accepted_tensor,
            initial_median_residual_deg=torch.stack(initial_residuals),
            final_median_residual_deg=torch.stack(final_residuals),
            diagnostics=diagnostics,
        )

    def _run_rotation_then_translation(
        self,
        poses_c2w: torch.Tensor,
        dense_depth: torch.Tensor,
        cache: Stage3MatchCache,
    ) -> Stage3BAOutput:
        rotation_solver = copy.copy(self)
        rotation_solver.pose_dof_mode = "rotation_only"
        translation_solver = copy.copy(self)
        translation_solver.pose_dof_mode = "translation_only"
        rotation = rotation_solver(poses_c2w, dense_depth, cache)
        translation = translation_solver(rotation.poses_c2w, rotation.dense_depth, cache)
        diagnostics: list[dict[str, Any]] = []
        for index, (rotation_diag, translation_diag) in enumerate(
            zip(rotation.diagnostics, translation.diagnostics, strict=True)
        ):
            combined = dict(translation_diag)
            combined.update(
                {
                    "pose_dof_mode": "rotation_then_translation",
                    "rotation_stage": rotation_diag,
                    "translation_stage": translation_diag,
                    "rotation_accepted": bool(rotation.accepted[index]),
                    "translation_accepted": bool(translation.accepted[index]),
                    "initial_objective": rotation_diag.get("initial_objective", float("nan")),
                    "final_objective": translation_diag.get("final_objective", float("nan")),
                    "accepted_steps": float(rotation_diag.get("accepted_steps", 0.0))
                    + float(translation_diag.get("accepted_steps", 0.0)),
                    "rotation_final_residual_deg": float(
                        rotation.final_median_residual_deg[index].detach().cpu()
                    ),
                    "translation_final_residual_deg": float(
                        translation.final_median_residual_deg[index].detach().cpu()
                    ),
                }
            )
            diagnostics.append(combined)
        return Stage3BAOutput(
            poses_c2w=translation.poses_c2w,
            dense_depth=translation.dense_depth,
            sparse_depth=translation.sparse_depth,
            depth_scale=translation.depth_scale,
            depth_shift=translation.depth_shift,
            depth_affine_accepted=translation.depth_affine_accepted,
            depth_affine_identity_error=translation.depth_affine_identity_error,
            depth_affine_fit_error=translation.depth_affine_fit_error,
            accepted=rotation.accepted | translation.accepted,
            initial_median_residual_deg=rotation.initial_median_residual_deg,
            final_median_residual_deg=translation.final_median_residual_deg,
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
        if initial_median < self.min_initial_median_residual_deg:
            identity_scale = torch.ones(views, device=device)
            zero_shift = torch.zeros(views, device=device)
            return (
                poses,
                source_depth.reshape(views, query_count),
                identity_scale,
                zero_shift,
                False,
                initial_median,
                initial_median,
                {
                    "reason": "below_min_initial_median_residual",
                    "num_factors": int(src.numel()),
                    "initial_geometry_residual_p50_deg": float(
                        torch.rad2deg(initial_geometry_residual).quantile(0.5).detach().cpu()
                    ),
                    "initial_geometry_residual_p90_deg": float(
                        torch.rad2deg(initial_geometry_residual).quantile(0.9).detach().cpu()
                    ),
                    "initial_parallax_p10_deg": float(
                        torch.rad2deg(initial_parallax).quantile(0.1).detach().cpu()
                    ),
                    "initial_parallax_p50_deg": float(
                        torch.rad2deg(initial_parallax).quantile(0.5).detach().cpu()
                    ),
                },
            )
        pose_dim = max(0, (views - 1) * 6)
        if self.pose_dof_mode == "rotation_only":
            active_pose_index = torch.tensor(
                [frame * 6 + axis for frame in range(max(0, views - 1)) for axis in range(3, 6)],
                device=device,
                dtype=torch.long,
            )
        elif self.pose_dof_mode == "translation_only":
            active_pose_index = torch.tensor(
                [frame * 6 + axis for frame in range(max(0, views - 1)) for axis in range(3)],
                device=device,
                dtype=torch.long,
            )
        else:
            active_pose_index = torch.arange(pose_dim, device=device, dtype=torch.long)
        depth_dim = views * query_count
        failed_reason: str | None = None
        current_damping = min(
            self.lm_damping_max,
            max(self.damping, self.lm_damping_min),
        )
        accepted_steps = 0
        gain_ratios: list[float] = []
        gauge_scales: list[float] = []
        gradient_norms: list[float] = []
        pose_step_norms: list[float] = []
        depth_step_norms: list[float] = []
        trial_objectives: list[float] = []
        trial_dampings: list[float] = []
        trial_predicted_reductions: list[float] = []
        trial_actual_reductions: list[float] = []
        trial_gain_ratios: list[float] = []
        max_factor_jacobian_norm = 0.0
        analytic_autodiff_max_abs = 0.0
        analytic_autodiff_max_rel = 0.0
        termination_reason: str | None = None
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
                if self.jacobian_mode == "analytic":
                    prior_residual, prior_jacobian = _right_pose_prior_residual_and_jacobian(
                        cur_pose,
                        poses,
                    )
                    if self.validate_analytic_jacobian:
                        prior_zero = torch.zeros(pose_dim, device=device, dtype=torch.float32)

                        def reference_pose_prior(delta: torch.Tensor) -> torch.Tensor:
                            updated = [cur_pose[0]]
                            for frame_idx in range(1, views):
                                step = delta[(frame_idx - 1) * 6 : frame_idx * 6]
                                updated.append(
                                    cur_pose[frame_idx] @ _se3_exp_out_of_place(step)
                                )
                            return self._pose_prior_vector(torch.stack(updated, dim=0), poses)

                        with torch.enable_grad():
                            reference_prior_residual = reference_pose_prior(prior_zero)
                            reference_prior_jacobian = torch.func.jacrev(reference_pose_prior)(
                                prior_zero
                            )
                        if not torch.allclose(
                            prior_residual,
                            reference_prior_residual,
                            atol=self.analytic_jacobian_atol,
                            rtol=self.analytic_jacobian_rtol,
                        ) or not torch.allclose(
                            prior_jacobian,
                            reference_prior_jacobian,
                            atol=self.analytic_jacobian_atol,
                            rtol=self.analytic_jacobian_rtol,
                        ):
                            failed_reason = "analytic_pose_prior_jacobian_mismatch"
                            break
                else:
                    prior_zero = torch.zeros(pose_dim, device=device, dtype=torch.float32)

                    def pose_prior_from_delta(delta: torch.Tensor) -> torch.Tensor:
                        updated = [cur_pose[0]]
                        for frame_idx in range(1, views):
                            step = delta[(frame_idx - 1) * 6 : frame_idx * 6]
                            transform = _se3_exp_out_of_place(step)
                            updated.append(
                                cur_pose[frame_idx] @ transform
                                if self.pose_update_side == "right"
                                else transform @ cur_pose[frame_idx]
                            )
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
                    return _factor_residual_from_local_delta(
                        delta,
                        sp,
                        tp,
                        sr,
                        tr,
                        ld,
                        self.pose_update_side,
                    )

                if self.jacobian_mode == "analytic":
                    residual, jacobian = _factor_residual_and_analytic_jacobian(
                        cur_pose[chunk_src],
                        cur_pose[chunk_tgt],
                        src_ray[start:stop],
                        tgt_ray[start:stop],
                        cur_log[chunk_depth],
                    )
                    if self.validate_analytic_jacobian:
                        with torch.enable_grad():
                            jacobian_fn = torch.func.jacrev(single, argnums=0)
                            reference_residual = torch.func.vmap(single)(
                                local_zero,
                                cur_pose[chunk_src],
                                cur_pose[chunk_tgt],
                                src_ray[start:stop],
                                tgt_ray[start:stop],
                                cur_log[chunk_depth],
                            )
                            reference_jacobian = torch.func.vmap(jacobian_fn)(
                                local_zero,
                                cur_pose[chunk_src],
                                cur_pose[chunk_tgt],
                                src_ray[start:stop],
                                tgt_ray[start:stop],
                                cur_log[chunk_depth],
                            )
                        absolute = (jacobian - reference_jacobian).abs()
                        relative = absolute / reference_jacobian.abs().clamp_min(1.0e-8)
                        analytic_autodiff_max_abs = max(
                            analytic_autodiff_max_abs,
                            float(absolute.max().detach().cpu()),
                        )
                        analytic_autodiff_max_rel = max(
                            analytic_autodiff_max_rel,
                            float(relative.max().detach().cpu()),
                        )
                        if not torch.allclose(
                            residual,
                            reference_residual,
                            atol=self.analytic_jacobian_atol,
                            rtol=self.analytic_jacobian_rtol,
                        ) or not torch.allclose(
                            jacobian,
                            reference_jacobian,
                            atol=self.analytic_jacobian_atol,
                            rtol=self.analytic_jacobian_rtol,
                        ):
                            failed_reason = "analytic_jacobian_mismatch"
                            break
                else:
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
                max_factor_jacobian_norm = max(
                    max_factor_jacobian_norm,
                    float(jacobian.norm(dim=(-2, -1)).max().detach().cpu()),
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

            if failed_reason is not None:
                break

            if not all(torch.isfinite(value).all() for value in (hpp, gp, hpd, hdd, gd)):
                failed_reason = "non_finite_normal_equations"
                break
            if initial_median > 1.0e-6 and max_factor_jacobian_norm < 1.0e-8:
                failed_reason = "zero_jacobian"
                break
            gradient_norm = float(torch.cat([gp, gd]).norm().detach().cpu())
            gradient_norms.append(gradient_norm)
            if gradient_norm <= self.gradient_tolerance:
                termination_reason = "converged_gradient"
                break
            pose_diagonal = (
                hpp.diagonal().clamp_min(self.lm_diagonal_floor)
                if pose_dim
                else torch.zeros(0, device=device, dtype=torch.float32)
            )
            depth_diagonal = hdd.clamp_min(self.lm_diagonal_floor)
            gauge_jacobian = self._baseline_gauge_jacobian(cur_pose, poses)

            def solve_step(damping: float, *, diagonal: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
                active_pose_dim = int(active_pose_index.numel())
                if active_pose_dim:
                    active_hpp = hpp.index_select(0, active_pose_index).index_select(1, active_pose_index)
                    active_gp = gp.index_select(0, active_pose_index)
                    active_hpd = hpd.index_select(1, active_pose_index)
                    active_pose_diagonal = pose_diagonal.index_select(0, active_pose_index)
                    if diagonal:
                        damped_hpp = active_hpp + torch.diag(active_pose_diagonal * float(damping))
                    else:
                        damped_hpp = active_hpp + torch.eye(active_pose_dim, device=device) * float(damping)
                else:
                    damped_hpp = hpp.new_zeros((0, 0))
                    active_gp = gp.new_zeros(0)
                    active_hpd = hpd.new_zeros((depth_dim, 0))
                damped_hdd = hdd + (
                    depth_diagonal * float(damping)
                    if diagonal
                    else torch.full_like(hdd, float(damping))
                )
                active_gauge_jacobian = None
                if gauge_jacobian is not None and active_pose_dim:
                    candidate_gauge = gauge_jacobian.index_select(0, active_pose_index)
                    if float(candidate_gauge.norm()) > 1.0e-10:
                        active_gauge_jacobian = candidate_gauge
                try:
                    active_pose_step, depth_step = _solve_diagonal_depth_schur(
                        damped_hpp,
                        active_gp,
                        active_hpd,
                        damped_hdd,
                        gd,
                        constraint=active_gauge_jacobian,
                    )
                except (RuntimeError, ValueError):
                    return None
                pose_step = torch.zeros(pose_dim, device=device, dtype=torch.float32)
                if active_pose_dim:
                    pose_step.index_copy_(0, active_pose_index, active_pose_step)
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
                    transform = _se3_exp_out_of_place(float(step_scale) * pose_step[frame - 1])
                    trial_pose[frame] = (
                        cur_pose[frame] @ transform
                        if self.pose_update_side == "right"
                        else transform @ cur_pose[frame]
                    )
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
            stop_after_accept = False
            if self.solver_mode == "standard_lm":
                for _trial in range(self.lm_max_trials):
                    trial_dampings.append(float(current_damping))
                    solved = solve_step(current_damping, diagonal=True)
                    if solved is None:
                        current_damping = min(self.lm_damping_max, current_damping * lm_nu)
                        lm_nu *= 2.0
                        continue
                    delta_pose, delta_depth = solved
                    pose_step_norm = float(delta_pose.norm().detach().cpu())
                    depth_step_norm = float(delta_depth.norm().detach().cpu())
                    pose_step_norms.append(pose_step_norm)
                    depth_step_norms.append(depth_step_norm)
                    if max(pose_step_norm, depth_step_norm) <= self.step_tolerance:
                        termination_reason = "converged_step"
                        break
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
                    relative_pose = (
                        torch.linalg.inv(cur_pose[1:]) @ trial_pose[1:]
                        if self.pose_update_side == "right"
                        else trial_pose[1:] @ torch.linalg.inv(cur_pose[1:])
                    )
                    # Gauge projection changes translations after the raw LM
                    # step.  Use the effective tangent step after projection.
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
                    trial_objectives.append(float(trial_objective.detach().cpu()))
                    trial_predicted_reductions.append(float(predicted_reduction.detach().cpu()))
                    trial_actual_reductions.append(float(actual_reduction.detach().cpu()))
                    trial_gain_ratios.append(float(rho.detach().cpu()))
                    if (
                        bool(torch.isfinite(trial_objective))
                        and bool(torch.isfinite(predicted_reduction))
                        and bool(torch.isfinite(actual_reduction))
                        and float(predicted_reduction) > 0.0
                        and float(actual_reduction) > 1.0e-10
                        and float(rho) > self.lm_acceptance_eta
                    ):
                        previous_objective = current_objective
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
                        relative_improvement = float(actual_reduction) / max(
                            abs(float(previous_objective)), 1.0e-12
                        )
                        if relative_improvement <= self.relative_objective_tolerance:
                            termination_reason = "converged_objective"
                            stop_after_accept = True
                        break
                    current_damping = min(self.lm_damping_max, current_damping * lm_nu)
                    lm_nu *= 2.0
            else:
                solved = solve_step(current_damping, diagonal=False)
                if solved is None:
                    failed_reason = "linear_solve_failure"
                    break
                delta_pose, delta_depth = solved
                pose_step_norms.append(float(delta_pose.norm().detach().cpu()))
                depth_step_norms.append(float(delta_depth.norm().detach().cpu()))
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
                    trial_objectives.append(float(trial_objective.detach().cpu()))
                    trial_dampings.append(float(current_damping))
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
                if termination_reason is None:
                    termination_reason = "lm_no_descent"
                break
            if stop_after_accept:
                break

        final_angle = self._angular_residual(cur_pose, cur_log, src, tgt, depth_index, src_ray, tgt_ray)
        final_median = float(torch.rad2deg(final_angle).median().detach().cpu())
        objective_improved = float(current_objective) < float(initial_objective) - 1.0e-10
        residual_limit = max(1.0e-6, initial_median * self.residual_worse_tolerance)
        residual_acceptable = final_median <= residual_limit
        finite_state = bool(torch.isfinite(cur_pose).all()) and bool(torch.isfinite(cur_log).all())
        anchor_acceptable = bool(torch.allclose(cur_pose[0], poses[0], atol=1.0e-6, rtol=1.0e-6))
        gauge_error = 0.0
        if self.gauge_mode == "initial_baseline" and views > 1:
            initial_offsets = poses[:, :3, 3] - poses[0, :3, 3]
            reference_index = int(initial_offsets.norm(dim=-1).argmax())
            initial_length = initial_offsets[reference_index].norm()
            current_length = (cur_pose[reference_index, :3, 3] - cur_pose[0, :3, 3]).norm()
            gauge_error = float(
                ((current_length - initial_length).abs() / initial_length.clamp_min(1.0e-8))
                .detach()
                .cpu()
            )
        gauge_acceptable = gauge_error <= 1.0e-5
        accepted = (
            failed_reason is None
            and math.isfinite(final_median)
            and objective_improved
            and residual_acceptable
            and finite_state
            and anchor_acceptable
            and gauge_acceptable
            and accepted_steps > 0
        )
        if not accepted:
            cur_pose = poses
            cur_log = log0
        optimized_depth = torch.exp(-cur_log).reshape(views, query_count)
        original_depth = source_depth.reshape(views, query_count)
        scales = torch.ones(views, device=device, dtype=torch.float32)
        shifts = torch.zeros(views, device=device, dtype=torch.float32)
        affine_accepted = torch.zeros(views, device=device, dtype=torch.bool)
        affine_identity_error = torch.full(
            (views,), float("inf"), device=device, dtype=torch.float32
        )
        affine_fit_error = torch.full(
            (views,), float("inf"), device=device, dtype=torch.float32
        )
        if accepted and self.dense_depth_mode == "affine":
            for frame in range(views):
                mask = cache.source_valid[batch_idx, frame].to(device=device)
                support = int(mask.sum())
                if support < self.min_affine_support:
                    continue
                median_depth = float(original_depth[frame, mask].median().detach().cpu())
                (
                    scales[frame],
                    shifts[frame],
                    affine_ok,
                    affine_identity_error[frame],
                    affine_fit_error[frame],
                ) = _weighted_affine_fit_with_acceptance(
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
                    min_support=self.min_affine_support,
                    min_relative_improvement=self.affine_min_relative_improvement,
                )
                affine_accepted[frame] = bool(affine_ok)
                if affine_ok:
                    dense_source = depth_map[frame]
                    dense_source_valid = (
                        torch.isfinite(dense_source)
                        & (dense_source >= self.min_depth)
                        & (dense_source <= self.max_depth)
                    )
                    dense_trial = scales[frame] * dense_source + shifts[frame]
                    dense_trial_valid = (
                        torch.isfinite(dense_trial)
                        & (dense_trial >= self.min_depth)
                        & (dense_trial <= self.max_depth)
                    )
                    if bool(dense_source_valid.any()) and not bool(
                        dense_trial_valid[dense_source_valid].all()
                    ):
                        scales[frame] = 1.0
                        shifts[frame] = 0.0
                        affine_accepted[frame] = False
        reason = "accepted" if accepted else (
            failed_reason
            or ("residual_worse" if not residual_acceptable else None)
            or ("non_finite_state" if not finite_state else None)
            or ("anchor_changed" if not anchor_acceptable else None)
            or ("gauge_violation" if not gauge_acceptable else None)
            or termination_reason
            or "objective_not_improved"
        )
        return cur_pose, optimized_depth, scales, shifts, accepted, initial_median, final_median, {
            "reason": reason,
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
            "finite_state": bool(finite_state),
            "anchor_acceptable": bool(anchor_acceptable),
            "gauge_error": float(gauge_error),
            "gauge_acceptable": bool(gauge_acceptable),
            "initial_objective": float(initial_objective.detach().cpu()),
            "final_objective": float(current_objective.detach().cpu()),
            "accepted_steps": int(accepted_steps),
            "final_damping": float(current_damping),
            "solver_mode": self.solver_mode,
            "dense_depth_mode": self.dense_depth_mode,
            "gauge_mode": self.gauge_mode,
            "pose_update_side": self.pose_update_side,
            "pose_dof_mode": self.pose_dof_mode,
            "jacobian_mode": self.jacobian_mode,
            "max_factor_jacobian_norm": float(max_factor_jacobian_norm),
            "analytic_autodiff_max_abs": float(analytic_autodiff_max_abs),
            "analytic_autodiff_max_rel": float(analytic_autodiff_max_rel),
            "gradient_norms": gradient_norms,
            "pose_step_norms": pose_step_norms,
            "depth_step_norms": depth_step_norms,
            "trial_objectives": trial_objectives,
            "trial_dampings": trial_dampings,
            "trial_predicted_reductions": trial_predicted_reductions,
            "trial_actual_reductions": trial_actual_reductions,
            "trial_gain_ratios": trial_gain_ratios,
            "termination_reason": termination_reason,
            "depth_affine_accepted": affine_accepted.detach().cpu().tolist(),
            "depth_affine_identity_error": affine_identity_error.detach().cpu().tolist(),
            "depth_affine_fit_error": affine_fit_error.detach().cpu().tolist(),
            "min_initial_median_residual_deg": self.min_initial_median_residual_deg,
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
        supported = count > 0
        average = (output / count.clamp_min(1.0)).clamp_min(1.0e-6)
        return torch.where(supported, average, torch.zeros_like(output))

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
        if self.pose_update_side == "right":
            row[frame_slice.start : frame_slice.start + 3] = (
                poses[reference_index, :3, :3].transpose(0, 1) @ direction
            )
        else:
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
        residual, _ = _right_pose_prior_residual_and_jacobian(current, initial)
        return residual
