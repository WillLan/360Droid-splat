"""Pose-guided dense descriptor matching on ERP spherical geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    spherical_angular_distance,
)

from .factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from .grid_utils import feature_uv_to_image_uv, image_uv_to_feature_uv, make_feature_grid, normalize_hw


@dataclass(frozen=True)
class PoseGuidedDenseMatcher:
    """Build dense spherical factors from pose-guided local descriptor search."""

    search_radius: int = 4
    topk: int = 1
    min_match_confidence: float = 0.2
    min_static_confidence: float = 0.2
    min_match_score: float = 0.0
    max_factors: int = 65536
    max_samples_per_edge: int | None = None
    use_wraparound: bool = True
    forward_backward: bool = True
    fb_tolerance: float = 1.5
    use_depth_consistency: bool = True
    depth_consistency_rel: float = 0.03
    depth_consistency_abs: float = 0.05

    def match(
        self,
        *,
        poses_c2w: torch.Tensor,
        depth: torch.Tensor,
        dense_descriptors: torch.Tensor,
        match_confidence: torch.Tensor,
        sky_prob: torch.Tensor,
        image_hw: tuple[int, int],
        feature_hw: tuple[int, int] | None = None,
        edge_pairs: torch.Tensor | list[tuple[int, int]] | None = None,
        static_confidence: torch.Tensor | None = None,
        inverse_depth: bool = False,
        temporal_radius: int = 2,
        max_edges: int | None = None,
    ) -> DenseSphereFactorGraph:
        """Return a dense spherical factor graph for the requested frame pairs."""

        descriptors = _normalize_descriptors(dense_descriptors)
        confidence = _normalize_scalar_map(match_confidence, name="match_confidence")
        sky = _normalize_scalar_map(sky_prob, name="sky_prob")
        static = None if static_confidence is None else _normalize_scalar_map(static_confidence, name="static_confidence")
        depth_t = _normalize_depth(depth, inverse_depth=inverse_depth)
        poses = poses_c2w.to(device=descriptors.device, dtype=descriptors.dtype)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(f"poses_c2w must have shape Nx4x4, got {tuple(poses.shape)}.")
        if int(poses.shape[0]) != int(descriptors.shape[0]):
            raise ValueError("poses_c2w and dense_descriptors must contain the same number of frames.")

        feature_size = normalize_hw(feature_hw or descriptors.shape[-2:], name="feature_hw")
        image_size = normalize_hw(image_hw, name="image_hw")
        if tuple(descriptors.shape[-2:]) != feature_size:
            raise ValueError(f"dense_descriptors shape {tuple(descriptors.shape[-2:])} does not match feature_hw {feature_size}.")
        for name, value in (("match_confidence", confidence), ("sky_prob", sky)):
            if tuple(value.shape[-2:]) != feature_size:
                raise ValueError(f"{name} shape {tuple(value.shape[-2:])} does not match feature_hw {feature_size}.")

        if edge_pairs is None:
            edges = DenseSphereFactorGraph.build_edges(
                int(descriptors.shape[0]),
                temporal_radius=temporal_radius,
                max_edges=max_edges,
                device=descriptors.device,
            )
        else:
            edges = torch.as_tensor(edge_pairs, dtype=torch.long, device=descriptors.device)
        if edges.ndim != 2 or int(edges.shape[1]) != 2:
            raise ValueError(f"edge_pairs must have shape Ex2, got {tuple(edges.shape)}.")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= int(descriptors.shape[0])):
            raise ValueError("edge_pairs contain frame indices outside the local prediction.")

        graph = DenseSphereFactorGraph(edges=edges.detach().clone(), metadata={"feature_hw": feature_size, "image_hw": image_size})
        if edges.numel() == 0:
            return graph

        per_edge_cap = self._samples_per_edge(feature_size[0] * feature_size[1], int(edges.shape[0]))
        full_grid = make_feature_grid(feature_size, device=descriptors.device, dtype=descriptors.dtype).view(-1, 2)
        sample_idx = _linspace_indices(full_grid.shape[0], per_edge_cap, device=descriptors.device)
        src_uv = full_grid[sample_idx]
        for src, tgt in edges.detach().cpu().tolist():
            factor = self._match_edge(
                src=int(src),
                tgt=int(tgt),
                src_uv=src_uv,
                poses=poses,
                depth=depth_t,
                descriptors=descriptors,
                confidence=confidence,
                sky_prob=sky,
                static_confidence=static,
                feature_hw=feature_size,
                image_hw=image_size,
            )
            graph.add_factor(factor)
        return graph

    def _samples_per_edge(self, total: int, edge_count: int) -> int:
        cap = int(total)
        if self.max_samples_per_edge is not None:
            cap = min(cap, int(self.max_samples_per_edge))
        if int(self.max_factors) > 0 and edge_count > 0:
            cap = min(cap, max(1, int(self.max_factors) // int(edge_count)))
        return max(1, cap)

    def _match_edge(
        self,
        *,
        src: int,
        tgt: int,
        src_uv: torch.Tensor,
        poses: torch.Tensor,
        depth: torch.Tensor,
        descriptors: torch.Tensor,
        confidence: torch.Tensor,
        sky_prob: torch.Tensor,
        static_confidence: torch.Tensor | None,
        feature_hw: tuple[int, int],
        image_hw: tuple[int, int],
    ) -> DenseSphereFactor:
        device, dtype = descriptors.device, descriptors.dtype
        height, width = image_hw
        src_image_uv = feature_uv_to_image_uv(src_uv, feature_hw, image_hw)
        src_depth = _sample_single_map(depth[src : src + 1], src_image_uv)[:, 0]
        src_bearing = erp_pixel_to_bearing(src_image_uv, height, width).to(device=device, dtype=dtype)
        src_cam = src_bearing * src_depth.clamp_min(1.0e-6).unsqueeze(-1)

        src_pose = poses[src]
        tgt_pose = poses[tgt]
        world = (src_pose[:3, :3] @ src_cam.T).T + src_pose[:3, 3].view(1, 3)
        tgt_cam = (tgt_pose[:3, :3].T @ (world - tgt_pose[:3, 3].view(1, 3)).T).T
        projected_tgt_depth = torch.linalg.norm(tgt_cam, dim=-1)
        projected_tgt_bearing = F.normalize(tgt_cam, dim=-1, eps=1.0e-12)
        projected_tgt_image_uv = bearing_to_erp_pixel(projected_tgt_bearing, height, width, wrap=self.use_wraparound).to(dtype=dtype)
        if self.use_wraparound:
            projected_tgt_image_uv = projected_tgt_image_uv.clone()
            projected_tgt_image_uv[..., 0] = torch.remainder(projected_tgt_image_uv[..., 0], float(width))
        projected_tgt_uv = image_uv_to_feature_uv(projected_tgt_image_uv, feature_hw, image_hw)
        if self.use_wraparound:
            projected_tgt_uv = projected_tgt_uv.clone()
            projected_tgt_uv[..., 0] = torch.remainder(projected_tgt_uv[..., 0], float(feature_hw[1]))

        src_desc = _sample_single_map(descriptors[src : src + 1], src_uv)
        best_uv, best_sim = _local_descriptor_match(
            src_desc,
            descriptors[tgt : tgt + 1],
            projected_tgt_uv,
            radius=int(self.search_radius),
            feature_hw=feature_hw,
            use_wraparound=self.use_wraparound,
        )
        match_score = ((best_sim + 1.0) * 0.5).clamp(0.0, 1.0)

        tgt_image_uv = feature_uv_to_image_uv(best_uv, feature_hw, image_hw)
        if self.use_wraparound:
            tgt_image_uv = tgt_image_uv.clone()
            tgt_image_uv[..., 0] = torch.remainder(tgt_image_uv[..., 0], float(width))
        tgt_bearing = erp_pixel_to_bearing(tgt_image_uv, height, width).to(device=device, dtype=dtype)

        depth_pass, depth_error = self._depth_consistency(
            target_depth=depth[tgt : tgt + 1],
            target_uv=tgt_image_uv,
            projected_depth=projected_tgt_depth,
        )
        fb_pass, fb_error = self._forward_backward(
            descriptors=descriptors,
            src=src,
            tgt=tgt,
            src_uv=src_uv,
            tgt_uv=best_uv,
            feature_hw=feature_hw,
        )

        src_match = _sample_single_map(confidence[src : src + 1], src_uv)[:, 0].clamp(0.0, 1.0)
        tgt_match = _sample_single_map(confidence[tgt : tgt + 1], best_uv)[:, 0].clamp(0.0, 1.0)
        src_sky = _sample_single_map(sky_prob[src : src + 1], src_uv)[:, 0].clamp(0.0, 1.0)
        tgt_sky = _sample_single_map(sky_prob[tgt : tgt + 1], best_uv)[:, 0].clamp(0.0, 1.0)
        src_non_sky = (1.0 - src_sky).clamp(0.0, 1.0)
        tgt_non_sky = (1.0 - tgt_sky).clamp(0.0, 1.0)
        if static_confidence is None:
            src_static = (src_match * src_non_sky).clamp(0.0, 1.0)
            tgt_static = (tgt_match * tgt_non_sky).clamp(0.0, 1.0)
        else:
            src_static = _sample_single_map(static_confidence[src : src + 1], src_uv)[:, 0].clamp(0.0, 1.0)
            tgt_static = _sample_single_map(static_confidence[tgt : tgt + 1], best_uv)[:, 0].clamp(0.0, 1.0)

        phi = math.pi * (src_image_uv[..., 1] / float(height) - 0.5)
        latitude_weight = torch.cos(phi).clamp_min(0.0)
        weight = (
            src_match
            * tgt_match
            * src_static
            * tgt_static
            * src_non_sky
            * tgt_non_sky
            * match_score
            * fb_pass.float()
            * depth_pass.float()
            * latitude_weight
        )
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        horizontal_ok = torch.ones_like(src_depth, dtype=torch.bool)
        if not self.use_wraparound:
            horizontal_ok = (best_uv[..., 0] >= 0.0) & (best_uv[..., 0] < float(feature_hw[1]))
        valid_mask = (
            torch.isfinite(src_depth)
            & (src_depth > 0.0)
            & torch.isfinite(projected_tgt_depth)
            & (projected_tgt_depth > 0.0)
            & horizontal_ok
            & (best_uv[..., 1] >= 0.0)
            & (best_uv[..., 1] < float(feature_hw[0]))
            & torch.isfinite(weight)
            & (src_match >= float(self.min_match_confidence))
            & (tgt_match >= float(self.min_match_confidence))
            & (src_static >= float(self.min_static_confidence))
            & (tgt_static >= float(self.min_static_confidence))
            & (match_score >= float(self.min_match_score))
            & fb_pass
            & depth_pass
            & (weight > 0.0)
        )
        angular_error = torch.rad2deg(spherical_angular_distance(projected_tgt_bearing, tgt_bearing)).detach()
        sky_filtered = (src_non_sky < float(self.min_static_confidence)) | (tgt_non_sky < float(self.min_static_confidence))

        return DenseSphereFactor(
            src=src,
            tgt=tgt,
            src_uv=src_uv.detach(),
            tgt_uv=best_uv.detach(),
            src_bearing=src_bearing.detach(),
            tgt_bearing=tgt_bearing.detach(),
            weight=weight.detach(),
            match_score=match_score.detach(),
            valid_mask=valid_mask.detach(),
            metadata={
                "projected_tgt_uv": projected_tgt_uv.detach(),
                "projected_tgt_bearing": projected_tgt_bearing.detach(),
                "fb_pass_mask": fb_pass.detach(),
                "fb_error": fb_error.detach(),
                "depth_consistency_mask": depth_pass.detach(),
                "depth_error": depth_error.detach(),
                "sky_filtered_mask": sky_filtered.detach(),
                "latitude_weight": latitude_weight.detach(),
                "angular_error_deg": angular_error.detach(),
            },
        )

    def _depth_consistency(
        self,
        *,
        target_depth: torch.Tensor,
        target_uv: torch.Tensor,
        projected_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_depth_consistency:
            ok = torch.ones_like(projected_depth, dtype=torch.bool)
            return ok, torch.zeros_like(projected_depth)
        sampled = _sample_single_map(target_depth, target_uv)[:, 0]
        error = (sampled - projected_depth).abs()
        ok = (
            torch.isfinite(sampled)
            & torch.isfinite(projected_depth)
            & (sampled > 0.0)
            & (projected_depth > 0.0)
            & (error <= float(self.depth_consistency_abs) + float(self.depth_consistency_rel) * sampled.abs().clamp_min(1.0e-6))
        )
        return ok, error

    def _forward_backward(
        self,
        *,
        descriptors: torch.Tensor,
        src: int,
        tgt: int,
        src_uv: torch.Tensor,
        tgt_uv: torch.Tensor,
        feature_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.forward_backward:
            ok = torch.ones(src_uv.shape[0], device=src_uv.device, dtype=torch.bool)
            return ok, torch.zeros(src_uv.shape[0], device=src_uv.device, dtype=src_uv.dtype)
        tgt_desc = _sample_single_map(descriptors[tgt : tgt + 1], tgt_uv)
        back_uv, _ = _local_descriptor_match(
            tgt_desc,
            descriptors[src : src + 1],
            src_uv,
            radius=int(self.search_radius),
            feature_hw=feature_hw,
            use_wraparound=self.use_wraparound,
        )
        error = _feature_uv_distance(back_uv, src_uv, width=feature_hw[1], use_wraparound=self.use_wraparound)
        return error <= float(self.fb_tolerance), error


def _normalize_descriptors(descriptors: torch.Tensor) -> torch.Tensor:
    if descriptors.ndim != 4:
        raise ValueError(f"dense_descriptors must have shape NxCxHfxWf, got {tuple(descriptors.shape)}.")
    return F.normalize(descriptors.float(), dim=1, eps=1.0e-6)


def _normalize_scalar_map(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape Nx1xHfxWf, got {tuple(value.shape)}.")
    return value.float().clamp(0.0, 1.0)


def _normalize_depth(depth: torch.Tensor, *, inverse_depth: bool) -> torch.Tensor:
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or int(depth.shape[1]) != 1:
        raise ValueError(f"depth must have shape Nx1xHxW, got {tuple(depth.shape)}.")
    value = depth.float()
    if inverse_depth:
        return value.clamp_min(1.0e-6).reciprocal()
    return value.clamp_min(1.0e-6)


def _uv_to_grid(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    norm_x = 2.0 * (uv[..., 0] - 0.5) / max(width - 1, 1) - 1.0
    norm_y = 2.0 * (uv[..., 1] - 0.5) / max(height - 1, 1) - 1.0
    return torch.stack([norm_x, norm_y], dim=-1)


def _sample_single_map(map_tensor: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if map_tensor.ndim != 4 or int(map_tensor.shape[0]) != 1:
        raise ValueError(f"map_tensor must have shape 1xCxHxW, got {tuple(map_tensor.shape)}.")
    if uv.ndim != 2 or int(uv.shape[-1]) != 2:
        raise ValueError(f"uv must have shape Px2, got {tuple(uv.shape)}.")
    height, width = int(map_tensor.shape[-2]), int(map_tensor.shape[-1])
    selected = map_tensor.expand(int(uv.shape[0]), -1, -1, -1)
    grid = _uv_to_grid(uv.to(device=map_tensor.device, dtype=map_tensor.dtype), height, width).view(-1, 1, 1, 2)
    sampled = F.grid_sample(selected, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, 0, 0]


def _candidate_uv(center_uv: torch.Tensor, *, radius: int, feature_hw: tuple[int, int], use_wraparound: bool) -> torch.Tensor:
    offsets = [(float(dx), float(dy)) for dy in range(-int(radius), int(radius) + 1) for dx in range(-int(radius), int(radius) + 1)]
    offset_t = torch.tensor(offsets, device=center_uv.device, dtype=center_uv.dtype)
    candidates = center_uv.unsqueeze(1) + offset_t.view(1, -1, 2)
    candidates = candidates.clone()
    if use_wraparound:
        candidates[..., 0] = torch.remainder(candidates[..., 0], float(feature_hw[1]))
    else:
        candidates[..., 0] = candidates[..., 0].clamp(0.5, float(feature_hw[1]) - 0.5)
    candidates[..., 1] = candidates[..., 1].clamp(0.5, float(feature_hw[0]) - 0.5)
    return candidates


def _sample_candidates(map_tensor: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if map_tensor.ndim != 4 or int(map_tensor.shape[0]) != 1:
        raise ValueError(f"map_tensor must have shape 1xCxHxW, got {tuple(map_tensor.shape)}.")
    p, k = int(candidates.shape[0]), int(candidates.shape[1])
    height, width = int(map_tensor.shape[-2]), int(map_tensor.shape[-1])
    selected = map_tensor.expand(p, -1, -1, -1)
    grid = _uv_to_grid(candidates.to(device=map_tensor.device, dtype=map_tensor.dtype), height, width).view(p, k, 1, 2)
    sampled = F.grid_sample(selected, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, :, 0].transpose(1, 2)


def _local_descriptor_match(
    query_desc: torch.Tensor,
    target_map: torch.Tensor,
    center_uv: torch.Tensor,
    *,
    radius: int,
    feature_hw: tuple[int, int],
    use_wraparound: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(query_desc.float(), dim=-1, eps=1.0e-6)
    candidates = _candidate_uv(center_uv, radius=radius, feature_hw=feature_hw, use_wraparound=use_wraparound)
    sampled = _sample_candidates(F.normalize(target_map.float(), dim=1, eps=1.0e-6), candidates)
    sampled = F.normalize(sampled, dim=-1, eps=1.0e-6)
    scores = (sampled * query.unsqueeze(1)).sum(dim=-1)
    best_idx = scores.argmax(dim=1)
    best_uv = candidates[torch.arange(candidates.shape[0], device=candidates.device), best_idx]
    best_score = scores[torch.arange(scores.shape[0], device=scores.device), best_idx]
    return best_uv, best_score


def _feature_uv_distance(a: torch.Tensor, b: torch.Tensor, *, width: int, use_wraparound: bool) -> torch.Tensor:
    delta = a - b
    if use_wraparound:
        du = torch.remainder(delta[..., 0] + float(width) * 0.5, float(width)) - float(width) * 0.5
    else:
        du = delta[..., 0]
    return torch.linalg.norm(torch.stack([du, delta[..., 1]], dim=-1), dim=-1)


def _linspace_indices(total: int, count: int, *, device: torch.device) -> torch.Tensor:
    if int(count) >= int(total):
        return torch.arange(int(total), device=device)
    return torch.linspace(0, int(total) - 1, steps=int(count), device=device).round().long()
