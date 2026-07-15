"""Panorama loop retrieval and spherical/3D geometric verification."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from geometry.panorama_loop_contracts import (
    DenseSphericalLoopMeasurement,
    LoopPoseMeasurement,
    PanoramaLoopVerification,
)
from frontend.pano_vggt.alignment import SubmapAligner
from geometry.spherical_pseudo_correspondence import sample_joint_valid_fibonacci_uv
from geometry.sim3 import sim3_components, weighted_umeyama
from geometry.spherical_erp import erp_pixel_to_unit_ray, sample_erp_with_wrap

from .window_packet import LocalGaussianWindowPacket


@dataclass(frozen=True)
class PanoramaLoopCandidate:
    """One retrieved window and the frame pair that produced its best score."""

    packet: LocalGaussianWindowPacket
    score: float
    source_frame_index: int
    target_frame_index: int


def circular_yaw_shift(source: torch.Tensor, target: torch.Tensor) -> tuple[int, float]:
    """Estimate the longitude roll aligning ``target`` to ``source``."""

    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target descriptor maps must both be CxHxW")
    height = int(source.shape[-2])
    rows = torch.arange(height, device=source.device, dtype=source.dtype) + 0.5
    area = torch.cos(math.pi * (rows / float(height) - 0.5)).clamp_min(1.0e-6)
    source_signature = (source * area.view(1, height, 1)).sum(dim=1)
    target_signature = (target * area.view(1, height, 1)).sum(dim=1)
    source_fft = torch.fft.rfft(source_signature.float(), dim=-1)
    target_fft = torch.fft.rfft(target_signature.float(), dim=-1)
    correlation = torch.fft.irfft(
        (source_fft.conj() * target_fft).sum(dim=0), n=int(source.shape[-1]), dim=-1
    )
    shift = int(correlation.argmax().item())
    score = float(correlation[shift].detach().cpu())
    return shift, score


def spherical_rotation_ransac(
    target_bearing: torch.Tensor,
    source_bearing: torch.Tensor,
    weight: torch.Tensor,
    *,
    threshold_rad: float,
    iterations: int = 128,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Estimate the target-to-source rotation with deterministic batched RANSAC."""

    if target_bearing.shape != source_bearing.shape or target_bearing.ndim != 2 or target_bearing.shape[-1] != 3:
        raise ValueError("Spherical rotation bearings must both have shape Nx3")
    count = int(target_bearing.shape[0])
    if count < 3:
        raise ValueError("Spherical rotation RANSAC requires at least three matches")
    device, dtype = target_bearing.device, target_bearing.dtype
    target = F.normalize(target_bearing, dim=-1, eps=1.0e-8)
    source = F.normalize(source_bearing.to(target), dim=-1, eps=1.0e-8)
    weights = weight.to(device=device, dtype=dtype).clamp_min(1.0e-8)
    hypotheses = max(1, int(iterations))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    # Weighted Gumbel top-k gives deterministic, non-replacement minimal sets
    # without a Python loop or a CUDA synchronization per hypothesis.
    uniform = torch.rand(hypotheses, count, device=device, dtype=dtype, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp(1.0e-7, 1.0 - 1.0e-7)))
    sample_indices = torch.topk(weights.log().view(1, -1) + gumbel, k=3, dim=-1).indices
    target_sample = target[sample_indices]
    source_sample = source[sample_indices]
    sample_weight = weights[sample_indices]
    sample_weight = sample_weight / sample_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_mean = (sample_weight[..., None] * target_sample).sum(dim=1)
    source_mean = (sample_weight[..., None] * source_sample).sum(dim=1)
    target_centered = target_sample - target_mean[:, None]
    source_centered = source_sample - source_mean[:, None]
    covariance = torch.einsum(
        "ki,kij,kil->kjl", sample_weight, target_centered, source_centered
    )
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=device, dtype=dtype).expand(hypotheses, -1, -1).clone()
    raw_rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    correction[:, -1, -1] = torch.where(
        torch.linalg.det(raw_rotation) < 0.0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    rotations = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    rotated = torch.einsum("kij,nj->kni", rotations, target)
    angular = torch.atan2(
        torch.cross(source.unsqueeze(0), rotated, dim=-1).norm(dim=-1),
        (source.unsqueeze(0) * rotated).sum(dim=-1).clamp(-1.0, 1.0),
    )
    inliers = angular <= max(float(threshold_rad), 1.0e-8)
    weighted_support = (inliers.to(dtype) * weights.view(1, -1)).sum(dim=-1)
    inlier_error = (
        inliers.to(dtype) * weights.view(1, -1) * angular
    ).sum(dim=-1) / weighted_support.clamp_min(1.0e-8)
    maximum_support = weighted_support.max()
    tied = weighted_support >= maximum_support - 1.0e-8
    ranked_error = torch.where(tied, inlier_error, torch.full_like(inlier_error, torch.inf))
    best = int(ranked_error.argmin().item())
    best_inliers = inliers[best]
    if int(best_inliers.sum()) >= 3:
        transform = weighted_umeyama(
            target[best_inliers], source[best_inliers], weights[best_inliers], allow_scale=False
        )
    else:
        transform = weighted_umeyama(target, source, weights, allow_scale=False)
    _, rotation, _ = sim3_components(transform)
    rotated = torch.einsum("ij,nj->ni", rotation, target)
    angular = torch.atan2(
        torch.cross(source, rotated, dim=-1).norm(dim=-1),
        (source * rotated).sum(dim=-1).clamp(-1.0, 1.0),
    )
    final_inliers = angular <= max(float(threshold_rad), 1.0e-8)
    inlier_ratio = float(final_inliers.float().mean().detach().cpu())
    residual = (
        float(angular[final_inliers].mean().detach().cpu())
        if bool(final_inliers.any())
        else float("inf")
    )
    return rotation, final_inliers, inlier_ratio, residual


def spherical_coverage_bins(
    bearing: torch.Tensor,
    *,
    latitude_bins: int = 3,
    longitude_bins: int = 4,
) -> tuple[int, int]:
    """Count occupied approximately equal-area bins on the unit sphere."""

    if bearing.ndim != 2 or int(bearing.shape[-1]) != 3:
        raise ValueError("bearing must have shape Nx3")
    lat_count = max(1, int(latitude_bins))
    lon_count = max(1, int(longitude_bins))
    if int(bearing.shape[0]) == 0:
        return 0, lat_count * lon_count
    value = F.normalize(bearing.float(), dim=-1, eps=1.0e-8)
    y = value[:, 1].clamp(-1.0, 1.0)
    longitude = torch.atan2(value[:, 0], value[:, 2])
    lat_index = torch.floor((y + 1.0) * (0.5 * lat_count)).long().clamp(0, lat_count - 1)
    lon_index = torch.floor(
        (longitude + math.pi) * (float(lon_count) / (2.0 * math.pi))
    ).long().clamp(0, lon_count - 1)
    occupied = torch.unique(lat_index * lon_count + lon_index)
    return int(occupied.numel()), lat_count * lon_count


def rotation_geodesic_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the SO(3) geodesic angle between two 3x3 rotations."""

    if tuple(first.shape) != (3, 3) or tuple(second.shape) != (3, 3):
        raise ValueError("rotation matrices must both have shape 3x3")
    relative = first.transpose(0, 1) @ second.to(first)
    skew = torch.stack(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    sine = 0.5 * skew.norm()
    cosine = 0.5 * (relative.trace() - 1.0)
    return float(torch.atan2(sine, cosine.clamp(-1.0, 1.0)).detach().cpu())


class PanoramaLoopDetector:
    def __init__(
        self,
        *,
        top_k: int = 5,
        exclude_recent_windows: int = 3,
        min_retrieval_score: float = 0.35,
        min_match_cosine: float = 0.45,
        min_matches: int = 32,
        max_matches: int = 512,
        min_inlier_ratio: float = 0.30,
        max_alignment_residual: float = 0.35,
        max_scale_change: float = 2.5,
        coincident_translation_threshold: float = 0.15,
        coincident_rotation_residual_deg: float = 2.0,
        rotation_ransac_iterations: int = 128,
        factor_queries_per_direction: int = 2048,
        fibonacci_oversample_factor: int = 8,
        fibonacci_seed: int = 123,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
        sky_threshold: float = 0.5,
        min_match_margin: float = 0.01,
        max_match_entropy: float = 0.95,
        forward_backward: bool = True,
        fb_tolerance_deg: float = 1.0,
        min_factor_weight: float = 0.01,
        target_area_correction: bool = True,
        depth_factor_weight: float = 0.1,
        s2_huber_delta_deg: float = 1.0,
        descriptor_mode: str = "latitude_bands",
        candidate_nms_radius: int = 2,
        max_verified_candidates: int = 3,
        max_accepted_loops: int = 1,
        verification_mode: str = "legacy",
        rotation_inlier_threshold_deg: float = 2.0,
        min_rotation_inlier_ratio: float | None = None,
        min_spherical_coverage_bins: int = 6,
        coverage_latitude_bins: int = 3,
        coverage_longitude_bins: int = 4,
        max_rotation_consistency_deg: float = 3.0,
        max_normalized_alignment_residual: float = 0.10,
        loop_dcs_phi: float | None = None,
    ) -> None:
        self.top_k = max(1, int(top_k))
        self.exclude_recent_windows = max(0, int(exclude_recent_windows))
        self.min_retrieval_score = float(min_retrieval_score)
        self.min_match_cosine = float(min_match_cosine)
        self.min_matches = max(3, int(min_matches))
        self.max_matches = max(self.min_matches, int(max_matches))
        self.coincident_translation_threshold = float(coincident_translation_threshold)
        self.coincident_rotation_residual = math.radians(float(coincident_rotation_residual_deg))
        self.rotation_ransac_iterations = max(1, int(rotation_ransac_iterations))
        self.factor_queries_per_direction = max(1, int(factor_queries_per_direction))
        self.fibonacci_oversample_factor = max(1, int(fibonacci_oversample_factor))
        self.fibonacci_seed = int(fibonacci_seed)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.sky_threshold = float(sky_threshold)
        self.min_match_margin = float(min_match_margin)
        self.max_match_entropy = float(max_match_entropy)
        self.forward_backward = bool(forward_backward)
        self.fb_tolerance = math.radians(float(fb_tolerance_deg))
        self.min_factor_weight = float(min_factor_weight)
        self.target_area_correction = bool(target_area_correction)
        self.depth_factor_weight = float(depth_factor_weight)
        self.s2_huber_delta_deg = float(s2_huber_delta_deg)
        self.descriptor_mode = str(descriptor_mode).strip().lower()
        self.so3_retrieval = self.descriptor_mode in {
            "so3_sh_gram",
            "so3",
            "spherical_harmonic_gram",
        }
        self.candidate_nms_radius = max(0, int(candidate_nms_radius))
        self.max_verified_candidates = max(1, int(max_verified_candidates))
        self.max_accepted_loops = max(1, int(max_accepted_loops))
        self.verification_mode = str(verification_mode).strip().lower()
        self.so3_verification = self.verification_mode in {
            "spherical_so3",
            "so3",
            "full_so3",
        }
        self.rotation_inlier_threshold = math.radians(float(rotation_inlier_threshold_deg))
        self.min_rotation_inlier_ratio = float(
            min_inlier_ratio if min_rotation_inlier_ratio is None else min_rotation_inlier_ratio
        )
        self.min_spherical_coverage_bins = max(1, int(min_spherical_coverage_bins))
        self.coverage_latitude_bins = max(1, int(coverage_latitude_bins))
        self.coverage_longitude_bins = max(1, int(coverage_longitude_bins))
        total_coverage_bins = self.coverage_latitude_bins * self.coverage_longitude_bins
        if self.min_spherical_coverage_bins > total_coverage_bins:
            raise ValueError(
                "min_spherical_coverage_bins cannot exceed latitude_bins*longitude_bins"
            )
        self.max_rotation_consistency = math.radians(float(max_rotation_consistency_deg))
        self.max_normalized_alignment_residual = float(max_normalized_alignment_residual)
        self.loop_dcs_phi = (
            None
            if loop_dcs_phi is None or float(loop_dcs_phi) <= 0.0
            else float(loop_dcs_phi)
        )
        self.aligner = SubmapAligner(
            align_mode="sim3",
            max_residual=float(max_alignment_residual),
            min_inlier_ratio=float(min_inlier_ratio),
            max_scale_change=float(max_scale_change),
            min_points=self.min_matches,
            return_rejected_transform=True,
        )
        self.memory: list[LocalGaussianWindowPacket] = []
        self._descriptor_database: torch.Tensor | None = None
        self._descriptor_offsets: list[tuple[int, int]] = []
        self._descriptor_rows = 0

    def add(self, packet: LocalGaussianWindowPacket) -> None:
        if self.so3_retrieval:
            descriptor = packet.retrieval_descriptors.detach().cpu().to(torch.float16)
            if descriptor.ndim != 2:
                raise ValueError("SO(3) retrieval descriptors must have shape SxD")
            rows, dimension = (int(value) for value in descriptor.shape)
            required = self._descriptor_rows + rows
            if self._descriptor_database is None:
                self._descriptor_database = torch.empty(
                    max(64, required), dimension, dtype=torch.float16
                )
            elif int(self._descriptor_database.shape[1]) != dimension:
                raise ValueError(
                    "Loop descriptor dimension changed within one retrieval database: "
                    f"database={int(self._descriptor_database.shape[1])}, incoming={dimension}."
                )
            elif int(self._descriptor_database.shape[0]) < required:
                capacity = max(required, int(self._descriptor_database.shape[0]) * 2)
                expanded = torch.empty(capacity, dimension, dtype=torch.float16)
                expanded[: self._descriptor_rows] = self._descriptor_database[
                    : self._descriptor_rows
                ]
                self._descriptor_database = expanded
            assert self._descriptor_database is not None
            start = self._descriptor_rows
            self._descriptor_database[start:required] = descriptor
            self._descriptor_offsets.append((start, required))
            self._descriptor_rows = required
        self.memory.append(packet)

    @property
    def descriptor_database_bytes(self) -> int:
        if self._descriptor_database is None:
            return 0
        return int(
            self._descriptor_rows
            * int(self._descriptor_database.shape[1])
            * self._descriptor_database.element_size()
        )

    def retrieve(self, packet: LocalGaussianWindowPacket) -> list[PanoramaLoopCandidate]:
        if not self.memory:
            return []
        cutoff = max(0, len(self.memory) - self.exclude_recent_windows)
        candidates = self.memory[:cutoff]
        if not candidates:
            return []
        if not self.so3_retrieval:
            # Preserve the historical frame-zero/yaw-only path bit-for-bit when
            # the new descriptor mode is disabled.
            query = packet.retrieval_descriptors[0].detach().float()
            scored: list[PanoramaLoopCandidate] = []
            for candidate in candidates:
                descriptor = candidate.retrieval_descriptors[0].to(query).float()
                score = float(F.cosine_similarity(query, descriptor, dim=0).detach().cpu())
                if score >= self.min_retrieval_score:
                    scored.append(PanoramaLoopCandidate(candidate, score, 0, 0))
            scored.sort(key=lambda item: item.score, reverse=True)
            return scored[: self.top_k]

        query = F.normalize(packet.retrieval_descriptors.detach().cpu().float(), dim=-1, eps=1.0e-8)
        descriptor_dim = int(query.shape[-1])
        if self._descriptor_database is None or len(self._descriptor_offsets) != len(self.memory):
            raise RuntimeError("SO(3) descriptor database is inconsistent with loop memory")
        if int(self._descriptor_database.shape[1]) != descriptor_dim:
            raise ValueError(
                "Loop descriptor dimension changed within one retrieval database: "
                f"query={descriptor_dim}, database={int(self._descriptor_database.shape[1])}."
            )
        offsets = self._descriptor_offsets[:cutoff]
        row_counts = [stop - start for start, stop in offsets]
        database_rows = offsets[-1][1]
        database = F.normalize(
            self._descriptor_database[:database_rows].float(),
            dim=-1,
            eps=1.0e-8,
        )
        similarity = database @ query.T
        scored = []
        cursor = 0
        for candidate, count in zip(candidates, row_counts):
            block = similarity[cursor : cursor + count]
            flat_index = int(block.reshape(-1).argmax().item())
            source_frame_index = flat_index // int(query.shape[0])
            target_frame_index = flat_index % int(query.shape[0])
            score = float(block[source_frame_index, target_frame_index].item())
            cursor += count
            if score >= self.min_retrieval_score:
                scored.append(
                    PanoramaLoopCandidate(
                        candidate,
                        score,
                        source_frame_index,
                        target_frame_index,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        selected: list[PanoramaLoopCandidate] = []
        for item in scored:
            if any(
                abs(int(item.packet.window_id) - int(previous.packet.window_id))
                <= self.candidate_nms_radius
                for previous in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= self.top_k:
                break
        return selected

    @staticmethod
    def _verification_uv(indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
        row = torch.div(indices, width, rounding_mode="floor").float() + 0.5
        col = torch.remainder(indices, width).float() + 0.5
        return torch.stack([col, row], dim=-1)

    @staticmethod
    def _to_observation_uv(
        uv: torch.Tensor,
        verification_hw: tuple[int, int],
        observation_hw: tuple[int, int],
    ) -> torch.Tensor:
        source_h, source_w = verification_hw
        target_h, target_w = observation_hw
        out = uv.clone()
        out[..., 0] = uv[..., 0] * float(target_w) / float(source_w)
        out[..., 1] = uv[..., 1] * float(target_h) / float(source_h)
        return out

    def _mutual_matches(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels, height, width = (int(value) for value in source.shape)
        if tuple(target.shape) != (channels, height, width):
            raise ValueError("Loop verification feature maps must share C/H/W")
        source_flat = F.normalize(source.float().reshape(channels, -1).T, dim=-1, eps=1.0e-8)
        target_flat = F.normalize(target.float().reshape(channels, -1).T, dim=-1, eps=1.0e-8)
        total = int(source_flat.shape[0])
        if total > self.max_matches * 4:
            stride = max(1, total // (self.max_matches * 4))
            source_indices = torch.arange(0, total, stride, device=source.device)[: self.max_matches * 4]
        else:
            source_indices = torch.arange(total, device=source.device)
        similarity = source_flat[source_indices] @ target_flat.T
        values, target_indices = torch.topk(similarity, k=min(2, int(similarity.shape[-1])), dim=-1)
        reverse = target_flat[target_indices[:, 0]] @ source_flat.T
        reverse_indices = reverse.argmax(dim=-1)
        mutual = reverse_indices == source_indices
        cosine = values[:, 0]
        margin = values[:, 0] - values[:, 1] if int(values.shape[1]) > 1 else values[:, 0]
        valid = mutual & (cosine >= self.min_match_cosine) & (margin > 0.0)
        source_indices = source_indices[valid]
        target_indices = target_indices[valid, 0]
        weight = (((cosine[valid] + 1.0) * 0.5) * torch.sigmoid(10.0 * margin[valid])).clamp(0.0, 1.0)
        if source_indices.numel() > self.max_matches:
            selected = torch.topk(weight, k=self.max_matches, largest=True).indices
            source_indices, target_indices, weight = source_indices[selected], target_indices[selected], weight[selected]
        return source_indices, target_indices, weight

    def _points_for_matches(
        self,
        packet: LocalGaussianWindowPacket,
        frame_index: int,
        verification_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_h, feature_w = tuple(int(v) for v in packet.verification_features.shape[-2:])
        image_h, image_w = packet.observation.image_size
        uv_feature = self._verification_uv(verification_indices, feature_h, feature_w).to(
            device=packet.observation.refined_depth.device
        )
        uv_image = self._to_observation_uv(uv_feature, (feature_h, feature_w), (image_h, image_w))
        depth_map = packet.observation.refined_depth[:, frame_index]
        depth = sample_erp_with_wrap(depth_map, uv_image.unsqueeze(0))[0, ..., 0]
        ray = erp_pixel_to_unit_ray(uv_image, image_h, image_w).to(depth)
        camera_point = depth[:, None] * ray
        pose = packet.local_poses_c2w[frame_index].to(camera_point)
        anchor_point = torch.einsum("ij,nj->ni", pose[:3, :3], camera_point) + pose[:3, 3]
        confidence = sample_erp_with_wrap(
            packet.observation.confidence[:, frame_index], uv_image.unsqueeze(0)
        )[0, ..., 0]
        valid = torch.isfinite(anchor_point).all(dim=-1) & torch.isfinite(depth) & (depth > 0.0)
        return anchor_point, confidence, valid

    @staticmethod
    def _scale_uv(uv: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        result = uv.clone()
        result[..., 0] *= float(target_w) / float(source_w)
        result[..., 1] *= float(target_h) / float(source_h)
        return result

    def _fibonacci_matches(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
        *,
        source_frame_index: int,
        target_frame_index: int,
        direction: int,
    ) -> dict[str, torch.Tensor | int]:
        source_device = source.verification_features.device
        target_device = target.verification_features.device
        work_device = source_device if source_device.type == "cuda" else target_device
        source_depth_map = source.observation.refined_depth[0, source_frame_index].detach().to(work_device)
        seed = (
            self.fibonacci_seed
            + 1_000_003 * int(source.window_id)
            + 10_007 * int(target.window_id)
            + 101 * int(source.frame_ids[source_frame_index])
            + 17 * int(direction)
        ) & 0x7FFFFFFF
        queries = sample_joint_valid_fibonacci_uv(
            source_depth_map,
            source_depth_map,
            count=self.factor_queries_per_direction,
            oversample_factor=self.fibonacci_oversample_factor,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            source_valid=source.finite_gaussian_mask[0, source_frame_index].detach(),
            target_valid=source.finite_gaussian_mask[0, source_frame_index].detach(),
            source_sky_probability=source.sky_prob[0, source_frame_index].detach(),
            target_sky_probability=source.sky_prob[0, source_frame_index].detach(),
            sky_threshold=self.sky_threshold,
            seed=seed,
        )
        source_uv = queries.uv
        if int(source_uv.shape[0]) == 0:
            return {"count": 0, "seed": seed}

        source_feature = source.verification_features[0, source_frame_index].to(source_depth_map)
        target_feature = target.verification_features[0, target_frame_index].to(source_depth_map)
        source_feature_hw = tuple(int(value) for value in source_feature.shape[-2:])
        target_feature_hw = tuple(int(value) for value in target_feature.shape[-2:])
        source_uv_feature = self._scale_uv(source_uv, source.observation.image_size, source_feature_hw)
        source_descriptor = F.normalize(
            sample_erp_with_wrap(source_feature, source_uv_feature).float(), dim=-1, eps=1.0e-8
        )
        target_flat = F.normalize(target_feature.float().reshape(int(target_feature.shape[0]), -1).T, dim=-1, eps=1.0e-8)
        similarity = source_descriptor @ target_flat.T
        target_height, target_width = target_feature_hw
        target_rows = torch.arange(target_height, device=similarity.device, dtype=similarity.dtype) + 0.5
        target_area = torch.cos(math.pi * (target_rows / float(target_height) - 0.5)).clamp_min(1.0e-6)
        ranking_logits = similarity / 0.07
        if self.target_area_correction:
            ranking_logits = ranking_logits + target_area.log().view(target_height, 1).expand(-1, target_width).reshape(1, -1)
        top_values, top_indices = torch.topk(ranking_logits, k=min(2, int(ranking_logits.shape[-1])), dim=-1)
        target_index = top_indices[:, 0]
        cosine = similarity.gather(1, target_index[:, None])[:, 0]
        margin = top_values[:, 0] - top_values[:, 1] if int(top_values.shape[1]) > 1 else top_values[:, 0]
        probability = torch.softmax(ranking_logits, dim=-1)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(2, int(probability.shape[-1]))), 1.0)

        target_uv_feature = self._verification_uv(target_index, target_height, target_width).to(source_uv)
        target_uv = self._scale_uv(target_uv_feature, target_feature_hw, target.observation.image_size)
        forward_backward_valid = torch.ones_like(cosine, dtype=torch.bool)
        if self.forward_backward:
            source_flat = F.normalize(source_feature.float().reshape(int(source_feature.shape[0]), -1).T, dim=-1, eps=1.0e-8)
            reverse_logits = target_flat[target_index] @ source_flat.T / 0.07
            if self.target_area_correction:
                source_height, source_width = source_feature_hw
                source_rows = torch.arange(source_height, device=similarity.device, dtype=similarity.dtype) + 0.5
                source_area = torch.cos(math.pi * (source_rows / float(source_height) - 0.5)).clamp_min(1.0e-6)
                reverse_logits = reverse_logits + source_area.log().view(source_height, 1).expand(-1, source_width).reshape(1, -1)
            reverse_index = reverse_logits.argmax(dim=-1)
            reverse_uv_feature = self._verification_uv(reverse_index, *source_feature_hw).to(source_uv)
            query_ray = erp_pixel_to_unit_ray(source_uv_feature, *source_feature_hw)
            reverse_ray = erp_pixel_to_unit_ray(reverse_uv_feature, *source_feature_hw)
            reverse_angle = torch.atan2(
                torch.cross(query_ray, reverse_ray, dim=-1).norm(dim=-1),
                (query_ray * reverse_ray).sum(dim=-1).clamp(-1.0, 1.0),
            )
            forward_backward_valid = reverse_angle <= self.fb_tolerance

        target_depth = sample_erp_with_wrap(
            target.observation.refined_depth[0, target_frame_index].detach().to(source_depth_map), target_uv
        )[..., 0]
        target_valid = sample_erp_with_wrap(
            target.finite_gaussian_mask[0, target_frame_index].detach().float().to(source_depth_map), target_uv
        )[..., 0] >= 0.5
        target_sky = sample_erp_with_wrap(
            target.sky_prob[0, target_frame_index].detach().to(source_depth_map), target_uv
        )[..., 0].clamp(0.0, 1.0)
        # Adapter scores, depth validity and sky masks are hard gates only.
        # Fibonacci already samples equal solid angle, so retained loop factors
        # carry unit measurement weight.
        weight = torch.ones_like(cosine)
        valid = (
            target_valid
            & forward_backward_valid
            & torch.isfinite(target_depth)
            & (target_depth >= self.min_depth)
            & (target_depth <= self.max_depth)
            & (target_sky < self.sky_threshold)
            & (cosine >= self.min_match_cosine)
            & (margin >= self.min_match_margin)
            & (entropy <= self.max_match_entropy)
        )
        selected = torch.nonzero(valid, as_tuple=False).flatten()
        raw_valid_count = int(selected.numel())
        if self.so3_verification and raw_valid_count > self.max_matches:
            quality = (
                ((cosine[selected] + 1.0) * 0.5)
                * torch.sigmoid(10.0 * margin[selected])
                * (1.0 - entropy[selected]).clamp_min(0.0)
            )
            selected = selected[torch.topk(quality, k=self.max_matches, largest=True).indices]
        keep = torch.zeros_like(valid)
        keep[selected] = True
        return {
            "count": int(selected.numel()),
            "raw_valid_count": raw_valid_count,
            "seed": seed,
            "source_uv": source_uv[keep],
            "target_uv": target_uv[keep],
            "source_bearing": queries.bearing[keep],
            "target_bearing": erp_pixel_to_unit_ray(target_uv[keep], *target.observation.image_size),
            "source_depth": queries.source_depth[keep],
            "target_depth": target_depth[keep],
            "weight": weight[keep],
            "top1_cosine": cosine[keep],
            "top2_margin": margin[keep],
            "entropy": entropy[keep],
        }

    @staticmethod
    def _anchor_points_from_match(
        packet: LocalGaussianWindowPacket,
        frame_index: int,
        bearing: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        camera = bearing * depth[:, None]
        pose = packet.local_poses_c2w[frame_index].to(camera)
        return camera @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]

    def _dense_factor_from_match(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
        source_frame_index: int,
        target_frame_index: int,
        match: dict[str, torch.Tensor | int],
        *,
        use_depth: bool,
        edge_type: str,
    ) -> DenseSphericalLoopMeasurement:
        return DenseSphericalLoopMeasurement(
            source=int(source.window_id),
            target=int(target.window_id),
            source_local_pose=source.local_poses_c2w[source_frame_index].detach(),
            target_local_pose=target.local_poses_c2w[target_frame_index].detach(),
            source_bearing=torch.as_tensor(match["source_bearing"]).detach(),
            target_bearing=torch.as_tensor(match["target_bearing"]).detach(),
            source_depth=torch.as_tensor(match["source_depth"]).detach(),
            target_depth=torch.as_tensor(match["target_depth"]).detach(),
            factor_weight=torch.as_tensor(match["weight"]).detach(),
            depth_factor_weight=self.depth_factor_weight,
            s2_huber_delta_deg=self.s2_huber_delta_deg,
            use_depth=bool(use_depth),
            edge_type=edge_type,
            dcs_phi=self.loop_dcs_phi if edge_type.startswith("loop") else None,
            metadata={
                "fibonacci_seed": int(match["seed"]),
                "num_matches": int(match["count"]),
                "source_frame_id": int(source.frame_ids[source_frame_index]),
                "target_frame_id": int(target.frame_ids[target_frame_index]),
            },
        )

    @staticmethod
    def _select_match_inliers(
        match: dict[str, torch.Tensor | int],
        inlier: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        """Select one direction's angular inliers without changing metadata."""

        count = int(match["count"])
        if tuple(inlier.shape) != (count,):
            raise ValueError("Loop inlier mask does not match directional correspondence count")
        selected: dict[str, torch.Tensor | int] = {}
        for key, value in match.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == count:
                selected[key] = value[inlier.to(device=value.device)]
            else:
                selected[key] = value
        selected["count"] = int(inlier.sum().item())
        return selected

    def verify_pair(
        self,
        source: LocalGaussianWindowPacket,
        target: LocalGaussianWindowPacket,
        *,
        retrieval_score: float = 1.0,
        edge_type: str = "loop",
        source_frame_index: int = 0,
        target_frame_index: int = 0,
    ) -> PanoramaLoopVerification:
        source_feature_raw = source.verification_features[0, source_frame_index]
        target_feature_raw = target.verification_features[0, target_frame_index]
        verification_device = (
            source_feature_raw.device
            if source_feature_raw.device.type == "cuda"
            else target_feature_raw.device
        )
        source_feature = source_feature_raw.to(verification_device)
        target_feature = target_feature_raw.to(verification_device)
        yaw_shift, yaw_score = circular_yaw_shift(source_feature, target_feature)
        try:
            forward_match = self._fibonacci_matches(
                source,
                target,
                source_frame_index=source_frame_index,
                target_frame_index=target_frame_index,
                direction=0,
            )
            reverse_match = self._fibonacci_matches(
                target,
                source,
                source_frame_index=target_frame_index,
                target_frame_index=source_frame_index,
                direction=1,
            )
        except ValueError as exc:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, 0, 0.0, float("inf"), f"feature_mismatch:{exc}")
        match_count = int(forward_match["count"]) + int(reverse_match["count"])
        if match_count < self.min_matches:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, match_count, 0.0, float("inf"), "too_few_fibonacci_matches")

        source_parts, target_parts, weight_parts = [], [], []
        source_depth_parts, target_depth_parts = [], []
        source_bearing_parts, target_bearing_parts = [], []
        if int(forward_match["count"]) > 0:
            source_parts.append(self._anchor_points_from_match(
                source, source_frame_index, torch.as_tensor(forward_match["source_bearing"]), torch.as_tensor(forward_match["source_depth"])
            ))
            target_parts.append(self._anchor_points_from_match(
                target, target_frame_index, torch.as_tensor(forward_match["target_bearing"]), torch.as_tensor(forward_match["target_depth"])
            ))
            weight_parts.append(torch.as_tensor(forward_match["weight"]))
            source_depth_parts.append(torch.as_tensor(forward_match["source_depth"]))
            target_depth_parts.append(torch.as_tensor(forward_match["target_depth"]))
            source_bearing_parts.append(torch.as_tensor(forward_match["source_bearing"]))
            target_bearing_parts.append(torch.as_tensor(forward_match["target_bearing"]))
        if int(reverse_match["count"]) > 0:
            # Reverse queries are target->source; swap them back to the
            # canonical source-window/target-window ordering for verification.
            source_parts.append(self._anchor_points_from_match(
                source, source_frame_index, torch.as_tensor(reverse_match["target_bearing"]), torch.as_tensor(reverse_match["target_depth"])
            ))
            target_parts.append(self._anchor_points_from_match(
                target, target_frame_index, torch.as_tensor(reverse_match["source_bearing"]), torch.as_tensor(reverse_match["source_depth"])
            ))
            weight_parts.append(torch.as_tensor(reverse_match["weight"]))
            source_depth_parts.append(torch.as_tensor(reverse_match["target_depth"]))
            target_depth_parts.append(torch.as_tensor(reverse_match["source_depth"]))
            source_bearing_parts.append(torch.as_tensor(reverse_match["target_bearing"]))
            target_bearing_parts.append(torch.as_tensor(reverse_match["source_bearing"]))
        source_point = torch.cat(source_parts, dim=0)
        target_point = torch.cat([value.to(source_point) for value in target_parts], dim=0)
        weight = torch.cat([value.to(source_point) for value in weight_parts], dim=0)
        source_depth = torch.cat([value.to(source_point) for value in source_depth_parts], dim=0)
        target_depth = torch.cat([value.to(source_point) for value in target_depth_parts], dim=0)
        source_bearing = torch.cat([value.to(source_point) for value in source_bearing_parts], dim=0)
        target_bearing = torch.cat([value.to(source_point) for value in target_bearing_parts], dim=0)
        if int(source_point.shape[0]) < self.min_matches:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), 0.0, float("inf"), "too_few_geometric_matches")

        rotation_seed = (
            self.fibonacci_seed
            + 65_537 * int(source.window_id)
            + 4_099 * int(target.window_id)
            + 257 * int(source.frame_ids[source_frame_index])
            + 17 * int(target.frame_ids[target_frame_index])
        ) & 0x7FFFFFFF
        rotation_measurement, angular_inlier, rotation_inlier_ratio, rotation_residual = spherical_rotation_ransac(
            target_bearing,
            source_bearing,
            weight,
            threshold_rad=(
                self.rotation_inlier_threshold
                if self.so3_verification
                else self.coincident_rotation_residual
            ),
            iterations=self.rotation_ransac_iterations,
            seed=rotation_seed,
        )

        source_coverage, total_coverage = spherical_coverage_bins(
            source_bearing[angular_inlier],
            latitude_bins=self.coverage_latitude_bins,
            longitude_bins=self.coverage_longitude_bins,
        )
        target_coverage, _ = spherical_coverage_bins(
            target_bearing[angular_inlier],
            latitude_bins=self.coverage_latitude_bins,
            longitude_bins=self.coverage_longitude_bins,
        )
        coverage = min(source_coverage, target_coverage)

        metadata = {
            "yaw_correlation": yaw_score,
            "source_frame_index": int(source_frame_index),
            "target_frame_index": int(target_frame_index),
            "source_frame_id": int(source.frame_ids[source_frame_index]),
            "target_frame_id": int(target.frame_ids[target_frame_index]),
            "rotation_inlier_ratio": rotation_inlier_ratio,
            "rotation_ransac_residual": rotation_residual,
            "rotation_ransac_iterations": self.rotation_ransac_iterations,
            "rotation_ransac_seed": rotation_seed,
            "rotation_inlier_count": int(angular_inlier.sum().item()),
            "spherical_coverage_bins": int(coverage),
            "source_spherical_coverage_bins": int(source_coverage),
            "target_spherical_coverage_bins": int(target_coverage),
            "total_spherical_coverage_bins": int(total_coverage),
            "num_matches": int(source_point.shape[0]),
            "forward_fibonacci_matches": int(forward_match["count"]),
            "reverse_fibonacci_matches": int(reverse_match["count"]),
            "forward_seed": int(forward_match["seed"]),
            "reverse_seed": int(reverse_match["seed"]),
        }
        if self.so3_verification:
            if rotation_inlier_ratio < self.min_rotation_inlier_ratio:
                return PanoramaLoopVerification(
                    False, None, source.window_id, target.window_id, retrieval_score,
                    yaw_shift, int(source_point.shape[0]), rotation_inlier_ratio,
                    rotation_residual, "insufficient_rotation_inlier_ratio", metadata,
                )
            if int(angular_inlier.sum().item()) < self.min_matches:
                return PanoramaLoopVerification(
                    False, None, source.window_id, target.window_id, retrieval_score,
                    yaw_shift, int(angular_inlier.sum().item()), rotation_inlier_ratio,
                    rotation_residual, "too_few_rotation_inliers", metadata,
                )
            if coverage < self.min_spherical_coverage_bins:
                return PanoramaLoopVerification(
                    False, None, source.window_id, target.window_id, retrieval_score,
                    yaw_shift, int(angular_inlier.sum().item()), rotation_inlier_ratio,
                    rotation_residual, "insufficient_spherical_coverage", metadata,
                )

            forward_count = int(forward_match["count"])
            forward_match = self._select_match_inliers(
                forward_match, angular_inlier[:forward_count]
            )
            reverse_match = self._select_match_inliers(
                reverse_match, angular_inlier[forward_count:]
            )
            source_point = source_point[angular_inlier]
            target_point = target_point[angular_inlier]
            weight = weight[angular_inlier]
            source_depth = source_depth[angular_inlier]
            target_depth = target_depth[angular_inlier]
            source_bearing = source_bearing[angular_inlier]
            target_bearing = target_bearing[angular_inlier]

        # Z_ij maps target-window coordinates into source-window coordinates.
        alignment = self.aligner.align(target_point, source_point, weight)
        measurement = alignment.as_matrix().to(source_point)
        scale, sim3_rotation, _ = sim3_components(measurement)
        source_local_rotation = source.local_poses_c2w[source_frame_index, :3, :3].to(sim3_rotation)
        target_local_rotation = target.local_poses_c2w[target_frame_index, :3, :3].to(sim3_rotation)
        predicted_camera_rotation = (
            source_local_rotation.transpose(0, 1) @ sim3_rotation @ target_local_rotation
        )
        rotation_consistency = rotation_geodesic_distance(
            rotation_measurement, predicted_camera_rotation
        )
        median_depth = torch.cat([source_depth, target_depth]).median().clamp_min(1.0e-8)
        normalized_alignment_residual = float(alignment.residual) / float(
            median_depth.detach().cpu()
        )
        metadata.update(
            {
                "alignment_residual": float(alignment.residual),
                "normalized_alignment_residual": normalized_alignment_residual,
                "alignment_inlier_ratio": float(alignment.inlier_ratio),
                "rotation_consistency_rad": rotation_consistency,
                "rotation_consistency_deg": math.degrees(rotation_consistency),
                "alignment_scale": float(scale.detach().cpu()),
                "verified_num_matches": int(source_point.shape[0]),
            }
        )
        if self.so3_verification:
            if not math.isfinite(rotation_consistency) or rotation_consistency > self.max_rotation_consistency:
                return PanoramaLoopVerification(
                    False, None, source.window_id, target.window_id, retrieval_score,
                    yaw_shift, int(source_point.shape[0]), rotation_inlier_ratio,
                    rotation_residual, "rotation_sim3_inconsistent", metadata,
                )
            if (
                not math.isfinite(normalized_alignment_residual)
                or normalized_alignment_residual > self.max_normalized_alignment_residual
            ):
                return PanoramaLoopVerification(
                    False, None, source.window_id, target.window_id, retrieval_score,
                    yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio),
                    float(alignment.residual), "normalized_alignment_residual_too_large", metadata,
                )
        if alignment.accepted:
            _, _, translation = sim3_components(measurement)
            if float(translation.norm().detach().cpu()) <= self.coincident_translation_threshold and rotation_inlier_ratio >= self.aligner.min_inlier_ratio:
                factor = LoopPoseMeasurement(
                    kind="coincident",
                    source=int(source.window_id),
                    target=int(target.window_id),
                    source_local_pose=source.local_poses_c2w[source_frame_index].detach(),
                    target_local_pose=target.local_poses_c2w[target_frame_index].detach(),
                    measured_source_to_target_rotation=rotation_measurement.detach(),
                    center_weight=max(1.0, float(source_point.shape[0]) * float(alignment.inlier_ratio)),
                    rotation_weight=max(1.0, float(source_point.shape[0]) * rotation_inlier_ratio),
                    edge_type="coincident_panorama" if edge_type == "loop" else edge_type,
                    dcs_phi=self.loop_dcs_phi if edge_type == "loop" else None,
                    metadata=metadata,
                )
                dense_factors = (
                    self._dense_factor_from_match(source, target, source_frame_index, target_frame_index, forward_match, use_depth=False, edge_type="loop_dense_spherical" if edge_type == "loop" else f"{edge_type}_dense_spherical")
                    if int(forward_match["count"]) > 0 else None,
                    self._dense_factor_from_match(target, source, target_frame_index, source_frame_index, reverse_match, use_depth=False, edge_type="loop_dense_spherical" if edge_type == "loop" else f"{edge_type}_dense_spherical")
                    if int(reverse_match["count"]) > 0 else None,
                )
                return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "coincident_panorama", metadata, tuple(value for value in dense_factors if value is not None))
            information = source_point.new_tensor(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5]
            ) * max(1.0, float(source_point.shape[0]) * float(alignment.inlier_ratio))
            factor = LoopPoseMeasurement(
                kind="sim3",
                source=int(source.window_id),
                target=int(target.window_id),
                measurement_target_to_source=measurement.detach(),
                information_diag=information.detach(),
                edge_type=edge_type,
                dcs_phi=self.loop_dcs_phi if edge_type == "loop" else None,
                metadata=metadata,
            )
            dense_factors = (
                self._dense_factor_from_match(source, target, source_frame_index, target_frame_index, forward_match, use_depth=True, edge_type="loop_dense_spherical" if edge_type == "loop" else f"{edge_type}_dense_spherical")
                if int(forward_match["count"]) > 0 else None,
                self._dense_factor_from_match(target, source, target_frame_index, source_frame_index, reverse_match, use_depth=True, edge_type="loop_dense_spherical" if edge_type == "loop" else f"{edge_type}_dense_spherical")
                if int(reverse_match["count"]) > 0 else None,
            )
            return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "sim3", metadata, tuple(value for value in dense_factors if value is not None))

        _, _, rejected_translation = sim3_components(measurement)
        near_coincident = (
            bool(torch.isfinite(rejected_translation).all())
            and float(rejected_translation.norm().detach().cpu())
            <= self.coincident_translation_threshold
        )
        if rotation_inlier_ratio >= self.aligner.min_inlier_ratio and near_coincident:
            factor = LoopPoseMeasurement(
                kind="coincident",
                source=int(source.window_id),
                target=int(target.window_id),
                source_local_pose=source.local_poses_c2w[source_frame_index].detach(),
                target_local_pose=target.local_poses_c2w[target_frame_index].detach(),
                measured_source_to_target_rotation=rotation_measurement.detach(),
                center_weight=max(1.0, float(source_point.shape[0]) * 0.25),
                rotation_weight=max(1.0, float(source_point.shape[0]) * rotation_inlier_ratio),
                edge_type="coincident_panorama" if edge_type == "loop" else edge_type,
                dcs_phi=self.loop_dcs_phi if edge_type == "loop" else None,
                metadata=metadata,
            )
            dense_factors = (
                self._dense_factor_from_match(source, target, source_frame_index, target_frame_index, forward_match, use_depth=False, edge_type="loop_coincident_dense_spherical" if edge_type == "loop" else f"{edge_type}_coincident_dense_spherical")
                if int(forward_match["count"]) > 0 else None,
                self._dense_factor_from_match(target, source, target_frame_index, source_frame_index, reverse_match, use_depth=False, edge_type="loop_coincident_dense_spherical" if edge_type == "loop" else f"{edge_type}_coincident_dense_spherical")
                if int(reverse_match["count"]) > 0 else None,
            )
            return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), rotation_inlier_ratio, rotation_residual, "rotation_only", metadata, tuple(value for value in dense_factors if value is not None))

        return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "geometric_verification_failed", metadata)

    def detect(self, packet: LocalGaussianWindowPacket) -> list[PanoramaLoopVerification]:
        results = []
        accepted = 0
        candidates = self.retrieve(packet)
        if self.so3_retrieval:
            candidates = candidates[: self.max_verified_candidates]
        for candidate in candidates:
            result = self.verify_pair(
                candidate.packet,
                packet,
                retrieval_score=candidate.score,
                edge_type="loop",
                source_frame_index=candidate.source_frame_index,
                target_frame_index=candidate.target_frame_index,
            )
            results.append(result)
            if result.accepted:
                accepted += 1
                if self.so3_retrieval and accepted >= self.max_accepted_loops:
                    break
        return results
