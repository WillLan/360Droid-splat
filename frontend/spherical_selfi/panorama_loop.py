"""Yaw-invariant retrieval and spherical/3D verification for panorama loops."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F

from backend.pano_gs.sim3_graph import CoincidentPanoramaFactor, Sim3GraphEdge
from frontend.pano_vggt.alignment import SubmapAligner
from geometry.sim3 import sim3_components, weighted_umeyama
from geometry.spherical_erp import erp_pixel_to_unit_ray, sample_erp_with_wrap

from .window_packet import LocalGaussianWindowPacket


@dataclass
class PanoramaLoopVerification:
    accepted: bool
    factor: Sim3GraphEdge | CoincidentPanoramaFactor | None
    source_window_id: int
    target_window_id: int
    retrieval_score: float
    yaw_shift_columns: int
    num_matches: int
    inlier_ratio: float
    residual: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    ) -> None:
        self.top_k = max(1, int(top_k))
        self.exclude_recent_windows = max(0, int(exclude_recent_windows))
        self.min_retrieval_score = float(min_retrieval_score)
        self.min_match_cosine = float(min_match_cosine)
        self.min_matches = max(3, int(min_matches))
        self.max_matches = max(self.min_matches, int(max_matches))
        self.coincident_translation_threshold = float(coincident_translation_threshold)
        self.coincident_rotation_residual = math.radians(float(coincident_rotation_residual_deg))
        self.aligner = SubmapAligner(
            align_mode="sim3",
            max_residual=float(max_alignment_residual),
            min_inlier_ratio=float(min_inlier_ratio),
            max_scale_change=float(max_scale_change),
            min_points=self.min_matches,
            return_rejected_transform=True,
        )
        self.memory: list[LocalGaussianWindowPacket] = []

    def add(self, packet: LocalGaussianWindowPacket) -> None:
        self.memory.append(packet)

    def retrieve(self, packet: LocalGaussianWindowPacket) -> list[tuple[LocalGaussianWindowPacket, float]]:
        if not self.memory:
            return []
        cutoff = max(0, len(self.memory) - self.exclude_recent_windows)
        candidates = self.memory[:cutoff]
        if not candidates:
            return []
        query = packet.retrieval_descriptors[0].detach().float()
        scored = []
        for candidate in candidates:
            descriptor = candidate.retrieval_descriptors[0].to(query).float()
            score = float(F.cosine_similarity(query, descriptor, dim=0).detach().cpu())
            if score >= self.min_retrieval_score:
                scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.top_k]

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
        source_feature = source.verification_features[0, source_frame_index]
        target_feature = target.verification_features[0, target_frame_index].to(source_feature)
        yaw_shift, yaw_score = circular_yaw_shift(source_feature, target_feature)
        try:
            source_index, target_index, match_weight = self._mutual_matches(source_feature, target_feature)
        except ValueError as exc:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, 0, 0.0, float("inf"), f"feature_mismatch:{exc}")
        if int(source_index.numel()) < self.min_matches:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_index.numel()), 0.0, float("inf"), "too_few_mutual_matches")

        source_point, source_confidence, source_valid = self._points_for_matches(
            source, source_frame_index, source_index
        )
        target_point, target_confidence, target_valid = self._points_for_matches(
            target, target_frame_index, target_index
        )
        target_point = target_point.to(source_point)
        target_confidence = target_confidence.to(source_point)
        target_valid = target_valid.to(device=source_point.device)
        match_weight = match_weight.to(source_point)
        valid = source_valid & target_valid & torch.isfinite(match_weight)
        weight = match_weight * source_confidence.to(source_point) * target_confidence
        valid &= weight > 0.0
        source_point, target_point, weight = source_point[valid], target_point[valid], weight[valid]
        source_index, target_index = source_index[valid], target_index[valid]
        if int(source_point.shape[0]) < self.min_matches:
            return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), 0.0, float("inf"), "too_few_geometric_matches")

        # Z_ij maps target-window coordinates into source-window coordinates.
        alignment = self.aligner.align(target_point, source_point, weight)
        measurement = alignment.as_matrix().to(source_point)

        feature_h, feature_w = tuple(int(v) for v in source_feature.shape[-2:])
        source_uv = self._verification_uv(source_index, feature_h, feature_w).to(source_point)
        target_uv = self._verification_uv(target_index, feature_h, feature_w).to(source_point)
        source_bearing = erp_pixel_to_unit_ray(source_uv, feature_h, feature_w)
        target_bearing = erp_pixel_to_unit_ray(target_uv, feature_h, feature_w)
        rotation_transform = weighted_umeyama(target_bearing, source_bearing, weight, allow_scale=False)
        _, rotation_measurement, _ = sim3_components(rotation_transform)
        rotated = torch.einsum("ij,nj->ni", rotation_measurement, target_bearing)
        angular = torch.atan2(
            torch.cross(source_bearing, rotated, dim=-1).norm(dim=-1),
            (source_bearing * rotated).sum(dim=-1).clamp(-1.0, 1.0),
        )
        angular_inlier = angular <= self.coincident_rotation_residual
        rotation_inlier_ratio = float(angular_inlier.float().mean().detach().cpu())

        metadata = {
            "yaw_correlation": yaw_score,
            "rotation_inlier_ratio": rotation_inlier_ratio,
            "alignment_residual": float(alignment.residual),
            "alignment_inlier_ratio": float(alignment.inlier_ratio),
            "num_matches": int(source_point.shape[0]),
        }
        if alignment.accepted:
            _, _, translation = sim3_components(measurement)
            if float(translation.norm().detach().cpu()) <= self.coincident_translation_threshold and rotation_inlier_ratio >= self.aligner.min_inlier_ratio:
                factor: Sim3GraphEdge | CoincidentPanoramaFactor = CoincidentPanoramaFactor(
                    source=int(source.window_id),
                    target=int(target.window_id),
                    source_local_pose=source.local_poses_c2w[source_frame_index].detach(),
                    target_local_pose=target.local_poses_c2w[target_frame_index].detach(),
                    measured_source_to_target_rotation=rotation_measurement.detach(),
                    center_weight=max(1.0, float(source_point.shape[0]) * float(alignment.inlier_ratio)),
                    rotation_weight=max(1.0, float(source_point.shape[0]) * rotation_inlier_ratio),
                    edge_type="coincident_panorama" if edge_type == "loop" else edge_type,
                    metadata=metadata,
                )
                return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "coincident_panorama", metadata)
            information = source_point.new_tensor(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5]
            ) * max(1.0, float(source_point.shape[0]) * float(alignment.inlier_ratio))
            factor = Sim3GraphEdge(
                source=int(source.window_id),
                target=int(target.window_id),
                measurement_target_to_source=measurement.detach(),
                information_diag=information.detach(),
                edge_type=edge_type,
                metadata=metadata,
            )
            return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "sim3", metadata)

        _, _, rejected_translation = sim3_components(measurement)
        near_coincident = (
            bool(torch.isfinite(rejected_translation).all())
            and float(rejected_translation.norm().detach().cpu())
            <= self.coincident_translation_threshold
        )
        if rotation_inlier_ratio >= self.aligner.min_inlier_ratio and near_coincident:
            factor = CoincidentPanoramaFactor(
                source=int(source.window_id),
                target=int(target.window_id),
                source_local_pose=source.local_poses_c2w[source_frame_index].detach(),
                target_local_pose=target.local_poses_c2w[target_frame_index].detach(),
                measured_source_to_target_rotation=rotation_measurement.detach(),
                center_weight=max(1.0, float(source_point.shape[0]) * 0.25),
                rotation_weight=max(1.0, float(source_point.shape[0]) * rotation_inlier_ratio),
                metadata=metadata,
            )
            return PanoramaLoopVerification(True, factor, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), rotation_inlier_ratio, float(angular[angular_inlier].mean().detach().cpu()) if bool(angular_inlier.any()) else float("inf"), "rotation_only", metadata)

        return PanoramaLoopVerification(False, None, source.window_id, target.window_id, retrieval_score, yaw_shift, int(source_point.shape[0]), float(alignment.inlier_ratio), float(alignment.residual), "geometric_verification_failed", metadata)

    def detect(self, packet: LocalGaussianWindowPacket) -> list[PanoramaLoopVerification]:
        results = []
        for candidate, score in self.retrieve(packet):
            result = self.verify_pair(candidate, packet, retrieval_score=score, edge_type="loop")
            results.append(result)
        return results
