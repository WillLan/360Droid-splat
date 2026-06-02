"""Online PanoVGGT long-sequence frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image

from .alignment import SimilarityTransform, SubmapAligner, sample_overlap_points
from .engine import PanoVGGTInferenceEngine, build_panovggt_engine
from .loop import FrontendPoseGraph, LoopManager, PoseGraphEdge
from .types import PanoVGGTLocalPrediction


@dataclass
class _FrameRecord:
    frame: PanoFrame
    image: torch.Tensor


def _relative_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


def _resize_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(field.shape[-2:]) == tuple(size):
        return field
    return F.interpolate(field.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]


class PanoVGGTLongTracker(PanoDROIDFrontend):
    """Chunked PanoVGGT tracker with overlap alignment and delayed emission."""

    def __init__(
        self,
        *,
        engine: PanoVGGTInferenceEngine | None = None,
        engine_config: dict | None = None,
        device: Optional[str] = None,
        chunk_size: int = 8,
        overlap: int = 4,
        emit_delay: int = 2,
        align_mode: str = "sim3",
        keyframe_threshold: float = 0.55,
        force_keyframe_interval: int = 10,
        loop_enable: bool = False,
        max_alignment_points: int = 4096,
        max_alignment_cache_frames: int = 128,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = min(max(0, int(overlap)), max(0, self.chunk_size - 1))
        self.emit_delay = max(0, int(emit_delay))
        self.keyframe_threshold = float(keyframe_threshold)
        self.force_keyframe_interval = int(force_keyframe_interval)
        self.max_alignment_points = int(max_alignment_points)
        self.max_alignment_cache_frames = int(max_alignment_cache_frames)
        self.engine = engine or build_panovggt_engine(engine_config or {}, device=self.device)
        self.aligner = SubmapAligner(align_mode=align_mode)
        self.pose_graph = FrontendPoseGraph()
        self.loop_manager = LoopManager(enabled=loop_enable)
        self.reset()

    def initialize(self, sequence_meta: dict) -> None:
        device = sequence_meta.get("device")
        if device is not None:
            self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.records: list[_FrameRecord] = []
        self.next_chunk_start = 0
        self.processed_ranges: set[tuple[int, int, int]] = set()
        self.pending_outputs: list[FrontendOutput] = []
        self.emitted_frame_ids: set[int] = set()
        self.pose_by_frame: dict[int, torch.Tensor] = {}
        self.depth_by_frame: dict[int, torch.Tensor] = {}
        self.conf_by_frame: dict[int, torch.Tensor] = {}
        self.global_points_by_frame: dict[int, torch.Tensor] = {}
        self.backend_pose_overrides: dict[int, torch.Tensor] = {}
        self.last_keyframe_id: Optional[int] = None
        self.chunk_count = 0

    def load_checkpoint(self, path: str) -> None:
        self.engine.load_checkpoint(path)

    def apply_backend_pose_updates(self, updates: dict[int, torch.Tensor]) -> None:
        """Inject refined backend keyframe poses for future chunk alignment."""

        for frame_id, pose in updates.items():
            pose_cpu = pose.detach().cpu().float()
            self.backend_pose_overrides[int(frame_id)] = pose_cpu
            self.pose_by_frame[int(frame_id)] = pose_cpu.to(self.device)

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = ensure_chw_image(frame.image).to(self.device)
        self.records.append(_FrameRecord(frame=frame, image=image.detach()))
        self._run_ready_chunks(final=False)
        return self._placeholder_output(frame)

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        out = self.pending_outputs
        self.pending_outputs = []
        return out

    def flush(self) -> list[FrontendOutput]:
        self._run_ready_chunks(final=True)
        return self.pop_ready_outputs()

    def _placeholder_output(self, frame: PanoFrame) -> FrontendOutput:
        pose = self.pose_by_frame.get(int(frame.frame_id))
        if pose is None:
            pose = torch.eye(4)
        return FrontendOutput(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pose_c2w=pose.detach().cpu(),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=False,
            ba_residual=None,
            tracking_status="buffering_panovggt_long",
        )

    def _run_ready_chunks(self, *, final: bool) -> None:
        while self.records:
            available = len(self.records) - self.next_chunk_start
            if available <= 0:
                break
            if not final and available < self.chunk_size:
                break
            start = self.next_chunk_start
            end = min(len(self.records), start + self.chunk_size)
            if end <= start:
                break
            self._process_chunk(start, end, final=final)
            if final:
                self.next_chunk_start = end
                if self.next_chunk_start >= len(self.records):
                    break
            else:
                self.next_chunk_start = max(end - self.overlap, start + 1)
                self._prune_records()
                if len(self.records) - self.next_chunk_start < self.chunk_size:
                    break

    def _process_chunk(self, start: int, end: int, *, final: bool) -> None:
        frame_ids = tuple(int(r.frame.frame_id) for r in self.records[start:end])
        range_key = (frame_ids[0], frame_ids[-1], len(frame_ids))
        if range_key in self.processed_ranges:
            return
        self.processed_ranges.add(range_key)

        images = torch.stack([r.image for r in self.records[start:end]], dim=0).to(self.device)
        pred = self.engine.infer(images)
        transform = self._align_chunk(pred, frame_ids)
        backend_correction = self._backend_feedback_correction(pred, frame_ids, transform)
        chunk_descriptor = self._chunk_descriptor(pred, images)
        loop_target = self.loop_manager.add_chunk(chunk_descriptor)
        if loop_target is not None:
            self.pose_graph.add_edge(
                PoseGraphEdge(
                    source_chunk=self.chunk_count,
                    target_chunk=int(loop_target),
                    transform=transform,
                    residual=transform.residual,
                    edge_type="loop",
                )
            )

        output_data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]] = {}
        for local_idx, record in enumerate(self.records[start:end]):
            frame_id = int(record.frame.frame_id)
            image_size = tuple(int(v) for v in record.image.shape[-2:])
            depth_local = pred.depth[local_idx].to(self.device)
            depth_global = (depth_local * float(transform.scale)).clamp_min(1e-6)
            inv_full = _resize_field(depth_global.reciprocal(), image_size).detach()
            conf_full = _resize_field(pred.confidence[local_idx].to(self.device), image_size).detach()
            if not transform.accepted:
                conf_full = 0.5 * conf_full
            pose = transform.apply_pose(pred.poses_c2w[local_idx].to(self.device)).detach()
            points = transform.apply_points(pred.point_maps[local_idx].to(self.device)).detach()
            if backend_correction is not None:
                pose = (backend_correction @ pose).detach()
                points = self._apply_pose_correction_to_points(points, backend_correction).detach()

            self.pose_by_frame[frame_id] = pose
            self.depth_by_frame[frame_id] = inv_full
            self.conf_by_frame[frame_id] = conf_full
            self.global_points_by_frame[frame_id] = points
            output_data[frame_id] = (
                pose,
                inv_full,
                conf_full,
                transform.residual,
                "tracked_panovggt_long" if transform.accepted else "tracked_panovggt_long_unaligned",
            )

        emit_end = end if final else max(start, end - self.emit_delay)
        for record in self.records[start:emit_end]:
            frame_id = int(record.frame.frame_id)
            if frame_id in self.emitted_frame_ids:
                continue
            data = output_data.get(frame_id)
            if data is not None:
                pose, inverse_depth, confidence, residual, status = data
                self.pending_outputs.append(
                    self._make_output(
                        record.frame,
                        pose=pose,
                        inverse_depth=inverse_depth,
                        confidence=confidence,
                        residual=residual,
                        status=status,
                    )
                )
                self.emitted_frame_ids.add(frame_id)
        self.chunk_count += 1
        self._trim_alignment_cache()

    def _backend_feedback_correction(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        transform: SimilarityTransform,
    ) -> torch.Tensor | None:
        for local_idx, frame_id in enumerate(frame_ids):
            refined = self.backend_pose_overrides.get(int(frame_id))
            if refined is None:
                continue
            pose = transform.apply_pose(pred.poses_c2w[local_idx].to(self.device)).detach()
            target = refined.to(device=self.device, dtype=pose.dtype)
            if target.shape == (4, 4) and torch.isfinite(target).all():
                return target @ torch.linalg.inv(pose)
        return None

    @staticmethod
    def _apply_pose_correction_to_points(points: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, 3)
        hom = torch.cat([flat, torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)], dim=-1)
        corrected = (correction.to(device=flat.device, dtype=flat.dtype) @ hom.T).T[:, :3]
        return corrected.reshape_as(points)

    def _align_chunk(self, pred: PanoVGGTLocalPrediction, frame_ids: tuple[int, ...]) -> SimilarityTransform:
        if not self.global_points_by_frame:
            transform = SimilarityTransform.identity(device=pred.depth.device, dtype=pred.depth.dtype)
            self.pose_graph.add_edge(
                PoseGraphEdge(
                    source_chunk=self.chunk_count,
                    target_chunk=self.chunk_count,
                    transform=transform,
                    residual=0.0,
                    edge_type="root",
                )
            )
            return transform

        source_parts = []
        target_parts = []
        weight_parts = []
        for local_idx, frame_id in enumerate(frame_ids):
            target = self.global_points_by_frame.get(int(frame_id))
            if target is None:
                continue
            source = pred.point_maps[local_idx].to(target.device)
            conf = pred.confidence[local_idx, 0].to(target.device)
            src, tgt, weights = sample_overlap_points(
                source,
                target,
                conf,
                None,
                max_points=max(1, self.max_alignment_points // max(1, self.overlap)),
            )
            if src.numel() == 0:
                continue
            source_parts.append(src)
            target_parts.append(tgt)
            weight_parts.append(weights)
        if not source_parts:
            return SimilarityTransform.identity(device=pred.depth.device, dtype=pred.depth.dtype)
        source_all = torch.cat(source_parts, dim=0)
        target_all = torch.cat(target_parts, dim=0)
        weight_all = torch.cat(weight_parts, dim=0)
        transform = self.aligner.align(source_all, target_all, weight_all)
        self.pose_graph.add_edge(
            PoseGraphEdge(
                source_chunk=self.chunk_count,
                target_chunk=max(0, self.chunk_count - 1),
                transform=transform,
                residual=transform.residual,
                edge_type="sequential",
            )
        )
        return transform

    def _make_output(
        self,
        frame: PanoFrame,
        *,
        pose: torch.Tensor,
        inverse_depth: torch.Tensor,
        confidence: torch.Tensor,
        residual: float,
        status: str,
    ) -> FrontendOutput:
        frame_id = int(frame.frame_id)
        prev_pose = self.pose_by_frame.get(frame_id - 1)
        relative = None if prev_pose is None else _relative_from_c2w(prev_pose.unsqueeze(0), pose.unsqueeze(0))[0]
        key_score = float(confidence.mean().detach().cpu())
        gap = (
            frame_id - int(self.last_keyframe_id)
            if self.last_keyframe_id is not None
            else self.force_keyframe_interval
        )
        is_keyframe = key_score >= self.keyframe_threshold or gap >= self.force_keyframe_interval
        if is_keyframe:
            self.last_keyframe_id = frame_id
        return FrontendOutput(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pose_c2w=pose.detach().cpu(),
            relative_pose=relative.detach().cpu() if relative is not None else None,
            pose_confidence=key_score,
            inverse_depth=inverse_depth.detach().cpu(),
            depth_confidence=confidence.detach().cpu(),
            spherical_flow=None,
            keyframe_score=key_score,
            is_keyframe=bool(is_keyframe),
            ba_residual=float(residual),
            tracking_status=status,
        )

    def _chunk_descriptor(self, pred: PanoVGGTLocalPrediction, images: torch.Tensor) -> torch.Tensor:
        if pred.descriptors is not None and pred.descriptors.numel() > 0:
            return pred.descriptors.float().mean(dim=0)
        return torch.cat(
            [images.mean(dim=(0, 2, 3)), images.std(dim=(0, 2, 3), unbiased=False)],
            dim=0,
        )

    def _prune_records(self) -> None:
        if self.next_chunk_start <= 0:
            return
        drop_count = min(self.next_chunk_start, max(0, len(self.records) - self.overlap))
        if drop_count <= 0:
            return
        self.records = self.records[drop_count:]
        self.next_chunk_start -= drop_count

    def _trim_alignment_cache(self) -> None:
        if self.max_alignment_cache_frames <= 0:
            return
        if len(self.global_points_by_frame) <= self.max_alignment_cache_frames:
            return
        live = {int(r.frame.frame_id) for r in self.records}
        for frame_id in sorted(self.global_points_by_frame):
            if len(self.global_points_by_frame) <= self.max_alignment_cache_frames:
                break
            if frame_id in live:
                continue
            self.global_points_by_frame.pop(frame_id, None)


def build_panovggt_frontend_from_config(config: dict) -> PanoVGGTLongTracker:
    frontend_cfg = config.get("Frontend", {})
    pano_cfg = config.get("PanoVGGT", {})
    return PanoVGGTLongTracker(
        engine_config=pano_cfg,
        chunk_size=int(pano_cfg.get("chunk_size", 8)),
        overlap=int(pano_cfg.get("overlap", 4)),
        emit_delay=int(pano_cfg.get("emit_delay", 2)),
        align_mode=str(pano_cfg.get("align_mode", "sim3")),
        keyframe_threshold=float(frontend_cfg.get("keyframe_threshold", 0.55)),
        force_keyframe_interval=int(frontend_cfg.get("force_keyframe_interval", 10)),
        loop_enable=bool(pano_cfg.get("loop_enable", False)),
        max_alignment_points=int(pano_cfg.get("max_alignment_points", 4096)),
        max_alignment_cache_frames=int(pano_cfg.get("max_alignment_cache_frames", 128)),
    )
