"""Online PanoVGGT long-sequence frontend."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Optional

import torch
import torch.nn.functional as F

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image

from .alignment import SimilarityTransform, SubmapAligner, sample_overlap_points
from .dense_ba_refiner import DenseBARefinerStats, PanoVGGTDenseBARefiner
from .dense_matcher import PoseGuidedDenseMatcher
from .engine import PanoVGGTInferenceEngine, build_panovggt_engine
from .keyframe_graph_refiner import KeyframeGraphBAStats, PanoVGGTKeyframeGraphRefiner
from .keyframe_memory import KeyframeCorrespondenceGraph, KeyframeMemory, KeyframeRecord
from .loop import FrontendPoseGraph, LoopManager, PoseGraphEdge
from .m3_config import parse_m3_sphere_config
from .types import PanoVGGTLocalPrediction


class PanoVGGTAlignmentError(RuntimeError):
    """Raised when a PanoVGGT chunk cannot be aligned into the global map."""


@dataclass
class _FrameRecord:
    frame: PanoFrame
    image: torch.Tensor


@dataclass
class _KeyframeAnchorRecord:
    frame_id: int
    image: torch.Tensor
    pose_c2w: torch.Tensor


@dataclass
class _ChunkAnchorContext:
    record: _KeyframeAnchorRecord
    full_index: int
    prepended: bool
    current_full_indices: tuple[int, ...]


@dataclass
class _AnchorFrameMetrics:
    anchor_frame_id: int
    frame_mean_pair_conf: float
    low_pair_conf_ratio: float
    match_coverage: float
    pair_conf_quantiles: dict[str, float]
    pair_confidence: torch.Tensor
    low_pair_conf: torch.Tensor
    matched_cells: torch.Tensor
    non_sky: torch.Tensor
    anchor_pose_c2w: torch.Tensor


@dataclass
class _JointInferenceContext:
    history_records: tuple[KeyframeRecord, ...]
    history_frame_ids: tuple[int, ...]
    history_count: int
    current_frame_ids: tuple[int, ...]


@dataclass
class _AlignmentDebug:
    overlap_points: int = 0
    history_points: int = 0
    history_ids: tuple[int, ...] = ()
    scale: float = 1.0
    residual: float = 0.0
    inlier_ratio: float = 0.0
    forced_accepted: bool = False


def _relative_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


def _resize_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(field.shape[-2:]) == tuple(size):
        return field
    return F.interpolate(field.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]


def _resize_nearest(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(field.shape[-2:]) == tuple(size):
        return field
    return F.interpolate(field.unsqueeze(0).float(), size=size, mode="nearest")[0]


def _resize_scalar_map(field: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    if field is None:
        return None
    tensor = field.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if int(tensor.shape[0]) != 1:
            tensor = tensor[:1]
    else:
        return None
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor[0]
    return _resize_field(tensor, size)[0]


def _resize_points(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(points.shape[:2]) == tuple(size):
        return points
    return F.interpolate(
        points.permute(2, 0, 1).unsqueeze(0).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)


def _slice_prediction(pred: PanoVGGTLocalPrediction, start: int, count: int) -> PanoVGGTLocalPrediction:
    n = int(pred.poses_c2w.shape[0])
    start_i = max(0, min(int(start), n))
    count_i = max(0, min(int(count), n - start_i))
    idx = slice(start_i, start_i + count_i)
    return replace(
        pred,
        poses_c2w=pred.poses_c2w[idx],
        depth=pred.depth[idx],
        confidence=pred.confidence[idx],
        chunk_world_points=pred.chunk_world_points[idx],
        local_points=None if pred.local_points is None else pred.local_points[idx],
        global_points=None if pred.global_points is None else pred.global_points[idx],
        descriptors=None if pred.descriptors is None else pred.descriptors[idx],
        dense_descriptors=None if pred.dense_descriptors is None else pred.dense_descriptors[idx],
        match_confidence=None if pred.match_confidence is None else pred.match_confidence[idx],
        static_confidence=None if pred.static_confidence is None else pred.static_confidence[idx],
        sky_logits=None if pred.sky_logits is None else pred.sky_logits[idx],
        sky_prob=None if pred.sky_prob is None else pred.sky_prob[idx],
    )


def _feature_uv_to_grid(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    norm_x = 2.0 * (uv[..., 0] - 0.5) / max(width - 1, 1) - 1.0
    norm_y = 2.0 * (uv[..., 1] - 0.5) / max(height - 1, 1) - 1.0
    return torch.stack([norm_x, norm_y], dim=-1)


def _sample_feature_map(map_tensor: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if map_tensor.ndim != 4 or int(map_tensor.shape[0]) != 1:
        raise ValueError(f"map_tensor must have shape 1xCxHxW, got {tuple(map_tensor.shape)}.")
    height, width = int(map_tensor.shape[-2]), int(map_tensor.shape[-1])
    selected = map_tensor.expand(int(uv.shape[0]), -1, -1, -1)
    grid = _feature_uv_to_grid(uv.to(device=map_tensor.device, dtype=map_tensor.dtype), height, width).view(-1, 1, 1, 2)
    sampled = F.grid_sample(selected, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, 0, 0]


def _scatter_feature_values(
    uv: torch.Tensor,
    values: torch.Tensor,
    *,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    out = torch.zeros(1, int(feature_hw[0]), int(feature_hw[1]), device=values.device, dtype=values.dtype)
    if uv.numel() == 0:
        return out
    x = (uv[:, 0] - 0.5).round().long().clamp(0, int(feature_hw[1]) - 1)
    y = (uv[:, 1] - 0.5).round().long().clamp(0, int(feature_hw[0]) - 1)
    out[0, y, x] = values.reshape(-1).to(out)
    return out


def _scatter_feature_values_and_mask(
    uv: torch.Tensor,
    values: torch.Tensor,
    *,
    feature_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros(1, int(feature_hw[0]), int(feature_hw[1]), device=values.device, dtype=values.dtype)
    seen = torch.zeros(1, int(feature_hw[0]), int(feature_hw[1]), device=values.device, dtype=torch.bool)
    if uv.numel() == 0:
        return out, seen
    x = (uv[:, 0] - 0.5).round().long().clamp(0, int(feature_hw[1]) - 1)
    y = (uv[:, 1] - 0.5).round().long().clamp(0, int(feature_hw[0]) - 1)
    out[0, y, x] = values.reshape(-1).to(out)
    seen[0, y, x] = True
    return out, seen


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    vals = values.detach().float().reshape(-1).cpu()
    qs = torch.quantile(vals, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
    }


def _limit_mask_per_cell(
    mask: torch.Tensor,
    score: torch.Tensor,
    *,
    cell_size: int,
    max_per_cell: int,
) -> torch.Tensor:
    if int(cell_size) <= 0 or int(max_per_cell) <= 0:
        return mask.bool()
    mask_t = mask.bool()
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.ndim != 3 or int(mask_t.shape[0]) != 1:
        raise ValueError(f"mask must have shape HxW or 1xHxW, got {tuple(mask.shape)}.")
    score_t = score.detach().float()
    if score_t.ndim == 2:
        score_t = score_t.unsqueeze(0)
    if tuple(score_t.shape) != tuple(mask_t.shape):
        raise ValueError(f"score shape {tuple(score.shape)} does not match mask shape {tuple(mask_t.shape)}.")
    _, height, width = mask_t.shape
    out = torch.zeros_like(mask_t)
    cell = max(1, int(cell_size))
    k = max(1, int(max_per_cell))
    for y0 in range(0, int(height), cell):
        y1 = min(int(height), y0 + cell)
        for x0 in range(0, int(width), cell):
            x1 = min(int(width), x0 + cell)
            ys, xs = torch.nonzero(mask_t[0, y0:y1, x0:x1], as_tuple=True)
            if ys.numel() == 0:
                continue
            if ys.numel() > k:
                cell_scores = score_t[0, y0 + ys, x0 + xs]
                _, order = torch.topk(cell_scores, k=k, largest=True)
                ys = ys.index_select(0, order)
                xs = xs.index_select(0, order)
            out[0, y0 + ys, x0 + xs] = True
    return out


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
        min_overlap_points: int = 4096,
        max_align_rmse: float = 0.25,
        min_inlier_ratio: float = 0.35,
        max_scale_jump: float = 2.0,
        force_accept_alignment: bool = True,
        require_aligned_world_points: bool = True,
        emit_unaligned: bool = False,
        novel_insertion_enabled: bool = False,
        novel_pair_conf_insert_threshold: float = 0.0,
        novel_insert_confidence_floor: float = 0.0,
        novel_spatial_cell_size: int = 0,
        novel_max_seeds_per_cell: int = 0,
        novel_insertion_strategy: str = "legacy",
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = min(max(0, int(overlap)), max(0, self.chunk_size - 1))
        self.emit_delay = max(0, int(emit_delay))
        self.keyframe_threshold = float(keyframe_threshold)
        self.force_keyframe_interval = int(force_keyframe_interval)
        self.max_alignment_points = int(max_alignment_points)
        self.max_alignment_cache_frames = int(max_alignment_cache_frames)
        self.min_overlap_points = int(min_overlap_points)
        self.force_accept_alignment = bool(force_accept_alignment)
        self.require_aligned_world_points = bool(require_aligned_world_points)
        self.emit_unaligned = bool(emit_unaligned)
        self.novel_insertion_enabled = bool(novel_insertion_enabled)
        self.novel_pair_conf_insert_threshold = float(novel_pair_conf_insert_threshold)
        self.novel_insert_confidence_floor = float(novel_insert_confidence_floor)
        self.novel_spatial_cell_size = max(0, int(novel_spatial_cell_size))
        self.novel_max_seeds_per_cell = max(0, int(novel_max_seeds_per_cell))
        self.novel_insertion_strategy = str(novel_insertion_strategy or "legacy").lower()
        self.engine = engine or build_panovggt_engine(engine_config or {}, device=self.device)
        self.m3_config = parse_m3_sphere_config({"PanoVGGT": engine_config or {}})
        self.dense_ba_refiner = PanoVGGTDenseBARefiner(self.m3_config)
        self.keyframe_graph_refiner = PanoVGGTKeyframeGraphRefiner(self.m3_config)
        self.aligner = SubmapAligner(
            align_mode=align_mode,
            max_residual=float(max_align_rmse),
            min_inlier_ratio=float(min_inlier_ratio),
            max_scale_change=float(max_scale_jump),
            min_points=max(3, int(min_overlap_points)),
            return_rejected_transform=self.force_accept_alignment,
        )
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
        self.sky_mask_by_frame: dict[int, torch.Tensor] = {}
        self.global_points_by_frame: dict[int, torch.Tensor] = {}
        self.backend_pose_overrides: dict[int, torch.Tensor] = {}
        memory_keyframes = max(1, int(self.m3_config.dense_ba.history_keyframes))
        if self.keyframe_graph_enabled:
            memory_keyframes = max(
                memory_keyframes,
                int(self.m3_config.keyframe_graph.window_keyframes) + int(self.m3_config.keyframe_graph.adjacent_history) + 1,
            )
        self.keyframe_memory = KeyframeMemory(max_keyframes=memory_keyframes)
        graph_cfg = self.m3_config.keyframe_graph
        graph_max_edges = max(16, int(graph_cfg.window_keyframes) * max(1, int(graph_cfg.adjacent_history) + 1) * 4)
        self.keyframe_correspondence_graph = KeyframeCorrespondenceGraph(max_edges=graph_max_edges)
        self.pending_keyframe_graph_pose_updates: dict[int, torch.Tensor] = {}
        self.last_keyframe_id: Optional[int] = None
        self.last_keyframe_anchor: _KeyframeAnchorRecord | None = None
        self.chunk_count = 0
        self.last_dense_ba_stats: DenseBARefinerStats | None = None
        self.dense_ba_stats_history: list[DenseBARefinerStats] = []
        self.last_keyframe_graph_stats: KeyframeGraphBAStats | None = None
        self.keyframe_graph_stats_history: list[KeyframeGraphBAStats] = []
        self.last_m3_debug: dict | None = None
        self.last_profile: dict | None = None
        self.last_alignment_debug = _AlignmentDebug()
        self.current_recent_history_ids: tuple[int, ...] = ()
        self.keyframe_decision_history: list[dict] = []
        self._pending_keyframe_decisions: list[dict] = []
        self._pending_insertion_hints: dict[int, dict[str, torch.Tensor]] = {}

    @property
    def keyframe_anchor_enabled(self) -> bool:
        return bool(self.m3_config.enabled and self.m3_config.keyframe_anchor.enabled)

    @property
    def dense_ba_history_enabled(self) -> bool:
        mode = str(self.m3_config.dense_ba.mode).lower()
        return bool(self.m3_config.enabled and self.m3_config.dense_ba.enabled and mode in {"history_window", "keyframe_graph"})

    @property
    def joint_inference_enabled(self) -> bool:
        return bool(self.m3_config.enabled and self.m3_config.joint_inference.enabled)

    @property
    def keyframe_graph_enabled(self) -> bool:
        return bool(self.m3_config.enabled and self.m3_config.keyframe_graph.enabled)

    def load_checkpoint(self, path: str) -> None:
        self.engine.load_checkpoint(path)

    def apply_backend_pose_updates(
        self,
        updates: dict[int, torch.Tensor],
        *,
        update_last_keyframe_anchor: bool = True,
    ) -> None:
        """Inject refined backend keyframe poses for future chunk alignment."""

        for frame_id, pose in updates.items():
            pose_cpu = pose.detach().cpu().float()
            fid = int(frame_id)
            self.backend_pose_overrides[fid] = pose_cpu
            self.pose_by_frame[fid] = pose_cpu.to(self.device)
            for record in self.keyframe_memory.records:
                if int(record.frame_id) == fid:
                    record.pose_c2w = pose_cpu
            if (
                update_last_keyframe_anchor
                and self.last_keyframe_anchor is not None
                and int(self.last_keyframe_anchor.frame_id) == fid
            ):
                self.last_keyframe_anchor.pose_c2w = pose_cpu

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = ensure_chw_image(frame.image).to(self.device)
        self.records.append(_FrameRecord(frame=frame, image=image.detach()))
        self._run_ready_chunks(final=False)
        return self._placeholder_output(frame)

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        out = self.pending_outputs
        self.pending_outputs = []
        return out

    def pop_keyframe_decisions(self) -> list[dict]:
        out = self._pending_keyframe_decisions
        self._pending_keyframe_decisions = []
        return out

    def pop_keyframe_graph_pose_updates(self) -> dict[int, torch.Tensor]:
        out = {
            int(frame_id): pose.detach().cpu().float()
            for frame_id, pose in self.pending_keyframe_graph_pose_updates.items()
        }
        self.pending_keyframe_graph_pose_updates = {}
        return out

    def consume_insertion_hints(self, frame_id: int) -> dict[str, torch.Tensor] | None:
        return self._pending_insertion_hints.pop(int(frame_id), None)

    def sky_mask_for_frame(
        self,
        frame_id: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        mask = self.sky_mask_by_frame.get(int(frame_id))
        if mask is None:
            return None
        out = mask.detach().cpu().bool()
        if image_size is not None and tuple(out.shape[-2:]) != tuple(int(v) for v in image_size):
            out = (_resize_nearest(out.float(), (int(image_size[0]), int(image_size[1]))) > 0.5).bool()
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
            world_points=None,
            world_points_confidence=None,
            valid_world_points_mask=None,
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

    def _recent_joint_history_records(self, frame_ids: tuple[int, ...]) -> tuple[KeyframeRecord, ...]:
        if not self.joint_inference_enabled:
            return ()
        cfg = self.m3_config.joint_inference
        if str(cfg.history_policy).lower() != "recent":
            return ()
        max_history = max(0, int(cfg.max_history_frames))
        if max_history <= 0:
            return ()
        current_ids = {int(fid) for fid in frame_ids}
        records: list[KeyframeRecord] = []
        for record in reversed(self.keyframe_memory.records):
            if int(record.frame_id) in current_ids:
                continue
            if record.image is None:
                continue
            records.append(record)
            if len(records) >= max_history:
                break
        return tuple(reversed(records))

    def _build_joint_inference_batch(
        self,
        images: torch.Tensor,
        frame_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, _JointInferenceContext]:
        records = self._recent_joint_history_records(frame_ids)
        if not records:
            return images, _JointInferenceContext((), (), 0, frame_ids)
        history_images = []
        size = tuple(int(v) for v in images.shape[-2:])
        for record in records:
            image = record.image.to(device=images.device, dtype=images.dtype)
            if tuple(image.shape[-2:]) != size:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            history_images.append(image)
        joint_images = torch.cat([torch.stack(history_images, dim=0), images], dim=0)
        history_ids = tuple(int(record.frame_id) for record in records)
        return joint_images, _JointInferenceContext(records, history_ids, len(records), frame_ids)

    def _joint_anchor_context(self, context: _JointInferenceContext) -> _ChunkAnchorContext | None:
        if not self.keyframe_anchor_enabled or context.history_count <= 0:
            return None
        anchor_idx = int(context.history_count) - 1
        record = context.history_records[anchor_idx]
        return _ChunkAnchorContext(
            record=_KeyframeAnchorRecord(
                frame_id=int(record.frame_id),
                image=record.image.detach().cpu().float() if record.image is not None else torch.empty(0),
                pose_c2w=record.pose_c2w.detach().cpu().float(),
            ),
            full_index=anchor_idx,
            prepended=True,
            current_full_indices=tuple(range(int(context.history_count), int(context.history_count) + len(context.current_frame_ids))),
        )

    def _process_chunk(self, start: int, end: int, *, final: bool) -> None:
        frame_ids = tuple(int(r.frame.frame_id) for r in self.records[start:end])
        range_key = (frame_ids[0], frame_ids[-1], len(frame_ids))
        if range_key in self.processed_ranges:
            return
        self.processed_ranges.add(range_key)

        profile: dict = {
            "chunk_index": int(self.chunk_count),
            "frame_start": int(frame_ids[0]),
            "frame_end": int(frame_ids[-1]),
            "frame_count": int(len(frame_ids)),
            "history_count": 0,
        }
        profile_total_start = time.perf_counter()
        profile_mark = profile_total_start

        def mark_profile(name: str) -> None:
            nonlocal profile_mark
            now = time.perf_counter()
            profile[f"{name}_sec"] = float(now - profile_mark)
            profile_mark = now

        def add_profile(name: str, elapsed: float) -> None:
            profile[name] = float(profile.get(name, 0.0)) + float(elapsed)

        images = torch.stack([r.image for r in self.records[start:end]], dim=0).to(self.device)
        mark_profile("stack_images")
        infer_images, joint_context = self._build_joint_inference_batch(images, frame_ids)
        self.current_recent_history_ids = tuple(joint_context.history_frame_ids)
        profile["history_count"] = int(joint_context.history_count)
        profile["recent_history_count"] = int(len(joint_context.history_frame_ids))
        mark_profile("joint_batch")
        pred_full0 = self.engine.infer(infer_images)
        mark_profile("engine_infer")
        full_factor_graph = getattr(self.engine, "last_dense_factor_graph", None)
        pred0 = (
            _slice_prediction(pred_full0, joint_context.history_count, len(frame_ids))
            if joint_context.history_count > 0
            else pred_full0
        )
        anchor_context = self._joint_anchor_context(joint_context)
        if anchor_context is None:
            anchor_context = self._anchor_context_from_current_chunk(frame_ids)
        if anchor_context is not None:
            anchor_metrics = self._compute_anchor_metrics(pred_full0 if joint_context.history_count > 0 else pred0, anchor_context)
        elif self.joint_inference_enabled:
            anchor_metrics = {}
        else:
            anchor_metrics = self._compute_anchor_metrics_sidepath(images, frame_ids)
        mark_profile("anchor_metrics")
        factor_graph = full_factor_graph
        if self.dense_ba_history_enabled:
            section_start = time.perf_counter()
            seed_transform = self._align_chunk(
                pred0,
                frame_ids,
                history_pred=pred_full0 if joint_context.history_count > 0 else None,
                history_records=joint_context.history_records,
            )
            add_profile("alignment_sec", time.perf_counter() - section_start)
            if joint_context.history_count > 0:
                pred_ba_seed = self._prediction_to_world(pred_full0, seed_transform)
                section_start = time.perf_counter()
                pred_refined, ba_stats = self.dense_ba_refiner.refine(
                    pred_ba_seed,
                    tuple((*joint_context.history_frame_ids, *frame_ids)),
                    factor_graph=factor_graph,
                    keyframe_memory=None,
                    current_start=joint_context.history_count,
                    current_count=len(frame_ids),
                    fixed_frames_override=1,
                )
                add_profile("dense_ba_sec", time.perf_counter() - section_start)
                pred_ba_input = _slice_prediction(pred_ba_seed, joint_context.history_count, len(frame_ids))
            else:
                pred_ba_input = self._prediction_to_world(pred0, seed_transform)
                section_start = time.perf_counter()
                pred_refined, ba_stats = self.dense_ba_refiner.refine(
                    pred_ba_input,
                    frame_ids,
                    factor_graph=factor_graph,
                    keyframe_memory=self.keyframe_memory,
                )
                add_profile("dense_ba_sec", time.perf_counter() - section_start)
            pred = pred_refined if ba_stats.used_refined else pred_ba_input
            transform = SimilarityTransform.identity(device=pred.depth.device, dtype=pred.depth.dtype)
            transform.accepted = bool(seed_transform.accepted)
            alignment_residual = float(seed_transform.residual)
        else:
            section_start = time.perf_counter()
            pred_refined, ba_stats = self.dense_ba_refiner.refine(
                pred0,
                frame_ids,
                factor_graph=factor_graph,
                keyframe_memory=self.keyframe_memory,
            )
            add_profile("dense_ba_sec", time.perf_counter() - section_start)
            pred = pred_refined if ba_stats.used_refined else pred0
            section_start = time.perf_counter()
            transform = self._align_chunk(
                pred,
                frame_ids,
                history_pred=pred_full0 if joint_context.history_count > 0 else None,
                history_records=joint_context.history_records,
            )
            add_profile("alignment_sec", time.perf_counter() - section_start)
            alignment_residual = float(transform.residual)
        mark_profile("dense_ba_and_alignment")
        factor_graph = getattr(self.dense_ba_refiner, "last_factor_graph", factor_graph) or factor_graph
        self.last_dense_ba_stats = ba_stats
        section_start = time.perf_counter()
        pred, transform, keyframe_graph_current_stats = self._post_align_current_to_last_ba(pred, frame_ids, transform)
        if keyframe_graph_current_stats is not None:
            self._record_keyframe_graph_stats(keyframe_graph_current_stats)
        add_profile("keyframe_graph_current_to_last_sec", time.perf_counter() - section_start)
        if ba_stats.enabled:
            self.dense_ba_stats_history.append(ba_stats)
            self.last_m3_debug = {
                "chunk_index": int(self.chunk_count),
                "frame_ids": frame_ids,
                "recent_history_ids": tuple(joint_context.history_frame_ids),
                "stats": ba_stats,
                "factor_graph": factor_graph,
                "sky_prob": pred0.sky_prob.detach().cpu() if pred0.sky_prob is not None else None,
                "feature_hw": pred0.feature_hw,
                "image_hw": pred0.image_hw,
                "images": images.detach().cpu(),
                "alignment": {
                    "overlap_points": float(self.last_alignment_debug.overlap_points),
                    "history_points": float(self.last_alignment_debug.history_points),
                    "overlap_alignment_points": float(self.last_alignment_debug.overlap_points),
                    "history_alignment_points": float(self.last_alignment_debug.history_points),
                    "history_ids": tuple(int(fid) for fid in self.last_alignment_debug.history_ids),
                    "scale": float(self.last_alignment_debug.scale),
                    "alignment_scale": float(self.last_alignment_debug.scale),
                    "residual": float(self.last_alignment_debug.residual),
                    "alignment_rmse": float(self.last_alignment_debug.residual),
                    "inlier_ratio": float(self.last_alignment_debug.inlier_ratio),
                    "forced_accepted": bool(self.last_alignment_debug.forced_accepted),
                },
                "keyframe_anchor": {
                    int(frame_ids[local_idx]): metrics.frame_mean_pair_conf
                    for local_idx, metrics in anchor_metrics.items()
                },
            }
            if keyframe_graph_current_stats is not None:
                self.last_m3_debug.setdefault("keyframe_graph", {}).update(keyframe_graph_current_stats.as_debug())
        else:
            self.last_m3_debug = None
            if keyframe_graph_current_stats is not None and keyframe_graph_current_stats.enabled:
                self.last_m3_debug = {
                    "chunk_index": int(self.chunk_count),
                    "frame_ids": frame_ids,
                    "recent_history_ids": tuple(joint_context.history_frame_ids),
                    "keyframe_graph": keyframe_graph_current_stats.as_debug(),
                    "alignment": {
                        "overlap_points": float(self.last_alignment_debug.overlap_points),
                        "history_points": float(self.last_alignment_debug.history_points),
                        "history_ids": tuple(int(fid) for fid in self.last_alignment_debug.history_ids),
                        "scale": float(self.last_alignment_debug.scale),
                        "alignment_scale": float(self.last_alignment_debug.scale),
                        "residual": float(self.last_alignment_debug.residual),
                        "alignment_rmse": float(self.last_alignment_debug.residual),
                        "inlier_ratio": float(self.last_alignment_debug.inlier_ratio),
                        "forced_accepted": bool(self.last_alignment_debug.forced_accepted),
                    },
                }
        mark_profile("m3_debug")
        backend_correction = self._backend_feedback_correction(pred, frame_ids, transform)
        mark_profile("backend_feedback_correction")
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
        mark_profile("loop_descriptor")

        output_data: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                _AnchorFrameMetrics | None,
                torch.Tensor | None,
                float,
                str,
            ],
        ] = {}
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
            points = transform.apply_points(pred.chunk_world_points[local_idx].to(self.device)).detach()
            if backend_correction is not None:
                pose = (backend_correction @ pose).detach()
                points = self._apply_pose_correction_to_points(points, backend_correction).detach()
            points_full = _resize_points(points, image_size).detach()
            sky_prob = pred.sky_prob[local_idx].detach() if pred.sky_prob is not None else None
            if sky_prob is not None:
                self.sky_mask_by_frame[frame_id] = self._sky_prob_to_mask(sky_prob, image_size).detach().cpu()
            valid_world = (
                torch.isfinite(points_full).all(dim=-1, keepdim=False)
                & torch.isfinite(inv_full[0])
                & torch.isfinite(conf_full[0])
                & (conf_full[0] > 0.0)
            ).unsqueeze(0)

            self.pose_by_frame[frame_id] = pose
            self.depth_by_frame[frame_id] = inv_full
            self.conf_by_frame[frame_id] = conf_full
            self.global_points_by_frame[frame_id] = points
            output_data[frame_id] = (
                pose,
                inv_full,
                conf_full,
                points_full,
                valid_world,
                anchor_metrics.get(local_idx),
                sky_prob,
                alignment_residual,
                "tracked_panovggt_long" + ba_stats.status_suffix,
            )
        mark_profile("output_fields")

        emit_end = end if final else max(start, end - self.emit_delay)
        for record in self.records[start:emit_end]:
            frame_id = int(record.frame.frame_id)
            if frame_id in self.emitted_frame_ids:
                continue
            data = output_data.get(frame_id)
            if data is not None:
                pose, inverse_depth, confidence, world_points, valid_world, anchor_metric, sky_prob, residual, status = data
                output = self._make_output(
                    record.frame,
                    pose=pose,
                    inverse_depth=inverse_depth,
                    confidence=confidence,
                    world_points=world_points,
                    valid_world_points_mask=valid_world,
                    anchor_metrics=anchor_metric,
                    sky_prob=sky_prob,
                    residual=residual,
                    status=status,
                )
                if output.is_keyframe and (
                    self.dense_ba_history_enabled or self.joint_inference_enabled or self.keyframe_graph_enabled
                ):
                    previous_keyframes = tuple(self.keyframe_memory.records)
                    keyframe_record = self._remember_keyframe_record(
                        frame_id,
                        frame_ids.index(frame_id),
                        pred,
                        pose,
                        image=record.image,
                        global_points=output.world_points,
                        confidence=output.depth_confidence,
                    )
                    graph_updates = self._handle_new_keyframe_graph_record(keyframe_record, previous_keyframes)
                    if graph_updates:
                        self._apply_keyframe_graph_update_to_output(output, graph_updates)
                self.pending_outputs.append(output)
                self.emitted_frame_ids.add(frame_id)
        mark_profile("emit_outputs")
        self.chunk_count += 1
        self._trim_alignment_cache()
        profile["total_sec"] = float(time.perf_counter() - profile_total_start)
        self.last_profile = profile
        if isinstance(self.last_m3_debug, dict):
            self.last_m3_debug["profile"] = dict(profile)

    def _anchor_context_from_current_chunk(
        self,
        frame_ids: tuple[int, ...],
    ) -> _ChunkAnchorContext | None:
        if not self.keyframe_anchor_enabled or self.last_keyframe_anchor is None:
            return None
        anchor = self.last_keyframe_anchor
        if int(anchor.frame_id) in frame_ids:
            anchor_idx = frame_ids.index(int(anchor.frame_id))
            return _ChunkAnchorContext(
                record=anchor,
                full_index=int(anchor_idx),
                prepended=False,
                current_full_indices=tuple(range(len(frame_ids))),
            )
        return None

    def _build_anchor_sidepath_batch(
        self,
        images: torch.Tensor,
        frame_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor | None, _ChunkAnchorContext | None]:
        if not self.keyframe_anchor_enabled or self.last_keyframe_anchor is None:
            return None, None
        anchor = self.last_keyframe_anchor
        if int(anchor.frame_id) in frame_ids:
            return None, None
        if not self.m3_config.keyframe_anchor.prepend_previous_keyframe:
            return None, None

        anchor_image = anchor.image.to(device=images.device, dtype=images.dtype)
        if tuple(anchor_image.shape[-2:]) != tuple(images.shape[-2:]):
            anchor_image = F.interpolate(
                anchor_image.unsqueeze(0),
                size=tuple(int(v) for v in images.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )[0]
        infer_images = torch.cat([anchor_image.unsqueeze(0), images], dim=0)
        return infer_images, _ChunkAnchorContext(
            record=anchor,
            full_index=0,
            prepended=True,
            current_full_indices=tuple(range(1, len(frame_ids) + 1)),
        )

    def _compute_anchor_metrics_sidepath(
        self,
        images: torch.Tensor,
        frame_ids: tuple[int, ...],
    ) -> dict[int, _AnchorFrameMetrics]:
        infer_images, anchor_context = self._build_anchor_sidepath_batch(images, frame_ids)
        if infer_images is None or anchor_context is None:
            return {}
        pred_anchor = self.engine.infer(infer_images)
        return self._compute_anchor_metrics(pred_anchor, anchor_context)

    def _compute_anchor_metrics(
        self,
        pred_full: PanoVGGTLocalPrediction,
        anchor_context: _ChunkAnchorContext,
    ) -> dict[int, _AnchorFrameMetrics]:
        if (
            pred_full.dense_descriptors is None
            or pred_full.match_confidence is None
            or pred_full.sky_prob is None
            or pred_full.feature_hw is None
            or pred_full.image_hw is None
        ):
            return {}
        edges = [
            (int(full_idx), int(anchor_context.full_index))
            for full_idx in anchor_context.current_full_indices
            if int(full_idx) != int(anchor_context.full_index)
        ]
        if not edges:
            return {}
        dense = self.m3_config.dense_matching
        feature_hw = tuple(int(v) for v in pred_full.feature_hw)
        full_factor_cap = max(1, len(edges) * int(feature_hw[0]) * int(feature_hw[1]))
        matcher = PoseGuidedDenseMatcher(
            search_radius=dense.search_radius,
            topk=dense.topk,
            min_match_confidence=0.0,
            min_static_confidence=0.0,
            min_match_score=0.0,
            max_factors=full_factor_cap,
            max_samples_per_edge=None,
            use_wraparound=dense.use_wraparound,
            forward_backward=dense.forward_backward,
            fb_tolerance=dense.fb_tolerance,
            use_depth_consistency=dense.use_depth_consistency,
            depth_consistency_rel=dense.depth_consistency_rel,
            depth_consistency_abs=dense.depth_consistency_abs,
        )
        graph = matcher.match(
            poses_c2w=pred_full.poses_c2w,
            depth=pred_full.depth,
            dense_descriptors=pred_full.dense_descriptors,
            match_confidence=pred_full.match_confidence,
            sky_prob=pred_full.sky_prob,
            static_confidence=pred_full.static_confidence,
            image_hw=pred_full.image_hw,
            feature_hw=feature_hw,
            edge_pairs=torch.tensor(edges, dtype=torch.long, device=pred_full.poses_c2w.device),
        )
        full_to_local = {int(full_idx): local_idx for local_idx, full_idx in enumerate(anchor_context.current_full_indices)}
        out: dict[int, _AnchorFrameMetrics] = {}
        for factor in graph.factors:
            src_full = int(factor.src)
            local_idx = full_to_local.get(src_full)
            if local_idx is None:
                continue
            src_conf = _sample_feature_map(pred_full.match_confidence[src_full : src_full + 1], factor.src_uv)[:, 0]
            tgt_conf = _sample_feature_map(pred_full.match_confidence[int(factor.tgt) : int(factor.tgt) + 1], factor.tgt_uv)[:, 0]
            match_score = factor.match_score.to(src_conf).clamp(0.0, 1.0)
            src_conf = src_conf.clamp(0.0, 1.0)
            tgt_conf = tgt_conf.clamp(0.0, 1.0)
            mode = str(self.m3_config.keyframe_anchor.pair_confidence_mode).lower()
            if mode in {"geometric_mean", "geomean", "geom"}:
                pair_conf = (match_score * src_conf * tgt_conf).clamp_min(1.0e-8).pow(1.0 / 3.0)
            elif mode in {"min", "minimum"}:
                pair_conf = torch.minimum(match_score, torch.minimum(src_conf, tgt_conf))
            elif mode in {"match_score", "score"}:
                pair_conf = match_score
            else:
                pair_conf = match_score * src_conf * tgt_conf
            pair_conf = pair_conf.clamp(0.0, 1.0)
            ok = torch.isfinite(pair_conf)
            if torch.is_tensor(factor.valid_mask) and factor.valid_mask.numel() == pair_conf.numel():
                ok = ok & factor.valid_mask.to(device=pair_conf.device).bool().reshape(-1)
            for key in ("fb_pass_mask", "depth_consistency_mask"):
                value = factor.metadata.get(key)
                if torch.is_tensor(value) and value.numel() == pair_conf.numel():
                    ok = ok & value.to(device=pair_conf.device).bool().reshape(-1)
            valid_uv = factor.src_uv.to(pair_conf.device)[ok]
            valid_pair = pair_conf[ok]
            pair_map, matched_cells = _scatter_feature_values_and_mask(valid_uv, valid_pair, feature_hw=feature_hw)

            sky = pred_full.sky_prob[src_full].detach().to(pair_map)
            non_sky = (sky <= float(self.m3_config.keyframe_anchor.sky_threshold)).float()
            low_pair = (pair_map < float(self.m3_config.keyframe_anchor.cell_pair_conf_threshold)) & matched_cells
            non_sky_cells = non_sky > 0.5
            valid_cells = non_sky_cells & matched_cells
            if bool(valid_cells.any()):
                values = pair_map[valid_cells]
                mean_pair = float(values.mean().detach().cpu())
                low_ratio = float(low_pair[valid_cells].float().mean().detach().cpu())
                pair_quantiles = _quantiles(values)
            else:
                mean_pair = 1.0
                low_ratio = 0.0
                pair_quantiles = {}
            non_sky_count = int(non_sky_cells.sum().detach().cpu())
            matched_count = int((non_sky_cells & matched_cells).sum().detach().cpu())
            match_coverage = float(matched_count / max(1, non_sky_count))
            out[local_idx] = _AnchorFrameMetrics(
                anchor_frame_id=int(anchor_context.record.frame_id),
                frame_mean_pair_conf=mean_pair,
                low_pair_conf_ratio=low_ratio,
                match_coverage=match_coverage,
                pair_conf_quantiles=pair_quantiles,
                pair_confidence=pair_map.detach(),
                low_pair_conf=low_pair.detach(),
                matched_cells=matched_cells.detach(),
                non_sky=non_sky_cells.detach(),
                anchor_pose_c2w=anchor_context.record.pose_c2w.detach(),
            )
        return out

    def _prediction_to_world(
        self,
        pred: PanoVGGTLocalPrediction,
        transform: SimilarityTransform,
    ) -> PanoVGGTLocalPrediction:
        poses = torch.stack([transform.apply_pose(pose.to(self.device)) for pose in pred.poses_c2w], dim=0).detach()
        depth = (pred.depth.to(self.device) * float(transform.scale)).clamp_min(1.0e-6).detach()
        local_points = None
        if pred.local_points is not None:
            local_points = (pred.local_points.to(self.device) * float(transform.scale)).detach()
        points = torch.stack([transform.apply_points(points.to(self.device)) for points in pred.chunk_world_points], dim=0).detach()
        return replace(
            pred,
            poses_c2w=poses.to(pred.poses_c2w),
            depth=depth.to(pred.depth),
            local_points=local_points.to(pred.depth) if local_points is not None else None,
            chunk_world_points=points.to(pred.chunk_world_points),
        )

    @staticmethod
    def _identity_like_transform(transform: SimilarityTransform, *, device: torch.device, dtype: torch.dtype) -> SimilarityTransform:
        identity = SimilarityTransform.identity(device=device, dtype=dtype)
        identity.accepted = bool(transform.accepted)
        identity.residual = float(transform.residual)
        identity.inlier_ratio = float(transform.inlier_ratio)
        return identity

    def _post_align_current_to_last_ba(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        transform: SimilarityTransform,
    ) -> tuple[PanoVGGTLocalPrediction, SimilarityTransform, KeyframeGraphBAStats | None]:
        if not self.keyframe_graph_enabled:
            return pred, transform, None
        world_pred = self._prediction_to_world(pred, transform)
        identity = self._identity_like_transform(transform, device=world_pred.depth.device, dtype=world_pred.depth.dtype)
        if not transform.accepted:
            stats = KeyframeGraphBAStats(
                enabled=True,
                stage="current_to_last",
                success=False,
                fallback_reason="alignment_not_accepted",
            )
            return world_pred, identity, stats
        if not self.m3_config.keyframe_graph.current_to_last_ba:
            stats = KeyframeGraphBAStats(
                enabled=True,
                stage="current_to_last",
                success=False,
                fallback_reason="disabled",
            )
            return world_pred, identity, stats
        last_keyframe = None
        if self.last_keyframe_id is not None:
            last_keyframe = self.keyframe_memory.get(int(self.last_keyframe_id))
        if last_keyframe is None and self.keyframe_memory.records:
            last_keyframe = self.keyframe_memory.records[-1]
        new_local_indices = tuple(
            local_idx
            for local_idx, frame_id in enumerate(frame_ids)
            if int(frame_id) not in self.global_points_by_frame
        )
        refined, stats = self.keyframe_graph_refiner.refine_current_to_last(
            world_pred,
            frame_ids,
            new_local_indices=new_local_indices,
            last_keyframe=last_keyframe,
        )
        return refined, identity, stats

    def _record_keyframe_graph_stats(self, stats: KeyframeGraphBAStats) -> None:
        self.last_keyframe_graph_stats = stats
        if stats.enabled:
            self.keyframe_graph_stats_history.append(stats)

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

    def _alignment_confidence(
        self,
        pred: PanoVGGTLocalPrediction,
        local_idx: int,
        point_hw: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        conf = _resize_scalar_map(pred.confidence[int(local_idx)], point_hw)
        if conf is None:
            conf = torch.ones(point_hw, device=device, dtype=dtype)
        else:
            conf = conf.to(device=device, dtype=dtype)

        alignment_cfg = self.m3_config.alignment
        if bool(self.m3_config.enabled and alignment_cfg.exclude_sky) and pred.sky_prob is not None:
            sky = _resize_scalar_map(pred.sky_prob[int(local_idx)], point_hw)
            if sky is not None:
                non_sky = sky.to(device=device) <= float(alignment_cfg.sky_threshold)
                conf = conf * non_sky.to(dtype=dtype)
        return conf

    def _align_chunk(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        *,
        history_pred: PanoVGGTLocalPrediction | None = None,
        history_records: tuple[KeyframeRecord, ...] = (),
    ) -> SimilarityTransform:
        self.last_alignment_debug = _AlignmentDebug(history_ids=tuple(int(r.frame_id) for r in history_records))
        if not self.global_points_by_frame:
            transform = self._root_transform(pred.poses_c2w[0])
            self.last_alignment_debug = _AlignmentDebug(
                history_ids=tuple(int(r.frame_id) for r in history_records),
                scale=float(transform.scale),
                residual=float(transform.residual),
                inlier_ratio=float(transform.inlier_ratio),
            )
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
        overlap_points = 0
        history_points = 0
        use_history = bool(self.m3_config.alignment.use_common_history and history_pred is not None and history_records)
        has_overlap_targets = any(int(frame_id) in self.global_points_by_frame for frame_id in frame_ids)
        history_targets: list[tuple[int, KeyframeRecord, torch.Tensor]] = []
        if use_history and history_pred is not None:
            for hist_idx, record in enumerate(history_records):
                target = record.global_points
                if target is None:
                    target = self.global_points_by_frame.get(int(record.frame_id))
                if target is not None:
                    history_targets.append((int(hist_idx), record, target))
        use_history = bool(use_history and history_targets)
        ratio = min(1.0, max(0.0, float(self.m3_config.alignment.history_point_budget_ratio)))
        total_budget = max(1, int(self.max_alignment_points))
        if use_history and has_overlap_targets:
            history_budget = int(total_budget * ratio)
            if ratio > 0.0:
                history_budget = max(1, history_budget)
            if ratio >= 1.0:
                history_budget = total_budget
            history_budget = min(total_budget, history_budget)
            overlap_budget = max(0, total_budget - history_budget)
        elif use_history:
            history_budget = total_budget
            overlap_budget = 0
        else:
            history_budget = 0
            overlap_budget = total_budget

        if overlap_budget > 0:
            per_overlap = max(1, overlap_budget // max(1, self.overlap))
            for local_idx, frame_id in enumerate(frame_ids):
                target = self.global_points_by_frame.get(int(frame_id))
                if target is None:
                    continue
                source = pred.chunk_world_points[local_idx].to(target.device)
                conf = self._alignment_confidence(
                    pred,
                    local_idx,
                    tuple(int(v) for v in source.shape[:2]),
                    device=target.device,
                    dtype=source.dtype,
                )
                src, tgt, weights = sample_overlap_points(
                    source,
                    target,
                    conf,
                    None,
                    max_points=per_overlap,
                )
                if src.numel() == 0:
                    continue
                source_parts.append(src)
                target_parts.append(tgt)
                weight_parts.append(weights)
                overlap_points += int(src.shape[0])

        if use_history and history_pred is not None and history_budget > 0:
            per_history = max(1, history_budget // max(1, len(history_targets)))
            for hist_idx, record, target in history_targets:
                target_t = target.to(device=history_pred.chunk_world_points.device, dtype=history_pred.chunk_world_points.dtype)
                source = history_pred.chunk_world_points[hist_idx].to(target_t)
                if tuple(source.shape[:2]) != tuple(target_t.shape[:2]):
                    source = _resize_points(source, tuple(int(v) for v in target_t.shape[:2]))
                conf = self._alignment_confidence(
                    history_pred,
                    hist_idx,
                    tuple(int(v) for v in source.shape[:2]),
                    device=target_t.device,
                    dtype=source.dtype,
                )
                src, tgt, weights = sample_overlap_points(
                    source,
                    target_t,
                    conf,
                    None,
                    max_points=per_history,
                )
                if src.numel() == 0:
                    continue
                source_parts.append(src)
                target_parts.append(tgt)
                weight_parts.append(weights)
                history_points += int(src.shape[0])

        if not source_parts:
            return self._handle_alignment_failure(
                "no overlapping or common-history frames with cached global world points",
                device=pred.depth.device,
                dtype=pred.depth.dtype,
            )
        source_all = torch.cat(source_parts, dim=0)
        target_all = torch.cat(target_parts, dim=0)
        weight_all = torch.cat(weight_parts, dim=0)
        if source_all.shape[0] < self.min_overlap_points:
            return self._handle_alignment_failure(
                f"only {source_all.shape[0]} overlap points, require {self.min_overlap_points}",
                device=pred.depth.device,
                dtype=pred.depth.dtype,
            )
        transform = self.aligner.align(source_all, target_all, weight_all)
        self.last_alignment_debug = _AlignmentDebug(
            overlap_points=int(overlap_points),
            history_points=int(history_points),
            history_ids=tuple(int(r.frame_id) for r in history_records),
            scale=float(transform.scale),
            residual=float(transform.residual),
            inlier_ratio=float(transform.inlier_ratio),
        )
        if not transform.accepted:
            if self.force_accept_alignment:
                transform.accepted = True
                self.last_alignment_debug.forced_accepted = True
                self.pose_graph.add_edge(
                    PoseGraphEdge(
                        source_chunk=self.chunk_count,
                        target_chunk=max(0, self.chunk_count - 1),
                        transform=transform,
                        residual=transform.residual,
                        edge_type="sequential_forced",
                    )
                )
                return transform
            return self._handle_alignment_failure(
                (
                    f"alignment rejected: rmse={transform.residual:.4f}, "
                    f"inlier_ratio={transform.inlier_ratio:.4f}, scale={transform.scale:.4f}"
                ),
                device=pred.depth.device,
                dtype=pred.depth.dtype,
                transform=transform,
            )
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

    def _root_transform(self, first_pose_c2w: torch.Tensor) -> SimilarityTransform:
        pose = first_pose_c2w.detach().float()
        rot = pose[:3, :3].T.contiguous()
        trans = -(rot @ pose[:3, 3])
        return SimilarityTransform(
            scale=1.0,
            rotation=rot.to(device=first_pose_c2w.device, dtype=first_pose_c2w.dtype),
            translation=trans.to(device=first_pose_c2w.device, dtype=first_pose_c2w.dtype),
            residual=0.0,
            inlier_ratio=1.0,
            accepted=True,
        )

    def _handle_alignment_failure(
        self,
        reason: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        transform: SimilarityTransform | None = None,
    ) -> SimilarityTransform:
        if self.require_aligned_world_points and not self.emit_unaligned:
            raise PanoVGGTAlignmentError(f"PanoVGGT chunk {self.chunk_count} alignment failed: {reason}")
        if transform is not None:
            return transform
        fallback = SimilarityTransform.identity(device=device, dtype=dtype)
        fallback.accepted = False
        return fallback

    def _make_output(
        self,
        frame: PanoFrame,
        *,
        pose: torch.Tensor,
        inverse_depth: torch.Tensor,
        confidence: torch.Tensor,
        world_points: torch.Tensor,
        valid_world_points_mask: torch.Tensor,
        anchor_metrics: _AnchorFrameMetrics | None,
        sky_prob: torch.Tensor | None,
        residual: float,
        status: str,
    ) -> FrontendOutput:
        frame_id = int(frame.frame_id)
        prev_pose = self.pose_by_frame.get(frame_id - 1)
        relative = None if prev_pose is None else _relative_from_c2w(prev_pose.unsqueeze(0), pose.unsqueeze(0))[0]
        key_score = float(confidence.mean().detach().cpu())
        is_keyframe, decision = self._decide_keyframe(
            frame_id=frame_id,
            pose=pose,
            inverse_depth=inverse_depth,
            confidence=confidence,
            key_score=key_score,
            anchor_metrics=anchor_metrics,
        )
        if self.keyframe_anchor_enabled:
            self._record_keyframe_decision(decision)
        output_valid = valid_world_points_mask
        output_world_confidence = confidence
        if self.novel_insertion_enabled and self.keyframe_anchor_enabled and is_keyframe:
            image_size = tuple(int(v) for v in world_points.shape[:2])
            first_keyframe = self.last_keyframe_id is None
            if self.novel_insertion_strategy == "pfgs360":
                hints = self._pfgs360_insertion_hints(
                    image_size=image_size,
                    anchor_metrics=anchor_metrics,
                    sky_prob=sky_prob,
                    first_keyframe=first_keyframe,
                )
                if hints:
                    self._pending_insertion_hints[frame_id] = hints
            else:
                output_valid, output_world_confidence = self._novel_world_mask_and_confidence(
                    valid_world_points_mask=valid_world_points_mask,
                    confidence=confidence,
                    image_size=image_size,
                    anchor_metrics=anchor_metrics,
                    sky_prob=sky_prob,
                    first_keyframe=first_keyframe,
                )
        if is_keyframe:
            self.last_keyframe_id = frame_id
            self.last_keyframe_anchor = _KeyframeAnchorRecord(
                frame_id=frame_id,
                image=ensure_chw_image(frame.image).detach().cpu().float(),
                pose_c2w=pose.detach().cpu().float(),
            )
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
            world_points=world_points.detach().cpu(),
            world_points_confidence=output_world_confidence.detach().cpu(),
            valid_world_points_mask=output_valid.detach().cpu(),
        )

    def _sky_prob_to_mask(self, sky_prob: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        tensor = sky_prob.detach().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if int(tensor.shape[0]) != 1:
                tensor = tensor[:1]
        else:
            raise ValueError(f"Expected sky probability as HxW or 1xHxW, got {tuple(sky_prob.shape)}")
        resized = _resize_field(tensor, (int(image_size[0]), int(image_size[1])))
        return (resized >= float(self.m3_config.keyframe_anchor.sky_threshold)).detach().cpu().bool()

    def _remember_keyframe_record(
        self,
        frame_id: int,
        local_idx: int,
        pred: PanoVGGTLocalPrediction,
        pose_c2w: torch.Tensor,
        *,
        image: torch.Tensor | None = None,
        global_points: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> KeyframeRecord | None:
        if (
            pred.dense_descriptors is None
            or pred.match_confidence is None
            or pred.sky_prob is None
            or pred.feature_hw is None
            or pred.image_hw is None
        ):
            return None
        idx = int(local_idx)
        if idx < 0 or idx >= int(pred.poses_c2w.shape[0]):
            return None
        static_confidence = None
        if pred.static_confidence is not None:
            static_confidence = pred.static_confidence[idx].detach().cpu().float()
        record = KeyframeRecord(
            frame_id=int(frame_id),
            pose_c2w=pose_c2w.detach().cpu().float(),
            depth_low=pred.depth[idx].detach().cpu().float(),
            dense_descriptors=pred.dense_descriptors[idx].detach().cpu().float(),
            match_confidence=pred.match_confidence[idx].detach().cpu().float(),
            sky_prob=pred.sky_prob[idx].detach().cpu().float(),
            static_confidence=static_confidence,
            feature_hw=tuple(int(v) for v in pred.feature_hw),
            image_hw=tuple(int(v) for v in pred.image_hw),
            image=None if image is None else image.detach().cpu().float(),
            global_points=None if global_points is None else global_points.detach().cpu().float(),
            confidence=None if confidence is None else confidence.detach().cpu().float(),
            frozen=True,
        )
        self.keyframe_memory.add(record)
        return record

    def _handle_new_keyframe_graph_record(
        self,
        record: KeyframeRecord | None,
        previous_records: tuple[KeyframeRecord, ...],
    ) -> dict[int, torch.Tensor]:
        if not self.keyframe_graph_enabled or record is None:
            return {}
        cfg = self.m3_config.keyframe_graph
        added_edges = 0
        edge_stats: list[KeyframeGraphBAStats] = []
        if cfg.adjacent_edges:
            history = max(1, int(cfg.adjacent_history))
            candidates = [
                item
                for item in previous_records
                if int(item.frame_id) != int(record.frame_id)
            ][-history:]
            for target in reversed(candidates):
                edge, stats = self.keyframe_graph_refiner.build_adjacent_edge(
                    source=record,
                    target=target,
                    edge_type="adjacent",
                )
                self._record_keyframe_graph_stats(stats)
                edge_stats.append(stats)
                if edge is None:
                    continue
                self.keyframe_correspondence_graph.add_edge(edge)
                added_edges += 1

        updates: dict[int, torch.Tensor] = {}
        if added_edges > 0 and len(self.keyframe_memory) % max(1, int(cfg.optimize_every_keyframes)) == 0:
            updates = self._optimize_keyframe_correspondence_graph()

        if isinstance(self.last_m3_debug, dict):
            graph_debug = self.last_m3_debug.setdefault("keyframe_graph", {})
            graph_debug.update(self.keyframe_correspondence_graph.metrics())
            graph_debug["keyframe_graph_added_edges"] = float(added_edges)
            if edge_stats:
                graph_debug.update(edge_stats[-1].as_debug())
        return updates

    def _optimize_keyframe_correspondence_graph(self) -> dict[int, torch.Tensor]:
        updates, stats = self.keyframe_graph_refiner.optimize_keyframe_graph(
            memory=self.keyframe_memory,
            graph=self.keyframe_correspondence_graph,
        )
        self._record_keyframe_graph_stats(stats)
        applied = self._apply_keyframe_graph_pose_updates(updates)
        if isinstance(self.last_m3_debug, dict):
            graph_debug = self.last_m3_debug.setdefault("keyframe_graph", {})
            graph_debug.update(stats.as_debug())
            graph_debug["keyframe_graph_pose_updates"] = float(len(applied))
        return applied

    def _apply_keyframe_graph_pose_updates(self, updates: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if not updates:
            return {}
        old_poses: dict[int, torch.Tensor] = {}
        for frame_id, pose in updates.items():
            fid = int(frame_id)
            current = self.pose_by_frame.get(fid)
            if current is None:
                record = self.keyframe_memory.get(fid)
                current = None if record is None else record.pose_c2w
            if torch.is_tensor(current) and tuple(current.shape) == (4, 4):
                old_poses[fid] = current.detach().to(device=self.device, dtype=pose.dtype)

        self.keyframe_memory.update_poses(updates)
        applied: dict[int, torch.Tensor] = {}
        for frame_id, pose in updates.items():
            fid = int(frame_id)
            new_pose = pose.detach().cpu().float()
            if tuple(new_pose.shape) != (4, 4) or not torch.isfinite(new_pose).all():
                continue
            self.pose_by_frame[fid] = new_pose.to(self.device)
            old_pose = old_poses.get(fid)
            if old_pose is not None:
                correction = new_pose.to(device=old_pose.device, dtype=old_pose.dtype) @ torch.linalg.inv(old_pose)
                points = self.global_points_by_frame.get(fid)
                if points is not None:
                    self.global_points_by_frame[fid] = self._apply_pose_correction_to_points(
                        points.to(device=correction.device, dtype=correction.dtype),
                        correction,
                    ).detach()
            if self.last_keyframe_anchor is not None and int(self.last_keyframe_anchor.frame_id) == fid:
                self.last_keyframe_anchor.pose_c2w = new_pose
            if self.m3_config.keyframe_graph.publish_pose_updates:
                self.pending_keyframe_graph_pose_updates[fid] = new_pose
            applied[fid] = new_pose
        return applied

    def _apply_keyframe_graph_update_to_output(
        self,
        output: FrontendOutput,
        updates: dict[int, torch.Tensor],
    ) -> None:
        new_pose = updates.get(int(output.frame_id))
        if new_pose is None:
            return
        old_pose = output.pose_c2w.detach().float()
        new_pose = new_pose.detach().cpu().float()
        correction = new_pose @ torch.linalg.inv(old_pose)
        output.pose_c2w = new_pose
        prev_pose = self.pose_by_frame.get(int(output.frame_id) - 1)
        if prev_pose is not None:
            relative = _relative_from_c2w(prev_pose.detach().cpu().unsqueeze(0), new_pose.unsqueeze(0))[0]
            output.relative_pose = relative.detach().cpu()
        if output.world_points is not None:
            output.world_points = self._apply_pose_correction_to_points(output.world_points.detach().float(), correction).detach().cpu()
        if "kf_graph" not in output.tracking_status:
            output.tracking_status = f"{output.tracking_status}_kf_graph"

    def _decide_keyframe(
        self,
        *,
        frame_id: int,
        pose: torch.Tensor,
        inverse_depth: torch.Tensor,
        confidence: torch.Tensor,
        key_score: float,
        anchor_metrics: _AnchorFrameMetrics | None,
    ) -> tuple[bool, dict]:
        last_keyframe_id = int(self.last_keyframe_id) if self.last_keyframe_id is not None else None
        if not self.keyframe_anchor_enabled:
            gap = (
                int(frame_id) - int(self.last_keyframe_id)
                if self.last_keyframe_id is not None
                else self.force_keyframe_interval
            )
            reasons = []
            if key_score >= self.keyframe_threshold:
                reasons.append("key_score")
            if gap >= self.force_keyframe_interval:
                reasons.append("force_keyframe_interval")
            accepted = bool(reasons)
            return accepted, {
                "frame_id": int(frame_id),
                "last_keyframe_id": last_keyframe_id,
                "accepted": accepted,
                "reasons": reasons,
                "keyframe_score": float(key_score),
                "legacy_gap": int(gap),
            }
        if self.last_keyframe_id is None:
            return True, {
                "frame_id": int(frame_id),
                "last_keyframe_id": None,
                "accepted": True,
                "reasons": ["first_frame"],
                "keyframe_score": float(key_score),
                "recent_history_ids": list(self.current_recent_history_ids),
            }
        cfg = self.m3_config.keyframe_anchor
        reasons: list[str] = []
        keyframe_gap = int(frame_id) - int(self.last_keyframe_id)
        decision: dict = {
            "frame_id": int(frame_id),
            "last_keyframe_id": last_keyframe_id,
            "accepted": False,
            "reasons": reasons,
            "keyframe_score": float(key_score),
            "keyframe_gap": int(keyframe_gap),
            "recent_history_ids": list(self.current_recent_history_ids),
        }
        coverage_deficit = 0.0
        matching_uncertainty = 0.0
        connectivity_deficit = 1.0
        if anchor_metrics is not None:
            coverage_deficit = max(0.0, min(1.0, 1.0 - float(anchor_metrics.match_coverage)))
            matching_uncertainty = max(0.0, min(1.0, 1.0 - float(anchor_metrics.frame_mean_pair_conf)))
            connectivity_deficit = max(0.0, min(1.0, float(anchor_metrics.low_pair_conf_ratio)))
            decision.update(
                {
                    "anchor_frame_id": int(anchor_metrics.anchor_frame_id),
                    "frame_mean_pair_conf": float(anchor_metrics.frame_mean_pair_conf),
                    "low_pair_conf_ratio": float(anchor_metrics.low_pair_conf_ratio),
                    "match_coverage": float(anchor_metrics.match_coverage),
                    "pair_conf_quantiles": dict(anchor_metrics.pair_conf_quantiles),
                }
            )
            if float(anchor_metrics.frame_mean_pair_conf) <= float(cfg.frame_mean_pair_conf_threshold):
                reasons.append("low_mean_pair_conf")
            if float(anchor_metrics.low_pair_conf_ratio) >= float(cfg.frame_low_pair_conf_ratio):
                reasons.append("high_low_pair_conf_ratio")
            if (
                float(cfg.match_coverage_threshold) > 0.0
                and float(anchor_metrics.match_coverage) <= float(cfg.match_coverage_threshold)
            ):
                reasons.append("low_match_coverage")
            anchor_pose = anchor_metrics.anchor_pose_c2w.to(device=pose.device, dtype=pose.dtype)
        elif self.last_keyframe_anchor is not None:
            anchor_pose = self.last_keyframe_anchor.pose_c2w.to(device=pose.device, dtype=pose.dtype)
        else:
            anchor_pose = None
        if anchor_pose is not None and anchor_pose.shape == (4, 4):
            translation = torch.linalg.norm(pose[:3, 3] - anchor_pose[:3, 3])
            translation_value = float(translation.detach().cpu())
            decision["translation_delta"] = translation_value
            if translation_value >= float(cfg.translation_threshold):
                reasons.append("translation")
            depth = inverse_depth.detach().float().clamp_min(1.0e-6).reciprocal()
            valid_depth = depth[torch.isfinite(depth)]
            if valid_depth.numel() > 0:
                median_depth = float(valid_depth.median().detach().cpu())
                decision["median_depth"] = median_depth
                if median_depth > 1.0e-6 and translation_value / median_depth >= float(cfg.translation_depth_ratio_threshold):
                    decision["translation_depth_ratio"] = float(translation_value / median_depth)
                    reasons.append("translation_depth_ratio")
                elif median_depth > 1.0e-6:
                    decision["translation_depth_ratio"] = float(translation_value / median_depth)
        parallax = max(0.0, min(1.0, float(decision.get("translation_depth_ratio", 0.0)) / max(float(cfg.translation_depth_ratio_threshold), 1.0e-6)))
        m3_score = (
            0.35 * float(coverage_deficit)
            + 0.30 * float(matching_uncertainty)
            + 0.20 * float(connectivity_deficit)
            + 0.15 * float(parallax)
        )
        decision.update(
            {
                "m3_keyframe_score": float(m3_score),
                "map_coverage_deficit": float(coverage_deficit),
                "matching_uncertainty": float(matching_uncertainty),
                "graph_connectivity_deficit": float(connectivity_deficit),
                "parallax_score": float(parallax),
            }
        )
        if self.joint_inference_enabled:
            decision["legacy_candidate_reasons"] = list(reasons)
            reasons.clear()
        m3_score_threshold = (
            float(cfg.m3_score_threshold)
            if float(cfg.m3_score_threshold) >= 0.0
            else float(self.keyframe_threshold)
        )
        decision["m3_score_threshold"] = float(m3_score_threshold)
        if m3_score >= float(m3_score_threshold):
            reasons.append("m3_score")
        min_interval = max(0, int(cfg.min_keyframe_interval))
        max_interval = max(0, int(cfg.max_keyframe_interval))
        if min_interval > 0 and keyframe_gap < min_interval:
            decision["suppressed_by_min_keyframe_interval"] = True
            decision["suppressed_by_min_interval"] = True
            decision["suppressed_reasons"] = list(reasons)
            decision["min_keyframe_interval"] = int(min_interval)
            reasons.clear()
        if max_interval > 0 and keyframe_gap >= max_interval:
            if "max_keyframe_interval" not in reasons:
                reasons.append("max_keyframe_interval")
            decision["max_keyframe_interval"] = int(max_interval)
            decision["forced_by_max_interval"] = True
        accepted = bool(reasons)
        decision["accepted"] = accepted
        return accepted, decision

    def _record_keyframe_decision(self, decision: dict) -> None:
        serializable = dict(decision)
        serializable["reasons"] = list(serializable.get("reasons", []))
        self.keyframe_decision_history.append(serializable)
        self._pending_keyframe_decisions.append(serializable)

    def _novel_world_mask_and_confidence(
        self,
        *,
        valid_world_points_mask: torch.Tensor,
        confidence: torch.Tensor,
        image_size: tuple[int, int],
        anchor_metrics: _AnchorFrameMetrics | None,
        sky_prob: torch.Tensor | None,
        first_keyframe: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = valid_world_points_mask.bool()
        if first_keyframe and sky_prob is not None:
            non_sky_feature = sky_prob.detach().float() <= float(self.m3_config.keyframe_anchor.sky_threshold)
        elif anchor_metrics is not None:
            non_sky_feature = anchor_metrics.non_sky.bool()
        elif sky_prob is not None:
            non_sky_feature = sky_prob.detach().float() <= float(self.m3_config.keyframe_anchor.sky_threshold)
        else:
            non_sky_feature = None
        if first_keyframe:
            if non_sky_feature is None:
                return valid, confidence
            non_sky = _resize_nearest(non_sky_feature.float(), image_size) > 0.5
            world_conf = confidence * _resize_nearest(non_sky_feature.float(), image_size).to(confidence)
            return valid & non_sky.to(valid.device), world_conf.clamp(0.0, 1.0)
        if anchor_metrics is None:
            return torch.zeros_like(valid), torch.zeros_like(confidence)
        non_sky_source = non_sky_feature if non_sky_feature is not None else anchor_metrics.non_sky
        low_pair = _resize_nearest(anchor_metrics.low_pair_conf.float(), image_size) > 0.5
        non_sky = _resize_nearest(non_sky_source.float(), image_size) > 0.5
        pair_conf = _resize_nearest(anchor_metrics.pair_confidence.float(), image_size).clamp(0.0, 1.0)
        sky_weight = _resize_nearest(non_sky_source.float(), image_size).clamp(0.0, 1.0)
        candidate = low_pair.to(device=valid.device) & non_sky.to(device=valid.device)
        if self.novel_pair_conf_insert_threshold > 0.0:
            loose_pair = pair_conf <= float(self.novel_pair_conf_insert_threshold)
            candidate = candidate | (loose_pair.to(device=valid.device) & non_sky.to(device=valid.device))
        world_conf = (1.0 - pair_conf.to(confidence)) * sky_weight.to(confidence)
        if self.novel_insert_confidence_floor > 0.0:
            floor = torch.full_like(world_conf, float(self.novel_insert_confidence_floor))
            world_conf = torch.where(candidate.to(device=world_conf.device), torch.maximum(world_conf, floor), world_conf)
        candidate = valid & candidate
        if self.novel_spatial_cell_size > 0 and self.novel_max_seeds_per_cell > 0:
            candidate = _limit_mask_per_cell(
                candidate,
                world_conf.to(device=candidate.device),
                cell_size=self.novel_spatial_cell_size,
                max_per_cell=self.novel_max_seeds_per_cell,
            ).to(device=valid.device)
        return candidate, world_conf.clamp(0.0, 1.0)

    def _pfgs360_insertion_hints(
        self,
        *,
        image_size: tuple[int, int],
        anchor_metrics: _AnchorFrameMetrics | None,
        sky_prob: torch.Tensor | None,
        first_keyframe: bool,
    ) -> dict[str, torch.Tensor]:
        hints: dict[str, torch.Tensor] = {}
        if first_keyframe and sky_prob is not None:
            non_sky_feature = sky_prob.detach().float() <= float(self.m3_config.keyframe_anchor.sky_threshold)
        elif anchor_metrics is not None:
            non_sky_feature = anchor_metrics.non_sky.bool()
        elif sky_prob is not None:
            non_sky_feature = sky_prob.detach().float() <= float(self.m3_config.keyframe_anchor.sky_threshold)
        else:
            non_sky_feature = None
        if non_sky_feature is not None:
            hints["non_sky"] = (_resize_nearest(non_sky_feature.float(), image_size) > 0.5).detach().cpu()
        if sky_prob is not None:
            hints["sky_mask"] = self._sky_prob_to_mask(sky_prob, image_size).detach().cpu()
        if anchor_metrics is not None:
            hints["pair_confidence"] = _resize_field(
                anchor_metrics.pair_confidence.detach().float(),
                image_size,
            ).clamp(0.0, 1.0).detach().cpu()
            hints["low_pair_conf"] = (
                _resize_nearest(anchor_metrics.low_pair_conf.float(), image_size) > 0.5
            ).detach().cpu()
            hints["matched_cells"] = (
                _resize_nearest(anchor_metrics.matched_cells.float(), image_size) > 0.5
            ).detach().cpu()
        return hints

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
    mapping_cfg = config.get("Mapping", {})
    novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
    m3_enabled = bool((pano_cfg.get("M3Sphere", {}) or {}).get("enabled", False)) if isinstance(pano_cfg, dict) else False
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
        min_overlap_points=int(pano_cfg.get("min_overlap_points", 4096)),
        max_align_rmse=float(pano_cfg.get("max_align_rmse", 0.25)),
        min_inlier_ratio=float(pano_cfg.get("min_inlier_ratio", 0.35)),
        max_scale_jump=float(pano_cfg.get("max_scale_jump", 2.0)),
        force_accept_alignment=bool(pano_cfg.get("force_accept_alignment", True)),
        require_aligned_world_points=bool(pano_cfg.get("require_aligned_world_points", True)),
        emit_unaligned=bool(pano_cfg.get("emit_unaligned", False)),
        novel_insertion_enabled=bool(novel_cfg.get("enabled", m3_enabled)),
        novel_pair_conf_insert_threshold=float(novel_cfg.get("pair_conf_insert_threshold", 0.0)),
        novel_insert_confidence_floor=float(novel_cfg.get("insert_confidence_floor", 0.0)),
        novel_spatial_cell_size=int(novel_cfg.get("spatial_cell_size", 0)),
        novel_max_seeds_per_cell=int(novel_cfg.get("max_seeds_per_cell", 0)),
        novel_insertion_strategy=str(novel_cfg.get("strategy", "legacy")),
    )
