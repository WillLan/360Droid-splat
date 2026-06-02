from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


def normalize_window_order(window: List[int]) -> List[int]:
    ordered: List[int] = []
    seen: Set[int] = set()
    for kf in sorted((int(v) for v in window), reverse=True):
        if kf in seen:
            continue
        seen.add(kf)
        ordered.append(kf)
    return ordered


@dataclass
class SubmapRecord:
    submap_id: int
    kf_ids: List[int] = field(default_factory=list)
    anchor_pose: Optional[np.ndarray] = None
    neighbor_submaps: Set[int] = field(default_factory=set)
    gaussian_indices: Set[int] = field(default_factory=set)
    replay_kf_ids: List[int] = field(default_factory=list)
    last_refine_step: int = -1


class SubmapManager:
    """Small helper to organise keyframes into overlapping temporal submaps."""

    def __init__(self, interval: int = 10, overlap_kfs: int = 3):
        self.interval = max(int(interval), 1)
        self.overlap_kfs = max(int(overlap_kfs), 0)
        self.submaps: Dict[int, SubmapRecord] = {}

    def assign_frame(self, frame_idx: int, pose_c2w: Optional[np.ndarray] = None) -> int:
        submap_id = int(frame_idx) // self.interval
        record = self.submaps.setdefault(submap_id, SubmapRecord(submap_id=submap_id))
        if frame_idx not in record.kf_ids:
            record.kf_ids.append(int(frame_idx))
        if pose_c2w is not None and record.anchor_pose is None:
            record.anchor_pose = np.asarray(pose_c2w, dtype=np.float64).copy()
        if submap_id - 1 in self.submaps:
            record.neighbor_submaps.add(submap_id - 1)
            self.submaps[submap_id - 1].neighbor_submaps.add(submap_id)
        if submap_id + 1 in self.submaps:
            record.neighbor_submaps.add(submap_id + 1)
            self.submaps[submap_id + 1].neighbor_submaps.add(submap_id)
        return submap_id

    def update_anchor_pose(self, submap_id: int, pose_c2w: np.ndarray) -> None:
        record = self.submaps.setdefault(submap_id, SubmapRecord(submap_id=submap_id))
        record.anchor_pose = np.asarray(pose_c2w, dtype=np.float64).copy()

    def get_active_submaps(self, submap_id: int) -> List[int]:
        if submap_id not in self.submaps:
            return [submap_id]
        record = self.submaps[submap_id]
        ids = {submap_id}
        ids.update(record.neighbor_submaps)
        return sorted(ids)

    def filter_window(self, window: List[int], frame_to_submap: Dict[int, int], active_submap_id: int) -> List[int]:
        if not window:
            return []
        window = normalize_window_order(window)
        allowed = set(self.get_active_submaps(active_submap_id))
        filtered = [kf for kf in window if frame_to_submap.get(kf, active_submap_id) in allowed]
        if not filtered:
            filtered = [window[0]]
        if self.overlap_kfs <= 0:
            return normalize_window_order(filtered)

        extras: List[int] = []
        seen_submaps: Set[int] = set(frame_to_submap.get(kf, active_submap_id) for kf in filtered)
        for submap_id in sorted(allowed):
            if submap_id in seen_submaps:
                continue
            record = self.submaps.get(submap_id)
            if record is None:
                continue
            extras.extend(record.kf_ids[-self.overlap_kfs :])
        merged: List[int] = []
        for kf in filtered + extras:
            if kf not in merged:
                merged.append(kf)
        return normalize_window_order(merged)

    def affected_submaps_from_frames(self, frame_ids: List[int], frame_to_submap: Dict[int, int]) -> List[int]:
        submap_ids = {frame_to_submap.get(int(frame_id), -1) for frame_id in frame_ids}
        submap_ids.discard(-1)
        expanded: Set[int] = set(submap_ids)
        for submap_id in list(submap_ids):
            expanded.update(self.get_active_submaps(submap_id))
        return sorted(expanded)

    def rebuild_gaussian_indices(self, anchor_submap) -> None:
        for record in self.submaps.values():
            record.gaussian_indices.clear()
        if anchor_submap is None:
            return
        if hasattr(anchor_submap, "detach"):
            anchor_submap = anchor_submap.detach().cpu().numpy()
        anchor_submap = np.asarray(anchor_submap, dtype=np.int32).reshape(-1)
        for gaussian_idx, submap_id in enumerate(anchor_submap.tolist()):
            record = self.submaps.get(int(submap_id))
            if record is not None:
                record.gaussian_indices.add(int(gaussian_idx))

    def mark_refined(self, submap_ids: List[int], step: int) -> None:
        for submap_id in submap_ids:
            record = self.submaps.get(int(submap_id))
            if record is not None:
                record.last_refine_step = int(step)
