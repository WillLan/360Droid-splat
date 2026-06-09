"""Small keyframe memory for future PanoVGGT-M3 current-to-history matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KeyframeRecord:
    frame_id: int
    pose_c2w: torch.Tensor
    depth_low: torch.Tensor
    dense_descriptors: torch.Tensor
    match_confidence: torch.Tensor
    sky_prob: torch.Tensor
    static_confidence: torch.Tensor | None = None
    feature_hw: tuple[int, int] | None = None
    image_hw: tuple[int, int] | None = None
    image: torch.Tensor | None = None
    global_points: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    frozen: bool = False


class KeyframeMemory:
    """Bounded in-memory cache for dense matching side data.

    Frozen records are intended to act as fixed historical anchors during
    history-window BA; they may be used as targets, but should not be optimized
    or pruned by local frontend refinement.
    """

    def __init__(self, max_keyframes: int = 64) -> None:
        self.max_keyframes = max(1, int(max_keyframes))
        self._records: list[KeyframeRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[KeyframeRecord, ...]:
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()

    def add(self, record: KeyframeRecord) -> None:
        self._records = [item for item in self._records if int(item.frame_id) != int(record.frame_id)]
        self._records.append(record)
        self._trim()

    def recent(self, count: int | None = None) -> tuple[KeyframeRecord, ...]:
        if count is None:
            return tuple(self._records)
        return tuple(self._records[-max(0, int(count)) :])

    def current_to_history_edges(self, current_local_index: int, history_count: int | None = None) -> torch.Tensor:
        """Return placeholder local edges from one current frame to cached history."""

        count = len(self.recent(history_count))
        if count == 0:
            return torch.empty(0, 2, dtype=torch.long)
        src = torch.full((count,), int(current_local_index), dtype=torch.long)
        tgt = torch.arange(count, dtype=torch.long)
        return torch.stack([src, tgt], dim=-1)

    def _trim(self) -> None:
        if len(self._records) <= self.max_keyframes:
            return
        frozen = [record for record in self._records if record.frozen]
        live = [record for record in self._records if not record.frozen]
        overflow = max(0, len(frozen) + len(live) - self.max_keyframes)
        self._records = [*frozen, *live[overflow:]]
