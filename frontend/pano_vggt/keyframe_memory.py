"""Small keyframe memory for future PanoVGGT-M3 current-to-history matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .factor_graph import DenseSphereFactor


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


@dataclass
class KeyframeCorrespondenceEdge:
    """Persistent dense correspondence edge between two keyframes."""

    src_kf_id: int
    tgt_kf_id: int
    edge_type: str
    factor: DenseSphereFactor
    metrics: dict[str, float]
    created_index: int = 0


class KeyframeCorrespondenceGraph:
    """Sparse persistent keyframe correspondence graph."""

    def __init__(self, max_edges: int = 256) -> None:
        self.max_edges = max(1, int(max_edges))
        self._edges: list[KeyframeCorrespondenceEdge] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._edges)

    @property
    def edges(self) -> tuple[KeyframeCorrespondenceEdge, ...]:
        return tuple(self._edges)

    def clear(self) -> None:
        self._edges.clear()
        self._next_index = 0

    def add_edge(self, edge: KeyframeCorrespondenceEdge) -> None:
        keyed = (
            int(edge.src_kf_id),
            int(edge.tgt_kf_id),
            str(edge.edge_type),
        )
        self._edges = [
            item
            for item in self._edges
            if (int(item.src_kf_id), int(item.tgt_kf_id), str(item.edge_type)) != keyed
        ]
        edge.created_index = self._next_index
        self._next_index += 1
        self._edges.append(edge)
        self._trim()

    def edges_for_nodes(self, node_ids: set[int] | list[int] | tuple[int, ...]) -> tuple[KeyframeCorrespondenceEdge, ...]:
        nodes = {int(node_id) for node_id in node_ids}
        return tuple(
            edge
            for edge in self._edges
            if int(edge.src_kf_id) in nodes and int(edge.tgt_kf_id) in nodes
        )

    def recent_node_ids(self, max_nodes: int) -> tuple[int, ...]:
        out: list[int] = []
        for edge in reversed(self._edges):
            for frame_id in (int(edge.src_kf_id), int(edge.tgt_kf_id)):
                if frame_id not in out:
                    out.append(frame_id)
                if len(out) >= int(max_nodes):
                    return tuple(reversed(out))
        return tuple(reversed(out))

    def metrics(self) -> dict[str, float]:
        if not self._edges:
            return {
                "keyframe_graph_edges": 0.0,
                "keyframe_graph_adjacent_edges": 0.0,
                "keyframe_graph_retrieval_edges": 0.0,
                "keyframe_graph_loop_edges": 0.0,
            }
        adjacent = sum(int(edge.edge_type == "adjacent") for edge in self._edges)
        retrieval = sum(int(edge.edge_type == "retrieval") for edge in self._edges)
        loop = sum(int(edge.edge_type == "loop") for edge in self._edges)
        valid = [float(edge.metrics.get("valid_factors", 0.0)) for edge in self._edges]
        ratios = [float(edge.metrics.get("valid_factor_ratio", 0.0)) for edge in self._edges]
        weights = [float(edge.metrics.get("mean_weight", 0.0)) for edge in self._edges]
        return {
            "keyframe_graph_edges": float(len(self._edges)),
            "keyframe_graph_adjacent_edges": float(adjacent),
            "keyframe_graph_retrieval_edges": float(retrieval),
            "keyframe_graph_loop_edges": float(loop),
            "keyframe_graph_mean_valid_factors": float(sum(valid) / len(valid)),
            "keyframe_graph_mean_valid_ratio": float(sum(ratios) / len(ratios)),
            "keyframe_graph_mean_weight": float(sum(weights) / len(weights)),
        }

    def _trim(self) -> None:
        if len(self._edges) <= self.max_edges:
            return
        self._edges = self._edges[-self.max_edges :]


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

    def get(self, frame_id: int) -> KeyframeRecord | None:
        fid = int(frame_id)
        for record in self._records:
            if int(record.frame_id) == fid:
                return record
        return None

    def recent(self, count: int | None = None) -> tuple[KeyframeRecord, ...]:
        if count is None:
            return tuple(self._records)
        return tuple(self._records[-max(0, int(count)) :])

    def update_poses(self, updates: dict[int, torch.Tensor]) -> None:
        if not updates:
            return
        for record in self._records:
            fid = int(record.frame_id)
            pose = updates.get(fid)
            if pose is None:
                continue
            old_pose = record.pose_c2w.detach().float()
            new_pose = pose.detach().cpu().float()
            if tuple(old_pose.shape) == (4, 4) and tuple(new_pose.shape) == (4, 4) and record.global_points is not None:
                correction = new_pose @ torch.linalg.inv(old_pose)
                points = record.global_points.detach().float()
                rot = correction[:3, :3].to(points)
                trans = correction[:3, 3].to(points)
                record.global_points = (torch.matmul(points, rot.T) + trans).detach().cpu().float()
            record.pose_c2w = new_pose

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
