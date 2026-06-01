"""Runtime DROID-style factor graph for panoramic tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .interfaces import FrontendOutput, PanoFrame
from .model import PanoDroidModel
from .projective_ops import project_edges
from .spherical_camera import pixel_grid, seam_aware_delta


def relative_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


@dataclass
class FactorGraphOutput:
    frame_id: int
    timestamp: float
    pose_c2w: torch.Tensor
    relative_pose: torch.Tensor
    inverse_depth: torch.Tensor
    depth_confidence: torch.Tensor
    spherical_flow: Optional[torch.Tensor]
    keyframe_score: float
    ba_residual: float
    pred: dict


class PanoFactorGraph:
    """Persistent active/inactive factor graph used by runtime tracking."""

    def __init__(
        self,
        model: PanoDroidModel,
        *,
        device: torch.device,
        window_size: int = 5,
        temporal_radius: int = 2,
        max_factors: int = 24,
        num_updates: Optional[int] = None,
        ba_iters_per_update: int = 2,
        ba_sample_stride: int = 1,
        fixed_frames: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.window_size = max(2, int(window_size))
        self.temporal_radius = max(1, int(temporal_radius))
        self.max_factors = max(1, int(max_factors))
        self.num_updates = num_updates
        self.ba_iters_per_update = int(ba_iters_per_update)
        self.ba_sample_stride = int(ba_sample_stride)
        self.fixed_frames = max(1, int(fixed_frames))
        self.reset()

    def reset(self) -> None:
        self.images: list[torch.Tensor] = []
        self.frame_ids: list[int] = []
        self.timestamps: list[float] = []
        self.poses_c2w: list[torch.Tensor] = []
        self.inverse_depth_low: Optional[torch.Tensor] = None
        self.edges: list[tuple[int, int]] = []
        self.inactive_edges: set[tuple[int, int]] = set()
        self.edge_age: dict[tuple[int, int], int] = {}
        self.edge_hidden: dict[tuple[int, int], torch.Tensor] = {}
        self.target: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.eta: Optional[torch.Tensor] = None
        self.last_pred: Optional[dict] = None
        self.last_output: Optional[FactorGraphOutput] = None

    @property
    def n_frames(self) -> int:
        return len(self.images)

    def add_frame(self, frame: PanoFrame, image: torch.Tensor) -> None:
        self.images.append(image.detach())
        self.frame_ids.append(int(frame.frame_id))
        self.timestamps.append(float(frame.timestamp))
        if not self.poses_c2w:
            self.poses_c2w.append(torch.eye(4, device=self.device, dtype=image.dtype))
        else:
            self.poses_c2w.append(self._predict_next_pose())

        while len(self.images) > self.window_size:
            self.remove_keyframe(0)
        self.add_neighborhood_factors(self.temporal_radius)
        self.add_proximity_factors()
        self._prune_active_factors()

    def _predict_next_pose(self) -> torch.Tensor:
        if len(self.poses_c2w) < 2:
            return self.poses_c2w[-1].detach().clone()
        prev2 = self.poses_c2w[-2]
        prev1 = self.poses_c2w[-1]
        pred = prev1.detach().clone()
        pred[:3, 3] = prev1[:3, 3] + (prev1[:3, 3] - prev2[:3, 3])
        pred[:3, :3] = prev1[:3, :3]
        return pred

    def add_factors(self, edges: list[tuple[int, int]]) -> None:
        n = self.n_frames
        for i, j in edges:
            edge = (int(i), int(j))
            if edge[0] == edge[1] or edge[0] < 0 or edge[1] < 0 or edge[0] >= n or edge[1] >= n:
                continue
            if edge in self.edges:
                continue
            self.edges.append(edge)
            self.edge_age[edge] = 0

    def add_neighborhood_factors(self, radius: int = 2) -> None:
        n = self.n_frames
        edges: list[tuple[int, int]] = []
        for i in range(n):
            for d in range(1, int(radius) + 1):
                j = i + d
                if j >= n:
                    continue
                edges.extend([(i, j), (j, i)])
        self.add_factors(edges)

    def add_proximity_factors(self) -> None:
        n = self.n_frames
        if n < 3 or len(self.edges) >= self.max_factors:
            return
        candidates: list[tuple[float, int, int]] = []
        existing = set(self.edges) | self.inactive_edges
        for i in range(n):
            for j in range(n):
                if i == j or abs(i - j) <= self.temporal_radius or (i, j) in existing:
                    continue
                candidates.append((self._projection_distance(i, j), i, j))
        candidates.sort(key=lambda x: x[0])
        self.add_factors([(i, j) for _, i, j in candidates[: max(0, self.max_factors - len(self.edges))]])

    def _projection_distance(self, i: int, j: int) -> float:
        if self.inverse_depth_low is None or self.inverse_depth_low.shape[0] <= max(i, j):
            ci = self.poses_c2w[i][:3, 3]
            cj = self.poses_c2w[j][:3, 3]
            return float(torch.linalg.norm((ci - cj).float()).detach().cpu())

        inv = self.inverse_depth_low
        h, w = int(inv.shape[-2]), int(inv.shape[-1])
        stride_y = max(1, h // 8)
        stride_x = max(1, w // 16)
        pixels = pixel_grid(h, w, device=inv.device, dtype=inv.dtype)[::stride_y, ::stride_x]
        ii = torch.tensor([i], device=inv.device, dtype=torch.long)
        jj = torch.tensor([j], device=inv.device, dtype=torch.long)
        poses = torch.stack(self.poses_c2w, dim=0).to(device=inv.device, dtype=inv.dtype).unsqueeze(0)
        with torch.no_grad():
            pred = project_edges(poses, inv.unsqueeze(0), ii, jj, height=h, width=w, pixels=pixels)
            src = pixels.view(1, 1, pixels.shape[0], pixels.shape[1], 2)
            score = seam_aware_delta(src, pred, w).norm(dim=-1).mean()
        return float(score.detach().cpu())

    def _prune_active_factors(self) -> None:
        if len(self.edges) <= self.max_factors:
            return
        scored = sorted(self.edges, key=lambda e: (max(e), -self.edge_age.get(e, 0)), reverse=True)
        keep = set(scored[: self.max_factors])
        for edge in list(self.edges):
            if edge not in keep:
                self.inactive_edges.add(edge)
                self.edge_hidden.pop(edge, None)
                self.edge_age.pop(edge, None)
        self.edges = [e for e in self.edges if e in keep]

    def remove_keyframe(self, local_index: int) -> None:
        idx = int(local_index)
        self.images.pop(idx)
        self.frame_ids.pop(idx)
        self.timestamps.pop(idx)
        self.poses_c2w.pop(idx)
        if self.inverse_depth_low is not None and self.inverse_depth_low.shape[0] > idx:
            self.inverse_depth_low = torch.cat(
                [self.inverse_depth_low[:idx], self.inverse_depth_low[idx + 1 :]], dim=0
            ).detach()

        def reindex(edge: tuple[int, int]) -> Optional[tuple[int, int]]:
            i, j = edge
            if i == idx or j == idx:
                return None
            return (i - 1 if i > idx else i, j - 1 if j > idx else j)

        old_hidden = self.edge_hidden
        new_edges: list[tuple[int, int]] = []
        new_age: dict[tuple[int, int], int] = {}
        new_hidden: dict[tuple[int, int], torch.Tensor] = {}
        new_inactive: set[tuple[int, int]] = set()
        for edge in self.edges:
            nxt = reindex(edge)
            if nxt is not None and nxt not in new_edges:
                new_edges.append(nxt)
                new_age[nxt] = self.edge_age.get(edge, 0)
                if edge in old_hidden:
                    new_hidden[nxt] = old_hidden[edge]
        for edge in self.inactive_edges:
            nxt = reindex(edge)
            if nxt is not None:
                new_inactive.add(nxt)
        self.edges = new_edges
        self.edge_age = new_age
        self.edge_hidden = new_hidden
        self.inactive_edges = new_inactive

    def _init_inverse_depth(self) -> Optional[torch.Tensor]:
        n = self.n_frames
        if self.inverse_depth_low is None:
            return None
        inv = self.inverse_depth_low.detach()
        if inv.shape[0] == n:
            return inv.unsqueeze(0)
        if inv.shape[0] == n - 1:
            return torch.cat([inv, inv[-1:].clone()], dim=0).unsqueeze(0)
        return None

    def _init_edge_hidden(self) -> Optional[torch.Tensor]:
        if not self.edges or not self.edge_hidden:
            return None
        vals = []
        for edge in self.edges:
            h = self.edge_hidden.get(edge)
            if h is None:
                return None
            vals.append(h)
        return torch.stack(vals, dim=0).unsqueeze(0)

    def update(self) -> Optional[FactorGraphOutput]:
        if self.n_frames < 2:
            return None
        if not self.edges:
            self.add_neighborhood_factors(1)
        images = torch.stack(self.images, dim=0).unsqueeze(0)
        init_poses = torch.stack(self.poses_c2w, dim=0).unsqueeze(0).to(images)
        init_inv = self._init_inverse_depth()
        if init_inv is not None:
            init_inv = init_inv.to(device=self.device, dtype=images.dtype)
        init_hidden = self._init_edge_hidden()
        if init_hidden is not None:
            init_hidden = init_hidden.to(device=self.device, dtype=images.dtype)

        pred = self.model.forward_graph(
            images,
            edges=self.edges,
            num_updates=self.num_updates or self.model.update_iters,
            poses_c2w=None,
            init_poses_c2w=init_poses,
            init_inverse_depth=init_inv,
            init_edge_hidden=init_hidden,
            ba_iters_per_update=self.ba_iters_per_update,
            fixed_frames=min(self.fixed_frames, self.n_frames),
            ba_sample_stride=self.ba_sample_stride,
        )
        refined = pred["refined_poses_c2w"][0].detach()
        self.poses_c2w = [refined[i].clone() for i in range(self.n_frames)]
        self.inverse_depth_low = pred["refined_inverse_depth"][0].detach()
        self.target = pred.get("target_steps", None)
        self.weight = pred.get("weight_steps", None)
        self.eta = pred.get("damping_steps", None)
        hidden = pred.get("edge_hidden")
        if torch.is_tensor(hidden):
            for idx, edge in enumerate(self.edges):
                self.edge_hidden[edge] = hidden[0, idx].detach()
        for edge in list(self.edge_age):
            self.edge_age[edge] = self.edge_age.get(edge, 0) + 1
        self.last_pred = pred

        pose_cur = self.poses_c2w[-1]
        pose_prev = self.poses_c2w[-2]
        rel = relative_from_c2w(pose_prev.unsqueeze(0), pose_cur.unsqueeze(0))[0]
        inv_full = pred.get("refined_inverse_depth_full")
        if torch.is_tensor(inv_full):
            inv = inv_full[0, -1].detach()
        else:
            inv = F.interpolate(
                self.inverse_depth_low[-1:].detach(),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[0]
        conf = self._edge_average(pred, "depth_confidence", self.n_frames - 1)
        if conf is None:
            conf = torch.ones_like(inv)
        flow = self._edge_average(pred, "spherical_flow", self.n_frames - 1)
        key = self._keyframe_score(pred, self.n_frames - 1)
        residual = pred["residual_steps"][:, -1].detach().norm(dim=-1).mean()
        output = FactorGraphOutput(
            frame_id=self.frame_ids[-1],
            timestamp=self.timestamps[-1],
            pose_c2w=pose_cur,
            relative_pose=rel,
            inverse_depth=inv,
            depth_confidence=conf,
            spherical_flow=flow,
            keyframe_score=key,
            ba_residual=float(residual.detach().cpu()),
            pred=pred,
        )
        self.last_output = output
        return output

    def _edge_average(self, pred: dict, key: str, local_idx: int) -> Optional[torch.Tensor]:
        value = pred.get(key)
        if not torch.is_tensor(value):
            return None
        ii = pred["edge_index_i"].to(value.device)
        mask = ii == int(local_idx)
        if bool(mask.any()):
            return value[0, mask].mean(dim=0).detach()
        return value[0, -1].detach()

    def _keyframe_score(self, pred: dict, local_idx: int) -> float:
        scores = pred["keyframe_score"][0].detach()
        edge_i = pred["edge_index_i"]
        edge_j = pred["edge_index_j"]
        mask = (edge_i == int(local_idx)) | (edge_j == int(local_idx))
        if bool(mask.any()):
            return float(scores[mask].mean().cpu())
        return float(scores.mean().cpu())

    def get_current_output(self, *, is_keyframe: bool, pose_confidence: float) -> FrontendOutput:
        if self.last_output is None:
            raise RuntimeError("PanoFactorGraph has no optimized output yet.")
        out = self.last_output
        return FrontendOutput(
            frame_id=out.frame_id,
            timestamp=out.timestamp,
            pose_c2w=out.pose_c2w.detach().cpu(),
            relative_pose=out.relative_pose.detach().cpu(),
            pose_confidence=float(pose_confidence),
            inverse_depth=out.inverse_depth.detach().cpu(),
            depth_confidence=out.depth_confidence.detach().cpu(),
            spherical_flow=out.spherical_flow.detach().cpu() if out.spherical_flow is not None else None,
            keyframe_score=out.keyframe_score,
            is_keyframe=bool(is_keyframe),
            ba_residual=out.ba_residual,
            tracking_status="tracked_graph",
        )
