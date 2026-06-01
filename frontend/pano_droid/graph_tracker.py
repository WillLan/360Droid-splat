"""Runtime DROID-style graph tracker for PanoDROID SLAM inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image
from .model import PanoDroidModel


def _relative_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


class PanoDroidGraphTracker(PanoDROIDFrontend):
    """Stateful graph tracker that mirrors the graph training path at inference."""

    def __init__(
        self,
        model: Optional[PanoDroidModel] = None,
        *,
        device: Optional[str] = None,
        window_size: int = 5,
        temporal_radius: int = 2,
        max_factors: int = 24,
        keyframe_threshold: float = 0.55,
        force_keyframe_interval: int = 10,
        num_updates: Optional[int] = None,
        ba_iters_per_update: int = 2,
        ba_sample_stride: int = 1,
        fixed_frames: int = 1,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model or PanoDroidModel()).to(self.device)
        self.window_size = max(2, int(window_size))
        self.temporal_radius = max(1, int(temporal_radius))
        self.max_factors = max(1, int(max_factors))
        self.keyframe_threshold = float(keyframe_threshold)
        self.force_keyframe_interval = int(force_keyframe_interval)
        self.num_updates = num_updates
        self.ba_iters_per_update = int(ba_iters_per_update)
        self.ba_sample_stride = int(ba_sample_stride)
        self.fixed_frames = max(1, int(fixed_frames))
        self.reset()

    def initialize(self, sequence_meta: dict) -> None:
        device = sequence_meta.get("device")
        if device is not None and torch.device(device) != self.device:
            self.device = torch.device(device)
            self.model.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.images: list[torch.Tensor] = []
        self.frame_ids: list[int] = []
        self.timestamps: list[float] = []
        self.metas: list[dict | None] = []
        self.poses_c2w: list[torch.Tensor] = []
        self.inverse_depth_low: Optional[torch.Tensor] = None
        self.last_keyframe_id: Optional[int] = None

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        model_cfg = payload.get("config", {}).get("Model")
        if model_cfg:
            self.model = PanoDroidModel(**model_cfg).to(self.device)
        load_checkpoint(path, self.model, map_location=self.device, strict=False)
        self.model.eval()

    def _append_frame(self, frame: PanoFrame, image: torch.Tensor) -> None:
        self.images.append(image.detach())
        self.frame_ids.append(int(frame.frame_id))
        self.timestamps.append(float(frame.timestamp))
        self.metas.append(frame.meta)
        if not self.poses_c2w:
            self.poses_c2w.append(torch.eye(4, device=self.device, dtype=image.dtype))
        else:
            self.poses_c2w.append(self._predict_next_pose())

        while len(self.images) > self.window_size:
            self.images.pop(0)
            self.frame_ids.pop(0)
            self.timestamps.pop(0)
            self.metas.pop(0)
            self.poses_c2w.pop(0)
            if self.inverse_depth_low is not None and self.inverse_depth_low.shape[0] >= len(self.images):
                self.inverse_depth_low = self.inverse_depth_low[1:].detach()

    def _predict_next_pose(self) -> torch.Tensor:
        if len(self.poses_c2w) < 2:
            return self.poses_c2w[-1].detach().clone()
        prev2 = self.poses_c2w[-2]
        prev1 = self.poses_c2w[-1]
        pred = prev1.detach().clone()
        pred[:3, 3] = prev1[:3, 3] + (prev1[:3, 3] - prev2[:3, 3])
        pred[:3, :3] = prev1[:3, :3]
        return pred

    def _build_edges(self, n_frames: int) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for i in range(n_frames):
            for d in range(1, self.temporal_radius + 1):
                j = i + d
                if j >= n_frames:
                    continue
                edges.append((i, j))
                edges.append((j, i))
        if not edges and n_frames >= 2:
            edges = [(n_frames - 2, n_frames - 1), (n_frames - 1, n_frames - 2)]
        if len(edges) > self.max_factors:
            edges = sorted(edges, key=lambda e: (max(e), min(e)), reverse=True)[: self.max_factors]
            edges = sorted(edges)
        return edges

    def _initial_inverse_depth(self, n_frames: int) -> Optional[torch.Tensor]:
        if self.inverse_depth_low is None:
            return None
        inv = self.inverse_depth_low.detach()
        if inv.shape[0] == n_frames:
            return inv.unsqueeze(0)
        if inv.shape[0] == n_frames - 1:
            inv_new = inv[-1:].clone()
            return torch.cat([inv, inv_new], dim=0).unsqueeze(0)
        return None

    def _current_edge_average(
        self,
        pred: dict,
        key: str,
        *,
        current_local_idx: int,
    ) -> Optional[torch.Tensor]:
        value = pred.get(key)
        if value is None:
            return None
        ii = pred["edge_index_i"].to(value.device)
        mask = ii == int(current_local_idx)
        if bool(mask.any()):
            return value[0, mask].mean(dim=0).detach()
        return value[0, -1].detach()

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = ensure_chw_image(frame.image).to(self.device)
        self.model.eval()
        self._append_frame(frame, image)

        if len(self.images) < 2:
            self.last_keyframe_id = int(frame.frame_id)
            return FrontendOutput(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                pose_c2w=self.poses_c2w[-1].detach().cpu(),
                relative_pose=None,
                pose_confidence=1.0,
                inverse_depth=None,
                depth_confidence=None,
                spherical_flow=None,
                keyframe_score=1.0,
                is_keyframe=True,
                ba_residual=None,
                tracking_status="initialized",
            )

        n = len(self.images)
        edges = self._build_edges(n)
        images = torch.stack(self.images, dim=0).unsqueeze(0)
        init_poses = torch.stack(self.poses_c2w, dim=0).unsqueeze(0).to(image)
        init_inv = self._initial_inverse_depth(n)
        if init_inv is not None:
            init_inv = init_inv.to(device=self.device, dtype=image.dtype)

        with torch.no_grad():
            pred = self.model.forward_graph(
                images,
                edges=edges,
                num_updates=self.num_updates or self.model.update_iters,
                poses_c2w=None,
                init_poses_c2w=init_poses,
                init_inverse_depth=init_inv,
                ba_iters_per_update=self.ba_iters_per_update,
                fixed_frames=min(self.fixed_frames, n),
                ba_sample_stride=self.ba_sample_stride,
            )

        refined = pred["refined_poses_c2w"][0].detach()
        self.poses_c2w = [refined[i].clone() for i in range(n)]
        self.inverse_depth_low = pred["refined_inverse_depth"][0].detach()

        pose_cur = self.poses_c2w[-1]
        pose_prev = self.poses_c2w[-2]
        relative_pose = _relative_from_c2w(pose_prev.unsqueeze(0), pose_cur.unsqueeze(0))[0]
        inv_full = pred.get("refined_inverse_depth_full")
        if inv_full is not None:
            inverse_depth = inv_full[0, -1].detach()
        else:
            inverse_depth = self.inverse_depth_low[-1].detach()
            inverse_depth = F.interpolate(
                inverse_depth.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[0]

        depth_conf = self._current_edge_average(pred, "depth_confidence", current_local_idx=n - 1)
        flow = self._current_edge_average(pred, "spherical_flow", current_local_idx=n - 1)
        if depth_conf is None:
            depth_conf = torch.ones_like(inverse_depth)
        pose_conf = float(depth_conf.mean().detach().cpu())

        key_scores = pred["keyframe_score"][0].detach()
        edge_i = pred["edge_index_i"]
        edge_j = pred["edge_index_j"]
        key_mask = (edge_i == n - 1) | (edge_j == n - 1)
        key_score = float(key_scores[key_mask].mean().cpu()) if bool(key_mask.any()) else float(key_scores.mean().cpu())
        gap = (
            int(frame.frame_id) - int(self.last_keyframe_id)
            if self.last_keyframe_id is not None
            else self.force_keyframe_interval
        )
        is_keyframe = key_score >= self.keyframe_threshold or gap >= self.force_keyframe_interval
        if is_keyframe:
            self.last_keyframe_id = int(frame.frame_id)

        residual = pred["residual_steps"][:, -1].detach().norm(dim=-1).mean()
        return FrontendOutput(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pose_c2w=pose_cur.detach().cpu(),
            relative_pose=relative_pose.detach().cpu(),
            pose_confidence=pose_conf,
            inverse_depth=inverse_depth.detach().cpu(),
            depth_confidence=depth_conf.detach().cpu(),
            spherical_flow=flow.detach().cpu() if flow is not None else None,
            keyframe_score=key_score,
            is_keyframe=bool(is_keyframe),
            ba_residual=float(residual.cpu()),
            tracking_status="tracked_graph",
        )
