"""Runtime DROID-style graph tracker for PanoDROID SLAM inference."""

from __future__ import annotations

from typing import Optional

import torch

from .checkpoint import load_checkpoint
from .factor_graph import PanoFactorGraph
from .interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image
from .model import PanoDroidModel


class PanoDroidGraphTracker(PanoDROIDFrontend):
    """Stateful tracker whose optimization is delegated to ``PanoFactorGraph``."""

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
        self.last_keyframe_id: Optional[int] = None
        self.graph = self._make_graph()

    def _make_graph(self) -> PanoFactorGraph:
        return PanoFactorGraph(
            self.model,
            device=self.device,
            window_size=self.window_size,
            temporal_radius=self.temporal_radius,
            max_factors=self.max_factors,
            num_updates=self.num_updates,
            ba_iters_per_update=self.ba_iters_per_update,
            ba_sample_stride=self.ba_sample_stride,
            fixed_frames=self.fixed_frames,
        )

    def initialize(self, sequence_meta: dict) -> None:
        device = sequence_meta.get("device")
        if device is not None and torch.device(device) != self.device:
            self.device = torch.device(device)
            self.model.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.last_keyframe_id = None
        self.graph = self._make_graph()

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        model_cfg = payload.get("config", {}).get("Model")
        if model_cfg:
            self.model = PanoDroidModel(**model_cfg).to(self.device)
        load_checkpoint(path, self.model, map_location=self.device, strict=False)
        self.model.eval()
        self.reset()

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = ensure_chw_image(frame.image).to(self.device)
        self.model.eval()
        self.graph.add_frame(frame, image)

        if self.graph.n_frames < 2:
            self.last_keyframe_id = int(frame.frame_id)
            return FrontendOutput(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                pose_c2w=self.graph.poses_c2w[-1].detach().cpu(),
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

        with torch.no_grad():
            out = self.graph.update()
        if out is None:
            raise RuntimeError("PanoFactorGraph failed to produce an output.")

        gap = (
            int(frame.frame_id) - int(self.last_keyframe_id)
            if self.last_keyframe_id is not None
            else self.force_keyframe_interval
        )
        is_keyframe = out.keyframe_score >= self.keyframe_threshold or gap >= self.force_keyframe_interval
        if is_keyframe:
            self.last_keyframe_id = int(frame.frame_id)

        pose_conf = float(out.depth_confidence.mean().detach().cpu())
        return FrontendOutput(
            frame_id=out.frame_id,
            timestamp=out.timestamp,
            pose_c2w=out.pose_c2w.detach().cpu(),
            relative_pose=out.relative_pose.detach().cpu(),
            pose_confidence=pose_conf,
            inverse_depth=out.inverse_depth.detach().cpu(),
            depth_confidence=out.depth_confidence.detach().cpu(),
            spherical_flow=out.spherical_flow.detach().cpu() if out.spherical_flow is not None else None,
            keyframe_score=out.keyframe_score,
            is_keyframe=bool(is_keyframe),
            ba_residual=out.ba_residual,
            tracking_status="tracked_graph",
        )
