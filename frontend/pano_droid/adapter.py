"""SLAM-facing adapter for PanoDROID-MVP."""

from __future__ import annotations

from typing import Optional

import torch

from .checkpoint import load_checkpoint
from .interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image
from .model import PanoDroidModel


class PanoDROIDFrontendAdapter(PanoDROIDFrontend):
    """Stateful tracker that exposes the requested frontend API."""

    def __init__(
        self,
        model: Optional[PanoDroidModel] = None,
        *,
        device: Optional[str] = None,
        keyframe_threshold: float = 0.55,
        force_keyframe_interval: int = 10,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model or PanoDroidModel()).to(self.device)
        self.keyframe_threshold = float(keyframe_threshold)
        self.force_keyframe_interval = int(force_keyframe_interval)
        self.prev_frame: Optional[PanoFrame] = None
        self.pose_c2w = torch.eye(4, device=self.device)
        self.last_keyframe_id: Optional[int] = None

    def initialize(self, sequence_meta: dict) -> None:
        device = sequence_meta.get("device")
        if device is not None and torch.device(device) != self.device:
            self.device = torch.device(device)
            self.model.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.prev_frame = None
        self.pose_c2w = torch.eye(4, device=self.device)
        self.last_keyframe_id = None

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        model_cfg = payload.get("config", {}).get("Model")
        if model_cfg:
            self.model = PanoDroidModel(**model_cfg).to(self.device)
        load_checkpoint(path, self.model, map_location=self.device, strict=True)
        self.model.eval()

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = ensure_chw_image(frame.image).to(self.device)
        if self.prev_frame is None:
            self.prev_frame = PanoFrame(
                image=image.detach().cpu(),
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                mask=frame.mask,
                meta=frame.meta,
            )
            self.last_keyframe_id = int(frame.frame_id)
            return FrontendOutput(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                pose_c2w=self.pose_c2w.detach().cpu(),
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

        prev_image = ensure_chw_image(self.prev_frame.image).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(prev_image.unsqueeze(0), image.unsqueeze(0))
        T_prev_to_cur = pred["relative_pose"][0]
        # If T maps previous camera coordinates to current camera coordinates,
        # c2w_cur = c2w_prev @ inv(T_prev_to_cur).
        self.pose_c2w = self.pose_c2w @ torch.linalg.inv(T_prev_to_cur)
        key_score = float(pred["keyframe_score"][0].detach().cpu())
        gap = (
            int(frame.frame_id) - int(self.last_keyframe_id)
            if self.last_keyframe_id is not None
            else self.force_keyframe_interval
        )
        is_keyframe = key_score >= self.keyframe_threshold or gap >= self.force_keyframe_interval
        if is_keyframe:
            self.last_keyframe_id = int(frame.frame_id)
        pose_conf = float(pred["confidence"].mean().detach().cpu())
        output = FrontendOutput(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pose_c2w=self.pose_c2w.detach().cpu(),
            relative_pose=T_prev_to_cur.detach().cpu(),
            pose_confidence=pose_conf,
            inverse_depth=pred["inverse_depth"][0].detach().cpu(),
            depth_confidence=pred["depth_confidence"][0].detach().cpu(),
            spherical_flow=pred["spherical_flow"][0].detach().cpu(),
            keyframe_score=key_score,
            is_keyframe=bool(is_keyframe),
            ba_residual=None,
            tracking_status="tracked",
        )
        self.prev_frame = PanoFrame(
            image=image.detach().cpu(),
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            mask=frame.mask,
            meta=frame.meta,
        )
        return output


def build_frontend_from_config(config: dict) -> PanoDROIDFrontendAdapter:
    model = PanoDroidModel(**config.get("Model", {}))
    frontend_cfg = config.get("Frontend", {})
    adapter = PanoDROIDFrontendAdapter(
        model,
        keyframe_threshold=float(frontend_cfg.get("keyframe_threshold", 0.55)),
        force_keyframe_interval=int(frontend_cfg.get("force_keyframe_interval", 10)),
    )
    ckpt = frontend_cfg.get("checkpoint")
    if ckpt:
        adapter.load_checkpoint(ckpt)
    return adapter

