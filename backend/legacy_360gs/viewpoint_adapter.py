"""Adapters from the current frontend API to legacy 360GS-SLAM viewpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame, ensure_chw_image
from mapping.gaussian_initializer import GaussianInitializer


@dataclass
class LegacyViewpointBundle:
    """CPU payload sent from the current online coordinator to the legacy backend."""

    frame_id: int
    viewpoint: Any
    depth_map: torch.Tensor
    valid_mask: torch.Tensor
    sky_mask: torch.Tensor
    pose_w2c: torch.Tensor


class LightweightPanoramaViewpoint:
    """Small viewpoint substitute for tests and fake backend runs.

    The real legacy backend receives ``utils.camera_utils.PanoramaCamera``.  This
    class keeps the same high-value attributes without importing CUDA renderer
    dependencies during unit tests.
    """

    def __init__(
        self,
        *,
        uid: int,
        image: torch.Tensor,
        depth: torch.Tensor,
        gt_T: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        self.uid = int(uid)
        self.original_image = image.detach().cpu().float()
        self.depth = depth.detach().cpu().float()
        self.mono_depth = depth.detach().cpu().float()
        self.R_gt = gt_T.detach().cpu().float()[:3, :3]
        self.T_gt = gt_T.detach().cpu().float()[:3, 3]
        self.R = torch.eye(3)
        self.T = torch.zeros(3)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.cam_rot_delta = torch.nn.Parameter(torch.zeros(3))
        self.cam_trans_delta = torch.nn.Parameter(torch.zeros(3))
        self.grad_mask = None
        self.erp_region_masks = None
        self.erp_sky_mask = None
        self.mdl_insert_mask = None
        self.mdl_overlap = None
        self.submap_id = -1
        self.config_training_overrides = {}

    def update_RT(self, R: torch.Tensor, T: torch.Tensor) -> None:
        self.R = R.detach().cpu().float()
        self.T = T.detach().cpu().float()


class LegacyViewpointAdapter:
    """Build legacy backend keyframe payloads from ``FrontendOutput`` objects."""

    def __init__(self, config: dict, *, use_legacy_camera: bool = True) -> None:
        self.config = config
        self.use_legacy_camera = bool(use_legacy_camera)
        mapping_cfg = config.get("Mapping", {})
        self.depth_min = float(mapping_cfg.get("depth_min", 0.05))
        self.depth_max = float(mapping_cfg.get("depth_max", 1.0e4))
        self.min_confidence = float(mapping_cfg.get("min_depth_confidence", 0.15))
        self.face_w = int(config.get("LegacyOnlineBackend", {}).get("face_w", 256))
        self.face_zfar = float(config.get("LegacyOnlineBackend", {}).get("face_zfar", 500.0))
        self.sky_helper = GaussianInitializer(
            max_seeds_per_keyframe=0,
            min_confidence=self.min_confidence,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            sky_mask_enable=bool(mapping_cfg.get("sky_mask_enable", False)),
            sky_mask_top_ratio=float(mapping_cfg.get("sky_mask_top_ratio", 0.58)),
            sky_mask_min_blue=float(mapping_cfg.get("sky_mask_min_blue", 0.35)),
            sky_mask_blue_margin=float(mapping_cfg.get("sky_mask_blue_margin", 0.05)),
            sky_mask_cloud_brightness=float(mapping_cfg.get("sky_mask_cloud_brightness", 0.72)),
            sky_mask_cloud_saturation=float(mapping_cfg.get("sky_mask_cloud_saturation", 0.22)),
            sky_mask_texture_threshold=float(mapping_cfg.get("sky_mask_texture_threshold", 0.08)),
        )

    def build(self, frame: PanoFrame, output: FrontendOutput) -> LegacyViewpointBundle:
        if output.inverse_depth is None:
            raise ValueError("Legacy backend keyframe requires inverse_depth.")

        image = ensure_chw_image(frame.image).detach().cpu().float()
        inv = output.inverse_depth.detach().cpu().float()
        if inv.ndim == 3:
            inv = inv[0]
        if inv.ndim != 2:
            raise ValueError(f"Expected HxW inverse depth, got {tuple(inv.shape)}")
        _, H, W = image.shape
        if tuple(inv.shape) != (H, W):
            raise ValueError(f"Inverse depth shape {tuple(inv.shape)} does not match image {(H, W)}")

        confidence = output.depth_confidence
        if confidence is None:
            conf = torch.ones_like(inv)
        else:
            conf = confidence.detach().cpu().float()
            if conf.ndim == 3:
                conf = conf[0]
            if tuple(conf.shape) != (H, W):
                raise ValueError(f"Depth confidence shape {tuple(conf.shape)} does not match image {(H, W)}")

        depth = inv.clamp_min(1e-6).reciprocal()
        valid = (
            torch.isfinite(depth)
            & (depth >= self.depth_min)
            & (depth <= self.depth_max)
            & torch.isfinite(conf)
            & (conf >= self.min_confidence)
        )
        if frame.mask is not None:
            mask = frame.mask.detach().cpu().bool()
            if mask.ndim == 3:
                mask = mask[0]
            if tuple(mask.shape) == (H, W):
                valid = valid & mask

        sky = torch.zeros((H, W), dtype=torch.bool)
        if self.sky_helper.sky_mask_enable:
            sky = self.sky_helper._sky_mask_from_image(image)[0].detach().cpu().bool()
            valid = valid & ~sky

        depth_map = torch.where(valid, depth, torch.zeros_like(depth)).float()
        pose_c2w = output.pose_c2w.detach().cpu().float()
        if tuple(pose_c2w.shape) != (4, 4):
            raise ValueError(f"Expected pose_c2w as 4x4, got {tuple(pose_c2w.shape)}")
        pose_w2c = torch.linalg.inv(pose_c2w)
        gt = frame.meta.get("gt_c2w") if isinstance(frame.meta, dict) else None
        gt_T = gt.detach().cpu().float() if isinstance(gt, torch.Tensor) else pose_c2w

        viewpoint = self._make_viewpoint(
            frame_id=int(output.frame_id),
            image=image,
            depth_map=depth_map,
            gt_T=gt_T,
            image_height=H,
            image_width=W,
        )
        viewpoint.update_RT(pose_w2c[:3, :3], pose_w2c[:3, 3])
        viewpoint.erp_sky_mask = sky.numpy().astype(bool)
        viewpoint.erp_region_masks = {"valid": valid.numpy().astype(bool)}
        viewpoint.depth_confidence = conf
        viewpoint.frame_path = frame.meta.get("path") if isinstance(frame.meta, dict) else None
        viewpoint.timestamp = float(frame.timestamp)
        viewpoint.motion_norm_m = float(output.ba_residual or 0.0)
        viewpoint.motion_rot_deg = 0.0
        return LegacyViewpointBundle(
            frame_id=int(output.frame_id),
            viewpoint=viewpoint,
            depth_map=depth_map,
            valid_mask=valid,
            sky_mask=sky,
            pose_w2c=pose_w2c,
        )

    def _make_viewpoint(
        self,
        *,
        frame_id: int,
        image: torch.Tensor,
        depth_map: torch.Tensor,
        gt_T: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> Any:
        if not self.use_legacy_camera:
            return LightweightPanoramaViewpoint(
                uid=frame_id,
                image=image,
                depth=depth_map,
                gt_T=gt_T,
                image_height=image_height,
                image_width=image_width,
            )

        from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
        from backend.legacy_360gs.utils.camera_utils import PanoramaCamera

        fx = image_width / 2.0
        fy = image_height / 2.0
        cx = image_width / 2.0 - 0.5
        cy = image_height / 2.0 - 0.5
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=self.face_zfar,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=image_width,
            H=image_height,
        ).transpose(0, 1)
        return PanoramaCamera(
            frame_id,
            image,
            depth_map,
            depth_map,
            gt_T,
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            360.0,
            180.0,
            image_height,
            image_width,
            face_w=self.face_w,
            face_zfar=self.face_zfar,
            device="cpu",
        )

