"""Adapters from the current frontend API to legacy 360GS-SLAM viewpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from geometry.pose import invert_c2w

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
    world_points: torch.Tensor | None = None
    world_points_confidence: torch.Tensor | None = None
    valid_world_points_mask: torch.Tensor | None = None


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
        self.global_world_points = None
        self.global_world_points_confidence = None
        self.global_world_points_valid_mask = None
        self.global_world_points_required = False
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
        frontend_mode = str(config.get("Frontend", {}).get("mode", "")).lower()
        seed_source = str(
            mapping_cfg.get(
                "seed_source",
                "world_points_only" if frontend_mode == "panovggt_long" else "depth_pose",
            )
        ).lower()
        self.require_world_points = bool(seed_source == "world_points_only" or frontend_mode == "panovggt_long")

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
        depth_valid = (
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
                depth_valid = depth_valid & mask

        sky = torch.zeros((H, W), dtype=torch.bool)
        if self.sky_helper.sky_mask_enable:
            sky = self.sky_helper._sky_mask_from_image(image)[0].detach().cpu().bool()
            depth_valid = depth_valid & ~sky

        world_points, world_conf, world_valid = self._world_points_payload(output, H, W)
        if world_points is None:
            if self.require_world_points:
                raise ValueError("PanoVGGT legacy backend requires global FrontendOutput.world_points.")
            valid = depth_valid
        else:
            world_valid = (
                world_valid
                & torch.isfinite(world_points).all(dim=-1)
                & torch.isfinite(world_conf)
                & (world_conf >= self.min_confidence)
                & ~sky
            )
            if frame.mask is not None:
                mask = frame.mask.detach().cpu().bool()
                if mask.ndim == 3:
                    mask = mask[0]
                if tuple(mask.shape) == (H, W):
                    world_valid = world_valid & mask
            valid = world_valid

        depth_map = torch.where(depth_valid, depth, torch.zeros_like(depth)).float()
        pose_c2w = output.pose_c2w.detach().cpu().float()
        if tuple(pose_c2w.shape) != (4, 4):
            raise ValueError(f"Expected pose_c2w as 4x4, got {tuple(pose_c2w.shape)}")
        pose_w2c = invert_c2w(pose_c2w)
        gt = frame.meta.get("gt_c2w") if isinstance(frame.meta, dict) else None
        gt_c2w = gt.detach().cpu().float() if isinstance(gt, torch.Tensor) else pose_c2w
        gt_T = invert_c2w(gt_c2w)

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
        viewpoint.global_world_points = None if world_points is None else world_points.float()
        viewpoint.global_world_points_confidence = None if world_conf is None else world_conf.float()
        viewpoint.global_world_points_valid_mask = None if world_valid is None else world_valid.bool()
        viewpoint.global_world_points_required = bool(self.require_world_points)
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
            world_points=world_points,
            world_points_confidence=world_conf,
            valid_world_points_mask=world_valid,
        )

    def _world_points_payload(
        self,
        output: FrontendOutput,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        points = output.world_points
        if points is None:
            return None, None, None
        pts = points.detach().cpu().float()
        if pts.ndim == 4 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim == 3 and pts.shape[-1] != 3 and pts.shape[0] == 3:
            pts = pts.permute(1, 2, 0)
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"Expected world_points as HxWx3, got {tuple(pts.shape)}")
        if tuple(pts.shape[:2]) != (height, width):
            raise ValueError(f"World-points shape {tuple(pts.shape[:2])} does not match image {(height, width)}")

        conf = output.world_points_confidence
        if conf is None:
            conf = output.depth_confidence
        if conf is None:
            conf_t = torch.ones((height, width), dtype=torch.float32)
        else:
            conf_t = conf.detach().cpu().float()
            if conf_t.ndim == 3:
                conf_t = conf_t[0]
            if tuple(conf_t.shape) != (height, width):
                raise ValueError(f"World-points confidence shape {tuple(conf_t.shape)} does not match image {(height, width)}")

        mask = output.valid_world_points_mask
        if mask is None:
            mask_t = torch.ones((height, width), dtype=torch.bool)
        else:
            mask_t = mask.detach().cpu().bool()
            if mask_t.ndim == 3:
                mask_t = mask_t[0]
            if tuple(mask_t.shape) != (height, width):
                raise ValueError(f"World-points mask shape {tuple(mask_t.shape)} does not match image {(height, width)}")
        return pts, conf_t, mask_t

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
            depth_map.numpy().astype("float32"),
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
