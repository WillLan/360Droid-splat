"""Geometry losses for PanoVGGT point/depth/pose fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid


@dataclass(frozen=True)
class PanoVGGTGeometryLossWeights:
    """Weights for explicit geometry supervision."""

    local_point: float = 1.0
    global_point: float = 0.5
    depth: float = 0.2
    pose_rot: float = 0.1
    pose_trans: float = 0.1
    smooth: float = 0.02


def weights_from_config(config: dict[str, Any]) -> PanoVGGTGeometryLossWeights:
    raw = config.get("Loss", {})
    return PanoVGGTGeometryLossWeights(
        local_point=float(raw.get("local_point_weight", raw.get("local_point", 1.0))),
        global_point=float(raw.get("global_point_weight", raw.get("global_point", 0.5))),
        depth=float(raw.get("depth_weight", raw.get("depth", 0.2))),
        pose_rot=float(raw.get("pose_rot_weight", raw.get("pose_rot", 0.1))),
        pose_trans=float(raw.get("pose_trans_weight", raw.get("pose_trans", 0.1))),
        smooth=float(raw.get("smooth_weight", raw.get("smooth", 0.02))),
    )


def build_erp_local_points(depths: torch.Tensor) -> torch.Tensor:
    """Build ERP local point maps from euclidean-range depth.

    Args:
        depths: Tensor with shape ``B x V x 1 x H x W``.

    Returns:
        Tensor with shape ``B x V x H x W x 3``.
    """

    if depths.ndim != 5 or int(depths.shape[2]) != 1:
        raise ValueError(f"depths must have shape BxVx1xHxW, got {tuple(depths.shape)}")
    b, v, _, h, w = [int(x) for x in depths.shape]
    grid = pixel_grid(h, w, device=depths.device, dtype=depths.dtype)
    bearing = erp_pixel_to_bearing(grid, h, w).to(device=depths.device, dtype=depths.dtype)
    return bearing.view(1, 1, h, w, 3) * depths[:, :, 0].view(b, v, h, w, 1)


def local_points_to_world(local_points: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    """Transform local camera-frame point maps to world coordinates."""

    if local_points.ndim != 5 or int(local_points.shape[-1]) != 3:
        raise ValueError(f"local_points must have shape BxVxHxWx3, got {tuple(local_points.shape)}")
    if poses_c2w.ndim != 4 or tuple(poses_c2w.shape[-2:]) != (4, 4):
        raise ValueError(f"poses_c2w must have shape BxVx4x4, got {tuple(poses_c2w.shape)}")
    rot = poses_c2w[:, :, :3, :3].to(device=local_points.device, dtype=local_points.dtype)
    trans = poses_c2w[:, :, :3, 3].to(device=local_points.device, dtype=local_points.dtype)
    return torch.einsum("bvij,bvhwj->bvhwi", rot, local_points) + trans[:, :, None, None, :]


def transform_world_to_first_camera(points_world: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    """Express world point maps in each sample's first-camera coordinate system."""

    if points_world.ndim != 5 or int(points_world.shape[-1]) != 3:
        raise ValueError(f"points_world must have shape BxVxHxWx3, got {tuple(points_world.shape)}")
    pose0 = poses_c2w[:, 0].to(device=points_world.device, dtype=points_world.dtype)
    rot0 = pose0[:, :3, :3]
    trans0 = pose0[:, :3, 3]
    centered = points_world - trans0[:, None, None, None, :]
    return torch.einsum("bij,bvhwj->bvhwi", rot0.transpose(-1, -2), centered)


def relative_poses_to_first(poses_c2w: torch.Tensor) -> torch.Tensor:
    """Return ``inv(T_0) @ T_i`` for every frame in a clip."""

    if poses_c2w.ndim != 4 or tuple(poses_c2w.shape[-2:]) != (4, 4):
        raise ValueError(f"poses_c2w must have shape BxVx4x4, got {tuple(poses_c2w.shape)}")
    first_w2c = torch.linalg.inv(poses_c2w[:, :1])
    return first_w2c @ poses_c2w


def rotation_geodesic_angle(rot_a: torch.Tensor, rot_b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle in radians between rotation matrices."""

    rel = rot_a.transpose(-1, -2) @ rot_b
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        dim=-1,
    )
    sin = 0.5 * torch.linalg.norm(skew, dim=-1)
    return torch.atan2(sin, cos)


def compose_geometry_mask(sample: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build ``valid_depth & ~sky_mask`` as ``B x V x 1 x H x W`` float mask."""

    depths = sample.get("depths")
    if not torch.is_tensor(depths):
        raise ValueError("sample['depths'] is required for geometry fine-tuning.")
    valid = sample.get("valid_depth")
    if torch.is_tensor(valid):
        mask = valid.to(device=device).bool()
    else:
        mask = torch.isfinite(depths.to(device=device)) & (depths.to(device=device) > 0.0)
    sky = sample.get("sky_mask")
    if torch.is_tensor(sky):
        mask = mask & ~sky.to(device=device).bool()
    return mask.to(dtype=dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, *, default: float = 0.0) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_t = mask.to(device=values.device, dtype=values.dtype)
    while mask_t.ndim < values.ndim:
        mask_t = mask_t.unsqueeze(-1)
    denom = mask_t.sum()
    if bool(denom.detach() <= 0):
        return values.new_tensor(float(default))
    return (values * mask_t).sum() / denom.clamp_min(1.0)


def _resize_depth_like(pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
    if pred_depth.ndim != 5:
        raise ValueError(f"pred depth must have shape BxVx1xHxW, got {tuple(pred_depth.shape)}")
    if tuple(pred_depth.shape[-2:]) == tuple(target_depth.shape[-2:]):
        return pred_depth
    b, v = int(pred_depth.shape[0]), int(pred_depth.shape[1])
    flat = pred_depth.reshape(b * v, 1, pred_depth.shape[-2], pred_depth.shape[-1])
    flat = F.interpolate(flat.float(), size=target_depth.shape[-2:], mode="bilinear", align_corners=False)
    return flat.view(b, v, 1, target_depth.shape[-2], target_depth.shape[-1]).to(dtype=pred_depth.dtype)


def _resize_points_like(points: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
    if points.ndim != 5 or int(points.shape[-1]) != 3:
        raise ValueError(f"points must have shape BxVxHxWx3, got {tuple(points.shape)}")
    if tuple(points.shape[2:4]) == tuple(target_depth.shape[-2:]):
        return points
    b, v = int(points.shape[0]), int(points.shape[1])
    flat = points.permute(0, 1, 4, 2, 3).reshape(b * v, 3, points.shape[2], points.shape[3])
    flat = F.interpolate(flat.float(), size=target_depth.shape[-2:], mode="bilinear", align_corners=False)
    return flat.view(b, v, 3, target_depth.shape[-2], target_depth.shape[-1]).permute(0, 1, 3, 4, 2).to(dtype=points.dtype)


def _point_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    scale = gt_depth[:, :, 0].clamp_min(1.0).unsqueeze(-1)
    diff = (pred - gt) / scale
    values = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.05, reduction="none").sum(dim=-1)
    return _masked_mean(values, mask[:, :, 0])


def _point_l1_metric(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    scale = gt_depth[:, :, 0].clamp_min(1.0)
    values = (pred - gt).norm(dim=-1) / scale
    return _masked_mean(values, mask[:, :, 0])


def _depth_smoothness(log_depth: torch.Tensor, images: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
    dx = (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs()
    dy = (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs()
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    if torch.is_tensor(images):
        gray = images.to(device=log_depth.device, dtype=log_depth.dtype).mean(dim=2, keepdim=True)
        if tuple(gray.shape[-2:]) != tuple(log_depth.shape[-2:]):
            b, v = int(gray.shape[0]), int(gray.shape[1])
            flat = gray.reshape(b * v, 1, gray.shape[-2], gray.shape[-1])
            flat = F.interpolate(flat, size=log_depth.shape[-2:], mode="bilinear", align_corners=False)
            gray = flat.view(b, v, 1, log_depth.shape[-2], log_depth.shape[-1])
        wx = torch.exp(-10.0 * (gray[..., :, 1:] - gray[..., :, :-1]).abs())
        wy = torch.exp(-10.0 * (gray[..., 1:, :] - gray[..., :-1, :]).abs())
        dx = dx * wx
        dy = dy * wy
    return 0.5 * (_masked_mean(dx, mask_x) + _masked_mean(dy, mask_y))


def _pose_losses(pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_pred = relative_poses_to_first(pred_poses)
    rel_gt = relative_poses_to_first(gt_poses.to(device=pred_poses.device, dtype=pred_poses.dtype))
    if int(pred_poses.shape[1]) <= 1:
        zero = pred_poses.new_tensor(0.0)
        return zero, zero, zero, zero
    rot_angle = rotation_geodesic_angle(rel_pred[:, 1:, :3, :3], rel_gt[:, 1:, :3, :3])
    trans_error = (rel_pred[:, 1:, :3, 3] - rel_gt[:, 1:, :3, 3]).norm(dim=-1)
    rot_loss = F.smooth_l1_loss(rot_angle, torch.zeros_like(rot_angle), beta=math.radians(1.0), reduction="mean")
    trans_loss = F.smooth_l1_loss(trans_error, torch.zeros_like(trans_error), beta=0.01, reduction="mean")
    return rot_loss, trans_loss, rot_angle.mean(), trans_error.mean()


def panovggt_geometry_loss(
    pred: dict[str, torch.Tensor | None],
    sample: dict[str, Any],
    weights: PanoVGGTGeometryLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked explicit geometry losses for PanoVGGT fine-tuning."""

    if not torch.is_tensor(pred.get("depth")):
        raise ValueError("pred['depth'] is required.")
    if not torch.is_tensor(pred.get("camera_poses")):
        raise ValueError("pred['camera_poses'] is required.")
    gt_depth = sample["depths"].to(device=pred["depth"].device, dtype=pred["depth"].dtype)  # type: ignore[index, union-attr]
    gt_pose = sample["poses_c2w"].to(device=gt_depth.device, dtype=gt_depth.dtype)
    pred_depth = _resize_depth_like(pred["depth"].clamp_min(1.0e-6), gt_depth)  # type: ignore[union-attr]
    pred_pose = pred["camera_poses"].to(device=gt_depth.device, dtype=gt_depth.dtype)  # type: ignore[union-attr]
    mask = compose_geometry_mask(sample, device=gt_depth.device, dtype=gt_depth.dtype)

    gt_local = build_erp_local_points(gt_depth)
    pred_local_raw = pred.get("local_points")
    if torch.is_tensor(pred_local_raw):
        pred_local = _resize_points_like(pred_local_raw.to(device=gt_depth.device, dtype=gt_depth.dtype), gt_depth)
    else:
        pred_local = build_erp_local_points(pred_depth)

    gt_world = local_points_to_world(gt_local, gt_pose)
    pred_world_raw = pred.get("world_points")
    if torch.is_tensor(pred_world_raw):
        pred_world = _resize_points_like(pred_world_raw.to(device=gt_depth.device, dtype=gt_depth.dtype), gt_depth)
    else:
        pred_world = local_points_to_world(pred_local, pred_pose)

    gt_first = transform_world_to_first_camera(gt_world, gt_pose)
    pred_first = transform_world_to_first_camera(pred_world, pred_pose)

    global_raw = pred.get("global_points")
    if torch.is_tensor(global_raw):
        pred_global = _resize_points_like(global_raw.to(device=gt_depth.device, dtype=gt_depth.dtype), gt_depth)
        pred_global_first = transform_world_to_first_camera(pred_global, pred_pose)
    else:
        pred_global_first = pred_first

    log_depth_diff = torch.log(pred_depth.clamp_min(1.0e-6)) - torch.log(gt_depth.clamp_min(1.0e-6))
    depth_loss = _masked_mean(
        F.smooth_l1_loss(log_depth_diff, torch.zeros_like(log_depth_diff), beta=0.05, reduction="none"),
        mask,
    )
    abs_rel_depth = _masked_mean((pred_depth - gt_depth).abs() / gt_depth.abs().clamp_min(1.0), mask)

    local_loss = _point_loss(pred_local, gt_local, mask, gt_depth)
    global_loss = _point_loss(pred_global_first, gt_first, mask, gt_depth)
    local_l1 = _point_l1_metric(pred_local, gt_local, mask, gt_depth)
    global_l1 = _point_l1_metric(pred_global_first, gt_first, mask, gt_depth)

    pose_rot_loss, pose_trans_loss, pose_rot_metric, pose_trans_metric = _pose_losses(pred_pose, gt_pose)
    smooth_loss = _depth_smoothness(torch.log(pred_depth.clamp_min(1.0e-6)), sample.get("images"), mask)

    total = (
        float(weights.local_point) * local_loss
        + float(weights.global_point) * global_loss
        + float(weights.depth) * depth_loss
        + float(weights.pose_rot) * pose_rot_loss
        + float(weights.pose_trans) * pose_trans_loss
        + float(weights.smooth) * smooth_loss
    )
    metrics = {
        "total_loss": total.detach(),
        "depth_loss": depth_loss.detach(),
        "local_point_loss": local_loss.detach(),
        "global_point_loss": global_loss.detach(),
        "pose_rot_loss": pose_rot_loss.detach(),
        "pose_trans_loss": pose_trans_loss.detach(),
        "smooth_loss": smooth_loss.detach(),
        "abs_rel_depth": abs_rel_depth.detach(),
        "local_point_l1": local_l1.detach(),
        "global_point_l1": global_l1.detach(),
        "pose_rpe_rot_deg": (pose_rot_metric.detach() * (180.0 / math.pi)),
        "pose_rpe_trans": pose_trans_metric.detach(),
        "valid_ratio": mask.detach().mean(),
    }
    return total, metrics
