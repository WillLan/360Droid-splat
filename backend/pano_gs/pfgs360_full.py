"""Panoramic PFGS360 backend primitives.

This module is deliberately independent from the SLAM orchestrator.  It ports
the depth-inlier-aware (DIA) geometry used by PFGS360 to ray-depth panoramic
images while preserving horizontal wrap at the longitude seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    pixel_grid,
)
from geometry.pose import invert_c2w


@dataclass(frozen=True)
class PFGS360DIAConfig:
    tangent_threshold: float = 0.008
    depth_relative_threshold: float = 0.05
    blur_threshold: float = 0.5
    min_depth: float = 0.1
    max_depth: float = 50.0
    gncc_radius: int = 2


def pfgs360_spherical_weight(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sine_weight: float = 0.8,
    floor: float = 0.2,
) -> torch.Tensor:
    rows = torch.arange(height, device=device, dtype=dtype) + 0.5
    weight = float(sine_weight) * torch.sin(rows * math.pi / float(height))
    weight = weight + float(floor)
    return weight.view(1, height, 1).expand(1, height, width)


def pfgs360_photometric_loss(
    render: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    l1_weight: float = 0.8,
    dssim_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PFGS360 spherical 0.8 L1 + 0.2 DSSIM objective."""

    if render.shape != target.shape or render.ndim != 3:
        raise ValueError("PFGS360 photometric tensors must be matching CxHxW")
    _, height, width = render.shape
    weight = pfgs360_spherical_weight(
        height,
        width,
        device=render.device,
        dtype=render.dtype,
    )
    if mask is not None:
        valid = mask.to(device=render.device, dtype=render.dtype)
        if valid.ndim == 2:
            valid = valid.unsqueeze(0)
        weight = weight * valid
    denom = weight.sum().clamp_min(1.0e-8)
    l1 = ((render - target).abs().mean(dim=0, keepdim=True) * weight).sum() / denom

    radius = 1

    def padded(value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value.unsqueeze(0), (radius, radius, 0, 0), mode="circular")
        return F.pad(value, (0, 0, radius, radius), mode="replicate")

    x, y = padded(render), padded(target)
    mu_x = F.avg_pool2d(x, 3, stride=1)
    mu_y = F.avg_pool2d(y, 3, stride=1)
    var_x = F.avg_pool2d(x.square(), 3, stride=1) - mu_x.square()
    var_y = F.avg_pool2d(y.square(), 3, stride=1) - mu_y.square()
    cov = F.avg_pool2d(x * y, 3, stride=1) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1.0e-8)
    dssim_map = ((1.0 - ssim.clamp(-1.0, 1.0)) * 0.5).mean(
        dim=1, keepdim=True
    )[0]
    dssim = (dssim_map * weight).sum() / denom
    total = float(l1_weight) * l1 + float(dssim_weight) * dssim
    return total, {"l1": l1.detach(), "dssim": dssim.detach()}


def _as_depth(depth: torch.Tensor) -> torch.Tensor:
    value = depth
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[0]) != 1:
        raise ValueError(f"Expected ray depth 1xHxW, got {tuple(depth.shape)}")
    return value


def sample_erp_with_wrap(
    image: torch.Tensor,
    pixels_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample an ERP tensor with longitude wrap and latitude clamp."""

    value = image
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or pixels_xy.shape[-1] != 2:
        raise ValueError("ERP sampling expects CxHxW and ...x2 pixels")
    channels, height, width = value.shape
    x = torch.remainder(pixels_xy[..., 0], float(width))
    y_raw = pixels_xy[..., 1]
    valid = torch.isfinite(x) & torch.isfinite(y_raw) & (y_raw >= 0.0) & (
        y_raw <= float(height - 1)
    )
    y = y_raw.clamp(0.0, float(height - 1))
    x0 = torch.floor(x).long()
    x1 = torch.remainder(x0 + 1, width)
    y0 = torch.floor(y).long()
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = (x - x0.to(x.dtype)).unsqueeze(0)
    wy = (y - y0.to(y.dtype)).unsqueeze(0)
    flat = value.reshape(channels, -1)

    def gather(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        index = (iy * width + ix).reshape(-1)
        return flat.index_select(1, index).reshape(channels, *ix.shape)

    v00 = gather(x0, y0)
    v10 = gather(x1, y0)
    v01 = gather(x0, y1)
    v11 = gather(x1, y1)
    sampled = (1.0 - wx) * (1.0 - wy) * v00
    sampled = sampled + wx * (1.0 - wy) * v10
    sampled = sampled + (1.0 - wx) * wy * v01 + wx * wy * v11
    sampled = torch.where(valid.unsqueeze(0), sampled, torch.zeros_like(sampled))
    return sampled, valid


def _world_points(depth: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    depth_1 = _as_depth(depth)
    height, width = int(depth_1.shape[-2]), int(depth_1.shape[-1])
    grid = pixel_grid(
        height,
        width,
        device=depth_1.device,
        dtype=depth_1.dtype,
    ).view(height, width, 2)
    bearing = erp_pixel_to_bearing(grid, height, width)
    points_cam = bearing * depth_1[0].unsqueeze(-1)
    rotation = c2w[:3, :3].to(points_cam)
    translation = c2w[:3, 3].to(points_cam)
    return torch.einsum("ij,hwj->hwi", rotation, points_cam) + translation


def warp_reference_to_current(
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    reference: torch.Tensor,
    reference_c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp a reference ERP tensor into the current view using ray depth."""

    depth = _as_depth(current_depth)
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    world = _world_points(depth, current_c2w)
    ref_w2c = invert_c2w(reference_c2w.to(world))
    ref_cam = torch.einsum("ij,hwj->hwi", ref_w2c[:3, :3], world) + ref_w2c[:3, 3]
    ref_ray_depth = torch.linalg.norm(ref_cam, dim=-1).clamp_min(1.0e-8)
    ref_bearing = ref_cam / ref_ray_depth.unsqueeze(-1)
    pixels = bearing_to_erp_pixel(ref_bearing, height, width)
    warped, valid = sample_erp_with_wrap(reference.to(depth), pixels)
    valid = valid & torch.isfinite(ref_ray_depth)
    return warped, valid, ref_ray_depth.unsqueeze(0)


def panoramic_pair_consistency(
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    reference_depth: torch.Tensor,
    reference_c2w: torch.Tensor,
    *,
    config: PFGS360DIAConfig | None = None,
) -> torch.Tensor:
    """Official-style tangent/depth consistency for panoramic ray depth."""

    cfg = config or PFGS360DIAConfig()
    depth = _as_depth(current_depth)
    ref_depth, projection_valid, projected_ref_depth = warp_reference_to_current(
        depth, current_c2w, _as_depth(reference_depth), reference_c2w
    )
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype).view(
        height, width, 2
    )
    current_bearing = erp_pixel_to_bearing(grid, height, width)

    current_world = _world_points(depth, current_c2w.to(depth))
    ref_w2c = invert_c2w(reference_c2w.to(depth))
    projected_ref_cam = (
        torch.einsum("ij,hwj->hwi", ref_w2c[:3, :3], current_world)
        + ref_w2c[:3, 3]
    )
    projected_ref_bearing = F.normalize(projected_ref_cam, dim=-1, eps=1.0e-8)
    reference_points_cam = projected_ref_bearing * ref_depth[0].unsqueeze(-1)
    reference_pose = reference_c2w.to(depth)
    world_from_reference = (
        torch.einsum(
            "ij,hwj->hwi", reference_pose[:3, :3], reference_points_cam
        )
        + reference_pose[:3, 3]
    )
    current_w2c = invert_c2w(current_c2w.to(depth))
    reproj_cam = (
        torch.einsum("ij,hwj->hwi", current_w2c[:3, :3], world_from_reference)
        + current_w2c[:3, 3]
    )
    reproj_depth = torch.linalg.norm(reproj_cam, dim=-1).clamp_min(1.0e-8)
    reproj_bearing = reproj_cam / reproj_depth.unsqueeze(-1)
    dot = (current_bearing * reproj_bearing).sum(dim=-1).clamp(-1.0, 1.0)
    tangent = 2.0 * torch.sqrt(
        ((1.0 - dot).clamp_min(0.0) / (1.0 + dot).clamp_min(1.0e-8))
    )
    current_rel = (reproj_depth - depth[0]).abs() / depth[0].clamp_min(1.0e-8)
    reference_rel = (ref_depth[0] - projected_ref_depth[0]).abs() / projected_ref_depth[
        0
    ].clamp_min(1.0e-8)
    valid = (
        projection_valid
        & torch.isfinite(depth[0])
        & torch.isfinite(ref_depth[0])
        & (depth[0] >= float(cfg.min_depth))
        & (depth[0] <= float(cfg.max_depth))
        & (ref_depth[0] >= float(cfg.min_depth))
        & (ref_depth[0] <= float(cfg.max_depth))
    )
    return (
        valid
        & (tangent < float(cfg.tangent_threshold))
        & (current_rel < float(cfg.depth_relative_threshold))
        & (reference_rel < float(cfg.depth_relative_threshold))
    ).unsqueeze(0)


def blur_panorama_mask(mask: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    value = mask.float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    padded = F.pad(value.unsqueeze(0), (1, 1, 0, 0), mode="circular")
    padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
    blurred = F.avg_pool2d(padded, kernel_size=3, stride=1)[0]
    return blurred > float(threshold)


def multi_reference_consistency(
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    references: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    config: PFGS360DIAConfig | None = None,
) -> torch.Tensor:
    if len(references) < 2:
        raise ValueError("PFGS360 DIA requires two reference panoramic views")
    cfg = config or PFGS360DIAConfig()
    masks = [
        panoramic_pair_consistency(
            current_depth,
            current_c2w,
            depth,
            pose,
            config=cfg,
        )
        for depth, pose in references[:2]
    ]
    return blur_panorama_mask(masks[0] & masks[1], threshold=cfg.blur_threshold)


def affine_align_depth(
    predicted_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
) -> tuple[torch.Tensor, float, float, int]:
    predicted = _as_depth(predicted_depth).float()
    rendered = _as_depth(rendered_depth).to(predicted)
    valid = mask.bool().to(predicted.device)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    valid = (
        valid
        & torch.isfinite(predicted)
        & torch.isfinite(rendered)
        & (predicted >= float(min_depth))
        & (predicted <= float(max_depth))
        & (rendered >= float(min_depth))
        & (rendered <= float(max_depth))
    )
    count = int(valid.sum().item())
    if count < 2:
        return predicted.clone(), 1.0, 0.0, count
    x = predicted[valid]
    y = rendered[valid]
    x_mean = x.mean()
    y_mean = y.mean()
    variance = (x - x_mean).square().sum().clamp_min(1.0e-8)
    scale = ((x - x_mean) * (y - y_mean)).sum() / variance
    shift = y_mean - scale * x_mean
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        scale = x.new_tensor(1.0)
        shift = x.new_tensor(0.0)
    aligned = (scale * predicted + shift).clamp(float(min_depth), float(max_depth))
    return aligned, float(scale.item()), float(shift.item()), count


def _local_gncc(a: torch.Tensor, b: torch.Tensor, radius: int) -> torch.Tensor:
    if a.ndim == 3:
        a = a.unsqueeze(0)
    if b.ndim == 3:
        b = b.unsqueeze(0)
    kernel = 2 * int(radius) + 1

    def pad(value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (radius, radius, 0, 0), mode="circular")
        return F.pad(value, (0, 0, radius, radius), mode="replicate")

    pa, pb = pad(a), pad(b)
    mean_a = F.avg_pool2d(pa, kernel, stride=1)
    mean_b = F.avg_pool2d(pb, kernel, stride=1)
    mean_ab = F.avg_pool2d(pa * pb, kernel, stride=1)
    var_a = F.avg_pool2d(pa.square(), kernel, stride=1) - mean_a.square()
    var_b = F.avg_pool2d(pb.square(), kernel, stride=1) - mean_b.square()
    cov = mean_ab - mean_a * mean_b
    ncc = cov / torch.sqrt(var_a.clamp_min(1.0e-6) * var_b.clamp_min(1.0e-6))
    return ncc.mean(dim=1, keepdim=True)[0]


def panoramic_patch_better_mask(
    image: torch.Tensor,
    rendered_depth: torch.Tensor,
    aligned_depth: torch.Tensor,
    c2w: torch.Tensor,
    references: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    radius: int = 2,
) -> torch.Tensor:
    """Keep pixels whose aligned-depth warp has better local GNCC."""

    if len(references) < 2:
        raise ValueError("PFGS360 patch filtering requires two reference views")
    current = image if image.ndim == 3 else image[0]
    masks: list[torch.Tensor] = []
    for ref_image, ref_pose in references[:2]:
        rendered_warp, rendered_valid, _ = warp_reference_to_current(
            rendered_depth, c2w, ref_image, ref_pose
        )
        aligned_warp, aligned_valid, _ = warp_reference_to_current(
            aligned_depth, c2w, ref_image, ref_pose
        )
        rendered_score = _local_gncc(current, rendered_warp, radius)
        aligned_score = _local_gncc(current, aligned_warp, radius)
        masks.append(
            (aligned_score > rendered_score)
            & rendered_valid.unsqueeze(0)
            & aligned_valid.unsqueeze(0)
        )
    return masks[0] & masks[1]


def backproject_panorama_depth(
    depth: torch.Tensor,
    image: torch.Tensor,
    c2w: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_1 = _as_depth(depth)
    world = _world_points(depth_1, c2w.to(depth_1))
    rgb = image if image.ndim == 3 else image[0]
    valid = mask.bool()
    if valid.ndim == 3:
        valid = valid[0]
    valid = valid & torch.isfinite(depth_1[0]) & (depth_1[0] > 0.0)
    return world[valid], rgb.permute(1, 2, 0).to(world)[valid]


def voxel_average_points(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    *,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(xyz.shape[0]) == 0:
        return xyz, rgb, torch.zeros(0, 3, device=xyz.device, dtype=torch.long)
    keys = torch.round(xyz / float(voxel_size)).long()
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.bincount(inverse, minlength=int(unique.shape[0])).to(xyz).unsqueeze(1)
    xyz_sum = torch.zeros(unique.shape[0], 3, device=xyz.device, dtype=xyz.dtype)
    rgb_sum = torch.zeros(unique.shape[0], 3, device=rgb.device, dtype=rgb.dtype)
    xyz_sum.index_add_(0, inverse, xyz)
    rgb_sum.index_add_(0, inverse, rgb)
    return xyz_sum / count, rgb_sum / count.to(rgb), unique


class PFGS360FullBackend:
    """Strict PFGS360 CAMERA -> DIA -> JOINT transaction."""

    PARAMETER_LRS = {
        "xyz": 1.6e-4,
        "features": 2.5e-3,
        "sh_rest": 1.25e-4,
        "opacity_logit": 5.0e-2,
        "scaling": 5.0e-3,
        "rotation": 1.0e-3,
    }

    def __init__(
        self,
        mapper,
        settings: dict | None = None,
        *,
        refined_anchor_update: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.mapper = mapper
        self.map = mapper.map
        self.settings = dict(settings or {})
        self.cfg = PFGS360DIAConfig(
            tangent_threshold=float(self.settings.get("tangent_threshold", 0.008)),
            depth_relative_threshold=float(
                self.settings.get("depth_relative_threshold", 0.05)
            ),
            blur_threshold=float(self.settings.get("blur_threshold", 0.5)),
            min_depth=float(self.settings.get("min_depth", 0.1)),
            max_depth=float(self.settings.get("max_depth", 50.0)),
            gncc_radius=int(self.settings.get("gncc_radius", 2)),
        )
        self.device = self.map.xyz.device
        self.dtype = self.map.xyz.dtype
        self.refined_anchor_update = refined_anchor_update

    @property
    def uses_refined_anchor_growth(self) -> bool:
        return str(self.settings.get("growth_source", "raw_depth")).strip().lower() == (
            "refined_anchor"
        )

    @staticmethod
    def _clip_pose_gradients(
        pose_parameters: list[torch.Tensor],
        clip_value: float,
    ) -> None:
        parameters_with_grad = [
            parameter for parameter in pose_parameters if parameter.grad is not None
        ]
        if parameters_with_grad:
            torch.nn.utils.clip_grad_value_(
                parameters_with_grad,
                float(clip_value),
            )

    def _state_storage_device(self) -> torch.device:
        mode = str(self.settings.get("state_storage_device", "cpu")).strip().lower()
        if mode == "cpu":
            return torch.device("cpu")
        if mode in {"map", "cuda"}:
            if self.device.type != "cuda":
                return self.device
            return self.device
        raise ValueError(
            "PFGS360 state_storage_device must be 'cpu', 'map', or 'cuda'"
        )

    def _snapshot(self) -> dict[str, object]:
        state_device = self._state_storage_device()
        return {
            "map": self.map.pfgs360_topology_snapshot(
                parameter_device=state_device,
            ),
            "poses": {
                int(frame_id): {
                    "base": pose.base_c2w.detach().cpu().clone(),
                    "delta": pose.delta.detach().cpu().clone(),
                }
                for frame_id, pose in self.mapper.pose_deltas.items()
            },
            "moments": {
                key: {
                    field: value.detach().clone() if torch.is_tensor(value) else value
                    for field, value in state.items()
                }
                for key, state in dict(
                    getattr(self.mapper, "_pfgs360_gaussian_moments", {}) or {}
                ).items()
            },
            "joint_steps": int(getattr(self.mapper, "_pfgs360_joint_steps", 0)),
        }

    def _restore(self, state: dict[str, object]) -> None:
        self.map.restore_pfgs360_topology_snapshot(dict(state["map"]))
        for frame_id, pose_state in dict(state["poses"]).items():
            pose = self.mapper.pose_deltas.get(int(frame_id))
            if pose is None:
                from backend.pano_gs.pose_param import PoseDelta

                pose = PoseDelta(pose_state["base"]).to(self.device)
                self.mapper.pose_deltas[int(frame_id)] = pose
            pose.rebase(pose_state["base"], preserve_delta=False)
            with torch.no_grad():
                pose.delta.copy_(pose_state["delta"].to(pose.delta))
        self.mapper._pfgs360_gaussian_moments = dict(state["moments"])
        self.mapper._pfgs360_joint_steps = int(state["joint_steps"])
        self.mapper._pending_pfgs360_anchor_admission = None
        self.mapper.optimizer = self.map.make_optimizer(
            lr=float(self.settings.get("fallback_lr", 2.0e-3))
        )

    def _observations(self, frame_ids: list[int] | tuple[int, ...]):
        observations = []
        for frame_id in dict.fromkeys(int(value) for value in frame_ids):
            observation = self.mapper.observations.get(frame_id)
            if observation is not None and frame_id in self.mapper.pose_deltas:
                observations.append(observation)
        observations.sort(key=lambda value: int(value.frame_id))
        return observations

    def _remap_moments(self) -> None:
        mapping = getattr(self.map, "_pfgs360_last_topology_mapping", None)
        moments = dict(getattr(self.mapper, "_pfgs360_gaussian_moments", {}) or {})
        if not torch.is_tensor(mapping) or not moments:
            return
        for name, state in moments.items():
            for field in ("exp_avg", "exp_avg_sq"):
                value = state.get(field)
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                state_mapping = mapping.detach().to(
                    device=value.device,
                    dtype=torch.long,
                )
                valid = state_mapping >= 0
                output = value.new_zeros((int(state_mapping.numel()), *value.shape[1:]))
                if bool(valid.any()):
                    output[valid] = value.index_select(0, state_mapping[valid])
                state[field] = output
        self.mapper._pfgs360_gaussian_moments = moments

    def _load_moments(self, optimizer: torch.optim.Optimizer) -> None:
        saved = dict(getattr(self.mapper, "_pfgs360_gaussian_moments", {}) or {})
        for group in optimizer.param_groups:
            name = str(group.get("name", ""))
            if name == "poses" or name not in saved or len(group["params"]) != 1:
                continue
            parameter = group["params"][0]
            source = saved[name]
            if not all(
                not torch.is_tensor(source.get(field))
                or tuple(source[field].shape) == tuple(parameter.shape)
                for field in ("exp_avg", "exp_avg_sq")
            ):
                continue
            optimizer.state[parameter] = {
                field: (
                    value.detach().clone().cpu()
                    if field == "step" and torch.is_tensor(value)
                    else value.detach().clone().to(parameter)
                    if torch.is_tensor(value)
                    else value
                )
                for field, value in source.items()
            }

    def _store_moments(self, optimizer: torch.optim.Optimizer) -> None:
        output: dict[str, dict[str, object]] = {}
        state_device = self._state_storage_device()
        for group in optimizer.param_groups:
            name = str(group.get("name", ""))
            if name == "poses" or len(group["params"]) != 1:
                continue
            parameter = group["params"][0]
            state = optimizer.state.get(parameter)
            if not state:
                continue
            output[name] = {
                field: (
                    value.detach().cpu().clone()
                    if field == "step" and torch.is_tensor(value)
                    else value.detach().to(state_device).clone()
                    if torch.is_tensor(value)
                    else value
                )
                for field, value in state.items()
            }
        self.mapper._pfgs360_gaussian_moments = output

    def _clear_opacity_moments(self, rows: torch.Tensor) -> None:
        state = dict(
            getattr(self.mapper, "_pfgs360_gaussian_moments", {}) or {}
        ).get("opacity_logit")
        if not state:
            return
        for field in ("exp_avg", "exp_avg_sq"):
            value = state.get(field)
            if not torch.is_tensor(value):
                continue
            mask = rows.detach().to(device=value.device, dtype=torch.bool)
            if int(value.shape[0]) == int(mask.numel()):
                value[mask] = 0

    def _pose(self, observation) -> torch.Tensor:
        return self.mapper.pose_deltas[int(observation.frame_id)]().to(
            device=self.device, dtype=self.dtype
        )

    def _render(self, observation, *, query_values: torch.Tensor | None = None) -> dict:
        target = observation.image.to(device=self.device, dtype=self.dtype)
        from backend.pano_gs.adapter import PanoRenderCamera

        render_camera = PanoRenderCamera(
            image_height=int(target.shape[-2]),
            image_width=int(target.shape[-1]),
            c2w=self._pose(observation),
        )
        return self.mapper.renderer.render(
            render_camera,
            self.map,
            query_values=query_values,
        )

    @staticmethod
    def _owner_state_equal(lhs: dict[str, object], rhs: dict[str, object]) -> bool:
        if bool(lhs.get("enabled", False)) != bool(rhs.get("enabled", False)):
            return False
        for key in ("reference", "current"):
            left = dict(lhs.get(key, {}) or {})
            right = dict(rhs.get(key, {}) or {})
            if set(left) != set(right):
                return False
            if any(not torch.equal(left[owner], right[owner]) for owner in left):
                return False
        return True

    def _references(self, observations, current) -> list:
        candidates = [
            value
            for value in observations
            if int(value.frame_id) != int(current.frame_id)
        ]
        candidates.sort(
            key=lambda value: (
                abs(int(value.frame_id) - int(current.frame_id)),
                int(value.frame_id),
            )
        )
        return candidates[:2]

    def _dia_valid_mask(
        self,
        observation,
        aligned_depth: torch.Tensor,
        alpha: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the DIA computation domain for the selected compatibility mode."""

        valid = (
            torch.isfinite(aligned_depth)
            & (aligned_depth >= self.cfg.min_depth)
            & (aligned_depth <= self.cfg.max_depth)
        )
        official_sky_only = str(
            self.settings.get("validity_gate", "legacy")
        ).strip().lower() == "pfgs360_official_sky_only"
        sky_mask = observation.sky_mask
        if official_sky_only and sky_mask is None:
            raise RuntimeError(
                "pfgs360_official_sky_only requires a panoramic sky mask"
            )
        if sky_mask is not None:
            valid &= ~sky_mask.to(device=self.device, dtype=torch.bool)
        if official_sky_only:
            return valid
        if observation.depth_confidence is not None:
            valid &= observation.depth_confidence.to(self.device) >= float(
                self.settings.get("min_depth_confidence", 0.05)
            )
        if torch.is_tensor(alpha):
            valid &= torch.isfinite(alpha) & (
                alpha >= float(self.settings.get("alpha_threshold", 0.05))
            )
        return valid

    def _bootstrap(self, observations, owner_window_id: int) -> dict[str, int]:
        if self.map.anchor_count() > 0 or not observations:
            return {"raw": 0, "unique": 0, "occupied": 0, "inserted": 0}
        if self.uses_refined_anchor_growth:
            if self.refined_anchor_update is None:
                raise RuntimeError(
                    "Refined-anchor PFGS360 bootstrap requires an anchor update handler"
                )
            output = self.refined_anchor_update(
                event="bootstrap",
                owner_window_id=int(owner_window_id),
                observations=tuple(observations),
                new_frame_ids=tuple(int(value.frame_id) for value in observations),
                mono_inlier_masks={},
                optimized_poses={
                    int(value.frame_id): self._pose(value).detach()
                    for value in observations
                },
                existing_anchor_visibility={},
            )
            self.mapper._pfgs360_gaussian_moments = {}
            return {
                "raw": int(output.get("candidate", 0)),
                "unique": int(output.get("selected", output.get("candidate", 0))),
                "occupied": 0,
                "inserted": int(output.get("inserted", 0)),
            }
        first = observations[0]
        if first.target_depth is None:
            raise RuntimeError("PFGS360 bootstrap requires refined panoramic ray depth")
        depth = first.target_depth.to(device=self.device, dtype=self.dtype)
        mask = torch.isfinite(depth) & (depth >= self.cfg.min_depth) & (
            depth <= self.cfg.max_depth
        )
        if first.sky_mask is not None:
            mask &= ~first.sky_mask.to(device=self.device, dtype=torch.bool)
        if first.depth_confidence is not None:
            mask &= first.depth_confidence.to(self.device) >= float(
                self.settings.get("min_depth_confidence", 0.05)
            )
        xyz, rgb = backproject_panorama_depth(
            depth,
            first.image.to(self.device),
            self._pose(first),
            mask,
        )
        output = self.map.append_pfgs360_points(
            xyz,
            rgb,
            owner_window_id=int(owner_window_id),
            frame_id=int(first.frame_id),
            voxel_size=float(self.settings.get("voxel_size", 0.01)),
            initial_opacity=float(self.settings.get("initial_opacity", 0.01)),
            min_raw_points=int(self.settings.get("min_raw_growth_points", 10)),
            min_unique_voxels=int(self.settings.get("min_unique_growth_voxels", 100)),
        )
        self._remap_moments()
        return output

    def _sampling_schedule(self, observations, steps: int, seed: int) -> list:
        if not observations:
            return []
        midpoint = max(1, len(observations) // 2)
        history = observations[:midpoint]
        recent = observations[midpoint:] or observations[-1:]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        schedule = []
        probability = float(self.settings.get("latter_half_sample_probability", 0.7))
        for _ in range(int(steps)):
            pool = recent if float(torch.rand((), generator=generator)) < probability else history
            index = int(torch.randint(len(pool), (), generator=generator))
            schedule.append(pool[index])
        return schedule

    def _render_consistency_masks(self, observations) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            depths = {
                int(obs.frame_id): self._render(obs)["depth"].detach()
                for obs in observations
            }
        output: dict[int, torch.Tensor] = {}
        for observation in observations:
            references = self._references(observations, observation)
            depth = depths.get(int(observation.frame_id))
            if not torch.is_tensor(depth) or len(references) < 2:
                shape = observation.image.shape[-2:]
                output[int(observation.frame_id)] = torch.ones(
                    1, *shape, device=self.device, dtype=torch.bool
                )
                continue
            reference_values = [
                (
                    depths[int(ref.frame_id)],
                    self._pose(ref),
                )
                for ref in references
            ]
            output[int(observation.frame_id)] = multi_reference_consistency(
                depth,
                self._pose(observation),
                reference_values,
                config=self.cfg,
            )
        return output

    def _camera_stage(self, observations, steps: int, seed: int) -> dict[str, float]:
        if not observations or int(steps) <= 0:
            return {"camera_steps": 0.0, "camera_loss": 0.0}
        fixed_frame = min(int(value.frame_id) for value in observations)
        pose_params = [
            self.mapper.pose_deltas[int(obs.frame_id)].delta
            for obs in observations
            if int(obs.frame_id) != fixed_frame
        ]
        if not pose_params:
            return {"camera_steps": 0.0, "camera_loss": 0.0}
        optimizer = torch.optim.Adam(
            pose_params,
            lr=float(self.settings.get("pose_lr", 1.0e-3)),
            eps=float(self.settings.get("adam_eps", 1.0e-15)),
            weight_decay=0.0,
        )
        consistency = self._render_consistency_masks(observations)
        schedule = self._sampling_schedule(observations, steps, seed)
        last_loss = 0.0
        gaussian_parameters = self.map.gaussian_parameters() + self.map.skybox_parameters()
        requires_grad = [parameter.requires_grad for parameter in gaussian_parameters]
        try:
            for parameter in gaussian_parameters:
                parameter.requires_grad_(False)
            for observation in schedule:
                optimizer.zero_grad(set_to_none=True)
                package = self._render(observation)
                target = observation.image.to(device=self.device, dtype=self.dtype)
                mask = consistency[int(observation.frame_id)]
                if observation.sky_mask is not None:
                    mask &= ~observation.sky_mask.to(device=self.device, dtype=torch.bool)
                loss, _ = pfgs360_photometric_loss(package["render"], target, mask=mask)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite PFGS360 CAMERA loss")
                loss.backward()
                self._clip_pose_gradients(
                    pose_params,
                    float(self.settings.get("pose_grad_clip_value", 1.0e-2)),
                )
                if not all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in pose_params
                ):
                    raise FloatingPointError("Non-finite PFGS360 CAMERA pose gradient")
                optimizer.step()
                last_loss = float(loss.detach().cpu())
        finally:
            for parameter, enabled in zip(gaussian_parameters, requires_grad):
                parameter.requires_grad_(enabled)
        return {
            "camera_steps": float(len(schedule)),
            "camera_loss": last_loss,
            "camera_trainable_poses": float(len(pose_params)),
        }

    def _dia(self, observations, new_frame_ids, owner_window_id: int) -> dict[str, float]:
        if not observations:
            return {"dia_render_views": 0.0}
        with torch.no_grad():
            packages = {}
            for observation in observations:
                rendered = self._render(observation)
                packages[int(observation.frame_id)] = {
                    "depth": rendered["depth"].detach(),
                    "alpha": (
                        None
                        if not torch.is_tensor(rendered.get("alpha"))
                        else rendered["alpha"].detach()
                    ),
                }
        render_consistency: dict[int, torch.Tensor] = {}
        aligned_depths: dict[int, torch.Tensor] = {}
        alignment_scales: list[float] = []
        for observation in observations:
            frame_id = int(observation.frame_id)
            references = self._references(observations, observation)
            rendered = packages[frame_id]["depth"]
            if len(references) >= 2:
                render_consistency[frame_id] = multi_reference_consistency(
                    rendered,
                    self._pose(observation),
                    [
                        (packages[int(ref.frame_id)]["depth"], self._pose(ref))
                        for ref in references
                    ],
                    config=self.cfg,
                )
            else:
                render_consistency[frame_id] = torch.ones_like(rendered, dtype=torch.bool)
            if observation.target_depth is None:
                aligned_depths[frame_id] = rendered.detach()
                alignment_scales.append(1.0)
            else:
                aligned, scale, _, _ = affine_align_depth(
                    observation.target_depth.to(self.device),
                    rendered,
                    render_consistency[frame_id],
                    min_depth=self.cfg.min_depth,
                    max_depth=self.cfg.max_depth,
                )
                aligned_depths[frame_id] = aligned
                alignment_scales.append(scale)

        inconsistent_hits = torch.zeros(self.map.anchor_count(), device=self.device, dtype=torch.int32)
        inlier_hits = torch.zeros_like(inconsistent_hits)
        mono_inlier_masks: dict[int, torch.Tensor] = {}
        growth_xyz: list[torch.Tensor] = []
        growth_rgb: list[torch.Tensor] = []
        mono_inlier_pixels = 0
        query_views = 0
        new_set = {int(value) for value in new_frame_ids}
        for observation in observations:
            frame_id = int(observation.frame_id)
            references = self._references(observations, observation)
            if len(references) < 2:
                continue
            aligned = aligned_depths[frame_id]
            mono_consistency = multi_reference_consistency(
                aligned,
                self._pose(observation),
                [(aligned_depths[int(ref.frame_id)], self._pose(ref)) for ref in references],
                config=self.cfg,
            )
            render_inconsistent = ~render_consistency[frame_id]
            alpha = packages[frame_id].get("alpha")
            valid = self._dia_valid_mask(observation, aligned, alpha)
            render_inconsistent &= valid
            patch_better = panoramic_patch_better_mask(
                observation.image.to(self.device),
                packages[frame_id]["depth"],
                aligned,
                self._pose(observation),
                [
                    (ref.image.to(self.device), self._pose(ref))
                    for ref in references
                ],
                radius=self.cfg.gncc_radius,
            )
            mono_inlier = render_inconsistent & mono_consistency & patch_better & valid
            mono_inlier_pixels += int(mono_inlier.sum().item())
            query = torch.stack(
                [render_inconsistent[0].float(), mono_inlier[0].float()], dim=-1
            )
            with torch.no_grad():
                package = self._render(observation, query_values=query)
            answers = package.get("query_answers")
            accum = package.get("accum_visible")
            if not (torch.is_tensor(answers) and torch.is_tensor(accum)):
                raise RuntimeError("PFGS360 DIA requires query_answers and accum_visible")
            if tuple(answers.shape) != (self.map.anchor_count(), 2):
                raise RuntimeError("PFGS360 DIA query answer shape mismatch")
            responsibility = answers / accum.view(-1, 1).clamp_min(1.0e-8)
            finite = torch.isfinite(responsibility).all(dim=-1) & (accum > 0.0)
            threshold = float(self.settings.get("query_responsibility_threshold", 0.8))
            inconsistent_hits += (finite & (responsibility[:, 0] >= threshold)).int()
            inlier_hits += (finite & (responsibility[:, 1] >= threshold)).int()
            query_views += 1
            if frame_id in new_set:
                mono_inlier_masks[frame_id] = mono_inlier.detach()
            if (
                not self.uses_refined_anchor_growth
                and frame_id in new_set
                and bool(mono_inlier.any())
            ):
                xyz, rgb = backproject_panorama_depth(
                    aligned,
                    observation.image.to(self.device),
                    self._pose(observation),
                    mono_inlier,
                )
                growth_xyz.append(xyz)
                growth_rgb.append(rgb)
            del package, answers, accum, responsibility

        cull = inlier_hits > 0
        reset = (inconsistent_hits > 0) & ~cull
        reset_count = int(reset.sum().item())
        cull_count = int(cull.sum().item())
        reset_applied = 0
        if reset_count >= int(self.settings.get("min_reset_gaussians", 100)):
            limit = math.log(0.01 / 0.99)
            with torch.no_grad():
                self.map.opacity_logit[reset] = torch.minimum(
                    self.map.opacity_logit[reset],
                    self.map.opacity_logit.new_tensor(limit),
                )
            reset_applied = reset_count
            self._clear_opacity_moments(reset)
        deleted = 0
        if cull_count >= int(self.settings.get("min_delete_gaussians", 100)):
            deleted = self.map.prune_anchors(cull)
            self._remap_moments()
        growth = {"raw": 0, "unique": 0, "occupied": 0, "inserted": 0}
        refined_stats: dict[str, Any] = {}
        if self.uses_refined_anchor_growth:
            if self.refined_anchor_update is None:
                raise RuntimeError(
                    "Refined-anchor PFGS360 growth requires an anchor update handler"
                )
            visibility: dict[int, torch.Tensor] = {}
            optimized_poses: dict[int, torch.Tensor] = {}
            with torch.no_grad():
                for observation in observations:
                    frame_id = int(observation.frame_id)
                    if frame_id not in new_set:
                        continue
                    package = self._render(observation)
                    accum = package.get("accum_visible")
                    radii = package.get("radii")
                    if torch.is_tensor(accum) and int(accum.numel()) == self.map.anchor_count():
                        visible = torch.isfinite(accum) & (accum > 0.0)
                    elif torch.is_tensor(radii) and int(radii.numel()) == self.map.anchor_count():
                        visible = torch.isfinite(radii) & (radii.reshape(-1) > 0.0)
                    else:
                        raise RuntimeError(
                            "Refined-anchor Hash requires per-Gaussian visibility"
                        )
                    visibility[frame_id] = visible.detach()
                    optimized_poses[frame_id] = self._pose(observation).detach()
            refined_stats = self.refined_anchor_update(
                event="growth",
                owner_window_id=int(owner_window_id),
                observations=tuple(observations),
                new_frame_ids=tuple(sorted(new_set)),
                mono_inlier_masks=mono_inlier_masks,
                optimized_poses=optimized_poses,
                existing_anchor_visibility=visibility,
            )
            self.mapper._pfgs360_gaussian_moments = {}
            growth = {
                "raw": int(refined_stats.get("candidate", 0)),
                "unique": int(refined_stats.get("selected", 0)),
                "occupied": 0,
                "inserted": int(refined_stats.get("inserted", 0)),
            }
        elif growth_xyz:
            growth = self.map.append_pfgs360_points(
                torch.cat(growth_xyz, dim=0),
                torch.cat(growth_rgb, dim=0),
                owner_window_id=int(owner_window_id),
                frame_id=max(new_set) if new_set else int(observations[-1].frame_id),
                voxel_size=float(self.settings.get("voxel_size", 0.01)),
                initial_opacity=float(self.settings.get("initial_opacity", 0.01)),
                min_raw_points=int(self.settings.get("min_raw_growth_points", 10)),
                min_unique_voxels=int(self.settings.get("min_unique_growth_voxels", 100)),
            )
            self._remap_moments()
        metrics = {
            "dia_render_views": float(len(observations)),
            "dia_query_views": float(query_views),
            "dia_mono_inlier_pixels": float(mono_inlier_pixels),
            "dia_reset_candidates": float(reset_count),
            "dia_reset_applied": float(reset_applied),
            "dia_delete_candidates": float(cull_count),
            "dia_deleted": float(deleted),
            "dia_growth_raw": float(growth["raw"]),
            "dia_growth_unique": float(growth["unique"]),
            "dia_growth_occupied": float(growth["occupied"]),
            "dia_growth_inserted": float(growth["inserted"]),
            "dia_alignment_scale_mean": float(sum(alignment_scales) / max(1, len(alignment_scales))),
        }
        metrics.update(
            {
                f"dia_anchor_{key}": float(value)
                for key, value in refined_stats.items()
                if isinstance(value, (int, float))
            }
        )
        return metrics

    def _joint_stage(self, observations, steps: int, seed: int) -> dict[str, float]:
        fixed_frame = min(int(value.frame_id) for value in observations)
        pose_params = [
            self.mapper.pose_deltas[int(obs.frame_id)].delta
            for obs in observations
            if int(obs.frame_id) != fixed_frame
        ]
        parameter_groups = []
        for name in self.map._gaussian_parameter_names():
            parameter = getattr(self.map, name)
            parameter_groups.append(
                {
                    "params": [parameter],
                    "lr": float(self.settings.get(f"{name}_lr", self.PARAMETER_LRS[name])),
                    "name": name,
                }
            )
        if pose_params:
            parameter_groups.append(
                {
                    "params": pose_params,
                    "lr": float(self.settings.get("joint_pose_lr", 1.0e-3)),
                    "name": "poses",
                }
            )
        optimizer = torch.optim.Adam(
            parameter_groups,
            eps=float(self.settings.get("adam_eps", 1.0e-15)),
            weight_decay=0.0,
        )
        self._load_moments(optimizer)
        schedule = self._sampling_schedule(observations, steps, seed)
        topology_refine_enabled = bool(
            self.settings.get("topology_refine_enabled", True)
        )
        grad_sum = (
            torch.zeros(self.map.anchor_count(), device=self.device)
            if topology_refine_enabled
            else None
        )
        grad_count = None if grad_sum is None else torch.zeros_like(grad_sum)
        max_radii = None if grad_sum is None else torch.zeros_like(grad_sum)
        last_loss = 0.0
        sky_parameters = self.map.skybox_parameters()
        sky_requires_grad = [parameter.requires_grad for parameter in sky_parameters]
        try:
            for parameter in sky_parameters:
                parameter.requires_grad_(False)
            for step_index, observation in enumerate(schedule):
                optimizer.zero_grad(set_to_none=True)
                package = self._render(observation)
                target = observation.image.to(device=self.device, dtype=self.dtype)
                mask = None
                if observation.sky_mask is not None:
                    mask = ~observation.sky_mask.to(device=self.device, dtype=torch.bool)
                loss, _ = pfgs360_photometric_loss(package["render"], target, mask=mask)
                if (step_index + 1) % int(self.settings.get("phys_ratio_every", 10)) == 0:
                    scale = self.map.get_scaling
                    ratio = scale.amax(dim=-1) / scale.amin(dim=-1).clamp_min(1.0e-8)
                    excessive = ratio - float(
                        self.settings.get("scale_ratio_threshold", 10.0)
                    )
                    positive = excessive[excessive > 0.0]
                    phys = (
                        positive.mean()
                        if int(positive.numel()) > 0
                        else excessive.new_zeros(())
                    )
                    loss = loss + float(self.settings.get("phys_ratio_weight", 0.1)) * phys
                distortion = package.get("render_distort")
                if torch.is_tensor(distortion):
                    balance = pfgs360_spherical_weight(
                        int(target.shape[-2]),
                        int(target.shape[-1]),
                        device=target.device,
                        dtype=target.dtype,
                    )
                    depth = package["depth"].clamp_min(1.0e-3)
                    loss = loss + float(
                        self.settings.get("distortion_weight", 0.01)
                    ) * (balance * distortion / depth).mean().nan_to_num()
                loss = loss + float(self.settings.get("opacity_regularizer_weight", 0.01)) * self.map.get_opacity.mean()
                loss = loss + float(self.settings.get("scale_regularizer_weight", 0.01)) * self.map.get_scaling.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite PFGS360 JOINT loss")
                loss.backward()
                if pose_params:
                    self._clip_pose_gradients(
                        pose_params,
                        float(self.settings.get("pose_grad_clip_value", 1.0e-2)),
                    )
                all_parameters = [parameter for group in parameter_groups for parameter in group["params"]]
                if not all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in all_parameters
                ):
                    raise FloatingPointError("Non-finite PFGS360 JOINT gradient")
                viewspace = package.get("viewspace_points")
                radii = package.get("radii")
                if topology_refine_enabled and torch.is_tensor(viewspace):
                    gradient = getattr(viewspace, "absgrad", None)
                    if gradient is None and (
                        bool(viewspace.is_leaf) or bool(viewspace.retains_grad)
                    ):
                        gradient = viewspace.grad
                    if torch.is_tensor(gradient) and int(gradient.shape[0]) == self.map.anchor_count():
                        magnitude = torch.linalg.norm(gradient.detach(), dim=-1)
                        visible = magnitude > 0.0
                        assert grad_sum is not None and grad_count is not None
                        grad_sum[visible] += magnitude[visible]
                        grad_count[visible] += 1.0
                if (
                    topology_refine_enabled
                    and torch.is_tensor(radii)
                    and int(radii.numel()) == self.map.anchor_count()
                ):
                    assert max_radii is not None
                    max_radii = torch.maximum(max_radii, radii.detach().reshape(-1).to(max_radii))
                optimizer.step()
                if not all(bool(torch.isfinite(value).all()) for value in self.map.gaussian_parameters()):
                    raise FloatingPointError("Non-finite PFGS360 Gaussian parameter")
                last_loss = float(loss.detach().cpu())
        finally:
            for parameter, enabled in zip(sky_parameters, sky_requires_grad):
                parameter.requires_grad_(enabled)
        self.mapper._pfgs360_joint_steps = int(
            getattr(self.mapper, "_pfgs360_joint_steps", 0)
        ) + len(schedule)
        refine_every = (
            int(self.settings.get("refine_every_joint_steps", 100))
            if topology_refine_enabled
            else 0
        )
        refine = {"split": 0, "duplicate": 0, "culled": 0, "after": self.map.anchor_count()}
        self._store_moments(optimizer)
        if refine_every > 0 and self.mapper._pfgs360_joint_steps % refine_every == 0:
            assert grad_sum is not None and grad_count is not None and max_radii is not None
            mean_grad = grad_sum / grad_count.clamp_min(1.0)
            refine = self.map.pfgs360_refine_topology(
                mean_grad,
                max_radii,
                grad_threshold=float(self.settings.get("absgrad_threshold", 8.0e-5)),
                split_scale_threshold=float(self.settings.get("split_scale_threshold", 0.01)),
                split_samples=int(self.settings.get("split_samples", 2)),
                cull_opacity=float(self.settings.get("cull_opacity", 0.005)),
                ood_distance=float(self.settings.get("ood_distance", 1.0e5)),
            )
            self._remap_moments()
        metrics = {
            "joint_steps": float(len(schedule)),
            "joint_loss": last_loss,
            "joint_trainable_poses": float(len(pose_params)),
        }
        if topology_refine_enabled:
            metrics.update(
                {
                    "refine_split": float(refine.get("split", 0)),
                    "refine_split_children": float(refine.get("split_children", 0)),
                    "refine_duplicate": float(refine.get("duplicate", 0)),
                    "refine_culled": float(refine.get("culled", 0)),
                }
            )
        return metrics

    def run(
        self,
        *,
        frame_ids: list[int] | tuple[int, ...],
        new_frame_ids: list[int] | tuple[int, ...],
        owner_window_id: int,
        camera_steps: int = 50,
        joint_steps: int = 50,
        seed: int = 123,
    ) -> dict[str, float]:
        snapshot_started = time.perf_counter()
        state = self._snapshot()
        snapshot_seconds = float(time.perf_counter() - snapshot_started)
        owner_before = self.map.lazy_owner_transform_state()
        try:
            observations = self._observations(frame_ids)
            if not observations:
                raise RuntimeError("PFGS360 full backend has no registered observations")
            metrics: dict[str, float] = {
                "strategy_pfgs360_full_50_50": 1.0,
                "visited_frames": float(len(observations)),
                "state_storage_on_map_device": float(
                    self._state_storage_device().type == self.device.type
                    and self._state_storage_device() == self.device
                ),
                "snapshot_seconds": snapshot_seconds,
            }
            bootstrap = self._bootstrap(observations, int(owner_window_id))
            metrics.update({f"bootstrap_{key}": float(value) for key, value in bootstrap.items()})
            refined_bootstrap = self.uses_refined_anchor_growth and int(
                bootstrap.get("inserted", 0)
            ) > 0
            if self.uses_refined_anchor_growth and self.map.anchor_count() == 0:
                raise RuntimeError("Refined-anchor bootstrap produced an empty map")
            started = time.perf_counter()
            metrics.update(self._camera_stage(observations, int(camera_steps), int(seed)))
            metrics["camera_seconds"] = float(time.perf_counter() - started)
            started = time.perf_counter()
            if refined_bootstrap:
                metrics.update(
                    {
                        "dia_render_views": 0.0,
                        "dia_query_views": 0.0,
                        "dia_first_window_passthrough": 1.0,
                    }
                )
            else:
                metrics.update(
                    self._dia(
                        observations,
                        tuple(int(value) for value in new_frame_ids),
                        int(owner_window_id),
                    )
                )
            metrics["dia_seconds"] = float(time.perf_counter() - started)
            started = time.perf_counter()
            metrics.update(
                self._joint_stage(
                    observations,
                    int(joint_steps),
                    int(seed) + 1,
                )
            )
            metrics["joint_seconds"] = float(time.perf_counter() - started)
            backend_cfg = dict(
                self.map.config.get("SphericalSelfiGlobalBackend", {}) or {}
            )
            voxel_cfg = dict(backend_cfg.get("voxel_fusion", {}) or {})
            maximum = max(0, int(voxel_cfg.get("max_total_gaussians", 0)))
            capacity_removed = 0
            if maximum > 0 and self.map.anchor_count() > maximum:
                quality = self.map._anchor_quality.to(self.device)
                score = quality + self.map.get_opacity.detach().view(-1)
                retained = torch.topk(score, k=maximum, largest=True).indices
                keep = torch.zeros(self.map.anchor_count(), device=self.device, dtype=torch.bool)
                keep[retained] = True
                capacity_removed = self.map.prune_anchors(~keep)
                self._remap_moments()
            if not self._owner_state_equal(
                self.map.lazy_owner_transform_state(), owner_before
            ):
                raise RuntimeError("PFGS360 optimization modified an owner Sim3 transform")
            if not all(bool(torch.isfinite(value).all()) for value in self.map.gaussian_parameters()):
                raise FloatingPointError("PFGS360 transaction ended with non-finite map state")
            metrics["anchors_after"] = float(self.map.anchor_count())
            metrics["capacity_removed"] = float(capacity_removed)
            metrics["window_rollback"] = 0.0
            return metrics
        except Exception:
            self._restore(state)
            raise
