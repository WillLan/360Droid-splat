"""Feed-forward anchor/scaffold Gaussian prediction heads for PanoVGGT features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import SH_C0
from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing

from .grid_utils import feature_uv_to_image_uv, make_feature_grid


def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(x / (1.0 - x))


def _normalize_quaternion(raw: torch.Tensor) -> torch.Tensor:
    quat = raw.clone()
    if quat.numel() == 0:
        return quat
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    return torch.where(norm > 1.0e-6, quat / norm.clamp_min(1.0e-6), identity)


@dataclass
class ExplicitGaussianSet:
    """Renderer-compatible explicit Gaussians materialized from anchor predictions."""

    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    config: dict[str, Any] | None = None
    active_sh_degree: int = 0
    max_sh_degree: int = 0

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    @property
    def get_features(self) -> torch.Tensor:
        return self.features

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        return ((self.features - 0.5) / SH_C0).unsqueeze(1)


@dataclass
class AnchorGaussianPrediction:
    """Batched anchor Gaussian state predicted from PanoVGGT priors."""

    base_depth: torch.Tensor
    source_bearing: torch.Tensor
    source_rot: torch.Tensor
    source_trans: torch.Tensor
    anchor_rgb: torch.Tensor
    anchor_valid: torch.Tensor
    anchor_feature: torch.Tensor
    log_depth_delta: torch.Tensor
    local_offsets: torch.Tensor
    log_scales: torch.Tensor
    quat_raw: torch.Tensor
    opacity_logit: torch.Tensor
    color_logit: torch.Tensor
    source_image_uv: torch.Tensor
    source_frame_index: torch.Tensor
    image_hw: tuple[int, int]
    feature_hw: tuple[int, int]
    min_scale: float = 1.0e-4
    max_scale: float = 0.25
    depth_delta_limit: float = 0.35

    @property
    def batch_size(self) -> int:
        return int(self.anchor_rgb.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(self.anchor_rgb.shape[1])

    @property
    def k_offsets(self) -> int:
        return int(self.local_offsets.shape[2])

    def current_anchor_xyz(self) -> torch.Tensor:
        depth_delta = self.log_depth_delta.clamp(-float(self.depth_delta_limit), float(self.depth_delta_limit))
        depth = self.base_depth.clamp_min(1.0e-6) * torch.exp(depth_delta)
        cam = self.source_bearing * depth
        return torch.einsum("bmij,bmj->bmi", self.source_rot, cam) + self.source_trans

    def materialize(self, batch_index: int, *, config: dict[str, Any] | None = None) -> ExplicitGaussianSet:
        idx = int(batch_index)
        anchor_xyz = self.current_anchor_xyz()[idx]
        valid = self.anchor_valid[idx].bool()
        xyz = anchor_xyz[:, None, :] + self.local_offsets[idx]
        scales = self.log_scales[idx].exp().clamp(float(self.min_scale), float(self.max_scale))
        rotation = _normalize_quaternion(self.quat_raw[idx])
        opacity = torch.sigmoid(self.opacity_logit[idx]).clamp(0.0, 1.0)
        color = torch.sigmoid(self.color_logit[idx]).clamp(0.0, 1.0)
        valid_g = valid[:, None].expand(-1, self.k_offsets).reshape(-1)
        if bool(valid_g.any()):
            xyz = xyz.reshape(-1, 3)[valid_g]
            scales = scales.reshape(-1, 3)[valid_g]
            rotation = rotation.reshape(-1, 4)[valid_g]
            opacity = opacity.reshape(-1, 1)[valid_g]
            color = color.reshape(-1, 3)[valid_g]
        else:
            device = anchor_xyz.device
            dtype = anchor_xyz.dtype
            xyz = torch.zeros(0, 3, device=device, dtype=dtype)
            scales = torch.zeros(0, 3, device=device, dtype=dtype)
            rotation = torch.zeros(0, 4, device=device, dtype=dtype)
            opacity = torch.zeros(0, 1, device=device, dtype=dtype)
            color = torch.zeros(0, 3, device=device, dtype=dtype)
        return ExplicitGaussianSet(
            xyz=torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0),
            scaling=torch.nan_to_num(scales, nan=float(self.min_scale), posinf=float(self.max_scale), neginf=float(self.min_scale)),
            rotation=torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0),
            opacity=torch.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0),
            features=torch.nan_to_num(color, nan=0.5, posinf=1.0, neginf=0.0),
            config=config,
        )

    def detached(self) -> "AnchorGaussianPrediction":
        return replace(
            self,
            base_depth=self.base_depth.detach(),
            source_bearing=self.source_bearing.detach(),
            source_rot=self.source_rot.detach(),
            source_trans=self.source_trans.detach(),
            anchor_rgb=self.anchor_rgb.detach(),
            anchor_valid=self.anchor_valid.detach(),
            anchor_feature=self.anchor_feature.detach(),
            log_depth_delta=self.log_depth_delta.detach(),
            local_offsets=self.local_offsets.detach(),
            log_scales=self.log_scales.detach(),
            quat_raw=self.quat_raw.detach(),
            opacity_logit=self.opacity_logit.detach(),
            color_logit=self.color_logit.detach(),
            source_image_uv=self.source_image_uv.detach(),
            source_frame_index=self.source_frame_index.detach(),
        )


class _ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, int(out_channels))
        while groups > 1 and int(out_channels) % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1),
            nn.GroupNorm(groups, int(out_channels)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PanoVGGTAnchorGaussianHead(nn.Module):
    """Predict compact anchor/scaffold Gaussian parameters from PanoVGGT features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        anchor_feat_dim: int = 64,
        k_offsets: int = 4,
        num_conv_blocks: int = 2,
        anchor_stride: int = 1,
        max_anchors: int = 4096,
        min_scale: float = 0.002,
        max_scale: float = 0.12,
        init_scale: float = 0.02,
        depth_delta_limit: float = 0.35,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive.")
        if int(k_offsets) <= 0:
            raise ValueError("k_offsets must be positive.")
        if int(num_conv_blocks) <= 0:
            raise ValueError("num_conv_blocks must be positive.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.anchor_feat_dim = int(anchor_feat_dim)
        self.k_offsets = int(k_offsets)
        self.anchor_stride = max(1, int(anchor_stride))
        self.max_anchors = int(max_anchors)
        self.min_scale = max(float(min_scale), 1.0e-6)
        self.max_scale = max(float(max_scale), self.min_scale)
        self.init_scale = min(max(float(init_scale), self.min_scale), self.max_scale)
        self.depth_delta_limit = float(depth_delta_limit)

        in_dim = self.feature_dim + 3 + 3 + 1 + 4
        blocks: list[nn.Module] = []
        cur = in_dim
        for _ in range(int(num_conv_blocks)):
            blocks.append(_ConvNormGELU(cur, self.hidden_dim))
            cur = self.hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.anchor_feat_proj = nn.Conv2d(self.hidden_dim, self.anchor_feat_dim, kernel_size=1)
        self.depth_delta_proj = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.offset_proj = nn.Conv2d(self.hidden_dim, self.k_offsets * 3, kernel_size=1)
        self.scale_proj = nn.Conv2d(self.hidden_dim, self.k_offsets * 3, kernel_size=1)
        self.rotation_proj = nn.Conv2d(self.hidden_dim, self.k_offsets * 4, kernel_size=1)
        self.opacity_proj = nn.Conv2d(self.hidden_dim, self.k_offsets, kernel_size=1)
        self.color_delta_proj = nn.Conv2d(self.hidden_dim, self.k_offsets * 3, kernel_size=1)
        self._init_predictions()

    def _init_predictions(self) -> None:
        with torch.no_grad():
            self.depth_delta_proj.weight.zero_()
            self.depth_delta_proj.bias.zero_()
            self.offset_proj.weight.zero_()
            self.offset_proj.bias.zero_()
            self.scale_proj.weight.zero_()
            self.scale_proj.bias.fill_(torch.log(torch.tensor(self.init_scale)).item())
            self.rotation_proj.weight.zero_()
            self.rotation_proj.bias.zero_()
            for idx in range(self.k_offsets):
                self.rotation_proj.bias[idx * 4] = 1.0
            self.opacity_proj.bias.fill_(0.0)
            self.color_delta_proj.weight.zero_()
            self.color_delta_proj.bias.zero_()

    def head_config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "anchor_feat_dim": self.anchor_feat_dim,
            "k_offsets": self.k_offsets,
            "num_conv_blocks": len(self.trunk),
            "anchor_stride": self.anchor_stride,
            "max_anchors": self.max_anchors,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "init_scale": self.init_scale,
            "depth_delta_limit": self.depth_delta_limit,
        }

    def forward(
        self,
        features: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        poses_c2w: torch.Tensor,
        *,
        world_points: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> AnchorGaussianPrediction:
        """Predict anchor Gaussian state for source views.

        Shapes:
        - ``features``: ``B x S x C x Hf x Wf``
        - ``images``: ``B x S x 3 x H x W``
        - ``depth``: ``B x S x 1 x H x W``
        - ``poses_c2w``: ``B x S x 4 x 4``
        """

        if features.ndim != 5:
            raise ValueError(f"features must have shape BxSxCxHfxWf, got {tuple(features.shape)}")
        if images.ndim != 5 or int(images.shape[2]) != 3:
            raise ValueError(f"images must have shape BxSx3xHxW, got {tuple(images.shape)}")
        if depth.ndim != 5 or int(depth.shape[2]) != 1:
            raise ValueError(f"depth must have shape BxSx1xHxW, got {tuple(depth.shape)}")
        if poses_c2w.ndim != 4 or tuple(poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"poses_c2w must have shape BxSx4x4, got {tuple(poses_c2w.shape)}")
        b, s, c, hf, wf = [int(v) for v in features.shape]
        if c != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {c}.")
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        feature_hw = (hf, wf)
        device = features.device
        dtype = features.dtype

        rgb_low = F.interpolate(images.reshape(b * s, 3, *image_hw).to(dtype), size=feature_hw, mode="bilinear", align_corners=False)
        depth_low = F.interpolate(depth.reshape(b * s, 1, *image_hw).to(dtype), size=feature_hw, mode="bilinear", align_corners=False).clamp_min(1.0e-6)
        uv_feature = make_feature_grid(feature_hw, device=device, dtype=dtype)
        uv_image = feature_uv_to_image_uv(uv_feature, feature_hw, image_hw)
        bearing = erp_pixel_to_bearing(uv_image, *image_hw).to(device=device, dtype=dtype)
        bearing_map = bearing.permute(2, 0, 1).view(1, 3, hf, wf).expand(b * s, -1, -1, -1)
        anchor_depth_low = depth_low
        anchor_bearing_map = bearing_map
        world_valid_low = torch.ones_like(depth_low, dtype=torch.bool)
        if world_points is not None:
            if world_points.ndim != 5 or int(world_points.shape[-1]) != 3:
                raise ValueError(f"world_points must have shape BxSxHxWx3, got {tuple(world_points.shape)}")
            if int(world_points.shape[0]) != b or int(world_points.shape[1]) != s:
                raise ValueError("world_points batch/frame dimensions must match features.")
            wp = world_points.to(device=device, dtype=dtype).permute(0, 1, 4, 2, 3).reshape(b * s, 3, int(world_points.shape[2]), int(world_points.shape[3]))
            wp_low = F.interpolate(wp, size=feature_hw, mode="bilinear", align_corners=False).view(b, s, 3, hf, wf)
            rot = poses_c2w[:, :, :3, :3].to(device=device, dtype=dtype)
            trans = poses_c2w[:, :, :3, 3].to(device=device, dtype=dtype)
            cam = torch.einsum("bsij,bsihw->bsjhw", rot, wp_low - trans.view(b, s, 3, 1, 1))
            depth_from_world = torch.linalg.norm(cam, dim=2, keepdim=True).clamp_min(1.0e-6)
            bearing_from_world = cam / depth_from_world
            finite_world = torch.isfinite(wp_low).all(dim=2, keepdim=True) & torch.isfinite(depth_from_world)
            anchor_depth_low = torch.where(finite_world.reshape(b * s, 1, hf, wf), depth_from_world.reshape(b * s, 1, hf, wf), depth_low)
            anchor_bearing_map = torch.where(
                finite_world.reshape(b * s, 1, hf, wf),
                bearing_from_world.reshape(b * s, 3, hf, wf),
                bearing_map,
            )
            world_valid_low = finite_world.reshape(b * s, 1, hf, wf)
        yy = torch.linspace(-1.0, 1.0, steps=hf, device=device, dtype=dtype).view(1, 1, hf, 1).expand(b * s, 1, hf, wf)
        xx = torch.linspace(-1.0, 1.0, steps=wf, device=device, dtype=dtype).view(1, 1, 1, wf).expand(b * s, 1, hf, wf)
        uv_embed = torch.cat(
            [
                torch.sin(torch.pi * xx),
                torch.cos(torch.pi * xx),
                torch.sin(0.5 * torch.pi * yy),
                torch.cos(0.5 * torch.pi * yy),
            ],
            dim=1,
        )

        x = torch.cat(
            [
                features.reshape(b * s, c, hf, wf),
                rgb_low,
                anchor_bearing_map,
                anchor_depth_low.log(),
                uv_embed,
            ],
            dim=1,
        )
        hidden = self.trunk(x)
        anchor_feat = self.anchor_feat_proj(hidden)
        depth_delta = self.depth_delta_proj(hidden).tanh() * float(self.depth_delta_limit)
        offsets = self.offset_proj(hidden).view(b * s, self.k_offsets, 3, hf, wf)
        scales = self.scale_proj(hidden).view(b * s, self.k_offsets, 3, hf, wf)
        rotations = self.rotation_proj(hidden).view(b * s, self.k_offsets, 4, hf, wf)
        opacity = self.opacity_proj(hidden).view(b * s, self.k_offsets, 1, hf, wf)
        color_delta = self.color_delta_proj(hidden).view(b * s, self.k_offsets, 3, hf, wf)

        select = self._anchor_selection_mask(feature_hw, device=device)
        if valid_mask is not None:
            mask_low = F.interpolate(
                valid_mask.reshape(b * s, 1, *valid_mask.shape[-2:]).float().to(device=device),
                size=feature_hw,
                mode="nearest",
            ) > 0.5
            select = select.view(1, 1, hf, wf) & mask_low
        else:
            select = select.view(1, 1, hf, wf).expand(b * s, -1, -1, -1)
        finite_depth = torch.isfinite(anchor_depth_low) & (anchor_depth_low > 1.0e-6) & world_valid_low
        select = (select & finite_depth).view(b, s, hf, wf)
        flat_select = select.reshape(b, s * hf * wf)
        indices = self._selected_indices(flat_select)

        flat_depth = anchor_depth_low.view(b, s, 1, hf, wf).permute(0, 1, 3, 4, 2).reshape(b, s * hf * wf, 1)
        flat_bearing = anchor_bearing_map.view(b, s, 3, hf, wf).permute(0, 1, 3, 4, 2).reshape(b, s * hf * wf, 3)
        flat_rgb = rgb_low.view(b, s, 3, hf, wf).permute(0, 1, 3, 4, 2).reshape(b, s * hf * wf, 3)
        flat_anchor_feat = anchor_feat.view(b, s, self.anchor_feat_dim, hf, wf).permute(0, 1, 3, 4, 2).reshape(b, s * hf * wf, self.anchor_feat_dim)
        flat_delta = depth_delta.view(b, s, 1, hf, wf).permute(0, 1, 3, 4, 2).reshape(b, s * hf * wf, 1)
        flat_offsets = offsets.view(b, s, self.k_offsets, 3, hf, wf).permute(0, 1, 4, 5, 2, 3).reshape(b, s * hf * wf, self.k_offsets, 3)
        flat_scales = scales.view(b, s, self.k_offsets, 3, hf, wf).permute(0, 1, 4, 5, 2, 3).reshape(b, s * hf * wf, self.k_offsets, 3)
        flat_rot = rotations.view(b, s, self.k_offsets, 4, hf, wf).permute(0, 1, 4, 5, 2, 3).reshape(b, s * hf * wf, self.k_offsets, 4)
        flat_opacity = opacity.view(b, s, self.k_offsets, 1, hf, wf).permute(0, 1, 4, 5, 2, 3).reshape(b, s * hf * wf, self.k_offsets, 1)
        flat_color_delta = color_delta.view(b, s, self.k_offsets, 3, hf, wf).permute(0, 1, 4, 5, 2, 3).reshape(b, s * hf * wf, self.k_offsets, 3)
        flat_valid = flat_select

        src_frame = torch.arange(s, device=device, dtype=torch.long).view(1, s, 1, 1).expand(b, -1, hf, wf).reshape(b, s * hf * wf)
        frame_uv = uv_image.view(1, 1, hf, wf, 2).expand(b, s, -1, -1, -1).reshape(b, s * hf * wf, 2)

        gather_1 = indices.unsqueeze(-1)
        gather_3 = indices.unsqueeze(-1).expand(-1, -1, 3)
        gather_feat = indices.unsqueeze(-1).expand(-1, -1, self.anchor_feat_dim)
        gather_k3 = indices.view(b, -1, 1, 1).expand(-1, -1, self.k_offsets, 3)
        gather_k4 = indices.view(b, -1, 1, 1).expand(-1, -1, self.k_offsets, 4)
        gather_k1 = indices.view(b, -1, 1, 1).expand(-1, -1, self.k_offsets, 1)

        source_index = torch.gather(src_frame, 1, indices)
        rot = poses_c2w[:, :, :3, :3].to(device=device, dtype=dtype)
        trans = poses_c2w[:, :, :3, 3].to(device=device, dtype=dtype)
        gather_rot = source_index.view(b, -1, 1, 1).expand(-1, -1, 3, 3)
        gather_trans = source_index.unsqueeze(-1).expand(-1, -1, 3)
        anchor_rgb = torch.gather(flat_rgb, 1, gather_3).clamp(0.0, 1.0)
        color_logit = _inv_sigmoid(anchor_rgb).view(b, -1, 1, 3) + torch.gather(flat_color_delta, 1, gather_k3)

        return AnchorGaussianPrediction(
            base_depth=torch.gather(flat_depth, 1, gather_1),
            source_bearing=torch.gather(flat_bearing, 1, gather_3),
            source_rot=torch.gather(rot, 1, gather_rot),
            source_trans=torch.gather(trans, 1, gather_trans),
            anchor_rgb=anchor_rgb,
            anchor_valid=torch.gather(flat_valid, 1, indices).bool(),
            anchor_feature=torch.gather(flat_anchor_feat, 1, gather_feat),
            log_depth_delta=torch.gather(flat_delta, 1, gather_1),
            local_offsets=torch.gather(flat_offsets, 1, gather_k3) * self.init_scale,
            log_scales=torch.gather(flat_scales, 1, gather_k3),
            quat_raw=torch.gather(flat_rot, 1, gather_k4),
            opacity_logit=torch.gather(flat_opacity, 1, gather_k1),
            color_logit=color_logit,
            source_image_uv=torch.gather(frame_uv, 1, indices.unsqueeze(-1).expand(-1, -1, 2)),
            source_frame_index=source_index,
            image_hw=image_hw,
            feature_hw=feature_hw,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            depth_delta_limit=self.depth_delta_limit,
        )

    def _anchor_selection_mask(self, feature_hw: tuple[int, int], *, device: torch.device) -> torch.Tensor:
        hf, wf = feature_hw
        yy, xx = torch.meshgrid(torch.arange(hf, device=device), torch.arange(wf, device=device), indexing="ij")
        return (yy % self.anchor_stride == 0) & (xx % self.anchor_stride == 0)

    def _selected_indices(self, flat_select: torch.Tensor) -> torch.Tensor:
        b, total = int(flat_select.shape[0]), int(flat_select.shape[1])
        max_anchors = total if self.max_anchors <= 0 else min(total, int(self.max_anchors))
        rows = []
        fallback = torch.linspace(0, total - 1, steps=max_anchors, device=flat_select.device).round().long()
        for batch_idx in range(b):
            idx = torch.nonzero(flat_select[batch_idx], as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = fallback
            if idx.numel() > max_anchors:
                keep = torch.linspace(0, idx.numel() - 1, steps=max_anchors, device=idx.device).round().long()
                idx = idx[keep]
            if idx.numel() < max_anchors:
                pad = idx[-1:].expand(max_anchors - idx.numel()) if idx.numel() else fallback[: max_anchors - idx.numel()]
                idx = torch.cat([idx, pad], dim=0)
            rows.append(idx[:max_anchors])
        return torch.stack(rows, dim=0)


class IterativeGaussianRefiner(nn.Module):
    """Recurrent feed-forward update that consumes render feedback per anchor."""

    def __init__(
        self,
        *,
        anchor_feat_dim: int = 64,
        hidden_dim: int = 128,
        k_offsets: int = 4,
        max_depth_step: float = 0.08,
        max_offset_step: float = 0.01,
        max_scale_step: float = 0.05,
        max_color_step: float = 0.10,
        max_opacity_step: float = 0.25,
    ) -> None:
        super().__init__()
        self.anchor_feat_dim = int(anchor_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_offsets = int(k_offsets)
        self.max_depth_step = float(max_depth_step)
        self.max_offset_step = float(max_offset_step)
        self.max_scale_step = float(max_scale_step)
        self.max_color_step = float(max_color_step)
        self.max_opacity_step = float(max_opacity_step)
        input_dim = self.anchor_feat_dim + 3 + 1 + 1 + 3 + 1 + 1 + 3
        self.input_proj = nn.Linear(input_dim, self.anchor_feat_dim)
        self.gru = nn.GRUCell(self.anchor_feat_dim, self.anchor_feat_dim)
        self.delta = nn.Sequential(
            nn.LayerNorm(self.anchor_feat_dim),
            nn.Linear(self.anchor_feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1 + self.k_offsets * (3 + 3 + 4 + 1 + 3)),
        )
        self._init_delta()

    def _init_delta(self) -> None:
        last = self.delta[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, pred: AnchorGaussianPrediction, feedback: dict[str, torch.Tensor]) -> AnchorGaussianPrediction:
        rgb_error = feedback["rgb_error"].to(pred.anchor_feature)
        depth_error = feedback["depth_error"].to(pred.anchor_feature)
        alpha = feedback["alpha"].to(pred.anchor_feature)
        view_dir = feedback["view_dir"].to(pred.anchor_feature)
        opacity_mean = torch.sigmoid(pred.opacity_logit).mean(dim=2)
        scale_mean = pred.log_scales.exp().mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        color_mean = torch.sigmoid(pred.color_logit).mean(dim=2)
        x = torch.cat(
            [
                pred.anchor_feature,
                rgb_error,
                depth_error,
                alpha,
                view_dir,
                opacity_mean,
                scale_mean,
                color_mean - pred.anchor_rgb,
            ],
            dim=-1,
        )
        b, m, _ = x.shape
        update_input = self.input_proj(x.reshape(b * m, -1))
        hidden = self.gru(update_input, pred.anchor_feature.reshape(b * m, -1)).view(b, m, -1)
        raw = self.delta(hidden).view(b, m, -1)
        cursor = 0
        depth_step = torch.tanh(raw[..., cursor : cursor + 1]) * self.max_depth_step
        cursor += 1
        offset_step = torch.tanh(raw[..., cursor : cursor + self.k_offsets * 3]).view(b, m, self.k_offsets, 3) * self.max_offset_step
        cursor += self.k_offsets * 3
        scale_step = torch.tanh(raw[..., cursor : cursor + self.k_offsets * 3]).view(b, m, self.k_offsets, 3) * self.max_scale_step
        cursor += self.k_offsets * 3
        rot_step = torch.tanh(raw[..., cursor : cursor + self.k_offsets * 4]).view(b, m, self.k_offsets, 4) * 0.05
        cursor += self.k_offsets * 4
        opacity_step = torch.tanh(raw[..., cursor : cursor + self.k_offsets]).view(b, m, self.k_offsets, 1) * self.max_opacity_step
        cursor += self.k_offsets
        color_step = torch.tanh(raw[..., cursor : cursor + self.k_offsets * 3]).view(b, m, self.k_offsets, 3) * self.max_color_step
        return replace(
            pred,
            anchor_feature=hidden,
            log_depth_delta=(pred.log_depth_delta + depth_step).clamp(-pred.depth_delta_limit, pred.depth_delta_limit),
            local_offsets=pred.local_offsets + offset_step,
            log_scales=pred.log_scales + scale_step,
            quat_raw=pred.quat_raw + rot_step,
            opacity_logit=pred.opacity_logit + opacity_step,
            color_logit=pred.color_logit + color_step,
        )


def sample_render_feedback(
    pred: AnchorGaussianPrediction,
    *,
    render_rgb: torch.Tensor,
    render_depth: torch.Tensor | None,
    render_alpha: torch.Tensor | None,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor | None,
    target_pose_c2w: torch.Tensor,
    batch_index: int,
) -> dict[str, torch.Tensor]:
    """Sample target-view render residuals at current anchor projections."""

    idx = int(batch_index)
    anchor_xyz = pred.current_anchor_xyz()[idx]
    device = anchor_xyz.device
    dtype = anchor_xyz.dtype
    target = target_rgb.to(device=device, dtype=dtype)
    rgb_err_map = (target - render_rgb.to(device=device, dtype=dtype)).unsqueeze(0)
    if render_depth is None or target_depth is None:
        depth_err_map = torch.zeros(1, 1, target.shape[-2], target.shape[-1], device=device, dtype=dtype)
    else:
        rd = render_depth.to(device=device, dtype=dtype)
        td = target_depth.to(device=device, dtype=dtype)
        depth_err_map = ((td - rd) / td.abs().clamp_min(1.0)).clamp(-1.0, 1.0).unsqueeze(0)
    if render_alpha is None:
        alpha_map = torch.zeros(1, 1, target.shape[-2], target.shape[-1], device=device, dtype=dtype)
    else:
        alpha_map = render_alpha.to(device=device, dtype=dtype).unsqueeze(0)

    c2w = target_pose_c2w.to(device=device, dtype=dtype)
    w2c = torch.linalg.inv(c2w)
    ones = torch.ones(anchor_xyz.shape[0], 1, device=device, dtype=dtype)
    cam = (w2c @ torch.cat([anchor_xyz, ones], dim=-1).T).T[:, :3]
    bearing = F.normalize(cam, dim=-1, eps=1.0e-6)
    height, width = int(target.shape[-2]), int(target.shape[-1])
    uv = bearing_to_erp_pixel(bearing, height, width)
    uv_x = torch.remainder(uv[:, 0], float(width)).clamp(0.0, max(float(width - 1), 1.0))
    uv_y = uv[:, 1].clamp(0.0, max(float(height - 1), 1.0))
    norm_x = 2.0 * uv_x / max(width - 1, 1) - 1.0
    norm_y = 2.0 * uv_y / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)

    rgb_error = F.grid_sample(rgb_err_map, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
    depth_error = F.grid_sample(depth_err_map, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
    alpha = F.grid_sample(alpha_map, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
    center = c2w[:3, 3]
    view_dir = F.normalize(anchor_xyz - center.view(1, 3), dim=-1, eps=1.0e-6)
    valid = pred.anchor_valid[idx].to(device=device).view(-1, 1).to(dtype=dtype)
    return {
        "rgb_error": (rgb_error * valid).unsqueeze(0),
        "depth_error": (depth_error * valid).unsqueeze(0),
        "alpha": (alpha * valid).unsqueeze(0),
        "view_dir": (view_dir * valid).unsqueeze(0),
    }


def merge_feedback(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample feedback dictionaries into a batched feedback dict."""

    if not parts:
        raise ValueError("feedback parts cannot be empty.")
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}
