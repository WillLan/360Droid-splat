"""Compact feed-forward Gaussian initialization for Pano-ReSplat."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import SH_C0
from frontend.pano_droid.spherical_camera import tangent_basis

from .grid_utils import feature_uv_to_image_uv, make_feature_grid
from .pano_resplat_geometry import erp_uv_to_bearing, world_to_camera
from .resplat_types import PanoGaussianState


@dataclass(frozen=True)
class PanoCompactGaussianInitializerConfig:
    type: str | None = None
    position_mode: str = "compact"
    latent_downsample: int = 1
    gaussians_per_cell: int = 2
    state_dim: int = 64
    sh_degree: int = 0
    max_gaussians: int = 0
    min_scale: float = 0.002
    max_scale: float = 0.12
    init_scale: float = 0.02
    use_world_points_as_base: bool = False
    use_local_offsets: bool = True
    max_offset_abs: float = 0.05
    max_offset_depth_ratio: float = 0.02


def _as_config(
    config: PanoCompactGaussianInitializerConfig | dict | None,
    **overrides,
) -> PanoCompactGaussianInitializerConfig:
    if config is None:
        base = {}
    elif isinstance(config, PanoCompactGaussianInitializerConfig):
        base = config.__dict__
    elif isinstance(config, dict):
        base = dict(config)
    else:
        raise TypeError(f"Unsupported initializer config type: {type(config)!r}")
    base.update({key: value for key, value in overrides.items() if value is not None})
    return PanoCompactGaussianInitializerConfig(**base)


class _ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int | None, out_channels: int) -> None:
        super().__init__()
        conv: nn.Module
        if in_channels is None:
            conv = nn.LazyConv2d(int(out_channels), kernel_size=3, padding=1)
        else:
            conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1)
        groups = min(8, int(out_channels))
        while groups > 1 and int(out_channels) % groups != 0:
            groups -= 1
        self.block = nn.Sequential(conv, nn.GroupNorm(groups, int(out_channels)), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PanoCompactGaussianInitializer(nn.Module):
    """Predict a compact ``PanoGaussianState`` from context views.

    The initializer uses PanoVGGT context features and geometry priors only.  It
    does not consume target images.  When ``use_world_points_as_base`` is true,
    pass ``world_points`` to ``forward`` as ``B x V x H x W x 3``.
    """

    def __init__(
        self,
        config: PanoCompactGaussianInitializerConfig | dict | None = None,
        *,
        position_mode: str | None = None,
        latent_downsample: int | None = None,
        gaussians_per_cell: int | None = None,
        state_dim: int | None = None,
        sh_degree: int | None = None,
        max_gaussians: int | None = None,
        min_scale: float | None = None,
        max_scale: float | None = None,
        init_scale: float | None = None,
        use_world_points_as_base: bool | None = None,
        use_local_offsets: bool | None = None,
        max_offset_abs: float | None = None,
        max_offset_depth_ratio: float | None = None,
    ) -> None:
        super().__init__()
        raise RuntimeError(
            "PanoCompactGaussianInitializer has been retired. "
            "Use PanoVGGTPointDecoderGaussianInitializer from "
            "frontend.pano_vggt.pano_resplat_point_decoder_init."
        )
        cfg = _as_config(
            config,
            position_mode=position_mode,
            latent_downsample=latent_downsample,
            gaussians_per_cell=gaussians_per_cell,
            state_dim=state_dim,
            sh_degree=sh_degree,
            max_gaussians=max_gaussians,
            min_scale=min_scale,
            max_scale=max_scale,
            init_scale=init_scale,
            use_world_points_as_base=use_world_points_as_base,
            use_local_offsets=use_local_offsets,
            max_offset_abs=max_offset_abs,
            max_offset_depth_ratio=max_offset_depth_ratio,
        )
        position_mode_value = str(cfg.position_mode).lower()
        type_value = str(cfg.type).lower() if cfg.type is not None else ""
        if type_value in {"panovggt_aligned", "pano_vggt_aligned"}:
            position_mode_value = "panovggt_aligned"
        if position_mode_value not in {"compact", "dense_world_points", "panovggt_aligned"}:
            raise ValueError(f"Unsupported Initializer.position_mode={cfg.position_mode!r}.")
        if int(cfg.sh_degree) < 0:
            raise ValueError("sh_degree must be non-negative.")
        if int(cfg.gaussians_per_cell) <= 0:
            raise ValueError("gaussians_per_cell must be positive.")
        if int(cfg.state_dim) <= 0:
            raise ValueError("state_dim must be positive.")
        if position_mode_value == "dense_world_points":
            if int(cfg.gaussians_per_cell) != 1:
                raise ValueError("dense_world_points mode requires gaussians_per_cell=1.")
            if bool(cfg.use_local_offsets):
                raise NotImplementedError("dense_world_points mode currently requires use_local_offsets=false.")
        if position_mode_value == "panovggt_aligned" and int(cfg.gaussians_per_cell) != 1:
            raise ValueError("panovggt_aligned mode currently requires gaussians_per_cell=1.")
        self.config = cfg
        self.position_mode = position_mode_value
        self.latent_downsample = max(1, int(cfg.latent_downsample))
        self.gaussians_per_cell = int(cfg.gaussians_per_cell)
        self.state_dim = int(cfg.state_dim)
        self.sh_degree = int(cfg.sh_degree)
        self.sh_dim = (self.sh_degree + 1) ** 2
        self.max_gaussians = int(cfg.max_gaussians)
        self.min_scale = max(float(cfg.min_scale), 1.0e-8)
        self.max_scale = max(float(cfg.max_scale), self.min_scale)
        self.init_scale = min(max(float(cfg.init_scale), self.min_scale), self.max_scale)
        self.use_world_points_as_base = bool(cfg.use_world_points_as_base)
        self.use_local_offsets = bool(cfg.use_local_offsets)
        self.max_offset_abs = max(0.0, float(cfg.max_offset_abs))
        self.max_offset_depth_ratio = max(0.0, float(cfg.max_offset_depth_ratio))

        self.trunk = nn.Sequential(
            _ConvNormGELU(None, self.state_dim),
            _ConvNormGELU(self.state_dim, self.state_dim),
        )
        self.dense_feature_proj = nn.Sequential(
            nn.LazyConv2d(self.state_dim, kernel_size=1),
            nn.GELU(),
        )
        k = self.gaussians_per_cell
        self.latent_proj = nn.Conv2d(self.state_dim, self.state_dim, kernel_size=1)
        self.offset_proj = nn.Conv2d(self.state_dim, k * 3, kernel_size=1)
        self.scale_proj = nn.Conv2d(self.state_dim, k * 3, kernel_size=1)
        self.rotation_proj = nn.Conv2d(self.state_dim, k * 4, kernel_size=1)
        self.opacity_proj = nn.Conv2d(self.state_dim, k, kernel_size=1)
        self.sh0_residual_proj = nn.Conv2d(self.state_dim, k * 3 * self.sh_dim, kernel_size=1)
        self.confidence_proj = nn.Conv2d(self.state_dim, k, kernel_size=1)
        self._init_predictions()

    def _init_predictions(self) -> None:
        with torch.no_grad():
            self.offset_proj.weight.zero_()
            self.offset_proj.bias.zero_()
            self.scale_proj.weight.zero_()
            self.scale_proj.bias.fill_(math.log(float(self.init_scale)))
            self.rotation_proj.weight.zero_()
            self.rotation_proj.bias.zero_()
            for idx in range(self.gaussians_per_cell):
                self.rotation_proj.bias[idx * 4] = 1.0
            self.opacity_proj.weight.zero_()
            self.opacity_proj.bias.zero_()
            self.sh0_residual_proj.weight.zero_()
            self.sh0_residual_proj.bias.zero_()
            self.confidence_proj.weight.zero_()
            self.confidence_proj.bias.zero_()

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        world_points: torch.Tensor | None = None,
    ) -> PanoGaussianState:
        """Predict compact Gaussians from context views.

        Required shapes:
        - images: B x V x 3 x H x W
        - features: B x V x C x Hf x Wf
        - depths: B x V x 1 x H x W
        - poses_c2w: B x V x 4 x 4
        """

        self._validate_inputs(images, features, depths, poses_c2w, valid_mask, world_points)
        if self.position_mode == "dense_world_points":
            return self._forward_dense_world_points(images, features, depths, poses_c2w, valid_mask, world_points=world_points)
        if self.position_mode == "panovggt_aligned":
            return self._forward_panovggt_aligned(images, features, depths, poses_c2w, valid_mask, world_points=world_points)

        input_dtype = features.dtype
        device = features.device
        param_dtype = next(self.parameters()).dtype
        b, v, c, hf, wf = [int(x) for x in features.shape]
        h, w = int(images.shape[-2]), int(images.shape[-1])
        latent_hw = self._latent_hw(hf, wf)
        lh, lw = latent_hw
        flat_count = b * v

        feat = F.interpolate(
            torch.nan_to_num(features.to(dtype=param_dtype), nan=0.0, posinf=0.0, neginf=0.0).reshape(flat_count, c, hf, wf),
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        rgb = F.interpolate(
            torch.nan_to_num(images.to(device=device, dtype=param_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).reshape(flat_count, 3, h, w),
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        depth_raw = depths.to(device=device, dtype=param_dtype)
        depth_valid = torch.isfinite(depth_raw) & (depth_raw > 1.0e-6)
        valid = depth_valid if valid_mask is None else depth_valid & self._normalize_valid_mask(valid_mask, depths.shape).to(device=device)
        depth_safe = torch.where(depth_valid, depth_raw, torch.ones_like(depth_raw))
        depth_low = F.interpolate(depth_safe.reshape(flat_count, 1, h, w), size=latent_hw, mode="bilinear", align_corners=False).clamp_min(1.0e-6)
        valid_low = F.interpolate(valid.reshape(flat_count, 1, h, w).float(), size=latent_hw, mode="nearest") > 0.5

        uv_image = feature_uv_to_image_uv(
            make_feature_grid(latent_hw, device=device, dtype=param_dtype),
            latent_hw,
            (h, w),
        )
        bearing = erp_uv_to_bearing(uv_image, (h, w)).to(device=device, dtype=param_dtype)
        bearing_map = bearing.permute(2, 0, 1).view(1, 3, lh, lw).expand(flat_count, -1, -1, -1)
        uv_norm = torch.stack(
            [
                2.0 * uv_image[..., 0] / float(max(w - 1, 1)) - 1.0,
                2.0 * uv_image[..., 1] / float(max(h - 1, 1)) - 1.0,
            ],
            dim=-1,
        ).permute(2, 0, 1).view(1, 2, lh, lw).expand(flat_count, -1, -1, -1)

        x = torch.cat([feat, rgb, depth_low.log(), bearing_map, uv_norm], dim=1)
        hidden = self.trunk(x)
        latent_cell = self.latent_proj(hidden)
        k = self.gaussians_per_cell
        offsets = self.offset_proj(hidden).view(b, v, k, 3, lh, lw).permute(0, 1, 4, 5, 2, 3)
        log_scales = self.scale_proj(hidden).view(b, v, k, 3, lh, lw).permute(0, 1, 4, 5, 2, 3)
        rotations = self.rotation_proj(hidden).view(b, v, k, 4, lh, lw).permute(0, 1, 4, 5, 2, 3)
        opacity = self.opacity_proj(hidden).view(b, v, k, 1, lh, lw).permute(0, 1, 4, 5, 2, 3)
        sh_raw = self._reshape_sh(self.sh0_residual_proj(hidden), b, v, k, lh, lw)
        confidence_delta = self.confidence_proj(hidden).view(b, v, k, 1, lh, lw).permute(0, 1, 4, 5, 2, 3)
        latent = latent_cell.view(b, v, self.state_dim, lh, lw).permute(0, 1, 3, 4, 2)
        latent = latent.unsqueeze(-2).expand(-1, -1, -1, -1, k, -1)

        base_cam, base_bearing, cell_valid = self._base_camera_points(
            depths=depth_low.view(b, v, 1, lh, lw),
            poses_c2w=poses_c2w.to(device=device, dtype=param_dtype),
            bearing=bearing,
            world_points=world_points,
            latent_hw=latent_hw,
            image_hw=(h, w),
            valid_low=valid_low.view(b, v, 1, lh, lw),
        )
        basis = tangent_basis(base_bearing)
        tangent_u = basis[..., 0]
        tangent_v = basis[..., 1]
        offset_step = torch.tanh(offsets) * float(self.max_scale) if self.use_local_offsets else torch.zeros_like(offsets)
        mean_cam = (
            base_cam.unsqueeze(-2)
            + offset_step[..., 0:1] * base_bearing.unsqueeze(-2)
            + offset_step[..., 1:2] * tangent_u.unsqueeze(-2)
            + offset_step[..., 2:3] * tangent_v.unsqueeze(-2)
        )
        rot = poses_c2w.to(device=device, dtype=param_dtype)[..., :3, :3]
        trans = poses_c2w.to(device=device, dtype=param_dtype)[..., :3, 3]
        means_world = torch.einsum("bvij,bvxykj->bvxyki", rot, mean_cam) + trans.view(b, v, 1, 1, 1, 3)

        rgb_cell = rgb.view(b, v, 3, lh, lw).permute(0, 1, 3, 4, 2).unsqueeze(-2).expand(-1, -1, -1, -1, k, -1)
        sh_coeffs = self._build_sh_coeffs(rgb_cell, sh_raw)
        log_scales = log_scales.clamp(math.log(float(self.min_scale)), math.log(float(self.max_scale)))
        valid_cells = cell_valid.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, k)
        finite = torch.isfinite(means_world).all(dim=-1) & torch.isfinite(log_scales).all(dim=-1) & torch.isfinite(sh_coeffs).all(dim=(-1, -2))
        valid_flat = (valid_cells & finite).reshape(b, -1)
        confidence = (valid_cells.unsqueeze(-1).to(dtype=param_dtype) * torch.sigmoid(opacity + confidence_delta)).reshape(b, -1, 1)

        state = PanoGaussianState(
            means=torch.nan_to_num(means_world.reshape(b, -1, 3).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(log_scales.reshape(b, -1, 3).to(dtype=input_dtype), nan=math.log(float(self.init_scale)), posinf=math.log(float(self.max_scale)), neginf=math.log(float(self.min_scale))),
            rotations_unnorm=torch.nan_to_num(rotations.reshape(b, -1, 4).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(opacity.reshape(b, -1, 1).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(sh_coeffs.reshape(b, -1, 3, self.sh_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(latent.reshape(b, -1, self.state_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=self._source_view_ids(b, v, lh, lw, k, device=device),
            source_uv=self._source_uv(b, v, lh, lw, k, uv_image.to(dtype=input_dtype)),
            valid_mask=valid_flat,
            confidence=torch.nan_to_num(confidence.to(dtype=input_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
        )
        return self._maybe_crop(state)

    def _forward_dense_world_points(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        world_points: torch.Tensor | None,
    ) -> PanoGaussianState:
        if world_points is None:
            raise ValueError("dense_world_points mode requires world_points with shape BxVxHxWx3.")
        input_dtype = features.dtype
        device = features.device
        param_dtype = next(self.parameters()).dtype
        b, v, c, hf, wf = [int(x) for x in features.shape]
        h, w = int(images.shape[-2]), int(images.shape[-1])
        flat_count = b * v
        k = 1

        feat = self.dense_feature_proj(
            torch.nan_to_num(features.to(dtype=param_dtype), nan=0.0, posinf=0.0, neginf=0.0).reshape(flat_count, c, hf, wf)
        )
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        rgb = torch.nan_to_num(
            images.to(device=device, dtype=param_dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0).reshape(flat_count, 3, h, w)
        depth_raw = depths.to(device=device, dtype=param_dtype)
        depth_valid = torch.isfinite(depth_raw) & (depth_raw > 1.0e-6)
        valid = depth_valid if valid_mask is None else depth_valid & self._normalize_valid_mask(valid_mask, depths.shape).to(device=device)
        depth_safe = torch.where(depth_valid, depth_raw, torch.ones_like(depth_raw)).reshape(flat_count, 1, h, w).clamp_min(1.0e-6)

        uv_image = make_feature_grid((h, w), device=device, dtype=param_dtype)
        bearing = erp_uv_to_bearing(uv_image, (h, w)).to(device=device, dtype=param_dtype)
        bearing_map = bearing.permute(2, 0, 1).view(1, 3, h, w).expand(flat_count, -1, -1, -1)
        uv_norm = torch.stack(
            [
                2.0 * uv_image[..., 0] / float(max(w - 1, 1)) - 1.0,
                2.0 * uv_image[..., 1] / float(max(h - 1, 1)) - 1.0,
            ],
            dim=-1,
        ).permute(2, 0, 1).view(1, 2, h, w).expand(flat_count, -1, -1, -1)

        x = torch.cat([feat, rgb, depth_safe.log(), bearing_map, uv_norm], dim=1)
        hidden = self.trunk(x)
        latent_cell = self.latent_proj(hidden)
        log_scales = self.scale_proj(hidden).view(b, v, k, 3, h, w).permute(0, 1, 4, 5, 2, 3)
        rotations = self.rotation_proj(hidden).view(b, v, k, 4, h, w).permute(0, 1, 4, 5, 2, 3)
        opacity = self.opacity_proj(hidden).view(b, v, k, 1, h, w).permute(0, 1, 4, 5, 2, 3)
        sh_raw = self._reshape_sh(self.sh0_residual_proj(hidden), b, v, k, h, w)
        confidence_delta = self.confidence_proj(hidden).view(b, v, k, 1, h, w).permute(0, 1, 4, 5, 2, 3)
        latent = latent_cell.view(b, v, self.state_dim, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-2)

        means_world = world_points.to(device=device, dtype=param_dtype).unsqueeze(-2)
        finite_world = torch.isfinite(means_world).all(dim=-1)
        rgb_cell = rgb.view(b, v, 3, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-2)
        sh_coeffs = self._build_sh_coeffs(rgb_cell, sh_raw)
        log_scales = log_scales.clamp(math.log(float(self.min_scale)), math.log(float(self.max_scale)))
        valid_cells = valid.reshape(b, v, 1, h, w).permute(0, 1, 3, 4, 2).expand(-1, -1, -1, -1, k)
        finite = finite_world & torch.isfinite(log_scales).all(dim=-1) & torch.isfinite(sh_coeffs).all(dim=(-1, -2))
        valid_flat = (valid_cells & finite).reshape(b, -1)
        confidence = (valid_cells.unsqueeze(-1).to(dtype=param_dtype) * torch.sigmoid(opacity + confidence_delta)).reshape(b, -1, 1)

        state = PanoGaussianState(
            means=torch.nan_to_num(means_world.reshape(b, -1, 3).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(log_scales.reshape(b, -1, 3).to(dtype=input_dtype), nan=math.log(float(self.init_scale)), posinf=math.log(float(self.max_scale)), neginf=math.log(float(self.min_scale))),
            rotations_unnorm=torch.nan_to_num(rotations.reshape(b, -1, 4).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(opacity.reshape(b, -1, 1).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(sh_coeffs.reshape(b, -1, 3, self.sh_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(latent.reshape(b, -1, self.state_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=self._source_view_ids(b, v, h, w, k, device=device),
            source_uv=self._source_uv(b, v, h, w, k, uv_image.to(dtype=input_dtype)),
            valid_mask=valid_flat,
            confidence=torch.nan_to_num(confidence.to(dtype=input_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
        )
        return self._maybe_crop(state)

    def _forward_panovggt_aligned(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        world_points: torch.Tensor | None,
    ) -> PanoGaussianState:
        if world_points is None:
            raise ValueError("panovggt_aligned mode requires world_points with shape BxVxHxWx3.")
        input_dtype = features.dtype
        device = features.device
        param_dtype = next(self.parameters()).dtype
        b, v, c, hf, wf = [int(x) for x in features.shape]
        h, w = int(images.shape[-2]), int(images.shape[-1])
        flat_count = b * v
        k = self.gaussians_per_cell

        feat = self.dense_feature_proj(
            torch.nan_to_num(features.to(dtype=param_dtype), nan=0.0, posinf=0.0, neginf=0.0).reshape(flat_count, c, hf, wf)
        )
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        rgb = torch.nan_to_num(
            images.to(device=device, dtype=param_dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0).reshape(flat_count, 3, h, w)
        depth_raw = depths.to(device=device, dtype=param_dtype)
        depth_valid = torch.isfinite(depth_raw) & (depth_raw > 1.0e-6)
        valid = depth_valid if valid_mask is None else depth_valid & self._normalize_valid_mask(valid_mask, depths.shape).to(device=device)
        depth_safe = torch.where(depth_valid, depth_raw, torch.ones_like(depth_raw)).reshape(flat_count, 1, h, w).clamp_min(1.0e-6)

        uv_image = make_feature_grid((h, w), device=device, dtype=param_dtype)
        bearing = erp_uv_to_bearing(uv_image, (h, w)).to(device=device, dtype=param_dtype)
        bearing_map = bearing.permute(2, 0, 1).view(1, 3, h, w).expand(flat_count, -1, -1, -1)
        uv_norm = torch.stack(
            [
                2.0 * uv_image[..., 0] / float(max(w - 1, 1)) - 1.0,
                2.0 * uv_image[..., 1] / float(max(h - 1, 1)) - 1.0,
            ],
            dim=-1,
        ).permute(2, 0, 1).view(1, 2, h, w).expand(flat_count, -1, -1, -1)

        base_world = world_points.to(device=device, dtype=param_dtype)
        base_cam = world_to_camera(base_world, poses_c2w.to(device=device, dtype=param_dtype))
        base_depth = torch.linalg.norm(base_cam, dim=-1, keepdim=True).clamp_min(1.0e-6)
        base_bearing = F.normalize(torch.nan_to_num(base_cam, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1.0e-6)

        world_chw = torch.nan_to_num(base_world, nan=0.0, posinf=0.0, neginf=0.0).permute(0, 1, 4, 2, 3).reshape(flat_count, 3, h, w)
        x = torch.cat([feat, rgb, depth_safe.log(), world_chw, bearing_map, uv_norm], dim=1)
        hidden = self.trunk(x)
        latent_cell = self.latent_proj(hidden)
        offsets = self.offset_proj(hidden).view(b, v, k, 3, h, w).permute(0, 1, 4, 5, 2, 3)
        log_scales = self.scale_proj(hidden).view(b, v, k, 3, h, w).permute(0, 1, 4, 5, 2, 3)
        rotations = self.rotation_proj(hidden).view(b, v, k, 4, h, w).permute(0, 1, 4, 5, 2, 3)
        opacity = self.opacity_proj(hidden).view(b, v, k, 1, h, w).permute(0, 1, 4, 5, 2, 3)
        sh_raw = self._reshape_sh(self.sh0_residual_proj(hidden), b, v, k, h, w)
        confidence_delta = self.confidence_proj(hidden).view(b, v, k, 1, h, w).permute(0, 1, 4, 5, 2, 3)
        latent = latent_cell.view(b, v, self.state_dim, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-2)

        if self.use_local_offsets:
            basis = tangent_basis(base_bearing)
            tangent_u = basis[..., 0]
            tangent_v = basis[..., 1]
            depth_bound = base_depth * float(self.max_offset_depth_ratio)
            if self.max_offset_abs > 0.0:
                abs_bound = torch.full_like(depth_bound, float(self.max_offset_abs))
                depth_bound = torch.minimum(depth_bound, abs_bound)
            offset_step = torch.tanh(offsets) * depth_bound.unsqueeze(-2)
        else:
            tangent_u = torch.zeros_like(base_bearing)
            tangent_v = torch.zeros_like(base_bearing)
            offset_step = torch.zeros_like(offsets)

        mean_cam = (
            base_cam.unsqueeze(-2)
            + offset_step[..., 0:1] * base_bearing.unsqueeze(-2)
            + offset_step[..., 1:2] * tangent_u.unsqueeze(-2)
            + offset_step[..., 2:3] * tangent_v.unsqueeze(-2)
        )
        pose = poses_c2w.to(device=device, dtype=param_dtype)
        rot = pose[..., :3, :3]
        trans = pose[..., :3, 3]
        means_world = torch.einsum("bvij,bvxykj->bvxyki", rot, mean_cam) + trans.view(b, v, 1, 1, 1, 3)

        rgb_cell = rgb.view(b, v, 3, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-2).expand(-1, -1, -1, -1, k, -1)
        sh_coeffs = self._build_sh_coeffs(rgb_cell, sh_raw)
        log_scales = log_scales.clamp(math.log(float(self.min_scale)), math.log(float(self.max_scale)))
        valid_cells = valid.reshape(b, v, 1, h, w).permute(0, 1, 3, 4, 2).expand(-1, -1, -1, -1, k)
        finite_world = torch.isfinite(base_world).all(dim=-1).unsqueeze(-1).expand_as(valid_cells)
        finite = (
            finite_world
            & torch.isfinite(means_world).all(dim=-1)
            & torch.isfinite(log_scales).all(dim=-1)
            & torch.isfinite(sh_coeffs).all(dim=(-1, -2))
            & torch.isfinite(base_depth).squeeze(-1).unsqueeze(-1).expand_as(valid_cells)
        )
        valid_flat = (valid_cells & finite).reshape(b, -1)
        confidence = (valid_cells.unsqueeze(-1).to(dtype=param_dtype) * torch.sigmoid(opacity + confidence_delta)).reshape(b, -1, 1)

        state = PanoGaussianState(
            means=torch.nan_to_num(means_world.reshape(b, -1, 3).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(log_scales.reshape(b, -1, 3).to(dtype=input_dtype), nan=math.log(float(self.init_scale)), posinf=math.log(float(self.max_scale)), neginf=math.log(float(self.min_scale))),
            rotations_unnorm=torch.nan_to_num(rotations.reshape(b, -1, 4).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(opacity.reshape(b, -1, 1).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(sh_coeffs.reshape(b, -1, 3, self.sh_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(latent.reshape(b, -1, self.state_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=self._source_view_ids(b, v, h, w, k, device=device),
            source_uv=self._source_uv(b, v, h, w, k, uv_image.to(dtype=input_dtype)),
            valid_mask=valid_flat,
            confidence=torch.nan_to_num(confidence.to(dtype=input_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
        )
        return self._maybe_crop(state)

    def _validate_inputs(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_mask: torch.Tensor | None,
        world_points: torch.Tensor | None,
    ) -> None:
        if images.ndim != 5 or int(images.shape[2]) != 3:
            raise ValueError(f"images must have shape BxVx3xHxW, got {tuple(images.shape)}")
        if features.ndim != 5:
            raise ValueError(f"features must have shape BxVxCxHfxWf, got {tuple(features.shape)}")
        if depths.ndim != 5 or int(depths.shape[2]) != 1:
            raise ValueError(f"depths must have shape BxVx1xHxW, got {tuple(depths.shape)}")
        if poses_c2w.ndim != 4 or tuple(poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"poses_c2w must have shape BxVx4x4, got {tuple(poses_c2w.shape)}")
        if tuple(images.shape[:2]) != tuple(features.shape[:2]) or tuple(images.shape[:2]) != tuple(depths.shape[:2]):
            raise ValueError("images, features, and depths must share B,V dimensions.")
        if tuple(poses_c2w.shape[:2]) != tuple(images.shape[:2]):
            raise ValueError("poses_c2w must share B,V dimensions with images.")
        if tuple(depths.shape[-2:]) != tuple(images.shape[-2:]):
            raise ValueError("depths must match image H,W.")
        if valid_mask is not None:
            self._normalize_valid_mask(valid_mask, depths.shape)
        if self.use_world_points_as_base or self.position_mode in {"dense_world_points", "panovggt_aligned"}:
            if world_points is None:
                raise ValueError("world point based initialization requires world_points with shape BxVxHxWx3.")
            if world_points.ndim != 5 or int(world_points.shape[-1]) != 3:
                raise ValueError(f"world_points must have shape BxVxHxWx3, got {tuple(world_points.shape)}")
            if tuple(world_points.shape[:2]) != tuple(images.shape[:2]) or tuple(world_points.shape[2:4]) != tuple(images.shape[-2:]):
                raise ValueError("world_points must share B,V,H,W with images.")

    @staticmethod
    def _normalize_valid_mask(valid_mask: torch.Tensor, depth_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        if valid_mask.ndim == 4:
            valid = valid_mask.unsqueeze(2)
        elif valid_mask.ndim == 5 and int(valid_mask.shape[2]) == 1:
            valid = valid_mask
        else:
            raise ValueError(f"valid_mask must have shape BxVxHxW or BxVx1xHxW, got {tuple(valid_mask.shape)}")
        if tuple(valid.shape) != tuple(depth_shape):
            raise ValueError(f"valid_mask shape {tuple(valid.shape)} must match depths shape {tuple(depth_shape)}")
        return valid.bool()

    def _reshape_sh(self, raw: torch.Tensor, b: int, v: int, k: int, h: int, w: int) -> torch.Tensor:
        return raw.view(b, v, k, 3, self.sh_dim, h, w).permute(0, 1, 5, 6, 2, 3, 4)

    def _build_sh_coeffs(self, rgb_cell: torch.Tensor, sh_raw: torch.Tensor) -> torch.Tensor:
        """Build SH coefficients from source RGB plus learned residuals.

        ``rgb_cell`` has shape ``B x V x H x W x K x 3`` and ``sh_raw`` has
        shape ``B x V x H x W x K x 3 x SH_DIM``.
        """

        sh_delta = torch.tanh(sh_raw)
        sh_coeffs = torch.zeros_like(sh_delta)
        rgb_init = (rgb_cell + 0.25 * sh_delta[..., 0]).clamp(0.0, 1.0)
        sh_coeffs[..., 0] = (rgb_init - 0.5) / SH_C0
        if self.sh_dim > 1:
            sh_coeffs[..., 1:] = 0.05 * sh_delta[..., 1:]
        return sh_coeffs

    def _latent_hw(self, hf: int, wf: int) -> tuple[int, int]:
        return (
            max(1, int(hf) // self.latent_downsample),
            max(1, int(wf) // self.latent_downsample),
        )

    def _base_camera_points(
        self,
        *,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        bearing: torch.Tensor,
        world_points: torch.Tensor | None,
        latent_hw: tuple[int, int],
        image_hw: tuple[int, int],
        valid_low: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, v, _, lh, lw = [int(x) for x in depths.shape]
        if self.use_world_points_as_base and world_points is not None:
            h, w = int(image_hw[0]), int(image_hw[1])
            wp = world_points.to(device=depths.device, dtype=depths.dtype).permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w)
            wp_low = F.interpolate(wp, size=latent_hw, mode="bilinear", align_corners=False).view(b, v, 3, lh, lw).permute(0, 1, 3, 4, 2)
            finite_world = torch.isfinite(wp_low).all(dim=-1)
            base_cam = world_to_camera(torch.nan_to_num(wp_low, nan=0.0, posinf=0.0, neginf=0.0), poses_c2w)
            base_depth = torch.linalg.norm(base_cam, dim=-1, keepdim=True).clamp_min(1.0e-6)
            base_bearing = F.normalize(base_cam, dim=-1, eps=1.0e-6)
            cell_valid = valid_low & finite_world.unsqueeze(2) & torch.isfinite(base_depth.permute(0, 1, 4, 2, 3)) & (base_depth.permute(0, 1, 4, 2, 3) > 1.0e-6)
            return base_cam, base_bearing, cell_valid

        base_bearing = bearing.view(1, 1, lh, lw, 3).expand(b, v, -1, -1, -1)
        base_cam = depths.permute(0, 1, 3, 4, 2) * base_bearing
        return base_cam, base_bearing, valid_low

    @staticmethod
    def _source_view_ids(b: int, v: int, lh: int, lw: int, k: int, *, device: torch.device) -> torch.Tensor:
        return torch.arange(v, device=device, dtype=torch.long).view(1, v, 1, 1, 1).expand(b, -1, lh, lw, k).reshape(b, -1)

    @staticmethod
    def _source_uv(b: int, v: int, lh: int, lw: int, k: int, uv_image: torch.Tensor) -> torch.Tensor:
        return uv_image.view(1, 1, lh, lw, 1, 2).expand(b, v, -1, -1, k, -1).reshape(b, -1, 2)

    def _maybe_crop(self, state: PanoGaussianState) -> PanoGaussianState:
        limit = int(self.max_gaussians)
        if limit <= 0 or state.num_gaussians <= limit:
            return state
        if state.confidence is None:
            idx = torch.linspace(0, state.num_gaussians - 1, steps=limit, device=state.means.device).round().long()
            indices = idx.view(1, -1).expand(state.batch_size, -1)
        else:
            indices = torch.topk(state.confidence.squeeze(-1), k=limit, dim=1, largest=True, sorted=True).indices
        return PanoGaussianState(
            means=self._gather_rows(state.means, indices),
            log_scales=self._gather_rows(state.log_scales, indices),
            rotations_unnorm=self._gather_rows(state.rotations_unnorm, indices),
            opacity_logits=self._gather_rows(state.opacity_logits, indices),
            sh_coeffs=self._gather_rows(state.sh_coeffs, indices),
            latent_features=self._gather_rows(state.latent_features, indices),
            source_view_ids=self._gather_rows(state.source_view_ids, indices),
            source_uv=self._gather_rows(state.source_uv, indices),
            valid_mask=self._gather_rows(state.valid_mask, indices),
            confidence=None if state.confidence is None else self._gather_rows(state.confidence, indices),
        )

    @staticmethod
    def _gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("values must have at least B,N dimensions.")
        gather_idx = indices
        while gather_idx.ndim < values.ndim:
            gather_idx = gather_idx.unsqueeze(-1)
        expand_shape = list(indices.shape) + list(values.shape[2:])
        return torch.gather(values, dim=1, index=gather_idx.expand(*expand_shape))
