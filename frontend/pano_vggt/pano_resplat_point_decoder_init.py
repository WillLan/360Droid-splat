"""PanoVGGT point-decoder-isomorphic Gaussian initializer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import SH_C0
from frontend.pano_droid.spherical_camera import tangent_basis

from .grid_utils import make_feature_grid
from .pano_resplat_geometry import world_to_camera
from .resplat_types import PanoGaussianState


INITIALIZER_TYPE = "panovggt_point_decoder_gaussian"
LEGACY_INITIALIZER_TYPES = {
    "compact",
    "dense_world_points",
    "panovggt_aligned",
    "pano_vggt_aligned",
}


@dataclass(frozen=True)
class PanoVGGTPointDecoderGaussianInitializerConfig:
    type: str | None = INITIALIZER_TYPE
    state_dim: int = 64
    sh_degree: int = 3
    patch_size: int = 14
    decoder_embed_dim: int = 1024
    decoder_depth: int = 5
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0
    init_scale: float = 0.015
    use_local_offsets: bool = True
    max_offset_abs: float = 0.05
    max_offset_depth_ratio: float = 0.02


def _as_config(config: PanoVGGTPointDecoderGaussianInitializerConfig | dict[str, Any] | None) -> PanoVGGTPointDecoderGaussianInitializerConfig:
    if config is None:
        raw: dict[str, Any] = {}
    elif isinstance(config, PanoVGGTPointDecoderGaussianInitializerConfig):
        raw = dict(config.__dict__)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raise TypeError(f"Unsupported initializer config type: {type(config)!r}")

    init_type = str(raw.get("type", INITIALIZER_TYPE)).lower()
    position_mode = raw.get("position_mode")
    if init_type in LEGACY_INITIALIZER_TYPES or (position_mode is not None and str(position_mode).lower() in LEGACY_INITIALIZER_TYPES):
        raise ValueError(
            "Legacy Pano-ReSplat initializer modes are disabled. "
            f"Use Initializer.type={INITIALIZER_TYPE!r}."
        )
    if init_type not in {"", "none", INITIALIZER_TYPE}:
        raise ValueError(f"Unsupported Initializer.type={raw.get('type')!r}; expected {INITIALIZER_TYPE!r}.")

    allowed = set(PanoVGGTPointDecoderGaussianInitializerConfig.__dataclass_fields__.keys())
    ignored = {
        "position_mode",
        "latent_downsample",
        "gaussians_per_cell",
        "max_gaussians",
        "min_scale",
        "max_scale",
        "use_world_points_as_base",
    }
    unknown = sorted(set(raw) - allowed - ignored)
    if unknown:
        raise ValueError(f"Unsupported Initializer keys for {INITIALIZER_TYPE}: {unknown}")
    filtered = {key: value for key, value in raw.items() if key in allowed}
    filtered["type"] = INITIALIZER_TYPE
    return PanoVGGTPointDecoderGaussianInitializerConfig(**filtered)


class _LinearPatchHead(nn.Module):
    """LinearPts3d-style token-to-pixel projection for arbitrary output dims."""

    def __init__(self, patch_size: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_dim = int(out_dim)
        self.proj = nn.Linear(int(in_dim), self.out_dim * self.patch_size * self.patch_size)

    def forward(self, tokens: torch.Tensor, token_hw: tuple[int, int], image_hw: tuple[int, int]) -> torch.Tensor:
        b, p, _ = [int(x) for x in tokens.shape]
        th, tw = int(token_hw[0]), int(token_hw[1])
        if p != th * tw:
            raise ValueError(f"Token count {p} does not match token_hw={token_hw}.")
        feat = self.proj(tokens).transpose(1, 2).reshape(
            b,
            self.out_dim * self.patch_size * self.patch_size,
            th,
            tw,
        )
        feat = F.pixel_shuffle(feat, self.patch_size)
        if tuple(feat.shape[-2:]) != tuple(image_hw):
            feat = F.interpolate(feat, size=image_hw, mode="bilinear", align_corners=False)
        return feat.permute(0, 2, 3, 1).contiguous()

    def zero_channel(self, channel_start: int, channel_count: int, *, bias: float = 0.0) -> None:
        start = int(channel_start) * self.patch_size * self.patch_size
        end = int(channel_start + channel_count) * self.patch_size * self.patch_size
        with torch.no_grad():
            self.proj.weight[start:end].zero_()
            self.proj.bias[start:end].fill_(float(bias))

    def set_channel_bias(self, channel: int, value: float) -> None:
        start = int(channel) * self.patch_size * self.patch_size
        end = int(channel + 1) * self.patch_size * self.patch_size
        with torch.no_grad():
            self.proj.weight[start:end].zero_()
            self.proj.bias[start:end].fill_(float(value))


class PanoVGGTPointDecoderGaussianInitializer(nn.Module):
    """Predict per-point Gaussian attributes with a PanoVGGT point-head-like decoder."""

    initializer_type = INITIALIZER_TYPE

    def __init__(self, config: PanoVGGTPointDecoderGaussianInitializerConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = _as_config(config)
        if int(cfg.state_dim) <= 0:
            raise ValueError("Initializer.state_dim must be positive.")
        if int(cfg.sh_degree) < 0:
            raise ValueError("Initializer.sh_degree must be non-negative.")
        if int(cfg.patch_size) <= 0:
            raise ValueError("Initializer.patch_size must be positive.")
        if int(cfg.decoder_embed_dim) <= 0:
            raise ValueError("Initializer.decoder_embed_dim must be positive.")
        if int(cfg.decoder_embed_dim) % int(cfg.decoder_num_heads) != 0:
            raise ValueError("Initializer.decoder_embed_dim must be divisible by decoder_num_heads.")

        self.config = cfg
        self.state_dim = int(cfg.state_dim)
        self.sh_degree = int(cfg.sh_degree)
        self.sh_dim = (self.sh_degree + 1) ** 2
        self.patch_size = int(cfg.patch_size)
        self.decoder_embed_dim = int(cfg.decoder_embed_dim)
        self.decoder_depth = int(cfg.decoder_depth)
        self.decoder_num_heads = int(cfg.decoder_num_heads)
        self.decoder_mlp_ratio = float(cfg.decoder_mlp_ratio)
        self.init_scale = float(cfg.init_scale)
        self.use_local_offsets = bool(cfg.use_local_offsets)
        self.max_offset_abs = max(0.0, float(cfg.max_offset_abs))
        self.max_offset_depth_ratio = max(0.0, float(cfg.max_offset_depth_ratio))

        self.token_proj = nn.LazyLinear(self.decoder_embed_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, self.decoder_embed_dim),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.decoder_embed_dim,
                    nhead=self.decoder_num_heads,
                    dim_feedforward=max(1, int(round(self.decoder_embed_dim * self.decoder_mlp_ratio))),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(0, self.decoder_depth))
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)

        self.out_channels = 3 + 3 + 4 + 1 + 3 * self.sh_dim + self.state_dim + 1
        self.pixel_head = _LinearPatchHead(self.patch_size, self.decoder_embed_dim, self.out_channels)
        self._init_prediction_channels()

    def _init_prediction_channels(self) -> None:
        cursor = 0
        self.pixel_head.zero_channel(cursor, 3, bias=0.0)  # offset
        cursor += 3
        self.pixel_head.zero_channel(cursor, 3, bias=math.log(max(self.init_scale, 1.0e-8)))  # log scale
        cursor += 3
        self.pixel_head.zero_channel(cursor, 4, bias=0.0)  # rotation
        self.pixel_head.set_channel_bias(cursor, 1.0)
        cursor += 4
        self.pixel_head.zero_channel(cursor, 1, bias=0.0)  # opacity
        cursor += 1
        self.pixel_head.zero_channel(cursor, 3 * self.sh_dim, bias=0.0)  # SH residual
        cursor += 3 * self.sh_dim
        # Keep latent rows trainable/random; they are consumed by the refiner.
        cursor += self.state_dim
        self.pixel_head.zero_channel(cursor, 1, bias=0.0)  # confidence residual

    @property
    def initializer_config(self) -> dict[str, Any]:
        return dict(self.config.__dict__)

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        poses_c2w: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        world_points: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        token_hw: tuple[int, int] | torch.Tensor | None = None,
    ) -> PanoGaussianState:
        self._validate_inputs(images, features, depths, poses_c2w, valid_mask, world_points)
        if world_points is None:
            raise ValueError("PanoVGGTPointDecoderGaussianInitializer requires world_points BxVxHxWx3.")

        input_dtype = features.dtype
        device = features.device
        param_dtype = next(self.parameters()).dtype
        b, v, _, h, w = [int(x) for x in images.shape]
        token_values, token_hw_tuple = self._tokens_from_inputs(
            features=features,
            tokens=tokens,
            token_hw=token_hw,
            dtype=param_dtype,
        )
        decoded_tokens = self._decode_tokens(token_values, token_hw_tuple)
        decoded = self.pixel_head(decoded_tokens, token_hw_tuple, (h, w)).view(b, v, h, w, self.out_channels)

        cursor = 0
        offsets = decoded[..., cursor : cursor + 3]
        cursor += 3
        log_scales = decoded[..., cursor : cursor + 3]
        cursor += 3
        rotations = decoded[..., cursor : cursor + 4]
        cursor += 4
        opacity = decoded[..., cursor : cursor + 1]
        cursor += 1
        sh_raw = decoded[..., cursor : cursor + 3 * self.sh_dim].view(b, v, h, w, 3, self.sh_dim)
        cursor += 3 * self.sh_dim
        latent = decoded[..., cursor : cursor + self.state_dim]
        cursor += self.state_dim
        confidence_delta = decoded[..., cursor : cursor + 1]

        depth_raw = depths.to(device=device, dtype=param_dtype)
        depth_valid = torch.isfinite(depth_raw) & (depth_raw > 1.0e-6)
        valid = depth_valid if valid_mask is None else depth_valid & self._normalize_valid_mask(valid_mask, depths.shape).to(device=device)

        base_world = world_points.to(device=device, dtype=param_dtype)
        base_cam = world_to_camera(base_world, poses_c2w.to(device=device, dtype=param_dtype))
        base_depth = torch.linalg.norm(base_cam, dim=-1, keepdim=True).clamp_min(1.0e-6)
        base_bearing = F.normalize(torch.nan_to_num(base_cam, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1.0e-6)

        if self.use_local_offsets:
            basis = tangent_basis(base_bearing)
            tangent_u = basis[..., 0]
            tangent_v = basis[..., 1]
            depth_bound = base_depth * self.max_offset_depth_ratio
            if self.max_offset_abs > 0.0:
                depth_bound = torch.minimum(depth_bound, torch.full_like(depth_bound, self.max_offset_abs))
            offset_step = torch.tanh(offsets) * depth_bound
        else:
            tangent_u = torch.zeros_like(base_bearing)
            tangent_v = torch.zeros_like(base_bearing)
            offset_step = torch.zeros_like(offsets)

        mean_cam = (
            base_cam
            + offset_step[..., 0:1] * base_bearing
            + offset_step[..., 1:2] * tangent_u
            + offset_step[..., 2:3] * tangent_v
        )
        pose = poses_c2w.to(device=device, dtype=param_dtype)
        means_world = torch.einsum("bvij,bvhwj->bvhwi", pose[..., :3, :3], mean_cam) + pose[..., :3, 3].view(b, v, 1, 1, 3)

        rgb = torch.nan_to_num(images.to(device=device, dtype=param_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        rgb_cell = rgb.permute(0, 1, 3, 4, 2)
        sh_coeffs = self._build_sh_coeffs(rgb_cell, sh_raw)

        valid_cells = valid.reshape(b, v, 1, h, w).permute(0, 1, 3, 4, 2).squeeze(-1)
        finite = (
            torch.isfinite(base_world).all(dim=-1)
            & torch.isfinite(means_world).all(dim=-1)
            & torch.isfinite(log_scales).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacity).all(dim=-1)
            & torch.isfinite(sh_coeffs).all(dim=(-1, -2))
            & torch.isfinite(latent).all(dim=-1)
            & torch.isfinite(base_depth).squeeze(-1)
        )
        valid_flat = (valid_cells & finite).reshape(b, -1)
        confidence = (valid_cells.unsqueeze(-1).to(dtype=param_dtype) * torch.sigmoid(opacity + confidence_delta)).reshape(b, -1, 1)

        uv_image = make_feature_grid((h, w), device=device, dtype=input_dtype)
        return PanoGaussianState(
            means=torch.nan_to_num(means_world.reshape(b, -1, 3).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            log_scales=torch.nan_to_num(log_scales.reshape(b, -1, 3).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            rotations_unnorm=torch.nan_to_num(rotations.reshape(b, -1, 4).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            opacity_logits=torch.nan_to_num(opacity.reshape(b, -1, 1).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            sh_coeffs=torch.nan_to_num(sh_coeffs.reshape(b, -1, 3, self.sh_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            latent_features=torch.nan_to_num(latent.reshape(b, -1, self.state_dim).to(dtype=input_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            source_view_ids=self._source_view_ids(b, v, h, w, device=device),
            source_uv=self._source_uv(b, v, h, w, uv_image),
            valid_mask=valid_flat,
            confidence=torch.nan_to_num(confidence.to(dtype=input_dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
        )

    def _tokens_from_inputs(
        self,
        *,
        features: torch.Tensor,
        tokens: torch.Tensor | None,
        token_hw: tuple[int, int] | torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        b, v = int(features.shape[0]), int(features.shape[1])
        if tokens is not None:
            if tokens.ndim == 4:
                token_values = tokens.reshape(b * v, int(tokens.shape[2]), int(tokens.shape[3]))
            elif tokens.ndim == 3 and int(tokens.shape[0]) == b * v:
                token_values = tokens
            else:
                raise ValueError(f"tokens must have shape BxVxPxC or (B*V)xPxC, got {tuple(tokens.shape)}")
            if token_hw is None:
                raise ValueError("token_hw is required when tokens are provided.")
            hw = self._normalize_token_hw(token_hw)
            return torch.nan_to_num(token_values.to(device=features.device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0), hw

        if features.ndim != 5:
            raise ValueError(f"features must have shape BxVxCxHfxWf, got {tuple(features.shape)}")
        _, _, c, hf, wf = [int(x) for x in features.shape]
        token_values = (
            torch.nan_to_num(features.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
            .reshape(b * v, c, hf, wf)
            .permute(0, 2, 3, 1)
            .reshape(b * v, hf * wf, c)
        )
        return token_values, (hf, wf)

    @staticmethod
    def _normalize_token_hw(value: tuple[int, int] | torch.Tensor) -> tuple[int, int]:
        if torch.is_tensor(value):
            vals = value.detach().cpu().flatten().tolist()
        else:
            vals = list(value)
        if len(vals) != 2:
            raise ValueError(f"token_hw must have two values, got {value!r}")
        return int(vals[0]), int(vals[1])

    def _decode_tokens(self, tokens: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        memory = self.token_proj(tokens)
        pos = self._positional_encoding(tokens.shape[0], token_hw, device=tokens.device, dtype=memory.dtype)
        memory = memory + self.pos_mlp(pos)
        x = memory
        for block in self.decoder_blocks:
            x = block(tgt=x, memory=memory)
        return self.decoder_norm(x)

    @staticmethod
    def _positional_encoding(batch: int, token_hw: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        th, tw = int(token_hw[0]), int(token_hw[1])
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=th, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, steps=tw, device=device, dtype=dtype),
            indexing="ij",
        )
        pos = torch.stack(
            [
                torch.sin(math.pi * xx),
                torch.cos(math.pi * xx),
                torch.sin(0.5 * math.pi * yy),
                torch.cos(0.5 * math.pi * yy),
            ],
            dim=-1,
        ).reshape(1, th * tw, 4)
        return pos.expand(int(batch), -1, -1)

    def _build_sh_coeffs(self, rgb_cell: torch.Tensor, sh_raw: torch.Tensor) -> torch.Tensor:
        sh_delta = torch.tanh(sh_raw)
        sh_coeffs = torch.zeros_like(sh_delta)
        rgb_init = (rgb_cell.unsqueeze(-1) + 0.25 * sh_delta[..., 0:1]).clamp(0.0, 1.0)
        sh_coeffs[..., 0] = ((rgb_init.squeeze(-1) - 0.5) / SH_C0)
        if self.sh_dim > 1:
            sh_coeffs[..., 1:] = 0.05 * sh_delta[..., 1:]
        return sh_coeffs

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
            raise ValueError("images/features/depths batch and view dimensions must match.")
        if tuple(images.shape[-2:]) != tuple(depths.shape[-2:]):
            raise ValueError("images and depths must have the same spatial size.")
        if world_points is not None and (world_points.ndim != 5 or int(world_points.shape[-1]) != 3):
            raise ValueError(f"world_points must have shape BxVxHxWx3, got {tuple(world_points.shape)}")
        if world_points is not None and tuple(world_points.shape[:2]) != tuple(images.shape[:2]):
            raise ValueError("world_points batch/view dimensions must match images.")
        if world_points is not None and tuple(world_points.shape[2:4]) != tuple(images.shape[-2:]):
            raise ValueError("world_points spatial size must match images.")
        if valid_mask is not None:
            self._normalize_valid_mask(valid_mask, depths.shape)

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

    @staticmethod
    def _source_view_ids(b: int, v: int, h: int, w: int, *, device: torch.device) -> torch.Tensor:
        return torch.arange(v, device=device, dtype=torch.long).view(1, v, 1, 1).expand(b, -1, h, w).reshape(b, -1)

    @staticmethod
    def _source_uv(b: int, v: int, h: int, w: int, uv_image: torch.Tensor) -> torch.Tensor:
        return uv_image.view(1, 1, h, w, 2).expand(b, v, -1, -1, -1).reshape(b, -1, 2)
