"""Context-view render feedback for Pano-ReSplat refinement."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .pano_resplat_geometry import project_world_to_erp_grid
from .resplat_types import PanoGaussianState, PanoRenderOutput


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _grid_from_pixel_uv(uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = int(image_hw[0]), int(image_hw[1])
    x = torch.remainder(uv[..., 0], float(max(w, 1)))
    y = uv[..., 1].clamp(0.0, float(max(h - 1, 0)))
    gx = 2.0 * x / float(max(w - 1, 1)) - 1.0
    gy = 2.0 * y / float(max(h - 1, 1)) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _sample_wrap(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Sample ``values`` with horizontal ERP wrap and vertical border padding."""

    if values.ndim != 4:
        raise ValueError(f"values must have shape BxCxHxW, got {tuple(values.shape)}")
    gx = torch.remainder((grid[..., 0] + 1.0) * 0.5, 1.0) * 2.0 - 1.0
    gy = grid[..., 1].clamp(-1.0, 1.0)
    wrapped = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(values, wrapped, mode="bilinear", padding_mode="border", align_corners=True)[..., 0]


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle vector to a rotation matrix."""

    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=-1,
    ).view(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(k)
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1.0e-8))
    small = theta2 < 1.0e-8
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-8),
    )
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def _bounded_vector(raw: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = raw.norm(dim=-1, keepdim=True)
    scale = torch.where(
        norm < 1.0e-6,
        torch.ones_like(norm),
        torch.tanh(norm) / norm.clamp_min(1.0e-6),
    )
    return raw * scale * float(max_norm)


class _FrozenFeatureExtractor(nn.Module):
    """Frozen image feature extractor used for ReSplat-style render error."""

    def __init__(self, out_dim: int = 64, *, backbone: str = "resnet18") -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.backbone_name = str(backbone)
        self.uses_torchvision = False
        model: nn.Module | None = None
        if self.backbone_name.lower() == "resnet18":
            try:
                from torchvision.models import resnet18  # type: ignore

                resnet = resnet18(weights=None)
                model = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                )
                self.uses_torchvision = True
            except Exception:
                model = None
        if model is None:
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, self.out_dim, 3, padding=1, bias=False),
                nn.GroupNorm(8, self.out_dim),
                nn.GELU(),
            )
        self.model = model
        self.proj = nn.Conv2d(64 if self.uses_torchvision else self.out_dim, self.out_dim, 1)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "_FrozenFeatureExtractor":  # noqa: D401
        super().train(False)
        return self

    def forward(self, image: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        x = image.clamp(0.0, 1.0)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.model(x)
        feat = self.proj(feat)
        if tuple(feat.shape[-2:]) != tuple(output_hw):
            feat = F.interpolate(feat, size=output_hw, mode="bilinear", align_corners=False)
        return feat


class PanoRenderErrorDecoder(nn.Module):
    """ReSplat-style low-resolution multi-view render-error decoder."""

    def __init__(self, channels: int = 64, *, down_factor: int = 4, num_heads: int = 4, num_blocks: int = 1) -> None:
        super().__init__()
        self.channels = int(channels)
        self.down_factor = max(1, int(down_factor))
        self.num_blocks = max(1, int(num_blocks))
        self.pre = nn.Conv2d(self.channels, self.channels, 3, padding=1, padding_mode="circular")
        low_channels = self.channels * self.down_factor * self.down_factor
        self.to_attn = nn.Sequential(nn.Linear(low_channels, self.channels), nn.LayerNorm(self.channels), nn.GELU())
        heads = max(1, int(num_heads))
        while self.channels % heads != 0 and heads > 1:
            heads -= 1
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(self.channels),
                        "attn": nn.MultiheadAttention(self.channels, heads, batch_first=True),
                        "norm2": nn.LayerNorm(self.channels),
                        "mlp": nn.Sequential(
                            nn.Linear(self.channels, self.channels * 4),
                            nn.GELU(),
                            nn.Linear(self.channels * 4, self.channels),
                        ),
                    }
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.from_attn = nn.Sequential(nn.Linear(self.channels, low_channels), nn.LayerNorm(low_channels), nn.GELU())
        self.post = nn.Conv2d(self.channels, self.channels, 3, padding=1, padding_mode="circular")

    def forward(self, error: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if error.ndim != 5:
            raise ValueError(f"error must have shape BxVxCxHxW, got {tuple(error.shape)}")
        b, v, c, h, w = [int(x) for x in error.shape]
        if c != self.channels:
            raise ValueError(f"Expected error channels={self.channels}, got {c}")
        x = _finite(error)
        if valid_mask is not None:
            valid = valid_mask
            if valid.ndim == 4:
                valid = valid.unsqueeze(2)
            if tuple(valid.shape[:2]) != (b, v):
                raise ValueError("valid_mask must share B,V with error")
            if tuple(valid.shape[-2:]) != (h, w):
                valid = F.interpolate(valid.float(), size=(h, w), mode="nearest") > 0.5
            x = x * valid.to(device=x.device, dtype=x.dtype)
        flat = x.reshape(b * v, c, h, w)
        flat = self.pre(flat)
        pad_h = (self.down_factor - h % self.down_factor) % self.down_factor
        pad_w = (self.down_factor - w % self.down_factor) % self.down_factor
        if pad_h or pad_w:
            flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="circular")
        hp, wp = int(flat.shape[-2]), int(flat.shape[-1])
        low = F.pixel_unshuffle(flat, self.down_factor)
        low_h, low_w = int(low.shape[-2]), int(low.shape[-1])
        tokens = low.reshape(b, v, low.shape[1], low_h, low_w).permute(0, 1, 3, 4, 2).reshape(b, v * low_h * low_w, low.shape[1])
        tokens = self.to_attn(tokens)
        for block in self.blocks:
            y = block["norm1"](tokens)
            attn, _ = block["attn"](y, y, y, need_weights=False)
            tokens = tokens + attn
            tokens = tokens + block["mlp"](block["norm2"](tokens))
        high = self.from_attn(tokens)
        high = high.reshape(b, v, low_h, low_w, -1).permute(0, 1, 4, 2, 3).reshape(b * v, -1, low_h, low_w)
        high = F.pixel_shuffle(high, self.down_factor)
        high = high[..., :hp, :wp]
        high = high[..., :h, :w]
        high = self.post(high)
        return _finite(high.reshape(b, v, self.channels, h, w))


class PanoSourceGroupPoseUpdateHead(nn.Module):
    """Predict bounded source-view Gaussian group SE(3) corrections."""

    def __init__(self, channels: int, hidden_dim: int = 128, *, max_rotation_deg: float = 1.0, max_translation: float = 0.03, translation_scale_ratio: float = 0.005) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_rotation_rad = math.radians(float(max_rotation_deg))
        self.max_translation = float(max_translation)
        self.translation_scale_ratio = float(translation_scale_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 6),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        state: PanoGaussianState,
        source_tokens: torch.Tensor,
        context_poses_c2w: torch.Tensor,
    ) -> tuple[PanoGaussianState, dict[str, torch.Tensor]]:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must have shape BxVxC, got {tuple(source_tokens.shape)}")
        b, v, _ = [int(x) for x in source_tokens.shape]
        raw = self.net(source_tokens)
        rot = torch.tanh(raw[..., :3]) * float(self.max_rotation_rad)
        scale = state.means.detach().norm(dim=-1)
        valid = state.valid_mask
        batch_scale = []
        for batch_idx in range(b):
            if bool(valid[batch_idx].any()):
                batch_scale.append(scale[batch_idx][valid[batch_idx]].mean())
            else:
                batch_scale.append(scale.new_tensor(1.0))
        batch_scale_t = torch.stack(batch_scale).view(b, 1, 1).clamp_min(1.0e-6)
        trans_limit = torch.full_like(batch_scale_t, float(self.max_translation))
        if self.translation_scale_ratio > 0.0:
            trans_limit = torch.minimum(trans_limit, batch_scale_t * float(self.translation_scale_ratio))
        trans = torch.tanh(raw[..., 3:6]) * trans_limit
        rot_m = _axis_angle_to_matrix(rot)
        centers = context_poses_c2w[..., :3, 3].to(device=state.means.device, dtype=state.means.dtype)
        means = state.means
        corrected = means.clone()
        for view_idx in range(v):
            mask = (state.source_view_ids == view_idx) & state.valid_mask
            if not bool(mask.any()):
                continue
            center = centers[:, view_idx]
            rel = means - center[:, None, :]
            rotated = torch.einsum("bij,bnj->bni", rot_m[:, view_idx], rel) + center[:, None, :] + trans[:, view_idx, None, :]
            corrected = torch.where(mask.unsqueeze(-1), rotated, corrected)
        out = PanoGaussianState(
            means=_finite(corrected),
            log_scales=state.log_scales,
            rotations_unnorm=state.rotations_unnorm,
            opacity_logits=state.opacity_logits,
            sh_coeffs=state.sh_coeffs,
            latent_features=state.latent_features,
            source_view_ids=state.source_view_ids,
            source_uv=state.source_uv,
            valid_mask=state.valid_mask,
            confidence=state.confidence,
        )
        metrics = {
            "group_rot_deg_abs": rot.norm(dim=-1).mean().detach() * (180.0 / math.pi),
            "group_trans_norm": trans.norm(dim=-1).mean().detach(),
        }
        return out, metrics


class PanoViewPoseResidualHead(nn.Module):
    """Predict bounded per-context-view pose residuals for refinement renders."""

    def __init__(self, channels: int, hidden_dim: int = 128, *, max_rotation_deg: float = 1.0, max_translation: float = 0.03) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_rotation_rad = math.radians(float(max_rotation_deg))
        self.max_translation = float(max_translation)
        self.net = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 6),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, source_tokens: torch.Tensor, poses_c2w: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must have shape BxVxC, got {tuple(source_tokens.shape)}")
        if poses_c2w.ndim != 4 or tuple(poses_c2w.shape[:2]) != tuple(source_tokens.shape[:2]) or tuple(poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError("poses_c2w must have shape BxVx4x4 and share B,V with source_tokens")
        raw = self.net(source_tokens)
        rot = _bounded_vector(raw[..., :3], float(self.max_rotation_rad))
        trans = _bounded_vector(raw[..., 3:6], float(self.max_translation))
        delta = torch.eye(4, device=poses_c2w.device, dtype=poses_c2w.dtype).view(1, 1, 4, 4).repeat(
            int(poses_c2w.shape[0]), int(poses_c2w.shape[1]), 1, 1
        )
        delta[..., :3, :3] = _axis_angle_to_matrix(rot.to(device=poses_c2w.device, dtype=poses_c2w.dtype))
        delta[..., :3, 3] = trans.to(device=poses_c2w.device, dtype=poses_c2w.dtype)
        refined = poses_c2w @ delta
        metrics = {
            "pose_rot_deg_abs": rot.norm(dim=-1).mean().detach() * (180.0 / math.pi),
            "pose_trans_norm": trans.norm(dim=-1).mean().detach(),
        }
        return _finite(refined), metrics


class PanoRenderFeedbackEncoder(nn.Module):
    """Encode per-Gaussian feedback from context-only render residuals."""

    def __init__(
        self,
        feedback_dim: int = 32,
        hidden_dim: int = 64,
        *,
        feedback_type: str = "legacy",
        error_dim: int = 64,
        mv_down_factor: int = 4,
        mv_attn_blocks: int = 1,
        mv_num_heads: int = 4,
        feature_backbone: str = "resnet18",
        enable_group_correction: bool = False,
        group_rotation_deg: float = 1.0,
        group_translation: float = 0.03,
        group_translation_scale_ratio: float = 0.005,
        enable_pose_residual: bool = False,
        pose_rotation_deg: float = 1.0,
        pose_translation: float = 0.03,
    ) -> None:
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.hidden_dim = int(hidden_dim)
        self.feedback_type = str(feedback_type).lower()
        self.error_dim = int(error_dim)
        self.enable_group_correction = bool(enable_group_correction)
        self.enable_pose_residual = bool(enable_pose_residual)
        self.legacy_input_dim = 14
        self.encoder = nn.Sequential(
            nn.Linear(self.legacy_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feedback_dim),
        )
        self.rgb_error_proj = nn.Conv2d(3, self.error_dim, 1)
        self.feature_extractor = _FrozenFeatureExtractor(self.error_dim, backbone=feature_backbone)
        self.error_fuse = nn.Sequential(
            nn.Conv2d(self.error_dim * 2, self.error_dim, 1),
            nn.GroupNorm(max(1, min(8, self.error_dim)), self.error_dim),
            nn.GELU(),
        )
        self.error_decoder = PanoRenderErrorDecoder(
            self.error_dim,
            down_factor=mv_down_factor,
            num_heads=mv_num_heads,
            num_blocks=mv_attn_blocks,
        )
        self.feedback_fuse = nn.Sequential(
            nn.Linear(self.error_dim * 2 + 1, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feedback_dim),
        )
        self.group_head = PanoSourceGroupPoseUpdateHead(
            self.error_dim,
            hidden_dim=max(self.hidden_dim, self.error_dim),
            max_rotation_deg=group_rotation_deg,
            max_translation=group_translation,
            translation_scale_ratio=group_translation_scale_ratio,
        )
        self.pose_head = PanoViewPoseResidualHead(
            self.error_dim,
            hidden_dim=max(self.hidden_dim, self.error_dim),
            max_rotation_deg=pose_rotation_deg,
            max_translation=pose_translation,
        )

    def forward(
        self,
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        context_depth: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feedback, _state, debug = self.refine_state_and_feedback(
            state,
            context_images,
            context_poses_c2w,
            context_render_output,
            context_depth=context_depth,
            context_valid_mask=context_valid_mask,
            apply_group_correction=False,
        )
        return feedback, debug

    def refine_state_and_feedback(
        self,
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        context_depth: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        *,
        apply_group_correction: bool = True,
    ) -> tuple[torch.Tensor, PanoGaussianState, dict[str, torch.Tensor]]:
        if self.feedback_type in {"legacy", "projection", "projected"}:
            feedback, debug = self._legacy_feedback(
                state,
                context_images,
                context_poses_c2w,
                context_render_output,
                context_depth=context_depth,
                context_valid_mask=context_valid_mask,
            )
            return feedback, state, debug
        if self.feedback_type not in {"resplat_pano_error_decoder", "ghosting"}:
            raise ValueError(f"Unsupported Feedback.type: {self.feedback_type!r}")
        return self._resplat_pano_feedback(
            state,
            context_images,
            context_poses_c2w,
            context_render_output,
            context_valid_mask=context_valid_mask,
            apply_group_correction=apply_group_correction,
        )

    def _resplat_pano_feedback(
        self,
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        *,
        context_valid_mask: torch.Tensor | None,
        apply_group_correction: bool,
    ) -> tuple[torch.Tensor, PanoGaussianState, dict[str, torch.Tensor]]:
        b, v, _, h, w = self._validate_context(state, context_images, context_poses_c2w)
        render_rgb, _render_depth, _render_alpha = self._unpack_render_output(context_render_output, b, v, h, w)
        target = torch.nan_to_num(context_images.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_rgb = torch.nan_to_num(render_rgb.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        valid_maps = None
        if context_valid_mask is not None:
            valid_maps = self._normalize_context_mask(context_valid_mask, b, v, h, w).to(device=state.means.device, dtype=state.means.dtype)
        rgb_error = render_rgb - target
        rgb_flat = rgb_error.reshape(b * v, 3, h, w)
        rgb_token = self.rgb_error_proj(rgb_flat)
        render_flat = render_rgb.reshape(b * v, 3, h, w)
        target_flat = target.reshape(b * v, 3, h, w)
        render_feat = self.feature_extractor(render_flat, (h, w))
        with torch.no_grad():
            target_feat = self.feature_extractor(target_flat, (h, w))
        feat_error = render_feat - target_feat.detach()
        fused = self.error_fuse(torch.cat([rgb_token, feat_error], dim=1)).reshape(b, v, self.error_dim, h, w)
        if valid_maps is not None:
            fused = fused * valid_maps.to(dtype=fused.dtype)
        decoded = self.error_decoder(fused, valid_maps)
        source_feedback = self._sample_source_feedback(decoded, state)
        projected_feedback, projected_valid_ratio = self._sample_projected_feedback(
            decoded,
            state,
            context_poses_c2w,
            valid_maps,
        )
        feedback_raw = torch.cat([source_feedback, projected_feedback, projected_valid_ratio], dim=-1)
        feedback = self.feedback_fuse(_finite(feedback_raw))
        source_tokens = self._pool_source_tokens(decoded, valid_maps)
        corrected_state = state
        group_metrics: dict[str, torch.Tensor] = {}
        if self.enable_group_correction and apply_group_correction:
            corrected_state, group_metrics = self.group_head(state, source_tokens, context_poses_c2w)
        pose_metrics: dict[str, torch.Tensor] = {}
        refined_context_poses = None
        if self.enable_pose_residual:
            refined_context_poses, pose_metrics = self.pose_head(
                source_tokens,
                context_poses_c2w.to(device=decoded.device, dtype=decoded.dtype),
            )
        debug = {
            "mean_abs_residual": rgb_error.abs().mean().detach(),
            "feature_error_norm": feat_error.detach().norm(dim=1).mean(),
            "decoded_error_norm": decoded.detach().norm(dim=2).mean(),
            "source_feedback_norm": source_feedback.detach().norm(dim=-1).mean(),
            "projected_feedback_norm": projected_feedback.detach().norm(dim=-1).mean(),
            "projected_valid_ratio": projected_valid_ratio.detach().mean(),
            **group_metrics,
            **pose_metrics,
        }
        if refined_context_poses is not None:
            debug["refined_context_poses_c2w"] = refined_context_poses
        return _finite(feedback), corrected_state, debug

    def _legacy_feedback(
        self,
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
        context_render_output: PanoRenderOutput | dict[str, Any],
        context_depth: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, v, _, h, w = self._validate_context(state, context_images, context_poses_c2w)
        render_rgb, render_depth, render_alpha = self._unpack_render_output(context_render_output, b, v, h, w)
        target = torch.nan_to_num(context_images.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_rgb = torch.nan_to_num(render_rgb.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        render_depth = torch.nan_to_num(render_depth.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        render_alpha = torch.nan_to_num(render_alpha.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        depth_target = None
        if context_depth is not None:
            if context_depth.ndim != 5 or tuple(context_depth.shape[:2]) != (b, v):
                raise ValueError(f"context_depth must have shape BxVx1xHxW, got {tuple(context_depth.shape)}")
            depth_target = torch.nan_to_num(context_depth.to(device=state.means.device, dtype=state.means.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        valid_maps = None
        if context_valid_mask is not None:
            valid_maps = self._normalize_context_mask(context_valid_mask, b, v, h, w).to(device=state.means.device, dtype=state.means.dtype)
        per_view = []
        weights = []
        valid_ratios = []
        alpha_means = []
        residual_maps = render_rgb - target
        abs_maps = residual_maps.abs()
        for view_idx in range(v):
            projection = project_world_to_erp_grid(
                state.means,
                context_poses_c2w[:, view_idx].to(device=state.means.device, dtype=state.means.dtype),
                (h, w),
            )
            grid = projection.grid.view(b, state.num_gaussians, 1, 2)
            residual = _sample_wrap(residual_maps[:, view_idx], grid).transpose(1, 2)
            abs_residual = _sample_wrap(abs_maps[:, view_idx], grid).transpose(1, 2)
            alpha = _sample_wrap(render_alpha[:, view_idx], grid).transpose(1, 2)
            depth_render = _sample_wrap(render_depth[:, view_idx], grid).transpose(1, 2)
            if valid_maps is None:
                valid_sample = torch.ones_like(alpha[..., 0], dtype=torch.bool)
            else:
                valid_sample = _sample_wrap(valid_maps[:, view_idx], grid).transpose(1, 2)[..., 0] > 0.5
            if depth_target is None:
                depth_error = torch.zeros_like(depth_render)
            else:
                depth_gt = _sample_wrap(depth_target[:, view_idx], grid).transpose(1, 2)
                depth_error = ((depth_render - depth_gt) / depth_gt.abs().clamp_min(1.0)).clamp(-1.0, 1.0)
            camera_center = context_poses_c2w[:, view_idx, :3, 3].to(device=state.means.device, dtype=state.means.dtype)
            view_dir = F.normalize(state.means - camera_center[:, None, :], dim=-1, eps=1.0e-6)
            valid = projection.mask & state.valid_mask & valid_sample
            weight = valid.unsqueeze(-1).to(dtype=state.means.dtype) * alpha.clamp(0.0, 1.0)
            feat = torch.cat(
                [
                    residual,
                    abs_residual,
                    alpha,
                    depth_error,
                    depth_render / depth_render.detach().abs().mean().clamp_min(1.0),
                    valid.unsqueeze(-1).to(dtype=state.means.dtype),
                    view_dir,
                ],
                dim=-1,
            )
            per_view.append(feat)
            weights.append(weight)
            valid_ratios.append(valid.to(dtype=state.means.dtype).mean())
            alpha_means.append(alpha.mean())
        stacked = torch.stack(per_view, dim=2)
        weight_t = torch.stack(weights, dim=2)
        denom = weight_t.sum(dim=2).clamp_min(1.0e-6)
        raw = (stacked * weight_t).sum(dim=2) / denom
        raw = torch.cat([raw, (weight_t.sum(dim=2) / float(max(v, 1))).clamp(0.0, 1.0)], dim=-1)
        feedback = self.encoder(_finite(raw))
        debug = {
            "valid_projection_ratio": torch.stack(valid_ratios).mean().detach(),
            "mean_alpha": torch.stack(alpha_means).mean().detach(),
            "mean_abs_residual": abs_maps.mean().detach(),
            "feedback_weight_mean": weight_t.mean().detach(),
        }
        return _finite(feedback), debug

    def _sample_source_feedback(self, decoded: torch.Tensor, state: PanoGaussianState) -> torch.Tensor:
        b, v, c, h, w = [int(x) for x in decoded.shape]
        grid = _grid_from_pixel_uv(state.source_uv.to(device=decoded.device, dtype=decoded.dtype), (h, w)).view(b, state.num_gaussians, 1, 2)
        out = decoded.new_zeros(b, state.num_gaussians, c)
        for view_idx in range(v):
            sampled = _sample_wrap(decoded[:, view_idx], grid).transpose(1, 2)
            mask = (state.source_view_ids.to(device=decoded.device) == view_idx) & state.valid_mask.to(device=decoded.device)
            out = torch.where(mask.unsqueeze(-1), sampled, out)
        return _finite(out)

    def _sample_projected_feedback(
        self,
        decoded: torch.Tensor,
        state: PanoGaussianState,
        context_poses_c2w: torch.Tensor,
        valid_maps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, v, c, h, w = [int(x) for x in decoded.shape]
        total = decoded.new_zeros(b, state.num_gaussians, c)
        count = decoded.new_zeros(b, state.num_gaussians, 1)
        source_ids = state.source_view_ids.to(device=decoded.device)
        for view_idx in range(v):
            projection = project_world_to_erp_grid(
                state.means,
                context_poses_c2w[:, view_idx].to(device=state.means.device, dtype=state.means.dtype),
                (h, w),
            )
            grid = projection.grid.view(b, state.num_gaussians, 1, 2).to(device=decoded.device, dtype=decoded.dtype)
            sampled = _sample_wrap(decoded[:, view_idx], grid).transpose(1, 2)
            valid = projection.mask.to(device=decoded.device) & state.valid_mask.to(device=decoded.device) & (source_ids != view_idx)
            if valid_maps is not None:
                valid_sample = _sample_wrap(valid_maps[:, view_idx].to(device=decoded.device, dtype=decoded.dtype), grid).transpose(1, 2)[..., 0] > 0.5
                valid = valid & valid_sample
            weight = valid.unsqueeze(-1).to(dtype=decoded.dtype)
            total = total + sampled * weight
            count = count + weight
        return _finite(total / count.clamp_min(1.0)), (count / float(max(v - 1, 1))).clamp(0.0, 1.0)

    @staticmethod
    def _pool_source_tokens(decoded: torch.Tensor, valid_maps: torch.Tensor | None) -> torch.Tensor:
        if valid_maps is None:
            return decoded.mean(dim=(-1, -2)).transpose(1, 2).transpose(1, 2)
        valid = valid_maps.to(device=decoded.device, dtype=decoded.dtype)
        denom = valid.sum(dim=(-1, -2)).clamp_min(1.0)
        return (decoded * valid).sum(dim=(-1, -2)) / denom

    @staticmethod
    def _validate_context(
        state: PanoGaussianState,
        context_images: torch.Tensor,
        context_poses_c2w: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if context_images.ndim != 5 or int(context_images.shape[2]) != 3:
            raise ValueError(f"context_images must have shape BxVx3xHxW, got {tuple(context_images.shape)}")
        if context_poses_c2w.ndim != 4 or tuple(context_poses_c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"context_poses_c2w must have shape BxVx4x4, got {tuple(context_poses_c2w.shape)}")
        b, v, _, h, w = [int(x) for x in context_images.shape]
        if state.batch_size != b:
            raise ValueError("state and context_images must share batch size.")
        if tuple(context_poses_c2w.shape[:2]) != (b, v):
            raise ValueError("context_poses_c2w must share B,V with context_images.")
        return b, v, 3, h, w

    @staticmethod
    def _normalize_context_mask(mask: torch.Tensor, b: int, v: int, h: int, w: int) -> torch.Tensor:
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        if mask.ndim != 5 or tuple(mask.shape[:2]) != (b, v) or int(mask.shape[2]) != 1:
            raise ValueError(f"context_valid_mask must have shape BxVxHxW or BxVx1xHxW, got {tuple(mask.shape)}")
        if tuple(mask.shape[-2:]) != (h, w):
            mask = F.interpolate(mask.float(), size=(h, w), mode="nearest")
        return mask.float()

    @staticmethod
    def _unpack_render_output(
        output: PanoRenderOutput | dict[str, Any],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, PanoRenderOutput):
            color, depth, alpha = output.color, output.depth, output.alpha
        elif isinstance(output, dict):
            color = output.get("color", output.get("render"))
            depth = output.get("depth")
            alpha = output.get("alpha", output.get("opacity"))
        else:
            raise TypeError(f"Unsupported context_render_output type: {type(output)!r}")
        if not torch.is_tensor(color):
            raise ValueError("context_render_output must contain color/render tensor.")
        if color.ndim == 4:
            color = color.unsqueeze(1)
        if color.ndim != 5 or tuple(color.shape[:2]) != (b, v):
            raise ValueError(f"render color must have shape BxVx3xHxW, got {tuple(color.shape)}")
        if not torch.is_tensor(depth):
            depth = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if not torch.is_tensor(alpha):
            alpha = torch.zeros(b, v, 1, h, w, device=color.device, dtype=color.dtype)
        elif alpha.ndim == 4:
            alpha = alpha.unsqueeze(1)
        return color, depth, alpha
