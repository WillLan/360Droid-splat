"""Full-resolution ERP recurrent refiner for Stage 3 per-pixel Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

from geometry.spherical_erp import sample_erp_with_wrap, unit_ray_to_erp_pixel
from .per_pixel_gaussian_observation import (
    BatchedExplicitPerPixelGaussianSet,
    SH_C0,
    ExplicitPerPixelGaussianSet,
    PerPixelGaussianObservation,
    normalize_quaternion,
    quaternion_multiply,
)
from .spherical_selfi_gaussian_head import ERPConv2d, erp_bilinear_resize

if TYPE_CHECKING:
    from .spherical_selfi_stage3_ba import Stage3BAOutput, Stage3MatchCache


def _groups(channels: int) -> int:
    count = min(8, int(channels))
    while count > 1 and int(channels) % count:
        count -= 1
    return count


def _channel_last_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 1, 3, 4, 2)


def _channel_first_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3)


def quaternion_inverse(value: torch.Tensor) -> torch.Tensor:
    normalized = normalize_quaternion(value)
    out = normalized.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quaternion_log_map(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    quaternion = normalize_quaternion(value)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, quaternion[..., :1].clamp_min(eps))
    scale = torch.where(norm > eps, angle / norm.clamp_min(eps), torch.full_like(norm, 2.0))
    return vector * scale


def quaternion_exp_map(rotation: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    angle = rotation.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(angle > eps, torch.sin(half) / angle.clamp_min(eps), 0.5 - angle.square() / 48.0)
    return normalize_quaternion(torch.cat([torch.cos(half), scale * rotation], dim=-1))


class ERPDepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.dilation = max(1, int(dilation))
        self.conv = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=0,
            dilation=self.dilation,
            groups=int(channels),
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pad = self.dilation
        value = F.pad(value, (pad, pad, 0, 0), mode="circular")
        value = F.pad(value, (0, 0, pad, pad), mode="replicate")
        return self.conv(value)


class SphericalContextBlock(nn.Module):
    """Lightweight ERP-aware residual context at full image resolution."""

    def __init__(self, channels: int = 64, *, dilation: int = 1) -> None:
        super().__init__()
        channels = int(channels)
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = ERPDepthwiseConv2d(channels, dilation=dilation)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(value))
        hidden = self.pointwise(self.depthwise(hidden))
        hidden = self.output(F.gelu(self.norm2(hidden)))
        return value + hidden


class SphericalConvGRUCell(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        combined = int(input_dim) + self.hidden_dim
        self.gates = ERPConv2d(combined, 2 * self.hidden_dim, kernel_size=3)
        self.candidate = ERPConv2d(combined, self.hidden_dim, kernel_size=3)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat([value, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([value, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class _ERPWrappedConv(nn.Module):
    """Wrap a pretrained Conv2d with circular/replicated ERP padding."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        conv.padding = (0, 0)
        self.conv = conv

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        kernel_h, kernel_w = self.conv.kernel_size
        dilation_h, dilation_w = self.conv.dilation
        pad_h = ((kernel_h - 1) * dilation_h) // 2
        pad_w = ((kernel_w - 1) * dilation_w) // 2
        if pad_w:
            value = F.pad(value, (pad_w, pad_w, 0, 0), mode="circular")
        if pad_h:
            value = F.pad(value, (0, 0, pad_h, pad_h), mode="replicate")
        return self.conv(value)


class _ERPWrappedMaxPool(nn.Module):
    def __init__(self, pool: nn.MaxPool2d) -> None:
        super().__init__()
        pool.padding = 0
        self.pool = pool

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        kernel = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]
        pad = (int(kernel) - 1) // 2
        if pad:
            value = F.pad(value, (pad, pad, 0, 0), mode="circular")
            value = F.pad(value, (0, 0, pad, pad), mode="replicate")
        return self.pool(value)


def _replace_resnet_convs(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _ERPWrappedConv(child))
        else:
            _replace_resnet_convs(child)


class FrozenERPResNet18(nn.Module):
    """Frozen ImageNet ResNet18 feature taps with ERP padding."""

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover - real training dependency
            raise RuntimeError("Stage 3 ResNet error features require torchvision.") from exc
        weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
        backbone = resnet18(weights=weights)
        _replace_resnet_convs(backbone)
        backbone.maxpool = _ERPWrappedMaxPool(backbone.maxpool)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.relu(self.bn1(self.conv1(image)))
        level1 = self.layer1(self.maxpool(stem))
        level2 = self.layer2(level1)
        return stem, level1, level2


@dataclass
class EncodedTargetReference:
    image: torch.Tensor
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


class ReSplatErrorEncoder(nn.Module):
    """Encode detached rendered-vs-target residuals at quarter resolution."""

    def __init__(self, *, use_resnet: bool = True, pretrained_resnet: bool = True) -> None:
        super().__init__()
        self.use_resnet = bool(use_resnet)
        self.backbone = FrozenERPResNet18(pretrained=pretrained_resnet) if self.use_resnet else None
        self.pixel_projection = nn.Sequential(
            nn.Conv2d(48, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
        )
        self.bottleneck = nn.Conv2d(256, 32, kernel_size=1)

    @torch.no_grad()
    def encode_reference(self, target: torch.Tensor) -> EncodedTargetReference:
        features = self.backbone(target.float()) if self.backbone is not None else None
        return EncodedTargetReference(image=target.detach(), features=features)

    def forward(self, rendered: torch.Tensor, reference: EncodedTargetReference) -> torch.Tensor:
        render = rendered.detach().float()
        target = reference.image.to(device=render.device, dtype=render.dtype)
        height, width = int(render.shape[-2]), int(render.shape[-1])
        quarter = (max(1, height // 4), max(1, width // 4))
        resize_shape = (quarter[0] * 4, quarter[1] * 4)
        residual = render - target
        if tuple(residual.shape[-2:]) != resize_shape:
            residual = F.interpolate(residual, size=resize_shape, mode="bilinear", align_corners=False)
        pixel = self.pixel_projection(F.pixel_unshuffle(residual, 4))
        if self.backbone is None:
            feature_error = torch.zeros_like(pixel)
        else:
            render_features = self.backbone(render)
            assert reference.features is not None
            resized = [
                F.interpolate(current - base.to(device=current.device), size=quarter, mode="bilinear", align_corners=False)
                for current, base in zip(render_features, reference.features)
            ]
            feature_error = self.feature_projection(torch.cat(resized, dim=1))
        return self.bottleneck(pixel + feature_error)


def scatter_materialized_visibility(
    materialized: ExplicitPerPixelGaussianSet,
    visibility_filter: torch.Tensor,
    *,
    frame_ids: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Scatter target-render visibility back to canonical source pixels."""

    device = materialized.xyz.device
    out = torch.zeros(int(frame_ids.numel()), 1, int(height), int(width), device=device, dtype=torch.bool)
    visible = visibility_filter.to(device=device).bool().reshape(-1)
    if int(visible.numel()) != int(materialized.source_frame_index.numel()):
        raise ValueError("visibility_filter must align with the materialized Gaussian set.")
    if not visible.any():
        return out
    ids = materialized.source_frame_index[visible]
    uv = materialized.source_pixel_uv[visible]
    x = torch.floor(uv[:, 0]).long().remainder(int(width))
    y = torch.floor(uv[:, 1]).long().clamp(0, int(height) - 1)
    for local, frame_id in enumerate(frame_ids.tolist()):
        mask = ids == int(frame_id)
        if mask.any():
            out[local, 0, y[mask], x[mask]] = True
    return out


def scatter_batched_materialized_visibility(
    materialized: BatchedExplicitPerPixelGaussianSet,
    visibility_filter: torch.Tensor,
    *,
    frame_ids: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Scatter ``CxN`` visibility to ``CxSx1xHxW`` source pixels."""

    device = materialized.xyz.device
    camera_count = materialized.num_cameras
    visible = visibility_filter.to(device=device).bool()
    expected = (camera_count, int(materialized.source_frame_index.numel()))
    if tuple(visible.shape) != expected:
        raise ValueError(f"visibility_filter must have shape {expected}.")
    output = torch.zeros(
        camera_count,
        int(frame_ids.numel()),
        1,
        int(height),
        int(width),
        device=device,
        dtype=torch.bool,
    )
    if not visible.any():
        return output
    ids = materialized.source_frame_index
    uv = materialized.source_pixel_uv
    x = torch.floor(uv[:, 0]).long().remainder(int(width))
    y = torch.floor(uv[:, 1]).long().clamp(0, int(height) - 1)
    for camera in range(camera_count):
        camera_visible = visible[camera]
        if not camera_visible.any():
            continue
        for local, frame_id in enumerate(frame_ids.tolist()):
            mask = camera_visible & (ids == int(frame_id))
            if mask.any():
                output[camera, local, 0, y[mask], x[mask]] = True
    return output


class SphericalErrorRouter(nn.Module):
    """Project target-view error maps back to their contributing source pixels."""

    def __init__(self, error_dim: int = 32, global_dim: int = 8) -> None:
        super().__init__()
        self.error_dim = int(error_dim)
        self.global_dim = int(global_dim)
        self.global_projection = nn.Linear(self.error_dim, self.global_dim)
        self.output_projection = nn.Conv2d(2 * self.error_dim + 1 + self.global_dim, self.error_dim, kernel_size=1)

    def forward(
        self,
        observation: PerPixelGaussianObservation,
        target_error_maps: torch.Tensor,
        rendered_depth: torch.Tensor,
        rendered_alpha: torch.Tensor,
        *,
        target_source_visibility: torch.Tensor | None = None,
        alpha_threshold: float = 0.05,
        depth_abs_threshold: float = 0.10,
        depth_rel_threshold: float = 0.05,
    ) -> torch.Tensor:
        if target_error_maps.ndim != 5:
            raise ValueError("target_error_maps must have shape BxSxCxH4xW4.")
        batch, views, channels, low_h, low_w = (int(value) for value in target_error_maps.shape)
        height, width = observation.image_size
        if (batch, views, channels) != (observation.batch_size, observation.num_source_views, self.error_dim):
            raise ValueError("Error maps and observation dimensions do not match.")
        if tuple(rendered_depth.shape) != (batch, views, 1, height, width):
            raise ValueError("rendered_depth must have shape BxSx1xHxW.")
        if tuple(rendered_alpha.shape) != tuple(rendered_depth.shape):
            raise ValueError("rendered_alpha must match rendered_depth.")

        centers = observation.centers_world()
        signed_sum = centers.new_zeros(batch, views, self.error_dim, height, width)
        absolute_sum = torch.zeros_like(signed_sum)
        count = centers.new_zeros(batch, views, 1, height, width)
        rows = torch.arange(low_h, device=centers.device, dtype=centers.dtype) + 0.5
        area = torch.cos(math.pi * (rows / float(low_h) - 0.5)).clamp_min(0.0).view(1, 1, low_h, 1)
        global_feature = (target_error_maps * area).sum(dim=(-2, -1)) / area.sum().clamp_min(1.0e-8) / float(low_w)
        global_token = self.global_projection(global_feature.float()).to(dtype=centers.dtype)

        for target in range(views):
            pose = observation.poses_c2w[:, target].to(device=centers.device, dtype=centers.dtype)
            rotation = pose[:, :3, :3]
            translation = pose[:, :3, 3]
            for source in range(views):
                point_target = torch.einsum(
                    "bij,bhwj->bhwi",
                    rotation.transpose(1, 2),
                    centers[:, source] - translation[:, None, None, :],
                )
                distance = point_target.norm(dim=-1)
                ray = F.normalize(point_target, dim=-1, eps=1.0e-8)
                uv = unit_ray_to_erp_pixel(ray, height, width)
                low_uv = uv.clone()
                low_uv[..., 0] *= float(low_w) / float(width)
                low_uv[..., 1] *= float(low_h) / float(height)
                sampled_error = sample_erp_with_wrap(target_error_maps[:, target], low_uv).permute(0, 3, 1, 2)
                sampled_depth = sample_erp_with_wrap(rendered_depth[:, target], uv)[..., 0]
                sampled_alpha = sample_erp_with_wrap(rendered_alpha[:, target], uv)[..., 0]
                # Clone before the in-place gates below; slicing the canonical
                # mask returns a view and must never mutate observation state.
                valid = observation.valid_mask[:, source, 0].bool().clone()
                valid &= torch.isfinite(distance) & torch.isfinite(sampled_depth) & torch.isfinite(sampled_alpha)
                valid &= sampled_alpha > float(alpha_threshold)
                valid &= (sampled_depth - distance).abs() <= float(depth_abs_threshold) + float(depth_rel_threshold) * distance
                if target_source_visibility is not None:
                    valid &= target_source_visibility[:, target, source, 0].bool()
                weight = valid.to(dtype=sampled_error.dtype).unsqueeze(1)
                signed_sum[:, source] += sampled_error * weight
                absolute_sum[:, source] += sampled_error.abs() * weight
                count[:, source] += weight

        denom = count.clamp_min(1.0)
        signed_mean = signed_sum / denom
        absolute_mean = absolute_sum / denom
        coverage = (count / max(1, views)).clamp(0.0, 1.0)
        source_global = global_token.mean(dim=1, keepdim=True).expand(-1, views, -1)
        global_map = source_global[:, :, :, None, None].expand(-1, -1, -1, height, width)
        flat = torch.cat([signed_mean, absolute_mean, coverage, global_map], dim=2)
        return self.output_projection(flat.reshape(batch * views, flat.shape[2], height, width)).reshape(
            batch, views, self.error_dim, height, width
        )


@dataclass
class Stage3RefinerOutput:
    observation: PerPixelGaussianObservation
    hidden: torch.Tensor
    raw_geometry: torch.Tensor
    raw_appearance: torch.Tensor
    normalized_update_energy: torch.Tensor
    profile: dict[str, float] | None = None


@dataclass
class Stage3RefinementResult:
    final_observation: PerPixelGaussianObservation
    initial_observation: PerPixelGaussianObservation
    snapshot_observations: dict[str, PerPixelGaussianObservation]
    ba_outputs: tuple["Stage3BAOutput", ...]
    match_cache: "Stage3MatchCache"
    diagnostics: dict[str, float]


class SphericalRecurrentGaussianRefiner(nn.Module):
    """Shared-weight, three-step per-pixel Gaussian update network."""

    depth_limits = (0.15, 0.10, 0.05)
    rotation_limits_deg = (5.0, 3.0, 2.0)
    scale_limits = (0.15, 0.10, 0.05)
    rgb_dc_limits = (0.25, 0.15, 0.10)
    rgb_ac_limits = (0.10, 0.075, 0.05)
    density_dc_limits = (1.0, 0.75, 0.5)
    density_ac_limits = (0.5, 0.35, 0.25)

    def __init__(self, *, adapter_dim: int = 24, hidden_dim: int = 32) -> None:
        super().__init__()
        self.adapter_dim = int(adapter_dim)
        self.hidden_dim = int(hidden_dim)
        self.static_encoder = nn.Sequential(
            ERPConv2d(self.adapter_dim + 3, 32, kernel_size=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Conv2d(39, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
        )
        self.fusion = nn.Conv2d(96, 64, kernel_size=1)
        self.context = SphericalContextBlock(64, dilation=1)
        self.gru = SphericalConvGRUCell(64, self.hidden_dim)
        self.hidden_initializer = nn.Conv2d(64, self.hidden_dim, kernel_size=1)
        self.geometry_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 64, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 7, kernel_size=1),
        )
        self.appearance_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 64, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 31, kernel_size=1),
        )
        self._zero_output_heads()

    def _zero_output_heads(self) -> None:
        for head in (self.geometry_head, self.appearance_head):
            output = head[-1]
            assert isinstance(output, nn.Conv2d)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _state_tensor(
        observation: PerPixelGaussianObservation,
        stage2_observation: PerPixelGaussianObservation,
    ) -> torch.Tensor:
        depth_ratio = torch.log(
            observation.refined_depth.clamp_min(1.0e-6)
            / stage2_observation.refined_depth.to(observation.refined_depth).clamp_min(1.0e-6)
        )
        current_q = _channel_last_quaternion(observation.local_quaternion)
        base_q = _channel_last_quaternion(stage2_observation.local_quaternion).to(current_q)
        relative_q = quaternion_multiply(current_q, quaternion_inverse(base_q))
        rotation = quaternion_log_map(relative_q).permute(0, 1, 4, 2, 3)
        rgb = observation.rgb_sh.reshape(
            observation.batch_size,
            observation.num_source_views,
            -1,
            *observation.image_size,
        )
        return torch.cat(
            [
                depth_ratio,
                rotation,
                observation.log_scale_multiplier,
                rgb,
                observation.density_sh,
                observation.confidence,
            ],
            dim=2,
        )

    def forward(
        self,
        observation: PerPixelGaussianObservation,
        stage2_observation: PerPixelGaussianObservation,
        adapter_features: torch.Tensor,
        images: torch.Tensor,
        error_features: torch.Tensor,
        *,
        iteration_index: int,
        hidden: torch.Tensor | None = None,
        min_geometry_depth: float = 0.05,
        max_geometry_depth: float = 20.0,
    ) -> Stage3RefinerOutput:
        batch, views, _, height, width = adapter_features.shape
        if tuple(images.shape) != (batch, views, 3, height, width):
            raise ValueError("images must match adapter feature B/S/H/W dimensions.")
        if tuple(error_features.shape) != (batch, views, 32, height, width):
            raise ValueError("error_features must have shape BxSx32xHxW.")
        index = int(iteration_index)
        if index < 0 or index >= 3:
            raise ValueError("iteration_index must be 0, 1, or 2.")
        static_input = torch.cat([adapter_features, images], dim=2).reshape(batch * views, self.adapter_dim + 3, height, width)
        static = self.static_encoder(static_input)
        state_value = self._state_tensor(observation, stage2_observation)
        state = self.state_encoder(state_value.reshape(batch * views, 39, height, width))
        error = error_features.reshape(batch * views, 32, height, width)
        fused = self.context(self.fusion(torch.cat([static, state, error], dim=1)))
        initialized_hidden = torch.tanh(self.hidden_initializer(torch.cat([static, state], dim=1)))
        if hidden is None:
            hidden = initialized_hidden
        else:
            # Keep the shared initializer in every DDP graph without changing
            # recurrent semantics on iterations two and three.
            hidden = hidden + initialized_hidden * 0.0
        if tuple(hidden.shape) != (batch * views, self.hidden_dim, height, width):
            raise ValueError("hidden has the wrong shape for this observation.")
        next_hidden = self.gru(fused, hidden)
        geometry = self.geometry_head(torch.cat([next_hidden, state, error], dim=1))
        appearance = self.appearance_head(torch.cat([next_hidden, static, error], dim=1))

        geometry = geometry.reshape(batch, views, 7, height, width)
        appearance = appearance.reshape(batch, views, 31, height, width)
        geometry_mask = (
            observation.valid_mask.bool()
            & torch.isfinite(observation.refined_depth)
            & (observation.refined_depth >= float(min_geometry_depth))
            & (observation.refined_depth <= float(max_geometry_depth))
        )
        appearance_mask = observation.valid_mask.bool()

        depth_delta = float(self.depth_limits[index]) * torch.tanh(geometry[:, :, 0:1])
        refined_depth = observation.refined_depth * (1.0 + depth_delta * geometry_mask)

        raw_rotation = torch.tanh(geometry[:, :, 1:4]).permute(0, 1, 3, 4, 2)
        max_angle = math.radians(float(self.rotation_limits_deg[index]))
        rotation_norm = raw_rotation.norm(dim=-1, keepdim=True)
        rotation_vector = raw_rotation * torch.clamp(max_angle / rotation_norm.clamp_min(1.0e-8), max=1.0)
        rotation_vector = rotation_vector * geometry_mask.permute(0, 1, 3, 4, 2)
        delta_quaternion = quaternion_exp_map(rotation_vector)
        quaternion = quaternion_multiply(delta_quaternion, _channel_last_quaternion(observation.local_quaternion))
        quaternion = _channel_first_quaternion(normalize_quaternion(quaternion))

        scale_delta = float(self.scale_limits[index]) * torch.tanh(geometry[:, :, 4:7])
        log_scale = observation.log_scale_multiplier + scale_delta * geometry_mask

        rgb_raw = appearance[:, :, :27].reshape(batch, views, 9, 3, height, width)
        rgb_limit = rgb_raw.new_full((1, 1, 9, 1, 1, 1), float(self.rgb_ac_limits[index]))
        rgb_limit[:, :, 0] = float(self.rgb_dc_limits[index])
        rgb_delta = rgb_limit * torch.tanh(rgb_raw)
        rgb = observation.rgb_sh + rgb_delta * appearance_mask.unsqueeze(2)

        density_raw = appearance[:, :, 27:31]
        density_delta = float(self.density_ac_limits[index]) * torch.tanh(density_raw)
        density_delta = density_delta.clone()
        density_delta[:, :, 0:1] = (
            float(self.density_dc_limits[index]) / float(SH_C0)
        ) * torch.tanh(density_raw[:, :, 0:1])
        density = observation.density_sh + density_delta * appearance_mask

        updated = observation.with_updates(
            refined_depth=refined_depth,
            local_quaternion=quaternion,
            log_scale_multiplier=log_scale,
            rgb_sh=rgb,
            density_sh=density,
        )
        energy = torch.stack(
            [
                depth_delta.square().mean(),
                rotation_vector.square().mean(),
                scale_delta.square().mean(),
                rgb_delta.square().mean(),
                density_delta.square().mean(),
            ]
        ).mean()
        return Stage3RefinerOutput(
            observation=updated,
            hidden=next_hidden,
            raw_geometry=geometry,
            raw_appearance=appearance,
            normalized_update_energy=energy,
            profile=None,
        )
