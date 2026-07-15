"""Spherical Selfi DPT-style adapter for Stage 1B."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from geometry.spherical_erp import DEFAULT_ERP_HEIGHT, DEFAULT_ERP_WIDTH
from .panovggt_feature_wrapper import normalize_stage_feature


@dataclass(frozen=True)
class LoadedSphericalSelfiAdapter:
    """A frozen adapter together with verified checkpoint provenance."""

    module: "SphericalSelfiDPTAdapter"
    checkpoint_path: str
    sha256: str
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num_groups(channels: int) -> int:
    groups = min(8, int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _normalize_size_list(
    value: list[tuple[int, int] | list[int] | None] | tuple[tuple[int, int] | list[int] | None, ...] | None,
    *,
    expected_len: int,
    name: str,
) -> list[tuple[int, int] | None]:
    if value is None:
        return [None] * expected_len
    if len(value) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} entries when provided.")
    normalized: list[tuple[int, int] | None] = []
    for item in value:
        if item is None:
            normalized.append(None)
            continue
        if len(item) != 2:
            raise ValueError(f"{name} entries must be (height, width), got {item!r}.")
        height, width = int(item[0]), int(item[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} entries must be positive, got {(height, width)!r}.")
        normalized.append((height, width))
    return normalized


def _normalize_optional_size(value: tuple[int, int] | list[int] | None, *, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must be (height, width), got {value!r}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)!r}.")
    return height, width


class _ERPConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        use_circular_padding: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.use_circular_padding = bool(use_circular_padding)
        padding = 0 if self.kernel_size > 1 and self.use_circular_padding else self.kernel_size // 2
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=self.kernel_size, padding=padding)
        groups = _num_groups(int(out_channels))
        self.norm = nn.GroupNorm(groups, int(out_channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size > 1 and self.use_circular_padding:
            pad = self.kernel_size // 2
            x = F.pad(x, (pad, pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        return self.act(self.norm(self.conv(x)))


class _FusionBlock(nn.Module):
    def __init__(self, channels: int, *, use_circular_padding: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _ERPConvNormAct(channels, channels, use_circular_padding=use_circular_padding),
            _ERPConvNormAct(channels, channels, use_circular_padding=use_circular_padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SphericalSelfiDPTAdapter(nn.Module):
    """Fuse 4-stage PanoVGGT features into full-resolution ERP descriptors."""

    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...],
        *,
        hidden_dim: int = 128,
        out_dim: int = 24,
        image_height: int = DEFAULT_ERP_HEIGHT,
        image_width: int = DEFAULT_ERP_WIDTH,
        use_circular_padding: bool = True,
        norm_output: bool = True,
        token_hw: list[tuple[int, int] | None] | tuple[tuple[int, int] | None, ...] | None = None,
        reassemble_sizes: list[tuple[int, int] | list[int] | None] | tuple[tuple[int, int] | list[int] | None, ...] | None = None,
        fusion_output_size: tuple[int, int] | list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"SphericalSelfiDPTAdapter requires 4 input stages, got {len(in_channels)}.")
        if int(out_dim) <= 0:
            raise ValueError("out_dim must be positive.")
        if int(image_height) <= 0 or int(image_width) <= 0:
            raise ValueError("image_height and image_width must be positive.")
        self.in_channels = [int(value) for value in in_channels]
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.use_circular_padding = bool(use_circular_padding)
        self.norm_output = bool(norm_output)
        self.token_hw = list(token_hw) if token_hw is not None else [None] * 4
        if len(self.token_hw) != 4:
            raise ValueError("token_hw must contain 4 entries when provided.")
        self.reassemble_sizes = _normalize_size_list(
            reassemble_sizes,
            expected_len=4,
            name="reassemble_sizes",
        )
        self.fusion_output_size = _normalize_optional_size(fusion_output_size, name="fusion_output_size")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(_num_groups(self.hidden_dim), self.hidden_dim),
                    nn.GELU(),
                )
                for channels in self.in_channels
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [_FusionBlock(self.hidden_dim, use_circular_padding=self.use_circular_padding) for _ in range(4)]
        )
        self.output_refine = _FusionBlock(self.hidden_dim, use_circular_padding=self.use_circular_padding)
        self.output_proj = nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1)

    def _infer_batch_views(self, stage_features: list[Any], batch_size: int | None, num_views: int | None) -> tuple[int, int]:
        first_tensor = None
        for value in stage_features:
            if torch.is_tensor(value):
                first_tensor = value
                break
            if isinstance(value, (list, tuple)):
                tensors = [item for item in value if torch.is_tensor(item)]
                if tensors:
                    first_tensor = tensors[-1]
                    break
            if isinstance(value, dict):
                tensors = [item for item in value.values() if torch.is_tensor(item)]
                if tensors:
                    first_tensor = tensors[-1]
                    break
        if first_tensor is None:
            raise ValueError("Could not infer batch/view dimensions because no tensor feature was provided.")
        if first_tensor.ndim == 5:
            return int(first_tensor.shape[0]), int(first_tensor.shape[1])
        if first_tensor.ndim == 4 and int(first_tensor.shape[-1]) == self.in_channels[0]:
            return int(first_tensor.shape[0]), int(first_tensor.shape[1])
        if first_tensor.ndim == 4 and batch_size is None and num_views is None:
            return 1, int(first_tensor.shape[0])
        if batch_size is None or num_views is None:
            raise ValueError("batch_size and num_views are required for flattened B*V features or B*V tokens.")
        if int(batch_size) * int(num_views) != int(first_tensor.shape[0]):
            raise ValueError(
                f"batch_size*num_views={int(batch_size) * int(num_views)} does not match feature leading dimension "
                f"{int(first_tensor.shape[0])}."
            )
        return int(batch_size), int(num_views)

    def _normalize_inputs(
        self,
        stage_features: list[Any] | tuple[Any, ...],
        *,
        batch_size: int | None,
        num_views: int | None,
    ) -> list[torch.Tensor]:
        if len(stage_features) != 4:
            raise ValueError(f"Expected 4 stage features, got {len(stage_features)}.")
        b, v = self._infer_batch_views(list(stage_features), batch_size, num_views)
        normalized = [
            normalize_stage_feature(
                feature,
                batch_size=b,
                num_views=v,
                image_hw=(self.image_height, self.image_width),
                token_hw=self.token_hw[idx],
            )
            for idx, feature in enumerate(stage_features)
        ]
        for idx, (feature, channels) in enumerate(zip(normalized, self.in_channels)):
            if int(feature.shape[2]) != int(channels):
                raise ValueError(
                    f"Stage {idx} expected {channels} channels, got {int(feature.shape[2])} "
                    f"from shape {tuple(feature.shape)}."
                )
        return normalized

    @staticmethod
    def _flatten_bv(feature: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, v, c, h, w = (int(dim) for dim in feature.shape)
        return feature.reshape(b * v, c, h, w), (b, v)

    @staticmethod
    def _restore_bv(feature: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        b, v = shape
        return feature.view(b, v, int(feature.shape[1]), int(feature.shape[2]), int(feature.shape[3]))

    def _forward_flat(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Run the adapter for an already flattened B*V feature batch."""

        projected: list[torch.Tensor] = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            target_size = self.reassemble_sizes[len(projected)]
            if target_size is not None and tuple(value.shape[-2:]) != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            projected.append(value)

        fused = self.fusion_blocks[-1](projected[-1])
        for idx in range(2, -1, -1):
            fused = F.interpolate(fused, size=projected[idx].shape[-2:], mode="bilinear", align_corners=False)
            fused = self.fusion_blocks[idx](fused + projected[idx])
        refine_size = self.fusion_output_size or (self.image_height, self.image_width)
        if tuple(fused.shape[-2:]) != refine_size:
            fused = F.interpolate(fused, size=refine_size, mode="bilinear", align_corners=False)
        fused = self.output_refine(fused)
        if tuple(fused.shape[-2:]) != (self.image_height, self.image_width):
            fused = F.interpolate(
                fused,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
        dense = self.output_proj(fused)
        if self.norm_output:
            dense = F.normalize(dense, dim=1, eps=1.0e-6)
        return dense

    def forward(
        self,
        stage_features: list[Any] | tuple[Any, ...],
        *,
        batch_size: int | None = None,
        num_views: int | None = None,
        flat_batch_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Return dense spherical features with shape ``B x V x out_dim x H x W``."""

        normalized = self._normalize_inputs(stage_features, batch_size=batch_size, num_views=num_views)
        flattened = [self._flatten_bv(feature)[0] for feature in normalized]
        restore_shape = (int(normalized[0].shape[0]), int(normalized[0].shape[1]))
        flat_count = int(flattened[0].shape[0])
        chunk_size = flat_count if flat_batch_chunk_size is None else int(flat_batch_chunk_size)
        if chunk_size <= 0:
            chunk_size = flat_count
        chunks = [
            self._forward_flat([feature[start : start + chunk_size] for feature in flattened])
            for start in range(0, flat_count, chunk_size)
        ]
        return self._restore_bv(torch.cat(chunks, dim=0), restore_shape)


def load_spherical_selfi_adapter_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_sha256: str | None = None,
    expected_out_dim: int = 24,
    expected_normalized: bool = True,
    expected_image_size: tuple[int, int] | None = None,
    expected_stage_hooks: list[str] | tuple[str, ...] | None = None,
    expected_token_hw: list[list[int] | tuple[int, int] | None] | tuple[list[int] | tuple[int, int] | None, ...] | None = None,
    expected_token_start_idx: list[int | None] | tuple[int | None, ...] | None = None,
    expected_pose_convention: str | None = None,
    expected_depth_convention: str | None = None,
) -> LoadedSphericalSelfiAdapter:
    """Load, validate, and freeze a ``spherical_selfi_adapter_v1`` checkpoint."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spherical Selfi adapter checkpoint does not exist: {path}")
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
        raise ValueError(
            "Spherical Selfi adapter checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, got {digest}."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "spherical_selfi_adapter_v1":
        actual = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"Unsupported Stage 1 adapter checkpoint format: {actual!r}.")
    state = payload.get("adapter")
    config = payload.get("adapter_config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Stage 1 adapter checkpoint must contain adapter and adapter_config mappings.")
    required = ("in_channels", "hidden_dim", "out_dim", "image_height", "image_width")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Stage 1 adapter checkpoint is missing config keys: {missing}.")
    if int(config["out_dim"]) != int(expected_out_dim):
        raise ValueError(
            f"Stage 1 adapter descriptor dimension must be {expected_out_dim}, got {config['out_dim']}."
        )
    if bool(config.get("norm_output", True)) != bool(expected_normalized):
        raise ValueError(
            "Stage 1 adapter normalization mismatch: "
            f"expected norm_output={bool(expected_normalized)}, got {config.get('norm_output')!r}."
        )
    image_size = (int(config["image_height"]), int(config["image_width"]))
    if expected_image_size is not None and image_size != tuple(int(value) for value in expected_image_size):
        raise ValueError(
            f"Stage 1 adapter image size mismatch: expected {expected_image_size}, got {image_size}."
        )
    training_config = payload.get("training_config")
    training_panovggt = training_config.get("panovggt", {}) if isinstance(training_config, dict) else {}
    if expected_stage_hooks is not None:
        saved_hooks = training_panovggt.get("stage_hooks")
        if saved_hooks is None or [str(value) for value in saved_hooks] != [str(value) for value in expected_stage_hooks]:
            raise ValueError(
                "Stage 1 adapter hook metadata mismatch: "
                f"expected {list(expected_stage_hooks)!r}, got {saved_hooks!r}."
            )
    if expected_token_hw is not None:
        def normalize_grids(values):
            return [None if value is None else [int(value[0]), int(value[1])] for value in values]

        saved_grids = training_panovggt.get("token_hw")
        if saved_grids is None or normalize_grids(saved_grids) != normalize_grids(expected_token_hw):
            raise ValueError(
                "Stage 1 adapter token-grid metadata mismatch: "
                f"expected {normalize_grids(expected_token_hw)!r}, got {saved_grids!r}."
            )
    if expected_token_start_idx is not None:
        saved_start = training_panovggt.get("token_start_idx")
        normalized_expected_start = [None if value is None else int(value) for value in expected_token_start_idx]
        normalized_saved_start = None if saved_start is None else [None if value is None else int(value) for value in saved_start]
        if normalized_saved_start != normalized_expected_start:
            raise ValueError(
                "Stage 1 adapter token-start metadata mismatch: "
                f"expected {normalized_expected_start!r}, got {saved_start!r}."
            )
    saved_pose = str(training_panovggt.get("pose_convention", "c2w"))
    saved_depth = str(training_panovggt.get("depth_convention", "euclidean_ray_depth"))
    if expected_pose_convention is not None and saved_pose != str(expected_pose_convention):
        raise ValueError(
            f"Stage 1 adapter pose convention mismatch: expected {expected_pose_convention!r}, got {saved_pose!r}."
        )
    if expected_depth_convention is not None and saved_depth != str(expected_depth_convention):
        raise ValueError(
            "Stage 1 adapter depth convention mismatch: "
            f"expected {expected_depth_convention!r}, got {saved_depth!r}."
        )
    adapter = SphericalSelfiDPTAdapter(
        [int(value) for value in config["in_channels"]],
        hidden_dim=int(config["hidden_dim"]),
        out_dim=int(config["out_dim"]),
        image_height=image_size[0],
        image_width=image_size[1],
        use_circular_padding=bool(config.get("use_circular_padding", True)),
        norm_output=bool(config.get("norm_output", True)),
        reassemble_sizes=config.get("reassemble_sizes"),
        fusion_output_size=config.get("fusion_output_size"),
    )
    adapter.load_state_dict(state, strict=True)
    adapter.to(device)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "format": payload["format"],
        "adapter_config": dict(config),
        "training_config": training_config,
        "stage_hooks": training_panovggt.get("stage_hooks"),
        "token_hw": training_panovggt.get("token_hw"),
        "pose_convention": saved_pose,
        "depth_convention": saved_depth,
        "global_step": int(payload.get("global_step", 0)),
        "metrics": dict(payload.get("metrics", {})),
        "best_val_angular_error": payload.get("best_val_angular_error"),
    }
    return LoadedSphericalSelfiAdapter(
        module=adapter,
        checkpoint_path=str(path),
        sha256=digest,
        metadata=metadata,
    )
