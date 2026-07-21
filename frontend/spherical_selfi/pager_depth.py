"""Config-gated PaGeR scale-invariant depth for spherical-Selfi windows.

The production provider imports PaGeR lazily from an external checkout.  Unit
tests can inject ``infer_batch_fn`` and therefore never need the external
repository, checkpoint, or its optional dependencies.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch


class PaGeRDepthError(RuntimeError):
    """Base error for PaGeR inference and alignment failures."""


class PaGeRDepthAlignmentError(PaGeRDepthError):
    """Raised when a frame cannot be aligned to the PanoVGGT window scale."""


@dataclass(frozen=True)
class PaGeRDepthConfig:
    """Validated internal representation of ``SphericalSelfiRuntime.pager_depth``."""

    enabled: bool = False
    repo_path: str | None = None
    checkpoint: str = "prs-eth/PaGeR"
    output: str = "scale_invariant"
    micro_batch_size: int = 1
    amp_dtype: str = "float16"
    cache_size: int = 128
    alignment_mode: str = "per_frame_panovggt_scale"
    min_valid_pixels: int = 4096
    min_valid_ratio: float = 0.05
    min_scale: float = 1.0e-3
    max_scale: float = 1.0e3
    apply_head_depth_residual: bool = False
    failure_policy: str = "error"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PaGeRDepthConfig":
        cfg = dict(value or {})
        result = cls(
            enabled=bool(cfg.get("enabled", False)),
            repo_path=(
                None if cfg.get("repo_path") in {None, ""} else str(cfg["repo_path"])
            ),
            checkpoint=str(cfg.get("checkpoint", "prs-eth/PaGeR")),
            output=str(cfg.get("output", "scale_invariant")).strip().lower(),
            micro_batch_size=int(cfg.get("micro_batch_size", 1)),
            amp_dtype=str(cfg.get("amp_dtype", "float16")).strip().lower(),
            cache_size=int(cfg.get("cache_size", 128)),
            alignment_mode=str(
                cfg.get("alignment_mode", "per_frame_panovggt_scale")
            ).strip().lower(),
            min_valid_pixels=int(cfg.get("min_valid_pixels", 4096)),
            min_valid_ratio=float(cfg.get("min_valid_ratio", 0.05)),
            min_scale=float(cfg.get("min_scale", 1.0e-3)),
            max_scale=float(cfg.get("max_scale", 1.0e3)),
            apply_head_depth_residual=bool(
                cfg.get("apply_head_depth_residual", False)
            ),
            failure_policy=str(cfg.get("failure_policy", "error")).strip().lower(),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.output != "scale_invariant":
            raise ValueError("pager_depth.output must be 'scale_invariant'.")
        if self.alignment_mode != "per_frame_panovggt_scale":
            raise ValueError(
                "pager_depth.alignment_mode must be 'per_frame_panovggt_scale'."
            )
        if self.failure_policy != "error":
            raise ValueError("pager_depth.failure_policy currently supports only 'error'.")
        if self.apply_head_depth_residual:
            raise ValueError(
                "PaGeR full-replacement mode requires "
                "pager_depth.apply_head_depth_residual=false."
            )
        if self.micro_batch_size <= 0:
            raise ValueError("pager_depth.micro_batch_size must be positive.")
        if self.cache_size < 0:
            raise ValueError("pager_depth.cache_size must be non-negative.")
        if self.amp_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "pager_depth.amp_dtype must be 'float16', 'bfloat16', or 'float32'."
            )
        if self.min_valid_pixels <= 0:
            raise ValueError("pager_depth.min_valid_pixels must be positive.")
        if not 0.0 < self.min_valid_ratio <= 1.0:
            raise ValueError("pager_depth.min_valid_ratio must be in (0, 1].")
        if not math.isfinite(self.min_scale) or self.min_scale <= 0.0:
            raise ValueError("pager_depth.min_scale must be finite and positive.")
        if not math.isfinite(self.max_scale) or self.max_scale <= self.min_scale:
            raise ValueError("pager_depth.max_scale must exceed min_scale.")
        if self.enabled and not self.repo_path:
            raise ValueError("pager_depth.repo_path is required when PaGeR is enabled.")
        if self.enabled and not self.checkpoint:
            raise ValueError("pager_depth.checkpoint is required when PaGeR is enabled.")


def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("Weighted median inputs must be equal-length vectors.")
    if values.numel() == 0:
        raise PaGeRDepthAlignmentError("Weighted median received no valid samples.")
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = sorted_weights.sum()
    if not bool(torch.isfinite(total)) or float(total.detach().cpu()) <= 0.0:
        raise PaGeRDepthAlignmentError("Weighted median received invalid weights.")
    cutoff = total * 0.5
    index = torch.searchsorted(sorted_weights.cumsum(dim=0), cutoff).clamp_max(
        sorted_values.numel() - 1
    )
    return sorted_values[index]


def align_pager_depth_to_panovggt(
    pager_depth: torch.Tensor,
    panovggt_depth: torch.Tensor,
    *,
    sky_mask: torch.Tensor | None = None,
    frame_ids: torch.Tensor | None = None,
    min_valid_pixels: int = 4096,
    min_valid_ratio: float = 0.05,
    min_scale: float = 1.0e-3,
    max_scale: float = 1.0e3,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Align every PaGeR frame to its PanoVGGT scale with a log weighted median.

    Both inputs use Euclidean ray depth and must have shape ``BxSx1xHxW``.
    Latitude-area weights prevent the oversampled ERP poles from dominating the
    robust scale fit.  No shift, mixing, or temporal smoothing is applied.
    """

    if pager_depth.ndim != 5 or pager_depth.shape[2] != 1:
        raise ValueError("pager_depth must have shape BxSx1xHxW.")
    if tuple(panovggt_depth.shape) != tuple(pager_depth.shape):
        raise ValueError("PaGeR and PanoVGGT depth tensors must have equal shapes.")
    if sky_mask is not None and tuple(sky_mask.shape) != tuple(pager_depth.shape):
        raise ValueError("sky_mask must match the depth tensor shape.")
    if min_valid_pixels <= 0 or not 0.0 < float(min_valid_ratio) <= 1.0:
        raise ValueError("Invalid PaGeR alignment support thresholds.")
    if min_scale <= 0.0 or max_scale <= min_scale:
        raise ValueError("Invalid PaGeR alignment scale bounds.")

    batch, views, _, height, width = (int(value) for value in pager_depth.shape)
    if frame_ids is not None and tuple(frame_ids.shape) != (batch, views):
        raise ValueError("frame_ids must have shape BxS when provided.")
    target = panovggt_depth.to(device=pager_depth.device, dtype=torch.float32)
    source = pager_depth.float()
    sky = None if sky_mask is None else sky_mask.to(device=source.device).bool()
    latitude = (
        (torch.arange(height, device=source.device, dtype=torch.float32) + 0.5)
        / float(height)
        * math.pi
        - 0.5 * math.pi
    )
    row_weights = latitude.cos().clamp_min(1.0e-6).view(height, 1)
    area_weights = row_weights.expand(height, width)
    aligned = torch.empty_like(source)
    diagnostics: list[dict[str, Any]] = []
    total_pixels = height * width

    for batch_index in range(batch):
        for view_index in range(views):
            source_frame = source[batch_index, view_index, 0]
            target_frame = target[batch_index, view_index, 0]
            valid = (
                torch.isfinite(source_frame)
                & torch.isfinite(target_frame)
                & (source_frame > 0.0)
                & (target_frame > 0.0)
            )
            if sky is not None:
                valid &= ~sky[batch_index, view_index, 0]
            valid_pixels = int(valid.sum().detach().cpu())
            valid_ratio = valid_pixels / float(max(1, total_pixels))
            frame_id = (
                int(frame_ids[batch_index, view_index].detach().cpu())
                if frame_ids is not None
                else view_index
            )
            if valid_pixels < int(min_valid_pixels) or valid_ratio < float(
                min_valid_ratio
            ):
                raise PaGeRDepthAlignmentError(
                    f"PaGeR frame {frame_id} has insufficient alignment support: "
                    f"{valid_pixels} pixels ({valid_ratio:.6f})."
                )

            log_ratio = target_frame[valid].log() - source_frame[valid].log()
            weights = area_weights[valid]
            log_scale = _weighted_median(log_ratio, weights)
            scale = log_scale.exp()
            scale_value = float(scale.detach().cpu())
            if (
                not math.isfinite(scale_value)
                or scale_value < float(min_scale)
                or scale_value > float(max_scale)
            ):
                raise PaGeRDepthAlignmentError(
                    f"PaGeR frame {frame_id} produced invalid scale {scale_value}."
                )
            absolute_log_error = (log_ratio - log_scale).abs()
            log_mad = _weighted_median(absolute_log_error, weights)
            aligned[batch_index, view_index, 0] = source_frame * scale
            diagnostics.append(
                {
                    "batch_index": batch_index,
                    "view_index": view_index,
                    "frame_id": frame_id,
                    "scale": scale_value,
                    "log_mad": float(log_mad.detach().cpu()),
                    "valid_pixels": valid_pixels,
                    "valid_ratio": valid_ratio,
                }
            )
    return aligned.to(dtype=pager_depth.dtype), diagnostics


class PaGeRDepthProvider:
    """Online PaGeR SI-depth provider with a CPU float32 frame LRU."""

    def __init__(
        self,
        config: PaGeRDepthConfig,
        *,
        device: torch.device,
        infer_batch_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("PaGeRDepthProvider requires an enabled configuration.")
        self.config = config
        self.device = torch.device(device)
        self._infer_batch_fn = infer_batch_fn
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._pager = None
        self._erp_to_cubemap = None
        self._skip_heads: set[str] | None = None
        self._face_size = 504
        self._cube_fov = 90.0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def reset(self) -> None:
        self._cache.clear()

    def _dtype(self) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[self.config.amp_dtype]

    def _load_runtime(self) -> None:
        if self._pager is not None:
            return
        repo = Path(str(self.config.repo_path)).expanduser().resolve()
        if not repo.is_dir():
            raise PaGeRDepthError(f"PaGeR repository does not exist: {repo}")
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        try:
            pager_module = importlib.import_module("src.pager")
            geometry_module = importlib.import_module("src.utils.geometry_utils")
            omega_module = importlib.import_module("omegaconf")
            hub_module = importlib.import_module("huggingface_hub")
        except Exception as exc:
            raise PaGeRDepthError(
                "Failed to import PaGeR. Install its editable package and runtime "
                f"dependencies from {repo}."
            ) from exc

        checkpoint = self.config.checkpoint
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.is_dir():
            model_config_path = checkpoint_path / "config.yaml"
            model_checkpoint: str | Path = checkpoint_path
        else:
            try:
                model_config_path = Path(
                    hub_module.hf_hub_download(
                        repo_id=checkpoint,
                        filename="config.yaml",
                    )
                )
                model_weights_path = Path(
                    hub_module.hf_hub_download(
                        repo_id=checkpoint,
                        filename="model.safetensors",
                    )
                )
            except Exception as exc:
                raise PaGeRDepthError(
                    f"Could not resolve PaGeR checkpoint {checkpoint!r}."
                ) from exc
            if model_config_path.parent != model_weights_path.parent:
                raise PaGeRDepthError(
                    "PaGeR config and weights resolved to different Hub snapshots."
                )
            model_checkpoint = model_weights_path.parent
        if not model_config_path.is_file():
            raise PaGeRDepthError(
                f"PaGeR checkpoint config does not exist: {model_config_path}"
            )
        model_cfg = omega_module.OmegaConf.load(str(model_config_path))
        modalities = {str(value) for value in model_cfg.modalities}
        if "depth" not in modalities:
            raise PaGeRDepthError(
                f"PaGeR checkpoint {checkpoint!r} does not contain a depth head."
            )
        self._face_size = int(getattr(model_cfg, "face_size", 504))
        self._cube_fov = float(getattr(model_cfg, "cube_fov", 90.0))
        # PaGeR explicitly disables autocast around its camera encoder.  Keep
        # parameters in the official FP32 storage dtype so that the FP32
        # extrinsic/intrinsic pose encoding and that encoder's linear layers
        # remain dtype-compatible.  ``amp_dtype`` controls the surrounding
        # image backbone/head autocast and therefore still provides mixed-
        # precision inference without mutating the released weights.
        weight_dtype = torch.float32
        try:
            pager = pager_module.Pager(
                model_checkpoint,
                cfg=model_cfg,
                device=self.device,
                weight_dtype=weight_dtype,
            )
            pager.model.to(self.device, dtype=weight_dtype)
            pager.get_intrinsics_extrinsics(
                image_size=self._face_size,
                fov=self._cube_fov,
            )
            pager.eval()
        except Exception as exc:
            raise PaGeRDepthError(
                f"Failed to initialize PaGeR checkpoint {checkpoint!r}."
            ) from exc
        self._pager = pager
        self._erp_to_cubemap = geometry_module.erp_to_cubemap
        self._skip_heads = modalities - {"depth"}

    @torch.no_grad()
    def _infer_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self._infer_batch_fn is not None:
            output = self._infer_batch_fn(images)
            if not torch.is_tensor(output):
                raise TypeError("Injected PaGeR inference must return a tensor.")
            return output
        self._load_runtime()
        assert self._pager is not None
        assert self._erp_to_cubemap is not None
        normalized_images = images.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
        ).view(3, 1, 1)
        cubemaps = torch.stack(
            [
                self._erp_to_cubemap(
                    (image - mean) / std,
                    face_w=self._face_size,
                    fov=self._cube_fov,
                )
                for image in normalized_images
            ],
            dim=0,
        )
        predictions = self._pager(
            cubemaps,
            dtype=self._dtype(),
            skip_heads=self._skip_heads,
        )
        if "depth" not in predictions:
            raise PaGeRDepthError("PaGeR inference did not return the depth head.")
        output_frames: list[torch.Tensor] = []
        output_size = (int(images.shape[-2]), int(images.shape[-1]))
        for index in range(int(images.shape[0])):
            processed, _ = self._pager.process_depth_output(
                predictions["depth"][index],
                output_size,
                sky_mask=None,
                log_scale=None,
            )
            value = processed.float()
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim != 3 or value.shape[0] != 1:
                raise PaGeRDepthError(
                    "PaGeR processed depth must have shape 1xHxW; got "
                    f"{tuple(value.shape)}."
                )
            output_frames.append(value)
        return torch.stack(output_frames, dim=0)

    def _store(self, frame_id: int, depth: torch.Tensor) -> None:
        if self.config.cache_size == 0:
            return
        self._cache[frame_id] = depth.detach().cpu().float().clone()
        self._cache.move_to_end(frame_id)
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return raw SI ray depth matching ``images`` spatial resolution."""

        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("PaGeR images must have shape BxSx3xHxW.")
        batch, views, _, height, width = (int(value) for value in images.shape)
        if tuple(frame_ids.shape) != (batch, views):
            raise ValueError("PaGeR frame_ids must have shape BxS.")
        flat_images = images.reshape(batch * views, 3, height, width)
        ids = [int(value) for value in frame_ids.detach().cpu().reshape(-1).tolist()]
        results: list[torch.Tensor | None] = [None] * len(ids)
        misses: list[int] = []
        cache_hits = 0
        for index, frame_id in enumerate(ids):
            cached = self._cache.get(frame_id)
            if cached is not None and tuple(cached.shape) == (1, height, width):
                results[index] = cached
                self._cache.move_to_end(frame_id)
                cache_hits += 1
            else:
                if cached is not None:
                    del self._cache[frame_id]
                misses.append(index)

        inference_start = time.perf_counter()
        for start in range(0, len(misses), self.config.micro_batch_size):
            indices = misses[start : start + self.config.micro_batch_size]
            prediction = self._infer_batch(flat_images[indices])
            if tuple(prediction.shape) != (len(indices), 1, height, width):
                raise PaGeRDepthError(
                    "PaGeR inference must return Nx1xHxW matching its input; got "
                    f"{tuple(prediction.shape)}."
                )
            prediction = prediction.detach().float()
            for local_index, flat_index in enumerate(indices):
                value = prediction[local_index]
                if not bool(torch.isfinite(value).any()) or not bool((value > 0.0).any()):
                    raise PaGeRDepthError(
                        f"PaGeR produced no finite positive depth for frame {ids[flat_index]}."
                    )
                cpu_value = value.cpu().clone()
                results[flat_index] = cpu_value
                self._store(ids[flat_index], cpu_value)
        inference_sec = float(time.perf_counter() - inference_start) if misses else 0.0
        if any(value is None for value in results):
            raise PaGeRDepthError("PaGeR failed to populate all requested frames.")
        stacked = torch.stack([value for value in results if value is not None], dim=0)
        stacked = stacked.reshape(batch, views, 1, height, width).to(self.device)
        cache_misses = len(misses)
        requested = len(ids)
        return stacked, {
            "inference_sec": inference_sec,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_ratio": cache_hits / float(max(1, requested)),
            "cache_entries": len(self._cache),
        }
