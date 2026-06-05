"""Inference adapter for trained PanoVGGT-M3 matching and sky heads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .matching_head import PanoVGGTMatchingSkyHead, _select_feature


_MATCHING_FORMATS = {
    "panovggt_m3_sphere_matching_head_v1",
    "panovggt_m3_sphere_matching_sky_bundle_v1",
}
_SKY_FORMATS = {
    "panovggt_m3_sphere_sky_head_v1",
    "panovggt_m3_sphere_matching_sky_bundle_v1",
}


@dataclass(frozen=True)
class MatchingSkyCheckpointInfo:
    """Metadata for a loaded matching/sky checkpoint bundle."""

    feature_hook: str | None
    descriptor_dim: int
    head_config: dict[str, Any]
    class_map: dict[str, Any]
    has_matching: bool
    has_sky: bool


class MatchingSkyAdapter(nn.Module):
    """Frozen inference wrapper around ``PanoVGGTMatchingSkyHead``."""

    def __init__(
        self,
        head: PanoVGGTMatchingSkyHead,
        *,
        info: MatchingSkyCheckpointInfo,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.head = head.to(device)
        self.info = info
        self.device = torch.device(device)
        self.head.eval()
        for param in self.head.parameters():
            param.requires_grad_(False)

    @property
    def feature_hook(self) -> str | None:
        return self.info.feature_hook

    @property
    def has_matching(self) -> bool:
        return self.info.has_matching

    @property
    def has_sky(self) -> bool:
        return self.info.has_sky

    @property
    def descriptor_dim(self) -> int:
        return self.info.descriptor_dim

    def forward(self, feature: torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor]) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        return run_matching_sky_head(self, feature)


def _load_payload(path: str | Path, *, device: torch.device | str) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"PanoVGGT-M3 head checkpoint must be a mapping, got {type(payload)!r}.")
    return payload


def _state_dict_from_payload(payload: dict[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, nn.Module):
        return value.state_dict()
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field {key!r} must be a state_dict mapping.")
    return {str(k).removeprefix("module."): v for k, v in value.items()}


def _clean_matching_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if not key.startswith("static_confidence_proj.")}


def _has_matching_payload(payload: dict[str, Any]) -> bool:
    fmt = payload.get("format")
    return "matching_head" in payload or "wrapper" in payload or fmt in _MATCHING_FORMATS


def _has_sky_payload(payload: dict[str, Any]) -> bool:
    fmt = payload.get("format")
    return "sky_mask_head" in payload or "wrapper" in payload or fmt in _SKY_FORMATS


def _first_present(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _infer_head_config_from_state(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    states: list[dict[str, torch.Tensor]] = []
    for payload in payloads:
        for key in ("matching_head", "sky_mask_head", "wrapper"):
            state = _state_dict_from_payload(payload, key)
            if state is not None:
                states.append(state)

    feature_dim = None
    hidden_dim = None
    num_conv_blocks = 0
    descriptor_dim = None
    for state in states:
        for key, value in state.items():
            short = key.removeprefix("matching_head.").removeprefix("sky_head.")
            if short.endswith("descriptor_proj.weight") and value.ndim >= 2:
                descriptor_dim = int(value.shape[0])
            if ".trunk." in f".{short}" and short.endswith("block.0.weight") and value.ndim == 4:
                feature_dim = int(value.shape[1]) if feature_dim is None else feature_dim
                hidden_dim = int(value.shape[0]) if hidden_dim is None else hidden_dim
                parts = short.split(".")
                if "trunk" in parts:
                    idx = parts.index("trunk")
                    if idx + 1 < len(parts):
                        try:
                            num_conv_blocks = max(num_conv_blocks, int(parts[idx + 1]) + 1)
                        except ValueError:
                            pass
    out: dict[str, Any] = {}
    if feature_dim is not None:
        out["feature_dim"] = feature_dim
    if hidden_dim is not None:
        out["hidden_dim"] = hidden_dim
    if num_conv_blocks > 0:
        out["num_conv_blocks"] = num_conv_blocks
    if descriptor_dim is not None:
        out["descriptor_dim"] = descriptor_dim
    return out


def build_matching_sky_head_from_checkpoint(
    payloads: list[dict[str, Any]] | dict[str, Any],
    *,
    descriptor_dim: int | None = None,
    feature_key: str | int | None = None,
    strict: bool = True,
) -> tuple[PanoVGGTMatchingSkyHead, MatchingSkyCheckpointInfo]:
    """Build and load a ``PanoVGGTMatchingSkyHead`` from checkpoint payloads."""

    if isinstance(payloads, dict):
        payload_list = [payloads]
    else:
        payload_list = payloads
    if not payload_list:
        raise ValueError("At least one PanoVGGT-M3 head checkpoint payload is required.")

    head_config = dict(_first_present(payload_list, "head_config") or {})
    inferred = _infer_head_config_from_state(payload_list)
    for key, value in inferred.items():
        head_config.setdefault(key, value)

    has_matching = any(_has_matching_payload(payload) for payload in payload_list)
    has_sky = any(_has_sky_payload(payload) for payload in payload_list)
    if not has_matching and not has_sky:
        raise ValueError("Checkpoint does not contain a matching_head, sky_mask_head, or wrapper state.")

    payload_descriptor = _first_present(payload_list, "descriptor_dim")
    requested_descriptor = int(descriptor_dim or payload_descriptor or head_config.get("descriptor_dim", 24))
    if strict and payload_descriptor is not None and int(payload_descriptor) != requested_descriptor:
        raise ValueError(
            "PanoVGGT-M3 descriptor_dim mismatch: "
            f"checkpoint={int(payload_descriptor)}, requested={requested_descriptor}."
        )
    if strict and "descriptor_dim" in head_config and int(head_config["descriptor_dim"]) != requested_descriptor:
        raise ValueError(
            "PanoVGGT-M3 head_config descriptor_dim mismatch: "
            f"head_config={int(head_config['descriptor_dim'])}, requested={requested_descriptor}."
        )

    if "feature_dim" not in head_config:
        raise ValueError("PanoVGGT-M3 checkpoint is missing head_config.feature_dim and it could not be inferred.")

    resolved_feature_key = feature_key if feature_key is not None else head_config.get("feature_key")
    wrapper = PanoVGGTMatchingSkyHead(
        int(head_config["feature_dim"]),
        descriptor_dim=requested_descriptor,
        hidden_dim=int(head_config.get("hidden_dim", 128)),
        num_conv_blocks=int(head_config.get("num_conv_blocks", 2)),
        feature_key=resolved_feature_key,
        train_matching=has_matching,
        train_sky=has_sky,
    )

    for payload in payload_list:
        wrapper_state = _state_dict_from_payload(payload, "wrapper")
        if wrapper_state is not None:
            wrapper.load_state_dict(wrapper_state, strict=strict)
        matching_state = _state_dict_from_payload(payload, "matching_head")
        if matching_state is not None:
            wrapper.matching_head.load_state_dict(_clean_matching_state(matching_state), strict=strict)
        sky_state = _state_dict_from_payload(payload, "sky_mask_head")
        if sky_state is not None:
            wrapper.sky_head.load_state_dict(sky_state, strict=strict)

    class_map = dict(_first_present(payload_list, "class_map") or {})
    feature_hook = _first_present(payload_list, "feature_hook")
    info = MatchingSkyCheckpointInfo(
        feature_hook=str(feature_hook) if feature_hook is not None else None,
        descriptor_dim=requested_descriptor,
        head_config={
            **head_config,
            "descriptor_dim": requested_descriptor,
            "feature_key": resolved_feature_key,
            "train_matching": has_matching,
            "train_sky": has_sky,
        },
        class_map=class_map,
        has_matching=has_matching,
        has_sky=has_sky,
    )
    return wrapper, info


def load_matching_sky_checkpoint(
    checkpoint: str | Path | None = None,
    *,
    matching_checkpoint: str | Path | None = None,
    sky_checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
    descriptor_dim: int | None = None,
    feature_hook: str | None = None,
    feature_key: str | int | None = None,
    strict: bool = True,
) -> MatchingSkyAdapter:
    """Load one combined or two separate PanoVGGT-M3 head checkpoints."""

    paths = [path for path in (checkpoint, matching_checkpoint, sky_checkpoint) if path]
    if not paths:
        raise ValueError("MatchingHead.enabled=true requires checkpoint, matching_checkpoint, or sky_checkpoint.")
    payloads = [_load_payload(path, device=device) for path in paths]
    head, info = build_matching_sky_head_from_checkpoint(
        payloads,
        descriptor_dim=descriptor_dim,
        feature_key=feature_key,
        strict=strict,
    )
    if feature_hook is not None:
        info = MatchingSkyCheckpointInfo(
            feature_hook=str(feature_hook),
            descriptor_dim=info.descriptor_dim,
            head_config=info.head_config,
            class_map=info.class_map,
            has_matching=info.has_matching,
            has_sky=info.has_sky,
        )
    return MatchingSkyAdapter(head, info=info, device=device)


def _infer_input_hw(inputs: tuple[Any, ...]) -> tuple[int, int] | None:
    for value in inputs:
        if torch.is_tensor(value) and value.ndim >= 4:
            return int(value.shape[-2]), int(value.shape[-1])
    return None


def _tensor_from_feature_container(feature: Any, *, feature_key: str | int | None = None) -> torch.Tensor:
    if isinstance(feature, dict):
        return _select_feature(feature, feature_key)
    if isinstance(feature, (list, tuple)):
        tensors = [item for item in feature if torch.is_tensor(item)]
        if not tensors:
            raise TypeError("Feature container returned no tensors.")
        return tensors[-1]
    if not torch.is_tensor(feature):
        raise TypeError(f"Feature hook returned unsupported output type {type(feature)!r}.")
    return feature


def normalize_pano_feature(
    feature: Any,
    *,
    input_hw: tuple[int, int] | None = None,
    patch_size: int = 14,
    feature_key: str | int | None = None,
) -> torch.Tensor:
    """Normalize captured PanoVGGT features to ``B x N x C x Hf x Wf`` or ``N x C x Hf x Wf``."""

    patch_start_idx = 0
    raw = feature
    if isinstance(raw, tuple) and len(raw) >= 2 and isinstance(raw[1], int):
        patch_start_idx = int(raw[1])
        raw = raw[0]
    tensor = _tensor_from_feature_container(raw, feature_key=feature_key)

    if tensor.ndim == 5:
        if tensor.shape[2] > tensor.shape[-1]:
            return tensor.contiguous()
        return tensor.permute(0, 1, 4, 2, 3).contiguous()
    if tensor.ndim == 4 and tensor.shape[1] > tensor.shape[-1]:
        return tensor.contiguous()
    if tensor.ndim == 4 and input_hw is not None:
        tokens = tensor
        if int(patch_start_idx) > 0:
            tokens = tokens[:, :, int(patch_start_idx) :, :]
        h_in, w_in = input_hw
        patch = max(1, int(patch_size))
        height_f = max(1, h_in // patch)
        width_f = max(1, w_in // patch)
        expected = height_f * width_f
        if int(tokens.shape[2]) != expected:
            side = int(round(math.sqrt(float(tokens.shape[2]))))
            if side * side == int(tokens.shape[2]):
                height_f, width_f = side, side
            else:
                raise ValueError(
                    "Captured PanoVGGT token count cannot be reshaped to a feature grid: "
                    f"tokens={int(tokens.shape[2])}, input_hw={input_hw}, patch_size={patch}."
                )
        return tokens.reshape(tokens.shape[0], tokens.shape[1], height_f, width_f, tokens.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
    if tensor.ndim == 4:
        return tensor.contiguous()
    raise ValueError(f"Captured PanoVGGT feature must be a 4D or 5D tensor, got {tuple(tensor.shape)}.")


def extract_features_with_hook(
    model: nn.Module,
    feature_hook: str,
    call_model: Callable[[], Any],
    *,
    patch_size: int = 14,
    feature_key: str | int | None = None,
) -> tuple[Any, torch.Tensor]:
    """Run ``call_model`` while capturing a named module output feature."""

    modules = dict(model.named_modules())
    if not feature_hook:
        raise ValueError("PanoVGGT.MatchingHead.feature_hook is required for feature-hook capture.")
    if str(feature_hook) not in modules:
        raise ValueError(f"PanoVGGT.MatchingHead.feature_hook={feature_hook!r} was not found in external PanoVGGT modules.")

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        captured["feature"] = normalize_pano_feature(
            output,
            input_hw=_infer_input_hw(inputs),
            patch_size=patch_size,
            feature_key=feature_key,
        )

    handle = modules[str(feature_hook)].register_forward_hook(hook_fn)
    try:
        output = call_model()
    finally:
        handle.remove()
    if "feature" not in captured:
        raise RuntimeError(f"PanoVGGT feature hook {feature_hook!r} did not capture any tensor.")
    return output, captured["feature"]


def call_forward_with_features(model: nn.Module, model_input: torch.Tensor, *, patch_size: int = 14) -> tuple[Any, torch.Tensor] | None:
    """Try explicit PanoVGGT methods that return both geometry output and features."""

    for name in ("infer_with_features", "inference_with_features", "predict_with_features", "forward_with_features"):
        method = getattr(model, name, None)
        if method is None:
            continue
        try:
            result = method(model_input)
        except TypeError:
            continue
        feature = None
        output = result
        if isinstance(result, tuple) and len(result) >= 2:
            output, feature = result[0], result[1]
        elif isinstance(result, dict):
            for key in ("features", "feature", "tokens"):
                if key in result and result[key] is not None:
                    feature = result[key]
                    break
        else:
            for key in ("features", "feature", "tokens"):
                if hasattr(result, key):
                    feature = getattr(result, key)
                    break
        if feature is None:
            raise RuntimeError(f"External PanoVGGT method {name} did not return features.")
        return output, normalize_pano_feature(feature, input_hw=tuple(int(v) for v in model_input.shape[-2:]), patch_size=patch_size)
    return None


def _drop_single_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 5:
        if int(value.shape[0]) != 1:
            raise ValueError(f"{name} must have B=1 during chunk inference, got {tuple(value.shape)}.")
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape NxCxHxW after batch normalization, got {tuple(value.shape)}.")
    return value.float()


def run_matching_sky_head(
    adapter: MatchingSkyAdapter,
    feature: torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor],
) -> dict[str, torch.Tensor | tuple[int, int] | int]:
    """Run a loaded matching/sky adapter and validate its output contract."""

    selected = _select_feature(feature, adapter.head.feature_key)
    if selected.ndim not in (4, 5):
        raise ValueError(f"Matching head feature must be 4D or 5D, got {tuple(selected.shape)}.")
    feature_hw = tuple(int(v) for v in selected.shape[-2:])
    with torch.no_grad():
        raw = adapter.head(feature)
    out: dict[str, torch.Tensor | tuple[int, int] | int] = {
        "feature_hw": feature_hw,
        "descriptor_dim": int(adapter.descriptor_dim),
    }
    if adapter.has_matching:
        if "dense_descriptors" not in raw or "match_confidence" not in raw:
            raise RuntimeError("Matching checkpoint did not produce dense_descriptors and match_confidence.")
        dense = _drop_single_batch(raw["dense_descriptors"], name="dense_descriptors")
        match_conf = _drop_single_batch(raw["match_confidence"], name="match_confidence").clamp(0.0, 1.0)
        if tuple(dense.shape[-2:]) != feature_hw or tuple(match_conf.shape[-2:]) != feature_hw:
            raise ValueError(
                "Matching head output spatial size must equal input feature size: "
                f"feature_hw={feature_hw}, dense={tuple(dense.shape[-2:])}, confidence={tuple(match_conf.shape[-2:])}."
            )
        if int(dense.shape[1]) != int(adapter.descriptor_dim):
            raise ValueError(f"Expected descriptor_dim={adapter.descriptor_dim}, got {int(dense.shape[1])}.")
        out["dense_descriptors"] = dense
        out["match_confidence"] = match_conf
    if adapter.has_sky:
        if "sky_logits" not in raw or "sky_prob" not in raw:
            raise RuntimeError("Sky checkpoint did not produce sky_logits and sky_prob.")
        sky_logits = _drop_single_batch(raw["sky_logits"], name="sky_logits")
        sky_prob = _drop_single_batch(raw["sky_prob"], name="sky_prob").clamp(0.0, 1.0)
        if tuple(sky_prob.shape[-2:]) != feature_hw:
            raise ValueError(
                "Sky head output spatial size must equal input feature size: "
                f"feature_hw={feature_hw}, sky={tuple(sky_prob.shape[-2:])}."
            )
        out["sky_logits"] = sky_logits
        out["sky_prob"] = sky_prob
    return out


def make_fake_matching_outputs(
    images: torch.Tensor,
    *,
    descriptor_dim: int = 24,
    feature_stride: int = 4,
) -> dict[str, torch.Tensor | tuple[int, int] | int]:
    """Build deterministic dense matching fields for explicit fake/synthetic tests."""

    if images.ndim != 4:
        raise ValueError(f"Fake matching images must have shape Nx3xHxW, got {tuple(images.shape)}.")
    n, _, height, width = images.shape
    stride = max(1, int(feature_stride))
    feature_hw = (max(1, int(math.ceil(height / float(stride)))), max(1, int(math.ceil(width / float(stride)))))
    small = F.interpolate(images.float(), size=feature_hw, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    device, dtype = small.device, small.dtype
    yy = torch.linspace(0.0, 1.0, steps=feature_hw[0], device=device, dtype=dtype).view(1, 1, feature_hw[0], 1).expand(n, 1, -1, feature_hw[1])
    xx = torch.linspace(0.0, 1.0, steps=feature_hw[1], device=device, dtype=dtype).view(1, 1, 1, feature_hw[1]).expand(n, 1, feature_hw[0], -1)
    luma = small.mean(dim=1, keepdim=True)
    base = torch.cat(
        [
            small,
            luma,
            xx,
            yy,
            torch.sin(2.0 * math.pi * xx),
            torch.cos(2.0 * math.pi * xx),
            torch.sin(math.pi * yy),
            torch.cos(math.pi * yy),
        ],
        dim=1,
    )
    repeats = int(math.ceil(float(descriptor_dim) / float(base.shape[1])))
    dense = base.repeat(1, repeats, 1, 1)[:, : int(descriptor_dim)]
    dense = F.normalize(dense, dim=1, eps=1.0e-6)

    sky_prior = (1.0 - yy).clamp(0.0, 1.0)
    blue_margin = (small[:, 2:3] - small[:, 0:1]).clamp_min(0.0)
    sky_prob = (0.10 + 0.70 * sky_prior + 0.20 * blue_margin).clamp(0.01, 0.99)
    sky_logits = torch.logit(sky_prob)
    match_confidence = (0.35 + 0.65 * luma).clamp(0.0, 1.0) * (1.0 - 0.5 * sky_prob)
    return {
        "dense_descriptors": dense,
        "match_confidence": match_confidence.clamp(0.0, 1.0),
        "sky_logits": sky_logits,
        "sky_prob": sky_prob,
        "feature_hw": feature_hw,
        "descriptor_dim": int(descriptor_dim),
    }
