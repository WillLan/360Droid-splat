"""Frozen PanoVGGT feature wrapper for Stage 1B.

This module is intentionally standalone. It does not integrate with the SLAM
frontend and does not alter the existing PanoVGGT inference path.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib import import_module
import inspect
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


def _import_attr(path: str) -> Any:
    module_name, attr = str(path).rsplit(".", 1)
    return getattr(import_module(module_name), attr)


@contextmanager
def _maybe_skip_dinov2_pretrain(enabled: bool):
    """Temporarily disable the external PanoVGGT DINOv2 preload hook."""

    if not enabled:
        yield
        return
    try:
        aggregator_mod = import_module("panovggt.models.aggregator")
        aggregator_cls = getattr(aggregator_mod, "Aggregator", None)
        original = getattr(aggregator_cls, "_try_load_dinov2", None) if aggregator_cls is not None else None
        if aggregator_cls is None or original is None:
            yield
            return

        def _skip(self, hub_name, url, patch_embed_key):
            return None

        aggregator_cls._try_load_dinov2 = _skip
        try:
            yield
        finally:
            aggregator_cls._try_load_dinov2 = original
    except ModuleNotFoundError:
        yield


def _instantiate_external_panovggt(cls: type, config: dict[str, Any]) -> nn.Module:
    kwargs = dict(config.get("model_kwargs", {}))
    config_path = config.get("config_path")
    if config_path is not None and cls.__name__ == "PanoVGGTModel":
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - external dependency
            raise ImportError("PanoVGGT config_path loading requires omegaconf.") from exc
        official = OmegaConf.load(str(config_path))
        OmegaConf.resolve(official)
        model_cfg = official.model
        kwargs = {
            "img_size": int(official.img_size),
            "patch_size": int(official.patch_size),
            "embed_dim": int(official.embed_dim),
            "enable_camera": bool(model_cfg.enable_camera),
            "enable_depth": bool(model_cfg.enable_depth),
            "enable_point": bool(model_cfg.enable_point),
            "aggregator": OmegaConf.to_container(model_cfg.aggregator, resolve=True),
            **kwargs,
        }
    signature = inspect.signature(cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return cls(**kwargs)
    return cls(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def build_external_panovggt_model(config: dict[str, Any]) -> nn.Module:
    """Build and load a real external PanoVGGT model from a config mapping."""

    repo_path = config.get("repo_path")
    if repo_path:
        repo = str(Path(repo_path).expanduser().resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    class_path = config.get("class_path")
    if not class_path:
        raise ValueError("panovggt.class_path is required for a real PanoVGGT model.")
    cls = _import_attr(str(class_path))
    with _maybe_skip_dinov2_pretrain(bool(config.get("skip_dinov2_pretrain", False))):
        model = _instantiate_external_panovggt(cls, config)
    checkpoint = config.get("checkpoint")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if isinstance(payload, dict) and key in payload:
                state = payload[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported PanoVGGT checkpoint payload: {checkpoint}")
        model.load_state_dict(
            {str(key).removeprefix("module."): value for key, value in state.items()},
            strict=bool(config.get("strict_checkpoint", False)),
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_frozen_panovggt_wrapper(
    config: dict[str, Any],
    *,
    device: torch.device | str,
    model: nn.Module | None = None,
) -> "PanoVGGTFeatureWrapper":
    """Build the reusable frozen four-stage feature wrapper used by Stages 1/2."""

    if model is None:
        model = build_external_panovggt_model(config)
    wrapper = PanoVGGTFeatureWrapper(
        model,
        stage_hooks=list(config.get("stage_hooks", [])),
        feature_keys=list(config.get("feature_keys", [None, None, None, None])),
        token_hw=list(config.get("token_hw", [None, None, None, None])),
        token_start_idx=list(config.get("token_start_idx", [None, None, None, None])),
        use_no_grad=bool(config.get("use_no_grad", True)),
        pose_convention=str(config.get("pose_convention", "c2w")),
        depth_convention=str(config.get("depth_convention", "euclidean_ray_depth")),
    )
    return wrapper.to(device)


@dataclass
class PanoVGGTFeatureOutput:
    """Frozen PanoVGGT geometry output plus captured 4-stage features."""

    images: torch.Tensor
    init_depth: torch.Tensor | None
    init_poses: torch.Tensor | None
    stage_features: list[torch.Tensor]
    optional_world_points: torch.Tensor | None = None
    feature_shapes: list[tuple[int, ...]] = field(default_factory=list)
    hook_names: list[str] = field(default_factory=list)
    pose_convention: str = "c2w"
    depth_convention: str = "euclidean_ray_depth"
    feature_convention: str = "B,V,C,H,W"


def _as_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if hasattr(output, "_asdict"):
        return dict(output._asdict())
    keys = (
        "depth",
        "depths",
        "init_depth",
        "poses",
        "camera_poses",
        "init_poses",
        "poses_c2w",
        "world_points",
        "point_maps",
        "points",
        "global_points",
    )
    out = {key: getattr(output, key) for key in keys if hasattr(output, key)}
    if out:
        return out
    if isinstance(output, (tuple, list)):
        names = ("camera_poses", "depth", "world_points")
        return {name: value for name, value in zip(names, output)}
    return {}


def _first_present(output: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = output.get(name)
        if value is not None:
            return value
    return None


def _select_from_container(value: Any, key: str | int | None) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        if key is None:
            if len(value) != 1:
                raise ValueError("feature_key is required for hook outputs with multiple dict entries.")
            return next(iter(value.values()))
        str_key = str(key)
        if str_key not in value:
            raise KeyError(f"Hook output dict does not contain key {str_key!r}.")
        return value[str_key]
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        if key is None:
            if tensors:
                return tensors[-1]
            raise TypeError("Hook output sequence did not contain any tensors.")
        if isinstance(key, int) or str(key).lstrip("-").isdigit():
            return value[int(key)]
        raise TypeError("Sequence hook outputs require an integer feature_key.")
    raise TypeError(f"Unsupported hook output type: {type(value)!r}.")


def _factor_grid(token_count: int, image_hw: tuple[int, int] | None) -> tuple[int, int]:
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if image_hw is None:
        side = int(round(math.sqrt(float(token_count))))
        if side * side == token_count:
            return side, side
        return 1, token_count
    target_ratio = float(image_hw[1]) / max(float(image_hw[0]), 1.0)
    best: tuple[float, int, int] | None = None
    for h in range(1, int(math.sqrt(float(token_count))) + 1):
        if token_count % h != 0:
            continue
        w = token_count // h
        score = abs((float(w) / float(h)) - target_ratio)
        if best is None or score < best[0]:
            best = (score, h, w)
    if best is None:
        return 1, token_count
    return best[1], best[2]


def _tokens_to_grid(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    image_hw: tuple[int, int] | None,
    token_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    if tokens.ndim == 4:
        if int(tokens.shape[0]) == batch_size and int(tokens.shape[1]) == num_views:
            b, v, n, c = (int(dim) for dim in tokens.shape)
            flat = tokens
        elif int(tokens.shape[0]) == batch_size * num_views:
            n, c = int(tokens.shape[1]), int(tokens.shape[2])
            flat = tokens.reshape(batch_size, num_views, n, c)
        else:
            raise ValueError(f"Cannot interpret token tensor shape {tuple(tokens.shape)}.")
    elif tokens.ndim == 3 and int(tokens.shape[0]) == batch_size * num_views:
        n, c = int(tokens.shape[1]), int(tokens.shape[2])
        flat = tokens.reshape(batch_size, num_views, n, c)
    else:
        raise ValueError(f"Token features must be BxVxNxC or B*VxNxC, got {tuple(tokens.shape)}.")
    _, _, token_count, channels = (int(dim) for dim in flat.shape)
    if token_hw is None:
        height, width = _factor_grid(token_count, image_hw)
    else:
        height, width = int(token_hw[0]), int(token_hw[1])
        if height * width != token_count:
            raise ValueError(f"token_hw={token_hw!r} does not match token count {token_count}.")
    return flat.reshape(batch_size, num_views, height, width, channels).permute(0, 1, 4, 2, 3).contiguous()


def normalize_stage_feature(
    feature: Any,
    *,
    batch_size: int,
    num_views: int,
    image_hw: tuple[int, int] | None,
    feature_key: str | int | None = None,
    token_hw: tuple[int, int] | None = None,
    token_start_idx: int | None = None,
) -> torch.Tensor:
    """Normalize map or token features to ``B x V x C x H x W``."""

    raw = feature
    if isinstance(raw, tuple) and len(raw) >= 2 and isinstance(raw[1], int):
        start = int(raw[1])
        raw = raw[0]
    else:
        start = 0
    if token_start_idx is not None:
        start = int(token_start_idx)
    tensor = _select_from_container(raw, feature_key)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Selected feature must be a tensor, got {type(tensor)!r}.")
    if tensor.ndim == 5:
        if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) != num_views:
            raise ValueError(
                f"Expected 5D feature as BxVxCxHxW with B,V={(batch_size, num_views)}, got {tuple(tensor.shape)}."
            )
        return tensor.contiguous()
    if tensor.ndim == 4 and int(tensor.shape[0]) == batch_size and int(tensor.shape[1]) == num_views:
        tokens = tensor
        if start > 0:
            tokens = tokens[..., start:, :]
        return _tokens_to_grid(tokens, batch_size=batch_size, num_views=num_views, image_hw=image_hw, token_hw=token_hw)
    if tensor.ndim == 4 and int(tensor.shape[0]) == batch_size * num_views:
        return tensor.reshape(batch_size, num_views, int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])).contiguous()
    if tensor.ndim == 3:
        tokens = tensor
        if start > 0:
            tokens = tokens[..., start:, :]
        return _tokens_to_grid(tokens, batch_size=batch_size, num_views=num_views, image_hw=image_hw, token_hw=token_hw)
    raise ValueError(f"Unsupported feature tensor shape {tuple(tensor.shape)}.")


class PanoVGGTFeatureWrapper(nn.Module):
    """Capture four frozen PanoVGGT feature stages through forward hooks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        stage_hooks: list[str] | tuple[str, ...],
        feature_keys: list[str | int | None] | tuple[str | int | None, ...] | None = None,
        token_hw: list[tuple[int, int] | None] | tuple[tuple[int, int] | None, ...] | None = None,
        token_start_idx: list[int | None] | tuple[int | None, ...] | None = None,
        use_no_grad: bool = True,
        pose_convention: str = "c2w",
        depth_convention: str = "euclidean_ray_depth",
    ) -> None:
        super().__init__()
        if len(stage_hooks) != 4:
            raise ValueError(f"Stage 1B requires exactly 4 feature hooks, got {len(stage_hooks)}.")
        self.model = model
        self.stage_hooks = [str(name) for name in stage_hooks]
        self.feature_keys = list(feature_keys) if feature_keys is not None else [None] * 4
        self.token_hw = list(token_hw) if token_hw is not None else [None] * 4
        self.token_start_idx = list(token_start_idx) if token_start_idx is not None else [None] * 4
        if len(self.feature_keys) != 4 or len(self.token_hw) != 4 or len(self.token_start_idx) != 4:
            raise ValueError("feature_keys, token_hw, and token_start_idx must each contain 4 entries when provided.")
        self.use_no_grad = bool(use_no_grad)
        self.pose_convention = str(pose_convention)
        self.depth_convention = str(depth_convention)
        self.freeze_panovggt()

    def freeze_panovggt(self) -> None:
        """Put the wrapped model in eval mode and disable all parameter grads."""

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.model.eval()
        return self

    def _call_model(self, images: torch.Tensor) -> Any:
        for name in ("infer", "inference", "predict", "forward"):
            method = getattr(self.model, name, None)
            if method is None:
                continue
            try:
                return method(images)
            except TypeError:
                continue
        return self.model(images)

    def forward(self, images: torch.Tensor) -> PanoVGGTFeatureOutput:
        """Run frozen PanoVGGT and return captured 4-stage features."""

        if images.ndim == 4:
            model_images = images.unsqueeze(0)
        elif images.ndim == 5:
            model_images = images
        else:
            raise ValueError(f"images must have shape Vx3xHxW or BxVx3xHxW, got {tuple(images.shape)}.")
        batch_size, num_views = int(model_images.shape[0]), int(model_images.shape[1])
        image_hw = (int(model_images.shape[-2]), int(model_images.shape[-1]))
        modules = dict(self.model.named_modules())
        missing = [name for name in self.stage_hooks if name not in modules]
        if missing:
            raise ValueError(f"PanoVGGT feature hook(s) not found: {missing}.")

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(stage_index: int):
            def hook_fn(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                captured[stage_index] = normalize_stage_feature(
                    output,
                    batch_size=batch_size,
                    num_views=num_views,
                    image_hw=image_hw,
                    feature_key=self.feature_keys[stage_index],
                    token_hw=self.token_hw[stage_index],
                    token_start_idx=self.token_start_idx[stage_index],
                )

            return hook_fn

        for idx, name in enumerate(self.stage_hooks):
            handles.append(modules[name].register_forward_hook(make_hook(idx)))
        try:
            context = torch.no_grad() if self.use_no_grad else nullcontext()
            with context:
                output = self._call_model(model_images)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 4:
            missing_idx = [idx for idx in range(4) if idx not in captured]
            raise RuntimeError(f"PanoVGGT feature hooks did not capture all stages; missing indices {missing_idx}.")
        stage_features = [captured[idx] for idx in range(4)]
        out_dict = _as_dict(output)
        depth = _first_present(out_dict, ("init_depth", "depth", "depths"))
        poses = _first_present(out_dict, ("init_poses", "poses_c2w", "camera_poses", "poses"))
        world_points = _first_present(out_dict, ("world_points", "point_maps", "points", "global_points"))
        return PanoVGGTFeatureOutput(
            images=model_images,
            init_depth=None if depth is None else torch.as_tensor(depth, device=model_images.device),
            init_poses=None if poses is None else torch.as_tensor(poses, device=model_images.device),
            stage_features=stage_features,
            optional_world_points=None if world_points is None else torch.as_tensor(world_points, device=model_images.device),
            feature_shapes=[tuple(feature.shape) for feature in stage_features],
            hook_names=list(self.stage_hooks),
            pose_convention=self.pose_convention,
            depth_convention=self.depth_convention,
        )
