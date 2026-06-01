"""PanoVGGT inference engines.

The real engine dynamically loads an external PanoVGGT checkout. Tests and
smoke runs use the fake engine so this repository stays self contained.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import inspect
import sys
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

from .types import PanoVGGTLocalPrediction


def _resize_images(images: torch.Tensor, image_size: tuple[int, int] | None) -> torch.Tensor:
    if image_size is None:
        return images
    if tuple(images.shape[-2:]) == tuple(image_size):
        return images
    return F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)


def _import_attr(path: str) -> Any:
    module_name, attr = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr)


def _as_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if hasattr(output, "_asdict"):
        return dict(output._asdict())
    keys = (
        "camera_poses",
        "poses",
        "depth",
        "depths",
        "confidence",
        "depth_confidence",
        "world_points",
        "local_points",
        "point_maps",
        "points",
        "points3d",
        "descriptors",
        "tokens",
        "features",
    )
    out = {key: getattr(output, key) for key in keys if hasattr(output, key)}
    if out:
        return out
    if isinstance(output, (tuple, list)):
        names = ("camera_poses", "depth", "world_points", "confidence")
        return {name: value for name, value in zip(names, output)}
    raise TypeError(f"Unsupported PanoVGGT output type: {type(output)!r}")


def _first_present(output: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in output and output[name] is not None:
            return output[name]
    return None


def _drop_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = _drop_batch_dim(depth)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth as Nx1xHxW, got {tuple(depth.shape)}")
    return depth.float().clamp_min(1e-6)


def _normalize_poses(poses: torch.Tensor) -> torch.Tensor:
    poses = _drop_batch_dim(poses)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected poses as Nx4x4, got {tuple(poses.shape)}")
    return poses.float()


def _normalize_confidence(confidence: torch.Tensor | None, depth: torch.Tensor) -> torch.Tensor:
    if confidence is None:
        return torch.isfinite(depth).to(depth.dtype)
    confidence = _drop_batch_dim(confidence)
    if confidence.ndim == 3:
        confidence = confidence.unsqueeze(1)
    elif confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence.permute(0, 3, 1, 2)
    if confidence.shape[-2:] != depth.shape[-2:]:
        confidence = F.interpolate(confidence.float(), size=depth.shape[-2:], mode="bilinear", align_corners=False)
    if confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError(f"Expected confidence as Nx1xHxW, got {tuple(confidence.shape)}")
    return confidence.float().clamp(0.0, 1.0)


def _build_point_maps(depth: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    n, _, height, width = depth.shape
    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    bearing = erp_pixel_to_bearing(grid, height, width).to(device=depth.device, dtype=depth.dtype)
    pts_cam = bearing.unsqueeze(0) * depth[:, 0].unsqueeze(-1)
    rot = poses_c2w[:, :3, :3]
    trans = poses_c2w[:, :3, 3]
    pts = torch.einsum("nij,nhwj->nhwi", rot, pts_cam) + trans.view(n, 1, 1, 3)
    return pts


def _normalize_point_maps(points: torch.Tensor | None, depth: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    if points is None:
        return _build_point_maps(depth, poses_c2w)
    points = _drop_batch_dim(points)
    if points.ndim == 4 and points.shape[1] == 3:
        points = points.permute(0, 2, 3, 1)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point maps as NxHxWx3, got {tuple(points.shape)}")
    if points.shape[1:3] != depth.shape[-2:]:
        points = F.interpolate(
            points.permute(0, 3, 1, 2).float(),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return points.float()


def normalize_panovggt_output(output: Any, images: torch.Tensor) -> PanoVGGTLocalPrediction:
    out = _as_dict(output)
    poses = _first_present(out, ("camera_poses", "poses", "pose", "extrinsics", "extrinsic"))
    depth = _first_present(out, ("depth", "depths", "depth_map", "depth_maps", "pred_depth"))
    if poses is None or depth is None:
        raise ValueError("PanoVGGT output must include camera poses and depth.")
    poses_t = _normalize_poses(torch.as_tensor(poses, device=images.device))
    depth_t = _normalize_depth(torch.as_tensor(depth, device=images.device))
    confidence_t = _normalize_confidence(
        _first_present(out, ("depth_confidence", "confidence", "conf", "scores")),
        depth_t,
    )
    points_t = _normalize_point_maps(
        _first_present(out, ("world_points", "local_points", "point_maps", "points", "points3d")),
        depth_t,
        poses_t,
    )
    descriptors = _first_present(out, ("descriptors", "descriptor", "tokens", "features"))
    descriptors_t = None if descriptors is None else _drop_batch_dim(torch.as_tensor(descriptors, device=images.device)).float()
    return PanoVGGTLocalPrediction(
        poses_c2w=poses_t,
        depth=depth_t,
        confidence=confidence_t,
        point_maps=points_t,
        descriptors=descriptors_t,
    )


class PanoVGGTInferenceEngine:
    """Base PanoVGGT inference engine."""

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class FakePanoVGGTInferenceEngine(PanoVGGTInferenceEngine):
    """Deterministic geometry prior for tests and local smoke runs."""

    def __init__(self, image_size: tuple[int, int] | None = (64, 128), translation_step: float = 0.08) -> None:
        self.image_size = image_size
        self.translation_step = float(translation_step)

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        images = _resize_images(images.float(), self.image_size)
        n, _, height, width = images.shape
        device = images.device
        dtype = images.dtype
        poses = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(n, 1, 1)
        poses[:, 0, 3] = torch.arange(n, device=device, dtype=dtype) * self.translation_step

        grid = pixel_grid(height, width, device=device, dtype=dtype)
        u = grid[..., 0] / float(width)
        v = grid[..., 1] / float(height)
        depth = 2.0 + 0.25 * torch.sin(2.0 * torch.pi * u) * torch.cos(torch.pi * v)
        depth = depth.clamp_min(0.2).view(1, 1, height, width).repeat(n, 1, 1, 1)
        image_luma = images.mean(dim=1, keepdim=True)
        confidence = (0.65 + 0.35 * image_luma).clamp(0.0, 1.0)
        points = _build_point_maps(depth, poses)
        descriptors = torch.cat(
            [images.mean(dim=(2, 3)), images.std(dim=(2, 3), unbiased=False)],
            dim=1,
        )
        return PanoVGGTLocalPrediction(
            poses_c2w=poses,
            depth=depth,
            confidence=confidence,
            point_maps=points,
            descriptors=descriptors,
        )

    def load_checkpoint(self, path: str) -> None:
        return None


class ExternalPanoVGGTInferenceEngine(PanoVGGTInferenceEngine):
    """Dynamic wrapper around an external PanoVGGT checkout."""

    DEFAULT_CLASS_PATHS = (
        "panovggt.models.panovggt_model.PanoVGGTModel",
        "panovggt.models.panovggt_model.PanoVGGT",
        "panovggt.models.PanoVGGT",
        "vggt.models.vggt.VGGT",
    )

    def __init__(
        self,
        *,
        repo_path: str | None,
        config_path: str | None = None,
        checkpoint: str | None = None,
        class_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        image_size: tuple[int, int] | None = (518, 1036),
        device: torch.device | str | None = None,
        amp: bool = True,
        input_batch_dim: bool = True,
        strict_checkpoint: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config_path = config_path
        self.image_size = image_size
        self.amp = bool(amp)
        self.input_batch_dim = bool(input_batch_dim)
        self.strict_checkpoint = bool(strict_checkpoint)
        if repo_path:
            repo = str(Path(repo_path).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
        self.model = self._build_model(class_path, model_kwargs or {}).to(self.device)
        if checkpoint:
            self.load_checkpoint(checkpoint)
        self.model.eval()

    def _build_model(self, class_path: str | None, model_kwargs: dict[str, Any]) -> torch.nn.Module:
        paths = (class_path,) if class_path else self.DEFAULT_CLASS_PATHS
        errors: list[str] = []
        for path in paths:
            if not path:
                continue
            try:
                cls = _import_attr(path)
                return self._instantiate(cls, model_kwargs)
            except Exception as exc:  # pragma: no cover - exercised only with external checkout
                errors.append(f"{path}: {exc}")
        joined = "\n".join(errors)
        raise ImportError(f"Could not construct external PanoVGGT model.\n{joined}")

    def _instantiate(self, cls: type, model_kwargs: dict[str, Any]) -> torch.nn.Module:
        if self.config_path is not None and cls.__name__ == "PanoVGGTModel":
            official = self._instantiate_official_panovggt(cls, model_kwargs)
            if official is not None:
                return official
        attempts: list[dict[str, Any]] = []
        if self.config_path is not None:
            attempts.append({**model_kwargs, "config_path": self.config_path})
            attempts.append({**model_kwargs, "cfg_path": self.config_path})
        attempts.append(model_kwargs)
        attempts.append({})
        signature = inspect.signature(cls)
        for kwargs in attempts:
            try:
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                    return cls(**kwargs)
                filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
                return cls(**filtered)
            except TypeError:
                continue
        return cls()

    def _instantiate_official_panovggt(
        self,
        cls: type,
        model_kwargs: dict[str, Any],
    ) -> torch.nn.Module | None:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - external dependency only
            raise ImportError("External PanoVGGT requires omegaconf for config loading.") from exc

        cfg = OmegaConf.load(self.config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        aggregator = OmegaConf.to_container(mc.aggregator, resolve=True)
        kwargs = {
            "img_size": int(cfg.img_size),
            "patch_size": int(cfg.patch_size),
            "embed_dim": int(cfg.embed_dim),
            "enable_camera": bool(mc.enable_camera),
            "enable_depth": bool(mc.enable_depth),
            "enable_point": bool(mc.enable_point),
            "aggregator": aggregator,
        }
        kwargs.update(model_kwargs)
        signature = inspect.signature(cls)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        }
        return cls(**filtered)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        state = payload
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if isinstance(payload, dict) and key in payload:
                state = payload[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint payload in {path}")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=self.strict_checkpoint)
        self.model.eval()

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        images = _resize_images(images.float().to(self.device), self.image_size)
        model_input = images.unsqueeze(0) if self.input_batch_dim else images
        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self._call_model(model_input)
            else:
                output = self._call_model(model_input)
        return normalize_panovggt_output(output, images)

    def _call_model(self, model_input: torch.Tensor) -> Any:
        for name in ("infer", "inference", "predict", "forward"):
            method = getattr(self.model, name, None)
            if method is None:
                continue
            try:
                return method(model_input)
            except TypeError:
                continue
        return self.model(model_input)


def build_panovggt_engine(config: dict, *, device: torch.device | str | None = None) -> PanoVGGTInferenceEngine:
    engine_name = str(config.get("engine", "external")).lower()
    size_cfg = config.get("image_size", [518, 1036])
    image_size = None if size_cfg is None else (int(size_cfg[0]), int(size_cfg[1]))
    if engine_name == "fake":
        return FakePanoVGGTInferenceEngine(
            image_size=image_size,
            translation_step=float(config.get("fake_translation_step", 0.08)),
        )
    return ExternalPanoVGGTInferenceEngine(
        repo_path=config.get("repo_path"),
        config_path=config.get("config_path"),
        checkpoint=config.get("checkpoint"),
        class_path=config.get("class_path"),
        model_kwargs=dict(config.get("model_kwargs", {})),
        image_size=image_size,
        device=device,
        amp=bool(config.get("amp", True)),
        input_batch_dim=bool(config.get("input_batch_dim", True)),
        strict_checkpoint=bool(config.get("strict_checkpoint", False)),
    )
