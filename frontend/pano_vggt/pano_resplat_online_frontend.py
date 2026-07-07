"""Online Pano-ReSplat wrapper for direct local-to-global Gaussian fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image

from .pano_resplat_frontend import PanoReSplatFrontend
from .tracker import PanoVGGTLongTracker, build_panovggt_frontend_from_config


@dataclass
class PanoReSplatOnlineArtifact:
    """Side-channel payload emitted after one complete local ReSplat window."""

    window_id: int
    frame_ids: tuple[int, ...]
    final_state: Any
    result: dict[str, Any]
    metrics: dict[str, float]


def _resize_chw(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(int(v) for v in image.shape[-2:]) == tuple(int(v) for v in size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(int(size[0]), int(size[1])),
        mode="bilinear",
        align_corners=False,
    )[0]


def _resize_scalar(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = field.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected scalar map as HxW or 1xHxW, got {tuple(field.shape)}")
    if tuple(int(v) for v in tensor.shape[-2:]) == tuple(int(v) for v in size):
        return tensor[:1]
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0][:1]


def _resize_mask(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = field.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected mask as HxW or 1xHxW, got {tuple(field.shape)}")
    if tuple(int(v) for v in tensor.shape[-2:]) != tuple(int(v) for v in size):
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="nearest")[0]
    return tensor[:1] > 0.5


def _resize_points(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = points.detach().float()
    if tensor.ndim != 3 or int(tensor.shape[-1]) != 3:
        raise ValueError(f"Expected world points as HxWx3, got {tuple(points.shape)}")
    if tuple(int(v) for v in tensor.shape[:2]) == tuple(int(v) for v in size):
        return tensor
    return F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).contiguous()


class PanoReSplatOnlineFrontend(PanoDROIDFrontend):
    """Wrap ``PanoVGGTLongTracker`` and emit a ReSplat Gaussian side artifact every 4 frames."""

    def __init__(
        self,
        *,
        tracker: PanoVGGTLongTracker,
        resplat_frontend: PanoReSplatFrontend,
        window_size: int = 4,
        stride: int = 4,
        image_height: int = 512,
        image_width: int = 1024,
        num_refine: int = 1,
        require_features: bool = True,
        allow_synthetic_features: bool = False,
        synthetic_feature_dim: int = 16,
        synthetic_feature_stride: int = 4,
    ) -> None:
        self.tracker = tracker
        self.resplat_frontend = resplat_frontend
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.image_size = (int(image_height), int(image_width))
        self.num_refine = max(0, int(num_refine))
        self.require_features = bool(require_features)
        self.allow_synthetic_features = bool(allow_synthetic_features)
        self.synthetic_feature_dim = max(1, int(synthetic_feature_dim))
        self.synthetic_feature_stride = max(1, int(synthetic_feature_stride))
        self._image_by_frame: dict[int, torch.Tensor] = {}
        self._output_by_frame: dict[int, FrontendOutput] = {}
        self._ready_ids: list[int] = []
        self._window_cursor = 0
        self._pending_artifacts: list[PanoReSplatOnlineArtifact] = []
        self._window_id = 0
        self.last_profile: dict[str, float | int] = {}

    def initialize(self, sequence_meta: dict) -> None:
        self.tracker.initialize(sequence_meta)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._image_by_frame = {}
        self._output_by_frame = {}
        self._ready_ids = []
        self._window_cursor = 0
        self._pending_artifacts = []
        self._window_id = 0
        self.last_profile = {}

    def load_checkpoint(self, path: str) -> None:
        self.tracker.load_checkpoint(path)

    def load_resplat_checkpoint(self, path: str) -> dict[str, Any]:
        from frontend.pano_vggt.train_resplat_gaussian import _load_checkpoint

        return _load_checkpoint(self.resplat_frontend, path)

    def track(self, frame: PanoFrame) -> FrontendOutput:
        image = _resize_chw(ensure_chw_image(frame.image).float(), self.image_size)
        resized = PanoFrame(
            image=image,
            timestamp=frame.timestamp,
            frame_id=int(frame.frame_id),
            mask=frame.mask,
            meta=frame.meta,
        )
        self._image_by_frame[int(frame.frame_id)] = image.detach().cpu()
        return self.tracker.track(resized)

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        outputs = self.tracker.pop_ready_outputs()
        self._remember_outputs(outputs)
        self._run_ready_windows()
        return outputs

    def flush(self) -> list[FrontendOutput]:
        outputs = self.tracker.flush()
        self._remember_outputs(outputs)
        self._run_ready_windows()
        return outputs

    def consume_resplat_artifacts(self) -> list[PanoReSplatOnlineArtifact]:
        out = self._pending_artifacts
        self._pending_artifacts = []
        return out

    def image_for_frame(self, frame_id: int) -> torch.Tensor | None:
        image = self._image_by_frame.get(int(frame_id))
        return None if image is None else image.detach().cpu()

    def _remember_outputs(self, outputs: list[FrontendOutput]) -> None:
        for out in outputs:
            fid = int(out.frame_id)
            self._output_by_frame[fid] = out
            if fid not in self._ready_ids:
                self._ready_ids.append(fid)
        self._ready_ids.sort()

    def _run_ready_windows(self) -> None:
        while self._window_cursor + self.window_size <= len(self._ready_ids):
            ids = tuple(int(v) for v in self._ready_ids[self._window_cursor : self._window_cursor + self.window_size])
            if any(fid not in self._output_by_frame for fid in ids):
                break
            artifact = self._run_window(ids)
            self._pending_artifacts.append(artifact)
            self._window_cursor += self.stride

    def _feature_for_frame(self, frame_id: int, image: torch.Tensor) -> torch.Tensor:
        feature_cache = getattr(self.tracker, "features_by_frame", {})
        feature = feature_cache.get(int(frame_id)) if isinstance(feature_cache, dict) else None
        if torch.is_tensor(feature):
            return feature.detach().float()
        if self.require_features and not self.allow_synthetic_features:
            raise RuntimeError(
                "pano_resplat_online requires PanoVGGT dense features/descriptors. "
                "Enable a PanoVGGT feature/matching output or set "
                "PanoReSplatOnline.allow_synthetic_features=true for smoke tests only."
            )
        pooled = F.interpolate(
            image.unsqueeze(0),
            size=(
                max(1, int(image.shape[-2]) // self.synthetic_feature_stride),
                max(1, int(image.shape[-1]) // self.synthetic_feature_stride),
            ),
            mode="bilinear",
            align_corners=False,
        )[0]
        channels = [pooled]
        while sum(int(t.shape[0]) for t in channels) < self.synthetic_feature_dim:
            channels.append(pooled.mean(dim=0, keepdim=True))
        return torch.cat(channels, dim=0)[: self.synthetic_feature_dim].contiguous()

    def _run_window(self, frame_ids: tuple[int, ...]) -> PanoReSplatOnlineArtifact:
        device = next(self.resplat_frontend.parameters()).device
        dtype = next(self.resplat_frontend.parameters()).dtype
        images: list[torch.Tensor] = []
        depths: list[torch.Tensor] = []
        poses: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        world_points: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for fid in frame_ids:
            out = self._output_by_frame[int(fid)]
            image = self._image_by_frame.get(int(fid))
            if image is None:
                raise RuntimeError(f"Missing cached ReSplat input image for frame {fid}.")
            image = image.float()
            if out.inverse_depth is None or out.world_points is None:
                raise RuntimeError(f"Frame {fid} is missing depth/world_points for ReSplat online window.")
            inv_depth = _resize_scalar(out.inverse_depth, self.image_size).clamp_min(1.0e-6)
            depth = inv_depth.reciprocal()
            valid = (
                _resize_mask(out.valid_world_points_mask, self.image_size)
                if out.valid_world_points_mask is not None
                else torch.isfinite(depth) & (depth > 0.0)
            )
            points = _resize_points(out.world_points, self.image_size)
            images.append(image)
            depths.append(depth)
            poses.append(out.pose_c2w.detach().float())
            valid_masks.append(valid)
            world_points.append(points)
            features.append(self._feature_for_frame(int(fid), image))

        context = {
            "images": torch.stack(images, dim=0).unsqueeze(0).to(device=device, dtype=dtype),
            "features": torch.stack(features, dim=0).unsqueeze(0).to(device=device, dtype=dtype),
            "depths": torch.stack(depths, dim=0).unsqueeze(0).to(device=device, dtype=dtype),
            "poses_c2w": torch.stack(poses, dim=0).unsqueeze(0).to(device=device, dtype=dtype),
            "valid_mask": torch.stack(valid_masks, dim=0).unsqueeze(0).to(device=device),
            "world_points": torch.stack(world_points, dim=0).unsqueeze(0).to(device=device, dtype=dtype),
        }
        self.resplat_frontend.eval()
        with torch.no_grad():
            result = self.resplat_frontend(
                context,
                target={"poses_c2w": context["poses_c2w"], "images": context["images"]},
                num_refine=self.num_refine,
                return_all=False,
            )
        final_state = result["final_state"].to(device="cpu", dtype=torch.float32)
        compact_debug = result.get("compactor_debug", {})
        update_metrics = result.get("update_metrics", [])
        metrics = {
            "window_id": float(self._window_id),
            "frame_count": float(len(frame_ids)),
            "gaussian_count": float(final_state.num_gaussians),
        }
        artifact = PanoReSplatOnlineArtifact(
            window_id=int(self._window_id),
            frame_ids=frame_ids,
            final_state=final_state,
            result={"compactor_debug": compact_debug, "update_metrics": update_metrics},
            metrics=metrics,
        )
        self.last_profile = {
            "chunk_index": int(self._window_id),
            "resplat_window_size": int(len(frame_ids)),
            "resplat_gaussians": int(final_state.num_gaussians),
        }
        self._window_id += 1
        return artifact


def _resplat_build_config(config: dict) -> dict[str, Any]:
    resplat_cfg = config.get("PanoReSplat", {})
    if not isinstance(resplat_cfg, dict) or not resplat_cfg:
        resplat_cfg = {}
    build_cfg: dict[str, Any] = dict(resplat_cfg)
    for key in ("Initializer", "Feedback", "Refiner", "Renderer", "TrainingRender", "Compactor"):
        if key not in build_cfg and isinstance(config.get(key), dict):
            build_cfg[key] = dict(config[key])
    return build_cfg


def build_pano_resplat_online_frontend_from_config(config: dict) -> PanoReSplatOnlineFrontend:
    online_cfg = config.get("PanoReSplatOnline", {})
    if not isinstance(online_cfg, dict):
        online_cfg = {}
    tracker = build_panovggt_frontend_from_config(config)
    frontend_cfg = config.get("Frontend", {})
    panovggt_ckpt = frontend_cfg.get("checkpoint") if isinstance(frontend_cfg, dict) else None
    if panovggt_ckpt:
        tracker.load_checkpoint(str(panovggt_ckpt))
    device = torch.device(str(online_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    from frontend.pano_vggt.train_resplat_gaussian import _build_frontend, _load_checkpoint

    resplat = _build_frontend(_resplat_build_config(config), device=device)
    ckpt = online_cfg.get("checkpoint") or config.get("PanoReSplat", {}).get("checkpoint", None)
    if ckpt:
        _load_checkpoint(resplat, str(ckpt))
    elif bool(online_cfg.get("require_checkpoint", True)):
        raise RuntimeError(
            "PanoReSplatOnline.require_checkpoint=true but no PanoReSplatOnline.checkpoint was provided."
        )
    return PanoReSplatOnlineFrontend(
        tracker=tracker,
        resplat_frontend=resplat,
        window_size=int(online_cfg.get("window_size", 4)),
        stride=int(online_cfg.get("stride", online_cfg.get("window_size", 4))),
        image_height=int(online_cfg.get("image_height", online_cfg.get("input_height", 512))),
        image_width=int(online_cfg.get("image_width", online_cfg.get("input_width", 1024))),
        num_refine=int(online_cfg.get("num_refine", 1)),
        require_features=bool(online_cfg.get("require_features", True)),
        allow_synthetic_features=bool(online_cfg.get("allow_synthetic_features", False)),
        synthetic_feature_dim=int(online_cfg.get("synthetic_feature_dim", 16)),
        synthetic_feature_stride=int(online_cfg.get("synthetic_feature_stride", 4)),
    )
