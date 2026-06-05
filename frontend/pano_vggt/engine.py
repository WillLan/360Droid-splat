"""PanoVGGT inference engines.

The real engine dynamically loads an external PanoVGGT checkout. Tests and
smoke runs use the fake engine so this repository stays self contained.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import inspect
import math
import sys
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

from .dense_matcher import PoseGuidedDenseMatcher
from .factor_graph import DenseSphereFactorGraph
from .m3_config import M3SphereConfig, parse_m3_sphere_config
from .matching_adapter import (
    MatchingSkyAdapter,
    call_forward_with_features,
    extract_features_with_hook,
    load_matching_sky_checkpoint,
    make_fake_matching_outputs,
    run_matching_sky_head,
)
from .types import PanoVGGTLocalPrediction


def _resize_images(images: torch.Tensor, image_size: tuple[int, int] | None) -> torch.Tensor:
    if image_size is None:
        return images
    if tuple(images.shape[-2:]) == tuple(image_size):
        return images
    return F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)


def _ceil_size_to_multiple(image_size: tuple[int, int], multiple: int) -> tuple[int, int]:
    if int(multiple) <= 1:
        return image_size
    h, w = int(image_size[0]), int(image_size[1])
    return (
        int(math.ceil(h / float(multiple)) * int(multiple)),
        int(math.ceil(w / float(multiple)) * int(multiple)),
    )


def _resize_prediction(pred: PanoVGGTLocalPrediction, image_size: tuple[int, int]) -> PanoVGGTLocalPrediction:
    if tuple(pred.depth.shape[-2:]) == tuple(image_size):
        return pred
    depth = F.interpolate(pred.depth.float(), size=image_size, mode="bilinear", align_corners=False).clamp_min(1e-6)
    confidence = F.interpolate(pred.confidence.float(), size=image_size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    points = _resize_points(pred.chunk_world_points, image_size)
    local_points = None if pred.local_points is None else _resize_points(pred.local_points, image_size)
    global_points = None if pred.global_points is None else _resize_points(pred.global_points, image_size)
    return PanoVGGTLocalPrediction(
        poses_c2w=pred.poses_c2w,
        depth=depth,
        confidence=confidence,
        chunk_world_points=points,
        local_points=local_points,
        global_points=global_points,
        descriptors=pred.descriptors,
        dense_descriptors=pred.dense_descriptors,
        match_confidence=pred.match_confidence,
        static_confidence=pred.static_confidence,
        sky_logits=pred.sky_logits,
        sky_prob=pred.sky_prob,
        feature_hw=pred.feature_hw,
        image_hw=image_size,
        descriptor_dim=pred.descriptor_dim,
        matching_debug=pred.matching_debug,
        ba_residual_angular=pred.ba_residual_angular,
        ba_valid_ratio=pred.ba_valid_ratio,
        ba_update_norm=pred.ba_update_norm,
    )


def _prediction_with_matching(
    pred: PanoVGGTLocalPrediction,
    matching: dict[str, Any],
    *,
    image_hw: tuple[int, int],
) -> PanoVGGTLocalPrediction:
    dense = matching.get("dense_descriptors", pred.dense_descriptors)
    match_conf = matching.get("match_confidence", pred.match_confidence)
    sky_logits = matching.get("sky_logits", pred.sky_logits)
    sky_prob = matching.get("sky_prob", pred.sky_prob)
    feature_hw = matching.get("feature_hw", pred.feature_hw)
    descriptor_dim = int(matching.get("descriptor_dim", pred.descriptor_dim))
    if dense is not None and int(dense.shape[0]) != int(pred.poses_c2w.shape[0]):
        raise ValueError("dense_descriptors frame count does not match PanoVGGT poses.")
    if match_conf is not None and int(match_conf.shape[0]) != int(pred.poses_c2w.shape[0]):
        raise ValueError("match_confidence frame count does not match PanoVGGT poses.")
    if sky_prob is not None and int(sky_prob.shape[0]) != int(pred.poses_c2w.shape[0]):
        raise ValueError("sky_prob frame count does not match PanoVGGT poses.")
    return replace(
        pred,
        dense_descriptors=dense,
        match_confidence=match_conf,
        sky_logits=sky_logits,
        sky_prob=sky_prob,
        feature_hw=feature_hw,
        image_hw=image_hw,
        descriptor_dim=descriptor_dim,
    )


def _dense_matcher_from_config(config: M3SphereConfig) -> PoseGuidedDenseMatcher:
    dense = config.dense_matching
    return PoseGuidedDenseMatcher(
        search_radius=dense.search_radius,
        topk=dense.topk,
        min_match_confidence=dense.min_match_confidence,
        min_static_confidence=dense.min_static_confidence,
        min_match_score=dense.min_match_score,
        max_factors=dense.max_factors,
        max_samples_per_edge=dense.max_samples_per_edge,
        use_wraparound=dense.use_wraparound,
        forward_backward=dense.forward_backward,
        fb_tolerance=dense.fb_tolerance,
        use_depth_consistency=dense.use_depth_consistency,
        depth_consistency_rel=dense.depth_consistency_rel,
        depth_consistency_abs=dense.depth_consistency_abs,
    )


def _run_dense_matching_if_enabled(
    pred: PanoVGGTLocalPrediction,
    config: M3SphereConfig,
) -> tuple[PanoVGGTLocalPrediction, DenseSphereFactorGraph | None]:
    if not (config.enabled and config.dense_matching.enabled):
        return pred, None
    if pred.dense_descriptors is None or pred.match_confidence is None:
        raise RuntimeError("PanoVGGT.DenseMatching.enabled=true requires dense_descriptors and match_confidence.")
    if pred.sky_prob is None:
        raise RuntimeError("PanoVGGT.DenseMatching.enabled=true requires sky_prob so sky can affect factor validity/weight.")
    if pred.feature_hw is None or pred.image_hw is None:
        raise RuntimeError("PanoVGGT.DenseMatching.enabled=true requires feature_hw and image_hw.")
    edges = DenseSphereFactorGraph.build_edges(
        int(pred.poses_c2w.shape[0]),
        temporal_radius=config.inference_window.temporal_radius,
        max_edges=config.inference_window.max_edges,
        device=pred.depth.device,
    )
    matcher = _dense_matcher_from_config(config)
    graph = matcher.match(
        poses_c2w=pred.poses_c2w,
        depth=pred.depth,
        dense_descriptors=pred.dense_descriptors,
        match_confidence=pred.match_confidence,
        sky_prob=pred.sky_prob,
        static_confidence=pred.static_confidence,
        image_hw=pred.image_hw,
        feature_hw=pred.feature_hw,
        edge_pairs=edges,
    )
    return replace(pred, matching_debug=graph.metrics()), graph


def _import_attr(path: str) -> Any:
    module_name, attr = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr)


@contextmanager
def _maybe_skip_dinov2_pretrain(enabled: bool):
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
        "global_points",
        "local_points",
        "point_maps",
        "points",
        "points3d",
        "extrinsics",
        "extrinsic",
        "descriptors",
        "dense_descriptors",
        "dense_descriptor",
        "match_confidence",
        "matching_confidence",
        "static_confidence",
        "sky_logits",
        "sky_prob",
        "feature_hw",
        "image_hw",
        "descriptor_dim",
        "matching_debug",
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


def _select_poses(output: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    for name in ("camera_poses", "poses", "pose"):
        value = _first_present(output, (name,))
        if value is not None:
            return _normalize_poses(torch.as_tensor(value, device=device))
    value = _first_present(output, ("extrinsics", "extrinsic"))
    if value is None:
        return None
    w2c = _normalize_poses(torch.as_tensor(value, device=device))
    return torch.linalg.inv(w2c)


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
    pts_cam = _build_local_points(depth)
    return _local_points_to_world(pts_cam, poses_c2w)


def _build_local_points(depth: torch.Tensor) -> torch.Tensor:
    n, _, height, width = depth.shape
    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    bearing = erp_pixel_to_bearing(grid, height, width).to(device=depth.device, dtype=depth.dtype)
    return bearing.unsqueeze(0) * depth[:, 0].unsqueeze(-1)


def _local_points_to_world(local_points: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
    n = local_points.shape[0]
    rot = poses_c2w[:, :3, :3]
    trans = poses_c2w[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", rot, local_points) + trans.view(n, 1, 1, 3)


def _resize_points(points: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if tuple(points.shape[1:3]) == tuple(image_size):
        return points.float()
    return F.interpolate(
        points.permute(0, 3, 1, 2).float(),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def _normalize_points_shape(points: torch.Tensor | None, depth: torch.Tensor) -> torch.Tensor | None:
    if points is None:
        return None
    points = _drop_batch_dim(points)
    if points.ndim == 4 and points.shape[-1] != 3 and points.shape[1] == 3:
        points = points.permute(0, 2, 3, 1)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point maps as NxHxWx3, got {tuple(points.shape)}")
    if points.shape[1:3] != depth.shape[-2:]:
        points = _resize_points(points, tuple(depth.shape[-2:]))
    return points.float()


def _normalize_optional_confidence(confidence: torch.Tensor | None) -> torch.Tensor | None:
    if confidence is None:
        return None
    confidence = _drop_batch_dim(confidence)
    if confidence.ndim == 3:
        confidence = confidence.unsqueeze(1)
    elif confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence.permute(0, 3, 1, 2)
    if confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError(f"Expected optional confidence as Nx1xHxW, got {tuple(confidence.shape)}")
    return confidence.float().clamp(0.0, 1.0)


def _normalize_optional_map(value: torch.Tensor | None, *, name: str, clamp_unit: bool = False) -> torch.Tensor | None:
    if value is None:
        return None
    value = _drop_batch_dim(value)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value.permute(0, 3, 1, 2)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"Expected {name} as Nx1xHxW, got {tuple(value.shape)}")
    value = value.float()
    return value.clamp(0.0, 1.0) if clamp_unit else value


def _normalize_dense_descriptors(descriptors: torch.Tensor | None) -> torch.Tensor | None:
    if descriptors is None:
        return None
    descriptors = _drop_batch_dim(descriptors)
    if descriptors.ndim != 4:
        raise ValueError(f"Expected dense descriptors as NxCxHxW, got {tuple(descriptors.shape)}")
    return descriptors.float()


def _normalize_hw_value(value: Any, *, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if len(value) != 2:
        raise ValueError(f"Expected {name} as two values, got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def normalize_panovggt_output(output: Any, images: torch.Tensor) -> PanoVGGTLocalPrediction:
    out = _as_dict(output)
    depth = _first_present(out, ("depth", "depths", "depth_map", "depth_maps", "pred_depth"))
    poses_t = _select_poses(out, images.device)
    if poses_t is None or depth is None:
        raise ValueError("PanoVGGT output must include camera poses and depth.")
    depth_t = _normalize_depth(torch.as_tensor(depth, device=images.device))
    confidence_t = _normalize_confidence(
        _first_present(out, ("depth_confidence", "confidence", "conf", "scores")),
        depth_t,
    )
    local_points_t = _normalize_points_shape(
        _first_present(out, ("local_points", "cam_points", "camera_points")),
        depth_t,
    )
    chunk_world_t = _normalize_points_shape(
        _first_present(out, ("world_points", "points", "point_maps", "points3d")),
        depth_t,
    )
    global_points_t = _normalize_points_shape(_first_present(out, ("global_points",)), depth_t)
    if chunk_world_t is None:
        if local_points_t is None:
            local_points_t = _build_local_points(depth_t)
        chunk_world_t = _local_points_to_world(local_points_t, poses_t)
    descriptors = _first_present(out, ("descriptors", "descriptor", "tokens", "features"))
    descriptors_t = None if descriptors is None else _drop_batch_dim(torch.as_tensor(descriptors, device=images.device)).float()
    dense_descriptor_value = _first_present(out, ("dense_descriptors", "dense_descriptor"))
    if dense_descriptor_value is None and descriptors_t is not None and descriptors_t.ndim == 4:
        dense_descriptor_value = descriptors_t
    dense_descriptors_t = _normalize_dense_descriptors(
        None if dense_descriptor_value is None else torch.as_tensor(dense_descriptor_value, device=images.device)
    )
    match_confidence_t = _normalize_optional_confidence(
        None
        if _first_present(out, ("match_confidence", "matching_confidence")) is None
        else torch.as_tensor(_first_present(out, ("match_confidence", "matching_confidence")), device=images.device)
    )
    static_confidence_t = _normalize_optional_confidence(
        None
        if _first_present(out, ("static_confidence",)) is None
        else torch.as_tensor(_first_present(out, ("static_confidence",)), device=images.device)
    )
    sky_logits_t = _normalize_optional_map(
        None
        if _first_present(out, ("sky_logits",)) is None
        else torch.as_tensor(_first_present(out, ("sky_logits",)), device=images.device),
        name="sky_logits",
    )
    sky_prob_t = _normalize_optional_map(
        None
        if _first_present(out, ("sky_prob", "sky_probability")) is None
        else torch.as_tensor(_first_present(out, ("sky_prob", "sky_probability")), device=images.device),
        name="sky_prob",
        clamp_unit=True,
    )
    feature_hw = _normalize_hw_value(_first_present(out, ("feature_hw",)), name="feature_hw")
    if feature_hw is None and dense_descriptors_t is not None:
        feature_hw = tuple(int(v) for v in dense_descriptors_t.shape[-2:])
    if feature_hw is None and match_confidence_t is not None:
        feature_hw = tuple(int(v) for v in match_confidence_t.shape[-2:])
    if feature_hw is None and static_confidence_t is not None:
        feature_hw = tuple(int(v) for v in static_confidence_t.shape[-2:])
    if feature_hw is None and sky_prob_t is not None:
        feature_hw = tuple(int(v) for v in sky_prob_t.shape[-2:])
    image_hw = _normalize_hw_value(_first_present(out, ("image_hw",)), name="image_hw")
    if image_hw is None:
        image_hw = tuple(int(v) for v in depth_t.shape[-2:])
    descriptor_dim_value = _first_present(out, ("descriptor_dim",))
    if descriptor_dim_value is None and dense_descriptors_t is not None:
        descriptor_dim_value = int(dense_descriptors_t.shape[1])
    descriptor_dim = int(descriptor_dim_value) if descriptor_dim_value is not None else 24
    matching_debug_value = _first_present(out, ("matching_debug",))
    matching_debug = None
    if isinstance(matching_debug_value, dict):
        matching_debug = {str(key): float(value) for key, value in matching_debug_value.items()}
    return PanoVGGTLocalPrediction(
        poses_c2w=poses_t,
        depth=depth_t,
        confidence=confidence_t,
        chunk_world_points=chunk_world_t,
        local_points=local_points_t,
        global_points=global_points_t,
        descriptors=descriptors_t,
        dense_descriptors=dense_descriptors_t,
        match_confidence=match_confidence_t,
        static_confidence=static_confidence_t,
        sky_logits=sky_logits_t,
        sky_prob=sky_prob_t,
        feature_hw=feature_hw,
        image_hw=image_hw,
        descriptor_dim=descriptor_dim,
        matching_debug=matching_debug,
    )


class PanoVGGTInferenceEngine:
    """Base PanoVGGT inference engine."""

    last_dense_factor_graph: DenseSphereFactorGraph | None = None

    def infer(self, images: torch.Tensor) -> PanoVGGTLocalPrediction:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class FakePanoVGGTInferenceEngine(PanoVGGTInferenceEngine):
    """Deterministic geometry prior for tests and local smoke runs."""

    def __init__(
        self,
        image_size: tuple[int, int] | None = (64, 128),
        translation_step: float = 0.08,
        *,
        m3_config: M3SphereConfig | None = None,
    ) -> None:
        self.image_size = image_size
        self.translation_step = float(translation_step)
        self.m3_config = m3_config or M3SphereConfig()
        self.last_dense_factor_graph: DenseSphereFactorGraph | None = None

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
        local_points = _build_local_points(depth)
        points = _local_points_to_world(local_points, poses)
        descriptors = torch.cat(
            [images.mean(dim=(2, 3)), images.std(dim=(2, 3), unbiased=False)],
            dim=1,
        )
        pred = PanoVGGTLocalPrediction(
            poses_c2w=poses,
            depth=depth,
            confidence=confidence,
            chunk_world_points=points,
            local_points=local_points,
            descriptors=descriptors,
        )
        if self.m3_config.enabled and self.m3_config.matching_head.enabled:
            if not self.m3_config.matching_head.allow_fake_matching:
                raise RuntimeError(
                    "PanoVGGT MatchingHead is enabled with engine=fake, but "
                    "MatchingHead.allow_fake_matching is false."
                )
            matching = make_fake_matching_outputs(
                images,
                descriptor_dim=self.m3_config.matching_head.descriptor_dim,
                feature_stride=self.m3_config.matching_head.fake_feature_stride,
            )
            pred = _prediction_with_matching(pred, matching, image_hw=(height, width))
        pred, self.last_dense_factor_graph = _run_dense_matching_if_enabled(pred, self.m3_config)
        return pred

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
        skip_dinov2_pretrain: bool = False,
        patch_multiple: int = 14,
        m3_config: M3SphereConfig | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config_path = config_path
        self.image_size = image_size
        self.amp = bool(amp)
        self.input_batch_dim = bool(input_batch_dim)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.skip_dinov2_pretrain = bool(skip_dinov2_pretrain)
        self.patch_multiple = int(patch_multiple)
        self.m3_config = m3_config or M3SphereConfig()
        self.matching_adapter: MatchingSkyAdapter | None = None
        self.last_dense_factor_graph: DenseSphereFactorGraph | None = None
        if repo_path:
            repo = str(Path(repo_path).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
        with _maybe_skip_dinov2_pretrain(self.skip_dinov2_pretrain):
            self.model = self._build_model(class_path, model_kwargs or {}).to(self.device)
        if checkpoint:
            self.load_checkpoint(checkpoint)
        self.model.eval()
        if self.m3_config.enabled and self.m3_config.matching_head.enabled:
            head_cfg = self.m3_config.matching_head
            self.matching_adapter = load_matching_sky_checkpoint(
                head_cfg.checkpoint,
                matching_checkpoint=head_cfg.matching_checkpoint,
                sky_checkpoint=head_cfg.sky_checkpoint,
                device=self.device,
                descriptor_dim=head_cfg.descriptor_dim,
                feature_hook=head_cfg.feature_hook,
                feature_key=head_cfg.feature_key,
                strict=head_cfg.strict,
            )

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
        target_size = tuple(int(v) for v in images.shape[-2:])
        model_size = _ceil_size_to_multiple(target_size, self.patch_multiple)
        model_images = _resize_images(images, model_size)
        model_input = model_images.unsqueeze(0) if self.input_batch_dim else model_images
        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, feature = self._call_model_and_maybe_capture_feature(model_input)
            else:
                output, feature = self._call_model_and_maybe_capture_feature(model_input)
        pred = normalize_panovggt_output(output, model_images)
        if self.matching_adapter is not None:
            if feature is None:
                raise RuntimeError("PanoVGGT matching adapter is enabled but no feature tensor was captured.")
            matching = run_matching_sky_head(self.matching_adapter, feature.to(self.device))
            pred = _prediction_with_matching(pred, matching, image_hw=model_size)
        pred = _resize_prediction(pred, target_size)
        pred, self.last_dense_factor_graph = _run_dense_matching_if_enabled(pred, self.m3_config)
        return pred

    def _call_model_and_maybe_capture_feature(self, model_input: torch.Tensor) -> tuple[Any, torch.Tensor | None]:
        if self.matching_adapter is None:
            return self._call_model(model_input), None
        feature_hook = self.matching_adapter.feature_hook
        if feature_hook:
            output, feature = extract_features_with_hook(
                self.model,
                feature_hook,
                lambda: self._call_model(model_input),
                patch_size=self.patch_multiple,
                feature_key=self.m3_config.matching_head.feature_key,
            )
            return output, feature
        explicit = call_forward_with_features(self.model, model_input, patch_size=self.patch_multiple)
        if explicit is not None:
            return explicit
        raise RuntimeError(
            "PanoVGGT MatchingHead is enabled, but no feature_hook was configured or stored in the checkpoint, "
            "and the external model does not expose forward_with_features/infer_with_features."
        )

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
    m3_config = parse_m3_sphere_config({"PanoVGGT": config})
    engine_name = str(config.get("engine", "external")).lower()
    size_cfg = config.get("image_size", [518, 1036])
    image_size = None if size_cfg is None else (int(size_cfg[0]), int(size_cfg[1]))
    if engine_name == "fake":
        return FakePanoVGGTInferenceEngine(
            image_size=image_size,
            translation_step=float(config.get("fake_translation_step", 0.08)),
            m3_config=m3_config,
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
        skip_dinov2_pretrain=bool(config.get("skip_dinov2_pretrain", False)),
        patch_multiple=int(config.get("patch_multiple", 14)),
        m3_config=m3_config,
    )
