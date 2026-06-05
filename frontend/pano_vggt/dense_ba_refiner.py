"""PanoVGGT-M3 dense BA refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

from .dense_matcher import PoseGuidedDenseMatcher
from .factor_graph import DenseSphereFactorGraph
from .keyframe_memory import KeyframeMemory
from .m3_config import M3SphereConfig
from .spherical_dense_ba import SphericalTangentDenseBA, SphericalTangentDenseBAOutput
from .types import PanoVGGTLocalPrediction


@dataclass(frozen=True)
class DenseBARefinerStats:
    enabled: bool = False
    shadow_mode: bool = True
    success: bool = False
    used_refined: bool = False
    fallback_reason: str | None = None
    mean_residual_deg: float = 0.0
    median_residual_deg: float = 0.0
    initial_mean_residual_deg: float = 0.0
    valid_factor_ratio: float = 0.0
    num_factors: int = 0
    valid_factors: int = 0
    pose_update_norm: dict[str, float] = field(default_factory=dict)
    depth_update_norm: dict[str, float] = field(default_factory=dict)

    @property
    def status_suffix(self) -> str:
        if not self.enabled:
            return ""
        mode = "shadow" if self.shadow_mode else "active"
        if self.success:
            return f"|dense_ba_{mode}_success"
        reason = self.fallback_reason or "unknown"
        return f"|dense_ba_{mode}_fallback:{reason}"

    def as_debug(self) -> dict[str, float]:
        return {
            "dense_ba_enabled": float(self.enabled),
            "dense_ba_shadow_mode": float(self.shadow_mode),
            "dense_ba_success": float(self.success),
            "dense_ba_used_refined": float(self.used_refined),
            "dense_ba_mean_residual_deg": float(self.mean_residual_deg),
            "dense_ba_median_residual_deg": float(self.median_residual_deg),
            "dense_ba_initial_mean_residual_deg": float(self.initial_mean_residual_deg),
            "dense_ba_valid_factor_ratio": float(self.valid_factor_ratio),
            "dense_ba_num_factors": float(self.num_factors),
            "dense_ba_valid_factors": float(self.valid_factors),
            "dense_ba_pose_update_mean": float(self.pose_update_norm.get("mean", 0.0)),
            "dense_ba_pose_update_max": float(self.pose_update_norm.get("max", 0.0)),
            "dense_ba_pose_update_rot_max_deg": float(self.pose_update_norm.get("rot_max_deg", 0.0)),
            "dense_ba_pose_update_trans_max": float(self.pose_update_norm.get("trans_max", 0.0)),
            "dense_ba_depth_update_mean": float(self.depth_update_norm.get("mean", 0.0)),
            "dense_ba_depth_update_max": float(self.depth_update_norm.get("max", 0.0)),
        }


class PanoVGGTDenseBARefiner:
    """Config-gated local dense BA refiner for PanoVGGT predictions."""

    def __init__(self, config: M3SphereConfig) -> None:
        self.config = config
        self.solver = SphericalTangentDenseBA(
            factor_chunk_size=config.dense_ba.factor_chunk_size,
            huber_delta_deg=config.dense_ba.huber_delta_deg,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.dense_ba.enabled)

    @property
    def shadow_mode(self) -> bool:
        return bool(self.config.dense_ba.shadow_mode)

    def refine(
        self,
        pred: PanoVGGTLocalPrediction,
        frame_ids: tuple[int, ...],
        *,
        factor_graph: DenseSphereFactorGraph | None = None,
        keyframe_memory: KeyframeMemory | None = None,
    ) -> tuple[PanoVGGTLocalPrediction, DenseBARefinerStats]:
        """Return a refined prediction or the original prediction with fallback stats."""

        if not self.enabled:
            return pred, self._stats(enabled=False, success=False, reason="disabled")
        if self.config.dense_ba.residual_mode != "tangent":
            return pred, self._stats(success=False, reason="unsupported_residual_mode")
        if self.config.dense_ba.mode != "local_chunk":
            return pred, self._stats(success=False, reason="unsupported_mode")
        missing = self._missing_prediction_fields(pred)
        if missing:
            return pred, self._stats(success=False, reason=f"missing_{missing}")
        feature_hw = pred.feature_hw
        image_hw = pred.image_hw
        assert feature_hw is not None and image_hw is not None

        graph = factor_graph if factor_graph is not None else self._build_factor_graph(pred)
        if graph is None or not graph.factors:
            return pred, self._stats(success=False, reason="empty_factors")

        graph_metrics = graph.metrics()
        num_factors = int(graph_metrics.get("num_factors", 0.0))
        valid_factors = int(graph_metrics.get("valid_factors", 0.0))
        valid_ratio = float(graph_metrics.get("valid_factor_ratio", 0.0))
        if valid_factors < int(self.config.dense_ba.min_num_factors):
            return pred, self._stats(success=False, reason="too_few_factors", graph_metrics=graph_metrics)
        if valid_ratio < float(self.config.dense_ba.min_valid_factor_ratio):
            return pred, self._stats(success=False, reason="low_valid_factor_ratio", graph_metrics=graph_metrics)

        depth_low = F.interpolate(pred.depth.float(), size=feature_hw, mode="bilinear", align_corners=False).clamp_min(1.0e-6)
        log_inv_depth_low = depth_low.reciprocal().clamp_min(1.0e-6).log()
        fixed_frames = self._fixed_frames(frame_ids)
        output = self.solver(
            pred.poses_c2w.to(device=pred.depth.device, dtype=pred.depth.dtype),
            log_inv_depth_low.to(device=pred.depth.device, dtype=pred.depth.dtype),
            graph,
            fixed_frames=fixed_frames,
            iters=self.config.dense_ba.iters,
            damping=self.config.dense_ba.lm,
            optimize_pose=self.config.dense_ba.optimize_pose,
            optimize_depth=self.config.dense_ba.optimize_depth,
            pose_prior_weight=self.config.dense_ba.pose_prior_weight,
            depth_prior_weight=self.config.dense_ba.depth_prior_weight,
            max_pose_update_deg=self.config.dense_ba.max_pose_update_deg,
            max_logdepth_update=self.config.dense_ba.max_logdepth_update,
        )
        if output.failed:
            return pred, self._stats(success=False, reason=str(output.debug.get("fallback_reason", "solver_failed")), graph_metrics=graph_metrics, output=output)
        fallback_reason = self._validate_output(pred, log_inv_depth_low, output, fixed_frames=fixed_frames)
        if fallback_reason is not None:
            return pred, self._stats(success=False, reason=fallback_reason, graph_metrics=graph_metrics, output=output)

        initial = float(output.debug.get("initial_mean_angular_residual_deg", output.mean_angular_residual_deg))
        if (
            self.config.dense_ba.fallback_if_residual_worse
            and output.mean_angular_residual_deg > initial * float(self.config.dense_ba.residual_worse_tolerance)
        ):
            return pred, self._stats(success=False, reason="residual_worse", graph_metrics=graph_metrics, output=output)

        refined = self._rebuild_prediction(pred, log_inv_depth_low, output)
        stats = self._stats(success=True, reason=None, graph_metrics=graph_metrics, output=output)
        refined = replace(
            refined,
            matching_debug={
                **(pred.matching_debug or {}),
                **graph_metrics,
                **stats.as_debug(),
            },
        )
        _ = keyframe_memory
        return refined, stats

    def _missing_prediction_fields(self, pred: PanoVGGTLocalPrediction) -> str | None:
        required = {
            "dense_descriptors": pred.dense_descriptors,
            "match_confidence": pred.match_confidence,
            "sky_prob": pred.sky_prob,
            "feature_hw": pred.feature_hw,
            "image_hw": pred.image_hw,
        }
        for name, value in required.items():
            if value is None:
                return name
        return None

    def _build_factor_graph(self, pred: PanoVGGTLocalPrediction) -> DenseSphereFactorGraph | None:
        if pred.dense_descriptors is None or pred.match_confidence is None or pred.sky_prob is None or pred.feature_hw is None or pred.image_hw is None:
            return None
        dense = self.config.dense_matching
        matcher = PoseGuidedDenseMatcher(
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
        edges = DenseSphereFactorGraph.build_edges(
            int(pred.poses_c2w.shape[0]),
            temporal_radius=self.config.inference_window.temporal_radius,
            max_edges=self.config.inference_window.max_edges,
            device=pred.depth.device,
        )
        return matcher.match(
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

    def _fixed_frames(self, frame_ids: tuple[int, ...]) -> int:
        fixed = int(self.config.dense_ba.fixed_frames)
        if self.config.dense_ba.fixed_policy == "first_frame":
            return max(1, min(fixed, len(frame_ids)))
        return max(1, min(fixed, len(frame_ids)))

    def _validate_output(
        self,
        pred: PanoVGGTLocalPrediction,
        log_inv_depth_low: torch.Tensor,
        output: SphericalTangentDenseBAOutput,
        *,
        fixed_frames: int,
    ) -> str | None:
        if not torch.isfinite(output.poses_c2w).all() or not torch.isfinite(output.inverse_depth).all():
            return "non_finite_outputs"
        if not torch.allclose(output.poses_c2w[:fixed_frames], pred.poses_c2w[:fixed_frames].to(output.poses_c2w), atol=1.0e-5):
            return "fixed_frame_changed"
        log_delta = output.log_inv_depth - log_inv_depth_low.to(output.log_inv_depth)
        if torch.isfinite(log_delta).all() and float(log_delta.abs().max().detach().cpu()) > float(self.config.dense_ba.max_logdepth_update) + 1.0e-5:
            return "depth_update_too_large"
        if output.pose_update_norm.get("rot_max_deg", 0.0) > float(self.config.dense_ba.max_pose_update_deg) + 0.2:
            return "pose_update_too_large"
        return None

    def _rebuild_prediction(
        self,
        pred: PanoVGGTLocalPrediction,
        log_inv_depth_low: torch.Tensor,
        output: SphericalTangentDenseBAOutput,
    ) -> PanoVGGTLocalPrediction:
        log_delta_low = output.log_inv_depth - log_inv_depth_low.to(output.log_inv_depth)
        log_delta_full = F.interpolate(
            log_delta_low.float(),
            size=pred.depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_inv = pred.depth.float().clamp_min(1.0e-6).reciprocal() * torch.exp(log_delta_full).clamp_min(1.0e-6)
        refined_depth = refined_inv.clamp_min(1.0e-6).reciprocal()
        local_points = _build_local_points(refined_depth)
        chunk_world_points = _local_points_to_world(local_points, output.poses_c2w.to(refined_depth))
        return replace(
            pred,
            poses_c2w=output.poses_c2w.to(pred.poses_c2w),
            depth=refined_depth.to(pred.depth),
            local_points=local_points.to(pred.depth),
            chunk_world_points=chunk_world_points.to(pred.depth),
            ba_residual_angular=float(math.radians(output.mean_angular_residual_deg)),
            ba_valid_ratio=float(output.valid_mask.float().mean().detach().cpu()) if output.valid_mask.numel() else 0.0,
            ba_update_norm={
                "pose_mean": float(output.pose_update_norm.get("mean", 0.0)),
                "pose_max": float(output.pose_update_norm.get("max", 0.0)),
                "depth_mean": float(output.depth_update_norm.get("mean", 0.0)),
                "depth_max": float(output.depth_update_norm.get("max", 0.0)),
            },
        )

    def _stats(
        self,
        *,
        enabled: bool | None = None,
        success: bool,
        reason: str | None,
        graph_metrics: dict[str, float] | None = None,
        output: SphericalTangentDenseBAOutput | None = None,
    ) -> DenseBARefinerStats:
        metrics = graph_metrics or {}
        initial = 0.0
        if output is not None:
            initial = float(output.debug.get("initial_mean_angular_residual_deg", 0.0))
        return DenseBARefinerStats(
            enabled=self.enabled if enabled is None else bool(enabled),
            shadow_mode=self.shadow_mode,
            success=bool(success),
            used_refined=bool(success and not self.shadow_mode),
            fallback_reason=reason,
            mean_residual_deg=0.0 if output is None else float(output.mean_angular_residual_deg),
            median_residual_deg=0.0 if output is None else float(output.median_angular_residual_deg),
            initial_mean_residual_deg=initial,
            valid_factor_ratio=float(metrics.get("valid_factor_ratio", 0.0)),
            num_factors=int(metrics.get("num_factors", 0.0)),
            valid_factors=int(metrics.get("valid_factors", 0.0)),
            pose_update_norm={} if output is None else dict(output.pose_update_norm),
            depth_update_norm={} if output is None else dict(output.depth_update_norm),
        )


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
