"""Config-gated streaming Stage-2 spherical-Selfi frontend."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time
from typing import Any

import torch

from backend.pano_gs.adapter import PFGS360Renderer
from geometry.pose import relative_c2w

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image, identity_pose
from frontend.pano_vggt.matching_adapter import (
    extract_features_with_hook,
    load_matching_sky_checkpoint,
    run_matching_sky_head,
)
from models.spherical_selfi_stage3_ba import (
    BlockSparseSphericalBA,
    Stage3MatchCache,
    build_stage3_match_cache,
    filter_stage3_match_cache_robust,
)
from models.sphereglue_local_ba import SphereGlueLocalBAMatcher
from models.spherical_voxel_anchor_refiner import (
    VoxelAnchorObservation,
    VoxelAnchorConfig,
    VoxelAnchorStage3Model,
    load_voxel_anchor_checkpoint,
    render_voxel_anchor_group,
    voxelize_per_pixel_gaussians,
)
from models.spherical_selfi_gaussian_head import erp_bilinear_resize
from training.train_spherical_selfi_gaussian_head import (
    build_frozen_feature_stack,
    build_head,
    extract_frozen_inputs,
    load_stage2_checkpoint,
)

from .window_packet import BoundaryMatchBlock, LocalGaussianWindowPacket, LocalGaussianWindowQueue


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested} requested for spherical-Selfi runtime but CUDA is unavailable")
    return requested


def _finite_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return scalar if math.isfinite(scalar) else None


def _boundary_matches_from_cache(
    cache: Stage3MatchCache | None,
    image_size: tuple[int, int],
) -> BoundaryMatchBlock | None:
    """Extract first/last matches and canonicalize both directions."""

    if cache is None or cache.batch_size != 1 or cache.num_views < 2:
        return None
    last = cache.num_views - 1
    height, width = (int(value) for value in image_size)
    entropy_scale = max(math.log(max(2, height * width)), 1.0e-8)
    values: dict[str, list[torch.Tensor]] = {
        "source_uv": [],
        "target_uv": [],
        "source_bearing": [],
        "target_bearing": [],
        "top1_cosine": [],
        "top2_margin": [],
        "normalized_entropy": [],
    }
    for edge_index, pair in enumerate(cache.edges.detach().cpu().tolist()):
        source_index, target_index = int(pair[0]), int(pair[1])
        if (source_index, target_index) not in {(0, last), (last, 0)}:
            continue
        keep = cache.valid_mask[0, edge_index].bool()
        if not bool(keep.any()):
            continue
        if source_index == 0:
            source_uv = cache.source_uv[0, 0, keep]
            target_uv = cache.target_uv[0, edge_index, keep]
            source_bearing = cache.source_ray[0, 0, keep]
            target_bearing = cache.target_ray[0, edge_index, keep]
        else:
            # Reverse queries are last->first. Swap them into the canonical
            # first->last direction before publishing the private packet.
            source_uv = cache.target_uv[0, edge_index, keep]
            target_uv = cache.source_uv[0, last, keep]
            source_bearing = cache.target_ray[0, edge_index, keep]
            target_bearing = cache.source_ray[0, last, keep]
        values["source_uv"].append(source_uv)
        values["target_uv"].append(target_uv)
        values["source_bearing"].append(source_bearing)
        values["target_bearing"].append(target_bearing)
        values["top1_cosine"].append(cache.top1_cosine[0, edge_index, keep])
        values["top2_margin"].append(cache.top2_margin[0, edge_index, keep])
        values["normalized_entropy"].append(
            (cache.entropy[0, edge_index, keep] / entropy_scale).clamp(0.0, 1.0)
        )
    if not values["source_uv"]:
        return None
    with torch.inference_mode(False):
        return BoundaryMatchBlock(
            **{
                name: torch.cat(parts, dim=0).detach().clone()
                for name, parts in values.items()
            }
        )


class SphericalSelfiWindowFrontend(PanoDROIDFrontend, LocalGaussianWindowQueue):
    def __init__(self, config: dict[str, Any]) -> None:
        LocalGaussianWindowQueue.__init__(self)
        self.config = config
        runtime = dict(config.get("SphericalSelfiRuntime", {}) or {})
        if not bool(runtime.get("enabled", False)):
            raise ValueError("SphericalSelfiRuntime.enabled must be true for this frontend mode")
        window_cfg = dict(runtime.get("window", {}) or {})
        self.window_size = max(2, int(window_cfg.get("size", 4)))
        self.window_stride = max(1, min(self.window_size, int(window_cfg.get("stride", 3))))
        self.expected_overlap = int(window_cfg.get("expected_overlap_frames", self.window_size - self.window_stride))
        if bool(window_cfg.get("enforce_exact_overlap", False)) and self.window_size - self.window_stride != self.expected_overlap:
            raise ValueError(
                "SphericalSelfiRuntime.window requires size-stride == expected_overlap_frames; "
                f"got {self.window_size}-{self.window_stride}!={self.expected_overlap}."
            )
        self.verification_size = tuple(int(v) for v in window_cfg.get("verification_size", (32, 64)))
        self.latitude_bands = max(1, int(window_cfg.get("latitude_bands", 8)))
        self.feature_device = _device(runtime.get("feature_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.head_device = _device(runtime.get("head_device", str(self.feature_device)))
        self.feature_amp = bool(runtime.get("feature_amp", False))
        image_cfg = dict(config.get("image", {}) or {})
        self.head_size = (
            int(image_cfg.get("head_height", image_cfg.get("height", 504))),
            int(image_cfg.get("head_width", image_cfg.get("width", 1008))),
        )
        self.wrapper, self.adapter, self.adapter_sha, _ = build_frozen_feature_stack(
            config, device=self.feature_device
        )
        self.head = build_head(config, device=self.head_device)
        checkpoint_cfg = dict(config.get("stage2_checkpoint", {}) or {})
        checkpoint = checkpoint_cfg.get("path")
        if not checkpoint:
            raise ValueError("stage2_checkpoint.path is required for spherical-Selfi runtime")
        load_stage2_checkpoint(
            checkpoint,
            head=self.head,
            expected_adapter_sha256=self.adapter_sha,
            map_location=self.head_device,
        )
        self.wrapper.eval()
        self.adapter.eval()
        self.head.eval()

        voxel_cfg = dict(config.get("VoxelAnchorRefiner", {}) or {})
        self.voxel_anchor_enabled = bool(voxel_cfg.get("enabled", False))
        self.voxel_anchor_model: VoxelAnchorStage3Model | None = None
        self.voxel_anchor_renderer: PFGS360Renderer | None = None
        self.voxel_anchor_config: VoxelAnchorConfig | None = None
        if self.voxel_anchor_enabled:
            checkpoint = voxel_cfg.get("checkpoint")
            if not checkpoint:
                raise ValueError(
                    "VoxelAnchorRefiner.checkpoint is required when VoxelAnchorRefiner.enabled=true"
                )
            self.voxel_anchor_config = VoxelAnchorConfig.from_mapping(voxel_cfg)
            self.voxel_anchor_model = VoxelAnchorStage3Model(self.voxel_anchor_config).to(
                self.head_device
            )
            load_voxel_anchor_checkpoint(
                str(checkpoint),
                model=self.voxel_anchor_model,
                map_location=self.head_device,
            )
            self.voxel_anchor_model.eval()
            renderer_cfg = dict(config.get("renderer", {}) or {})
            self.voxel_anchor_renderer = PFGS360Renderer(
                config=config,
                extra_gsplat360_roots=list(
                    renderer_cfg.get("extra_gsplat360_roots", []) or []
                ),
                allow_fallback=False,
            )

        sky_cfg = dict(runtime.get("sky", {}) or {})
        self.sky_enabled = bool(sky_cfg.get("enabled", False))
        self.sky_required = bool(sky_cfg.get("required", self.sky_enabled))
        self.sky_threshold = float(sky_cfg.get("threshold", 0.5))
        self.sky_adapter = None
        self.sky_feature_hook: str | None = None
        self.sky_feature_key: str | int | None = None
        self.sky_patch_size = int(sky_cfg.get("patch_size", 14))
        self.fibonacci_config = dict(runtime.get("fibonacci", {}) or {})
        if self.sky_enabled:
            checkpoint = sky_cfg.get("checkpoint")
            if not checkpoint:
                raise ValueError("SphericalSelfiRuntime.sky.checkpoint is required when sky.enabled=true")
            self.sky_adapter = load_matching_sky_checkpoint(
                sky_checkpoint=checkpoint,
                device=self.head_device,
                descriptor_dim=sky_cfg.get("descriptor_dim"),
                feature_hook=sky_cfg.get("feature_hook"),
                feature_key=sky_cfg.get("feature_key"),
                strict=bool(sky_cfg.get("strict", True)),
            )
            if self.sky_required and not self.sky_adapter.has_sky:
                raise ValueError("Configured spherical-Selfi sky checkpoint does not contain a sky head")
            self.sky_feature_hook = self.sky_adapter.feature_hook
            self.sky_feature_key = self.sky_adapter.head.feature_key
            if not self.sky_feature_hook:
                raise ValueError("Sky checkpoint must provide feature_hook, or sky.feature_hook must be configured")

        local_ba = dict(runtime.get("local_ba", {}) or {})
        self.local_ba_enabled = bool(local_ba.get("enabled", False))
        self.local_ba_matching = dict(local_ba.get("matching", {}) or {})
        self.local_ba_matcher_name = str(
            self.local_ba_matching.get("type", "adapter")
        ).strip().lower()
        if self.local_ba_matcher_name not in {"adapter", "superpoint_sphereglue"}:
            raise ValueError(
                "SphericalSelfiRuntime.local_ba.matching.type must be "
                "'adapter' or 'superpoint_sphereglue'."
            )
        self.sphereglue_local_ba_matcher = None
        if self.local_ba_enabled and self.local_ba_matcher_name == "superpoint_sphereglue":
            self.sphereglue_local_ba_matcher = SphereGlueLocalBAMatcher(
                self.local_ba_matching,
                device=self.head_device,
            )
        self.local_ba = BlockSparseSphericalBA(
            iterations=int(local_ba.get("iterations", 3)),
            damping=float(local_ba.get("damping", 1.0e-4)),
            huber_delta_deg=float(local_ba.get("huber_delta_deg", 0.5)),
            pose_prior_weight=float(local_ba.get("pose_prior_weight", 1.0e-3)),
            depth_prior_weight=float(local_ba.get("depth_prior_weight", 1.0e-2)),
            max_pose_update_deg=float(local_ba.get("max_pose_update_deg", 5.0)),
            max_translation_update=float(local_ba.get("max_translation_update", 0.05)),
            max_logdepth_update=float(local_ba.get("max_logdepth_update", 0.35)),
            factor_chunk_size=int(local_ba.get("factor_chunk_size", 2048)),
            min_factors=int(local_ba.get("min_factors", 256)),
            residual_worse_tolerance=1.0,
            min_affine_support=int(local_ba.get("min_affine_support", 64)),
            min_depth=float(local_ba.get("min_depth", 0.05)),
            max_depth=float(local_ba.get("max_depth", 20.0)),
            solver_mode=str(local_ba.get("solver_mode", "backtracking_gn")),
            dense_depth_mode=str(local_ba.get("dense_depth_mode", "affine")),
            gauge_mode=str(local_ba.get("gauge_mode", "none")),
            lm_max_trials=int(local_ba.get("lm_max_trials", 4)),
            lm_acceptance_eta=float(local_ba.get("lm_acceptance_eta", 1.0e-4)),
            lm_damping_min=float(local_ba.get("lm_damping_min", 1.0e-8)),
            lm_damping_max=float(local_ba.get("lm_damping_max", 1.0e8)),
            lm_diagonal_floor=float(local_ba.get("lm_diagonal_floor", 1.0e-6)),
            max_initial_residual_deg=local_ba.get("max_initial_residual_deg"),
            min_parallax_deg=float(local_ba.get("min_parallax_deg", 0.0)),
            pose_update_side=str(local_ba.get("pose_update_side", "left")),
            pose_dof_mode=str(local_ba.get("pose_dof_mode", "se3")),
            min_initial_median_residual_deg=float(
                local_ba.get("min_initial_median_residual_deg", 0.0)
            ),
            jacobian_mode=str(local_ba.get("jacobian_mode", "autodiff_reference")),
            validate_analytic_jacobian=bool(
                local_ba.get("validate_analytic_jacobian", False)
            ),
            analytic_jacobian_atol=float(local_ba.get("analytic_jacobian_atol", 1.0e-5)),
            analytic_jacobian_rtol=float(local_ba.get("analytic_jacobian_rtol", 1.0e-4)),
            gradient_tolerance=float(local_ba.get("gradient_tolerance", 1.0e-8)),
            step_tolerance=float(local_ba.get("step_tolerance", 1.0e-8)),
            relative_objective_tolerance=float(
                local_ba.get("relative_objective_tolerance", 1.0e-6)
            ),
            affine_min_relative_improvement=float(
                local_ba.get("affine_min_relative_improvement", 1.0e-3)
            ),
        )
        self.local_ba_outlier_config = dict(local_ba.get("outlier_refinement", {}) or {})
        self.local_ba_outlier_enabled = bool(
            self.local_ba_outlier_config.get("enabled", False)
        )
        self.local_ba_second = copy.copy(self.local_ba)
        self.local_ba_second.iterations = int(
            self.local_ba_outlier_config.get("second_stage_iterations", 10)
        )
        self.frames: list[PanoFrame] = []
        self.frame_buffer_start = 0
        self.next_window_start = 0
        self.window_index = 0
        self.ready_outputs: list[FrontendOutput] = []
        self.pending_outputs: dict[int, FrontendOutput] = {}
        self.emitted_frame_ids: set[int] = set()
        self.sky_prob_by_frame: dict[int, torch.Tensor] = {}
        self.sky_mask_by_frame: dict[int, torch.Tensor] = {}
        self._local_ba_diagnostics: list[dict[str, Any]] = []
        self.last_processed_frame_id: int | None = None

    def initialize(self, sequence_meta: dict) -> None:
        _ = sequence_meta

    def reset(self) -> None:
        self.frames.clear()
        self.frame_buffer_start = 0
        self.next_window_start = 0
        self.window_index = 0
        self.ready_outputs.clear()
        self.pending_outputs.clear()
        self.emitted_frame_ids.clear()
        self.sky_prob_by_frame.clear()
        self.sky_mask_by_frame.clear()
        self._local_ba_diagnostics.clear()
        self._local_gaussian_windows.clear()
        self.last_processed_frame_id = None

    def load_checkpoint(self, path: str) -> None:
        load_stage2_checkpoint(
            Path(path),
            head=self.head,
            expected_adapter_sha256=self.adapter_sha,
            map_location=self.head_device,
        )
        self.head.eval()

    @staticmethod
    def _pending_output(frame: PanoFrame) -> FrontendOutput:
        return FrontendOutput(
            frame_id=int(frame.frame_id),
            timestamp=float(frame.timestamp),
            pose_c2w=identity_pose(),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=False,
            ba_residual=None,
            tracking_status="pending_spherical_selfi_window",
        )

    def _run_local_ba(
        self,
        observation,
        dense_features,
        images,
        static_valid_mask=None,
    ):
        if not self.local_ba_enabled:
            return observation, None, None, 0.0, 0.0
        cfg = self.local_ba_matching
        combined_valid = (
            observation.valid_mask
            if static_valid_mask is None
            else observation.valid_mask & static_valid_mask.bool()
        )
        if self.head_device.type == "cuda":
            torch.cuda.synchronize(self.head_device)
        matching_start = time.perf_counter()
        with torch.no_grad():
            if self.local_ba_matcher_name == "superpoint_sphereglue":
                if self.sphereglue_local_ba_matcher is None:
                    raise RuntimeError("SphereGlue local BA matcher was not initialized")
                cache = self.sphereglue_local_ba_matcher.build_cache(
                    images,
                    observation.refined_depth,
                    static_valid_mask=combined_valid,
                )
            else:
                fibonacci_seed = int(self.fibonacci_config.get("seed", 123)) + int(
                    self.window_index
                )
                generator = torch.Generator(device=self.head_device)
                generator.manual_seed(fibonacci_seed)
                cache = build_stage3_match_cache(
                    dense_features,
                    observation.refined_depth,
                    num_queries=int(cfg.get("num_queries", 2048)),
                    min_depth=float(cfg.get("min_depth", 0.05)),
                    max_depth=float(cfg.get("max_depth", 20.0)),
                    temperature=float(cfg.get("temperature", 0.07)),
                    query_chunk_size=int(cfg.get("query_chunk_size", 32)),
                    fibonacci_oversample_factor=int(cfg.get("fibonacci_oversample_factor", 8)),
                    use_spherical_area_correction=bool(cfg.get("use_spherical_area_correction", True)),
                    forward_backward=bool(cfg.get("forward_backward", True)),
                    fb_tolerance_deg=float(cfg.get("fb_tolerance_deg", 1.0)),
                    min_factor_weight=float(cfg.get("min_factor_weight", 0.01)),
                    factor_weight_mode=str(
                        cfg.get("factor_weight_mode", "descriptor_confidence")
                    ),
                    static_valid_mask=combined_valid,
                    generator=generator,
                )
                cache.metadata["fibonacci_seed"] = fibonacci_seed
        if self.head_device.type == "cuda":
            torch.cuda.synchronize(self.head_device)
        matching_sec = float(time.perf_counter() - matching_start)
        ba_start = time.perf_counter()
        with torch.inference_mode(False):
            ba_poses = observation.poses_c2w.detach().clone()
            ba_depth = observation.refined_depth.detach().clone()
            ba_cache = cache.detached_clone()
            if self.local_ba.jacobian_mode == "autodiff_reference":
                with torch.enable_grad():
                    stage1_result = self.local_ba(ba_poses, ba_depth, ba_cache)
            else:
                with torch.no_grad():
                    stage1_result = self.local_ba(ba_poses, ba_depth, ba_cache)
            result = stage1_result
            published_cache = cache
            if self.local_ba_outlier_enabled:
                filter_cfg = self.local_ba_outlier_config
                filtered_cache, filter_diagnostics = filter_stage3_match_cache_robust(
                    ba_cache,
                    stage1_result.poses_c2w.detach(),
                    stage1_result.dense_depth.detach(),
                    angular_mad_scale=float(filter_cfg.get("angular_mad_scale", 3.0)),
                    angular_min_deg=float(filter_cfg.get("angular_min_deg", 1.0)),
                    angular_max_deg=float(filter_cfg.get("angular_max_deg", 5.0)),
                    sim3_irls_iterations=int(filter_cfg.get("sim3_irls_iterations", 3)),
                    sim3_mad_scale=float(filter_cfg.get("sim3_mad_scale", 3.0)),
                    sim3_min_residual=float(filter_cfg.get("sim3_min_residual", 0.01)),
                    sim3_max_relative_depth=float(
                        filter_cfg.get("sim3_max_relative_depth", 0.05)
                    ),
                )
                min_inliers = max(
                    self.local_ba_second.min_factors,
                    int(filter_cfg.get("min_inliers", self.local_ba_second.min_factors)),
                )
                min_ratio = float(filter_cfg.get("min_inlier_ratio", 0.3))
                stage2_supported = all(
                    int(value["post_filter_inliers"]) >= min_inliers
                    and float(value["post_filter_inlier_ratio"]) >= min_ratio
                    for value in filter_diagnostics
                )
                if stage2_supported:
                    published_cache = filtered_cache
                    if self.local_ba_second.jacobian_mode == "autodiff_reference":
                        with torch.enable_grad():
                            stage2_result = self.local_ba_second(
                                stage1_result.poses_c2w.detach().clone(),
                                stage1_result.dense_depth.detach().clone(),
                                filtered_cache,
                            )
                    else:
                        with torch.no_grad():
                            stage2_result = self.local_ba_second(
                                stage1_result.poses_c2w.detach().clone(),
                                stage1_result.dense_depth.detach().clone(),
                                filtered_cache,
                            )
                    raw_stage2_accepted = stage2_result.accepted.clone()
                    stage2_result.accepted = stage1_result.accepted | raw_stage2_accepted
                    stage2_result.initial_median_residual_deg = (
                        stage1_result.initial_median_residual_deg
                    )
                    for batch_idx, filter_diag in enumerate(filter_diagnostics):
                        stage1_diag = dict(stage1_result.diagnostics[batch_idx])
                        stage2_diag = dict(stage2_result.diagnostics[batch_idx])
                        stage2_accepted = bool(raw_stage2_accepted[batch_idx])
                        combined = dict(stage2_diag)
                        combined.update(filter_diag)
                        combined.update(
                            {
                                "stage1_iterations": int(self.local_ba.iterations),
                                "stage1_accepted": bool(stage1_result.accepted[batch_idx]),
                                "stage1_reason": stage1_diag.get("reason"),
                                "stage2_attempted": True,
                                "stage2_iterations": int(self.local_ba_second.iterations),
                                "stage2_accepted": stage2_accepted,
                                "stage2_reason": stage2_diag.get("reason"),
                                "stage2_min_inliers": min_inliers,
                                "stage2_min_inlier_ratio": min_ratio,
                                "accepted_steps": int(stage1_diag.get("accepted_steps", 0))
                                + int(stage2_diag.get("accepted_steps", 0)),
                                "gradient_norms": list(stage1_diag.get("gradient_norms", []))
                                + list(stage2_diag.get("gradient_norms", [])),
                                "pose_step_norms": list(stage1_diag.get("pose_step_norms", []))
                                + list(stage2_diag.get("pose_step_norms", [])),
                                "depth_step_norms": list(stage1_diag.get("depth_step_norms", []))
                                + list(stage2_diag.get("depth_step_norms", [])),
                                "trial_gain_ratios": list(
                                    stage1_diag.get("trial_gain_ratios", [])
                                )
                                + list(stage2_diag.get("trial_gain_ratios", [])),
                                "published_pose_updated": bool(
                                    stage1_diag.get("published_pose_updated", False)
                                    or stage2_diag.get("published_pose_updated", False)
                                ),
                            }
                        )
                        if stage2_accepted:
                            combined["reason"] = "accepted_two_stage"
                        elif bool(stage1_result.accepted[batch_idx]):
                            combined["reason"] = "stage2_rejected_stage1_retained"
                        stage2_result.diagnostics[batch_idx] = combined
                    result = stage2_result
                else:
                    for batch_idx, filter_diag in enumerate(filter_diagnostics):
                        stage1_result.diagnostics[batch_idx].update(filter_diag)
                        stage1_result.diagnostics[batch_idx].update(
                            {
                                "stage1_iterations": int(self.local_ba.iterations),
                                "stage1_accepted": bool(stage1_result.accepted[batch_idx]),
                                "stage1_reason": stage1_result.diagnostics[batch_idx].get(
                                    "reason"
                                ),
                                "stage2_attempted": False,
                                "stage2_iterations": int(self.local_ba_second.iterations),
                                "stage2_accepted": False,
                                "stage2_reason": "insufficient_post_filter_inliers",
                                "stage2_min_inliers": min_inliers,
                                "stage2_min_inlier_ratio": min_ratio,
                                "reason": "insufficient_post_filter_inliers_stage1_retained",
                            }
                        )
        if self.head_device.type == "cuda":
            torch.cuda.synchronize(self.head_device)
        ba_sec = float(time.perf_counter() - ba_start)
        updated = observation.with_geometry(
            poses_c2w=result.poses_c2w.detach(),
            refined_depth=result.dense_depth.detach(),
        )
        return updated, published_cache, result, matching_sec, ba_sec

    def _run_voxel_anchor_refiner(
        self,
        observation,
        adapter_features: torch.Tensor,
        images: torch.Tensor,
        sky_prob: torch.Tensor | None,
    ) -> VoxelAnchorObservation | None:
        if not self.voxel_anchor_enabled:
            return None
        assert self.voxel_anchor_config is not None
        assert self.voxel_anchor_model is not None
        assert self.voxel_anchor_renderer is not None

        target_images = images.to(self.head_device)
        if tuple(target_images.shape[-2:]) != tuple(observation.image_size):
            batch, views = int(target_images.shape[0]), int(target_images.shape[1])
            target_images = erp_bilinear_resize(
                target_images.reshape(batch * views, 3, *target_images.shape[-2:]),
                observation.image_size,
            ).reshape(batch, views, 3, *observation.image_size)
        target_valid = observation.valid_mask.bool()
        if sky_prob is not None:
            resized_sky = sky_prob.to(target_valid.device)
            if tuple(resized_sky.shape[-2:]) != tuple(observation.image_size):
                batch, views = int(resized_sky.shape[0]), int(resized_sky.shape[1])
                resized_sky = erp_bilinear_resize(
                    resized_sky.reshape(batch * views, 1, *resized_sky.shape[-2:]),
                    observation.image_size,
                ).reshape(batch, views, 1, *observation.image_size)
            target_valid = target_valid & (resized_sky < self.sky_threshold)

        with torch.inference_mode():
            current = voxelize_per_pixel_gaussians(
                observation,
                adapter_features.to(self.head_device),
                target_images,
                self.voxel_anchor_config,
                valid_mask=target_valid,
            )
            reference = self.voxel_anchor_model.encode_references(target_images)
            hidden = None
            for iteration in range(self.voxel_anchor_config.iterations):
                feedback = render_voxel_anchor_group(self.voxel_anchor_renderer, current)
                output = self.voxel_anchor_model.forward_step(
                    current,
                    feedback,
                    reference,
                    target_valid,
                    iteration_index=iteration,
                    hidden=hidden,
                )
                current, hidden = output.observation, output.hidden
                if iteration < self.voxel_anchor_config.iterations - 1:
                    current = current.detach_parameters()
                    hidden = hidden.detach()
        return current.detach_for_backend()

    def _run_window(self, frames: list[PanoFrame]) -> None:
        images = torch.stack([ensure_chw_image(frame.image).float() for frame in frames], dim=0).unsqueeze(0)
        frame_ids = torch.tensor([[int(frame.frame_id) for frame in frames]], device=self.head_device)
        with torch.inference_mode():
            def frozen_forward():
                return extract_frozen_inputs(
                    self.wrapper,
                    self.adapter,
                    images,
                    feature_device=self.feature_device,
                    train_device=self.head_device,
                    head_size=self.head_size,
                    feature_amp=self.feature_amp,
                )

            captured_feature = None
            if self.sky_adapter is not None:
                frozen_inputs, captured_feature = extract_features_with_hook(
                    self.wrapper.model,
                    str(self.sky_feature_hook),
                    frozen_forward,
                    patch_size=self.sky_patch_size,
                    feature_key=self.sky_feature_key,
                )
                dense, rgb, depth, poses = frozen_inputs
            else:
                dense, rgb, depth, poses = frozen_forward()
            observation = self.head(
                dense,
                rgb,
                depth,
                poses,
                frame_ids=frame_ids,
            )
            sky_prob = None
            if self.sky_adapter is not None:
                sky_output = run_matching_sky_head(
                    self.sky_adapter,
                    captured_feature.to(self.sky_adapter.device),
                )
                if "sky_prob" not in sky_output:
                    if self.sky_required:
                        raise RuntimeError("Sky head did not return sky_prob")
                else:
                    feature_sky = sky_output["sky_prob"]
                    if not torch.is_tensor(feature_sky):
                        raise TypeError("Sky head sky_prob must be a tensor")
                    views = int(feature_sky.shape[0])
                    sky_prob = erp_bilinear_resize(
                        feature_sky.to(self.head_device),
                        observation.image_size,
                    ).reshape(1, views, 1, *observation.image_size)
        initial_poses_c2w = observation.poses_c2w.detach().cpu().float().clone()
        ba_valid = None if sky_prob is None else sky_prob < self.sky_threshold
        observation, match_cache, ba_result, matching_sec, ba_sec = self._run_local_ba(
            observation,
            dense,
            images,
            static_valid_mask=ba_valid,
        )
        anchor_observation = self._run_voxel_anchor_refiner(
            observation,
            dense,
            images,
            sky_prob,
        )

        gt_values = [
            None
            if frame.meta is None or frame.meta.get("gt_c2w") is None
            else torch.as_tensor(frame.meta["gt_c2w"]).detach().cpu().float()
            for frame in frames
        ]
        gt_poses_c2w = (
            torch.stack([value for value in gt_values if value is not None], dim=0)
            if all(value is not None for value in gt_values)
            else None
        )
        self._local_ba_diagnostics.append(
            {
                "window_id": int(self.window_index),
                "frame_ids": tuple(int(frame.frame_id) for frame in frames),
                "matcher": (
                    self.local_ba_matcher_name if self.local_ba_enabled else "none"
                ),
                "initial_poses_c2w": initial_poses_c2w[0],
                "refined_poses_c2w": observation.poses_c2w[0].detach().cpu().float(),
                "gt_poses_c2w": gt_poses_c2w,
                "accepted": False if ba_result is None else bool(ba_result.accepted[0]),
                "initial_median_residual_deg": (
                    None
                    if ba_result is None
                    else _finite_optional_float(ba_result.initial_median_residual_deg[0])
                ),
                "final_median_residual_deg": (
                    None
                    if ba_result is None
                    else _finite_optional_float(ba_result.final_median_residual_deg[0])
                ),
                "num_factors": 0 if match_cache is None else int(match_cache.num_factors),
                "matching_sec": float(matching_sec),
                "ba_sec": float(ba_sec),
                "ba_diagnostics": None if ba_result is None else dict(ba_result.diagnostics[0]),
                "matching_metadata": None if match_cache is None else dict(match_cache.metadata),
            }
        )

        match_quality = {}
        if match_cache is not None:
            match_quality = {
                "top1_cosine": match_cache.top1_cosine.detach(),
                "top2_margin": match_cache.top2_margin.detach(),
                "entropy": match_cache.entropy.detach(),
                "factor_weight": match_cache.factor_weight.detach() if match_cache.factor_weight is not None else torch.ones_like(match_cache.entropy),
            }
        packet = LocalGaussianWindowPacket.from_observation(
            window_id=self.window_index,
            observation=observation,
            adapter_features=dense,
            frame_ids=[int(frame.frame_id) for frame in frames],
            verification_size=self.verification_size,
            latitude_bands=self.latitude_bands,
            sky_prob=sky_prob,
            sky_threshold=self.sky_threshold,
            anchor_observation=anchor_observation,
            boundary_matches=_boundary_matches_from_cache(
                match_cache, observation.image_size
            ),
            match_quality=match_quality,
            metadata={
                "local_ba_enabled": self.local_ba_enabled,
                "local_ba_matcher": self.local_ba_matcher_name,
                "local_ba_accepted": None if ba_result is None else bool(ba_result.accepted[0]),
                "input_anchor_pose_c2w": poses[0, 0].detach().cpu(),
                "fibonacci": dict(self.fibonacci_config),
                "voxel_anchor_refiner_enabled": self.voxel_anchor_enabled,
                "voxel_anchor_count": (
                    0 if anchor_observation is None else anchor_observation.num_anchors
                ),
            },
        )
        self.enqueue_local_gaussian_window(packet)
        for index, frame in enumerate(frames):
            if int(frame.frame_id) not in self.emitted_frame_ids:
                self.sky_prob_by_frame[int(frame.frame_id)] = packet.sky_prob[0, index].detach().cpu().float()
                self.sky_mask_by_frame[int(frame.frame_id)] = packet.sky_mask[0, index].detach().cpu().bool()
        world_points = observation.centers_world()[0]
        for index, frame in enumerate(frames):
            pose = observation.poses_c2w[0, index].detach().cpu().float()
            inverse_depth = observation.refined_depth[0, index].detach().cpu().float().clamp_min(1.0e-6).reciprocal()
            confidence = observation.confidence[0, index].detach().cpu().float()
            valid = packet.finite_gaussian_mask[0, index].detach().cpu().bool()
            previous_pose = None
            if index > 0:
                previous_pose = relative_c2w(
                    observation.poses_c2w[0, index - 1],
                    observation.poses_c2w[0, index],
                )
                previous_pose = previous_pose.detach().cpu().float()
            residual = None
            if ba_result is not None:
                residual = float(ba_result.final_median_residual_deg[0].detach().cpu())
            self.pending_outputs[int(frame.frame_id)] = FrontendOutput(
                frame_id=int(frame.frame_id),
                timestamp=float(frame.timestamp),
                pose_c2w=pose,
                relative_pose=previous_pose,
                pose_confidence=float(confidence.mean()),
                inverse_depth=inverse_depth,
                depth_confidence=confidence,
                spherical_flow=None,
                keyframe_score=float(1.0 - confidence.mean()),
                is_keyframe=True,
                ba_residual=residual,
                tracking_status=(
                    "tracked_spherical_selfi_stage2_ba"
                    if self.local_ba_enabled and ba_result is not None and bool(ba_result.accepted[0])
                    else "tracked_spherical_selfi_stage2"
                ),
                world_points=world_points[index].detach().cpu().float(),
                world_points_confidence=confidence,
                valid_world_points_mask=valid,
            )
        self.window_index += 1
        self.last_processed_frame_id = int(frames[-1].frame_id)

    def _emit_oldest(self, count: int) -> None:
        candidates = sorted(
            (frame_id for frame_id in self.pending_outputs if frame_id not in self.emitted_frame_ids)
        )
        for frame_id in candidates[: max(0, int(count))]:
            self.ready_outputs.append(self.pending_outputs.pop(frame_id))
            self.emitted_frame_ids.add(frame_id)
        for frame_id in [value for value in self.pending_outputs if value in self.emitted_frame_ids]:
            self.pending_outputs.pop(frame_id, None)

    def track(self, frame: PanoFrame) -> FrontendOutput:
        self.frames.append(frame)
        while self.frame_buffer_start + len(self.frames) >= self.next_window_start + self.window_size:
            local_start = self.next_window_start - self.frame_buffer_start
            stop = local_start + self.window_size
            self._run_window(self.frames[local_start:stop])
            self.next_window_start += self.window_stride
            # Emit the first complete window in full.  Later windows naturally
            # emit only their three unseen frames because the overlap frame is
            # already marked emitted.  This registers all four RGB targets
            # before the backend starts that window's 20-step optimization.
            self._emit_oldest(self.window_size)
            prune = self.next_window_start - self.frame_buffer_start
            if prune > 0:
                del self.frames[:prune]
                self.frame_buffer_start = self.next_window_start
        pending = self.pending_outputs.get(int(frame.frame_id))
        if pending is not None:
            return pending
        for output in reversed(self.ready_outputs):
            if int(output.frame_id) == int(frame.frame_id):
                return output
        return self._pending_output(frame)

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        outputs = list(self.ready_outputs)
        self.ready_outputs.clear()
        return outputs

    def consume_local_ba_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = list(self._local_ba_diagnostics)
        self._local_ba_diagnostics.clear()
        return diagnostics

    def sky_mask_for_frame(
        self,
        frame_id: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        mask = self.sky_mask_by_frame.pop(int(frame_id), None)
        if mask is None:
            return None
        self.sky_prob_by_frame.pop(int(frame_id), None)
        output = mask.detach().cpu().bool()
        if image_size is not None and tuple(output.shape[-2:]) != tuple(int(v) for v in image_size):
            output = erp_bilinear_resize(
                output.float().unsqueeze(0), tuple(int(v) for v in image_size)
            )[0] >= 0.5
        return output

    def sky_probability_for_frame(
        self,
        frame_id: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        probability = self.sky_prob_by_frame.get(int(frame_id))
        if probability is None:
            return None
        output = probability.detach().cpu().float()
        if image_size is not None and tuple(output.shape[-2:]) != tuple(int(v) for v in image_size):
            output = erp_bilinear_resize(
                output.unsqueeze(0), tuple(int(v) for v in image_size)
            )[0]
        return output.clamp(0.0, 1.0)

    def flush(self) -> list[FrontendOutput]:
        absolute_end = self.frame_buffer_start + len(self.frames)
        remaining = absolute_end - self.next_window_start
        if remaining >= 2:
            local_start = self.next_window_start - self.frame_buffer_start
            partial = self.frames[local_start:]
            if self.last_processed_frame_id != int(partial[-1].frame_id):
                self._run_window(partial)
                self.next_window_start = absolute_end
                self.frames.clear()
                self.frame_buffer_start = absolute_end
        self._emit_oldest(len(self.pending_outputs))
        return self.pop_ready_outputs()


def build_spherical_selfi_frontend_from_config(config: dict[str, Any]) -> SphericalSelfiWindowFrontend:
    return SphericalSelfiWindowFrontend(config)
