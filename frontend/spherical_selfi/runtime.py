"""Config-gated streaming Stage-2 spherical-Selfi frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from geometry.pose import relative_c2w

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image, identity_pose
from frontend.pano_vggt.matching_adapter import (
    extract_features_with_hook,
    load_matching_sky_checkpoint,
    run_matching_sky_head,
)
from models.spherical_selfi_stage3_ba import BlockSparseSphericalBA, build_stage3_match_cache
from models.spherical_selfi_gaussian_head import erp_bilinear_resize
from training.train_spherical_selfi_gaussian_head import (
    build_frozen_feature_stack,
    build_head,
    extract_frozen_inputs,
    load_stage2_checkpoint,
)

from .window_packet import LocalGaussianWindowPacket, LocalGaussianWindowQueue


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested} requested for spherical-Selfi runtime but CUDA is unavailable")
    return requested


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

    def _run_local_ba(self, observation, dense_features, static_valid_mask=None):
        if not self.local_ba_enabled:
            return observation, None, None
        cfg = self.local_ba_matching
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
            static_valid_mask=(
                observation.valid_mask
                if static_valid_mask is None
                else observation.valid_mask & static_valid_mask.bool()
            ),
        )
        result = self.local_ba(observation.poses_c2w, observation.refined_depth, cache)
        updated = observation.with_geometry(
            poses_c2w=result.poses_c2w,
            refined_depth=result.dense_depth,
        )
        return updated, cache, result

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
            ba_valid = None if sky_prob is None else sky_prob < self.sky_threshold
            observation, match_cache, ba_result = self._run_local_ba(
                observation, dense, static_valid_mask=ba_valid
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
            match_quality=match_quality,
            metadata={
                "local_ba_enabled": self.local_ba_enabled,
                "local_ba_accepted": None if ba_result is None else bool(ba_result.accepted[0]),
                "input_anchor_pose_c2w": poses[0, 0].detach().cpu(),
                "fibonacci": dict(self.fibonacci_config),
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
