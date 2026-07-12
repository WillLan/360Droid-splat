"""Config-gated streaming Stage-2 spherical-Selfi frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame, ensure_chw_image, identity_pose
from models.spherical_selfi_stage3_ba import BlockSparseSphericalBA, build_stage3_match_cache
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
        self.window_stride = max(1, min(self.window_size, int(window_cfg.get("stride", 2))))
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
        self.frames_since_window = 0
        self.window_index = 0
        self.ready_outputs: list[FrontendOutput] = []
        self.pending_outputs: dict[int, FrontendOutput] = {}
        self.emitted_frame_ids: set[int] = set()
        self.last_processed_frame_id: int | None = None

    def initialize(self, sequence_meta: dict) -> None:
        _ = sequence_meta

    def reset(self) -> None:
        self.frames.clear()
        self.frames_since_window = 0
        self.window_index = 0
        self.ready_outputs.clear()
        self.pending_outputs.clear()
        self.emitted_frame_ids.clear()
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

    def _run_local_ba(self, observation, dense_features):
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
            static_valid_mask=observation.valid_mask,
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
            dense, rgb, depth, poses = extract_frozen_inputs(
                self.wrapper,
                self.adapter,
                images,
                feature_device=self.feature_device,
                train_device=self.head_device,
                head_size=self.head_size,
                feature_amp=self.feature_amp,
            )
            observation = self.head(
                dense,
                rgb,
                depth,
                poses,
                frame_ids=frame_ids,
            )
            observation, match_cache, ba_result = self._run_local_ba(observation, dense)

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
            match_quality=match_quality,
            metadata={
                "local_ba_enabled": self.local_ba_enabled,
                "local_ba_accepted": None if ba_result is None else bool(ba_result.accepted[0]),
                "input_anchor_pose_c2w": poses[0, 0].detach().cpu(),
            },
        )
        self.enqueue_local_gaussian_window(packet)
        world_points = observation.centers_world()[0]
        for index, frame in enumerate(frames):
            pose = observation.poses_c2w[0, index].detach().cpu().float()
            inverse_depth = observation.refined_depth[0, index].detach().cpu().float().clamp_min(1.0e-6).reciprocal()
            confidence = observation.confidence[0, index].detach().cpu().float()
            valid = packet.valid_mask[0, index].detach().cpu().bool()
            previous_pose = None
            if index > 0:
                previous_pose = torch.linalg.inv(observation.poses_c2w[0, index - 1]) @ observation.poses_c2w[0, index]
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
            self.ready_outputs.append(self.pending_outputs[frame_id])
            self.emitted_frame_ids.add(frame_id)

    def track(self, frame: PanoFrame) -> FrontendOutput:
        self.frames.append(frame)
        self.frames_since_window += 1
        if len(self.frames) >= self.window_size and (
            self.window_index == 0 or self.frames_since_window >= self.window_stride
        ):
            self._run_window(self.frames[-self.window_size :])
            self.frames_since_window = 0
            self._emit_oldest(self.window_stride)
        return self.pending_outputs.get(int(frame.frame_id), self._pending_output(frame))

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        outputs = list(self.ready_outputs)
        self.ready_outputs.clear()
        return outputs

    def flush(self) -> list[FrontendOutput]:
        unprocessed_new = self.frames_since_window > 0
        if unprocessed_new and len(self.frames) >= 2:
            count = min(self.window_size, len(self.frames))
            last_ids = [int(frame.frame_id) for frame in self.frames[-count:]]
            if self.last_processed_frame_id != last_ids[-1]:
                self._run_window(self.frames[-count:])
        self._emit_oldest(len(self.pending_outputs))
        return self.pop_ready_outputs()


def build_spherical_selfi_frontend_from_config(config: dict[str, Any]) -> SphericalSelfiWindowFrontend:
    return SphericalSelfiWindowFrontend(config)
