"""PanoVGGT frontend with the legacy online 360GS-SLAM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import traceback

import torch

from backend.legacy_360gs.online import LegacyBackendSnapshot, LegacyOnlineBackendClient
from backend.legacy_360gs.viewpoint_adapter import LegacyViewpointAdapter
from frontend.pano_droid.adapter import build_frontend_from_config
from frontend.pano_droid.interfaces import PanoFrame
from system.pano_droid_gs_slam import SlamRuntimeLogger, iter_sequence_frames


class PanoVGGTLegacyOnlineSlamSystem:
    """Queue-based online SLAM runner using PanoVGGT tracking and legacy backend mapping."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.frontend = build_frontend_from_config(config)
        output_dir = Path(config.get("Results", {}).get("save_dir", "outputs/pano_vggt_legacy_online"))
        output_dir.mkdir(parents=True, exist_ok=True)
        backend_cfg = config.get("LegacyOnlineBackend", {})
        backend_impl = str(backend_cfg.get("backend_impl", backend_cfg.get("impl", "legacy"))).lower()
        self.adapter = LegacyViewpointAdapter(config, use_legacy_camera=(backend_impl != "fake"))
        self.backend = LegacyOnlineBackendClient(config, save_dir=output_dir)
        self.output_dir = output_dir
        self.feedback_enable = bool(backend_cfg.get("feedback_enable", True))
        self.submit_all_frontend_outputs = bool(backend_cfg.get("submit_all_frontend_outputs", True))
        self.synchronous_windows = bool(backend_cfg.get("synchronous_windows", True))
        self.window_wait_timeout_s = float(backend_cfg.get("window_wait_timeout_s", 1800.0))
        self.color_refinement_on_stop = bool(backend_cfg.get("color_refinement_on_stop", False))
        self.join_timeout_s = float(backend_cfg.get("join_timeout_s", 30.0))
        self.latest_anchor_count = 0
        self.backend_poses: dict[int, torch.Tensor] = {}
        self.last_backend_tag: str | None = None

    def run(self, *, max_frames: int | None = None) -> dict[str, Any]:
        self.frontend.initialize({"config": self.config})
        logger = SlamRuntimeLogger(self.config, self.output_dir)
        self.backend.start()
        frame_cache: dict[int, PanoFrame] = {}
        frame_count = 0
        backend_frames = 0
        last_status = None
        backend_initialized = False

        def handle_snapshots(snapshots: list[LegacyBackendSnapshot]) -> None:
            for snapshot in snapshots:
                self.last_backend_tag = snapshot.tag
                if snapshot.anchor_count > 0:
                    self.latest_anchor_count = int(snapshot.anchor_count)
                if snapshot.poses_c2w:
                    self.backend_poses.update(snapshot.poses_c2w)
                    if self.feedback_enable:
                        apply_updates = getattr(self.frontend, "apply_backend_pose_updates", None)
                        if callable(apply_updates):
                            apply_updates(snapshot.poses_c2w)
                logger.observe_backend_snapshot(snapshot, step=frame_count)

        def process_outputs(outputs) -> None:
            nonlocal backend_frames, last_status, backend_initialized
            ready = sorted(outputs, key=lambda item: int(item.frame_id))
            output_records = []
            backend_bundles = []
            for out in ready:
                last_status = out.tracking_status
                source_frame = frame_cache.pop(int(out.frame_id), None)
                if source_frame is None:
                    continue
                output_records.append((out, source_frame))
                should_submit = (
                    out.inverse_depth is not None
                    and (self.submit_all_frontend_outputs or bool(out.is_keyframe))
                )
                if should_submit:
                    backend_bundles.append(self.adapter.build(source_frame, out))

            if backend_bundles:
                if not backend_initialized:
                    first = backend_bundles[0]
                    self.backend.submit_init(
                        frame_id=first.frame_id,
                        viewpoint=first.viewpoint,
                        depth_map=first.depth_map,
                    )
                    backend_initialized = True
                    backend_frames += 1
                    if self.synchronous_windows:
                        handle_snapshots(
                            self.backend.wait_for_frame(
                                first.frame_id,
                                timeout_s=self.window_wait_timeout_s,
                            )
                        )
                    backend_bundles = backend_bundles[1:]
                if backend_bundles:
                    target_frame_id = self.backend.submit_window(
                        [
                            (bundle.frame_id, bundle.viewpoint, bundle.depth_map)
                            for bundle in backend_bundles
                        ]
                    )
                    backend_frames += len(backend_bundles)
                    if self.synchronous_windows and target_frame_id is not None:
                        handle_snapshots(
                            self.backend.wait_for_frame(
                                target_frame_id,
                                timeout_s=self.window_wait_timeout_s,
                            )
                        )
                handle_snapshots(self.backend.poll())

            for out, source_frame in output_records:
                backend_pose = self.backend_poses.get(int(out.frame_id))
                logger.observe(
                    out,
                    source_frame,
                    anchor_count=self.latest_anchor_count,
                    keyframe_count=backend_frames,
                    backend_loss=None,
                    backend_pose_c2w=backend_pose,
                    backend_render_pkg=None,
                )

        try:
            for frame in iter_sequence_frames(self.config):
                if max_frames is not None and frame_count >= int(max_frames):
                    break
                handle_snapshots(self.backend.poll())
                frame_cache[int(frame.frame_id)] = frame
                out = self.frontend.track(frame)
                last_status = out.tracking_status
                pop_ready = getattr(self.frontend, "pop_ready_outputs", None)
                outputs = pop_ready() if callable(pop_ready) else [out]
                process_outputs(outputs)
                frame_count += 1

            flush = getattr(self.frontend, "flush", None)
            if callable(flush):
                process_outputs(flush())

            handle_snapshots(self.backend.stop(
                color_refinement=self.color_refinement_on_stop,
                join_timeout_s=self.join_timeout_s,
            ))
            final_backend_traj = logger.log_final_backend_trajectory(
                sorted(self.backend_poses.items()),
                step=frame_count,
            )
            summary = {
                "frames": frame_count,
                "keyframes": backend_frames,
                "anchors": self.latest_anchor_count,
                "backend_last_tag": self.last_backend_tag,
                "last_status": last_status,
                "final_backend_trajectory": final_backend_traj,
                "runtime_mode": "legacy_online",
            }
            with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.finish(summary)
            return summary
        except Exception as exc:
            system_traceback = traceback.format_exc()
            stop_error = None
            stop_traceback = None
            try:
                handle_snapshots(self.backend.stop(join_timeout_s=5.0))
            except Exception as stop_exc:
                stop_error = repr(stop_exc)
                stop_traceback = traceback.format_exc()
            failed = {
                "frames": frame_count,
                "keyframes": backend_frames,
                "runtime_mode": "legacy_online_failed",
                "error": repr(exc),
                "backend_stop_error": stop_error,
            }
            with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, indent=2)
            with open(self.output_dir / "system_error.log", "w", encoding="utf-8") as f:
                f.write(system_traceback)
                if stop_traceback is not None:
                    f.write("\n--- backend stop error ---\n")
                    f.write(stop_traceback)
            logger.finish(failed)
            raise
