"""Queue-based online bridge to the legacy 360GS-SLAM backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import queue
import time
import traceback
from types import SimpleNamespace
from typing import Any

import torch
import torch.multiprocessing as mp

from backend.legacy_360gs.config import build_legacy_config, namespace_from_mapping
from backend.legacy_360gs.utils.multiprocessing_utils import pack_queue_message, unpack_queue_message


@dataclass
class LegacyBackendSnapshot:
    tag: str
    poses_c2w: dict[int, torch.Tensor] = field(default_factory=dict)
    anchor_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    render_path: str | None = None
    depth_path: str | None = None
    raw: Any = None


def _pose_from_legacy_rt(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = R.detach().cpu().float()
    w2c[:3, 3] = T.detach().cpu().float()
    return torch.linalg.inv(w2c)


def _snapshot_from_message(message: Any, save_dir: str | None = None) -> LegacyBackendSnapshot | None:
    data = unpack_queue_message(message)
    if not isinstance(data, (list, tuple)) or not data:
        return None
    tag = str(data[0])
    if tag == "error":
        return LegacyBackendSnapshot(tag=tag, metrics={"error": 1.0}, raw=data)
    if len(data) >= 4 and tag in {"sync_backend", "init", "keyframe", "color_refinement"}:
        gaussians = data[1]
        keyframes = data[3]
        poses: dict[int, torch.Tensor] = {}
        if isinstance(keyframes, list):
            for item in keyframes:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                poses[int(item[0])] = _pose_from_legacy_rt(item[1], item[2])
        anchor_count = 0
        if hasattr(gaussians, "get_xyz"):
            try:
                anchor_count = int(gaussians.get_xyz.shape[0])
            except Exception:
                anchor_count = 0
        elif isinstance(gaussians, dict):
            anchor_count = int(gaussians.get("anchor_count", 0))
        frame_id = max(poses.keys()) if poses else None
        render_path = None
        depth_path = None
        if save_dir and frame_id is not None:
            for ext in ("png", "jpg", "jpeg"):
                candidate = Path(save_dir) / "kf_renders_opt" / f"kf_{frame_id:04d}.{ext}"
                if candidate.exists():
                    render_path = str(candidate)
                    break
            for ext in ("png", "jpg", "jpeg"):
                candidate = Path(save_dir) / "kf_depths_opt" / f"kf_{frame_id:04d}.{ext}"
                if candidate.exists():
                    depth_path = str(candidate)
                    break
        return LegacyBackendSnapshot(
            tag=tag,
            poses_c2w=poses,
            anchor_count=anchor_count,
            render_path=render_path,
            depth_path=depth_path,
            raw=data,
        )
    return LegacyBackendSnapshot(tag=tag, raw=data)


def _make_background(config: dict[str, Any], device: str) -> torch.Tensor:
    bg = config.get("Training", {}).get("render_background_rgb", [0.0, 0.0, 0.0])
    return torch.tensor([float(v) for v in bg], dtype=torch.float32, device=device)


def _run_real_legacy_backend_process(
    config: dict[str, Any],
    save_dir: str,
    backend_queue: mp.Queue,
    frontend_queue: mp.Queue,
) -> None:
    try:
        cfg = build_legacy_config(config)
        device = str(cfg.get("LegacyOnlineBackend", {}).get("device", "cuda"))
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device).index or 0)

        from backend.legacy_360gs.gaussian_splatting.scene.gaussian_model import GaussianModel
        from backend.legacy_360gs.gaussian_splatting.scene.pano_scaffold_model import PanoScaffoldModel
        from backend.legacy_360gs.utils.slam_backend import BackEnd

        model_params = namespace_from_mapping(cfg["model_params"])
        opt_params = namespace_from_mapping(cfg["opt_params"])
        pipeline_params = namespace_from_mapping(cfg["pipeline_params"])
        map_mode = cfg.get("MapRepresentation", {}).get("mode", "legacy_gaussian_panorama")
        if map_mode == "anchor_scaffold_panorama":
            gaussians = PanoScaffoldModel(model_params.sh_degree, config=cfg)
        else:
            gaussians = GaussianModel(model_params.sh_degree, config=cfg)
        gaussians.init_lr(float(cfg["opt_params"].get("init_lr", 6)))
        gaussians.training_setup(opt_params)

        backend = BackEnd(cfg, save_dir=save_dir)
        backend.gaussians = gaussians
        backend.background = _make_background(cfg, device)
        backend.cameras_extent = float(cfg.get("LegacyOnlineBackend", {}).get("cameras_extent", 6.0))
        backend.pipeline_params = pipeline_params
        backend.opt_params = opt_params
        backend.frontend_queue = frontend_queue
        backend.backend_queue = backend_queue
        backend.live_mode = bool(cfg.get("LegacyOnlineBackend", {}).get("live_mode", False))
        backend.set_hyperparams()
        backend.run()
    except Exception:
        err = traceback.format_exc()
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "backend_error.log"), "w", encoding="utf-8") as f:
            f.write(err)
        frontend_queue.put(pack_queue_message(["error", err]))
        raise


def _run_fake_backend_process(
    config: dict[str, Any],
    save_dir: str,
    backend_queue: mp.Queue,
    frontend_queue: mp.Queue,
) -> None:
    del config
    os.makedirs(save_dir, exist_ok=True)
    keyframes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    anchor_count = 0
    while True:
        data = unpack_queue_message(backend_queue.get())
        tag = data[0]
        if tag == "stop":
            break
        if tag in {"pause", "unpause", "color_refinement"}:
            continue
        if tag == "register":
            frame_id = int(data[1])
            viewpoint = data[2]
            depth_map = data[3]
            keyframes[frame_id] = (viewpoint.R.detach().cpu(), viewpoint.T.detach().cpu())
            world_valid = getattr(viewpoint, "global_world_points_valid_mask", None)
            if world_valid is not None:
                valid = torch.as_tensor(world_valid).detach().cpu().bool()
            else:
                valid = torch.as_tensor(depth_map).detach().cpu() > 0.01
            anchor_count += int(valid.sum().item())
            continue
        if tag in {"init", "keyframe"}:
            frame_id = int(data[1])
            viewpoint = data[2]
            depth_map = data[3] if tag == "init" else data[4]
            keyframes[frame_id] = (viewpoint.R.detach().cpu(), viewpoint.T.detach().cpu())
            world_valid = getattr(viewpoint, "global_world_points_valid_mask", None)
            if world_valid is not None:
                valid = torch.as_tensor(world_valid).detach().cpu().bool()
            else:
                valid = torch.as_tensor(depth_map).detach().cpu() > 0.01
            anchor_count += int(valid.sum().item())
            payload = [
                tag,
                {"anchor_count": anchor_count},
                {},
                [(fid, rt[0], rt[1]) for fid, rt in sorted(keyframes.items())],
            ]
            frontend_queue.put(pack_queue_message(payload))


class LegacyOnlineBackendClient:
    """Main-process client for the legacy backend worker."""

    def __init__(self, config: dict[str, Any], *, save_dir: str | Path) -> None:
        self.config = build_legacy_config(config)
        self.save_dir = str(save_dir)
        backend_cfg = self.config.get("LegacyOnlineBackend", {})
        start_method = str(self.config.get("Runtime", {}).get("multiprocessing_start_method", "spawn"))
        self.ctx = mp.get_context(start_method)
        self.backend_queue: mp.Queue = self.ctx.Queue()
        self.frontend_queue: mp.Queue = self.ctx.Queue()
        backend_impl = str(backend_cfg.get("backend_impl", backend_cfg.get("impl", "legacy"))).lower()
        target = _run_fake_backend_process if backend_impl == "fake" else _run_real_legacy_backend_process
        self.process = self.ctx.Process(
            target=target,
            args=(self.config, self.save_dir, self.backend_queue, self.frontend_queue),
            daemon=False,
        )
        self.keyframe_ids: list[int] = []
        self.window_size = int(self.config.get("Training", {}).get("window_size", 8))
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.process.start()
            self.started = True

    def submit_init(self, *, frame_id: int, viewpoint: Any, depth_map: torch.Tensor) -> None:
        self._remember_keyframe(frame_id)
        self.backend_queue.put(pack_queue_message(["init", int(frame_id), viewpoint, depth_map.detach().cpu()]))

    def submit_keyframe(self, *, frame_id: int, viewpoint: Any, depth_map: torch.Tensor) -> None:
        current_window = self._remember_keyframe(frame_id)
        theta = torch.zeros(1, 3)
        self.backend_queue.put(
            pack_queue_message(
                ["keyframe", int(frame_id), viewpoint, current_window, depth_map.detach().cpu(), theta]
            )
        )

    def submit_window(self, bundles: list[tuple[int, Any, torch.Tensor]]) -> int | None:
        """Register a frontend prediction window and optimise it once.

        All frames except the last are registered without running mapping.  The
        final frame triggers the legacy ``keyframe`` path with a current window
        that includes every frame submitted here, so the backend performs one
        mapping burst for the PanoVGGT submap instead of queueing one burst per
        frame.
        """

        if not bundles:
            return None
        normalized = [(int(fid), viewpoint, depth_map.detach().cpu()) for fid, viewpoint, depth_map in bundles]
        for frame_id, viewpoint, depth_map in normalized[:-1]:
            self._remember_keyframe(frame_id)
            self.backend_queue.put(pack_queue_message(["register", frame_id, viewpoint, depth_map]))
        frame_id, viewpoint, depth_map = normalized[-1]
        self.submit_keyframe(frame_id=frame_id, viewpoint=viewpoint, depth_map=depth_map)
        return int(frame_id)

    def wait_for_frame(self, frame_id: int, *, timeout_s: float = 1800.0) -> list[LegacyBackendSnapshot]:
        """Wait until a backend snapshot contains ``frame_id``."""

        target = int(frame_id)
        snapshots: list[LegacyBackendSnapshot] = []
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            polled = self.poll()
            snapshots.extend(polled)
            if any(target in snapshot.poses_c2w for snapshot in polled):
                return snapshots
            if self.started and self.process.exitcode not in (0, None):
                raise RuntimeError(f"Legacy backend process exited with code {self.process.exitcode}")
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for legacy backend frame {target}")

    def poll(self) -> list[LegacyBackendSnapshot]:
        snapshots: list[LegacyBackendSnapshot] = []
        while True:
            try:
                msg = self.frontend_queue.get_nowait()
            except queue.Empty:
                break
            snapshot = _snapshot_from_message(msg, save_dir=self.save_dir)
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def stop(
        self,
        *,
        color_refinement: bool = False,
        join_timeout_s: float = 30.0,
        save_final_artifacts: bool = True,
    ) -> list[LegacyBackendSnapshot]:
        if not self.started:
            return []
        if color_refinement:
            self.backend_queue.put(pack_queue_message(["color_refinement"]))
        self.backend_queue.put(pack_queue_message(["stop", {"save_final_artifacts": bool(save_final_artifacts)}]))
        snapshots: list[LegacyBackendSnapshot] = []
        deadline = time.monotonic() + float(join_timeout_s)
        while self.process.is_alive() and time.monotonic() < deadline:
            self.process.join(timeout=0.1)
            snapshots.extend(self.poll())
        snapshots.extend(self.poll())
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.exitcode not in (0, None):
            raise RuntimeError(f"Legacy backend process exited with code {self.process.exitcode}")
        return snapshots

    def _remember_keyframe(self, frame_id: int) -> list[int]:
        fid = int(frame_id)
        if fid not in self.keyframe_ids:
            self.keyframe_ids.append(fid)
        if len(self.keyframe_ids) > self.window_size:
            self.keyframe_ids = self.keyframe_ids[-self.window_size :]
        return list(self.keyframe_ids)
