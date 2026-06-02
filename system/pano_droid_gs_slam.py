"""PanoDROID front-end plus panoramic Gaussian backend SLAM runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper
from frontend.pano_droid.adapter import build_frontend_from_config
from frontend.pano_droid.dataset import discover_erp_images, load_erp_image
from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from mapping.gaussian_initializer import GaussianInitializer


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_sequence_frames(config: dict) -> Iterable[PanoFrame]:
    ds_cfg = config.get("Dataset", {})
    if ds_cfg.get("synthetic", False):
        from frontend.pano_droid.dataset import SyntheticPanoPairDataset

        ds = SyntheticPanoPairDataset(
            length=int(ds_cfg.get("synthetic_length", 4)),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
        )
        yielded_first = False
        for idx in range(len(ds)):
            sample = ds[idx]
            if not yielded_first:
                yielded_first = True
                yield PanoFrame(image=sample["image0"], timestamp=float(idx), frame_id=idx)
            yield PanoFrame(image=sample["image1"], timestamp=float(idx + 1), frame_id=idx + 1)
        return

    root = ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.dataset_path is required unless Dataset.synthetic=true.")
    files = discover_erp_images(root, sequence=ds_cfg.get("sequence"))
    begin = int(ds_cfg.get("begin", 0))
    end = ds_cfg.get("end")
    files = files[begin:end]
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    resize = (int(h), int(w)) if h is not None and w is not None else None
    for local_idx, path in enumerate(files):
        frame_id = begin + local_idx
        yield PanoFrame(
            image=load_erp_image(path, resize=resize),
            timestamp=float(frame_id),
            frame_id=frame_id,
            meta={"path": path},
        )


def _scalar_to_rgb(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = valid & np.isfinite(values)
    if not bool(valid.any()):
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(values[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0.0
    return (255.0 * rgb).astype(np.uint8)


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().cpu().float().clamp(0.0, 1.0)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected source image as 3xHxW, got {tuple(img.shape)}")
    arr = (255.0 * img.permute(1, 2, 0).numpy()).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _inverse_depth_to_pil(inverse_depth: torch.Tensor) -> Image.Image:
    inv = inverse_depth.detach().cpu().float()
    if inv.ndim == 3:
        inv = inv[0]
    if inv.ndim != 2:
        raise ValueError(f"Expected inverse depth as HxW or 1xHxW, got {tuple(inv.shape)}")
    valid = torch.isfinite(inv) & (inv > 1e-6)
    depth = torch.zeros_like(inv)
    depth[valid] = 1.0 / inv[valid].clamp_min(1e-6)
    return Image.fromarray(_scalar_to_rgb(depth.numpy(), valid.numpy()), mode="RGB")


def _resize_to_max_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.size[0] <= int(max_width):
        return image
    scale = float(max_width) / float(image.size[0])
    return image.resize((int(max_width), max(1, int(image.size[1] * scale))), Image.BILINEAR)


class SlamRuntimeLogger:
    """Runtime W&B and local visualization logger for online SLAM runs."""

    def __init__(self, config: dict, output_dir: Path) -> None:
        wb_cfg = config.get("WeightsAndBiases", {})
        vis_cfg = config.get("Visualization", {})
        self.output_dir = output_dir
        self.log_every = max(1, int(wb_cfg.get("log_every", vis_cfg.get("log_every", 10))))
        self.log_keyframes = bool(wb_cfg.get("log_keyframes", True))
        mode = str(wb_cfg.get("mode") or "online")
        self.wandb_enabled = bool(wb_cfg.get("enabled", False)) and mode != "disabled"
        self.save_local = bool(vis_cfg.get("save_local", self.wandb_enabled))
        self.visualization_dir = output_dir / "visualizations"
        if self.save_local:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.run = None
        self._wandb = None
        self._step = 0
        self._pose_history: list[tuple[int, np.ndarray]] = []

        if self.wandb_enabled:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "WeightsAndBiases.enabled=true requires the 'wandb' package. "
                    "Install it or set WeightsAndBiases.mode=disabled."
                ) from exc
            self._wandb = wandb
            self.run = wandb.init(
                project=str(wb_cfg.get("project") or "360Droid-splat"),
                entity=wb_cfg.get("entity") or None,
                name=wb_cfg.get("run_name") or None,
                mode=mode,
                dir=str(output_dir),
                config=config,
                tags=wb_cfg.get("tags") or None,
                group=wb_cfg.get("group") or None,
            )

    @property
    def run_url(self) -> str | None:
        if self.run is None:
            return None
        url = getattr(self.run, "url", None)
        return str(url) if url else None

    def observe(
        self,
        output: FrontendOutput,
        source_frame: PanoFrame,
        *,
        anchor_count: int,
        keyframe_count: int,
        backend_loss: float | None,
    ) -> None:
        self._step += 1
        pose = output.pose_c2w.detach().cpu().float()
        if pose.shape == (4, 4):
            self._pose_history.append((int(output.frame_id), pose[:3, 3].numpy()))

        payload: dict[str, float | int | str] = {
            "slam/frame_id": int(output.frame_id),
            "slam/keyframe": int(bool(output.is_keyframe)),
            "slam/keyframe_score": float(output.keyframe_score),
            "slam/pose_confidence": float(output.pose_confidence),
            "slam/anchors": int(anchor_count),
            "slam/keyframes": int(keyframe_count),
            "slam/status": str(output.tracking_status),
        }
        if output.ba_residual is not None:
            payload["slam/ba_residual"] = float(output.ba_residual)
        if backend_loss is not None:
            payload["backend/loss"] = float(backend_loss)

        if self.run is not None:
            self.run.log(payload, step=self._step)

        if not self._should_visualize(output):
            return
        image_payload = {}
        depth_path = None
        if output.inverse_depth is not None:
            depth_path = self._save_depth_panel(output, source_frame)
            if self.run is not None and self._wandb is not None:
                image_payload["slam/depth"] = self._wandb.Image(str(depth_path))
        traj_path = self._save_trajectory_panel(output)
        if self.run is not None and self._wandb is not None:
            image_payload["slam/trajectory"] = self._wandb.Image(str(traj_path))
            if depth_path is not None:
                image_payload["slam/depth_png"] = str(depth_path)
            image_payload["slam/trajectory_png"] = str(traj_path)
            self.run.log(image_payload, step=self._step)

    def _should_visualize(self, output: FrontendOutput) -> bool:
        if not self.save_local:
            return False
        return self._step == 1 or self._step % self.log_every == 0 or (
            self.log_keyframes and bool(output.is_keyframe)
        )

    def _save_depth_panel(self, output: FrontendOutput, source_frame: PanoFrame) -> Path:
        rgb = _image_tensor_to_pil(source_frame.image)
        depth = _inverse_depth_to_pil(output.inverse_depth)
        if depth.size != rgb.size:
            depth = depth.resize(rgb.size, Image.BILINEAR)
        w, h = rgb.size
        canvas = Image.new("RGB", (2 * w, h + 26), "white")
        canvas.paste(rgb, (0, 26))
        canvas.paste(depth, (w, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "source ERP", fill=(0, 0, 0))
        draw.text((w + 8, 6), "pred depth", fill=(0, 0, 0))
        canvas = _resize_to_max_width(canvas, 1600)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_depth.png"
        canvas.save(path)
        return path

    def _save_trajectory_panel(self, output: FrontendOutput) -> Path:
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_trajectory.png"
        positions = np.asarray([p for _, p in self._pose_history], dtype=np.float32)
        frame_ids = np.asarray([fid for fid, _ in self._pose_history], dtype=np.float32)
        if positions.size == 0:
            image = Image.new("RGB", (900, 640), "white")
            ImageDraw.Draw(image).text((20, 20), "no valid trajectory", fill=(0, 0, 0))
            image.save(path)
            return path
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8.5, 6.2), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#1f77b4", linewidth=2.0)
            sc = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=frame_ids,
                cmap="viridis",
                s=28,
                depthshade=True,
            )
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="limegreen", s=80, label="start")
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c="red", s=80, label="latest")
            center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
            radius = max(float((positions.max(axis=0) - positions.min(axis=0)).max()) * 0.55, 1e-3)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Online camera trajectory")
            ax.view_init(elev=24, azim=-58)
            ax.legend(loc="upper left")
            fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label="frame id")
            fig.tight_layout()
            fig.savefig(path, facecolor="white")
            plt.close(fig)
        except Exception:
            self._save_topdown_trajectory(path, positions)
        return path

    @staticmethod
    def _save_topdown_trajectory(path: Path, positions: np.ndarray) -> None:
        image = Image.new("RGB", (900, 640), "white")
        draw = ImageDraw.Draw(image)
        pts = positions[:, [0, 2]]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = np.maximum(mx - mn, 1e-3)
        norm = (pts - mn) / span
        xy = np.stack([40 + norm[:, 0] * 820, 600 - norm[:, 1] * 560], axis=-1)
        draw.line([tuple(p) for p in xy], fill=(31, 119, 180), width=3)
        for p in xy:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(31, 119, 180))
        draw.text((20, 20), "top-down trajectory fallback", fill=(0, 0, 0))
        image.save(path)

    def finish(self, summary: dict | None = None) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(summary)
        self.run.finish()


class PanoDroidGSSlamSystem:
    """Small orchestration layer matching the original SLAM staging."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.frontend = build_frontend_from_config(config)
        self.initializer = GaussianInitializer(
            max_seeds_per_keyframe=int(config.get("Mapping", {}).get("max_seeds_per_keyframe", 2048)),
            min_confidence=float(config.get("Mapping", {}).get("min_depth_confidence", 0.15)),
            voxel_sizes=tuple(config.get("Hierarchical", {}).get("voxel_size_lis", [0.12, 0.45, 1.8])),
        )
        self.map = PanoGaussianMap(config=config)
        render_cfg = config.get("Renderer", {})
        self.renderer = PFGS360Renderer(
            config=config,
            extra_gsplat360_roots=list(render_cfg.get("extra_gsplat360_roots", [])),
            allow_fallback=bool(render_cfg.get("allow_smoke_fallback", True)),
        )
        self.mapper = PanoGaussianMapper(
            self.map,
            renderer=self.renderer,
            lr=float(config.get("Mapping", {}).get("lr", 2e-3)),
        )

    def run(self, *, max_frames: int | None = None) -> dict:
        self.frontend.initialize({"config": self.config})
        output_dir = Path(self.config.get("Results", {}).get("save_dir", "outputs/pano_droid_gs_slam"))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = SlamRuntimeLogger(self.config, output_dir)
        refine_steps = int(self.config.get("Mapping", {}).get("refine_steps_per_keyframe", 0))
        frame_cache: dict[int, PanoFrame] = {}
        frame_count = 0
        keyframes = 0
        last_status = None

        def process_output(out) -> None:
            nonlocal keyframes, last_status
            last_status = out.tracking_status
            source_frame = frame_cache.pop(int(out.frame_id), None)
            if source_frame is None:
                self.mapper.stats.notes.append(f"frame {out.frame_id}: missing source frame for frontend output")
                return
            backend_loss = None
            if out.is_keyframe and out.inverse_depth is not None:
                seeds = self.initializer.from_frontend_output(out, source_frame.image)
                if self.mapper.uses_joint_optimization:
                    self.mapper.insert_keyframe(seeds, out, image=source_frame.image)
                else:
                    self.mapper.insert_keyframe(seeds, out)
                keyframes += 1
                if self.mapper.uses_joint_optimization:
                    metrics = self.mapper.optimize_after_keyframe()
                    backend_loss = metrics.get("loss")
                elif refine_steps > 0:
                    metrics = self.mapper.refine_on_keyframe(
                        image=source_frame.image,
                        c2w=out.pose_c2w,
                        steps=refine_steps,
                    )
                    backend_loss = metrics.get("loss")
            logger.observe(
                out,
                source_frame,
                anchor_count=self.map.anchor_count(),
                keyframe_count=keyframes,
                backend_loss=backend_loss,
            )

        try:
            for frame in iter_sequence_frames(self.config):
                if max_frames is not None and frame_count >= int(max_frames):
                    break
                frame_cache[int(frame.frame_id)] = frame
                out = self.frontend.track(frame)
                last_status = out.tracking_status
                pop_ready = getattr(self.frontend, "pop_ready_outputs", None)
                outputs = pop_ready() if callable(pop_ready) else [out]
                for ready in outputs:
                    process_output(ready)
                frame_count += 1

            flush = getattr(self.frontend, "flush", None)
            if callable(flush):
                for ready in flush():
                    process_output(ready)

            final_metrics = self.mapper.finalize_optimization()
            summary = {
                "frames": frame_count,
                "keyframes": keyframes,
                "anchors": self.map.anchor_count(),
                "last_tracking_status": last_status,
                "map_mode": self.map.map_mode,
                "renderer": self.config.get("Training", {}).get("panorama_render_mode", "pfgs360_gsplat"),
                "backend_last_loss": self.mapper.stats.last_loss,
                "backend_last_phase": self.mapper.stats.last_phase,
                "backend_optimization_steps": self.mapper.stats.optimization_steps,
                "backend_pose_delta_norm": self.mapper.stats.last_pose_delta_norm,
                "backend_final_metrics": final_metrics,
                "wandb_run_url": logger.run_url,
                "visualization_dir": str(logger.visualization_dir) if logger.save_local else None,
                "notes": self.mapper.stats.notes,
            }
            with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.finish(summary)
            return summary
        except BaseException as exc:
            logger.finish({"failed": True, "error": repr(exc)})
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pano_droid_gs_slam.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("WeightsAndBiases", {})["enabled"] = True
    if args.wandb_mode is not None:
        cfg.setdefault("WeightsAndBiases", {})["mode"] = args.wandb_mode
    if args.run_name:
        cfg.setdefault("WeightsAndBiases", {})["run_name"] = args.run_name
    system = PanoDroidGSSlamSystem(cfg)
    print(json.dumps(system.run(max_frames=args.max_frames), indent=2))


if __name__ == "__main__":
    main()
