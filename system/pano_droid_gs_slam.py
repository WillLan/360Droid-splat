"""PanoDROID front-end plus panoramic Gaussian backend SLAM runner."""

from __future__ import annotations

import argparse
import json
import shutil
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


def _load_panocity_gt_poses(root: str) -> dict[str, np.ndarray]:
    root_path = Path(root)
    block_dir = root_path.parent if root_path.name == "pano_images" else root_path
    image_dir = block_dir / "pano_images"
    if not image_dir.is_dir():
        return {}
    pose_files = sorted(block_dir.glob("*poses*.json"))
    if not pose_files:
        return {}
    pose_path = next((p for p in pose_files if ".1." not in p.name), pose_files[0])
    with open(pose_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", payload)
    if not isinstance(frames, list):
        return {}
    out: dict[str, np.ndarray] = {}
    for frame in frames:
        name = frame.get("name")
        mat = frame.get("transformation_matrix")
        if name is None or mat is None:
            continue
        c2w = np.asarray(mat, dtype=np.float32)
        if c2w.shape != (4, 4):
            continue
        image_path = image_dir / str(name)
        out[str(image_path.resolve())] = c2w
        out[str(name)] = c2w
    return out


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
    gt_poses = _load_panocity_gt_poses(root)
    for local_idx, path in enumerate(files):
        frame_id = begin + local_idx
        gt = gt_poses.get(str(Path(path).resolve()))
        if gt is None:
            gt = gt_poses.get(Path(path).name)
        meta = {"path": path}
        if gt is not None:
            meta["gt_c2w"] = torch.from_numpy(gt).float()
        yield PanoFrame(
            image=load_erp_image(path, resize=resize),
            timestamp=float(frame_id),
            frame_id=frame_id,
            meta=meta,
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


def _pose_xyz_from_meta(frame: PanoFrame) -> np.ndarray | None:
    meta = frame.meta or {}
    gt = meta.get("gt_c2w")
    if gt is None:
        return None
    gt_t = torch.as_tensor(gt).detach().cpu().float()
    if gt_t.shape != (4, 4):
        return None
    return gt_t[:3, 3].numpy()


def _align_xyz_umeyama_sim3(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, bool]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if int(valid.sum()) < 3:
        return pred, False
    src = pred[valid]
    tgt = gt[valid]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    cov = src_c.T @ tgt_c / max(1, src.shape[0])
    u, s, vh = np.linalg.svd(cov)
    rot = vh.T @ u.T
    if np.linalg.det(rot) < 0:
        vh = vh.copy()
        vh[-1] *= -1.0
        rot = vh.T @ u.T
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s) / max(var_src, 1e-8))
    trans = tgt_mean - scale * (rot @ src_mean)
    return scale * (pred @ rot.T) + trans, True


def _compute_ape_translation(
    pred: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], bool]:
    """Return Sim(3)-aligned predictions and APE translation per frame."""

    aligned, sim3_aligned = _align_xyz_umeyama_sim3(pred, gt)
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.isfinite(aligned).all(axis=1) & np.isfinite(gt).all(axis=1)
    ape = np.full((len(aligned),), np.nan, dtype=np.float32)
    ape[valid] = np.linalg.norm(aligned[valid] - gt[valid], axis=1)
    finite = np.isfinite(ape)
    metrics: dict[str, float] = {}
    if finite.any():
        vals = ape[finite].astype(np.float64)
        metrics = {
            "rmse": float(np.sqrt(np.mean(vals * vals))),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
        }
    return aligned, ape, metrics, sim3_aligned


def _align_xyz_for_plot(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    aligned, _ = _align_xyz_umeyama_sim3(pred, gt)
    return aligned


def _xyz_to_y_up_plot(xyz: np.ndarray) -> np.ndarray:
    """Map project coordinates to Matplotlib coordinates with original Y vertical."""

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        return xyz.reshape(-1, 3)
    return np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)


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
        self._frontend_pose_history: list[tuple[int, np.ndarray]] = []
        self._backend_pose_history: list[tuple[int, np.ndarray]] = []
        self._gt_pose_history: list[tuple[int, np.ndarray]] = []

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
        backend_pose_c2w: torch.Tensor | None = None,
        backend_render_pkg: dict | None = None,
    ) -> None:
        self._step += 1
        pose = output.pose_c2w.detach().cpu().float()
        if pose.shape == (4, 4):
            self._frontend_pose_history.append((int(output.frame_id), pose[:3, 3].numpy()))
        backend_pose = backend_pose_c2w.detach().cpu().float() if backend_pose_c2w is not None else None
        if backend_pose is not None and backend_pose.shape == (4, 4):
            self._backend_pose_history.append((int(output.frame_id), backend_pose[:3, 3].numpy()))
        gt_xyz = _pose_xyz_from_meta(source_frame)
        if gt_xyz is not None:
            self._gt_pose_history.append((int(output.frame_id), gt_xyz))

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
        if output.valid_world_points_mask is not None:
            valid_world = output.valid_world_points_mask.detach().cpu().bool()
            payload["frontend/valid_world_points"] = int(valid_world.sum().item())
        if output.world_points is not None:
            payload["frontend/world_points_finite"] = int(torch.isfinite(output.world_points).all(dim=-1).sum().item())

        if self.run is not None:
            self.run.log(payload, step=self._step)

        if not self._should_visualize(output):
            return
        image_payload = {}
        depth_path = None
        if output.inverse_depth is not None:
            depth_path = self._save_depth_panel(output, source_frame)
            if self.run is not None and self._wandb is not None:
                image_payload["frontend/depth"] = self._wandb.Image(str(depth_path))
                image_payload["slam/depth"] = self._wandb.Image(str(depth_path))
        frontend_traj_path = self._save_trajectory_panel(
            output,
            kind="frontend",
            pred_history=self._frontend_pose_history,
        )
        backend_traj_path = self._save_trajectory_panel(
            output,
            kind="backend",
            pred_history=self._backend_pose_history,
        )
        backend_rgb_path = None
        backend_depth_path = None
        if backend_render_pkg is not None:
            backend_rgb_path = self._save_backend_render_panel(output, source_frame, backend_render_pkg)
            backend_depth_path = self._save_backend_depth_panel(output, backend_render_pkg)
        if self.run is not None and self._wandb is not None:
            image_payload["frontend/trajectory_vs_gt"] = self._wandb.Image(str(frontend_traj_path))
            image_payload["backend/trajectory_vs_gt"] = self._wandb.Image(str(backend_traj_path))
            image_payload["slam/trajectory"] = self._wandb.Image(str(frontend_traj_path))
            if depth_path is not None:
                image_payload["slam/depth_png"] = str(depth_path)
            if backend_rgb_path is not None:
                image_payload["backend/render_vs_gt_panorama"] = self._wandb.Image(str(backend_rgb_path))
                image_payload["backend/render_vs_gt_png"] = str(backend_rgb_path)
            if backend_depth_path is not None:
                image_payload["backend/render_depth"] = self._wandb.Image(str(backend_depth_path))
                image_payload["backend/render_depth_png"] = str(backend_depth_path)
            image_payload["frontend/trajectory_png"] = str(frontend_traj_path)
            image_payload["backend/trajectory_png"] = str(backend_traj_path)
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

    def _save_trajectory_panel(
        self,
        output: FrontendOutput,
        *,
        kind: str,
        pred_history: list[tuple[int, np.ndarray]],
    ) -> Path:
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_{kind}_trajectory_vs_gt.png"
        legacy_path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_trajectory.png" if kind == "frontend" else None
        positions = np.asarray([p for _, p in pred_history], dtype=np.float32)
        frame_ids = np.asarray([fid for fid, _ in pred_history], dtype=np.float32)
        if positions.size == 0:
            image = Image.new("RGB", (900, 640), "white")
            ImageDraw.Draw(image).text((20, 20), "no valid trajectory", fill=(0, 0, 0))
            image.save(path)
            if legacy_path is not None:
                shutil.copyfile(path, legacy_path)
            return path
        gt_by_id = {fid: xyz for fid, xyz in self._gt_pose_history}
        gt_positions = np.asarray([gt_by_id.get(int(fid), np.full(3, np.nan)) for fid in frame_ids], dtype=np.float32)
        has_gt = bool(np.isfinite(gt_positions).all(axis=1).any())
        positions_plot = positions
        ape_errors = None
        ape_metrics: dict[str, float] = {}
        sim3_aligned = False
        if has_gt:
            positions_plot, ape_errors, ape_metrics, sim3_aligned = _compute_ape_translation(
                positions,
                gt_positions,
            )
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            fig = plt.figure(figsize=(8.5, 6.2), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            plot_pred = _xyz_to_y_up_plot(positions_plot)
            all_for_limits = [plot_pred]
            valid_gt = np.zeros((len(frame_ids),), dtype=bool)
            plot_gt = None
            if has_gt:
                valid_gt = np.isfinite(gt_positions).all(axis=1)
                plot_gt_all = _xyz_to_y_up_plot(gt_positions)
                plot_gt = plot_gt_all[valid_gt]
                all_for_limits.append(plot_gt)
                ax.plot(
                    plot_gt[:, 0],
                    plot_gt[:, 1],
                    plot_gt[:, 2],
                    color="#6b7280",
                    linestyle="--",
                    linewidth=1.6,
                    label="GT",
                )
            if len(plot_pred) >= 2:
                segments = np.stack([plot_pred[:-1], plot_pred[1:]], axis=1)
                if ape_errors is not None and np.isfinite(ape_errors).any():
                    endpoint_errors = np.stack([ape_errors[:-1], ape_errors[1:]], axis=1)
                    finite_counts = np.isfinite(endpoint_errors).sum(axis=1)
                    segment_errors = np.divide(
                        np.nansum(endpoint_errors, axis=1),
                        np.maximum(finite_counts, 1),
                    )
                    segment_errors[finite_counts == 0] = np.nan
                    finite = np.isfinite(segment_errors)
                    if finite.any():
                        fill = float(np.nanmedian(segment_errors[finite]))
                        segment_errors = np.where(finite, segment_errors, fill)
                    else:
                        segment_errors = np.zeros((len(segments),), dtype=np.float32)
                    line_collection = Line3DCollection(segments, cmap="turbo", linewidth=2.2)
                    line_collection.set_array(segment_errors.astype(np.float32))
                    line_collection.set_label(f"{kind} pred")
                    ax.add_collection3d(line_collection)
                    fig.colorbar(
                        line_collection,
                        ax=ax,
                        shrink=0.75,
                        pad=0.08,
                        label="APE translation",
                    )
                else:
                    line_collection = Line3DCollection(segments, colors="#1f77b4", linewidth=2.2)
                    line_collection.set_label(f"{kind} pred")
                    ax.add_collection3d(line_collection)
            else:
                ax.plot(
                    plot_pred[:, 0],
                    plot_pred[:, 1],
                    plot_pred[:, 2],
                    color="#1f77b4",
                    linewidth=2.2,
                    label=f"{kind} pred",
                )
            ax.scatter(plot_pred[0, 0], plot_pred[0, 1], plot_pred[0, 2], c="limegreen", s=64, label="start")
            ax.scatter(plot_pred[-1, 0], plot_pred[-1, 1], plot_pred[-1, 2], c="red", s=64, label="latest")
            limits_pts = np.concatenate([arr for arr in all_for_limits if arr.size], axis=0)
            center = 0.5 * (limits_pts.min(axis=0) + limits_pts.max(axis=0))
            radius = max(float((limits_pts.max(axis=0) - limits_pts.min(axis=0)).max()) * 0.55, 1e-3)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            if has_gt:
                align_text = "Sim(3) Umeyama" if sim3_aligned else "unaligned (<3 GT matches)"
                metric_text = ""
                if ape_metrics:
                    metric_text = (
                        f"\nAPE trans. {align_text}: "
                        f"RMSE={ape_metrics['rmse']:.3f}, "
                        f"mean={ape_metrics['mean']:.3f}, "
                        f"max={ape_metrics['max']:.3f}"
                    )
                ax.set_title(f"{kind} trajectory APE vs GT{metric_text}")
            else:
                ax.set_title(f"{kind} trajectory")
            ax.view_init(elev=24, azim=-58)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(path, facecolor="white")
            plt.close(fig)
        except Exception:
            self._save_topdown_trajectory(path, positions_plot)
        if legacy_path is not None:
            shutil.copyfile(path, legacy_path)
        return path

    def _save_backend_render_panel(
        self,
        output: FrontendOutput,
        source_frame: PanoFrame,
        render_pkg: dict,
    ) -> Path:
        target = _image_tensor_to_pil(source_frame.image)
        render = _image_tensor_to_pil(render_pkg["render"])
        if render.size != target.size:
            render = render.resize(target.size, Image.BILINEAR)
        w, h = target.size
        canvas = Image.new("RGB", (2 * w, h + 26), "white")
        canvas.paste(target, (0, 26))
        canvas.paste(render, (w, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "target panorama", fill=(0, 0, 0))
        draw.text((w + 8, 6), "backend render", fill=(0, 0, 0))
        canvas = _resize_to_max_width(canvas, 1600)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_backend_render_vs_gt.png"
        canvas.save(path)
        return path

    def _save_backend_depth_panel(self, output: FrontendOutput, render_pkg: dict) -> Path:
        depth = render_pkg.get("depth")
        if depth is None:
            image = Image.new("RGB", (900, 480), "white")
            ImageDraw.Draw(image).text((20, 20), "no backend depth", fill=(0, 0, 0))
        else:
            image = _inverse_depth_to_pil(depth.detach().clamp_min(1e-6).reciprocal())
            image = _resize_to_max_width(image, 1200)
        path = self.visualization_dir / f"frame_{int(output.frame_id):06d}_backend_render_depth.png"
        image.save(path)
        return path

    def log_final_backend_trajectory(self, poses: list[tuple[int, torch.Tensor]], *, step: int) -> str | None:
        if not poses or not self.save_local:
            return None
        history = [(int(fid), pose.detach().cpu().float()[:3, 3].numpy()) for fid, pose in poses if pose.shape == (4, 4)]
        if not history:
            return None
        dummy = FrontendOutput(
            frame_id=history[-1][0],
            timestamp=float(history[-1][0]),
            pose_c2w=torch.eye(4),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=False,
            ba_residual=None,
            tracking_status="final_backend_trajectory",
        )
        path = self._save_trajectory_panel(dummy, kind="backend_final", pred_history=history)
        if self.run is not None and self._wandb is not None:
            self.run.log({"backend/final_trajectory_vs_gt": self._wandb.Image(str(path))}, step=int(step))
        return str(path)

    def observe_backend_snapshot(self, snapshot, *, step: int) -> None:
        """Log asynchronous legacy backend snapshots."""

        poses = getattr(snapshot, "poses_c2w", {}) or {}
        for frame_id, pose in sorted(poses.items()):
            pose_cpu = pose.detach().cpu().float()
            if pose_cpu.shape != (4, 4):
                continue
            self._backend_pose_history = [
                item for item in self._backend_pose_history if item[0] != int(frame_id)
            ]
            self._backend_pose_history.append((int(frame_id), pose_cpu[:3, 3].numpy()))
        if not self.save_local or not poses:
            return
        latest_id = int(max(poses))
        dummy = FrontendOutput(
            frame_id=latest_id,
            timestamp=float(latest_id),
            pose_c2w=torch.eye(4),
            relative_pose=None,
            pose_confidence=0.0,
            inverse_depth=None,
            depth_confidence=None,
            spherical_flow=None,
            keyframe_score=0.0,
            is_keyframe=True,
            ba_residual=None,
            tracking_status=f"backend_snapshot_{getattr(snapshot, 'tag', 'unknown')}",
        )
        path = self._save_trajectory_panel(dummy, kind="backend", pred_history=self._backend_pose_history)
        if self.run is not None and self._wandb is not None:
            payload = {
                "backend/trajectory_vs_gt": self._wandb.Image(str(path)),
                "backend/trajectory_png": str(path),
            }
            render_path = getattr(snapshot, "render_path", None)
            if render_path:
                payload["backend/render_vs_gt_panorama"] = self._wandb.Image(str(render_path))
                payload["backend/render_vs_gt_png"] = str(render_path)
            depth_path = getattr(snapshot, "depth_path", None)
            if depth_path:
                payload["backend/render_depth"] = self._wandb.Image(str(depth_path))
                payload["backend/render_depth_png"] = str(depth_path)
            self.run.log(payload, step=self._step + 1)

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
        runtime_mode = str(config.get("Runtime", {}).get("mode", "sync_mvp")).lower()
        self._delegate = None
        if runtime_mode == "legacy_online":
            from system.legacy_online_slam import PanoVGGTLegacyOnlineSlamSystem

            self._delegate = PanoVGGTLegacyOnlineSlamSystem(config)
            return
        self.frontend = build_frontend_from_config(config)
        mapping_cfg = config.get("Mapping", {})
        frontend_mode = str(config.get("Frontend", {}).get("mode", "graph")).lower()
        default_seed_source = "world_points_only" if frontend_mode == "panovggt_long" else "depth_pose"
        self.initializer = GaussianInitializer(
            max_seeds_per_keyframe=int(mapping_cfg.get("max_seeds_per_keyframe", 2048)),
            min_confidence=float(mapping_cfg.get("min_depth_confidence", 0.15)),
            sky_mask_enable=bool(mapping_cfg.get("sky_mask_enable", False)),
            sky_mask_top_ratio=float(mapping_cfg.get("sky_mask_top_ratio", 0.58)),
            sky_mask_min_blue=float(mapping_cfg.get("sky_mask_min_blue", 0.35)),
            sky_mask_blue_margin=float(mapping_cfg.get("sky_mask_blue_margin", 0.05)),
            sky_mask_cloud_brightness=float(mapping_cfg.get("sky_mask_cloud_brightness", 0.72)),
            sky_mask_cloud_saturation=float(mapping_cfg.get("sky_mask_cloud_saturation", 0.22)),
            sky_mask_texture_threshold=float(mapping_cfg.get("sky_mask_texture_threshold", 0.08)),
            voxel_sizes=tuple(config.get("Hierarchical", {}).get("voxel_size_lis", [0.12, 0.45, 1.8])),
            seed_source=str(mapping_cfg.get("seed_source", default_seed_source)),
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
        if self._delegate is not None:
            return self._delegate.run(max_frames=max_frames)
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
            backend_pose = self.mapper.refined_pose_c2w(int(out.frame_id))
            render_pose = backend_pose if backend_pose is not None else out.pose_c2w.detach().cpu()
            backend_render_pkg = None
            if self.map.anchor_count() > 0:
                try:
                    backend_render_pkg = self.mapper.render_view(image=source_frame.image, c2w=render_pose)
                except Exception as exc:
                    self.mapper.stats.notes.append(f"frame {out.frame_id}: backend visualization render failed: {exc!r}")
            logger.observe(
                out,
                source_frame,
                anchor_count=self.map.anchor_count(),
                keyframe_count=keyframes,
                backend_loss=backend_loss,
                backend_pose_c2w=backend_pose,
                backend_render_pkg=backend_render_pkg,
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
            final_backend_traj = logger.log_final_backend_trajectory(
                self.mapper.refined_keyframe_poses(),
                step=frame_count,
            )
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
                "backend_final_trajectory_png": final_backend_traj,
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
