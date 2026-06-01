"""Visualization helpers for graph training diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _normalize_image(x: np.ndarray, *, valid: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = valid & np.isfinite(x)
    if not bool(valid.any()):
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(x[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * y).astype(np.uint8)


def _make_depth_panel(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    *,
    max_width: int = 1200,
) -> Image.Image:
    valid = gt_depth > 0.0
    pred_vis = _normalize_image(pred_depth, valid=np.isfinite(pred_depth))
    gt_vis = _normalize_image(gt_depth, valid=valid)
    err = np.abs(pred_depth - gt_depth)
    err_vis = _normalize_image(err, valid=valid)
    panels = [pred_vis, gt_vis, err_vis]
    images = [Image.fromarray(p).convert("RGB") for p in panels]
    w, h = images[0].size
    canvas = Image.new("RGB", (w * 3, h + 24), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["pred depth", "gt depth", "abs error"]
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * w, 24))
        draw.text((idx * w + 8, 6), labels[idx], fill=(0, 0, 0))
    if canvas.size[0] > max_width:
        scale = max_width / float(canvas.size[0])
        canvas = canvas.resize((max_width, max(1, int(canvas.size[1] * scale))), Image.BILINEAR)
    return canvas


def _positions_from_pred(
    relative_pose: torch.Tensor,
    edges: list[tuple[int, int]],
    gt_c2w: torch.Tensor,
) -> torch.Tensor:
    n = int(gt_c2w.shape[0])
    pred = torch.full((n, 3), float("nan"), dtype=gt_c2w.dtype, device=gt_c2w.device)
    pred_c2w = [None for _ in range(n)]
    pred_c2w[0] = gt_c2w[0]
    pred[0] = gt_c2w[0, :3, 3]
    edge_map = {tuple(edge): idx for idx, edge in enumerate(edges)}
    for i in range(n - 1):
        idx = edge_map.get((i, i + 1))
        if idx is None or pred_c2w[i] is None:
            continue
        T_i_to_j = relative_pose[idx]
        pred_c2w[i + 1] = pred_c2w[i] @ torch.linalg.inv(T_i_to_j)
        pred[i + 1] = pred_c2w[i + 1][:3, 3]
    return pred


def _make_trajectory_panel(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> Image.Image:
    canvas = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(canvas)
    pts = []
    for arr in (gt_xyz, pred_xyz):
        finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 2])
        if finite.any():
            pts.append(arr[finite][:, [0, 2]])
    if not pts:
        draw.text((20, 20), "no valid trajectory", fill=(0, 0, 0))
        return canvas
    all_pts = np.concatenate(pts, axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-3)
    margin = 40

    def project(arr: np.ndarray) -> list[tuple[int, int]]:
        xy = arr[:, [0, 2]]
        out = []
        for p in xy:
            if not np.isfinite(p).all():
                out.append(None)
                continue
            q = (p - mn) / span
            x = int(margin + q[0] * (640 - 2 * margin))
            y = int(480 - margin - q[1] * (480 - 2 * margin))
            out.append((x, y))
        return out

    gt = project(gt_xyz)
    pred = project(pred_xyz)
    for seq, color in ((gt, (0, 120, 255)), (pred, (220, 40, 40))):
        prev = None
        for p in seq:
            if p is None:
                prev = None
                continue
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)
            if prev is not None:
                draw.line((prev[0], prev[1], p[0], p[1]), fill=color, width=2)
            prev = p
    draw.text((20, 16), "trajectory x-z: GT blue, pred red", fill=(0, 0, 0))
    return canvas


def save_graph_diagnostics(
    batch: dict,
    pred: dict,
    *,
    output_dir: str | Path,
    step: int,
) -> dict[str, float | str]:
    """Save trajectory and depth comparison images for one graph batch."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    edges = list(pred["edges"])
    gt_c2w = batch["poses_c2w"][0].detach().cpu().float()
    rel = pred["relative_pose"][0].detach().cpu().float()
    pred_xyz = _positions_from_pred(rel, edges, gt_c2w)
    gt_xyz = gt_c2w[:, :3, 3]

    traj = _make_trajectory_panel(_to_numpy(gt_xyz), _to_numpy(pred_xyz))
    traj_path = output_path / f"step_{int(step):07d}_trajectory.png"
    traj.save(traj_path)

    src_idx = int(edges[0][0])
    gt_depth = batch["depths"][0, src_idx, 0].detach().cpu().float()
    pred_inv = pred["inverse_depth"][0, 0, 0].detach().cpu().float()
    pred_depth = torch.zeros_like(pred_inv)
    valid_pred = pred_inv > 1e-6
    pred_depth[valid_pred] = 1.0 / pred_inv[valid_pred].clamp_min(1e-6)
    depth = _make_depth_panel(_to_numpy(pred_depth), _to_numpy(gt_depth))
    depth_path = output_path / f"step_{int(step):07d}_depth.png"
    depth.save(depth_path)

    finite_traj = torch.isfinite(pred_xyz).all(dim=-1)
    if bool(finite_traj.any()):
        traj_rmse = torch.sqrt(((pred_xyz[finite_traj] - gt_xyz[finite_traj]) ** 2).sum(dim=-1).mean())
    else:
        traj_rmse = torch.tensor(float("nan"))
    valid_depth = gt_depth > 0.0
    if bool(valid_depth.any()):
        depth_mae = (pred_depth[valid_depth] - gt_depth[valid_depth]).abs().mean()
    else:
        depth_mae = torch.tensor(float("nan"))
    metrics = {
        "trajectory_png": str(traj_path),
        "depth_png": str(depth_path),
        "trajectory_rmse": float(traj_rmse),
        "depth_mae": float(depth_mae),
    }
    metrics_path = output_path / f"step_{int(step):07d}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics

