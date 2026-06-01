"""Visualization helpers for graph training diagnostics."""

from __future__ import annotations

import json
from io import BytesIO
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
    """Render GT and predicted camera centers in a shared 3D coordinate frame."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    gt_valid = np.isfinite(gt_xyz).all(axis=1)
    pred_valid = np.isfinite(pred_xyz).all(axis=1)
    if not bool(gt_valid.any() or pred_valid.any()):
        canvas = Image.new("RGB", (900, 640), "white")
        ImageDraw.Draw(canvas).text((20, 20), "no valid trajectory", fill=(0, 0, 0))
        return canvas

    all_pts = np.concatenate([gt_xyz[gt_valid], pred_xyz[pred_valid]], axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = (mn + mx) * 0.5
    radius = max(float((mx - mn).max()) * 0.55, 1e-3)
    n_frames = max(gt_xyz.shape[0], pred_xyz.shape[0])
    frame_ids = np.arange(n_frames, dtype=np.float32)

    fig = plt.figure(figsize=(9.0, 6.4), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=0.0, vmax=max(1.0, float(n_frames - 1)))
    cmap = plt.get_cmap("viridis")

    def plot_track(
        arr: np.ndarray,
        valid: np.ndarray,
        *,
        label: str,
        marker: str,
        linestyle: str,
        line_color: str,
    ) -> None:
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return
        pts = arr[idx]
        colors = cmap(norm(idx.astype(np.float32)))
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            linestyle=linestyle,
            linewidth=2.0,
            color=line_color,
            alpha=0.75,
            label=label,
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=colors,
            marker=marker,
            s=46,
            depthshade=True,
            edgecolors=line_color,
            linewidths=0.6,
        )

    plot_track(gt_xyz, gt_valid, label="GT", marker="o", linestyle="-", line_color="#1f77b4")
    plot_track(pred_xyz, pred_valid, label="Pred", marker="^", linestyle="--", line_color="#d62728")

    if gt_valid.any():
        start = gt_xyz[np.flatnonzero(gt_valid)[0]]
        ax.scatter(start[0], start[1], start[2], c="limegreen", marker="o", s=120, label="Start")
    if pred_valid.any():
        end = pred_xyz[np.flatnonzero(pred_valid)[-1]]
        ax.scatter(end[0], end[1], end[2], c="red", marker="^", s=120, label="Pred end")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D trajectory: GT blue/circles, Pred red/triangles")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)
    ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(frame_ids)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.08)
    cbar.set_label("frame index")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
