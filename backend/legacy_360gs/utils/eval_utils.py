import json
import os
from typing import Any
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")  # must run before evo.tools.plot (headless / no TkAgg)

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from backend.legacy_360gs.gaussian_splatting.gaussian_renderer import render
from backend.legacy_360gs.gaussian_splatting.utils.image_utils import psnr
from backend.legacy_360gs.gaussian_splatting.utils.loss_utils import ssim
from backend.legacy_360gs.gaussian_splatting.utils.system_utils import mkdir_p
from backend.legacy_360gs.utils.camera_utils import PanoramaCamera
from backend.legacy_360gs.utils.logging_utils import Log
from backend.legacy_360gs.utils.pano_masking import get_viewpoint_ignore_mask


evo = None
metrics = None
trajectory = None
PoseRelation = None
Unit = None
PosePath3D = None
PoseTrajectory3D = None
plot = None
PlotMode = None


def _ensure_evo():
    global evo, metrics, trajectory, PoseRelation, Unit
    global PosePath3D, PoseTrajectory3D, plot, PlotMode
    if evo is not None:
        return

    import evo as _evo
    from evo.core import metrics as _metrics, trajectory as _trajectory
    from evo.core.metrics import PoseRelation as _PoseRelation, Unit as _Unit
    from evo.core.trajectory import PosePath3D as _PosePath3D
    from evo.core.trajectory import PoseTrajectory3D as _PoseTrajectory3D
    # evo.tools.plot calls mpl.use(SETTINGS.plot_backend) at import time 鈥?set Agg
    # *before* importing plot, or headless runs fail on TkAgg.
    from evo.tools.settings import SETTINGS as _SETTINGS

    _SETTINGS.plot_backend = "Agg"
    from evo.tools import plot as _plot
    from evo.tools.plot import PlotMode as _PlotMode

    evo = _evo
    metrics = _metrics
    trajectory = _trajectory
    PoseRelation = _PoseRelation
    Unit = _Unit
    PosePath3D = _PosePath3D
    PoseTrajectory3D = _PoseTrajectory3D
    plot = _plot
    PlotMode = _PlotMode


def _clone_pose_path(traj: Any):
    _ensure_evo()
    return PosePath3D(poses_se3=[pose.copy() for pose in traj.poses_se3])


def _align_trajectory_compat(
    traj_est: Any,
    traj_ref: Any,
    correct_scale: bool = False,
):
    _ensure_evo()
    if hasattr(trajectory, "align_trajectory"):
        return trajectory.align_trajectory(
            traj_est, traj_ref, correct_scale=correct_scale
        )
    traj_est_aligned = _clone_pose_path(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=correct_scale)
    return traj_est_aligned


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, return_stats=False):
    _ensure_evo()
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = _align_trajectory_compat(
        traj_est, traj_ref, correct_scale=monocular
    )
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE [m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))))
    plt.close(fig) 

    if return_stats:
        return ape_stat, ape_stats
    return ape_stat

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, BA=False):
    _ensure_evo()
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []
    raw_translation_errors = []
    raw_rotation_errors = []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)
        raw_translation_errors.append(
            float(np.linalg.norm(pose_est[:3, 3] - pose_gt[:3, 3]))
        )
        raw_rotation_errors.append(
            _rotation_error_deg(pose_est[:3, :3], pose_gt[:3, :3])
        )

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    if BA:
        label_evo = "after BA"
    elif final:
        label_evo = "final"
    else:
        label_evo = "{:04}".format(iterations)

    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    try:
        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
        wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    except Exception as e:
        Log(f"ATE evaluation failed ({e}); skipping evo plot", tag="Eval")
        ate = float("nan")

    aligned_for_plot = _align_poses_for_plot(
        trj_est_np, trj_gt_np, correct_scale=monocular
    )
    # Always save a simple matplotlib trajectory comparison (est vs GT)
    _save_trajectory_comparison(
        aligned_for_plot, trj_gt_np, trj_id, plot_dir, label_evo
    )

    metrics_payload = {
        "label": label_evo,
        "count": len(trj_id),
        "correct_scale": bool(monocular),
        "ate_rmse": None if np.isnan(ate) else float(ate),
        "raw_translation_error_m": _array_stats(raw_translation_errors),
        "raw_rotation_error_deg": _array_stats(raw_rotation_errors),
    }
    with open(
        os.path.join(plot_dir, f"pose_metrics_{label_evo}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_payload, f, indent=4)
    return ate


def _pose_matrix_from_rt(R, T):
    pose = np.eye(4, dtype=np.float64)
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    pose[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    pose[0:3, 3] = np.asarray(T, dtype=np.float64)
    return pose


def _rotation_error_deg(R_est, R_gt):
    rel = np.asarray(R_est, dtype=np.float64) @ np.asarray(R_gt, dtype=np.float64).T
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _array_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def save_pose_dict_artifact(pose_dict, save_dir, stem):
    mkdir_p(save_dir)
    npy_path = os.path.join(save_dir, f"{stem}.npy")
    json_path = os.path.join(save_dir, f"{stem}.json")
    np.save(npy_path, pose_dict)
    serializable = {
        str(int(frame_id)): np.asarray(pose, dtype=np.float64).tolist()
        for frame_id, pose in sorted(pose_dict.items())
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return {"npy": npy_path, "json": json_path}


def eval_pose_dict(
    frames,
    pose_dict,
    save_dir,
    label,
    frame_ids=None,
    monocular=False,
    correct_scale=None,
):
    _ensure_evo()
    mkdir_p(save_dir)
    if correct_scale is None:
        correct_scale = bool(monocular)

    if frame_ids is None:
        ordered_ids = sorted(
            int(frame_id)
            for frame_id in pose_dict.keys()
            if int(frame_id) in frames
        )
    else:
        ordered_ids = [
            int(frame_id)
            for frame_id in frame_ids
            if int(frame_id) in pose_dict and int(frame_id) in frames
        ]

    if len(ordered_ids) < 2:
        metrics_path = os.path.join(save_dir, f"pose_metrics_{label}.json")
        payload = {
            "label": label,
            "count": len(ordered_ids),
            "frame_ids": ordered_ids,
            "skipped": True,
            "reason": "need at least 2 poses",
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    trj_data = {"trj_id": [], "trj_est": [], "trj_gt": []}
    poses_est_c2w = []
    poses_gt_c2w = []
    raw_translation_errors = []
    raw_rotation_errors = []

    for frame_id in ordered_ids:
        frame = frames[frame_id]
        pose_est_w2c = np.asarray(pose_dict[frame_id], dtype=np.float64)
        pose_gt_w2c = _pose_matrix_from_rt(frame.R_gt, frame.T_gt)
        pose_est_c2w = np.linalg.inv(pose_est_w2c)
        pose_gt_c2w = np.linalg.inv(pose_gt_w2c)

        trj_data["trj_id"].append(int(frame_id))
        trj_data["trj_est"].append(pose_est_c2w.tolist())
        trj_data["trj_gt"].append(pose_gt_c2w.tolist())
        poses_est_c2w.append(pose_est_c2w)
        poses_gt_c2w.append(pose_gt_c2w)
        raw_translation_errors.append(
            float(np.linalg.norm(pose_est_c2w[:3, 3] - pose_gt_c2w[:3, 3]))
        )
        raw_rotation_errors.append(
            _rotation_error_deg(pose_est_c2w[:3, :3], pose_gt_c2w[:3, :3])
        )

    with open(
        os.path.join(save_dir, f"trj_{label}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=2)

    try:
        ate_rmse, ape_stats = evaluate_evo(
            poses_gt=poses_gt_c2w,
            poses_est=poses_est_c2w,
            plot_dir=save_dir,
            label=label,
            monocular=correct_scale,
            return_stats=True,
        )
    except Exception as e:
        Log(f"[eval_pose_dict] evo evaluation failed for {label}: {e}", tag="Eval")
        ate_rmse = float("nan")
        ape_stats = None

    aligned_for_plot = _align_poses_for_plot(
        poses_est_c2w, poses_gt_c2w, correct_scale=correct_scale
    )
    _save_trajectory_comparison(
        aligned_for_plot, poses_gt_c2w, ordered_ids, save_dir, label
    )

    payload = {
        "label": label,
        "count": len(ordered_ids),
        "frame_ids": ordered_ids,
        "correct_scale": bool(correct_scale),
        "ate_rmse": float(ate_rmse),
        "ape_translation_stats": ape_stats,
        "raw_translation_error_m": _array_stats(raw_translation_errors),
        "raw_rotation_error_deg": _array_stats(raw_rotation_errors),
        "files": {
            "trajectory_json": os.path.join(save_dir, f"trj_{label}.json"),
            "evo_plot": os.path.join(save_dir, f"evo_2dplot_{label}.png"),
            "trajectory_plot": os.path.join(save_dir, f"trajectory_{label}.png"),
            "ape_stats_json": os.path.join(save_dir, f"stats_{label}.json"),
        },
    }
    with open(
        os.path.join(save_dir, f"pose_metrics_{label}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2)
    return payload


def _align_poses_for_plot(poses_est, poses_gt, correct_scale=False):
    try:
        traj_ref = PosePath3D(poses_se3=poses_gt)
        traj_est = PosePath3D(poses_se3=poses_est)
        traj_est_aligned = _align_trajectory_compat(
            traj_est, traj_ref, correct_scale=correct_scale
        )
        return [pose.copy() for pose in traj_est_aligned.poses_se3]
    except Exception as e:
        Log(f"[_align_poses_for_plot] alignment failed ({e}), using raw poses", tag="Eval")
        return [pose.copy() for pose in poses_est]


def _meter_axis_limits(values, tick_m=5.0, min_span_m=10.0):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -min_span_m * 0.5, min_span_m * 0.5
    lo = float(finite.min())
    hi = float(finite.max())
    center = 0.5 * (lo + hi)
    span = max(hi - lo, float(min_span_m))
    half = 0.5 * span
    lo = np.floor((center - half) / tick_m) * tick_m
    hi = np.ceil((center + half) / tick_m) * tick_m
    if hi <= lo:
        hi = lo + tick_m
    return float(lo), float(hi)


def _set_meter_ticks(ax, axes=("x", "y"), tick_m=5.0):
    locator = MultipleLocator(float(tick_m))
    if "x" in axes:
        ax.xaxis.set_major_locator(locator)
    if "y" in axes:
        ax.yaxis.set_major_locator(MultipleLocator(float(tick_m)))
    if "z" in axes and hasattr(ax, "zaxis"):
        ax.zaxis.set_major_locator(MultipleLocator(float(tick_m)))


def _equal_span_3d_limits(xlim, ylim, zlim, tick_m=5.0):
    """Return three axis limits with the same metric span."""
    limits = [xlim, ylim, zlim]
    centers = [0.5 * (lo + hi) for lo, hi in limits]
    span = max(float(hi - lo) for lo, hi in limits)
    span = max(span, float(tick_m))
    half = 0.5 * span
    out = []
    for center in centers:
        lo = np.floor((center - half) / tick_m) * tick_m
        hi = np.ceil((center + half) / tick_m) * tick_m
        out.append((float(lo), float(hi)))
    return out[0], out[1], out[2]


def _save_trajectory_comparison(poses_est, poses_gt, frame_ids, plot_dir, label):
    """Save scale-aligned 2D/3D trajectory comparisons."""
    if len(poses_est) < 2:
        return
    est = np.array([p[:3, 3] for p in poses_est])   # (N,3) c2w translation
    gt  = np.array([p[:3, 3] for p in poses_gt])
    tick_m = 5.0
    xlim = _meter_axis_limits(np.concatenate([gt[:, 0], est[:, 0]]), tick_m=tick_m)
    ylim = _meter_axis_limits(np.concatenate([gt[:, 1], est[:, 1]]), tick_m=tick_m)
    zlim = _meter_axis_limits(np.concatenate([gt[:, 2], est[:, 2]]), tick_m=tick_m)
    xlim3d, zlim3d, ylim3d = _equal_span_3d_limits(xlim, zlim, ylim, tick_m=tick_m)

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    fig.suptitle(f"Trajectory comparison [{label}]  (N={len(est)} keyframes)",
                 fontsize=13)

    # ---- left: bird-eye XZ ----
    ax.plot(gt[:, 0],  gt[:, 2],  'o--', color='gray',   lw=1.5, ms=4, label='GT')
    ax.plot(est[:, 0], est[:, 2], 's-',  color='#e74c3c', lw=1.5, ms=4, label='Est')
    # Mark start
    ax.plot(gt[0, 0],  gt[0, 2],  'o', color='green', ms=8)
    ax.plot(est[0, 0], est[0, 2], 's', color='green', ms=8)
    for i, fid in enumerate(frame_ids):
        if i % max(1, len(frame_ids) // 8) == 0:
            ax.annotate(str(fid), (est[i, 0], est[i, 2]), fontsize=7, color='#c0392b')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
    ax.set_title('Bird-eye (X-Z)')
    ax.set_xlim(*xlim); ax.set_ylim(*zlim)
    _set_meter_ticks(ax, axes=("x", "y"), tick_m=tick_m)
    ax.legend(); ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.3)

    # ---- right: altitude Y vs frame ----
    xs = list(range(len(frame_ids)))
    ax2.plot(xs, gt[:, 1],  'o--', color='gray',   lw=1.5, ms=4, label='GT')
    ax2.plot(xs, est[:, 1], 's-',  color='#e74c3c', lw=1.5, ms=4, label='Est')
    ax2.set_xticks(xs[::max(1, len(xs) // 8)])
    ax2.set_xticklabels([str(frame_ids[i]) for i in xs[::max(1, len(xs) // 8)]], fontsize=7)
    ax2.set_xlabel('Keyframe index'); ax2.set_ylabel('Y [m]')
    ax2.set_ylim(*ylim)
    _set_meter_ticks(ax2, axes=("y",), tick_m=tick_m)
    ax2.set_title('Altitude (Y, 5m scale)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ---- third: scale-aligned 3D trajectory ----
    ax3.plot(gt[:, 0], gt[:, 2], gt[:, 1], 'o--', color='gray', lw=1.2, ms=3, label='GT')
    ax3.plot(est[:, 0], est[:, 2], est[:, 1], 's-', color='#e74c3c', lw=1.2, ms=3, label='Est')
    ax3.scatter(gt[0, 0], gt[0, 2], gt[0, 1], color='green', s=40)
    ax3.scatter(est[0, 0], est[0, 2], est[0, 1], color='green', s=40)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Z [m]')
    ax3.set_zlabel('Y [m]')
    ax3.set_title('3D trajectory (scale-aligned)')
    ax3.set_xlim(*xlim3d); ax3.set_ylim(*zlim3d); ax3.set_zlim(*ylim3d)
    _set_meter_ticks(ax3, axes=("x", "y", "z"), tick_m=tick_m)
    try:
        ax3.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"trajectory_{label}.png"), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    Log(f"Saved trajectory comparison 鈫?plot/trajectory_{label}.png", tag="Eval")

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    datatype,
    kf_indices,
    iteration="final",
    include_kf=True,
):
    """Render every frame and save GT|render comparison images with PSNR.

    Args:
        include_kf: if True, also render keyframe indices (default True so that
                    the final evaluation covers all frames).
    """
    interval = 1
    end_idx = len(frames) - 1 if iteration in ("final", "before_opt", "after_opt") else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # Output directories
    viz_dir = os.path.join(save_dir, "all_frames", str(iteration))
    render_dir = os.path.join(save_dir, "render_rgb", str(iteration))
    depth_dir = os.path.join(save_dir, "render_depth", str(iteration))
    for d in (viz_dir, render_dir, depth_dir):
        mkdir_p(d)

    for idx in range(0, end_idx, interval):
        if not include_kf and idx in kf_indices:
            continue
        frame = frames[idx]
        gt_image, _, _, _ = dataset[idx]

        with torch.no_grad():
            if isinstance(frame, PanoramaCamera):
                from backend.legacy_360gs.utils.panoramic_renderer import render_panorama_for_config
                import torch as _torch
                _z = _torch.zeros(1, 3, device="cuda")
                render_pkg = render_panorama_for_config(
                    frame,
                    gaussians,
                    pipe,
                    background,
                    config=dataset.config,
                    theta=_z,
                    rho=_z,
                )
            else:
                render_pkg = render(frame, gaussians, pipe, background)
        rendering = render_pkg["render"]

        # Depth visualisation
        depth_np = render_pkg["depth"].squeeze().detach().cpu().numpy()
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_vis = ((depth_np - d_min) / max(d_max - d_min, 1e-8) * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(os.path.join(depth_dir, f"{idx:04d}.png"))

        image = torch.clamp(rendering, 0.0, 1.0)
        gt_image = gt_image.to(image.device)
        valid_eval_mask = None
        if isinstance(frame, PanoramaCamera):
            ignore_mask = get_viewpoint_ignore_mask(frame, dataset.config, device=image.device)
            if ignore_mask.ndim == 2:
                ignore_mask = ignore_mask.unsqueeze(0)
            valid_eval_mask = (~ignore_mask).to(device=image.device, dtype=image.dtype)
            image = image * valid_eval_mask
            gt_image = gt_image * valid_eval_mask

        psnr_score = psnr(image.unsqueeze(0), gt_image.unsqueeze(0))
        ssim_score = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        lpips_score = cal_lpips(image.unsqueeze(0), gt_image.unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        # ---- GT | Render side-by-side with PSNR ----
        gt_np   = (gt_image.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pred_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        H, W = gt_np.shape[:2]
        gap = 4
        is_kf = idx in kf_indices
        canvas = np.zeros((H + 28, W * 2 + gap, 3), dtype=np.uint8)
        canvas[28:, :W]       = gt_np[:, :, ::-1] if gt_np.shape[2] == 3 else gt_np
        canvas[28:, W + gap:] = pred_np[:, :, ::-1] if pred_np.shape[2] == 3 else pred_np
        # Annotate header bar
        label = f"Frame {idx:04d}{'  [KF]' if is_kf else ''}   PSNR: {psnr_score.item():.2f} dB   SSIM: {ssim_score.item():.3f}"
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GT", (6, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Render", (W + gap + 6, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Downscale to max 1920 px wide
        if canvas.shape[1] > 1920:
            scale = 1920 / canvas.shape[1]
            canvas = cv2.resize(canvas,
                                (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(viz_dir, f"{idx:04d}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Save plain rendered image
        cv2.imwrite(os.path.join(render_dir, f"{idx:04d}.jpg"),
                    pred_np[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

    output = dict()
    if psnr_array:
        output["mean_psnr"]  = float(np.mean(psnr_array))
        output["mean_ssim"]  = float(np.mean(ssim_array))
        output["mean_lpips"] = float(np.mean(lpips_array))
    else:
        output = {"mean_psnr": 0.0, "mean_ssim": 0.0, "mean_lpips": 0.0}

    Log(
        f'[{iteration}] mean psnr: {output["mean_psnr"]:.2f}, '
        f'ssim: {output["mean_ssim"]:.3f}, lpips: {output["mean_lpips"]:.4f}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)
    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
