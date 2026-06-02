from typing import Iterable, Optional

import torch

from backend.legacy_360gs.utils.camera_utils import PanoramaCamera
from backend.legacy_360gs.utils.panoramic_renderer import render_erp_direct
from backend.legacy_360gs.utils.slam_utils import (
    _get_panorama_supervision,
    align_mono_depth_to_render_torch,
    erp_area_weight,
)


def _score_stats(tensor: Optional[torch.Tensor]) -> dict:
    if tensor is None or tensor.numel() == 0:
        return {
            "count": 0,
            "nonzero": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }
    flat = tensor.detach().view(-1).to(dtype=torch.float32)
    nz = flat[flat > 0]
    return {
        "count": int(flat.numel()),
        "nonzero": int((flat > 0).sum().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "nonzero_mean": float(nz.mean().item()) if nz.numel() > 0 else 0.0,
    }


def _build_metric_map(viewpoint, render_pkg, config: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    image = render_pkg["render"]
    depth = render_pkg["depth"]
    supervision = _get_panorama_supervision(
        viewpoint,
        config,
        device=image.device,
        dtype=image.dtype,
        depth_shape=depth.shape if depth is not None else None,
    )

    gt_image = supervision["gt_image"]
    mono_depth = supervision["mono_depth"]
    depth_valid = supervision["depth_valid"]
    nonsky_mask = supervision["nonsky_mask"]
    sky_rgb_mask = supervision["sky_rgb_mask"]

    tr_cfg = config.get("Training", {})
    exclude_sky = bool(tr_cfg.get("fastgs_exclude_sky", True))
    valid_rgb_mask = nonsky_mask if exclude_sky else (nonsky_mask | sky_rgb_mask)
    area_w = erp_area_weight(
        image.shape[1],
        image.shape[2],
        device=image.device,
        dtype=image.dtype,
        viewpoint=viewpoint,
        config=config,
    )

    rgb_residual = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
    rgb_score = rgb_residual * area_w
    rgb_thresh = float(tr_cfg.get("fastgs_loss_thresh_rgb", 0.12))
    rgb_metric = valid_rgb_mask & (rgb_score > rgb_thresh)

    depth_metric = torch.zeros_like(valid_rgb_mask, dtype=torch.bool)
    depth_score = torch.zeros_like(area_w)
    depth_bad_ratio = image.new_tensor(0.0)
    if mono_depth is not None and depth is not None:
        mono_depth, _, depth_valid = align_mono_depth_to_render_torch(
            depth,
            mono_depth,
            depth_valid & nonsky_mask,
            config,
        )
        depth_rel = torch.abs(depth - mono_depth) / mono_depth.clamp_min(1.0)
        depth_score = depth_rel * area_w
        depth_thresh = float(tr_cfg.get("fastgs_loss_thresh_depth", 0.08))
        depth_metric = depth_valid & nonsky_mask & (depth_score > depth_thresh)
        valid_depth_count = (depth_valid & nonsky_mask).float().sum().clamp_min(1.0)
        depth_bad_ratio = depth_metric.float().sum() / valid_depth_count

    metric_map = (rgb_metric | depth_metric).squeeze(0).to(dtype=torch.int32)
    rgb_bad_ratio = rgb_metric.float().sum() / valid_rgb_mask.float().sum().clamp_min(1.0)

    if valid_rgb_mask.any():
        photo_rgb = rgb_score[valid_rgb_mask].mean()
    else:
        photo_rgb = image.new_tensor(0.0)
    if depth_metric.any():
        photo_depth = depth_score[depth_valid & nonsky_mask].mean()
    else:
        photo_depth = image.new_tensor(0.0)
    photometric = photo_rgb + photo_depth

    skip_bad_view = bool(tr_cfg.get("fastgs_skip_bad_views", True))
    max_depth_bad_ratio = float(tr_cfg.get("fastgs_max_depth_bad_ratio", 0.50))
    max_rgb_bad_ratio = float(tr_cfg.get("fastgs_max_rgb_bad_ratio", 0.65))
    max_photometric = float(tr_cfg.get("fastgs_max_photometric_loss", 0.25))
    skipped = False
    if skip_bad_view and (
        float(depth_bad_ratio.item()) > max_depth_bad_ratio
        or float(rgb_bad_ratio.item()) > max_rgb_bad_ratio
        or float(photometric.item()) > max_photometric
    ):
        metric_map.zero_()
        skipped = True

    stats = {
        "uid": int(getattr(viewpoint, "uid", -1)),
        "metric_pixels": int(metric_map.sum().item()),
        "rgb_metric_pixels": int(rgb_metric.sum().item()),
        "depth_metric_pixels": int(depth_metric.sum().item()),
        "valid_rgb_pixels": int(valid_rgb_mask.sum().item()),
        "valid_depth_pixels": int((depth_valid & nonsky_mask).sum().item()),
        "photometric_loss": float(photometric.item()),
        "rgb_bad_ratio": float(rgb_bad_ratio.item()),
        "depth_bad_ratio": float(depth_bad_ratio.item()),
        "skipped_bad_view": skipped,
    }
    return metric_map, photometric.detach(), stats


def build_fastgs_metric_map_erp(
    viewpoint,
    gaussians,
    background: torch.Tensor,
    config: dict,
    theta: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    render_pkg: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if render_pkg is None:
        render_pkg = render_erp_direct(
            viewpoint,
            gaussians,
            background,
            theta=theta,
            rho=rho,
        )
    return _build_metric_map(viewpoint, render_pkg, config)


def _sample_replay_views(
    replay_viewpoints: Optional[Iterable],
    replay_limit: int,
) -> list:
    replay_views = [
        vp
        for vp in (replay_viewpoints or [])
        if isinstance(vp, PanoramaCamera)
    ]
    replay_views = sorted(
        replay_views,
        key=lambda vp: int(getattr(vp, "uid", -1)),
        reverse=True,
    )
    if replay_limit <= 0 or len(replay_views) <= replay_limit:
        return replay_views
    return replay_views[:replay_limit]


def compute_gaussian_score_fastgs_erp(
    viewpoints: Iterable,
    gaussians,
    background: torch.Tensor,
    config: dict,
    replay_viewpoints: Optional[Iterable] = None,
    optimize_uids: Optional[set[int]] = None,
):
    tr_cfg = config.get("Training", {})
    replay_limit = int(tr_cfg.get("fastgs_score_replay", 2))
    primary_views = [vp for vp in viewpoints if isinstance(vp, PanoramaCamera)]
    replay_views = _sample_replay_views(replay_viewpoints, replay_limit)
    camlist = primary_views + replay_views
    if not camlist:
        return None, None, {
            "num_views": 0,
            "window_uids": [],
            "replay_uids": [],
            "view_stats": [],
            "importance_stats": _score_stats(None),
            "pruning_stats": _score_stats(None),
        }

    full_metric_counts = None
    full_metric_score = None
    view_stats = []

    for viewpoint in camlist:
        theta = None
        rho = None
        if optimize_uids is not None and int(getattr(viewpoint, "uid", -1)) not in optimize_uids:
            theta = torch.zeros(1, 3, device=background.device)
            rho = torch.zeros(1, 3, device=background.device)

        metric_map, photometric_loss, metric_stats = build_fastgs_metric_map_erp(
            viewpoint,
            gaussians,
            background,
            config,
            theta=theta,
            rho=rho,
        )
        stats_pkg = render_erp_direct(
            viewpoint,
            gaussians,
            background,
            theta=theta,
            rho=rho,
            get_flag=True,
            metric_map=metric_map,
        )

        accum_metric_counts = stats_pkg["accum_metric_counts"].to(dtype=torch.float32)
        metric_stats["gaussians_hit"] = int((accum_metric_counts > 0).sum().item())
        view_stats.append(metric_stats)

        if full_metric_counts is None:
            full_metric_counts = accum_metric_counts.clone()
            full_metric_score = photometric_loss * accum_metric_counts
        else:
            full_metric_counts += accum_metric_counts
            full_metric_score += photometric_loss * accum_metric_counts

    denom = max(len(camlist), 1)
    importance_score = torch.div(
        full_metric_counts.to(dtype=torch.int32),
        denom,
        rounding_mode="floor",
    )

    score_min = float(full_metric_score.min().item())
    score_max = float(full_metric_score.max().item())
    if score_max - score_min > 1e-8:
        pruning_score = (full_metric_score - score_min) / (score_max - score_min)
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    debug_info = {
        "num_views": len(camlist),
        "window_uids": [int(getattr(vp, "uid", -1)) for vp in primary_views],
        "replay_uids": [int(getattr(vp, "uid", -1)) for vp in replay_views],
        "view_stats": view_stats,
        "importance_stats": _score_stats(importance_score),
        "pruning_stats": _score_stats(pruning_score),
    }
    return importance_score, pruning_score, debug_info
