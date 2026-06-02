"""Spherical consistency helpers for ERP panoramic SLAM.

The routines here are intentionally lightweight and dependency-free.  They
adapt the PFGS360 idea of using spherical reprojection/depth consistency as a
reliability signal, while keeping the online SLAM path conservative.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from backend.legacy_360gs.utils.erp_geometry import erp_dense_pixel_center_bearings


def _as_numpy_mask(mask, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = mask_np.astype(bool)
    if shape is not None and mask_np.shape != shape:
        return None
    return mask_np


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample an ERP image at floating pixel coordinates.

    Longitude wraps around; latitude outside the image is marked invalid.
    """
    h, w = image.shape[:2]
    valid = np.isfinite(u) & np.isfinite(v) & (v >= 0.0) & (v <= h - 1.0)
    u = np.mod(u, w)
    v = np.clip(v, 0.0, h - 1.0)

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = u - x0
    wy = v - y0

    vals = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    return vals, valid


def _project_cam_to_erp(points_cam: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points_cam, axis=-1)
    radius_safe = np.maximum(radius, 1e-12)
    x = points_cam[..., 0] / radius_safe
    y = points_cam[..., 1] / radius_safe
    z = points_cam[..., 2] / radius_safe
    lam = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1.0, 1.0))
    u = w * (lam / (2.0 * math.pi) + 0.5) - 0.5
    v = h * (phi / math.pi + 0.5) - 0.5
    return u, v, radius


def _camera_points_from_depth(depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    dx, dy, dz = erp_dense_pixel_center_bearings(h, w)
    return np.stack([dx * depth, dy * depth, dz * depth], axis=-1)


def _depth_abs_rel(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    *,
    rel_mode: str = "mono",
) -> tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(render_depth - mono_depth)
    if str(rel_mode).lower() in {"symmetric", "sym", "balanced"}:
        denom = np.maximum(render_depth + mono_depth, 1e-3)
        rel = 2.0 * abs_err / denom
    else:
        rel = abs_err / np.maximum(mono_depth, 1e-3)
    return abs_err, rel


def _depth_edge_guard(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    valid: np.ndarray,
    *,
    edge_rel_thresh: float = 0.08,
) -> np.ndarray:
    """Return pixels away from large local depth discontinuities."""
    h, w = valid.shape
    edge = np.zeros((h, w), dtype=bool)

    def _accumulate_edges(depth: np.ndarray) -> None:
        depth_valid = valid & np.isfinite(depth) & (depth > 0.01)
        if w > 1:
            right = np.roll(depth, -1, axis=1)
            right_valid = depth_valid & np.roll(depth_valid, -1, axis=1)
            jump = np.abs(depth - right) / np.maximum(np.maximum(depth, right), 1e-3)
            e = right_valid & (jump > float(edge_rel_thresh))
            edge[:] |= e | np.roll(e, 1, axis=1)
        if h > 1:
            down = np.empty_like(depth)
            down[:-1, :] = depth[1:, :]
            down[-1, :] = depth[-1, :]
            down_valid = np.zeros_like(depth_valid)
            down_valid[:-1, :] = depth_valid[:-1, :] & depth_valid[1:, :]
            jump = np.abs(depth - down) / np.maximum(np.maximum(depth, down), 1e-3)
            e = down_valid & (jump > float(edge_rel_thresh))
            edge[:] |= e
            edge[1:, :] |= e[:-1, :]

    _accumulate_edges(render_depth)
    _accumulate_edges(mono_depth)
    dilated = edge.copy()
    if w > 1:
        dilated |= np.roll(edge, 1, axis=1) | np.roll(edge, -1, axis=1)
    if h > 1:
        dilated[1:, :] |= edge[:-1, :]
        dilated[:-1, :] |= edge[1:, :]
    return valid & (~dilated)


def warp_depth_to_reference_erp(
    src_depth: np.ndarray,
    src_w2c: np.ndarray,
    ref_w2c: np.ndarray,
    *,
    src_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Forward-warp an ERP depth map into a reference camera.

    Multiple source pixels can land on the same reference pixel; keep the
    nearest radial depth as a simple z-buffer.
    """
    src_depth = np.asarray(src_depth, dtype=np.float32)
    if src_depth.ndim == 3:
        src_depth = src_depth[0]
    if src_depth.ndim != 2:
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, empty.astype(bool), {"valid_projected": 0, "coverage": 0.0}

    h, w = src_depth.shape
    valid = _as_numpy_mask(src_valid, (h, w))
    if valid is None:
        valid = np.isfinite(src_depth) & (src_depth > 0.01)
    else:
        valid = valid & np.isfinite(src_depth) & (src_depth > 0.01)

    src_pts = _camera_points_from_depth(src_depth)
    src_R = np.asarray(src_w2c[:3, :3], dtype=np.float64)
    src_T = np.asarray(src_w2c[:3, 3], dtype=np.float64)
    ref_R = np.asarray(ref_w2c[:3, :3], dtype=np.float64)
    ref_T = np.asarray(ref_w2c[:3, 3], dtype=np.float64)

    world = np.einsum("ij,hwj->hwi", src_R.T, src_pts - src_T.reshape(1, 1, 3))
    ref_pts = np.einsum("ij,hwj->hwi", ref_R, world) + ref_T.reshape(1, 1, 3)
    u_ref, v_ref, ref_radius = _project_cam_to_erp(ref_pts, h, w)

    valid_proj = (
        valid
        & np.isfinite(u_ref)
        & np.isfinite(v_ref)
        & np.isfinite(ref_radius)
        & (ref_radius > 0.01)
        & (v_ref >= 0.0)
        & (v_ref <= h - 1.0)
    )
    out_flat = np.full((h * w,), np.inf, dtype=np.float32)
    if np.any(valid_proj):
        xs = np.mod(np.round(u_ref[valid_proj]).astype(np.int64), w)
        ys = np.clip(np.round(v_ref[valid_proj]).astype(np.int64), 0, h - 1)
        flat = ys * w + xs
        np.minimum.at(out_flat, flat, ref_radius[valid_proj].astype(np.float32))

    warped = out_flat.reshape(h, w)
    warped_valid = np.isfinite(warped)
    warped = np.where(warped_valid, warped, 0.0).astype(np.float32)
    return warped, warped_valid, {
        "valid_projected": int(valid_proj.sum()),
        "coverage": float(warped_valid.mean()),
    }


def spherical_depth_consistency_mask(
    src_depth: np.ndarray,
    src_w2c: np.ndarray,
    ref_depth: np.ndarray,
    ref_w2c: np.ndarray,
    *,
    src_valid: np.ndarray | None = None,
    ref_valid: np.ndarray | None = None,
    eps_tan: float = 0.008,
    eps_depth_rel: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return pixels in ``src`` whose depth is consistent with ``ref``.

    Consistency is checked by projecting a source depth point into the reference
    ERP, sampling reference depth, backprojecting to source, and comparing both
    tangent angular error and radial depth.
    """
    src_depth = np.asarray(src_depth, dtype=np.float32)
    ref_depth = np.asarray(ref_depth, dtype=np.float32)
    if src_depth.ndim != 2 or ref_depth.shape != src_depth.shape:
        mask = np.zeros_like(src_depth, dtype=bool)
        return mask, {"coverage": 0.0, "valid_projected": 0, "reason": "shape_mismatch"}

    h, w = src_depth.shape
    src_valid_np = _as_numpy_mask(src_valid, (h, w))
    ref_valid_np = _as_numpy_mask(ref_valid, (h, w))
    if src_valid_np is None:
        src_valid_np = np.isfinite(src_depth) & (src_depth > 0.01)
    if ref_valid_np is None:
        ref_valid_np = np.isfinite(ref_depth) & (ref_depth > 0.01)

    src_pts = _camera_points_from_depth(src_depth)
    src_R = np.asarray(src_w2c[:3, :3], dtype=np.float64)
    src_T = np.asarray(src_w2c[:3, 3], dtype=np.float64)
    ref_R = np.asarray(ref_w2c[:3, :3], dtype=np.float64)
    ref_T = np.asarray(ref_w2c[:3, 3], dtype=np.float64)

    world = np.einsum("ij,hwj->hwi", src_R.T, src_pts - src_T.reshape(1, 1, 3))
    ref_pts_from_src = np.einsum("ij,hwj->hwi", ref_R, world) + ref_T.reshape(1, 1, 3)
    u_ref, v_ref, ref_radius_pred = _project_cam_to_erp(ref_pts_from_src, h, w)

    sampled_ref_depth, valid_proj = _bilinear_sample(ref_depth, u_ref, v_ref)
    sampled_ref_valid_f, valid_valid = _bilinear_sample(ref_valid_np.astype(np.float32), u_ref, v_ref)
    sampled_ref_valid = sampled_ref_valid_f > 0.5

    ref_bearing = ref_pts_from_src / np.maximum(ref_radius_pred[..., None], 1e-12)
    ref_pts = ref_bearing * sampled_ref_depth[..., None]
    world_back = np.einsum("ij,hwj->hwi", ref_R.T, ref_pts - ref_T.reshape(1, 1, 3))
    src_pts_back = np.einsum("ij,hwj->hwi", src_R, world_back) + src_T.reshape(1, 1, 3)
    back_radius = np.linalg.norm(src_pts_back, axis=-1)

    src_unit = src_pts / np.maximum(np.linalg.norm(src_pts, axis=-1, keepdims=True), 1e-12)
    back_unit = src_pts_back / np.maximum(back_radius[..., None], 1e-12)
    dot = np.clip(np.sum(src_unit * back_unit, axis=-1), -1.0 + 1e-7, 1.0 - 1e-7)
    tan_err = 2.0 * np.sqrt(np.maximum(1.0 - dot, 0.0) / np.maximum(1.0 + dot, 1e-12))
    depth_rel = np.abs(back_radius - src_depth) / np.maximum(src_depth, 1e-3)

    mask = (
        src_valid_np
        & sampled_ref_valid
        & valid_proj
        & valid_valid
        & np.isfinite(tan_err)
        & np.isfinite(depth_rel)
        & (tan_err <= float(eps_tan))
        & (depth_rel <= float(eps_depth_rel))
    )
    stats = {
        "coverage": float(mask.mean()),
        "valid_projected": int((src_valid_np & valid_proj & sampled_ref_valid).sum()),
        "mean_tan_err": float(np.nanmean(tan_err[mask])) if mask.any() else 0.0,
        "mean_depth_rel": float(np.nanmean(depth_rel[mask])) if mask.any() else 0.0,
    }
    return mask, stats


def depth_render_consistency_mask(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    opacity_min: float = 0.15,
    rel_thresh: float = 0.10,
    abs_thresh: float = 0.0,
    rel_mode: str = "mono",
) -> tuple[np.ndarray, dict[str, Any]]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    mono_depth = np.asarray(mono_depth, dtype=np.float32)
    if render_depth.ndim == 3:
        render_depth = render_depth[0]
    if mono_depth.ndim == 3:
        mono_depth = mono_depth[0]
    valid = np.isfinite(render_depth) & np.isfinite(mono_depth) & (render_depth > 0.01) & (mono_depth > 0.01)
    mask_np = _as_numpy_mask(valid_mask, render_depth.shape)
    if mask_np is not None:
        valid &= mask_np
    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float32)
        if op.ndim == 3:
            op = op[0]
        if op.shape == render_depth.shape:
            valid &= op > float(opacity_min)
    abs_err, rel = _depth_abs_rel(render_depth, mono_depth, rel_mode=rel_mode)
    if float(abs_thresh) > 0.0:
        mask = valid & ((rel <= float(rel_thresh)) | (abs_err <= float(abs_thresh)))
    else:
        mask = valid & (rel <= float(rel_thresh))
    valid_count = int(valid.sum())
    consistent_count = int(mask.sum())
    return mask, {
        "coverage": float(mask.mean()),
        "valid_ratio": float(valid.mean()),
        "valid_pixels": valid_count,
        "consistent_pixels": consistent_count,
        "consistent_valid_ratio": float(consistent_count / max(valid_count, 1)),
        "mean_rel": float(rel[mask].mean()) if mask.any() else 0.0,
        "mean_rel_valid": float(rel[valid].mean()) if valid.any() else 0.0,
        "median_rel_valid": float(np.median(rel[valid])) if valid.any() else 0.0,
        "mean_abs_valid": float(abs_err[valid].mean()) if valid.any() else 0.0,
        "depth_rel_mode": str(rel_mode),
        "depth_abs_thresh": float(abs_thresh),
    }


def depth_render_novelty_mask(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    opacity_min: float = 0.15,
    rel_thresh: float = 0.10,
    abs_thresh: float = 0.0,
    rel_mode: str = "mono",
    edge_guard: bool = False,
    edge_rel_thresh: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return pixels where the current map does not explain the DAP depth."""
    render_depth = np.asarray(render_depth, dtype=np.float32)
    mono_depth = np.asarray(mono_depth, dtype=np.float32)
    if render_depth.ndim == 3:
        render_depth = render_depth[0]
    if mono_depth.ndim == 3:
        mono_depth = mono_depth[0]
    valid = np.isfinite(mono_depth) & (mono_depth > 0.01)
    mask_np = _as_numpy_mask(valid_mask, mono_depth.shape)
    if mask_np is not None:
        valid &= mask_np

    render_valid = np.isfinite(render_depth) & (render_depth > 0.01)
    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float32)
        if op.ndim == 3:
            op = op[0]
        if op.shape == render_depth.shape:
            render_valid &= op > float(opacity_min)

    abs_err, rel = _depth_abs_rel(render_depth, mono_depth, rel_mode=rel_mode)
    coverage_hole = valid & (~render_valid)
    conflict_valid = valid & render_valid
    edge_guard_removed = 0
    if bool(edge_guard):
        guarded = _depth_edge_guard(
            render_depth,
            mono_depth,
            conflict_valid,
            edge_rel_thresh=edge_rel_thresh,
        )
        edge_guard_removed = int(conflict_valid.sum() - guarded.sum())
        conflict_valid &= guarded
    depth_conflict = conflict_valid & (rel > float(rel_thresh))
    if float(abs_thresh) > 0.0:
        depth_conflict &= abs_err > float(abs_thresh)
    novelty = coverage_hole | depth_conflict
    valid_count = int(valid.sum())
    return novelty, {
        "novelty_ratio": float(novelty.sum() / max(valid_count, 1)),
        "valid_ratio": float(valid.mean()),
        "valid_pixels": valid_count,
        "novelty_pixels": int(novelty.sum()),
        "coverage_hole_pixels": int(coverage_hole.sum()),
        "coverage_hole_ratio": float(coverage_hole.sum() / max(valid_count, 1)),
        "depth_conflict_pixels": int(depth_conflict.sum()),
        "depth_conflict_ratio": float(depth_conflict.sum() / max(valid_count, 1)),
        "mean_rel": float(rel[novelty].mean()) if novelty.any() else 0.0,
        "mean_abs": float(abs_err[novelty].mean()) if novelty.any() else 0.0,
        "depth_abs_thresh": float(abs_thresh),
        "depth_rel_mode": str(rel_mode),
        "edge_guard_enabled": bool(edge_guard),
        "edge_guard_removed_pixels": int(edge_guard_removed),
    }


def depth_render_novelty_components(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    opacity_min: float = 0.15,
    rel_thresh: float = 0.10,
    abs_thresh: float = 0.0,
    rel_mode: str = "mono",
    edge_guard: bool = False,
    edge_rel_thresh: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return all/coverage/depth-conflict novelty masks.

    The all-novelty mask is useful for diagnostics and keyframe decisions.
    For DVO-driven map growth, the coverage-hole component is usually safer:
    it asks for new Gaussians only where the current map has no reliable
    support, while depth conflicts can be handled by refinement or pruning.
    """
    novelty, stats = depth_render_novelty_mask(
        render_depth,
        mono_depth,
        valid_mask=valid_mask,
        opacity=opacity,
        opacity_min=opacity_min,
        rel_thresh=rel_thresh,
        abs_thresh=abs_thresh,
        rel_mode=rel_mode,
        edge_guard=edge_guard,
        edge_rel_thresh=edge_rel_thresh,
    )

    render_depth = np.asarray(render_depth, dtype=np.float32)
    mono_depth = np.asarray(mono_depth, dtype=np.float32)
    if render_depth.ndim == 3:
        render_depth = render_depth[0]
    if mono_depth.ndim == 3:
        mono_depth = mono_depth[0]
    valid = np.isfinite(mono_depth) & (mono_depth > 0.01)
    mask_np = _as_numpy_mask(valid_mask, mono_depth.shape)
    if mask_np is not None:
        valid &= mask_np
    render_valid = np.isfinite(render_depth) & (render_depth > 0.01)
    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float32)
        if op.ndim == 3:
            op = op[0]
        if op.shape == render_depth.shape:
            render_valid &= op > float(opacity_min)
    abs_err, rel = _depth_abs_rel(render_depth, mono_depth, rel_mode=rel_mode)
    coverage_hole = valid & (~render_valid)
    conflict_valid = valid & render_valid
    if bool(edge_guard):
        conflict_valid &= _depth_edge_guard(
            render_depth,
            mono_depth,
            conflict_valid,
            edge_rel_thresh=edge_rel_thresh,
        )
    depth_conflict = conflict_valid & (rel > float(rel_thresh))
    if float(abs_thresh) > 0.0:
        depth_conflict &= abs_err > float(abs_thresh)
    return novelty, coverage_hole, depth_conflict, stats


def sample_mask_at_uv(mask, xy: np.ndarray, *, default: bool = False) -> np.ndarray:
    mask_np = _as_numpy_mask(mask)
    if mask_np is None or xy is None or len(xy) == 0:
        return np.full((0 if xy is None else len(xy),), bool(default), dtype=bool)
    h, w = mask_np.shape
    pts = np.asarray(xy)
    xs = np.mod(np.round(pts[:, 0]).astype(np.int64), w)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)
    return mask_np[ys, xs].astype(bool)


def latitude_weights_for_uv(xy: np.ndarray, height: int) -> np.ndarray:
    if xy is None or len(xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    v = np.asarray(xy, dtype=np.float32)[:, 1]
    lat = math.pi * ((v + 0.5) / float(height) - 0.5)
    return np.clip(np.cos(lat), 0.0, 1.0).astype(np.float32)


def combine_masks(*masks, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    out = None
    for mask in masks:
        mask_np = _as_numpy_mask(mask, shape)
        if mask_np is None:
            continue
        out = mask_np.copy() if out is None else (out & mask_np)
    return out


def build_dia_insert_mask(
    render_depth: np.ndarray,
    mono_depth: np.ndarray,
    *,
    valid_insert: np.ndarray,
    opacity: np.ndarray | None = None,
    rel_thresh: float = 0.08,
    opacity_min: float = 0.15,
    max_insert_ratio: float = 0.35,
    rng: np.random.Generator | None = None,
    apply_cap: bool = True,
    far_depth_start: float | None = None,
    far_rel_thresh_mult: float = 1.0,
    far_max_insert_ratio: float | None = None,
    disable_insert_beyond: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a conservative depth-inlier insertion mask.

    Candidate pixels are valid DAP pixels in regions where the current map is
    absent or inconsistent with aligned DAP.  The mask is capped by
    ``max_insert_ratio`` to prevent a single keyframe from exploding the map.
    """
    render_depth = np.asarray(render_depth, dtype=np.float32)
    mono_depth = np.asarray(mono_depth, dtype=np.float32)
    valid = _as_numpy_mask(valid_insert, mono_depth.shape)
    if valid is None:
        valid = np.isfinite(mono_depth) & (mono_depth > 0.01)

    render_valid = np.isfinite(render_depth) & (render_depth > 0.01)
    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float32)
        if op.ndim == 3:
            op = op[0]
        if op.shape == render_depth.shape:
            render_valid &= op > float(opacity_min)

    far_start = None
    if far_depth_start is not None and np.isfinite(float(far_depth_start)):
        far_start = float(far_depth_start)
    far_mask = np.zeros_like(valid, dtype=bool)
    if far_start is not None:
        far_mask = valid & np.isfinite(mono_depth) & (mono_depth >= far_start)
    beyond_mask = np.zeros_like(valid, dtype=bool)
    if disable_insert_beyond is not None and np.isfinite(float(disable_insert_beyond)):
        beyond_mask = valid & np.isfinite(mono_depth) & (mono_depth >= float(disable_insert_beyond))

    rel = np.abs(render_depth - mono_depth) / np.maximum(mono_depth, 1e-3)
    rel_thresh_map = np.full_like(mono_depth, float(rel_thresh), dtype=np.float32)
    if far_start is not None:
        rel_thresh_map[far_mask] = float(rel_thresh) * max(1.0, float(far_rel_thresh_mult))
    candidate = valid & ((~render_valid) | (rel > rel_thresh_map))
    candidate_before_beyond = candidate.copy()
    candidate &= ~beyond_mask
    n_candidate = int(candidate.sum())
    max_pixels = int(max(1, valid.sum() * float(max_insert_ratio))) if int(valid.sum()) > 0 else 0
    capped = False
    if apply_cap:
        candidate, cap_stats = cap_dia_insert_mask(
            candidate,
            valid,
            max_insert_ratio=max_insert_ratio,
            rng=rng,
            depth_map=mono_depth,
            far_depth_start=far_start,
            far_max_insert_ratio=far_max_insert_ratio,
            disable_insert_beyond=disable_insert_beyond,
        )
        capped = bool(cap_stats.get("capped", False))
        max_pixels = int(cap_stats.get("max_insert_pixels", max_pixels))

    return candidate, {
        "valid_pixels": int(valid.sum()),
        "far_valid_pixels": int(far_mask.sum()),
        "far_start_depth": float(far_start) if far_start is not None else 0.0,
        "far_rel_thresh_mult": float(far_rel_thresh_mult),
        "disable_insert_beyond_depth": (
            float(disable_insert_beyond)
            if disable_insert_beyond is not None and np.isfinite(float(disable_insert_beyond))
            else 0.0
        ),
        "beyond_insert_suppressed_pixels": int((candidate_before_beyond & beyond_mask).sum()),
        "candidate_pixels": n_candidate,
        "candidate_pixels_before_cap": n_candidate,
        "max_insert_pixels": max_pixels,
        "insert_pixels": int(candidate.sum()),
        "far_insert_pixels": int((candidate & far_mask).sum()),
        "insert_ratio": float(candidate.mean()),
        "capped": bool(capped),
    }


def cap_dia_insert_mask(
    candidate_mask: np.ndarray,
    valid_insert: np.ndarray | None,
    *,
    max_insert_ratio: float,
    rng: np.random.Generator | None = None,
    depth_map: np.ndarray | None = None,
    far_depth_start: float | None = None,
    far_max_insert_ratio: float | None = None,
    disable_insert_beyond: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Randomly cap an insertion mask by a ratio of valid pixels."""
    candidate = np.asarray(candidate_mask, dtype=bool).copy()
    valid = _as_numpy_mask(valid_insert, candidate.shape)
    if valid is None:
        valid = np.ones_like(candidate, dtype=bool)
    depth = None
    if depth_map is not None:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        if depth.shape != candidate.shape:
            depth = None
    beyond = np.zeros_like(candidate, dtype=bool)
    if depth is not None and disable_insert_beyond is not None and np.isfinite(float(disable_insert_beyond)):
        beyond = np.isfinite(depth) & (depth >= float(disable_insert_beyond))
        candidate &= ~beyond
        valid = valid & ~beyond

    far_start = None
    if depth is not None and far_depth_start is not None and np.isfinite(float(far_depth_start)):
        far_start = float(far_depth_start)

    def _cap_single(mask: np.ndarray, valid_mask: np.ndarray, ratio: float) -> tuple[np.ndarray, dict[str, Any]]:
        mask = mask.copy()
        n_mask = int(mask.sum())
        n_valid_mask = int(valid_mask.sum())
        max_mask_pixels = int(max(1, n_valid_mask * float(ratio))) if n_valid_mask > 0 else 0
        was_capped = False
        if max_mask_pixels > 0 and n_mask > max_mask_pixels:
            local_rng = rng or np.random.default_rng()
            ids = np.flatnonzero(mask.reshape(-1))
            keep_ids = local_rng.choice(ids, size=max_mask_pixels, replace=False)
            capped_mask = np.zeros(mask.size, dtype=bool)
            capped_mask[keep_ids] = True
            mask = capped_mask.reshape(mask.shape)
            was_capped = True
        elif max_mask_pixels <= 0:
            mask.fill(False)
        return mask, {
            "candidate_pixels_before_cap": n_mask,
            "valid_pixels": n_valid_mask,
            "max_insert_pixels": max_mask_pixels,
            "insert_pixels": int(mask.sum()),
            "capped": bool(was_capped),
        }

    n_candidate = int(candidate.sum())
    n_valid = int(valid.sum())
    if far_start is not None and far_max_insert_ratio is not None:
        far_valid = valid & np.isfinite(depth) & (depth >= far_start)
        near_valid = valid & ~far_valid
        near_candidate = candidate & near_valid
        far_candidate = candidate & far_valid
        near_candidate, near_stats = _cap_single(
            near_candidate, near_valid, float(max_insert_ratio)
        )
        far_candidate, far_stats = _cap_single(
            far_candidate, far_valid, float(far_max_insert_ratio)
        )
        candidate = near_candidate | far_candidate
        max_pixels = int(near_stats["max_insert_pixels"] + far_stats["max_insert_pixels"])
        capped = bool(near_stats["capped"] or far_stats["capped"])
        return candidate, {
            "candidate_pixels_before_cap": n_candidate,
            "valid_pixels": n_valid,
            "max_insert_pixels": max_pixels,
            "insert_pixels": int(candidate.sum()),
            "insert_ratio": float(candidate.mean()) if candidate.size else 0.0,
            "far_valid_pixels": int(far_valid.sum()),
            "far_candidate_pixels_before_cap": int(far_stats["candidate_pixels_before_cap"]),
            "far_max_insert_pixels": int(far_stats["max_insert_pixels"]),
            "far_insert_pixels": int((candidate & far_valid).sum()),
            "beyond_insert_suppressed_pixels": int((np.asarray(candidate_mask, dtype=bool) & beyond).sum()),
            "capped": bool(capped),
        }

    candidate, stats = _cap_single(candidate, valid, float(max_insert_ratio))
    max_pixels = int(stats["max_insert_pixels"])
    capped = bool(stats["capped"])
    return candidate, {
        "candidate_pixels_before_cap": n_candidate,
        "valid_pixels": n_valid,
        "max_insert_pixels": max_pixels,
        "insert_pixels": int(candidate.sum()),
        "insert_ratio": float(candidate.mean()) if candidate.size else 0.0,
        "beyond_insert_suppressed_pixels": int((np.asarray(candidate_mask, dtype=bool) & beyond).sum()),
        "capped": bool(capped),
    }


def cap_insert_mask_by_score(
    candidate_mask: np.ndarray,
    valid_insert: np.ndarray | None,
    *,
    max_insert_ratio: float | None = None,
    max_insert_pixels: int | None = None,
    score_map: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cap candidate pixels by ratio/absolute budget, preferring high scores."""
    candidate = np.asarray(candidate_mask, dtype=bool).copy()
    valid = _as_numpy_mask(valid_insert, candidate.shape)
    if valid is None:
        valid = np.ones_like(candidate, dtype=bool)
    candidate &= valid

    n_candidate = int(candidate.sum())
    n_valid = int(valid.sum())
    ratio_cap = n_candidate
    ratio_for_stats = -1.0
    if max_insert_ratio is not None:
        try:
            ratio = float(max_insert_ratio)
        except (TypeError, ValueError):
            ratio = float("nan")
        if np.isfinite(ratio) and ratio > 0.0:
            ratio_for_stats = float(ratio)
            ratio_cap = int(max(1, n_valid * ratio)) if n_valid > 0 else 0

    absolute_cap = n_candidate
    if max_insert_pixels is not None:
        try:
            absolute_cap = int(max_insert_pixels)
        except (TypeError, ValueError):
            absolute_cap = n_candidate
        if absolute_cap < 0:
            absolute_cap = n_candidate

    max_pixels = int(max(0, min(ratio_cap, absolute_cap)))
    score_used = False
    if max_pixels <= 0:
        candidate.fill(False)
    elif n_candidate > max_pixels:
        ids = np.flatnonzero(candidate.reshape(-1))
        keep_ids = None
        if score_map is not None:
            score = np.asarray(score_map, dtype=np.float32)
            if score.ndim == 3:
                score = score[0]
            if tuple(score.shape) == tuple(candidate.shape):
                flat_score = score.reshape(-1)[ids]
                finite = np.isfinite(flat_score)
                if finite.any():
                    ranked_score = np.where(finite, flat_score, -np.inf)
                    top_local = np.argpartition(ranked_score, -max_pixels)[-max_pixels:]
                    keep_ids = ids[top_local]
                    score_used = True
        if keep_ids is None:
            local_rng = rng or np.random.default_rng()
            keep_ids = local_rng.choice(ids, size=max_pixels, replace=False)
        capped = np.zeros(candidate.size, dtype=bool)
        capped[keep_ids] = True
        candidate = capped.reshape(candidate.shape)

    return candidate, {
        "candidate_pixels_before_cap": n_candidate,
        "valid_pixels": n_valid,
        "max_insert_ratio": float(ratio_for_stats),
        "max_insert_pixels": max_pixels,
        "max_insert_pixels_by_ratio": int(ratio_cap),
        "max_insert_pixels_by_absolute": int(absolute_cap),
        "insert_pixels": int(candidate.sum()),
        "insert_ratio": float(candidate.mean()) if candidate.size else 0.0,
        "capped": bool(n_candidate > max_pixels),
        "cap_score_used": bool(score_used),
    }


def depth_projection_support_mask(
    src_depth: np.ndarray,
    src_w2c: np.ndarray,
    ref_depth: np.ndarray,
    ref_w2c: np.ndarray,
    *,
    src_valid: np.ndarray | None = None,
    ref_valid: np.ndarray | None = None,
    rel_thresh: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Check whether source-depth points agree with a reference ERP depth map.

    The source depth is backprojected to 3D, transformed into the reference
    camera, and compared against the reference aligned mono depth at the
    projected ERP pixels using only radial relative depth error.
    """
    src_depth = np.asarray(src_depth, dtype=np.float32)
    ref_depth = np.asarray(ref_depth, dtype=np.float32)
    if src_depth.ndim == 3:
        src_depth = src_depth[0]
    if ref_depth.ndim == 3:
        ref_depth = ref_depth[0]
    if src_depth.ndim != 2 or ref_depth.shape != src_depth.shape:
        mask = np.zeros_like(src_depth if src_depth.ndim == 2 else ref_depth, dtype=bool)
        return mask, {
            "supported_pixels": 0,
            "valid_projected": 0,
            "mean_rel_valid": 0.0,
            "reason": "shape_mismatch",
        }

    h, w = src_depth.shape
    src_valid_np = _as_numpy_mask(src_valid, (h, w))
    if src_valid_np is None:
        src_valid_np = np.isfinite(src_depth) & (src_depth > 0.01)
    else:
        src_valid_np = src_valid_np & np.isfinite(src_depth) & (src_depth > 0.01)
    ref_valid_np = _as_numpy_mask(ref_valid, (h, w))
    if ref_valid_np is None:
        ref_valid_np = np.isfinite(ref_depth) & (ref_depth > 0.01)
    else:
        ref_valid_np = ref_valid_np & np.isfinite(ref_depth) & (ref_depth > 0.01)

    src_pts = _camera_points_from_depth(src_depth)
    src_R = np.asarray(src_w2c[:3, :3], dtype=np.float64)
    src_T = np.asarray(src_w2c[:3, 3], dtype=np.float64)
    ref_R = np.asarray(ref_w2c[:3, :3], dtype=np.float64)
    ref_T = np.asarray(ref_w2c[:3, 3], dtype=np.float64)

    world = np.einsum("ij,hwj->hwi", src_R.T, src_pts - src_T.reshape(1, 1, 3))
    ref_pts = np.einsum("ij,hwj->hwi", ref_R, world) + ref_T.reshape(1, 1, 3)
    u_ref, v_ref, pred_ref_depth = _project_cam_to_erp(ref_pts, h, w)

    sampled_ref_depth, valid_proj = _bilinear_sample(ref_depth, u_ref, v_ref)
    sampled_ref_valid_f, valid_valid = _bilinear_sample(
        ref_valid_np.astype(np.float32),
        u_ref,
        v_ref,
    )
    sampled_ref_valid = sampled_ref_valid_f > 0.5
    rel = np.abs(pred_ref_depth - sampled_ref_depth) / np.maximum(sampled_ref_depth, 1e-3)
    valid_compare = (
        src_valid_np
        & valid_proj
        & valid_valid
        & sampled_ref_valid
        & np.isfinite(pred_ref_depth)
        & np.isfinite(sampled_ref_depth)
        & np.isfinite(rel)
        & (pred_ref_depth > 0.01)
        & (sampled_ref_depth > 0.01)
    )
    mask = valid_compare & (rel <= float(rel_thresh))
    return mask, {
        "supported_pixels": int(mask.sum()),
        "valid_projected": int(valid_compare.sum()),
        "mean_rel_valid": float(rel[valid_compare].mean()) if valid_compare.any() else 0.0,
        "mean_rel_supported": float(rel[mask].mean()) if mask.any() else 0.0,
    }
