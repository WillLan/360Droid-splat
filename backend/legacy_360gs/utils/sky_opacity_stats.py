"""Diagnostic: opacity distribution of Gaussians whose ERP projection lands
in the ``sky_mask`` region of recent keyframes, stratified by ERP latitude
bands.

Used to choose a safe per-band opacity threshold for sky-direction anchor
pruning, without harming far-ground anchors near the horizon that may be
mis-classified as sky by the segmentation model.

Conventions
-----------
World -> body-cam: ``p_cam = R @ p_world + T`` (matches
``pano_scaffold_model._create_pcd_from_erp_depth``).

Body-cam ERP direction encoding (matches
``NeuralSkyMLP._erp_pixel_directions`` and
``utils.erp_geometry.erp_dense_pixel_center_bearings``)::

    d_x = cos(el) * sin(az)
    d_y = -sin(el)                   # y points DOWN -> el>0 is upward (sky)
    d_z = cos(el) * cos(az)
    u   = (az + pi) / (2*pi) * W     # az in [-pi, pi)
    v   = (pi/2 - el) / pi * H       # el in (pi/2, -pi/2]

So north latitude (lat_deg > 0) is the upper hemisphere, the place we
expect "real" sky to live. Low-lat in_sky hits typically come from the
segmentation model misclassifying distant ground/buildings/trees as sky.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from backend.legacy_360gs.utils.panoramic_renderer import _get_body_cam_sky_mask


DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 30.0),
    (30.0, 60.0),
    (60.0, 90.0),
)
DEFAULT_OPACITY_THRESHOLDS: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80)


def _world_to_cam(
    xyz_world: torch.Tensor, R: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    """``p_cam = R @ p_world + T``. Inputs in arbitrary device/dtype."""
    R = R.to(device=xyz_world.device, dtype=xyz_world.dtype)
    T = T.to(device=xyz_world.device, dtype=xyz_world.dtype).view(3)
    return xyz_world @ R.t() + T


def _erp_uv_lat_from_cam(
    p_cam: torch.Tensor, H: int, W: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map per-anchor body-cam vectors to ERP integer pixel ``(u, v)`` and
    signed latitude in degrees. Returns ``(u, v, lat_deg)``.
    """
    norms = p_cam.norm(dim=1).clamp_min(1e-6)
    d = p_cam / norms.unsqueeze(-1)
    az = torch.atan2(d[:, 0], d[:, 2])
    el = -torch.asin(d[:, 1].clamp(-1.0, 1.0))
    u = ((az + math.pi) / (2.0 * math.pi) * float(W)).long().clamp(0, W - 1)
    v = ((math.pi * 0.5 - el) / math.pi * float(H)).long().clamp(0, H - 1)
    lat_deg = el * (180.0 / math.pi)
    return u, v, lat_deg


def _summarize(
    arr: np.ndarray, thresholds: Sequence[float]
) -> dict:
    if arr.size == 0:
        return {"n_samples": 0}
    ps = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    out = {
        "n_samples": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(ps[0]),
        "p05": float(ps[1]),
        "p10": float(ps[2]),
        "p25": float(ps[3]),
        "p50": float(ps[4]),
        "p75": float(ps[5]),
        "p90": float(ps[6]),
        "p95": float(ps[7]),
        "p99": float(ps[8]),
    }
    for thr in thresholds:
        key = f"{thr:.2f}".rstrip("0").rstrip(".")
        out[f"n_le_{key}"] = int((arr <= thr).sum())
        out[f"frac_le_{key}"] = float((arr <= thr).mean())
    return out


def collect_sky_opacity_stats(
    gaussians,
    keyframes: Sequence,
    frame_idx: int,
    out_path: Optional[str],
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    opacity_thresholds: Sequence[float] = DEFAULT_OPACITY_THRESHOLDS,
    extra_meta: Optional[dict] = None,
) -> Optional[dict]:
    """For each KF in ``keyframes``, project every anchor's centre into the
    body-cam ERP, look up ``sky_mask`` at that pixel and bin the hits by
    latitude band. Emit a single JSON line with per-band opacity
    distribution + threshold-prune counts.

    A single anchor can contribute multiple samples (one per KF where its
    projection falls into sky_mask), so the *n_samples* aggregates over
    (anchor, KF) pairs. We also report *n_unique_anchors* per band.

    Returns the constructed payload (also written to ``out_path`` if given).
    """
    if not keyframes:
        return None

    xyz_attr = getattr(gaussians, "_xyz", None)
    opa_attr = getattr(gaussians, "_opacity", None)
    if xyz_attr is None or opa_attr is None:
        return None
    if xyz_attr.numel() == 0:
        return None

    device = xyz_attr.device
    xyz_world = xyz_attr.detach().to(device)
    opacity = torch.sigmoid(opa_attr.detach().to(device)).view(-1)
    is_sky_anchor = getattr(gaussians, "_is_sky_anchor", None)
    if isinstance(is_sky_anchor, torch.Tensor) and is_sky_anchor.shape[0] == xyz_world.shape[0]:
        is_sky_anchor_bool = is_sky_anchor.to(device=device, dtype=torch.bool).view(-1)
    else:
        is_sky_anchor_bool = torch.zeros(xyz_world.shape[0], dtype=torch.bool, device=device)
    N = int(xyz_world.shape[0])

    band_keys = [f"lat_{int(lo):02d}_{int(hi):02d}" for (lo, hi) in bands]
    band_samples = {k: {"op": [], "anchors": set()} for k in band_keys}
    baseline_samples = {"op": [], "anchors": set()}

    n_kf_used = 0
    for cam in keyframes:
        H = int(getattr(cam, "image_height", 0) or 0)
        W = int(getattr(cam, "image_width", 0) or 0)
        if H <= 0 or W <= 0:
            continue
        sky_mask = _get_body_cam_sky_mask(cam, H, W, device=device, dtype=torch.float32)
        if sky_mask is None:
            continue
        sky_mask_bool = sky_mask.bool().view(H, W)

        R = cam.R
        T = cam.T
        if not isinstance(R, torch.Tensor) or not isinstance(T, torch.Tensor):
            continue
        p_cam = _world_to_cam(xyz_world, R, T)
        u, v, lat_deg = _erp_uv_lat_from_cam(p_cam, H, W)

        hit = sky_mask_bool[v, u] & (~is_sky_anchor_bool)
        # Baseline: anchors in upper hemisphere but NOT inside sky_mask
        # (used to compare "low-lat sky_mask hits" against "low-lat non-sky-mask hits"
        # so the user can tell whether opacity alone separates them).
        baseline = (~sky_mask_bool[v, u]) & (~is_sky_anchor_bool) & (lat_deg >= 0.0)
        n_kf_used += 1

        if hit.any():
            # Use [lo, hi] for every band (inclusive on both ends). With the
            # user-supplied non-overlapping bands like (0,30),(30,60),(60,90)
            # this only matters at the measure-zero boundary samples and lets
            # the zenith (lat = 90 exact) land in the highest band.
            for (lo, hi), key in zip(bands, band_keys):
                m = hit & (lat_deg >= lo) & (lat_deg <= hi)
                if not m.any():
                    continue
                band_samples[key]["op"].append(opacity[m].detach().cpu().numpy())
                idx = m.nonzero(as_tuple=False).view(-1).tolist()
                band_samples[key]["anchors"].update(int(x) for x in idx)

        if baseline.any():
            baseline_samples["op"].append(opacity[baseline].detach().cpu().numpy())
            idx = baseline.nonzero(as_tuple=False).view(-1).tolist()
            baseline_samples["anchors"].update(int(x) for x in idx)

    if n_kf_used == 0:
        return None

    payload = {
        "frame_idx": int(frame_idx),
        "n_keyframes": int(n_kf_used),
        "n_anchors_total": int(N),
        "n_anchors_legacy_sky": int(is_sky_anchor_bool.sum().item()),
        "bands": {},
    }
    for key in band_keys:
        payload_band = band_samples[key]
        if not payload_band["op"]:
            payload["bands"][key] = {"n_samples": 0, "n_unique_anchors": 0}
            continue
        arr = np.concatenate(payload_band["op"])
        s = _summarize(arr, opacity_thresholds)
        s["n_unique_anchors"] = len(payload_band["anchors"])
        payload["bands"][key] = s

    # Baseline (upper-hemisphere, non-sky-mask): for comparison
    if baseline_samples["op"]:
        arr = np.concatenate(baseline_samples["op"])
        s = _summarize(arr, opacity_thresholds)
        s["n_unique_anchors"] = len(baseline_samples["anchors"])
        payload["baseline_upper_nonsky"] = s
    else:
        payload["baseline_upper_nonsky"] = {"n_samples": 0, "n_unique_anchors": 0}

    if extra_meta:
        for k, v in extra_meta.items():
            payload.setdefault(k, v)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload
