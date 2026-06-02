"""
MDL-driven novelty scoring and pixel selection for panoramic 3DGS SLAM.

This module implements three primitives used by the backend to decide **which
pixels** should be back-projected and initialised as new Gaussians when a
new keyframe arrives:

  compute_overlap(alpha, area_w, tau_A)
      -> scalar overlap ratio in [0, 1]

  compute_novelty(render_rgb, gt_rgb, render_depth, gt_depth, alpha, area_w)
      -> (H, W) novelty map in [0, 1]

  select_pixels_greedy_mdl(novelty, area_w, config)
      -> (H, W) bool mask of selected pixels

All tensors are expected on CUDA (float32).  The area weight ``area_w``
(shape 1脳H脳W, mean鈮?) is the cos-latitude spherical area weight produced by
``utils.slam_utils.erp_area_weight``; pass ``torch.ones(1,H,W)`` for pinhole.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def compute_overlap(
    alpha: torch.Tensor,
    area_w: torch.Tensor,
    tau_A: float = 0.5,
) -> torch.Tensor:
    """Fraction of the image that is already well-explained by the current map.

    Overlap = 危_i  area_w_i * [alpha_i >= tau_A]
              鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
                     危_i  area_w_i

    Args:
        alpha:   (1, H, W) rendered opacity / transmittance in [0, 1].
        area_w:  (1, H, W) spherical area weight (mean鈮?).
        tau_A:   opacity threshold above which a pixel is considered "covered".

    Returns:
        Scalar tensor in [0, 1].
    """
    covered = (alpha >= tau_A).float()
    return (covered * area_w).sum() / area_w.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

def compute_novelty(
    render_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    render_depth: torch.Tensor,
    gt_depth: Optional[torch.Tensor],
    alpha: torch.Tensor,
    area_w: torch.Tensor,
    lambda_d: float = 1.0,
    residual_rescue_weight: float = 0.35,
    residual_rescue_alpha_thresh: float = 0.35,
    residual_rescue_rel_thresh: float = 1.75,
    rgb_eps: float = 1e-3,
    depth_eps: float = 1e-2,
) -> torch.Tensor:
    """Per-pixel novelty score: how poorly is this pixel explained by the map?

    Baseline MDL favours low-coverage pixels:

        score_i = (1 - alpha_i) * residual_i

    but in online SLAM we also need a "repair" path for pixels that are
    already covered by large Gaussians yet still badly rendered.  We therefore
    augment the score with a residual-rescue term:

        score_i =
            (1 - alpha_i) * residual_i
            + rescue_weight * rescue_gate_i * residual_i

    where ``rescue_gate`` activates only for sufficiently high-opacity pixels
    whose residual exceeds a per-frame relative threshold.

    where
        r_c = Charbonnier(render_rgb - gt_rgb)  (averaged across colour channels)
        r_d = Charbonnier(render_depth - gt_depth)  (0 when gt_depth unavailable)

    The result is multiplied by the spherical area weight so that equatorial
    pixels (larger solid angle) naturally receive higher priority.

    Args:
        render_rgb:   (3, H, W) rendered colour.
        gt_rgb:       (3, H, W) ground-truth colour.
        render_depth: (1, H, W) rendered radial depth.
        gt_depth:     (1, H, W) metric depth from DAP/GT, or None.
        alpha:        (1, H, W) rendered opacity.
        area_w:       (1, H, W) cos-latitude area weight.
        lambda_d:     relative weight for depth residual.

    Returns:
        (1, H, W) novelty map (non-negative, unnormalised).
    """
    def _charbonnier(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(x * x + eps * eps)

    r_c = _charbonnier(render_rgb - gt_rgb, rgb_eps).mean(dim=0, keepdim=True)  # (1,H,W)

    if gt_depth is not None:
        depth_valid = (gt_depth > 0.01) & (alpha > 0.05)
        r_d = _charbonnier(render_depth - gt_depth, depth_eps)
        r_d = r_d * depth_valid.float()
    else:
        r_d = torch.zeros_like(render_depth)

    residual = r_c + lambda_d * r_d
    alpha_clamped = alpha.clamp(0.0, 1.0)
    novelty = (1.0 - alpha_clamped) * residual

    # Rescue high-residual regions even when they are already "covered" by
    # blurred / oversized Gaussians.  This is the case the original MDL term
    # systematically misses in online SLAM.
    residual_mean = residual.detach().mean()
    residual_floor = residual_mean * residual_rescue_rel_thresh
    rescue_mask = (
        (alpha_clamped >= residual_rescue_alpha_thresh)
        & (residual >= residual_floor)
    ).float()
    if rescue_mask.any():
        rescue_alpha = (
            (alpha_clamped - residual_rescue_alpha_thresh)
            / max(1.0 - residual_rescue_alpha_thresh, 1e-6)
        ).clamp(0.0, 1.0)
        novelty = novelty + residual_rescue_weight * rescue_alpha * residual * rescue_mask

    novelty = novelty * area_w
    return novelty.clamp_min(0.0)


# ---------------------------------------------------------------------------
# MDL greedy pixel selection
# ---------------------------------------------------------------------------

def select_pixels_greedy_mdl(
    novelty: torch.Tensor,
    area_w: torch.Tensor,
    config: dict,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select pixels for new Gaussian initialisation using a greedy MDL criterion.

    Pixels are ranked by their novelty score.  We add pixels in descending
    order until one of the two stopping criteria is met:

        1. The number of selected pixels reaches ``B_max``.
        2. The marginal novelty gain of the next pixel falls below
           ``lambda_mdl * mean_novelty``.

    Args:
        novelty:     (1, H, W) novelty map from ``compute_novelty``.
        area_w:      (1, H, W) cos-latitude area weight.
        config:      SLAM config dict; reads keys from ``Training``:
                       mdl_B_max      (int,   default 4096)
                       mdl_lambda_mdl (float, default 0.05)
                       mdl_min_novelty (float, default 1e-4)
        valid_mask:  Optional (1, H, W) bool mask to restrict candidates
                     (e.g. non-sky, non-zero depth, within rgb boundary).

    Returns:
        (H, W) bool tensor 鈥?True for selected pixels.
    """
    training_cfg = config.get("Training", {})
    B_max = int(training_cfg.get("mdl_B_max", 4096))
    lambda_mdl = float(training_cfg.get("mdl_lambda_mdl", 0.05))
    min_novelty = float(training_cfg.get("mdl_min_novelty", 1e-4))

    _, H, W = novelty.shape
    scores = novelty.squeeze(0)         # (H, W)

    if valid_mask is not None:
        scores = scores * valid_mask.squeeze(0).float()

    flat = scores.reshape(-1)           # (H*W,)
    if flat.sum() < min_novelty:
        # Nothing novel: return empty mask (caller will fall back to full back-proj)
        return torch.zeros(H, W, dtype=torch.bool, device=novelty.device)

    sorted_idx = torch.argsort(flat, descending=True)
    mean_novelty = flat[flat > min_novelty].mean().item()
    threshold = lambda_mdl * mean_novelty

    selected = torch.zeros(H * W, dtype=torch.bool, device=novelty.device)
    count = 0
    for idx in sorted_idx:
        val = flat[idx].item()
        if val < threshold or count >= B_max:
            break
        selected[idx] = True
        count += 1

    return selected.reshape(H, W)


# ---------------------------------------------------------------------------
# Convenience: compute everything and return mask
# ---------------------------------------------------------------------------

def mdl_pixel_mask(
    render_pkg: dict,
    gt_rgb: torch.Tensor,
    gt_depth: Optional[torch.Tensor],
    config: dict,
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-in-one helper used by slam_backend.add_next_kf.

    Args:
        render_pkg:  Output dict of render_erp_direct / render with keys
                     ``render``, ``depth``, ``opacity``/``alpha``.
        gt_rgb:      (3, H, W) ground-truth colour.
        gt_depth:    (1, H, W) metric depth, or None.
        config:      SLAM config dict.
        valid_mask:  Optional pixel validity mask.

    Returns:
        pixel_mask (H, W bool), overlap (scalar), novelty (1, H, W).
    """
    from backend.legacy_360gs.utils.slam_utils import erp_area_weight, _is_panorama_config

    rendered = render_pkg["render"]                         # (3, H, W)
    depth    = render_pkg["depth"]                         # (1, H, W)
    alpha    = render_pkg.get("opacity", render_pkg.get("alpha"))  # (1, H, W)

    _, H, W = rendered.shape
    is_pano = _is_panorama_config(config)
    if is_pano:
        area_w = erp_area_weight(H, W, device=rendered.device, dtype=rendered.dtype)
    else:
        area_w = torch.ones(1, H, W, device=rendered.device, dtype=rendered.dtype)

    training_cfg = config.get("Training", {})
    tau_A    = float(training_cfg.get("mdl_tau_alpha", 0.5))
    lambda_d = float(training_cfg.get("mdl_lambda_depth", 1.0))
    residual_rescue_weight = float(
        training_cfg.get("mdl_residual_rescue_weight", 0.35)
    )
    residual_rescue_alpha_thresh = float(
        training_cfg.get("mdl_residual_rescue_alpha_thresh", 0.35)
    )
    residual_rescue_rel_thresh = float(
        training_cfg.get("mdl_residual_rescue_rel_thresh", 1.75)
    )

    overlap = compute_overlap(alpha, area_w, tau_A)
    novelty = compute_novelty(
        rendered, gt_rgb.to(rendered.device),
        depth, gt_depth,
        alpha, area_w,
        lambda_d=lambda_d,
        residual_rescue_weight=residual_rescue_weight,
        residual_rescue_alpha_thresh=residual_rescue_alpha_thresh,
        residual_rescue_rel_thresh=residual_rescue_rel_thresh,
    )
    pixel_mask = select_pixels_greedy_mdl(novelty, area_w, config, valid_mask)
    return pixel_mask, overlap, novelty


def compute_mdl_mask_from_render_pkg(
    render_pkg: dict,
    viewpoint,
    depth_map,
    config: dict,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Optional[torch.Tensor], float, torch.Tensor]:
    """Compute the raw MDL mask for a keyframe render.

    Returns:
        pixel_mask: (H, W) bool tensor on the current device, or None when the
            selected ratio is below ``mdl_min_select_ratio`` and the caller
            should fall back to its default dense insertion logic.
        overlap: scalar float overlap ratio.
        novelty: (1, H, W) novelty tensor.
    """
    gt_rgb = viewpoint.original_image.to(device=device, dtype=dtype)

    gt_depth = None
    if depth_map is not None:
        if isinstance(depth_map, torch.Tensor):
            gt_depth = depth_map.to(device=device, dtype=dtype)
        else:
            gt_depth = torch.from_numpy(np.asarray(depth_map).astype(np.float32)).to(
                device=device, dtype=dtype
            )
        if gt_depth.ndim == 2:
            gt_depth = gt_depth.unsqueeze(0)

    pixel_mask, overlap, novelty = mdl_pixel_mask(
        render_pkg, gt_rgb, gt_depth, config
    )

    n_selected = int(pixel_mask.sum().item())
    n_total = int(pixel_mask.numel())
    min_ratio = float(config.get("Training", {}).get("mdl_min_select_ratio", 0.005))
    if n_selected < n_total * min_ratio:
        return None, float(overlap.item()), novelty
    return pixel_mask, float(overlap.item()), novelty
