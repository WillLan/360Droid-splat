"""
Online acceleration utilities for panoramic GS-SLAM (Phase 4).

Two main components:

1. ``schedule_budget(overlap, motion, config)``
   Dynamically scales the per-keyframe map-optimisation budget based on the
   current frame's overlap with the existing map and the magnitude of the
   inter-frame motion.  High overlap + low motion 鈫?fewer mapping iters
   (fast path).  Low overlap + large motion 鈫?full mapping budget.

2. ``EmaStabilityTracker``
   Tracks per-Gaussian stability across keyframes using an exponential moving
   average (EMA) of the per-Gaussian ``n_touched`` contribution.  Gaussians
   whose EMA falls below a threshold are flagged as "unstable" and prioritised
   during optimisation; stable Gaussians have their gradients zeroed before
   ``optimizer.step()`` to avoid unnecessary computation.

Both components are gated by the ``enable_accel`` flag in the config; when the
flag is False every function is a no-op so the system falls back to the static
budget defined in the base config.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Dynamic budget scheduling
# ---------------------------------------------------------------------------

class BudgetSchedule:
    """Container for the current per-keyframe resource budget.

    ``track_iters`` / ``render_scale`` are kept for config compatibility, but
    the current panoramic GS-SLAM pipeline only consumes ``map_iters``.
    """
    __slots__ = ("track_iters", "map_iters", "render_scale")

    def __init__(self, track_iters: int, map_iters: int, render_scale: float):
        self.track_iters  = track_iters
        self.map_iters    = map_iters
        self.render_scale = render_scale

    def __repr__(self) -> str:
        return (
            f"BudgetSchedule(track={self.track_iters}, "
            f"map={self.map_iters}, scale={self.render_scale:.2f})"
        )


def schedule_budget(
    overlap: float,
    motion: float,
    config: dict,
) -> BudgetSchedule:
    """Compute the resource budget for the current keyframe.

    The active backend mapping budget is a linear interpolation between a
    *minimal* and a *full* budget based on a combined novelty score:

        novelty = (1 - overlap) * motion_factor

    where ``motion_factor = clamp(motion / motion_ref, 0, 1)``.

    High novelty (low overlap or large motion) 鈫?full budget.
    Low novelty (high overlap AND small motion)  鈫?minimal budget.

    Config keys read from ``Training`` (all optional, with defaults):
        accel_map_iters_min     (int,   default 10)
        accel_map_iters_max     (int,   default mapping_itr_num or 150)
        accel_motion_ref        (float, default 0.3)  # normalisation reference

    Args:
        overlap:  Overlap ratio in [0, 1] from ``compute_overlap`` / frontend.
        motion:   Inter-frame motion magnitude (e.g. translation norm in metres).
        config:   SLAM config dict.

    Returns:
        ``BudgetSchedule`` with the computed budget.  Only ``map_iters`` is
        currently used at runtime; the remaining fields are legacy-compatible
        placeholders.
    """
    training_cfg = config.get("Training", {})

    if not bool(training_cfg.get("enable_accel", False)):
        # Return static budget from config (no-op path)
        return BudgetSchedule(
            track_iters  = int(training_cfg.get("tracking_itr_num", 100)),
            map_iters    = int(training_cfg.get("mapping_itr_num", 150)),
            render_scale = 1.0,
        )

    track_min = int(training_cfg.get("accel_track_iters_min", 30))
    track_max = int(training_cfg.get("accel_track_iters_max",
                                      training_cfg.get("tracking_itr_num", 100)))
    map_min   = int(training_cfg.get("accel_map_iters_min", 10))
    map_max   = int(training_cfg.get("accel_map_iters_max",
                                      training_cfg.get("mapping_itr_num", 150)))
    scale_min = float(training_cfg.get("accel_render_scale_min", 0.5))
    scale_max = float(training_cfg.get("accel_render_scale_max", 1.0))
    motion_ref = float(training_cfg.get("accel_motion_ref", 0.3))

    motion_factor = min(1.0, motion / max(motion_ref, 1e-6))
    # novelty 鈭?[0, 1]: 0 = fully covered and static, 1 = novel and fast-moving
    novelty = (1.0 - float(overlap)) * motion_factor
    novelty = max(0.0, min(1.0, novelty))

    track_iters  = int(track_min + novelty * (track_max - track_min) + 0.5)
    map_iters    = int(map_min   + novelty * (map_max  - map_min)    + 0.5)
    render_scale = scale_min + novelty * (scale_max - scale_min)

    return BudgetSchedule(track_iters=track_iters,
                          map_iters=map_iters,
                          render_scale=render_scale)


# ---------------------------------------------------------------------------
# EMA Stability Tracker
# ---------------------------------------------------------------------------

class EmaStabilityTracker:
    """Track per-Gaussian stability using EMA of ``n_touched``.

    Each time ``update`` is called with the per-Gaussian ``n_touched`` tensor
    from the current mapping iteration, the EMA score is updated:

        ema[i] = alpha * ema[i] + (1 - alpha) * n_touched_normalised[i]

    A Gaussian is considered *stable* when its EMA score exceeds
    ``stable_threshold``.  Stable Gaussians have their gradients zeroed via
    ``mask_stable_gradients`` before the optimiser step.

    The tracker automatically handles Gaussian count changes (densification /
    pruning) by reinitialising itself when the count changes.

    Config keys read from ``Training``:
        accel_ema_alpha         (float, default 0.9)
        accel_stable_threshold  (float, default 0.6)
        accel_stable_top_frac   (float, default 0.8)  fraction of Gaussians to
                                 freeze (those with highest EMA score)
    """

    def __init__(self, config: dict):
        self.config = config
        training_cfg = config.get("Training", {})
        self.alpha      = float(training_cfg.get("accel_ema_alpha", 0.9))
        self.threshold  = float(training_cfg.get("accel_stable_threshold", 0.6))
        self.top_frac   = float(training_cfg.get("accel_stable_top_frac", 0.8))
        self._ema: Optional[torch.Tensor] = None
        self._n: int = 0

    def reset(self) -> None:
        """Reset tracker state (called after map reset)."""
        self._ema = None
        self._n = 0

    def update(self, n_touched: torch.Tensor) -> None:
        """Update EMA scores with the latest per-Gaussian touch counts.

        Args:
            n_touched: (N,) int or float tensor on any device.
        """
        if not bool(self.config.get("Training", {}).get("enable_accel", False)):
            return

        n = n_touched.shape[0]
        touched = n_touched.float().cpu()
        # Normalise to [0, 1] per batch
        t_max = touched.max().item()
        if t_max > 0:
            touched = touched / t_max

        if self._ema is None or self._n != n:
            self._ema = touched.clone()
            self._n = n
        else:
            self._ema = self.alpha * self._ema + (1.0 - self.alpha) * touched

    def stable_mask(self, device="cuda") -> Optional[torch.Tensor]:
        """Return a boolean mask of *stable* Gaussians (True = stable, freeze).

        Returns None when the tracker has not been initialised yet or when
        ``enable_accel`` is False.

        Args:
            device: target device for the returned mask.
        """
        if not bool(self.config.get("Training", {}).get("enable_accel", False)):
            return None
        if self._ema is None:
            return None

        threshold = self.threshold
        # Additionally cap the frozen fraction at top_frac (to always keep some
        # Gaussians active even if all have high EMA).
        ema = self._ema
        k_freeze = int(self.top_frac * ema.shape[0])
        if k_freeze < ema.shape[0]:
            kth_val = torch.topk(ema, k_freeze, largest=True).values[-1].item()
            threshold = max(threshold, kth_val)

        return (ema >= threshold).to(device=device)

    def mask_stable_gradients(self, gaussians) -> None:
        """Zero out gradients for stable Gaussians on all optimisable params.

        Must be called **after** ``loss.backward()`` and **before**
        ``optimizer.step()``.

        Args:
            gaussians: ``GaussianModel`` instance.
        """
        if not bool(self.config.get("Training", {}).get("enable_accel", False)):
            return
        mask = self.stable_mask(device="cuda")
        if mask is None or not mask.any():
            return

        # Parameters whose first dimension is the Gaussian index
        param_tensors = [
            gaussians._xyz,
            gaussians._features_dc,
            gaussians._features_rest,
            gaussians._opacity,
            gaussians._scaling,
            gaussians._rotation,
        ]
        for p in param_tensors:
            if p.grad is not None and p.shape[0] == mask.shape[0]:
                p.grad[mask] = 0.0
