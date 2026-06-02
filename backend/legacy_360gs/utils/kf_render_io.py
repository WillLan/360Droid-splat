"""Save keyframe GT-vs-render comparison canvases (lossless PNG or JPEG)."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def save_kf_canvas(
    canvas_bgr,
    path_without_ext: str,
    results_cfg: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write ``canvas_bgr`` to disk. Extension chosen by config.

    Args:
        canvas_bgr: OpenCV BGR uint8 image (H, W, 3).
        path_without_ext: Full path without extension, e.g.
            ``.../kf_renders_opt/kf_0067``.
        results_cfg: ``config['Results']`` subset. Keys:
            - ``kf_render_format``: ``\"png\"`` (default) or ``\"jpeg\"``.
            - ``kf_jpeg_quality``: 1--100, default 95 (only for JPEG).

    Returns:
        Path written (with extension).
    """
    import cv2

    cfg = results_cfg or {}
    fmt = str(cfg.get("kf_render_format", "png")).lower().strip()
    if fmt in ("jpg", "jpeg"):
        q = int(cfg.get("kf_jpeg_quality", 95))
        q = max(1, min(100, q))
        out = path_without_ext + ".jpg"
        cv2.imwrite(out, canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return out
    out = path_without_ext + ".png"
    cv2.imwrite(out, canvas_bgr)
    return out


def kf_render_extension(results_cfg: Optional[Mapping[str, Any]] = None) -> str:
    cfg = results_cfg or {}
    fmt = str(cfg.get("kf_render_format", "png")).lower().strip()
    if fmt in ("jpg", "jpeg"):
        return ".jpg"
    return ".png"
