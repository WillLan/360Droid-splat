"""Compatibility adapter for the graph-based PanoDROID frontend."""

from __future__ import annotations

from .graph_tracker import PanoDroidGraphTracker
from .model import PanoDroidModel


class PanoDROIDFrontendAdapter(PanoDroidGraphTracker):
    """Backward-compatible name for the default graph tracker."""


def build_frontend_from_config(config: dict) -> PanoDROIDFrontendAdapter:
    model = PanoDroidModel(**config.get("Model", {}))
    frontend_cfg = config.get("Frontend", {})
    graph_cfg = config.get("Graph", {})
    ba_cfg = config.get("BA", {})
    adapter = PanoDROIDFrontendAdapter(
        model,
        keyframe_threshold=float(frontend_cfg.get("keyframe_threshold", 0.55)),
        force_keyframe_interval=int(frontend_cfg.get("force_keyframe_interval", 10)),
        window_size=int(frontend_cfg.get("window_size", graph_cfg.get("window_size", 5))),
        temporal_radius=int(graph_cfg.get("temporal_radius", 2)),
        max_factors=int(graph_cfg.get("max_factors", graph_cfg.get("max_edges_per_step", 24))),
        num_updates=frontend_cfg.get("num_updates"),
        ba_iters_per_update=int(ba_cfg.get("iters_per_update", graph_cfg.get("ba_iters_per_update", 2))),
        ba_sample_stride=int(ba_cfg.get("sample_stride", graph_cfg.get("ba_sample_stride", 1))),
        fixed_frames=int(graph_cfg.get("fixed_frames_inference", 1)),
    )
    ckpt = frontend_cfg.get("checkpoint")
    if ckpt:
        adapter.load_checkpoint(ckpt)
    return adapter
