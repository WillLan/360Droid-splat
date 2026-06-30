"""PanoAnchorSplat local-window feed-forward frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from frontend.pano_droid.interfaces import FrontendOutput, PanoDROIDFrontend, PanoFrame

from .pano_anchor_splat_decoder import PanoAnchorGaussianDecoder
from .pano_anchor_splat_encoder import PanoAnchorFeatureEncoder
from .pano_anchor_splat_error import PanoAnchorRenderErrorEncoder
from .pano_anchor_splat_prior import PanoVGGTPriorProvider
from .pano_anchor_splat_refiner import PanoAnchorGaussianRefinerLite
from .pano_anchor_splat_types import PanoAnchorSet, PanoAnchorSplatConfig, PanoAnchorSplatOutput, PanoAnchorSplatPrior
from .pano_anchor_splat_voxel import PanoVoxelAnchorBuilder
from .pano_resplat_renderer import PanoGaussianRendererAdapter
from .resplat_types import PanoGaussianState, PanoRenderOutput


class PanoAnchorSplatFrontend(nn.Module, PanoDROIDFrontend):
    """AnchorSplat-style PanoVGGT prior to Gaussian local-window renderer.

    This module is intentionally a local-window feed-forward frontend.  The
    online ``track`` API remains a clear error until the mapping integration is
    explicitly wired without changing ``FrontendOutput``.
    """

    def __init__(
        self,
        config: PanoAnchorSplatConfig | dict | None = None,
        *,
        prior_provider: PanoVGGTPriorProvider | None = None,
        anchor_builder: PanoVoxelAnchorBuilder | None = None,
        feature_encoder: PanoAnchorFeatureEncoder | None = None,
        decoder: PanoAnchorGaussianDecoder | None = None,
        error_encoder: PanoAnchorRenderErrorEncoder | None = None,
        refiner: PanoAnchorGaussianRefinerLite | None = None,
        renderer: PanoGaussianRendererAdapter | None = None,
        renderer_backend: str = "soft_splat",
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, PanoAnchorSplatConfig) else PanoAnchorSplatConfig.from_dict(config)
        self.prior_provider = prior_provider or PanoVGGTPriorProvider(detach=True)
        self.anchor_builder = anchor_builder or PanoVoxelAnchorBuilder(self.config)
        self.feature_encoder = feature_encoder or PanoAnchorFeatureEncoder(self.config)
        self.decoder = decoder or PanoAnchorGaussianDecoder(self.config)
        self.error_encoder = error_encoder or PanoAnchorRenderErrorEncoder(self.config)
        self.refiner = refiner or PanoAnchorGaussianRefinerLite(self.config)
        self.renderer = renderer or PanoGaussianRendererAdapter()
        self.renderer_backend = str(renderer_backend)
        self.sequence_meta: dict[str, Any] = {}

    def forward(
        self,
        context: PanoAnchorSplatPrior | Mapping[str, Any],
        target: Mapping[str, torch.Tensor] | None = None,
        *,
        num_refine: int = 0,
        return_dataclass: bool = False,
    ) -> dict[str, Any] | PanoAnchorSplatOutput:
        prior = self.prior_provider(context)
        anchors = self.anchor_builder(prior)
        anchor_tokens = self.feature_encoder(anchors)
        init_state = self.decoder(anchors, anchor_tokens)
        state = init_state
        context_renders: list[PanoRenderOutput] = []
        error_debug: list[dict[str, torch.Tensor]] = []
        refiner_metrics: list[dict[str, torch.Tensor]] = []
        for _ in range(max(0, int(num_refine))):
            render_output = self._render_context_views(state, prior.poses_c2w, tuple(int(x) for x in prior.images.shape[-2:]))
            context_renders.append(render_output)
            error_tokens, debug = self.error_encoder(
                anchors,
                prior.images,
                prior.poses_c2w,
                render_output,
                context_depth=prior.depths,
                context_valid_mask=prior.valid_mask,
            )
            state, metrics = self.refiner(state, anchors, anchor_tokens, error_tokens)
            error_debug.append(debug)
            refiner_metrics.append(metrics)

        target_render = None
        if target is not None and "poses_c2w" in target:
            target_render = self._render_target_views(state, target["poses_c2w"], target, tuple(int(x) for x in prior.images.shape[-2:]))

        debug: dict[str, Any] = {
            "anchor_stats": dict(self.anchor_builder.last_stats),
            "error_debug": error_debug,
            "refiner_metrics": refiner_metrics,
            "config": self.config.to_dict(),
        }
        if return_dataclass:
            return PanoAnchorSplatOutput(
                anchors=anchors,
                init_state=init_state,
                final_state=state,
                target_render=target_render,
                context_renders=context_renders,
                debug=debug,
            )
        return {
            "anchors": anchors,
            "anchor_tokens": anchor_tokens,
            "init_state": init_state,
            "final_state": state,
            "target_render": target_render,
            "context_renders": context_renders,
            "debug": debug,
        }

    def _render_context_views(
        self,
        state: PanoGaussianState,
        poses_c2w: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> PanoRenderOutput:
        if poses_c2w.ndim != 4:
            raise ValueError(f"context poses_c2w must have shape BxVx4x4, got {tuple(poses_c2w.shape)}")
        colors = []
        depths = []
        alphas = []
        packages = []
        backends = []
        for view_idx in range(int(poses_c2w.shape[1])):
            out = self.renderer.render_state(
                state,
                poses_c2w[:, view_idx],
                image_hw,
                renderer_backend=self.renderer_backend,
            )
            colors.append(out.color)
            depths.append(out.depth)
            alphas.append(out.alpha)
            packages.append(out.extras.get("packages"))
            backends.append(out.extras.get("backend"))
        return PanoRenderOutput(
            color=torch.stack(colors, dim=1),
            depth=torch.stack(depths, dim=1),
            alpha=torch.stack(alphas, dim=1),
            extras={"packages": packages, "backend": backends},
        )

    def _render_target_views(
        self,
        state: PanoGaussianState,
        poses_c2w: torch.Tensor,
        target: Mapping[str, torch.Tensor],
        default_hw: tuple[int, int],
    ) -> PanoRenderOutput:
        image_hw = tuple(int(x) for x in target["images"].shape[-2:]) if "images" in target else default_hw
        if poses_c2w.ndim == 3:
            return self.renderer.render_state(state, poses_c2w, image_hw, renderer_backend=self.renderer_backend)
        if poses_c2w.ndim != 4:
            raise ValueError(f"target poses_c2w must have shape Bx4x4 or BxTx4x4, got {tuple(poses_c2w.shape)}")
        colors = []
        depths = []
        alphas = []
        packages = []
        backends = []
        for view_idx in range(int(poses_c2w.shape[1])):
            out = self.renderer.render_state(
                state,
                poses_c2w[:, view_idx],
                image_hw,
                renderer_backend=self.renderer_backend,
            )
            colors.append(out.color)
            depths.append(out.depth)
            alphas.append(out.alpha)
            packages.append(out.extras.get("packages"))
            backends.append(out.extras.get("backend"))
        return PanoRenderOutput(
            color=torch.stack(colors, dim=1),
            depth=torch.stack(depths, dim=1),
            alpha=torch.stack(alphas, dim=1),
            extras={"packages": packages, "backend": backends},
        )

    def initialize(self, sequence_meta: dict) -> None:
        self.sequence_meta = dict(sequence_meta)

    def track(self, frame: PanoFrame) -> FrontendOutput:
        raise NotImplementedError(
            "PanoAnchorSplatFrontend is a local-window feed-forward Gaussian frontend. "
            "Use forward(context, target) with offline PanoVGGT priors; online track() integration is intentionally not wired yet."
        )

    def reset(self) -> None:
        self.sequence_meta = {}

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if state_dict is None and isinstance(payload, dict):
            state_dict = payload.get("model")
        if state_dict is None:
            state_dict = payload
        self.load_state_dict(state_dict, strict=False)


def build_pano_anchor_splat_frontend_from_config(config: dict[str, Any]) -> PanoAnchorSplatFrontend:
    """Build the config-gated AnchorSplat frontend from a project config."""

    anchor_cfg = dict(config.get("PanoAnchorSplat", {}))
    cfg = PanoAnchorSplatConfig.from_dict(anchor_cfg)
    prior_cfg = dict(config.get("PanoAnchorSplatPrior", config.get("Prior", {})))
    renderer_cfg = dict(config.get("Renderer", {}))
    provider = PanoVGGTPriorProvider(
        cache_root=prior_cfg.get("cache_root"),
        cache_key_field=str(prior_cfg.get("cache_key_field", "sequence_id")),
        strict_cache=bool(prior_cfg.get("strict_cache", True)),
        detach=True,
    )
    renderer = PanoGaussianRendererAdapter(
        config={"Training": dict(config.get("TrainingRender", {})), "Renderer": renderer_cfg},
        extra_gsplat360_roots=list(renderer_cfg.get("extra_gsplat360_roots", [])),
        allow_soft_splat_fallback=bool(renderer_cfg.get("allow_soft_splat_fallback", True)),
        soft_sigma_px=float(renderer_cfg.get("soft_sigma_px", 1.25)),
        soft_max_points=int(renderer_cfg.get("soft_max_points", 4096)),
    )
    backend = str(renderer_cfg.get("backend", "soft_splat"))
    frontend = PanoAnchorSplatFrontend(cfg, prior_provider=provider, renderer=renderer, renderer_backend=backend)
    ckpt = config.get("Frontend", {}).get("checkpoint") or anchor_cfg.get("checkpoint")
    if ckpt:
        frontend.load_checkpoint(str(Path(ckpt)))
    return frontend
