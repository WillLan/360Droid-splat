"""Pano-ReSplat feed-forward frontend orchestration."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pano_resplat_feedback import PanoRenderFeedbackEncoder
from .pano_resplat_init import PanoCompactGaussianInitializer
from .pano_resplat_refiner import PanoGaussianUpdateBlock
from .pano_resplat_renderer import PanoGaussianRendererAdapter
from .resplat_types import PanoGaussianState, PanoRenderOutput


class PanoReSplatFrontend(nn.Module):
    """Initializer + recurrent context-feedback Gaussian refinement."""

    def __init__(
        self,
        *,
        initializer: PanoCompactGaussianInitializer | None = None,
        feedback_encoder: PanoRenderFeedbackEncoder | None = None,
        update_block: PanoGaussianUpdateBlock | None = None,
        renderer: PanoGaussianRendererAdapter | None = None,
        renderer_backend: str = "soft_splat",
    ) -> None:
        super().__init__()
        self.initializer = initializer or PanoCompactGaussianInitializer()
        self.feedback_encoder = feedback_encoder or PanoRenderFeedbackEncoder()
        latent_dim = int(getattr(self.initializer, "state_dim", 64))
        sh_dim = int(getattr(self.initializer, "sh_dim", 1))
        feedback_dim = int(getattr(self.feedback_encoder, "feedback_dim", 32))
        self.update_block = update_block or PanoGaussianUpdateBlock(
            feedback_dim=feedback_dim,
            latent_dim=latent_dim,
            sh_dim=sh_dim,
            hidden_dim=latent_dim,
        )
        self.renderer = renderer or PanoGaussianRendererAdapter()
        self.renderer_backend = str(renderer_backend)

    def forward(
        self,
        context: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        *,
        num_refine: int = 0,
        return_all: bool = True,
    ) -> dict[str, Any]:
        images = context["images"]
        features = context["features"]
        depths = context["depths"]
        poses = context["poses_c2w"]
        valid_mask = context.get("valid_mask", context.get("valid_depth"))
        world_points = context.get("world_points")
        state = self.initializer(
            images,
            features,
            depths,
            poses,
            valid_mask,
            world_points=world_points,
        )
        states: list[PanoGaussianState] = [state]
        context_renders: list[PanoRenderOutput] = []
        feedback_debug: list[dict[str, torch.Tensor]] = []
        update_metrics: list[dict[str, torch.Tensor]] = []
        for _iter_idx in range(max(0, int(num_refine))):
            render_output = self._render_context_views(state, poses, tuple(int(x) for x in images.shape[-2:]))
            context_renders.append(render_output)
            feedback, debug = self.feedback_encoder(
                state,
                images,
                poses,
                render_output,
                context_depth=depths,
                context_valid_mask=valid_mask,
            )
            state, metrics = self.update_block(state, feedback)
            states.append(state)
            feedback_debug.append(debug)
            update_metrics.append(metrics)

        target_render = None
        if target is not None and "poses_c2w" in target:
            target_render = self._render_target_views(state, target["poses_c2w"], target, tuple(int(x) for x in images.shape[-2:]))

        result: dict[str, Any] = {
            "init_state": states[0],
            "final_state": state,
            "target_render": target_render,
            "context_renders": context_renders,
            "feedback_debug": feedback_debug,
            "update_metrics": update_metrics,
        }
        if return_all:
            result["states"] = states
        return result

    def _render_context_views(
        self,
        state: PanoGaussianState,
        poses_c2w: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> PanoRenderOutput:
        colors = []
        depths = []
        alphas = []
        packages = []
        backends = []
        v = int(poses_c2w.shape[1])
        for view_idx in range(v):
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
        target: dict[str, torch.Tensor],
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
