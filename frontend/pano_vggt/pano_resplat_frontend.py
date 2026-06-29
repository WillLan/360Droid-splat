"""Pano-ReSplat feed-forward frontend orchestration."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pano_resplat_feedback import PanoRenderFeedbackEncoder
from .pano_resplat_point_decoder_init import PanoVGGTPointDecoderGaussianInitializer
from .pano_resplat_refiner import PanoGaussianUpdateBlock
from .pano_resplat_renderer import PanoGaussianRendererAdapter
from .pano_resplat_voxel import VoxelGaussianCompactor
from .resplat_types import PanoGaussianState, PanoRenderOutput


class PanoReSplatFrontend(nn.Module):
    """Initializer + recurrent context-feedback Gaussian refinement."""

    def __init__(
        self,
        *,
        initializer: PanoVGGTPointDecoderGaussianInitializer | None = None,
        feedback_encoder: PanoRenderFeedbackEncoder | None = None,
        update_block: PanoGaussianUpdateBlock | None = None,
        compactor: VoxelGaussianCompactor | None = None,
        renderer: PanoGaussianRendererAdapter | None = None,
        renderer_backend: str = "soft_splat",
    ) -> None:
        super().__init__()
        self.initializer = initializer or PanoVGGTPointDecoderGaussianInitializer()
        self.feedback_encoder = feedback_encoder or PanoRenderFeedbackEncoder()
        self.compactor = compactor
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
        render_poses = poses
        valid_mask = context.get("valid_mask", context.get("valid_depth"))
        world_points = context.get("world_points")
        dense_state = self.initializer(
            images,
            features,
            depths,
            poses,
            valid_mask,
            world_points=world_points,
            tokens=context.get("tokens"),
            token_hw=context.get("token_hw"),
        )
        state = self._compact_init_state(dense_state)
        states: list[PanoGaussianState] = [state]
        poses_by_iter: list[torch.Tensor] = [render_poses]
        context_renders: list[PanoRenderOutput] = []
        feedback_debug: list[dict[str, torch.Tensor]] = []
        update_metrics: list[dict[str, torch.Tensor]] = []
        for _iter_idx in range(max(0, int(num_refine))):
            render_output = self._render_context_views(state, render_poses, tuple(int(x) for x in images.shape[-2:]))
            context_renders.append(render_output)
            feedback, state_for_update, debug = self.feedback_encoder.refine_state_and_feedback(
                state,
                images,
                render_poses,
                render_output,
                context_depth=depths,
                context_valid_mask=valid_mask,
            )
            next_render_poses = debug.pop("refined_context_poses_c2w", None)
            if torch.is_tensor(next_render_poses):
                render_poses = next_render_poses
            state, metrics = self.update_block(state_for_update, feedback)
            states.append(state)
            poses_by_iter.append(render_poses)
            feedback_debug.append(debug)
            update_metrics.append(metrics)

        target_render = None
        if target is not None and "poses_c2w" in target:
            target_poses = target["poses_c2w"]
            if target_poses.ndim == 4 and tuple(target_poses.shape[:2]) == tuple(render_poses.shape[:2]):
                target_poses = poses_by_iter[-1]
            target_render = self._render_target_views(state, target_poses, target, tuple(int(x) for x in images.shape[-2:]))

        result: dict[str, Any] = {
            "init_state": states[0],
            "dense_init_state": dense_state,
            "final_state": state,
            "target_render": target_render,
            "context_renders": context_renders,
            "feedback_debug": feedback_debug,
            "update_metrics": update_metrics,
            "compactor_debug": self._compactor_debug(),
            "refined_context_poses_c2w": render_poses,
            "context_poses_by_iter": poses_by_iter,
        }
        if return_all:
            result["states"] = states
        return result

    def _compact_init_state(self, state: PanoGaussianState) -> PanoGaussianState:
        if self.compactor is None:
            return state
        return self.compactor(state)

    def _compactor_debug(self) -> dict[str, torch.Tensor]:
        if self.compactor is None:
            return {}
        return dict(getattr(self.compactor, "last_stats", {}))

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
