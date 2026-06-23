"""Renderer adapter for Pano-ReSplat Gaussian states."""

from __future__ import annotations

from typing import Any

import torch

from backend.pano_gs import PFGS360Renderer, PanoRenderCamera

from .pano_resplat_geometry import camera_to_erp_uv
from .resplat_types import PanoGaussianState, PanoRenderOutput, state_to_explicit_gaussian_set


def _soft_splat_render(
    gaussians,
    camera: PanoRenderCamera,
    *,
    sigma_px: float = 1.25,
    max_points: int = 4096,
) -> dict[str, torch.Tensor]:
    """Small differentiable ERP splat fallback for smoke tests."""

    xyz = gaussians.get_xyz
    height, width = int(camera.image_height), int(camera.image_width)
    device = xyz.device
    dtype = xyz.dtype
    if int(xyz.shape[0]) > int(max_points):
        idx = torch.linspace(0, int(xyz.shape[0]) - 1, steps=int(max_points), device=device).round().long()
        xyz = xyz.index_select(0, idx)
        color = gaussians.get_features.index_select(0, idx)
        opacity = gaussians.get_opacity.index_select(0, idx)
        scale = gaussians.get_scaling.index_select(0, idx)
    else:
        color = gaussians.get_features
        opacity = gaussians.get_opacity
        scale = gaussians.get_scaling
    if int(xyz.shape[0]) == 0:
        render = torch.zeros(3, height, width, device=device, dtype=dtype)
        alpha = torch.zeros(1, height, width, device=device, dtype=dtype)
        return {"render": render, "depth": alpha.clone(), "alpha": alpha, "opacity": alpha}

    c2w = camera.c2w.to(device=device, dtype=dtype)
    ones = torch.ones(xyz.shape[0], 1, device=device, dtype=dtype)
    cam = (torch.linalg.inv(c2w) @ torch.cat([xyz, ones], dim=-1).T).T[:, :3]
    uv, depth, valid, _bearing = camera_to_erp_uv(cam, (height, width), require_forward=False)
    if not bool(valid.any()):
        render = torch.zeros(3, height, width, device=device, dtype=dtype)
        alpha = torch.zeros(1, height, width, device=device, dtype=dtype)
        return {"render": render, "depth": alpha.clone(), "alpha": alpha, "opacity": alpha}

    uv = uv[valid]
    depth = depth[valid]
    color = color[valid].clamp(0.0, 1.0)
    opacity = opacity[valid].clamp(0.0, 1.0)
    scale = scale[valid].clamp_min(1.0e-8)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = torch.remainder(xs.reshape(1, height, width) - uv[:, 0].view(-1, 1, 1) + float(width) * 0.5, float(width))
    dx = dx - float(width) * 0.5
    dy = ys.reshape(1, height, width) - uv[:, 1].view(-1, 1, 1)
    sigma = (
        float(sigma_px)
        + (scale.mean(dim=-1) / depth).clamp(0.0, 0.10) * float(max(height, width))
    ).view(-1, 1, 1)
    weight = torch.exp(-0.5 * (dx.square() + dy.square()) / sigma.square().clamp_min(1.0e-6))
    weight = weight * opacity.view(-1, 1, 1)
    denom = weight.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
    rgb = (weight.unsqueeze(1) * color.view(-1, 3, 1, 1)).sum(dim=0) / denom
    depth_img = (weight * depth.view(-1, 1, 1)).sum(dim=0, keepdim=True) / denom
    alpha = weight.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    return {"render": rgb.clamp(0.0, 1.0), "depth": depth_img, "alpha": alpha, "opacity": alpha}


class PanoGaussianRendererAdapter:
    """Render batched ``PanoGaussianState`` through gsplat360 or soft splatting."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        extra_gsplat360_roots: list[str] | None = None,
        allow_soft_splat_fallback: bool = True,
        soft_sigma_px: float = 1.25,
        soft_max_points: int = 4096,
    ) -> None:
        self.config = config or {}
        self.extra_gsplat360_roots = extra_gsplat360_roots or []
        self.allow_soft_splat_fallback = bool(allow_soft_splat_fallback)
        self.soft_sigma_px = float(soft_sigma_px)
        self.soft_max_points = int(soft_max_points)
        self._renderer: PFGS360Renderer | None = None

    def _gsplat_renderer(self) -> PFGS360Renderer:
        if self._renderer is None:
            self._renderer = PFGS360Renderer(
                config=self.config,
                extra_gsplat360_roots=self.extra_gsplat360_roots,
                allow_fallback=False,
            )
        return self._renderer

    def render_state(
        self,
        state: PanoGaussianState,
        poses_c2w: torch.Tensor,
        image_hw: tuple[int, int],
        renderer_backend: str = "auto",
    ) -> PanoRenderOutput:
        """Render a batched state.

        ``poses_c2w`` accepts ``B x 4 x 4`` or a single ``4 x 4`` pose.
        ``renderer_backend`` is ``"auto"``, ``"gsplat360"``, or ``"soft_splat"``.
        """

        backend = str(renderer_backend or "auto").lower()
        if backend not in {"auto", "gsplat360", "soft_splat"}:
            raise ValueError(f"Unsupported renderer_backend={renderer_backend!r}")
        height, width = int(image_hw[0]), int(image_hw[1])
        pose_batch = poses_c2w
        if pose_batch.ndim == 2:
            pose_batch = pose_batch.unsqueeze(0).expand(state.batch_size, -1, -1)
        if pose_batch.ndim != 3 or tuple(pose_batch.shape[-2:]) != (4, 4):
            raise ValueError(f"poses_c2w must have shape Bx4x4 or 4x4, got {tuple(poses_c2w.shape)}")
        if int(pose_batch.shape[0]) != state.batch_size:
            raise ValueError(f"poses batch size {int(pose_batch.shape[0])} does not match state batch size {state.batch_size}")

        colors = []
        depths = []
        alphas = []
        packages = []
        backends_used = []
        for batch_idx in range(state.batch_size):
            gaussians = state_to_explicit_gaussian_set(state, batch_idx, config=self.config)
            camera = PanoRenderCamera(
                image_height=height,
                image_width=width,
                c2w=pose_batch[batch_idx].to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype),
            )
            if backend == "soft_splat":
                pkg = _soft_splat_render(
                    gaussians,
                    camera,
                    sigma_px=self.soft_sigma_px,
                    max_points=self.soft_max_points,
                )
                used = "soft_splat"
            else:
                try:
                    pkg = self._gsplat_renderer().render(camera, gaussians)
                    used = "gsplat360"
                except (ImportError, ModuleNotFoundError, NotImplementedError, RuntimeError):
                    if backend == "gsplat360" and not self.allow_soft_splat_fallback:
                        raise
                    if not self.allow_soft_splat_fallback:
                        raise
                    pkg = _soft_splat_render(
                        gaussians,
                        camera,
                        sigma_px=self.soft_sigma_px,
                        max_points=self.soft_max_points,
                    )
                    used = "soft_splat"
            render = pkg["render"]
            depth = pkg.get("depth")
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not torch.is_tensor(render):
                raise RuntimeError("Renderer package missing tensor key 'render'.")
            if not torch.is_tensor(depth):
                depth = torch.zeros(1, height, width, device=render.device, dtype=render.dtype)
            if not torch.is_tensor(alpha):
                alpha = torch.zeros(1, height, width, device=render.device, dtype=render.dtype)
            colors.append(render)
            depths.append(depth)
            alphas.append(alpha)
            packages.append(pkg)
            backends_used.append(used)

        return PanoRenderOutput(
            color=torch.stack(colors, dim=0),
            depth=torch.stack(depths, dim=0),
            alpha=torch.stack(alphas, dim=0),
            extras={"packages": packages, "backend": backends_used},
        )
