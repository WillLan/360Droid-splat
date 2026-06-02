"""Adapter for PFGS360/gsplat360 equirectangular rendering.

The production path calls ``gsplat360.rasterization`` with
``camera_model="equirectangular"``.  A small CPU/PyTorch fallback is kept only
for tests and smoke runs where the CUDA extension is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import torch

from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel


@dataclass
class PanoRenderCamera:
    image_height: int
    image_width: int
    c2w: torch.Tensor

    @property
    def w2c(self) -> torch.Tensor:
        return torch.linalg.inv(self.c2w)


RenderPackage = dict[str, torch.Tensor | None]


def _optional_gsplat360(extra_roots: list[str] | None = None):
    try:
        from gsplat360 import rasterization
        from gsplat360.cuda import _backend as gsplat360_backend

        if gsplat360_backend._C is None:
            raise ImportError("gsplat360 CUDA extension is not loaded.")
        return rasterization
    except ModuleNotFoundError:
        pass

    for root in extra_roots or []:
        candidate = Path(root)
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from gsplat360 import rasterization
            from gsplat360.cuda import _backend as gsplat360_backend

            if gsplat360_backend._C is None:
                raise ImportError("gsplat360 CUDA extension is not loaded.")
            return rasterization
        except ModuleNotFoundError:
            continue
    return None


def _identity_quats(n: int, *, device, dtype) -> torch.Tensor:
    quat = torch.zeros(n, 4, device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def _blank_package(camera: PanoRenderCamera, gaussians, background: torch.Tensor) -> RenderPackage:
    H = int(camera.image_height)
    W = int(camera.image_width)
    total = int(gaussians.get_xyz.shape[0])
    bg = background.to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    render = bg.view(3, 1, 1).expand(3, H, W).clone()
    depth = torch.zeros((1, H, W), device=render.device, dtype=render.dtype)
    alpha = torch.zeros((1, H, W), device=render.device, dtype=render.dtype)
    return {
        "render": render,
        "gs_only": render,
        "sky_bg_only": torch.zeros_like(render),
        "sky_bg_alpha": torch.zeros_like(alpha),
        "depth": depth,
        "opacity": alpha,
        "alpha": alpha,
        "render_distort": None,
        "radii": torch.zeros(total, device=render.device, dtype=torch.int32),
        "n_touched": torch.zeros(total, device=render.device, dtype=torch.int32),
        "accum_metric_counts": None,
        "viewspace_points": torch.zeros(total, 2, device=render.device, dtype=render.dtype),
        "visibility_filter": torch.zeros(total, device=render.device, dtype=torch.bool),
    }


class PFGS360Renderer:
    """Render ``anchor_scaffold_panorama`` maps through gsplat360 when available."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        extra_gsplat360_roots: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.config = config or {}
        self.extra_gsplat360_roots = extra_gsplat360_roots or []
        self.allow_fallback = bool(allow_fallback)

    def _training_cfg(self, gaussians) -> dict[str, Any]:
        cfg = getattr(gaussians, "config", None) or self.config
        return cfg.get("Training", {}) if isinstance(cfg, dict) else {}

    def render(
        self,
        camera: PanoRenderCamera,
        gaussians,
        *,
        background: torch.Tensor | None = None,
    ) -> RenderPackage:
        if background is None:
            background = torch.zeros(3, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
        if int(gaussians.get_xyz.shape[0]) == 0:
            return _blank_package(camera, gaussians, background)

        rasterization = _optional_gsplat360(self.extra_gsplat360_roots)
        if rasterization is None:
            if not self.allow_fallback:
                raise ImportError(
                    "gsplat360 is unavailable. Install/build the PFGS360 gsplat360 package "
                    "or enable the explicit smoke fallback."
                )
            return self._render_fallback(camera, gaussians, background)

        return self._render_gsplat360(rasterization, camera, gaussians, background)

    def _render_gsplat360(
        self,
        rasterization,
        camera: PanoRenderCamera,
        gaussians,
        background: torch.Tensor,
    ) -> RenderPackage:
        H = int(camera.image_height)
        W = int(camera.image_width)
        xyz = gaussians.get_xyz
        device = xyz.device
        dtype = xyz.dtype
        training_cfg = self._training_cfg(gaussians)
        if bool(training_cfg.get("pfgs360_packed", False)):
            raise NotImplementedError("PFGS360 backend requires pfgs360_packed=False.")
        render_mode = str(training_cfg.get("pfgs360_render_mode", "RGB+ED"))
        if render_mode != "RGB+ED":
            raise NotImplementedError("PFGS360 backend expects pfgs360_render_mode='RGB+ED'.")

        viewmat = camera.w2c.to(device=device, dtype=dtype)
        K = torch.tensor(
            [[W / 2.0, 0.0, (W - 1.0) / 2.0], [0.0, H / 2.0, (H - 1.0) / 2.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        render, alpha, render_distort, info = rasterization(
            means=xyz,
            quats=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            opacities=gaussians.get_opacity.squeeze(-1),
            colors=gaussians.get_features,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            packed=False,
            backgrounds=background.to(device=device, dtype=dtype).view(1, 3),
            near_plane=float(training_cfg.get("pfgs360_near_plane", 0.01)),
            far_plane=float(training_cfg.get("pfgs360_far_plane", 1.0e5)),
            radius_clip=float(training_cfg.get("pfgs360_radius_clip", 0.0)),
            render_mode=render_mode,
            sh_degree=int(getattr(gaussians, "active_sh_degree", 0)),
            sparse_grad=False,
            absgrad=bool(training_cfg.get("pfgs360_absgrad", True)),
            distloss=bool(training_cfg.get("pfgs360_distloss", False)),
            rasterize_mode=str(training_cfg.get("pfgs360_rasterize_mode", "antialiased")),
            camera_model="equirectangular",
            ret_visible=True,
        )
        rgb = render[0, ..., :3].permute(2, 0, 1).contiguous()
        depth = render[0, ..., 3:4].permute(2, 0, 1).contiguous()
        opacity = alpha[0].permute(2, 0, 1).contiguous()
        means2d = info["means2d"]
        if means2d.requires_grad:
            means2d.retain_grad()
        radii = info["radii"][0]
        visibility_filter = radii > 0
        n_touched = info.get("accum_times")
        if n_touched is not None:
            n_touched = n_touched[0].to(device=device, dtype=torch.int32)
        else:
            n_touched = visibility_filter.to(dtype=torch.int32)
        return {
            "render": rgb,
            "gs_only": rgb,
            "sky_bg_only": torch.zeros_like(rgb),
            "sky_bg_alpha": torch.zeros_like(opacity),
            "depth": depth,
            "opacity": opacity,
            "alpha": opacity,
            "render_distort": render_distort[0] if render_distort is not None else None,
            "radii": radii,
            "n_touched": n_touched,
            "accum_metric_counts": None,
            "viewspace_points": means2d,
            "visibility_filter": visibility_filter,
        }

    def _render_fallback(
        self,
        camera: PanoRenderCamera,
        gaussians,
        background: torch.Tensor,
    ) -> RenderPackage:
        """Naive ERP point renderer for tests.

        This fallback is intentionally not a replacement for gsplat360.  It
        only lets the project run integration tests without the CUDA extension.
        """
        H = int(camera.image_height)
        W = int(camera.image_width)
        xyz = gaussians.get_xyz.detach()
        device = xyz.device
        dtype = xyz.dtype
        render = background.to(device=device, dtype=dtype).view(3, 1, 1).expand(3, H, W).clone()
        depth = torch.zeros(1, H, W, device=device, dtype=dtype)
        alpha = torch.zeros(1, H, W, device=device, dtype=dtype)
        if xyz.numel() == 0:
            return _blank_package(camera, gaussians, background)

        w2c = camera.w2c.to(device=device, dtype=dtype)
        xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        valid = cam[:, 2] > 1e-4
        if not bool(valid.any()):
            return _blank_package(camera, gaussians, background)

        cam_v = cam[valid]
        colors = gaussians.get_features.detach()[valid, :3].clamp(0.0, 1.0)
        opacity = gaussians.get_opacity.detach()[valid, 0].clamp(0.0, 1.0)
        pixels = bearing_to_erp_pixel(cam_v, H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        dist = torch.linalg.norm(cam_v, dim=-1)
        for row in range(cam_v.shape[0]):
            a = opacity[row]
            u = int(ui[row])
            v = int(vi[row])
            render[:, v, u] = (1.0 - a) * render[:, v, u] + a * colors[row]
            depth[:, v, u] = dist[row]
            alpha[:, v, u] = torch.maximum(alpha[:, v, u], a)
        total = int(xyz.shape[0])
        radii = torch.zeros(total, device=device, dtype=torch.int32)
        radii[valid] = 1
        return {
            "render": render,
            "gs_only": render,
            "sky_bg_only": torch.zeros_like(render),
            "sky_bg_alpha": torch.zeros_like(alpha),
            "depth": depth,
            "opacity": alpha,
            "alpha": alpha,
            "render_distort": None,
            "radii": radii,
            "n_touched": radii,
            "accum_metric_counts": None,
            "viewspace_points": pixels,
            "visibility_filter": radii > 0,
        }
