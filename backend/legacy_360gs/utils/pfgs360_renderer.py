"""Adapter for PFGS360/gsplat360 equirectangular rasterization.

This keeps the local SLAM renderer interface while routing panorama renders
through the PFGS360 baseline's ``gsplat360.rasterization`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from backend.legacy_360gs.utils.pose_utils import SE3_exp


def _import_gsplat360_rasterization():
    try:
        from gsplat360 import rasterization
        from gsplat360.cuda import _backend as gsplat360_backend

        if gsplat360_backend._C is None:
            raise ImportError(
                "PFGS360 gsplat360 CUDA extension is not loaded. Make sure nvcc "
                "is available or use the PFGS360 environment where gsplat360 is built."
            )
        return rasterization
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        vendored_root = repo_root / "external_baselines" / "gsplat360"
        if vendored_root.is_dir() and str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        try:
            from gsplat360 import rasterization
            from gsplat360.cuda import _backend as gsplat360_backend

            if gsplat360_backend._C is None:
                raise ImportError(
                    "PFGS360 gsplat360 CUDA extension is not loaded. Make sure nvcc "
                    "is available or use the PFGS360 environment where gsplat360 is built."
                )
            return rasterization
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PFGS360 gsplat360 renderer is unavailable. Install the "
                "external_baselines/gsplat360 package (and its CUDA extension) "
                "or run in the PFGS360 environment."
            ) from exc


def _as_delta(delta, fallback: torch.Tensor, *, device, dtype) -> torch.Tensor:
    if delta is None:
        delta = fallback
    delta = delta.reshape(-1)[-3:]
    return delta.to(device=device, dtype=dtype)


def _build_viewmat(body_cam, theta=None, rho=None) -> torch.Tensor:
    """Build a differentiable row-major world-to-camera matrix for gsplat360."""
    dtype = body_cam.R.dtype
    device = body_cam.R.device
    base = torch.eye(4, device=device, dtype=dtype)
    base[:3, :3] = body_cam.R.to(device=device, dtype=dtype)
    base[:3, 3] = body_cam.T.to(device=device, dtype=dtype)

    rot_delta = _as_delta(theta, body_cam.cam_rot_delta, device=device, dtype=dtype)
    trans_delta = _as_delta(rho, body_cam.cam_trans_delta, device=device, dtype=dtype)
    tau = torch.cat([trans_delta, rot_delta], dim=0)
    return SE3_exp(tau) @ base


def _dummy_intrinsics(body_cam, *, device, dtype) -> torch.Tensor:
    # gsplat360's equirectangular projection ignores K, but the API requires it.
    fx = float(getattr(body_cam, "fx", max(1.0, body_cam.image_width / 2.0)))
    fy = float(getattr(body_cam, "fy", max(1.0, body_cam.image_height / 2.0)))
    cx = float(getattr(body_cam, "cx", (body_cam.image_width - 1) / 2.0))
    cy = float(getattr(body_cam, "cy", (body_cam.image_height - 1) / 2.0))
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _blank_package(body_cam, gaussians, background: torch.Tensor) -> dict:
    H = int(body_cam.image_height)
    W = int(body_cam.image_width)
    total = int(gaussians.get_xyz.shape[0])
    blank_rgb = background.view(3, 1, 1).expand(3, H, W).clone()
    blank_depth = torch.zeros((1, H, W), device=background.device, dtype=background.dtype)
    blank_alpha = torch.zeros((1, H, W), device=background.device, dtype=background.dtype)
    return {
        "render": blank_rgb,
        "gs_only": blank_rgb,
        "sky_bg_only": torch.zeros_like(blank_rgb),
        "sky_bg_alpha": torch.zeros_like(blank_alpha),
        "depth": blank_depth,
        "opacity": blank_alpha,
        "alpha": blank_alpha,
        "radii": torch.zeros((total,), device=background.device, dtype=torch.int32),
        "n_touched": torch.zeros((total,), device=background.device, dtype=torch.int32),
        "accum_metric_counts": None,
        "viewspace_points": torch.zeros((total, 2), device=background.device, dtype=background.dtype),
        "visibility_filter": torch.zeros((total,), device=background.device, dtype=torch.bool),
    }


def render_pfgs360_erp(
    body_cam,
    gaussians,
    background: torch.Tensor,
    theta=None,
    rho=None,
    get_flag: bool = False,
    metric_map: torch.Tensor | None = None,
    skip_erp_sky_bg: bool = False,
) -> dict:
    """Render ERP with PFGS360's gsplat360 equirectangular rasterizer."""
    if get_flag or metric_map is not None:
        raise NotImplementedError(
            "pfgs360_gsplat does not expose the current FastGS metric-map query API."
        )

    total = int(gaussians.get_xyz.shape[0])
    if total == 0:
        return _blank_package(body_cam, gaussians, background)

    rasterization = _import_gsplat360_rasterization()

    H = int(body_cam.image_height)
    W = int(body_cam.image_width)
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    training_cfg = getattr(gaussians, "config", {}).get("Training", {}) or {}
    if bool(training_cfg.get("pfgs360_packed", False)):
        raise NotImplementedError(
            "pfgs360_gsplat currently requires pfgs360_packed=False so "
            "densification can consume per-Gaussian means2d/radii tensors."
        )
    render_mode = str(training_cfg.get("pfgs360_render_mode", "RGB+ED"))
    if render_mode != "RGB+ED":
        raise NotImplementedError("pfgs360_gsplat currently expects pfgs360_render_mode='RGB+ED'.")

    viewmat = _build_viewmat(body_cam, theta=theta, rho=rho).to(device=device, dtype=dtype)
    K = _dummy_intrinsics(body_cam, device=device, dtype=dtype)

    render, alpha, render_distort, info = rasterization(
        means=gaussians.get_xyz,
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
        sh_degree=int(gaussians.active_sh_degree),
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
        scale = torch.tensor([W / 2.0, H / 2.0], device=means2d.device, dtype=means2d.dtype)
        means2d.register_hook(lambda grad: grad * scale)

    radii = info["radii"][0]
    visibility_filter = radii > 0
    accum_times = info.get("accum_times")
    if accum_times is not None:
        n_touched = accum_times[0].to(device=device, dtype=torch.int32)
    else:
        n_touched = visibility_filter.to(dtype=torch.int32)

    gs_only = rgb
    sky_bg_only = torch.zeros_like(rgb)
    sky_bg_alpha = torch.zeros_like(opacity)
    bg_enabled = bool(
        training_cfg.get("enable_erp_sky_background", False)
        or training_cfg.get("enable_neural_sky_bg", False)
    )
    if bg_enabled and not skip_erp_sky_bg:
        from backend.legacy_360gs.utils.panoramic_renderer import _get_body_cam_sky_mask, compose_erp_sky_background

        sky_bg = gaussians.get_erp_sky_background(body_cam)
        sky_mask = _get_body_cam_sky_mask(body_cam, H, W, device=rgb.device, dtype=rgb.dtype)
        alpha_mode = str(training_cfg.get("erp_sky_bg_alpha_mode", "fixed")).lower()
        alpha_val = float(training_cfg.get("erp_sky_bg_alpha", 0.65))
        if alpha_mode == "opacity":
            alpha_src = (1.0 - opacity.clamp(0.0, 1.0)) * alpha_val
        else:
            alpha_src = alpha_val
        rgb, sky_bg_only, sky_bg_alpha = compose_erp_sky_background(
            rgb, sky_bg, sky_mask, alpha_src
        )

    return {
        "render": rgb,
        "gs_only": gs_only,
        "sky_bg_only": sky_bg_only,
        "sky_bg_alpha": sky_bg_alpha,
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
