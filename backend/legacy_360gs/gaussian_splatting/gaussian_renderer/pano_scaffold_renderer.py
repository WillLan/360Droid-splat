from __future__ import annotations

import torch


def render_pano_scaffold_erp(
    body_cam,
    gaussians,
    background: torch.Tensor,
    theta=None,
    rho=None,
    get_flag: bool = False,
    metric_map: torch.Tensor | None = None,
    skip_erp_sky_bg: bool = False,
) -> dict:
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettingsERP,
            GaussianRasterizerERP,
        )
    except ImportError as exc:
        raise ImportError(
            "diff_gaussian_rasterization with ERP support not found. "
            "Compile submodules/diff-gaussian-rasterization with `pip install -e .`"
        ) from exc

    H = int(body_cam.image_height)
    W = int(body_cam.image_width)
    total = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype if total > 0 else background.dtype

    selection = gaussians.build_active_anchor_selection(body_cam)
    active_idx = selection.indices
    num_active = int(active_idx.numel())

    if total == 0 or num_active == 0:
        blank_rgb = background.view(3, 1, 1).expand(3, H, W).clone()
        blank_depth = torch.zeros((1, H, W), device=background.device, dtype=background.dtype)
        blank_alpha = torch.zeros((1, H, W), device=background.device, dtype=background.dtype)
        gaussians.record_render_stats(selection, radii=torch.zeros((0,), device=background.device))
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
            "accum_metric_counts": torch.zeros((H * W,), device=background.device, dtype=torch.int32)
            if get_flag
            else None,
            "viewspace_points": torch.zeros((total, 3), device=background.device, dtype=background.dtype),
            "visibility_filter": torch.zeros((total,), device=background.device, dtype=torch.bool),
            "anchor_visible_mask": selection.selection_mask,
            "selection_mask": selection.selection_mask,
        }

    settings = GaussianRasterizationSettingsERP(
        image_height=H,
        image_width=W,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=body_cam.world_view_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=body_cam.camera_center,
        prefiltered=False,
        debug=False,
        get_flag=bool(get_flag),
        metric_map=None,
    )
    rasterizer = GaussianRasterizerERP(raster_settings=settings)

    means3D = gaussians.get_xyz.index_select(0, active_idx)
    full_viewspace = torch.zeros((total, 3), device=device, dtype=dtype, requires_grad=True)
    full_viewspace.register_hook(
        lambda g: g * torch.tensor([W / 2.0, H / 2.0, 1.0], device=g.device, dtype=g.dtype)
    )
    means2D_active = full_viewspace.index_select(0, active_idx)

    if theta is None:
        theta = body_cam.cam_rot_delta.unsqueeze(0)
    if rho is None:
        rho = body_cam.cam_trans_delta.unsqueeze(0)

    metric_map_flat = None
    if metric_map is not None:
        metric_map_t = metric_map.to(device=device)
        if metric_map_t.ndim == 3:
            metric_map_t = metric_map_t.squeeze(0)
        metric_map_t = metric_map_t.to(dtype=torch.int32)
        metric_map_flat = metric_map_t.contiguous().view(-1)
    elif get_flag:
        metric_map_flat = torch.zeros(H * W, dtype=torch.int32, device=device)
    rasterizer.raster_settings = rasterizer.raster_settings._replace(metric_map=metric_map_flat)

    rendered, radii_active, depth, opacity, n_touched_active, accum_metric_counts = rasterizer(
        means3D=means3D,
        means2D=means2D_active,
        opacities=gaussians.get_opacity.index_select(0, active_idx),
        shs=gaussians.get_features.index_select(0, active_idx),
        scales=gaussians.get_scaling.index_select(0, active_idx),
        rotations=gaussians.get_rotation.index_select(0, active_idx),
        theta=theta,
        rho=rho,
    )

    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    if opacity.dim() == 2:
        opacity = opacity.unsqueeze(0)

    gs_only = rendered
    sky_bg_only = torch.zeros_like(rendered)
    sky_bg_alpha = torch.zeros_like(opacity)
    _tr_cfg = getattr(gaussians, "config", {}).get("Training", {})
    _bg_on = bool(
        _tr_cfg.get("enable_erp_sky_background", False)
        or _tr_cfg.get("enable_neural_sky_bg", False)
    )
    if _bg_on and not skip_erp_sky_bg:
        from backend.legacy_360gs.utils.panoramic_renderer import _get_body_cam_sky_mask, compose_erp_sky_background

        sky_bg = gaussians.get_erp_sky_background(body_cam)
        sky_mask = _get_body_cam_sky_mask(
            body_cam, H, W, device=rendered.device, dtype=rendered.dtype
        )
        tr = getattr(gaussians, "config", {}).get("Training", {}) or {}
        alpha_mode = str(tr.get("erp_sky_bg_alpha_mode", "fixed")).lower()
        alpha_val = float(tr.get("erp_sky_bg_alpha", 0.65))
        if alpha_mode == "opacity":
            alpha_src = (1.0 - opacity.clamp(0.0, 1.0)) * alpha_val
        else:
            alpha_src = alpha_val
        rendered, sky_bg_only, sky_bg_alpha = compose_erp_sky_background(
            rendered, sky_bg, sky_mask, alpha_src
        )

    full_radii = torch.zeros((total,), device=device, dtype=radii_active.dtype)
    full_radii.index_copy_(0, active_idx, radii_active)
    full_n_touched = torch.zeros((total,), device=device, dtype=n_touched_active.dtype)
    full_n_touched.index_copy_(0, active_idx, n_touched_active)
    visibility_filter = full_radii > 0

    gaussians.record_render_stats(selection, radii=radii_active)
    return {
        "render": rendered,
        "gs_only": gs_only,
        "sky_bg_only": sky_bg_only,
        "sky_bg_alpha": sky_bg_alpha,
        "depth": depth,
        "opacity": opacity,
        "alpha": opacity,
        "radii": full_radii,
        "n_touched": full_n_touched,
        "accum_metric_counts": accum_metric_counts,
        "viewspace_points": full_viewspace,
        "visibility_filter": visibility_filter,
        "anchor_visible_mask": visibility_filter,
        "selection_mask": selection.selection_mask,
    }
