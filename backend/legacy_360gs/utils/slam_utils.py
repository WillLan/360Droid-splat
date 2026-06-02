import torch
import torch.nn.functional as F

from backend.legacy_360gs.gaussian_splatting.utils.loss_utils import ssim
from backend.legacy_360gs.utils.pano_masking import get_viewpoint_ignore_mask


def erp_area_weight(H: int, W: int, device, dtype, viewpoint=None, config=None):
    """Per-pixel spherical area weight for ERP images.

    ERP tiles the sphere uniformly in longitude/latitude, so pixels near the
    poles correspond to a much smaller solid angle than equatorial pixels.
    Weighting by cos(latitude) = cos(蟺*(v/H 鈭?0.5)) makes the loss consistent
    with the spherical measure d惟 = cos(蠁) d位 d蠁.

    Returns a (1, H, W) tensor normalised so that its mean equals 1.
    """
    v = torch.arange(H, device=device, dtype=dtype) + 0.5
    lat = torch.pi * (v / H - 0.5)           # latitude in [-蟺/2, 蟺/2]
    w = torch.cos(lat)                         # (H,)
    w = w.view(1, H, 1).expand(1, H, W)       # (1, H, W)
    if viewpoint is not None and config is not None:
        region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
        bottom_pole = region_masks.get("bottom_pole", None)
        if bottom_pole is not None:
            if isinstance(bottom_pole, torch.Tensor):
                bottom_pole = bottom_pole.to(device=device, dtype=torch.bool)
            else:
                bottom_pole = torch.from_numpy(bottom_pole).to(device=device, dtype=torch.bool)
            if bottom_pole.ndim == 2:
                bottom_pole = bottom_pole.unsqueeze(0)
            floor = float(
                config.get("Training", {}).get("erp_bottom_pole_area_weight_floor", 0.0)
            )
            if floor > 0:
                w = torch.where(bottom_pole, torch.clamp(w, min=floor), w)
    w = w / w.mean().clamp_min(1e-8)          # normalise to mean=1
    return w


def erp_top_latitude_mask(H: int, W: int, device, config):
    top_deg = float(config.get("Training", {}).get("erp_region_top_pole_deg", 65.0))
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    lat_deg = 180.0 * (v / H - 0.5)
    mask = (lat_deg <= -top_deg).view(1, H, 1).expand(1, H, W)
    return mask

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def _is_panorama_config(config):
    return config.get("Dataset", {}).get("type", "") == "panorama"


def _charbonnier(residual, eps):
    return torch.sqrt(residual * residual + eps * eps)


def _depth_alignment_enabled(config):
    return bool(config.get("Training", {}).get("dap_align_to_render_depth", True))


def align_mono_depth_to_render_torch(
    render_depth,
    mono_depth,
    valid_mask,
    config,
    opacity=None,
):
    """Align per-frame DAP depth to current rendered metric depth.

    DAP is monocular and can drift in scale.  We keep it useful as a relative
    geometric prior by fitting it to render_depth on reliable non-sky pixels,
    then callers can use the aligned map for robust residuals.
    """
    if mono_depth is None or render_depth is None:
        return mono_depth, mono_depth.new_tensor(1.0) if mono_depth is not None else None, valid_mask
    if not _depth_alignment_enabled(config):
        return mono_depth, mono_depth.new_tensor(1.0), valid_mask

    tr_cfg = config.get("Training", {})
    min_depth = float(tr_cfg.get("dap_align_min_depth", 0.05))
    max_depth = float(tr_cfg.get("dap_align_max_depth", tr_cfg.get("dap_depth_max_valid", 99.9)))
    min_pixels = int(tr_cfg.get("dap_align_min_pixels", 512))
    opacity_min = float(tr_cfg.get("dap_align_min_opacity", 0.2))
    align_model = str(tr_cfg.get("dap_align_model", "scale")).lower()

    valid = valid_mask & torch.isfinite(mono_depth) & torch.isfinite(render_depth)
    valid = valid & (mono_depth > min_depth) & (mono_depth < max_depth)
    valid = valid & (render_depth > min_depth) & (render_depth < max_depth)
    if opacity is not None:
        valid = valid & (opacity > opacity_min)

    if int(valid.sum().item()) < min_pixels:
        return mono_depth, mono_depth.new_tensor(1.0), valid_mask

    ratios = (render_depth[valid] / mono_depth[valid].clamp_min(min_depth)).detach()
    ratios = ratios[torch.isfinite(ratios)]
    if int(ratios.numel()) < min_pixels:
        return mono_depth, mono_depth.new_tensor(1.0), valid_mask

    if align_model in {"scale_shift", "affine"}:
        x = mono_depth[valid].detach().to(torch.float32)
        y = render_depth[valid].detach().to(torch.float32)
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        denom = (x_centered * x_centered).mean().clamp_min(1e-8)
        scale = (x_centered * (y - y_mean)).mean() / denom
        shift = y_mean - scale * x_mean
        aligned = mono_depth * scale.to(dtype=mono_depth.dtype, device=mono_depth.device)
        aligned = aligned + shift.to(dtype=mono_depth.dtype, device=mono_depth.device)
    else:
        scale = torch.median(ratios)
        aligned = mono_depth * scale.to(dtype=mono_depth.dtype, device=mono_depth.device)
    aligned_valid = valid_mask & torch.isfinite(aligned) & (aligned > min_depth) & (aligned < max_depth)
    return aligned, scale, aligned_valid


def robust_relative_depth_loss(
    render_depth,
    mono_depth,
    valid_mask,
    area_w,
    config,
    charbonnier_eps,
    opacity=None,
    loss_type=None,
):
    aligned_depth, _, aligned_valid = align_mono_depth_to_render_torch(
        render_depth,
        mono_depth,
        valid_mask,
        config,
        opacity=opacity,
    )
    if aligned_depth is None:
        return render_depth.new_tensor(0.0), aligned_depth, aligned_valid
    valid = aligned_valid
    if not valid.any():
        return render_depth.new_tensor(0.0), aligned_depth, valid

    tr_cfg = config.get("Training", {})
    depth_loss_type = str(loss_type or tr_cfg.get("erp_depth_loss_type", "relative_charbonnier")).lower()
    if depth_loss_type in {"berhu", "reverse_huber"}:
        abs_err = torch.abs(render_depth - aligned_depth)
        c = float(tr_cfg.get("erp_depth_berhu_threshold", 0.0))
        if c <= 0.0:
            c = 0.2 * float(abs_err[valid].detach().max().clamp_min(1e-6).item())
        c_t = render_depth.new_tensor(c).clamp_min(1e-6)
        berhu = torch.where(
            abs_err <= c_t,
            abs_err,
            (abs_err * abs_err + c_t * c_t) / (2.0 * c_t),
        )
        loss = (berhu * area_w)[valid].mean()
        return loss, aligned_depth, valid

    rel_clip = float(tr_cfg.get("dap_depth_rel_clip", 0.2))
    denom_min = float(tr_cfg.get("dap_depth_loss_min_denom", 1.0))
    rel = (render_depth - aligned_depth) / aligned_depth.clamp_min(denom_min)
    rel = rel.clamp(min=-rel_clip, max=rel_clip)
    loss = (_charbonnier(rel, charbonnier_eps) * area_w)[valid].mean()
    return loss, aligned_depth, valid


def align_mono_depth_to_render_np(
    render_depth,
    mono_depth,
    valid_mask,
    config,
    opacity=None,
    *,
    return_stats=False,
):
    import numpy as np

    if mono_depth is None or render_depth is None:
        result = (mono_depth, 1.0, valid_mask)
        if return_stats:
            return (*result, {"align_shift": 0.0, "align_model": "none", "fit_pixels": 0})
        return result
    if not _depth_alignment_enabled(config):
        result = (mono_depth.astype(np.float32, copy=True), 1.0, valid_mask)
        if return_stats:
            return (*result, {"align_shift": 0.0, "align_model": "disabled", "fit_pixels": 0})
        return result

    tr_cfg = config.get("Training", {})
    min_depth = float(tr_cfg.get("dap_align_min_depth", 0.05))
    max_depth = float(tr_cfg.get("dap_align_max_depth", tr_cfg.get("dap_depth_max_valid", 99.9)))
    min_pixels = int(tr_cfg.get("dap_align_min_pixels", 512))
    opacity_min = float(tr_cfg.get("dap_align_min_opacity", 0.2))
    align_model = str(tr_cfg.get("dap_align_model", "scale")).lower()

    render_depth = np.asarray(render_depth, dtype=np.float32)
    mono_depth = np.asarray(mono_depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(mono_depth) & np.isfinite(render_depth)
    valid &= (mono_depth > min_depth) & (mono_depth < max_depth)
    valid &= (render_depth > min_depth) & (render_depth < max_depth)
    if opacity is not None:
        valid &= np.asarray(opacity, dtype=np.float32) > opacity_min

    fit_stats = {"align_shift": 0.0, "align_model": align_model, "fit_pixels": int(valid.sum())}
    if int(valid.sum()) < min_pixels:
        result = (mono_depth.copy(), 1.0, np.asarray(valid_mask, dtype=bool))
        if return_stats:
            return (*result, fit_stats)
        return result

    ratios = render_depth[valid] / np.clip(mono_depth[valid], min_depth, None)
    ratios = ratios[np.isfinite(ratios)]
    if int(ratios.size) < min_pixels:
        result = (mono_depth.copy(), 1.0, np.asarray(valid_mask, dtype=bool))
        if return_stats:
            fit_stats["fit_pixels"] = int(ratios.size)
            return (*result, fit_stats)
        return result

    if align_model in {"scale_shift", "affine"}:
        x = mono_depth[valid].astype(np.float64)
        y = render_depth[valid].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if int(x.size) < min_pixels:
            result = (mono_depth.copy(), 1.0, np.asarray(valid_mask, dtype=bool))
            if return_stats:
                fit_stats["fit_pixels"] = int(x.size)
                return (*result, fit_stats)
            return result

        def _fit_scale_shift(x_fit, y_fit):
            design = np.stack([x_fit, np.ones_like(x_fit)], axis=1)
            scale_fit, shift_fit = np.linalg.lstsq(design, y_fit, rcond=None)[0]
            return float(scale_fit), float(shift_fit)

        scale, shift = _fit_scale_shift(x, y)
        trim_q = float(tr_cfg.get("dap_align_lstsq_trim_quantile", 0.05))
        if 0.0 < trim_q < 0.5 and int(x.size) >= min_pixels * 2:
            residual = np.abs(scale * x + shift - y)
            keep = residual <= np.quantile(residual, 1.0 - trim_q)
            if int(keep.sum()) >= min_pixels:
                x = x[keep]
                y = y[keep]
                scale, shift = _fit_scale_shift(x, y)

        if not np.isfinite(scale) or not np.isfinite(shift):
            scale, shift = float(np.median(ratios)), 0.0
        aligned = (mono_depth * scale + shift).astype(np.float32)
        fit_stats.update(
            {
                "align_shift": shift,
                "fit_pixels": int(x.size),
                "fit_rmse": float(np.sqrt(np.mean((scale * x + shift - y) ** 2))) if x.size else 0.0,
            }
        )
    else:
        scale = float(np.median(ratios))
        aligned = (mono_depth * scale).astype(np.float32)

    aligned_valid = np.asarray(valid_mask, dtype=bool).copy()
    aligned_valid &= np.isfinite(aligned) & (aligned > min_depth) & (aligned < max_depth)
    if return_stats:
        return aligned, scale, aligned_valid, fit_stats
    return aligned, scale, aligned_valid

def _get_erp_sky_rgb_weight(config, phase: str) -> float:
    training_cfg = config.get("Training", {})
    if phase == "tracking":
        return float(
            training_cfg.get(
                "erp_tracking_sky_rgb_weight",
                training_cfg.get("erp_sky_rgb_weight", 0.15),
            )
        )
    if phase == "mapping":
        return float(
            training_cfg.get(
                "erp_mapping_sky_rgb_weight",
                training_cfg.get("erp_sky_rgb_weight", 0.15),
            )
        )
    raise ValueError(f"Unsupported ERP sky RGB weight phase: {phase}")


def _get_panorama_supervision(viewpoint, config, device, dtype, depth_shape=None):
    gt_image = viewpoint.original_image.to(device=device, dtype=dtype)
    _, h, w = gt_image.shape
    mask_shape = (1, h, w) if depth_shape is None else depth_shape

    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    ignore_mask = get_viewpoint_ignore_mask(viewpoint, config, device=device)
    if ignore_mask.ndim == 2:
        ignore_mask = ignore_mask.unsqueeze(0)
    ignore_mask = ignore_mask.view(*mask_shape)
    rgb_mask = rgb_mask & (~ignore_mask)

    grad_mask = getattr(viewpoint, "grad_mask", None)
    if grad_mask is not None:
        rgb_mask = rgb_mask * grad_mask.view(*mask_shape)

    mono_depth_np = getattr(viewpoint, "mono_depth", None)
    depth_valid_max = float(
        config["Training"].get(
            "dap_depth_max_valid",
            config["Training"].get("ransac", {}).get("depth_max", 80.0),
        )
    )
    sky_threshold = float(
        config["Training"].get("erp_sky_depth_threshold", depth_valid_max)
    )
    if mono_depth_np is None:
        mono_depth = None
        depth_valid = torch.zeros(mask_shape, device=device, dtype=torch.bool)
        sky_mask = torch.zeros(mask_shape, device=device, dtype=torch.bool)
    else:
        mono_depth = torch.from_numpy(mono_depth_np).to(device=device, dtype=dtype)[None]
        depth_valid = (mono_depth > 0.01) & (mono_depth < depth_valid_max)
        sky_mask = mono_depth >= sky_threshold
        if depth_shape is not None and tuple(mono_depth.shape) != tuple(depth_shape):
            mono_depth = mono_depth.view(*depth_shape)
            depth_valid = depth_valid.view(*depth_shape)
            sky_mask = sky_mask.view(*depth_shape)
    erp_sky_mask = getattr(viewpoint, "erp_sky_mask", None)
    if erp_sky_mask is not None:
        if isinstance(erp_sky_mask, torch.Tensor):
            erp_sky_mask = erp_sky_mask.to(device=device, dtype=torch.bool)
        else:
            erp_sky_mask = torch.from_numpy(erp_sky_mask).to(device=device, dtype=torch.bool)
        if erp_sky_mask.ndim == 2:
            erp_sky_mask = erp_sky_mask.unsqueeze(0)
        sky_mask = sky_mask | erp_sky_mask.view(*mask_shape)
    elif getattr(viewpoint, "erp_region_masks", None):
        region_sky = viewpoint.erp_region_masks.get("sky", None)
        if region_sky is not None:
            if isinstance(region_sky, torch.Tensor):
                region_sky = region_sky.to(device=device, dtype=torch.bool)
            else:
                region_sky = torch.from_numpy(region_sky).to(device=device, dtype=torch.bool)
            if region_sky.ndim == 2:
                region_sky = region_sky.unsqueeze(0)
            sky_mask = sky_mask | region_sky.view(*mask_shape)

    nonsky_mask = rgb_mask & (~sky_mask)
    sky_rgb_mask = rgb_mask & sky_mask
    return {
        "gt_image": gt_image,
        "nonsky_mask": nonsky_mask,
        "sky_mask": sky_mask,
        "sky_rgb_mask": sky_rgb_mask,
        "mono_depth": mono_depth,
        "depth_valid": depth_valid,
        "depth_valid_max": depth_valid_max,
    }

def _tv_loss(image: torch.Tensor) -> torch.Tensor:
    if image is None or image.numel() == 0:
        return torch.tensor(0.0)
    loss_h = torch.abs(image[:, :, 1:] - image[:, :, :-1]).mean()
    loss_v = torch.abs(image[:, 1:, :] - image[:, :-1, :]).mean()
    return loss_h + loss_v


def _get_structured_region_weights(viewpoint, device, dtype, shape):
    region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
    horizon = region_masks.get("horizon", None)
    parallax = region_masks.get("parallax", None)
    top_pole = region_masks.get("top_pole", None)
    bottom_pole = region_masks.get("bottom_pole", None)

    weights = torch.ones(shape, device=device, dtype=dtype)

    def _to_mask(mask_like):
        if mask_like is None:
            return None
        if isinstance(mask_like, torch.Tensor):
            mask = mask_like.to(device=device, dtype=torch.bool)
        else:
            mask = torch.from_numpy(mask_like).to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.view(*shape)

    horizon = _to_mask(horizon)
    parallax = _to_mask(parallax)
    top_pole = _to_mask(top_pole)
    bottom_pole = _to_mask(bottom_pole)

    if horizon is not None:
        weights = weights + 0.25 * horizon.float()
    if parallax is not None:
        weights = weights + 0.10 * parallax.float()
    top_weight = float(
        getattr(viewpoint, "config_training_overrides", {}).get("erp_top_pole_struct_weight", -1.0)
    )
    bottom_weight = float(
        getattr(viewpoint, "config_training_overrides", {}).get("erp_bottom_pole_struct_weight", -1.0)
    )
    if top_weight < 0.0:
        top_weight = 0.2
    if bottom_weight < 0.0:
        bottom_weight = 0.85
    if top_pole is not None:
        weights = torch.where(top_pole, weights.new_full((), top_weight), weights)
    if bottom_pole is not None:
        weights = torch.where(bottom_pole, weights.new_full((), bottom_weight), weights)
    return weights.clamp_min(0.1)


def get_loss_tracking(
    config, image, depth, opacity, viewpoint, initialization=False, return_details=False
):
    _ = initialization
    image_ab = image

    if _is_panorama_config(config):
        return get_loss_tracking_erp(
            config, image_ab, depth, opacity, viewpoint, return_details=return_details
        )

    if config["Training"]["monocular"] and config["Dataset"]["depth_loss"]:
        #return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_erp(config, image, depth, opacity, viewpoint, return_details=False):
    supervision = _get_panorama_supervision(
        viewpoint,
        config,
        device=image.device,
        dtype=image.dtype,
        depth_shape=depth.shape,
    )
    gt_image = supervision["gt_image"]
    nonsky_mask = supervision["nonsky_mask"]
    sky_rgb_mask = supervision["sky_rgb_mask"]

    sky_weight = _get_erp_sky_rgb_weight(config, phase="tracking")
    lambda_dssim = float(config["Training"].get("erp_tracking_lambda_dssim", 0.1))
    depth_weight = float(config["Training"].get("erp_tracking_depth_weight", 0.1))
    coverage_weight = float(config["Training"].get("erp_tracking_coverage_weight", 0.05))
    opacity_mix = float(config["Training"].get("erp_tracking_opacity_mix", 0.5))
    coverage_floor = float(config["Training"].get("erp_tracking_min_opacity", 0.08))
    charbonnier_eps = float(
        config["Training"].get("erp_tracking_charbonnier_eps", 1e-3)
    )
    use_area_weight = bool(config["Training"].get("erp_area_weight", True))

    _, H, W = gt_image.shape
    area_w = (
        erp_area_weight(H, W, device=image.device, dtype=image.dtype, viewpoint=viewpoint, config=config)
        if use_area_weight
        else torch.ones(1, H, W, device=image.device, dtype=image.dtype)
    )

    pixel_weights = (nonsky_mask.float() + sky_weight * sky_rgb_mask.float()) * area_w
    structured_w = _get_structured_region_weights(
        viewpoint, image.device, image.dtype, depth.shape
    )
    pixel_weights = pixel_weights * structured_w
    raw_consistency_mask = getattr(viewpoint, "erp_consistency_mask", None)
    consistency_mask = None
    consistency_ratio = image.new_tensor(0.0)
    if (
        bool(config["Training"].get("enable_sca_refine_mask", False))
        and raw_consistency_mask is not None
    ):
        if isinstance(raw_consistency_mask, torch.Tensor):
            consistency_mask = raw_consistency_mask.to(device=image.device, dtype=torch.bool)
        else:
            consistency_mask = torch.from_numpy(raw_consistency_mask).to(
                device=image.device, dtype=torch.bool
            )
        if consistency_mask.ndim == 2:
            consistency_mask = consistency_mask.unsqueeze(0)
        consistency_mask = consistency_mask.view(*depth.shape)
        min_ratio = float(config["Training"].get("sca_refine_min_mask_ratio", 0.08))
        consistency_ratio = consistency_mask.float().mean()
        if float(consistency_ratio.detach().item()) >= min_ratio:
            floor = float(config["Training"].get("sca_refine_inconsistent_rgb_weight", 0.20))
            consistency_w = torch.where(
                consistency_mask,
                torch.ones_like(pixel_weights),
                pixel_weights.new_full((), floor),
            )
            pixel_weights = pixel_weights * consistency_w
        else:
            consistency_mask = None
    effective_opacity = opacity_mix * opacity.clamp(0.0, 1.0) + (1.0 - opacity_mix)

    residual = image - gt_image
    l_rgb = (_charbonnier(residual, charbonnier_eps) * pixel_weights * effective_opacity).mean()
    l_ssim = 1.0 - ssim(image * pixel_weights, gt_image * pixel_weights)

    mono_depth = supervision["mono_depth"]
    depth_valid = supervision["depth_valid"] & nonsky_mask
    if depth_weight > 0.0:
        if consistency_mask is not None:
            depth_valid = depth_valid & consistency_mask
        if mono_depth is not None:
            depth_valid = depth_valid & (opacity > 0.1)
        if mono_depth is not None and depth is not None and depth_valid.any():
            l_depth, mono_depth, depth_valid = robust_relative_depth_loss(
                depth,
                mono_depth,
                depth_valid,
                area_w,
                config,
                charbonnier_eps,
                opacity=opacity,
            )
        else:
            l_depth = image.new_tensor(0.0)
    else:
        l_depth = image.new_tensor(0.0)

    if nonsky_mask.any():
        l_coverage = (
            F.relu(coverage_floor - opacity.clamp(0.0, 1.0))
            * nonsky_mask.float()
            * area_w
        ).mean()
    else:
        l_coverage = image.new_tensor(0.0)

    loss = (1.0 - lambda_dssim) * l_rgb + lambda_dssim * l_ssim
    loss = loss + depth_weight * l_depth + coverage_weight * l_coverage

    if return_details:
        details = {
            "rgb": l_rgb.detach(),
            "ssim": l_ssim.detach(),
            "depth": l_depth.detach(),
            "coverage": l_coverage.detach(),
            "coverage_ratio": (opacity > 0.1).float().mean().detach(),
            "nonsky_coverage_ratio": (
                ((opacity > 0.1) & nonsky_mask).float().sum()
                / nonsky_mask.float().sum().clamp_min(1.0)
            ).detach(),
            "depth_valid_ratio": (
                depth_valid.float().sum() / depth_valid.numel()
            ).detach(),
            "consistency_ratio": consistency_ratio.detach(),
        }
        return loss, details
    return loss

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    # Some cameras (e.g. panoramic faces during certain stages) may not have a
    # gradient mask initialised yet; fall back to a full-ones mask in that case.
    grad_mask = getattr(viewpoint, "grad_mask", None)
    if grad_mask is not None:
        rgb_pixel_mask = rgb_pixel_mask * grad_mask

    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    return l1.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.mono_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def get_loss_mapping(
    config, image, viewpoint, depth=None, initialization=False, monodepth=True
):
    _ = initialization
    image_ab = image

    if _is_panorama_config(config):
        return get_loss_mapping_erp(config, image_ab, viewpoint, depth=depth)

    if config["Training"]["monocular"] and monodepth:
        return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_erp(config, image, viewpoint, depth=None):
    supervision = _get_panorama_supervision(
        viewpoint,
        config,
        device=image.device,
        dtype=image.dtype,
        depth_shape=depth.shape if depth is not None else None,
    )
    gt_image = supervision["gt_image"]
    nonsky_mask = supervision["nonsky_mask"]
    sky_rgb_mask = supervision["sky_rgb_mask"]

    sky_weight = _get_erp_sky_rgb_weight(config, phase="mapping")
    mapping_lambda_dssim = float(
        config["Training"].get(
            "erp_mapping_lambda_dssim",
            config.get("opt_params", {}).get("lambda_dssim", 0.2),
        )
    )
    mapping_depth_weight = float(
        config["Training"].get("erp_mapping_depth_weight", 1.0 - config["Training"].get("alpha", 0.95))
    )
    charbonnier_eps = float(
        config["Training"].get("erp_mapping_charbonnier_eps", 1e-3)
    )
    use_area_weight = bool(config["Training"].get("erp_area_weight", True))

    _, H, W = gt_image.shape
    area_w = (
        erp_area_weight(H, W, device=image.device, dtype=image.dtype, viewpoint=viewpoint, config=config)
        if use_area_weight
        else torch.ones(1, H, W, device=image.device, dtype=image.dtype)
    )

    pixel_weights = (nonsky_mask.float() + sky_weight * sky_rgb_mask.float()) * area_w
    structured_w = _get_structured_region_weights(
        viewpoint,
        image.device,
        image.dtype,
        depth.shape if depth is not None else (1, H, W),
    )
    pixel_weights = pixel_weights * structured_w
    l_rgb = (_charbonnier(image - gt_image, charbonnier_eps) * pixel_weights).mean()
    l_ssim = 1.0 - ssim(image * pixel_weights, gt_image * pixel_weights)

    mono_depth = supervision["mono_depth"]
    depth_valid = supervision["depth_valid"] & nonsky_mask
    if mapping_depth_weight > 0.0:
        if mono_depth is not None and depth is not None and depth_valid.any():
            l_depth, mono_depth, depth_valid = robust_relative_depth_loss(
                depth,
                mono_depth,
                depth_valid,
                area_w,
                config,
                charbonnier_eps,
                loss_type=config["Training"].get("erp_mapping_depth_loss_type", "berhu"),
            )
        else:
            l_depth = image.new_tensor(0.0)
    else:
        l_depth = image.new_tensor(0.0)

    return (
        (1.0 - mapping_lambda_dssim) * l_rgb
        + mapping_lambda_dssim * l_ssim
        + mapping_depth_weight * l_depth
    )


def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    mono_depth_np = getattr(viewpoint, "mono_depth", None)
    if mono_depth_np is None:
        # No depth supervision available; fall back to RGB-only mapping loss
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(
            1, gt_image.shape[1], gt_image.shape[2]
        )
        return torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask).mean()

    gt_depth = torch.from_numpy(mono_depth_np).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
