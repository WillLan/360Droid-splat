import numpy as np
import torch


def _get_ignore_rect_reference_size(config, width: int, height: int):
    training = config.get("Training", {}) if config else {}
    dataset = config.get("Dataset", {}) if config else {}
    ref_size = training.get("erp_ignore_rect_reference_size", None)
    if ref_size is not None and len(ref_size) == 2:
        return float(ref_size[0]), float(ref_size[1])
    calib = dataset.get("Calibration", {}) if dataset else {}
    ref_w = dataset.get("erp_resize_width", calib.get("width", width))
    ref_h = dataset.get("erp_resize_height", calib.get("height", height))
    return float(ref_w), float(ref_h)


def build_erp_ignore_mask(height: int, width: int, config, device=None):
    training = config.get("Training", {}) if config else {}
    rects = training.get("erp_ignore_rects", []) or []
    if device is not None:
        mask = torch.zeros((1, height, width), dtype=torch.bool, device=device)
    else:
        mask = np.zeros((height, width), dtype=bool)
    if not rects:
        return mask

    ref_w, ref_h = _get_ignore_rect_reference_size(config, width, height)
    scale_x = float(width) / max(ref_w, 1.0)
    scale_y = float(height) / max(ref_h, 1.0)

    for rect in rects:
        if rect is None or len(rect) != 4:
            continue
        x, y, w, h = [float(v) for v in rect]
        x0 = max(0, min(width, int(np.floor(x * scale_x))))
        y0 = max(0, min(height, int(np.floor(y * scale_y))))
        x1 = max(0, min(width, int(np.ceil((x + w) * scale_x))))
        y1 = max(0, min(height, int(np.ceil((y + h) * scale_y))))
        if x1 <= x0 or y1 <= y0:
            continue
        if device is not None:
            mask[:, y0:y1, x0:x1] = True
        else:
            mask[y0:y1, x0:x1] = True
    return mask


def get_viewpoint_ignore_mask(viewpoint, config, device=None):
    region_masks = getattr(viewpoint, "erp_region_masks", None) or {}
    ignore_mask = region_masks.get("ignore", None)
    if ignore_mask is not None:
        if isinstance(ignore_mask, torch.Tensor):
            mask = ignore_mask.to(device=device, dtype=torch.bool) if device is not None else ignore_mask.cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(ignore_mask, dtype=bool)
            mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool) if device is not None else mask_np
        if device is None and mask.ndim == 3:
            mask = mask[0]
        if device is not None and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask
    return build_erp_ignore_mask(
        int(getattr(viewpoint, "image_height", 0)),
        int(getattr(viewpoint, "image_width", 0)),
        config,
        device=device,
    )
