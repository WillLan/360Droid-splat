import math
from typing import Dict

import torch


def erp_latitude_map(height: int, width: int, device, dtype) -> torch.Tensor:
    """Return ERP latitude in radians for each pixel center.

    Output shape: (1, H, W), range [-pi/2, pi/2].
    """
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    lat = torch.pi * (v / height - 0.5)
    return lat.view(1, height, 1).expand(1, height, width)


def build_pano_region_masks(
    sky_mask: torch.Tensor,
    horizon_deg: float = 18.0,
    top_pole_deg: float = 65.0,
    bottom_pole_deg: float = 55.0,
) -> Dict[str, torch.Tensor]:
    """Partition an ERP frame into sky / horizon / parallax / top/bottom poles.

    Args:
        sky_mask: (1, H, W) bool tensor.
        horizon_deg: absolute latitude threshold for horizon band.
        top_pole_deg: top-pole latitude threshold in degrees.
        bottom_pole_deg: bottom-pole latitude threshold in degrees.
    """
    if sky_mask.ndim != 3 or sky_mask.shape[0] != 1:
        raise ValueError(f"Expected sky_mask with shape (1,H,W), got {tuple(sky_mask.shape)}")

    _, height, width = sky_mask.shape
    lat = erp_latitude_map(height, width, device=sky_mask.device, dtype=torch.float32)
    lat_deg = lat * (180.0 / math.pi)
    abs_lat_deg = lat_deg.abs()

    top_pole = (lat_deg <= -float(top_pole_deg)) & (~sky_mask)
    bottom_pole = (lat_deg >= float(bottom_pole_deg)) & (~sky_mask)
    horizon = (abs_lat_deg <= float(horizon_deg)) & (~sky_mask)
    parallax = (~sky_mask) & (~top_pole) & (~bottom_pole) & (~horizon)
    degenerate = top_pole | bottom_pole

    return {
        "sky": sky_mask.bool(),
        "horizon": horizon.bool(),
        "parallax": parallax.bool(),
        "top_pole": top_pole.bool(),
        "bottom_pole": bottom_pole.bool(),
        "degenerate": degenerate.bool(),
    }
