#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from backend.legacy_360gs.gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from backend.legacy_360gs.gaussian_splatting.utils.sh_utils import RGB2SH
from backend.legacy_360gs.gaussian_splatting.utils.system_utils import mkdir_p
from backend.legacy_360gs.utils.erp_geometry import erp_dense_pixel_center_bearings
from backend.legacy_360gs.utils.pano_masking import get_viewpoint_ignore_mask


class GaussianModel:
    REGION_TAG_DEFAULT = 0
    REGION_TAG_UPPER_SKY = 1
    REGION_TAG_HORIZON_BG = 2
    REGION_TAG_BOTTOM_POLE_GROUND = 3
    REGION_TAG_POLAR_CAP_SKY = 4

    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()
        self._is_sky = torch.zeros(0, dtype=torch.bool)  # CPU, sky Gaussian flag
        # Layer label (CPU int8): 0=near, 1=far, 2=sky.  Only populated when
        # enable_layered_map=True in config; otherwise stays empty/zero and is
        # ignored by the backend.
        self._layer = torch.zeros(0, dtype=torch.int8)   # CPU
        self._layer_pending: Optional[torch.Tensor] = None  # stash from _create_pcd_from_erp_depth
        self._region_tag = torch.zeros(0, dtype=torch.int8)     # CPU
        self._anchor_kf = torch.empty(0, dtype=torch.int32)     # CPU
        self._anchor_submap = torch.empty(0, dtype=torch.int32) # CPU
        self._birth_frame = torch.empty(0, dtype=torch.int32)   # CPU
        self._region_tag_pending: Optional[torch.Tensor] = None
        self._anchor_kf_pending: Optional[torch.Tensor] = None
        self._anchor_submap_pending: Optional[torch.Tensor] = None
        self._birth_frame_pending: Optional[torch.Tensor] = None
        self._erp_sky_bg: Optional[nn.Parameter] = None
        self._neural_sky_bg: Optional[nn.Module] = None

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_layer(self) -> torch.Tensor:
        """Return the per-Gaussian layer label (CPU int8, 0=near,1=far,2=sky)."""
        return self._layer

    def layer_mask(self, layer_id: int, device=None) -> torch.Tensor:
        """Boolean mask for Gaussians in a given layer.

        Args:
            layer_id: 0=near, 1=far, 2=sky.
            device:   target device; defaults to 'cuda'.

        Returns:
            (N,) bool tensor on `device`.
        """
        dev = device if device is not None else "cuda"
        if self._layer.shape[0] != self._xyz.shape[0]:
            # Layer not populated (enable_layered_map=False): fall back to _is_sky for layer 2
            if layer_id == 2:
                return self._is_sky.to(device=dev, dtype=torch.bool)
            return torch.ones(self._xyz.shape[0], device=dev, dtype=torch.bool) if layer_id == 0 \
                else torch.zeros(self._xyz.shape[0], device=dev, dtype=torch.bool)
        return (self._layer == layer_id).to(device=dev)

    def region_tag_mask(self, region_tag: int, device=None) -> torch.Tensor:
        dev = device if device is not None else "cuda"
        if self._region_tag.shape[0] != self._xyz.shape[0]:
            return torch.zeros((self._xyz.shape[0],), device=dev, dtype=torch.bool)
        return (self._region_tag == int(region_tag)).to(device=dev)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    _FALLBACK_SKY_RGB = np.array([0.53, 0.81, 0.92], dtype=np.float64)
    _GRAY_SKY_RGB     = np.array([0.5,  0.5,  0.5 ], dtype=np.float64)

    def _get_erp_image_np(self, cam) -> np.ndarray:
        return (
            cam.original_image.detach()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

    def _get_sky_sampling_masks(self, cam, depthmap: np.ndarray):
        tr = self.config.get("Training", {}) if self.config else {}
        mono = getattr(cam, "mono_depth", None)
        if mono is None:
            return None, None
        mono = np.asarray(mono, dtype=np.float64)
        if mono.shape != depthmap.shape:
            return None, None
        depth_valid_max = float(
            tr.get(
                "dap_depth_max_valid",
                tr.get("ransac", {}).get("depth_max", 80.0),
            )
        )
        sky_threshold = float(tr.get("erp_sky_depth_threshold", depth_valid_max))
        rgb_th = float(tr.get("rgb_boundary_threshold", 0.01))
        img_f = self._get_erp_image_np(cam)
        rgb_ok = img_f.sum(axis=-1) > rgb_th
        depth_ok = np.isfinite(mono) & (mono > 0.01)
        sky_from_mono = depth_ok & (mono >= sky_threshold)
        erp_sky = getattr(cam, "erp_sky_mask", None)
        if erp_sky is not None and tuple(erp_sky.shape) == depthmap.shape:
            sky_pixels = sky_from_mono | erp_sky.astype(bool)
        else:
            sky_pixels = sky_from_mono
        return img_f, sky_pixels & rgb_ok

    def _estimate_sky_band_rgb_erp(self, cam, depthmap: np.ndarray, band: str) -> np.ndarray:
        img_f, sky_pixels = self._get_sky_sampling_masks(cam, depthmap)
        if img_f is None or sky_pixels is None:
            return self._FALLBACK_SKY_RGB.copy()

        h, _ = depthmap.shape
        rows = (np.arange(h, dtype=np.float64) + 0.5) / h
        lat_deg = 180.0 * (rows - 0.5)
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        horizon_band_deg = float(
            tr_cfg.get("sky_horizon_band_deg", 12.0)
        )
        top_cap_deg = float(tr_cfg.get("sky_top_cap_deg", 70.0))
        if band == "polar_cap":
            row_mask = lat_deg <= -top_cap_deg
        elif band == "upper":
            row_mask = (lat_deg > -top_cap_deg) & (lat_deg <= -horizon_band_deg)
        elif band == "upper_all":
            row_mask = lat_deg <= -horizon_band_deg
        elif band == "horizon":
            row_mask = (lat_deg > -horizon_band_deg) & (lat_deg <= 0.0)
        else:
            row_mask = np.ones_like(lat_deg, dtype=bool)
        mask = sky_pixels & row_mask[:, None]
        min_px = int(tr_cfg.get("sky_color_auto_min_pixels", 256))
        if int(mask.sum()) < min_px and band == "polar_cap":
            mask = sky_pixels & (lat_deg[:, None] <= -horizon_band_deg)
        if int(mask.sum()) < min_px and band in ("upper", "upper_all"):
            mask = sky_pixels & (lat_deg[:, None] <= -horizon_band_deg)
        if int(mask.sum()) < min_px:
            mask = sky_pixels
        if int(mask.sum()) < max(64, min_px // 4):
            return self._FALLBACK_SKY_RGB.copy()
        return np.clip(img_f[mask].reshape(-1, 3).mean(axis=0), 0.0, 1.0).astype(np.float64)

    def _estimate_sky_rgb_erp(self, cam, depthmap: np.ndarray) -> np.ndarray:
        """
        Return the initial RGB colour used for sky Gaussians.

        Controlled by ``erp_sky_color_init_mode`` (Training config):
          "auto"     鈥?(default) detect mean sky colour from the image.
                       Gives smallest initial residual; gradient is weak.
          "gray"     鈥?use neutral grey [0.5, 0.5, 0.5].
                       Produces a meaningful RGB residual for any sky tone,
                       driving stronger gradient for colour AND scale.
          "fallback" 鈥?use the hard-coded sky-blue _FALLBACK_SKY_RGB.
          "black"    鈥?use [0, 0, 0].  NOTE: scale gradient is 鈭?Gaussian
                       colour, so pure black will kill scale gradients.
          "white"    鈥?use [1, 1, 1].

        Falls back to _FALLBACK_SKY_RGB if mode=="auto" is disabled or
        too few sky pixels are found.
        """
        tr = self.config.get("Training", {}) if self.config else {}

        # --- non-auto modes: return fixed colour immediately ---
        init_mode = str(tr.get("erp_sky_color_init_mode", "auto")).lower()
        if init_mode == "gray":
            return self._GRAY_SKY_RGB.copy()
        if init_mode == "fallback":
            return self._FALLBACK_SKY_RGB.copy()
        if init_mode == "black":
            return np.zeros(3, dtype=np.float64)
        if init_mode == "white":
            return np.ones(3, dtype=np.float64)
        if init_mode == "sampled":
            return self._estimate_sky_band_rgb_erp(cam, depthmap, band="all")

        # --- init_mode == "auto" ---
        if not tr.get("erp_auto_sky_color", True):
            return self._FALLBACK_SKY_RGB.copy()

        mono = getattr(cam, "mono_depth", None)
        if mono is None:
            return self._FALLBACK_SKY_RGB.copy()
        mono = np.asarray(mono, dtype=np.float64)
        if mono.shape != depthmap.shape:
            return self._FALLBACK_SKY_RGB.copy()

        depth_valid_max = float(
            tr.get(
                "dap_depth_max_valid",
                tr.get("ransac", {}).get("depth_max", 80.0),
            )
        )
        sky_threshold = float(tr.get("erp_sky_depth_threshold", depth_valid_max))
        rgb_th = float(tr.get("rgb_boundary_threshold", 0.01))
        min_px = int(tr.get("sky_color_auto_min_pixels", 256))

        depth_ok = np.isfinite(mono) & (mono > 0.01)
        sky_from_mono = depth_ok & (mono >= sky_threshold)
        erp_sky = getattr(cam, "erp_sky_mask", None)
        if erp_sky is not None and tuple(erp_sky.shape) == depthmap.shape:
            sky_pixels = sky_from_mono | erp_sky.astype(bool)
        else:
            sky_pixels = sky_from_mono

        img_f = self._get_erp_image_np(cam)
        rgb_ok = img_f.sum(axis=-1) > rgb_th
        ignore_mask = get_viewpoint_ignore_mask(cam, self.config, device=None)
        if ignore_mask.ndim == 3:
            ignore_mask = ignore_mask[0]
        rgb_ok = rgb_ok & (~ignore_mask)
        mask = sky_pixels & rgb_ok
        if int(mask.sum()) < min_px:
            return self._FALLBACK_SKY_RGB.copy()

        colors = img_f[mask].reshape(-1, 3)
        sky_rgb = np.clip(colors.mean(axis=0), 0.0, 1.0).astype(np.float64)
        return sky_rgb

    def _build_erp_sky_mask_tensor(self, cam, H: int, W: int, device: str):
        sky_mask = getattr(cam, "erp_sky_mask", None)
        if sky_mask is not None:
            if isinstance(sky_mask, torch.Tensor):
                sky_mask_t = sky_mask.to(device=device, dtype=torch.bool)
            else:
                sky_mask_t = torch.from_numpy(np.asarray(sky_mask).astype(bool)).to(
                    device=device
                )
            if sky_mask_t.ndim == 2:
                sky_mask_t = sky_mask_t.unsqueeze(0)
            return sky_mask_t.view(1, H, W)

        region_masks = getattr(cam, "erp_region_masks", None) or {}
        region_sky = region_masks.get("sky", None)
        if region_sky is not None:
            if isinstance(region_sky, torch.Tensor):
                region_sky = region_sky.to(device=device, dtype=torch.bool)
            else:
                region_sky = torch.from_numpy(np.asarray(region_sky).astype(bool)).to(
                    device=device
                )
            if region_sky.ndim == 2:
                region_sky = region_sky.unsqueeze(0)
            return region_sky.view(1, H, W)

        mono = getattr(cam, "mono_depth", None)
        if mono is None:
            return None
        mono = torch.from_numpy(np.asarray(mono).astype(np.float32)).to(device=device)
        if mono.ndim == 2:
            mono = mono.unsqueeze(0)
        if mono.shape[-2:] != (H, W):
            return None
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        depth_valid_max = float(
            tr_cfg.get(
                "dap_depth_max_valid",
                tr_cfg.get("ransac", {}).get("depth_max", 80.0),
            )
        )
        sky_threshold = float(tr_cfg.get("erp_sky_depth_threshold", depth_valid_max))
        return mono >= sky_threshold

    def _init_erp_sky_background_from_camera(self, cam, H: int, W: int):
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        down = max(1, int(tr_cfg.get("erp_sky_bg_res_down", 4)))
        h_bg = max(1, H // down)
        w_bg = max(1, W // down)
        device = "cuda"
        img = cam.original_image.detach().to(device=device, dtype=torch.float32).unsqueeze(0)
        ignore_mask = get_viewpoint_ignore_mask(cam, self.config, device=device)
        if ignore_mask.ndim == 2:
            ignore_mask = ignore_mask.unsqueeze(0)
        valid_mask = ~ignore_mask
        mean_fill = img.mean(dim=(-1, -2), keepdim=True)
        img = torch.where(valid_mask.unsqueeze(0), img, mean_fill)
        img_lr = F.interpolate(img, size=(h_bg, w_bg), mode="bilinear", align_corners=False)

        sky_mask = self._build_erp_sky_mask_tensor(cam, H, W, device=device)
        if sky_mask is not None:
            sky_mask = sky_mask & valid_mask
            sky_mask_lr = F.interpolate(
                sky_mask.float().unsqueeze(0), size=(h_bg, w_bg), mode="bilinear", align_corners=False
            ).squeeze(0) > 0.25
            if sky_mask.any():
                mean_sky = (
                    img.squeeze(0)[:, sky_mask.squeeze(0)].mean(dim=1, keepdim=True).view(3, 1, 1)
                )
            else:
                mean_sky = torch.from_numpy(self._FALLBACK_SKY_RGB).to(
                    device=device, dtype=torch.float32
                ).view(3, 1, 1)
            init_lr = mean_sky.expand(3, h_bg, w_bg).clone()
            init_lr[:, sky_mask_lr.squeeze(0)] = img_lr.squeeze(0)[:, sky_mask_lr.squeeze(0)]
        else:
            init_lr = img_lr.squeeze(0).clone()
        return init_lr.contiguous()

    def _register_erp_sky_bg_optimizer_group(self):
        if self.optimizer is None or self._erp_sky_bg is None:
            return
        for group in self.optimizer.param_groups:
            if group.get("name") == "erp_sky_bg":
                group["params"][0] = self._erp_sky_bg
                return
        lr = float(
            self.config.get("Training", {}).get(
                "erp_sky_bg_lr",
                self.config.get("opt_params", {}).get("feature_lr", 0.0025),
            )
        )
        self.optimizer.add_param_group(
            {"params": [self._erp_sky_bg], "lr": lr, "name": "erp_sky_bg"}
        )

    def ensure_erp_sky_background(self, cam):
        if not bool(self.config.get("Training", {}).get("enable_erp_sky_background", False)):
            return None
        H = int(getattr(cam, "image_height", 0))
        W = int(getattr(cam, "image_width", 0))
        if H <= 0 or W <= 0:
            return None
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        down = max(1, int(tr_cfg.get("erp_sky_bg_res_down", 4)))
        target_shape = (3, max(1, H // down), max(1, W // down))
        if self._erp_sky_bg is not None and tuple(self._erp_sky_bg.shape) == target_shape:
            self._register_erp_sky_bg_optimizer_group()
            return self._erp_sky_bg
        if getattr(cam, "original_image", None) is None:
            return None
        init_lr = self._init_erp_sky_background_from_camera(cam, H, W)
        self._erp_sky_bg = nn.Parameter(init_lr.requires_grad_(True))
        self._register_erp_sky_bg_optimizer_group()
        return self._erp_sky_bg

    def get_erp_sky_background(self, cam):
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        if bool(tr_cfg.get("enable_neural_sky_bg", False)):
            return self.get_neural_sky_background(cam)
        bg_lr = self.ensure_erp_sky_background(cam)
        if bg_lr is None:
            return None
        H = int(getattr(cam, "image_height", 0))
        W = int(getattr(cam, "image_width", 0))
        bg = F.interpolate(
            bg_lr.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        return bg.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Neural sky background (MLP over ERP ray direction).
    # Used in place of the 2D ``_erp_sky_bg`` texture when
    # ``Training.enable_neural_sky_bg`` is True. See gaussian_splatting/
    # scene/neural_sky.py for the model definition.
    # ------------------------------------------------------------------
    def _register_neural_sky_optimizer_group(self):
        if self.optimizer is None or self._neural_sky_bg is None:
            return
        params = list(self._neural_sky_bg.parameters())
        if not params:
            return
        for group in self.optimizer.param_groups:
            if group.get("name") == "neural_sky_bg":
                group["params"] = params
                return
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        lr = float(
            tr_cfg.get(
                "neural_sky_lr",
                tr_cfg.get(
                    "erp_sky_bg_lr",
                    self.config.get("opt_params", {}).get("feature_lr", 0.0025),
                ),
            )
        )
        self.optimizer.add_param_group(
            {"params": params, "lr": lr, "name": "neural_sky_bg"}
        )

    def ensure_neural_sky_background(self, cam):
        tr_cfg = self.config.get("Training", {}) if self.config else {}
        if not bool(tr_cfg.get("enable_neural_sky_bg", False)):
            return None
        H = int(getattr(cam, "image_height", 0))
        W = int(getattr(cam, "image_width", 0))
        if H <= 0 or W <= 0:
            return None
        if self._neural_sky_bg is None:
            from backend.legacy_360gs.gaussian_splatting.scene.neural_sky import NeuralSkyMLP

            hidden_dim = int(tr_cfg.get("neural_sky_hidden_dim", 64))
            n_layers = int(tr_cfg.get("neural_sky_n_layers", 3))
            n_freq = int(tr_cfg.get("neural_sky_n_freq", 6))
            mlp = NeuralSkyMLP(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_freq=n_freq,
            ).to("cuda")
            self._neural_sky_bg = mlp
        self._register_neural_sky_optimizer_group()
        return self._neural_sky_bg

    def get_neural_sky_background(self, cam):
        mlp = self.ensure_neural_sky_background(cam)
        if mlp is None:
            return None
        H = int(getattr(cam, "image_height", 0))
        W = int(getattr(cam, "image_width", 0))
        device = torch.device("cuda")
        dtype = torch.float32
        if hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0:
            device = self._xyz.device
            dtype = self._xyz.dtype
        return mlp(H, W, device=device, dtype=dtype).clamp(0.0, 1.0)

    def _create_fibonacci_sky_band(
        self,
        cam,
        radius: float = 300.0,
        n_samples: int = 4096,
        elev_min_deg: float = 0.0,
        elev_max_deg: float = 90.0,
        sky_rgb: Optional[np.ndarray] = None,
        azimuth_offset_deg: float = 0.0,
    ):
        """
        Generate a Fibonacci-sampled upper hemisphere of points at `radius`
        metres from the camera centre in world space.  Used to model sky /
        distant background so that far-field ERP pixels have Gaussian coverage.

        Args:
            sky_rgb: optional (3,) float RGB in [0,1]; default legacy blue-grey.

        Returns (pts_world, rgb) both as float64 numpy arrays of shape (N, 3).
        """
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        i = np.arange(n_samples, dtype=np.float64)
        elev_min = np.deg2rad(float(elev_min_deg))
        elev_max = np.deg2rad(float(elev_max_deg))
        sin_min = np.sin(elev_min)
        sin_max = np.sin(elev_max)
        sin_elev = sin_min + (sin_max - sin_min) * (i / max(n_samples - 1, 1))
        cos_elev = np.sqrt(1.0 - sin_elev ** 2)
        azimuth = 2.0 * np.pi * i / golden + np.deg2rad(float(azimuth_offset_deg))

        dx = cos_elev * np.sin(azimuth)
        dy = -sin_elev
        dz = cos_elev * np.cos(azimuth)

        pts_cam = np.stack([dx, dy, dz], axis=-1) * radius  # (N, 3)

        # c2w: P_world = R^T (P_cam - T)
        R_np = cam.R.float().cpu().numpy().astype(np.float64)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)
        pts_world = (R_np.T @ pts_cam.T).T - (R_np.T @ T_np)

        if sky_rgb is None:
            sky_color = self._FALLBACK_SKY_RGB
        else:
            sky_color = np.asarray(sky_rgb, dtype=np.float64).reshape(3)
            sky_color = np.clip(sky_color, 0.0, 1.0)
        rgb = np.tile(sky_color, (n_samples, 1))

        return pts_world, rgb

    def _create_multi_shell_sky_band(
        self,
        cam,
        radius: float,
        total_samples: int,
        elev_min_deg: float,
        elev_max_deg: float,
        sky_rgb: np.ndarray,
        shells: int = 1,
        radius_jitter: float = 0.0,
        azimuth_step_deg: float = 23.0,
    ):
        shells = max(1, int(shells))
        total_samples = max(1, int(total_samples))
        sample_splits = np.full((shells,), total_samples // shells, dtype=np.int32)
        sample_splits[: total_samples % shells] += 1
        pts_list = []
        rgb_list = []
        center = 0.5 * (shells - 1)
        for shell_idx, shell_samples in enumerate(sample_splits.tolist()):
            if shell_samples <= 0:
                continue
            shell_radius = float(radius + (shell_idx - center) * radius_jitter)
            shell_radius = max(10.0, shell_radius)
            shell_pts, shell_rgb = self._create_fibonacci_sky_band(
                cam=cam,
                radius=shell_radius,
                n_samples=int(shell_samples),
                elev_min_deg=elev_min_deg,
                elev_max_deg=elev_max_deg,
                sky_rgb=sky_rgb,
                azimuth_offset_deg=float(shell_idx * azimuth_step_deg),
            )
            pts_list.append(shell_pts)
            rgb_list.append(shell_rgb)
        return np.concatenate(pts_list, axis=0), np.concatenate(rgb_list, axis=0)

    def clamp_scaling_ratios(self, max_ratio: float):
        if self._scaling.numel() == 0:
            return
        scaling = self.get_scaling
        s_min = scaling.min(dim=1, keepdim=True).values.clamp(min=1e-6)
        max_allowed = s_min * max_ratio
        scaling_clamped = torch.minimum(scaling, max_allowed.expand_as(scaling))
        self._scaling.data.copy_(self.scaling_inverse_activation(scaling_clamped))

    def clamp_max_scaling(self, max_abs_scale: float):
        """Prevent any single scaling axis from exceeding an absolute world-unit limit."""
        if self._scaling.numel() == 0 or max_abs_scale <= 0:
            return
        scaling = self.get_scaling
        clamped = torch.clamp(scaling, max=max_abs_scale)
        self._scaling.data.copy_(self.scaling_inverse_activation(clamped))

    def prune_needles(self, max_ratio: float = 20.0, protect_sky: bool = True):
        """Remove needle-like Gaussians (one dominant axis) using s_max/s_median metric.
        Pancake Gaussians (two large axes, one small) are intentionally preserved."""
        if self._scaling.numel() == 0:
            return 0
        s = self.get_scaling
        s_sorted, _ = s.sort(dim=1, descending=True)  # (N, 3): [max, mid, min]
        s_mid = s_sorted[:, 1].clamp(min=1e-6)
        needle_ratio = s_sorted[:, 0] / s_mid
        region_ratio = torch.full_like(needle_ratio, float(max_ratio))
        if self._region_tag.shape[0] == needle_ratio.shape[0]:
            region_tag = self._region_tag.to(device=needle_ratio.device)
            region_ratio = torch.where(
                region_tag == self.REGION_TAG_HORIZON_BG,
                region_ratio * 1.5,
                region_ratio,
            )
            region_ratio = torch.where(
                region_tag == self.REGION_TAG_BOTTOM_POLE_GROUND,
                region_ratio * 1.25,
                region_ratio,
            )
            region_ratio = torch.where(
                region_tag == self.REGION_TAG_UPPER_SKY,
                torch.full_like(region_ratio, float("inf")),
                region_ratio,
            )
            region_ratio = torch.where(
                region_tag == self.REGION_TAG_POLAR_CAP_SKY,
                torch.full_like(region_ratio, float("inf")),
                region_ratio,
            )
        needle_mask = needle_ratio > region_ratio
        if protect_sky and self._is_sky.shape[0] == needle_mask.shape[0]:
            sky = self._is_sky.to(needle_mask.device)
            needle_mask = needle_mask & ~sky
        n_pruned = int(needle_mask.sum().item())
        if n_pruned > 0:
            self.prune_points(needle_mask)
        return n_pruned

    def get_sky_isotropic_loss(self):
        """Encourage sky Gaussians to be roughly isotropic (equal scales)."""
        if self._scaling.numel() == 0:
            return self._scaling.new_zeros(())
        sky = self._is_sky.to(self._scaling.device)
        if not sky.any():
            return self._scaling.new_zeros(())
        s = self.get_scaling[sky]  # (N_sky, 3)
        s_mean = s.mean(dim=1, keepdim=True)
        return ((s - s_mean) ** 2).mean()

    # Process input camera info and image data, then call create_pcd_from_image_and_depth to generate point cloud
    def create_pcd_from_image(
        self,
        cam_info,
        init=False,
        scale=2.0,
        depthmap=None,
        anchor_submap=-1,
        birth_frame=None,
    ):
        cam = cam_info

        # For panoramic (ERP) cameras with a provided depthmap, use spherical
        # backprojection instead of the pinhole O3D pipeline.  The body
        # PanoramaCamera has face-level intrinsics (90掳 FoV) stored in fx/fy
        # which are meaningless for a full ERP image.
        if depthmap is not None:
            from backend.legacy_360gs.utils.camera_utils import PanoramaCamera
            if isinstance(cam, PanoramaCamera):
                return self._create_pcd_from_erp_depth(
                    cam,
                    depthmap,
                    init=init,
                    anchor_submap=anchor_submap,
                    birth_frame=birth_frame,
                )

        image_ab = cam.original_image.clamp(0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def _create_pcd_from_erp_depth(
        self,
        cam,
        depthmap: np.ndarray,
        init: bool = False,
        anchor_submap: int = -1,
        birth_frame: Optional[int] = None,
    ):
        """
        Build a Gaussian point cloud from an ERP panoramic depth map using
        spherical backprojection.  Bearings match ``utils.erp_geometry`` /
        spherical RANSAC (pixel centers u=c+0.5, v=r+0.5).

        Args:
            cam:        PanoramaCamera with current R, T (w2c) pose.
            depthmap:   (H, W) float32 ERP depth map in metres.
            init:       True for the first keyframe (uses pcd_downsample_init).

        Insertion consumes all valid non-sky depth pixels directly.
        """
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]

        H, W = depthmap.shape

        # Pixel-center (u,v) -> unit bearing; same closed form as utils.erp_geometry
        # and spherical RANSAC (lam, phi from u/W, v/H).
        dx, dy, dz = erp_dense_pixel_center_bearings(H, W)

        depth_valid_max = float(
            self.config.get("Training", {}).get(
                "dap_depth_max_valid",
                self.config.get("Training", {}).get("ransac", {}).get("depth_max", 80.0),
            )
        )
        sky_mask = getattr(cam, "erp_sky_mask", None)
        if sky_mask is not None and sky_mask.shape != depthmap.shape:
            sky_mask = None
        region_masks = getattr(cam, "erp_region_masks", None) or {}
        bottom_pole_mask = region_masks.get("bottom_pole", None)
        if bottom_pole_mask is not None:
            if isinstance(bottom_pole_mask, torch.Tensor):
                bottom_pole_mask = bottom_pole_mask.cpu().numpy().astype(bool)
            else:
                bottom_pole_mask = np.asarray(bottom_pole_mask, dtype=bool)
            if bottom_pole_mask.ndim == 3:
                bottom_pole_mask = bottom_pole_mask[0]
            if bottom_pole_mask.shape != depthmap.shape:
                bottom_pole_mask = None

        # Valid geometric depth mask: only keep finite non-sky depths inside the
        # configured DAP valid range.
        depth_valid = (depthmap > 0.01) & (depthmap < depth_valid_max)
        if sky_mask is not None:
            depth_valid = depth_valid & (~sky_mask)

        # 3D points in camera space. Depth is radial ERP depth in metres.
        pts_cam = np.stack(
            [dx * depthmap, dy * depthmap, dz * depthmap], axis=-1
        )  # (H, W, 3)
        pts_cam_valid = pts_cam.reshape(-1, 3)[depth_valid.reshape(-1)]  # (N, 3)

        # Transform to world: P_world = R^T (P_cam 鈭?T)  [w2c: P_cam = R路P_world + T]
        R_np = cam.R.float().cpu().numpy().astype(np.float64)   # (3, 3)
        T_np = cam.T.float().cpu().numpy().astype(np.float64)   # (3,)
        pts_world = (R_np.T @ pts_cam_valid.T).T - (R_np.T @ T_np)  # (N, 3)

        # RGB at valid pixels
        image_ab = cam.original_image.clamp(0.0, 1.0)
        rgb_np = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb_valid = rgb_np.reshape(-1, 3)[depth_valid.reshape(-1)].astype(np.float64) / 255.0
        region_tag_valid = np.full((pts_cam_valid.shape[0],), self.REGION_TAG_DEFAULT, dtype=np.int8)
        if bottom_pole_mask is not None:
            region_tag_valid = np.where(
                bottom_pole_mask.reshape(-1)[depth_valid.reshape(-1)],
                self.REGION_TAG_BOTTOM_POLE_GROUND,
                region_tag_valid,
            )

        idx = None  # subsample indices for layer labels (aligned with pts_world rows)
        if pts_world.shape[0] == 0:
            pts_world = np.zeros((1, 3), dtype=np.float64)
            rgb_valid = np.zeros((1, 3), dtype=np.float64)
            region_tag_valid = np.full((1,), self.REGION_TAG_DEFAULT, dtype=np.int8)
        elif downsample_factor > 1:
            n_keep = max(1, pts_world.shape[0] // downsample_factor)
            idx = np.random.choice(pts_world.shape[0], n_keep, replace=False)
            pts_world = pts_world[idx]
            rgb_valid = rgb_valid[idx]
            region_tag_valid = region_tag_valid[idx]
        else:
            idx = None  # no random subsample; all valid points in order

        n_sky_added = 0
        sky_region_tags = np.zeros((0,), dtype=np.int8)
        add_sky_support = init
        sky_support_interval = int(
            self.config.get("Training", {}).get("sky_support_kf_interval", 0)
        )
        if (not init) and sky_support_interval > 0 and (cam.uid % sky_support_interval == 0):
            add_sky_support = True

        if add_sky_support:
            sky_radius = float(
                self.config.get("Dataset", {}).get("sky_hemisphere_radius", 300.0)
            )
            tr_cfg = self.config.get("Training", {})
            sky_mode = str(tr_cfg.get("sky_support_mode", "single_band")).lower()
            sky_pts_list = []
            sky_rgb_list = []
            if sky_mode in {"stratified_dual_band", "stratified_triple_band"}:
                upper_samples = int(
                    tr_cfg.get(
                        "sky_support_upper_samples",
                        self.config.get("Dataset", {}).get("sky_hemisphere_samples", 4096) // 2,
                    )
                )
                horizon_samples = int(tr_cfg.get("sky_support_horizon_samples", 256))
                top_cap_samples = int(tr_cfg.get("sky_support_top_cap_samples", 512))
                if not init:
                    refresh_cap = int(
                        tr_cfg.get(
                            "sky_support_refresh_max_new",
                            max(horizon_samples, top_cap_samples),
                        )
                    )
                    horizon_samples = min(horizon_samples, refresh_cap)
                    top_cap_samples = min(top_cap_samples, refresh_cap)
                    upper_samples = min(upper_samples, max(refresh_cap, refresh_cap // 2))
                horizon_band_deg = float(tr_cfg.get("sky_horizon_band_deg", 12.0))
                top_cap_deg = float(tr_cfg.get("sky_top_cap_deg", 70.0))
                upper_shells = int(tr_cfg.get("sky_support_upper_shells", 2))
                upper_radius_jitter = float(
                    tr_cfg.get("sky_support_upper_radius_jitter", 18.0)
                )
                upper_azimuth_step = float(
                    tr_cfg.get("sky_support_upper_azimuth_step_deg", 23.0)
                )
                upper_rgb = self._estimate_sky_band_rgb_erp(cam, depthmap, band="upper")
                horizon_rgb = self._estimate_sky_band_rgb_erp(cam, depthmap, band="horizon")
                upper_max_deg = 90.0
                upper_min_deg = horizon_band_deg
                if sky_mode == "stratified_triple_band":
                    upper_max_deg = max(upper_min_deg + 1e-3, top_cap_deg)
                    polar_rgb = self._estimate_sky_band_rgb_erp(
                        cam, depthmap, band="polar_cap"
                    )
                    polar_pts, polar_cols = self._create_fibonacci_sky_band(
                        cam=cam,
                        radius=sky_radius,
                        n_samples=max(1, top_cap_samples),
                        elev_min_deg=top_cap_deg,
                        elev_max_deg=90.0,
                        sky_rgb=polar_rgb,
                    )
                    sky_pts_list.append(polar_pts)
                    sky_rgb_list.append(polar_cols)
                upper_pts, upper_cols = self._create_multi_shell_sky_band(
                    cam=cam,
                    radius=sky_radius,
                    total_samples=max(1, upper_samples),
                    elev_min_deg=upper_min_deg,
                    elev_max_deg=upper_max_deg,
                    sky_rgb=upper_rgb,
                    shells=upper_shells,
                    radius_jitter=upper_radius_jitter,
                    azimuth_step_deg=upper_azimuth_step,
                )
                horizon_pts, horizon_cols = self._create_fibonacci_sky_band(
                    cam=cam,
                    radius=sky_radius,
                    n_samples=max(1, horizon_samples),
                    elev_min_deg=0.0,
                    elev_max_deg=horizon_band_deg,
                    sky_rgb=horizon_rgb,
                )
                sky_pts_list.extend([upper_pts, horizon_pts])
                sky_rgb_list.extend([upper_cols, horizon_cols])
                region_tag_chunks = []
                if sky_mode == "stratified_triple_band":
                    region_tag_chunks.append(
                        np.full(
                            (polar_pts.shape[0],),
                            self.REGION_TAG_POLAR_CAP_SKY,
                            dtype=np.int8,
                        )
                    )
                region_tag_chunks.extend(
                    [
                        np.full((upper_pts.shape[0],), self.REGION_TAG_UPPER_SKY, dtype=np.int8),
                        np.full((horizon_pts.shape[0],), self.REGION_TAG_HORIZON_BG, dtype=np.int8),
                    ]
                )
                sky_region_tags = np.concatenate(region_tag_chunks, axis=0)
            else:
                sky_samples = int(
                    self.config.get("Dataset", {}).get(
                        "sky_hemisphere_samples",
                        4096,
                    )
                )
                if not init:
                    sky_samples = int(tr_cfg.get("sky_support_samples", 512))
                sky_rgb_vec = self._estimate_sky_rgb_erp(cam, depthmap)
                sky_pts, sky_rgb = self._create_fibonacci_sky_band(
                    cam=cam,
                    radius=sky_radius,
                    n_samples=sky_samples,
                    elev_min_deg=0.0,
                    elev_max_deg=90.0,
                    sky_rgb=sky_rgb_vec,
                )
                sky_pts_list.append(sky_pts)
                sky_rgb_list.append(sky_rgb)
                sky_region_tags = np.full((sky_pts.shape[0],), self.REGION_TAG_UPPER_SKY, dtype=np.int8)

            sky_pts = np.concatenate(sky_pts_list, axis=0)
            sky_rgb = np.concatenate(sky_rgb_list, axis=0)
            pts_world = np.concatenate([pts_world, sky_pts], axis=0)
            rgb_valid = np.concatenate([rgb_valid, sky_rgb], axis=0)
            region_tag_valid = np.concatenate([region_tag_valid, sky_region_tags], axis=0)
            n_sky_added = sky_pts.shape[0]

        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                med = float(np.median(depthmap[depth_valid])) if depth_valid.any() else 1.0
                point_size = min(0.05, point_size * med)

        pcd = BasicPointCloud(
            points=pts_world,
            colors=rgb_valid,
            normals=np.zeros((pts_world.shape[0], 3)),
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(pts_world).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(rgb_valid).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32, device="cuda",
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        if n_sky_added > 0:
            sky_scale = float(
                self.config.get("Training", {}).get("sky_support_scale", 0.2)
            )
            scales[-n_sky_added:] = np.log(sky_scale)
            if region_tag_valid.shape[0] == scales.shape[0]:
                tr_cfg = self.config.get("Training", {})
                upper_mult = float(tr_cfg.get("sky_support_upper_scale_mult", 1.35))
                horizon_mult = float(tr_cfg.get("sky_support_horizon_scale_mult", 1.0))
                polar_mult = float(tr_cfg.get("sky_support_polar_scale_mult", 1.15))
                region_tag_t = torch.from_numpy(region_tag_valid).to(device=scales.device)
                if upper_mult > 0 and upper_mult != 1.0:
                    upper_mask = region_tag_t == self.REGION_TAG_UPPER_SKY
                    if upper_mask.any():
                        scales[upper_mask] = scales[upper_mask] + np.log(upper_mult)
                if horizon_mult > 0 and horizon_mult != 1.0:
                    horizon_mask = region_tag_t == self.REGION_TAG_HORIZON_BG
                    if horizon_mask.any():
                        scales[horizon_mask] = scales[horizon_mask] + np.log(horizon_mult)
                if polar_mult > 0 and polar_mult != 1.0:
                    polar_mask_t = region_tag_t == self.REGION_TAG_POLAR_CAP_SKY
                    if polar_mask_t.any():
                        scales[polar_mask_t] = scales[polar_mask_t] + np.log(polar_mult)
        bottom_scale_mult = float(
            self.config.get("Training", {}).get("bottom_pole_scale_mult", 1.0)
        )
        if bottom_scale_mult > 0 and bottom_scale_mult != 1.0 and region_tag_valid.shape[0] == scales.shape[0]:
            bottom_mask_t = torch.from_numpy(
                region_tag_valid == self.REGION_TAG_BOTTOM_POLE_GROUND
            ).to(device=scales.device, dtype=torch.bool)
            if bottom_mask_t.any():
                scales[bottom_mask_t] = scales[bottom_mask_t] + np.log(bottom_scale_mult)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(
            0.5 * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"
            )
        )
        self._n_sky_pending = n_sky_added  # stash for extend_from_pcd

        # ----- Layer assignment (enable_layered_map only) -----
        enable_layered = bool(
            self.config.get("Training", {}).get("enable_layered_map", False)
        )
        if enable_layered:
            near_max = float(self.config.get("Training", {}).get("layer_near_max_depth", 80.0))
            far_max  = float(self.config.get("Training", {}).get("layer_far_max_depth", 100.0))
            n_geom = pts_world.shape[0] - n_sky_added
            if n_geom > 0:
                # depths at valid pixels (same order as pts_cam_valid before subsample).
                # Must flatten depthmap first: depthmap[bool_flat] is ambiguous for 2D (512脳1024).
                flat_d = np.asarray(depthmap).reshape(-1)
                flat_m = depth_valid.reshape(-1)
                valid_depths = flat_d[flat_m]
                if idx is not None:
                    valid_depths = valid_depths[idx]
                geom_layers = np.zeros(n_geom, dtype=np.int8)
                geom_layers[(valid_depths >= near_max) & (valid_depths < far_max)] = 1
                geom_layers[valid_depths >= far_max] = 2
            else:
                geom_layers = np.zeros(0, dtype=np.int8)
            sky_layers = np.full(n_sky_added, 2, dtype=np.int8)
            all_layers = np.concatenate([geom_layers, sky_layers])
            self._layer_pending = torch.from_numpy(all_layers)  # CPU int8
        else:
            self._layer_pending = None

        self._region_tag_pending = torch.from_numpy(
            np.asarray(region_tag_valid, dtype=np.int8)
        )
        if birth_frame is None:
            birth_frame = int(cam.uid)
        n_total = pts_world.shape[0]
        self._anchor_kf_pending = torch.full((n_total,), int(cam.uid), dtype=torch.int32)
        self._anchor_submap_pending = torch.full(
            (n_total,), int(anchor_submap), dtype=torch.int32
        )
        self._birth_frame_pending = torch.full(
            (n_total,), int(birth_frame), dtype=torch.int32
        )

        return fused_point_cloud, features, scales, rots, opacities

    # Create point cloud from RGB and depth, apply random downsampling, store as BasicPointCloud, and complete 3DGS initialization
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()     
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())   
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(         
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities
    
    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def apply_sky_scaling_grad_multiplier(self, mult: float) -> None:
        """Scale ``_scaling.grad`` rows for sky Gaussians only (equivalent to higher LR)."""
        if mult is None or mult <= 1.0:
            return
        g = self._scaling.grad
        if g is None:
            return
        if self._is_sky.shape[0] != g.shape[0]:
            return
        sky = self._is_sky.to(device=g.device, dtype=torch.bool)
        g[sky] *= mult

    def apply_sky_all_grad_multiplier(self, mult: float) -> None:
        """Scale gradients of ALL parameters for sky Gaussians (equivalent to higher LR).

        Affects: xyz, features_dc, features_rest, opacity, scaling, rotation.
        Call after loss.backward() and before optimizer.step().

        NOTE: Adam normalises gradients by the RMS of past gradients, so this
        multiplier is cancelled out in steady state.  Pair with
        ``reset_sky_adam_state()`` (called after optimizer.step()) to keep the
        Adam second-moment buffer small so the multiplier remains effective.
        """
        if mult is None or mult <= 1.0:
            return
        n = self._is_sky.shape[0]
        for param in (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ):
            g = param.grad
            if g is None or g.shape[0] != n:
                continue
            sky = self._is_sky.to(device=g.device, dtype=torch.bool)
            g[sky] *= mult

    def reset_sky_adam_state(self) -> None:
        """Zero Adam first/second moment buffers for sky Gaussians.

        Adam normalises every gradient by the RMS of past gradients, so
        multiplying the gradient by K is cancelled by an equal K in
        sqrt(exp_avg_sq).  However, when the state is reset to zero the
        bias-correction factors dominate: the first step after reset has
        magnitude 鈮?lr 脳 (1-尾鈧?/鈭?1-尾鈧? 鈮?3脳 the steady-state step.
        This gives a genuine (though moderate) boost that decays back to
        normal as the state re-accumulates over the next few iterations.

        Call BEFORE ``optimizer.step()`` at a configurable interval
        (config key ``sky_adam_reset_interval``).  Calling every step (=1)
        is safe but turns Adam into sign-SGD for sky Gaussians; calling
        every N>1 steps gives periodic boost bursts.
        """
        if self.optimizer is None:
            return
        n = self._is_sky.shape[0]
        if n == 0:
            return
        sky_cpu = self._is_sky  # bool, CPU
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            state = self.optimizer.state.get(param, None)
            if state is None:
                continue
            if param.shape[0] != n:
                continue
            sky_dev = sky_cpu.to(device=param.device, dtype=torch.bool)
            if "exp_avg" in state:
                state["exp_avg"][sky_dev] = 0.0
            if "exp_avg_sq" in state:
                state["exp_avg_sq"][sky_dev] = 0.0

    # ------------------------------------------------------------------
    # True per-element LR boost for sky Gaussians via post-step scaling
    # ------------------------------------------------------------------

    def save_sky_params(self) -> dict:
        """Snapshot current parameter values for sky Gaussians.

        Returns a dict mapping ``id(param) -> (param_ref, sky_data_clone)``
        for every optimizer param group whose tensor has the right length.

        Call this AFTER any densification / opacity-reset that replaces the
        underlying parameter tensors (so the ids are stable), and BEFORE
        ``optimizer.step()``.
        """
        n = self._is_sky.shape[0]
        if n == 0 or self.optimizer is None:
            return {}
        sky_cpu = self._is_sky
        saved: dict = {}
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            if param.shape[0] != n:
                continue
            sky_dev = sky_cpu.to(device=param.device, dtype=torch.bool)
            saved[id(param)] = (param, param.data[sky_dev].clone())
        return saved

    def amplify_sky_adam_update(self, mult: float, old_sky_params: dict) -> None:
        """Multiply the Adam update that was just applied to sky Gaussians by *mult*.

        Adam normalises every gradient by the running RMS of past gradients,
        so gradient-scaling (``apply_sky_all_grad_multiplier``) is cancelled
        internally.  This function instead operates on the *actual update* that
        Adam already committed:

            螖  = new_param  鈭?old_param        (computed by Adam normally)
            new_param_sky 鈫?old_param + mult 脳 螖   (re-scaled in-place)

        This is equivalent to running Adam with a learning-rate of
        ``mult 脳 base_lr`` for sky Gaussians only, without needing separate
        parameter groups.

        Typical usage in the training loop::

            old_sky = gaussians.save_sky_params()   # snapshot before step
            optimizer.step()
            gaussians.amplify_sky_adam_update(mult, old_sky)  # re-scale after
            optimizer.zero_grad(set_to_none=True)

        Args:
            mult: Effective LR multiplier (1.0 = no change).
            old_sky_params: dict returned by ``save_sky_params()``.
        """
        if mult is None or mult <= 1.0 or not old_sky_params:
            return
        n = self._is_sky.shape[0]
        if n == 0:
            return
        sky_cpu = self._is_sky
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            if id(param) not in old_sky_params or param.shape[0] != n:
                continue
            _, old_data = old_sky_params[id(param)]
            sky_dev = sky_cpu.to(device=param.device, dtype=torch.bool)
            delta = param.data[sky_dev] - old_data      # Adam's update for sky
            param.data[sky_dev] = old_data + mult * delta  # amplify

    def extend_from_pcd(
        self,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        kf_id,
        anchor_kf=None,
        anchor_submap=None,
        birth_frame=None,
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_anchor_kf=anchor_kf,
            new_anchor_submap=anchor_submap,
            new_birth_frame=birth_frame,
        )
        # densification_postfix already appended n_new False entries to _is_sky.
        # Just flip the trailing n_sky entries to True for actual sky points.
        # (_n_sky_pending is set by _create_pcd_from_erp_depth; 0 for all other paths.)
        n_sky = getattr(self, "_n_sky_pending", 0)
        self._n_sky_pending = 0
        if n_sky > 0:
            self._is_sky[-n_sky:] = True

        # Apply pending layer assignments computed in _create_pcd_from_erp_depth.
        layer_pending = getattr(self, "_layer_pending", None)
        self._layer_pending = None
        if layer_pending is not None and layer_pending.shape[0] > 0:
            n_new = layer_pending.shape[0]
            if self._layer.shape[0] >= n_new:
                self._layer[-n_new:] = layer_pending
        region_tag_pending = getattr(self, "_region_tag_pending", None)
        self._region_tag_pending = None
        if region_tag_pending is not None and region_tag_pending.shape[0] > 0:
            n_new = region_tag_pending.shape[0]
            if self._region_tag.shape[0] >= n_new:
                self._region_tag[-n_new:] = region_tag_pending.to(dtype=torch.int8)
        anchor_kf_pending = getattr(self, "_anchor_kf_pending", None)
        self._anchor_kf_pending = None
        if anchor_kf_pending is not None and anchor_kf_pending.shape[0] > 0:
            n_new = anchor_kf_pending.shape[0]
            if self._anchor_kf.shape[0] >= n_new:
                self._anchor_kf[-n_new:] = anchor_kf_pending.to(dtype=torch.int32)
        anchor_submap_pending = getattr(self, "_anchor_submap_pending", None)
        self._anchor_submap_pending = None
        if anchor_submap_pending is not None and anchor_submap_pending.shape[0] > 0:
            n_new = anchor_submap_pending.shape[0]
            if self._anchor_submap.shape[0] >= n_new:
                self._anchor_submap[-n_new:] = anchor_submap_pending.to(dtype=torch.int32)
        birth_frame_pending = getattr(self, "_birth_frame_pending", None)
        self._birth_frame_pending = None
        if birth_frame_pending is not None and birth_frame_pending.shape[0] > 0:
            n_new = birth_frame_pending.shape[0]
            if self._birth_frame.shape[0] >= n_new:
                self._birth_frame[-n_new:] = birth_frame_pending.to(dtype=torch.int32)

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None,
        anchor_submap=-1, birth_frame=None,
    ):
        """Extend the Gaussian map from a new keyframe.

        The ERP insertion path back-projects all valid non-sky depth pixels.
        """
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(
                cam_info, init, scale=scale, depthmap=depthmap,
                anchor_submap=anchor_submap,
                birth_frame=birth_frame,
            )
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        if bool(self.config.get("Training", {}).get("enable_erp_sky_background", False)) and self._erp_sky_bg is not None:
            l.append(
                {
                    "params": [self._erp_sky_bg],
                    "lr": float(
                        self.config.get("Training", {}).get(
                            "erp_sky_bg_lr", training_args.feature_lr
                        )
                    ),
                    "name": "erp_sky_bg",
                }
            )
        # Neural sky background MLP (Splatfacto-W style direction-only MLP).
        # When enabled, this replaces the 2D ``_erp_sky_bg`` texture path inside
        # ``get_erp_sky_background``. Lazy-init on first KF render is also OK,
        # but adding here gives the optimizer the param group up front.
        if (
            bool(self.config.get("Training", {}).get("enable_neural_sky_bg", False))
            and self._neural_sky_bg is not None
        ):
            tr_cfg = self.config.get("Training", {})
            l.append(
                {
                    "params": list(self._neural_sky_bg.parameters()),
                    "lr": float(
                        tr_cfg.get(
                            "neural_sky_lr",
                            tr_cfg.get("erp_sky_bg_lr", training_args.feature_lr),
                        )
                    ),
                    "name": "neural_sky_bg",
                }
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonsky(self):
        """Reset opacity to 0.01 for non-sky Gaussians only, preserving sky Gaussian opacities."""
        opacities_new = self._opacity.clone().detach()
        if self._is_sky.shape[0] == opacities_new.shape[0]:
            sky_mask = self._is_sky.to(opacities_new.device)
            opacities_new[~sky_mask] = inverse_sigmoid(
                torch.ones_like(opacities_new[~sky_mask]) * 0.01
            )
        else:
            opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters, protected_mask=None
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        # Preserve sky Gaussian opacities 鈥?they should not be forcibly reset.
        if self._is_sky.shape[0] == opacities_new.shape[0]:
            sky_mask = self._is_sky.to(opacities_new.device)
            opacities_new[sky_mask] = self.get_opacity[sky_mask]
        # Preserve protected anchors (e.g. early-KF anchors outside current
        # mapping window) so their opacity is not hard-overwritten + Adam state
        # is not cleared by the optimizer replacement step below.
        if protected_mask is not None and protected_mask.shape[0] == opacities_new.shape[0]:
            pm = protected_mask.to(opacities_new.device).bool()
            opacities_new[pm] = self.get_opacity[pm]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        n = self._xyz.shape[0]
        self._is_sky = torch.zeros(n, dtype=torch.bool)
        self._layer = torch.zeros(n, dtype=torch.int8)
        self._region_tag = torch.zeros(n, dtype=torch.int8)
        self._anchor_kf = torch.full((n,), -1, dtype=torch.int32)
        self._anchor_submap = torch.full((n,), -1, dtype=torch.int32)
        self._birth_frame = torch.full((n,), -1, dtype=torch.int32)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        self._is_sky = self._is_sky[valid_points_mask.cpu()]
        if self._layer.shape[0] == valid_points_mask.shape[0]:
            self._layer = self._layer[valid_points_mask.cpu()]
        if self._region_tag.shape[0] == valid_points_mask.shape[0]:
            self._region_tag = self._region_tag[valid_points_mask.cpu()]
        if self._anchor_kf.shape[0] == valid_points_mask.shape[0]:
            self._anchor_kf = self._anchor_kf[valid_points_mask.cpu()]
        if self._anchor_submap.shape[0] == valid_points_mask.shape[0]:
            self._anchor_submap = self._anchor_submap[valid_points_mask.cpu()]
        if self._birth_frame.shape[0] == valid_points_mask.shape[0]:
            self._birth_frame = self._birth_frame[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                # Skip groups whose params are not per-Gaussian (e.g. the
                # learnable sky background / neural sky MLP). They live
                # outside the (N, ...) tensor world and don't need to grow
                # when new Gaussians are appended.
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        sky_clone_mask=None,
        new_region_tag=None,
        new_anchor_kf=None,
        new_anchor_submap=None,
        new_birth_frame=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        n_new = new_xyz.shape[0]
        new_sky_flags = torch.zeros(n_new, dtype=torch.bool)
        if sky_clone_mask is not None:
            new_sky_flags[: sky_clone_mask.shape[0]] = sky_clone_mask
        self._is_sky = torch.cat([self._is_sky, new_sky_flags])

        # _layer: append zeros for new Gaussians; caller may overwrite afterwards
        # (extend_from_pcd via _layer_pending, or split/clone propagation).
        new_layer_flags = torch.zeros(n_new, dtype=torch.int8)
        self._layer = torch.cat([self._layer, new_layer_flags])
        if new_region_tag is None:
            new_region_tag = torch.zeros(n_new, dtype=torch.int8)
        self._region_tag = torch.cat(
            [self._region_tag, new_region_tag.detach().cpu().to(dtype=torch.int8)]
        )
        if new_anchor_kf is None:
            fill_value = int(new_kf_ids[0].item()) if new_kf_ids is not None and n_new > 0 else -1
            new_anchor_kf = torch.full((n_new,), fill_value, dtype=torch.int32)
        if new_anchor_submap is None:
            new_anchor_submap = torch.full((n_new,), -1, dtype=torch.int32)
        if new_birth_frame is None:
            fill_value = int(new_kf_ids[0].item()) if new_kf_ids is not None and n_new > 0 else -1
            new_birth_frame = torch.full((n_new,), fill_value, dtype=torch.int32)
        self._anchor_kf = torch.cat(
            [self._anchor_kf, new_anchor_kf.detach().cpu().to(dtype=torch.int32)]
        )
        self._anchor_submap = torch.cat(
            [self._anchor_submap, new_anchor_submap.detach().cpu().to(dtype=torch.int32)]
        )
        self._birth_frame = torch.cat(
            [self._birth_frame, new_birth_frame.detach().cpu().to(dtype=torch.int32)]
        )

    def _exclude_sky_from_densify_mask(self, mask: torch.Tensor, operation: str = "split") -> torch.Tensor:
        """Exclude sky Gaussians from densification mask.

        Controlled by config flags:
          split: honour ``allow_sky_split`` (default False).
                 When True, split children inherit the parent's _is_sky flag.
          clone: honour ``allow_sky_clone`` (default False).
        """
        if mask.ndim != 1 or self._is_sky.shape[0] != mask.shape[0]:
            return mask
        sky = self._is_sky.to(device=mask.device, dtype=torch.bool)
        if operation == "clone":
            if self.config.get("Training", {}).get("allow_sky_clone", False):
                return mask
        elif operation == "split":
            if self.config.get("Training", {}).get("allow_sky_split", False):
                return mask
        return torch.logical_and(mask, torch.logical_not(sky))

    def _fastgs_importance_gate(
        self, importance_score: Optional[torch.Tensor], target_len: int, device
    ) -> Optional[torch.Tensor]:
        if importance_score is None:
            return None
        score = torch.as_tensor(importance_score, device=device)
        if score.ndim > 1:
            score = score.view(-1)
        gate = torch.zeros((target_len,), device=device, dtype=torch.bool)
        n_valid = min(target_len, int(score.shape[0]))
        if n_valid <= 0:
            return gate
        thr = int(self.config.get("Training", {}).get("fastgs_importance_px_min", 5))
        gate[:n_valid] = score[:n_valid].to(dtype=torch.float32) >= float(thr)
        return gate

    def _fastgs_prune_mask(
        self, pruning_score: Optional[torch.Tensor], target_len: int, device
    ) -> torch.Tensor:
        mask = torch.zeros((target_len,), device=device, dtype=torch.bool)
        if pruning_score is None:
            return mask

        score = torch.as_tensor(pruning_score, device=device, dtype=torch.float32)
        if score.ndim > 1:
            score = score.view(-1)
        n_valid = min(target_len, int(score.shape[0]))
        if n_valid <= 0:
            return mask

        tr_cfg = self.config.get("Training", {})
        prune_thr = float(tr_cfg.get("fastgs_prune_score_thresh", 0.90))
        mask[:n_valid] = score[:n_valid] > prune_thr

        if bool(tr_cfg.get("fastgs_exclude_sky", True)) and self._is_sky.shape[0] == target_len:
            mask = mask & ~self._is_sky.to(device=device, dtype=torch.bool)

        if bool(tr_cfg.get("fastgs_exclude_high_opacity", True)):
            high_opacity_thr = float(tr_cfg.get("fastgs_high_opacity_thr", 0.80))
            mask = mask & ~(self.get_opacity.squeeze().to(device=device) > high_opacity_thr)

        if bool(tr_cfg.get("fastgs_exclude_region_protected", True)) and self._region_tag.shape[0] == target_len:
            region_tag = self._region_tag.to(device=device)
            protected = (
                (region_tag == self.REGION_TAG_UPPER_SKY)
                | (region_tag == self.REGION_TAG_POLAR_CAP_SKY)
                | (region_tag == self.REGION_TAG_HORIZON_BG)
            )
            if self.n_obs.shape[0] == target_len:
                bottom = region_tag == self.REGION_TAG_BOTTOM_POLE_GROUND
                min_bottom_obs = int(
                    tr_cfg.get("region_min_obs_before_prune_bottom_pole", 4)
                )
                protected = protected | (
                    bottom & (self.n_obs.to(device=device) < min_bottom_obs)
                )
            mask = mask & ~protected

        return mask

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        current_kf_id=None,
        init_phase: bool = False,
        importance_score: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        large_enough_mask = (
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent
        )
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, large_enough_mask)
        if self._region_tag.shape[0] == selected_pts_mask.shape[0]:
            region_tag = self._region_tag.to(device=selected_pts_mask.device)
            upper_sky = region_tag == self.REGION_TAG_UPPER_SKY
            polar_cap = region_tag == self.REGION_TAG_POLAR_CAP_SKY
            horizon_bg = region_tag == self.REGION_TAG_HORIZON_BG
            bottom_pole = region_tag == self.REGION_TAG_BOTTOM_POLE_GROUND
            if not bool(
                self.config.get("Training", {}).get("region_enable_split_upper_sky", False)
            ):
                selected_pts_mask = selected_pts_mask & ~(upper_sky | polar_cap)
            horizon_grad_mult = float(
                self.config.get("Training", {}).get("region_horizon_split_grad_mult", 1.15)
            )
            selected_pts_mask = selected_pts_mask & (
                ~horizon_bg | (padded_grad >= grad_threshold * horizon_grad_mult)
            )
            bottom_grad_mult = float(
                self.config.get("Training", {}).get("region_bottom_pole_split_grad_mult", 0.75)
            )
            boosted_bottom = (
                bottom_pole
                & large_enough_mask
                & (padded_grad >= grad_threshold * bottom_grad_mult)
            )
            selected_pts_mask = selected_pts_mask | boosted_bottom
        importance_gate = self._fastgs_importance_gate(
            importance_score, n_init_points, selected_pts_mask.device
        )
        if importance_gate is not None:
            selected_pts_mask = selected_pts_mask & importance_gate
        if exclude_mask is not None and exclude_mask.shape[0] == selected_pts_mask.shape[0]:
            selected_pts_mask = selected_pts_mask & ~exclude_mask.to(
                device=selected_pts_mask.device, dtype=torch.bool
            )
        selected_pts_mask = self._exclude_sky_from_densify_mask(selected_pts_mask)
        if not selected_pts_mask.any():
            return {"count": 0, "selected_mask": selected_pts_mask, "n_children": 0}
        n_selected = int(selected_pts_mask.sum().item())
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        split_max_ratio = float(
            self.config.get("Training", {}).get("split_child_max_ratio", 10.0)
        )
        if split_max_ratio > 1.0:
            _s = self.scaling_activation(new_scaling)
            apply_ratio_cap = torch.ones(
                (_s.shape[0],), device=_s.device, dtype=torch.bool
            )
            if init_phase and self._is_sky.shape[0] == selected_pts_mask.shape[0]:
                sky_children = self._is_sky[selected_pts_mask.cpu()].repeat(N)
                apply_ratio_cap = ~sky_children.to(device=_s.device, dtype=torch.bool)
            if apply_ratio_cap.any():
                _s_capped = _s[apply_ratio_cap]
                _s_sorted, _s_order = _s_capped.sort(dim=1, descending=True)
                _s_mid = _s_sorted[:, 1].clamp(min=1e-6)
                _max_allowed = _s_mid * split_max_ratio
                _s_sorted[:, 0] = torch.minimum(_s_sorted[:, 0], _max_allowed)
                _s_restored = torch.zeros_like(_s_capped)
                _s_restored.scatter_(1, _s_order, _s_sorted)
                _s = _s.clone()
                _s[apply_ratio_cap] = _s_restored
                new_scaling = self.scaling_inverse_activation(_s)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        if current_kf_id is not None:
            new_kf_id = torch.full_like(new_kf_id, current_kf_id)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        # Propagate _is_sky flag to split children so sky Gaussians keep their tag.
        sky_split_mask = None
        if self._is_sky.shape[0] == selected_pts_mask.shape[0]:
            sky_split_mask = self._is_sky[selected_pts_mask.cpu()].repeat(N)

        # _layer: read parents *before* densification_postfix (which extends _layer).
        n_split = int(selected_pts_mask.sum().item()) * N
        layer_children = None
        if (
            self._layer.shape[0] > 0
            and self._layer.shape[0] == selected_pts_mask.shape[0]
        ):
            layer_children = self._layer[selected_pts_mask.cpu()].repeat(N).clone()
        parent_region_tag = None
        parent_anchor_kf = None
        parent_anchor_submap = None
        parent_birth_frame = None
        if self._region_tag.shape[0] == selected_pts_mask.shape[0]:
            parent_region_tag = self._region_tag[selected_pts_mask.cpu()].repeat(N).clone()
        if self._anchor_kf.shape[0] == selected_pts_mask.shape[0]:
            parent_anchor_kf = self._anchor_kf[selected_pts_mask.cpu()].repeat(N).clone()
        if self._anchor_submap.shape[0] == selected_pts_mask.shape[0]:
            parent_anchor_submap = self._anchor_submap[selected_pts_mask.cpu()].repeat(N).clone()
        if self._birth_frame.shape[0] == selected_pts_mask.shape[0]:
            parent_birth_frame = self._birth_frame[selected_pts_mask.cpu()].repeat(N).clone()
        if parent_region_tag is not None:
            bottom_children = (
                parent_region_tag.to(device=new_scaling.device)
                == self.REGION_TAG_BOTTOM_POLE_GROUND
            )
            if bottom_children.any():
                bottom_scale_mult = float(
                    self.config.get("Training", {}).get("bottom_pole_scale_mult", 1.0)
                )
                if bottom_scale_mult > 0.0 and bottom_scale_mult != 1.0:
                    scaled = self.scaling_activation(new_scaling)
                    scaled[bottom_children] *= bottom_scale_mult
                    new_scaling = self.scaling_inverse_activation(
                        scaled.clamp_min(1e-6)
                    )
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            sky_clone_mask=sky_split_mask,
            new_region_tag=parent_region_tag,
            new_anchor_kf=parent_anchor_kf,
            new_anchor_submap=parent_anchor_submap,
            new_birth_frame=parent_birth_frame,
        )
        if layer_children is not None and n_split > 0:
            self._layer[-n_split:] = layer_children

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)
        return {
            "count": n_selected,
            "selected_mask": selected_pts_mask,
            "n_children": n_split,
        }

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        scene_extent,
        current_kf_id=None,
        importance_score: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
    ):
        grad_norm = torch.norm(grads, dim=-1)
        selected_pts_mask = torch.where(grad_norm >= grad_threshold, True, False)
        too_large = (
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        if self._region_tag.shape[0] == selected_pts_mask.shape[0]:
            region_tag = self._region_tag.to(device=selected_pts_mask.device)
            upper_sky = region_tag == self.REGION_TAG_UPPER_SKY
            polar_cap = region_tag == self.REGION_TAG_POLAR_CAP_SKY
            horizon_bg = region_tag == self.REGION_TAG_HORIZON_BG
            bottom_pole = region_tag == self.REGION_TAG_BOTTOM_POLE_GROUND
            selected_pts_mask = selected_pts_mask & ~(upper_sky | polar_cap)
            if not bool(
                self.config.get("Training", {}).get("region_enable_clone_horizon_bg", True)
            ):
                selected_pts_mask = selected_pts_mask & ~horizon_bg
            horizon_grad_mult = float(
                self.config.get("Training", {}).get("region_horizon_clone_grad_mult", 1.10)
            )
            selected_pts_mask = selected_pts_mask & (
                ~horizon_bg | (grad_norm >= grad_threshold * horizon_grad_mult)
            )
            bottom_grad_mult = float(
                self.config.get("Training", {}).get("region_bottom_pole_clone_grad_mult", 0.75)
            )
            boosted_bottom = (
                bottom_pole
                & (~too_large)
                & (grad_norm >= grad_threshold * bottom_grad_mult)
            )
            selected_pts_mask = selected_pts_mask | boosted_bottom
        importance_gate = self._fastgs_importance_gate(
            importance_score, selected_pts_mask.shape[0], selected_pts_mask.device
        )
        if importance_gate is not None:
            selected_pts_mask = selected_pts_mask & importance_gate
        if exclude_mask is not None and exclude_mask.shape[0] == selected_pts_mask.shape[0]:
            selected_pts_mask = selected_pts_mask & ~exclude_mask.to(
                device=selected_pts_mask.device, dtype=torch.bool
            )
        allow_sky_clone = self.config.get("Training", {}).get("allow_sky_clone", False)
        if allow_sky_clone and self._is_sky.shape[0] == too_large.shape[0]:
            sky = self._is_sky.to(too_large.device)
            too_large = too_large & ~sky
        selected_pts_mask = selected_pts_mask & ~too_large
        selected_pts_mask = self._exclude_sky_from_densify_mask(
            selected_pts_mask, operation="clone"
        )
        if not selected_pts_mask.any():
            return {"count": 0, "selected_mask": selected_pts_mask}

        sky_clone_mask = (
            self._is_sky[selected_pts_mask.cpu()].clone()
            if allow_sky_clone
            else None
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        if current_kf_id is not None:
            new_kf_id = torch.full_like(new_kf_id, current_kf_id)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        # _layer: read parents *before* densification_postfix (which extends _layer).
        n_clone = int(selected_pts_mask.sum().item())
        layer_clones = None
        if (
            self._layer.shape[0] > 0
            and self._layer.shape[0] == selected_pts_mask.shape[0]
        ):
            layer_clones = self._layer[selected_pts_mask.cpu()].clone()
        parent_region_tag = None
        parent_anchor_kf = None
        parent_anchor_submap = None
        parent_birth_frame = None
        if self._region_tag.shape[0] == selected_pts_mask.shape[0]:
            parent_region_tag = self._region_tag[selected_pts_mask.cpu()].clone()
        if self._anchor_kf.shape[0] == selected_pts_mask.shape[0]:
            parent_anchor_kf = self._anchor_kf[selected_pts_mask.cpu()].clone()
        if self._anchor_submap.shape[0] == selected_pts_mask.shape[0]:
            parent_anchor_submap = self._anchor_submap[selected_pts_mask.cpu()].clone()
        if self._birth_frame.shape[0] == selected_pts_mask.shape[0]:
            parent_birth_frame = self._birth_frame[selected_pts_mask.cpu()].clone()
        if parent_region_tag is not None:
            bottom_clones = (
                parent_region_tag.to(device=new_scaling.device)
                == self.REGION_TAG_BOTTOM_POLE_GROUND
            )
            if bottom_clones.any():
                bottom_scale_mult = float(
                    self.config.get("Training", {}).get("bottom_pole_scale_mult", 1.0)
                )
                if bottom_scale_mult > 0.0 and bottom_scale_mult != 1.0:
                    scaled = self.scaling_activation(new_scaling)
                    scaled[bottom_clones] *= bottom_scale_mult
                    new_scaling = self.scaling_inverse_activation(
                        scaled.clamp_min(1e-6)
                    )
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            sky_clone_mask=sky_clone_mask,
            new_region_tag=parent_region_tag,
            new_anchor_kf=parent_anchor_kf,
            new_anchor_submap=parent_anchor_submap,
            new_birth_frame=parent_birth_frame,
        )
        if layer_clones is not None and n_clone > 0:
            self._layer[-n_clone:] = layer_clones
        return {"count": n_clone, "selected_mask": selected_pts_mask}

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        current_kf_id=None,
        screen_prune_sky_only: bool = False,
        apply_world_size_prune: bool = True,
        init_phase: bool = False,
        importance_score: Optional[torch.Tensor] = None,
        pruning_score: Optional[torch.Tensor] = None,
        fastgs_enabled: bool = False,
        fastgs_vcd_only: bool = True,
        current_window=None,
    ):
        # ``current_window`` is only consumed by ``PanoScaffoldModel`` (Fix2);
        # accepted here as a keyword to keep the signature compatible.
        del current_window
        profile_detailed = bool(
            self.config.get("Training", {}).get("profile_backend_detailed", False)
        )

        def _sync():
            if profile_detailed and torch.cuda.is_available():
                torch.cuda.synchronize()

        phase_ms = {
            "grads_ms": 0.0,
            "fastgs_mask_ms": 0.0,
            "clone_ms": 0.0,
            "split_ms": 0.0,
            "prune_mask_ms": 0.0,
            "prune_apply_ms": 0.0,
        }

        _sync()
        t0 = time.perf_counter()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        _sync()
        phase_ms["grads_ms"] = (time.perf_counter() - t0) * 1000.0

        fastgs_prune_mask = torch.zeros(
            (self.get_xyz.shape[0],), device=self.get_xyz.device, dtype=torch.bool
        )
        _sync()
        t0 = time.perf_counter()
        if fastgs_enabled and pruning_score is not None:
            fastgs_prune_mask = self._fastgs_prune_mask(
                pruning_score, self.get_xyz.shape[0], self.get_xyz.device
            )
        _sync()
        phase_ms["fastgs_mask_ms"] = (time.perf_counter() - t0) * 1000.0

        _sync()
        t0 = time.perf_counter()
        clone_stats = self.densify_and_clone(
            grads,
            max_grad,
            extent,
            current_kf_id=current_kf_id,
            importance_score=importance_score if fastgs_enabled else None,
            exclude_mask=fastgs_prune_mask if fastgs_enabled else None,
        )
        _sync()
        phase_ms["clone_ms"] = (time.perf_counter() - t0) * 1000.0
        n_clone = int(clone_stats.get("count", 0))
        if fastgs_prune_mask.numel() > 0 and n_clone > 0:
            fastgs_prune_mask = torch.cat(
                (
                    fastgs_prune_mask,
                    torch.zeros((n_clone,), device=fastgs_prune_mask.device, dtype=torch.bool),
                )
            )

        _sync()
        t0 = time.perf_counter()
        split_stats = self.densify_and_split(
            grads,
            max_grad,
            extent,
            current_kf_id=current_kf_id,
            init_phase=init_phase,
            importance_score=importance_score if fastgs_enabled else None,
            exclude_mask=fastgs_prune_mask if fastgs_enabled else None,
        )
        _sync()
        phase_ms["split_ms"] = (time.perf_counter() - t0) * 1000.0
        n_split = int(split_stats.get("count", 0))
        split_selected_mask = split_stats.get("selected_mask", None)
        n_split_children = int(split_stats.get("n_children", 0))
        if fastgs_prune_mask.numel() > 0:
            if (
                split_selected_mask is not None
                and split_selected_mask.shape[0] == fastgs_prune_mask.shape[0]
            ):
                fastgs_prune_mask = torch.cat(
                    (
                        fastgs_prune_mask[~split_selected_mask],
                        torch.zeros(
                            (n_split_children,),
                            device=fastgs_prune_mask.device,
                            dtype=torch.bool,
                        ),
                    )
                )
            elif n_split_children > 0:
                fastgs_prune_mask = torch.cat(
                    (
                        fastgs_prune_mask,
                        torch.zeros(
                            (n_split_children,),
                            device=fastgs_prune_mask.device,
                            dtype=torch.bool,
                        ),
                    )
                )

        debug_prune_stats = bool(
            self.config.get("Training", {}).get("debug_prune_stats", False)
        )

        # Build per-reason masks first, then OR them into the final prune_mask.
        _sync()
        t0 = time.perf_counter()
        m_opacity = (self.get_opacity < min_opacity).squeeze()

        m_screen = torch.zeros_like(m_opacity)
        m_world = torch.zeros_like(m_opacity)
        protect_sky_screen = bool(
            self.config.get("Training", {}).get("protect_sky_screen_prune", True)
        )
        protect_sky_world = bool(
            self.config.get("Training", {}).get("protect_sky_world_prune", True)
        )
        if max_screen_size:
            m_screen = self.max_radii2D > max_screen_size
            # ERP sky Gaussians naturally project to large screen radii; protect by default.
            if protect_sky_screen and self._is_sky.shape[0] == m_screen.shape[0]:
                m_screen = m_screen & ~self._is_sky.to(m_screen.device)

            if apply_world_size_prune:
                m_world = self.get_scaling.max(dim=1).values > 0.1 * extent
                # Sky Gaussians are intentionally large; protect by default.
                if protect_sky_world and self._is_sky.shape[0] == m_world.shape[0]:
                    m_world = m_world & ~self._is_sky.to(m_world.device)

        prune_ratio = float(
            self.config.get("Training", {}).get("gaussian_ratio_prune", 0.0)
        )
        # Prune needle-like Gaussians only when explicitly enabled (ratio_prune > 1).
        # WARNING: do NOT set this below ~50 for outdoor scenes 鈥?flat ground/wall
        # Gaussians are natural high-ratio pancakes and will be wrongly pruned.
        m_ratio = torch.zeros_like(m_opacity)
        if prune_ratio > 1.0:
            s = self.get_scaling
            m_ratio = (
                s.max(dim=1).values / s.min(dim=1).values.clamp(min=1e-6)
            ) > prune_ratio
            if init_phase and self._is_sky.shape[0] == m_ratio.shape[0]:
                m_ratio = m_ratio & ~self._is_sky.to(m_ratio.device)

        if self._region_tag.shape[0] == m_opacity.shape[0]:
            region_tag = self._region_tag.to(device=m_opacity.device)
            upper_sky = region_tag == self.REGION_TAG_UPPER_SKY
            polar_cap = region_tag == self.REGION_TAG_POLAR_CAP_SKY
            horizon_bg = region_tag == self.REGION_TAG_HORIZON_BG
            bottom = region_tag == self.REGION_TAG_BOTTOM_POLE_GROUND
            upper_min_opacity = float(
                self.config.get("Training", {}).get("region_min_opacity_upper_sky", 0.002)
            )
            m_opacity = torch.where(
                upper_sky | polar_cap,
                self.get_opacity.squeeze() < upper_min_opacity,
                m_opacity,
            )
            m_screen = m_screen & ~horizon_bg
            m_world = m_world & ~horizon_bg
            m_ratio = m_ratio & ~(upper_sky | polar_cap | horizon_bg)
            if self.n_obs.shape[0] == m_opacity.shape[0]:
                min_bottom_obs = int(
                    self.config.get("Training", {}).get(
                        "region_min_obs_before_prune_bottom_pole", 4
                    )
                )
                immature_bottom = bottom & (
                    self.n_obs.to(device=m_opacity.device) < min_bottom_obs
                )
                m_opacity = m_opacity & ~immature_bottom
                m_screen = m_screen & ~immature_bottom
                m_world = m_world & ~immature_bottom
                m_ratio = m_ratio & ~immature_bottom

        # Combine all prune reasons.
        prune_mask = m_opacity | m_screen | m_world | m_ratio
        fastgs_prune_candidates = int(fastgs_prune_mask.sum().item())
        fastgs_pruned = 0
        if fastgs_enabled and pruning_score is not None and not fastgs_vcd_only:
            if fastgs_prune_mask.shape[0] == prune_mask.shape[0]:
                fastgs_pruned = int((fastgs_prune_mask & ~prune_mask).sum().item())
                prune_mask = prune_mask | fastgs_prune_mask
            else:
                fastgs_prune_candidates = 0

        if debug_prune_stats:
            # Raw counts per-reason (may overlap).
            n_total = int(m_opacity.numel())
            n_opacity = int(m_opacity.sum().item())
            n_screen = int(m_screen.sum().item())
            n_world = int(m_world.sum().item())
            n_ratio = int(m_ratio.sum().item())
            n_union = int(prune_mask.sum().item())

            # Incremental unique contributions in the same order as the summary.
            u = torch.zeros_like(prune_mask)
            n_opacity_u = int((m_opacity & ~u).sum().item())
            u |= m_opacity
            n_screen_u = int((m_screen & ~u).sum().item())
            u |= m_screen
            n_world_u = int((m_world & ~u).sum().item())
            u |= m_world
            n_ratio_u = int((m_ratio & ~u).sum().item())
            u |= m_ratio

            print(
                "[prune_stats] total=%d union=%d | "
                "opacity=%d(+%d) screen=%d(+%d) world=%d(+%d) ratio=%d(+%d) | "
                "max_screen_size=%s screen_sky_only=%s world_prune=%s ratio_thr=%.3g"
                % (
                    n_total,
                    n_union,
                    n_opacity,
                    n_opacity_u,
                    n_screen,
                    n_screen_u,
                    n_world,
                    n_world_u,
                    n_ratio,
                    n_ratio_u,
                    str(max_screen_size),
                    str(screen_prune_sky_only),
                    str(apply_world_size_prune),
                    prune_ratio,
                )
            )
            with torch.no_grad():
                op = self.get_opacity.detach().flatten().float().cpu().numpy()
            mo = float(min_opacity)
            pct = np.percentile(op, [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99])
            (
                p01,
                p1,
                p5,
                p10,
                p25,
                p50,
                p75,
                p90,
                p95,
                p99,
            ) = (float(x) for x in pct)
            hist10, _ = np.histogram(np.clip(op, 0.0, 1.0), bins=10, range=(0.0, 1.0))
            # Bins aligned with common early/late opacity thresholds (e.g. 0.002 / 0.005).
            b_lo = int(np.sum(op < 0.002))
            b_mid = int(np.sum((op >= 0.002) & (op < 0.005)))
            b_5_10 = int(np.sum((op >= 0.005) & (op < 0.01)))
            b_10_50 = int(np.sum((op >= 0.01) & (op < 0.05)))
            b_hi = int(np.sum(op >= 0.05))
            print(
                "[prune_stats] opacity_dist (activated 伪) n=%d min=%.6g max=%.6g mean=%.6g std=%.6g | "
                "p0.1=%.6g p1=%.6g p5=%.6g p10=%.6g p25=%.6g p50=%.6g p75=%.6g p90=%.6g p95=%.6g p99=%.6g | "
                "below_min_opacity(%.6g)=%d"
                % (
                    int(op.size),
                    float(op.min()),
                    float(op.max()),
                    float(op.mean()),
                    float(op.std()),
                    p01,
                    p1,
                    p5,
                    p10,
                    p25,
                    p50,
                    p75,
                    p90,
                    p95,
                    p99,
                    mo,
                    int((op < mo).sum()),
                )
            )
            print(
                "[prune_stats] opacity_hist [0,1) uniform10=%s"
                % ",".join(str(int(c)) for c in hist10)
            )
            print(
                "[prune_stats] opacity_hist thr_bins "
                "[0,0.002)=%d [0.002,0.005)=%d [0.005,0.01)=%d [0.01,0.05)=%d [0.05,1]=%d"
                % (b_lo, b_mid, b_5_10, b_10_50, b_hi)
            )

        _sync()
        phase_ms["prune_mask_ms"] = (time.perf_counter() - t0) * 1000.0
        _sync()
        t0 = time.perf_counter()
        self.prune_points(prune_mask)
        _sync()
        phase_ms["prune_apply_ms"] = (time.perf_counter() - t0) * 1000.0
        return {
            "n_clone": n_clone,
            "n_split": n_split,
            "fastgs_prune_candidates": fastgs_prune_candidates,
            "fastgs_pruned": fastgs_pruned,
            "n_pruned": int(prune_mask.sum().item()),
            "phase_ms": phase_ms,
        }

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        grad = viewspace_point_tensor.grad
        if grad is not None and grad.dim() == 3 and grad.shape[0] == 1:
            grad = grad[0]
        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
