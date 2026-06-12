"""Minimal panoramic Gaussian map and mapper.

The map is intentionally compact: it exposes the attributes expected by the
PFGS360 adapter while keeping anchor-scaffold metadata local to this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from backend.pano_gs.adapter import PFGS360Renderer, PanoRenderCamera
from backend.pano_gs.losses import BackendLossWeights, backend_render_loss
from backend.pano_gs.pose_param import PoseDelta
from frontend.pano_droid.interfaces import FrontendOutput
from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing, pixel_grid
from mapping.gaussian_initializer import GaussianSeedBatch


@dataclass
class MapperStats:
    n_keyframes: int = 0
    n_anchors: int = 0
    last_loss: float | None = None
    last_phase: str | None = None
    last_pose_delta_norm: float | None = None
    optimization_steps: int = 0
    last_backend: str = "pfgs360_gsplat"
    fallback_renderer: bool = False
    last_inserted_anchors: int = 0
    last_skipped_voxel: int = 0
    last_skipped_budget: int = 0
    last_window_size: int = 0
    last_window_keyframes: list[int] = field(default_factory=list)
    last_window_observations: list[int] = field(default_factory=list)
    last_feedforward_current_frames: list[int] = field(default_factory=list)
    last_feedforward_history_frames: list[int] = field(default_factory=list)
    last_sampled_keyframes: list[int] = field(default_factory=list)
    last_trainable_pose_count: int = 0
    last_feedforward_opacity_resets: int = 0
    last_feedforward_pruned: int = 0
    last_replace_deleted: int = 0
    last_replace_fused: int = 0
    last_replace_compacted: int = 0
    last_sky_pruned: int = 0
    last_hash_hits: int = 0
    last_hash_near_hits: int = 0
    last_suppressed_insert: int = 0
    last_outlier_resets: int = 0
    last_outlier_pruned: int = 0
    last_render_missing_pixels: int = 0
    last_render_depth_mismatch_pixels: int = 0
    last_render_bad_pixels: int = 0
    last_missing_seed_candidates: int = 0
    last_depth_mismatch_seed_candidates: int = 0
    last_skipped_missing_budget: int = 0
    last_skipped_depth_mismatch_budget: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class MapperKeyframe:
    frame_id: int
    image: torch.Tensor
    gaussian_start: int
    gaussian_end: int
    sky_mask: torch.Tensor | None = None
    target_depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None


@dataclass
class MapperObservation:
    frame_id: int
    image: torch.Tensor
    pose_c2w: torch.Tensor
    is_keyframe: bool = False
    sky_mask: torch.Tensor | None = None
    target_depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None


@dataclass
class KeyframeRenderDiagnostic:
    frame_id: int
    target: torch.Tensor
    render: torch.Tensor
    depth: torch.Tensor | None
    loss: float
    psnr: float
    anchor_count: int
    phase: str | None


@dataclass
class DepthInsertionDiagnostic:
    frame_id: int
    render_depth: torch.Tensor | None
    predicted_depth: torch.Tensor | None
    rel_depth_error: torch.Tensor | None
    missing_mask: torch.Tensor | None
    depth_mismatch_mask: torch.Tensor | None
    render_bad_mask: torch.Tensor | None
    depth_scale: float = 1.0
    depth_shift: float = 0.0


class PanoGaussianMap(nn.Module):
    """Anchor-scaffold panorama map with gsplat360-compatible accessors."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        sh_degree: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.map_mode = "anchor_scaffold_panorama"
        self.active_sh_degree = min(int(sh_degree), 0)
        self.device_hint = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._reset_parameters()
        self._reset_skybox()
        self._anchor_level = torch.zeros(0, dtype=torch.int8)
        self._anchor_voxel_size = torch.zeros(0, dtype=torch.float32)
        self._anchor_grid_coord = torch.zeros(0, 3, dtype=torch.int32)
        self._anchor_obs_count = torch.zeros(0, dtype=torch.int32)
        self._anchor_conf_accum = torch.zeros(0, dtype=torch.float32)
        self._anchor_birth_frame = torch.zeros(0, dtype=torch.int32)
        self._anchor_last_seen_kf = torch.zeros(0, dtype=torch.int32)
        self._anchor_last_update_kf_ord = torch.zeros(0, dtype=torch.int32)
        self._anchor_inlier_obs = torch.zeros(0, dtype=torch.int32)
        self._anchor_outlier_obs = torch.zeros(0, dtype=torch.int32)

    def _reset_parameters(self) -> None:
        device = self.device_hint
        self.xyz = nn.Parameter(torch.zeros(0, 3, device=device))
        self.rotation = nn.Parameter(torch.zeros(0, 4, device=device))
        self.scaling = nn.Parameter(torch.zeros(0, 3, device=device))
        self.opacity_logit = nn.Parameter(torch.zeros(0, 1, device=device))
        self.features = nn.Parameter(torch.zeros(0, 3, device=device))

    def _reset_skybox(self) -> None:
        cfg = self.config.get("SkyBox", {}) if isinstance(self.config, dict) else {}
        self.skybox_enabled = bool(cfg.get("enabled", False))
        self.skybox_optimize = bool(cfg.get("optimize", True))
        self.skybox_optimization_mask_enable = bool(cfg.get("optimization_mask_enable", True))
        self.skybox_force_sky_render = bool(cfg.get("force_sky_render", False))
        self.skybox_init_fallback_to_full_image = bool(cfg.get("init_fallback_to_full_image", False))
        self.skybox_resolution = max(4, int(cfg.get("resolution", 512)))
        self.skybox_lr = float(cfg.get("lr", 1.0e-2))
        self._skybox_initialized = False
        self.skybox_logits: nn.Parameter | None = None
        if self.skybox_enabled:
            init = torch.full(
                (6, 3, self.skybox_resolution, self.skybox_resolution),
                0.5,
                device=self.device_hint,
                dtype=torch.float32,
            )
            self.skybox_logits = nn.Parameter(self._inv_sigmoid(init))

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_rotation(self) -> torch.Tensor:
        if self.rotation.numel() == 0:
            return self.rotation
        return torch.nn.functional.normalize(self.rotation, dim=-1, eps=1e-12)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.scaling) + 1e-5

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.sigmoid(self.features)

    @staticmethod
    def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-5, 1.0 - 1e-5)
        return torch.log(x / (1.0 - x))

    @property
    def has_skybox(self) -> bool:
        return bool(self.skybox_enabled and self.skybox_logits is not None)

    @property
    def get_skybox_faces(self) -> torch.Tensor | None:
        if self.skybox_logits is None:
            return None
        return torch.sigmoid(self.skybox_logits)

    def gaussian_parameters(self) -> list[nn.Parameter]:
        return [self.xyz, self.rotation, self.scaling, self.opacity_logit, self.features]

    def skybox_parameters(self) -> list[nn.Parameter]:
        if self.skybox_logits is None or not self.skybox_optimize:
            return []
        return [self.skybox_logits]

    def anchor_count(self) -> int:
        return int(self.xyz.shape[0])

    def add_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        voxel_size: float | None = None,
        last_update_kf_ord: int | None = None,
    ) -> int:
        if len(seeds) == 0:
            return 0
        device = self.xyz.device
        dtype = self.xyz.dtype
        xyz = seeds.xyz.to(device=device, dtype=dtype)
        rgb = seeds.rgb.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        conf = seeds.confidence.to(device=device, dtype=dtype).view(-1, 1).clamp(1e-4, 1.0)
        scale = seeds.scale.to(device=device, dtype=dtype).view(-1, 1).expand(-1, 3)
        quat = torch.zeros(xyz.shape[0], 4, device=device, dtype=dtype)
        quat[:, 0] = 1.0

        new_xyz = torch.cat([self.xyz.detach(), xyz], dim=0)
        new_rot = torch.cat([self.rotation.detach(), quat], dim=0)
        new_scaling = torch.cat([self.scaling.detach(), torch.log(torch.expm1(scale.clamp_min(1e-5)))], dim=0)
        new_opacity = torch.cat([self.opacity_logit.detach(), self._inv_sigmoid(conf)], dim=0)
        new_features = torch.cat([self.features.detach(), self._inv_sigmoid(rgb)], dim=0)

        self.xyz = nn.Parameter(new_xyz)
        self.rotation = nn.Parameter(new_rot)
        self.scaling = nn.Parameter(new_scaling)
        self.opacity_logit = nn.Parameter(new_opacity)
        self.features = nn.Parameter(new_features)

        self._anchor_level = torch.cat([self._anchor_level, seeds.level.detach().cpu().to(torch.int8)], dim=0)
        self._anchor_voxel_size = torch.cat(
            [self._anchor_voxel_size, seeds.scale.detach().cpu().to(torch.float32)], dim=0
        )
        if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == int(len(seeds)):
            grid = seeds.grid_coord.detach().cpu().to(torch.int32)
        elif voxel_size is not None and float(voxel_size) > 0.0:
            grid = torch.floor(seeds.xyz.detach().cpu() / float(voxel_size)).to(torch.int32)
        else:
            grid = torch.floor(seeds.xyz.detach().cpu() / seeds.scale.detach().cpu().view(-1, 1).clamp_min(1e-6)).to(torch.int32)
        self._anchor_grid_coord = torch.cat([self._anchor_grid_coord, grid.to(torch.int32)], dim=0)
        self._anchor_obs_count = torch.cat(
            [self._anchor_obs_count, torch.ones(len(seeds), dtype=torch.int32)], dim=0
        )
        self._anchor_conf_accum = torch.cat(
            [self._anchor_conf_accum, seeds.confidence.detach().cpu().to(torch.float32)], dim=0
        )
        frame_ids = torch.full((len(seeds),), int(seeds.frame_id), dtype=torch.int32)
        self._anchor_birth_frame = torch.cat([self._anchor_birth_frame, frame_ids], dim=0)
        self._anchor_last_seen_kf = torch.cat([self._anchor_last_seen_kf, frame_ids], dim=0)
        update_ord = int(seeds.frame_id) if last_update_kf_ord is None else int(last_update_kf_ord)
        self._anchor_last_update_kf_ord = torch.cat(
            [self._anchor_last_update_kf_ord, torch.full((len(seeds),), update_ord, dtype=torch.int32)],
            dim=0,
        )
        self._anchor_inlier_obs = torch.cat([self._anchor_inlier_obs, torch.zeros(len(seeds), dtype=torch.int32)], dim=0)
        self._anchor_outlier_obs = torch.cat([self._anchor_outlier_obs, torch.zeros(len(seeds), dtype=torch.int32)], dim=0)
        return int(xyz.shape[0])

    def prune_anchors(self, prune_mask: torch.Tensor) -> int:
        if self.anchor_count() == 0:
            return 0
        mask = prune_mask.detach().to(device=self.xyz.device, dtype=torch.bool).view(-1)
        if int(mask.shape[0]) != self.anchor_count():
            raise ValueError(f"Prune mask length {int(mask.shape[0])} does not match anchor count {self.anchor_count()}")
        keep = ~mask
        n_pruned = int(mask.sum().item())
        if n_pruned <= 0:
            return 0
        self.xyz = nn.Parameter(self.xyz.detach()[keep])
        self.rotation = nn.Parameter(self.rotation.detach()[keep])
        self.scaling = nn.Parameter(self.scaling.detach()[keep])
        self.opacity_logit = nn.Parameter(self.opacity_logit.detach()[keep])
        self.features = nn.Parameter(self.features.detach()[keep])
        keep_cpu = keep.detach().cpu()
        self._anchor_level = self._anchor_level[keep_cpu]
        self._anchor_voxel_size = self._anchor_voxel_size[keep_cpu]
        self._anchor_grid_coord = self._anchor_grid_coord[keep_cpu]
        self._anchor_obs_count = self._anchor_obs_count[keep_cpu]
        self._anchor_conf_accum = self._anchor_conf_accum[keep_cpu]
        self._anchor_birth_frame = self._anchor_birth_frame[keep_cpu]
        self._anchor_last_seen_kf = self._anchor_last_seen_kf[keep_cpu]
        self._anchor_last_update_kf_ord = self._anchor_last_update_kf_ord[keep_cpu]
        self._anchor_inlier_obs = self._anchor_inlier_obs[keep_cpu]
        self._anchor_outlier_obs = self._anchor_outlier_obs[keep_cpu]
        return n_pruned

    def make_optimizer(self, *, lr: float = 2e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        param_groups = [{"params": self.gaussian_parameters(), "lr": float(lr), "name": "gaussians"}]
        sky_params = self.skybox_parameters()
        if sky_params:
            param_groups.append({"params": sky_params, "lr": self.skybox_lr, "name": "skybox"})
        return torch.optim.AdamW(param_groups, weight_decay=float(weight_decay))

    def initialize_skybox_from_image(
        self,
        image: torch.Tensor,
        c2w: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None = None,
        force: bool = False,
    ) -> bool:
        if not self.has_skybox:
            return False
        if self._skybox_initialized and not force:
            return False
        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] != 3:
            return False
        device = self.skybox_logits.device
        img = img.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        _, H, W = img.shape
        if sky_mask is None:
            sky = self._sky_mask_from_image(img)
        else:
            sky = sky_mask.detach().bool().to(device=device)
            if sky.ndim == 3:
                sky = sky[0]
            if tuple(sky.shape[-2:]) != (H, W):
                sky = F.interpolate(sky.float().view(1, 1, *sky.shape[-2:]), size=(H, W), mode="nearest")[0, 0] > 0.5
        if not bool(sky.any()):
            if not self.skybox_init_fallback_to_full_image:
                return False
            sky = torch.ones(H, W, device=device, dtype=torch.bool)
        dirs = self._world_dirs_for_erp(H, W, c2w.to(device=device, dtype=torch.float32))
        face, uv = self._directions_to_cubemap(dirs.reshape(-1, 3))
        sky_flat = sky.reshape(-1)
        img_flat = img.permute(1, 2, 0).reshape(-1, 3)
        selected = torch.nonzero(sky_flat, as_tuple=False).flatten()
        if selected.numel() == 0:
            return False
        face_sel = face[selected]
        uv_sel = uv[selected]
        s = int(self.skybox_resolution)
        ix = ((uv_sel[:, 0] + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        iy = ((uv_sel[:, 1] + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        linear = face_sel.long() * (s * s) + iy * s + ix
        accum = torch.zeros(6 * s * s, 3, device=device, dtype=torch.float32)
        counts = torch.zeros(6 * s * s, 1, device=device, dtype=torch.float32)
        accum.index_add_(0, linear, img_flat[selected])
        counts.index_add_(0, linear, torch.ones(selected.numel(), 1, device=device, dtype=torch.float32))
        mean_sky = img_flat[selected].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
        flat = mean_sky.expand(6 * s * s, 3).clone()
        filled = counts[:, 0] > 0
        flat[filled] = accum[filled] / counts[filled].clamp_min(1.0)
        faces = flat.view(6, s, s, 3).permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        with torch.no_grad():
            self.skybox_logits.copy_(self._inv_sigmoid(faces))
        self._skybox_initialized = True
        return True

    def sample_skybox(self, world_dirs: torch.Tensor) -> torch.Tensor:
        faces = self.get_skybox_faces
        if faces is None:
            raise RuntimeError("SkyBox is disabled.")
        original_shape = world_dirs.shape[:-1]
        dirs = torch.nn.functional.normalize(world_dirs.reshape(-1, 3).to(faces), dim=-1, eps=1e-8)
        face, uv = self._directions_to_cubemap(dirs)
        out = torch.zeros(dirs.shape[0], 3, device=faces.device, dtype=faces.dtype)
        for face_idx in range(6):
            mask = face == face_idx
            if not bool(mask.any()):
                continue
            grid = uv[mask].view(1, -1, 1, 2).to(device=faces.device, dtype=faces.dtype)
            sampled = F.grid_sample(
                faces[face_idx : face_idx + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            out[mask] = sampled[0, :, :, 0].T
        return out.view(*original_shape, 3)

    def skybox_erp_preview(self, *, height: int = 256, width: int = 512, c2w: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.has_skybox:
            return None
        pose = torch.eye(4, device=self.skybox_logits.device, dtype=torch.float32) if c2w is None else c2w
        dirs = self._world_dirs_for_erp(int(height), int(width), pose.to(device=self.skybox_logits.device, dtype=torch.float32))
        rgb = self.sample_skybox(dirs).permute(2, 0, 1).contiguous()
        return rgb.detach().cpu().clamp(0.0, 1.0)

    def _world_dirs_for_erp(self, H: int, W: int, c2w: torch.Tensor) -> torch.Tensor:
        grid = pixel_grid(H, W, device=c2w.device, dtype=c2w.dtype).view(-1, 2)
        dirs_cam = erp_pixel_to_bearing(grid, H, W).to(device=c2w.device, dtype=c2w.dtype)
        rot = c2w[:3, :3]
        dirs_world = (rot @ dirs_cam.T).T
        return dirs_world.view(H, W, 3)

    @staticmethod
    def _directions_to_cubemap(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dirs = torch.nn.functional.normalize(directions.float(), dim=-1, eps=1e-8)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        ax, ay, az = x.abs(), y.abs(), z.abs()
        face = torch.zeros(dirs.shape[0], device=dirs.device, dtype=torch.long)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        is_x = (ax >= ay) & (ax >= az)
        is_y = (ay > ax) & (ay >= az)
        is_z = ~(is_x | is_y)
        pos = is_x & (x >= 0)
        neg = is_x & (x < 0)
        face[pos] = 0
        u[pos] = -z[pos] / ax[pos].clamp_min(1e-8)
        v[pos] = -y[pos] / ax[pos].clamp_min(1e-8)
        face[neg] = 1
        u[neg] = z[neg] / ax[neg].clamp_min(1e-8)
        v[neg] = -y[neg] / ax[neg].clamp_min(1e-8)
        pos = is_y & (y >= 0)
        neg = is_y & (y < 0)
        face[pos] = 2
        u[pos] = x[pos] / ay[pos].clamp_min(1e-8)
        v[pos] = z[pos] / ay[pos].clamp_min(1e-8)
        face[neg] = 3
        u[neg] = x[neg] / ay[neg].clamp_min(1e-8)
        v[neg] = -z[neg] / ay[neg].clamp_min(1e-8)
        pos = is_z & (z >= 0)
        neg = is_z & (z < 0)
        face[pos] = 4
        u[pos] = x[pos] / az[pos].clamp_min(1e-8)
        v[pos] = -y[pos] / az[pos].clamp_min(1e-8)
        face[neg] = 5
        u[neg] = -x[neg] / az[neg].clamp_min(1e-8)
        v[neg] = -y[neg] / az[neg].clamp_min(1e-8)
        return face, torch.stack([u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)], dim=-1)

    def _sky_mask_from_image(self, image: torch.Tensor) -> torch.Tensor:
        cfg = self.config.get("SkyBox", {}) if isinstance(self.config, dict) else {}
        top_ratio = float(cfg.get("sky_mask_top_ratio", self.config.get("Mapping", {}).get("sky_mask_top_ratio", 0.58)))
        min_blue = float(cfg.get("sky_mask_min_blue", self.config.get("Mapping", {}).get("sky_mask_min_blue", 0.35)))
        blue_margin = float(cfg.get("sky_mask_blue_margin", self.config.get("Mapping", {}).get("sky_mask_blue_margin", 0.05)))
        cloud_brightness = float(
            cfg.get("sky_mask_cloud_brightness", self.config.get("Mapping", {}).get("sky_mask_cloud_brightness", 0.72))
        )
        cloud_saturation = float(
            cfg.get("sky_mask_cloud_saturation", self.config.get("Mapping", {}).get("sky_mask_cloud_saturation", 0.22))
        )
        texture_threshold = float(
            cfg.get("sky_mask_texture_threshold", self.config.get("Mapping", {}).get("sky_mask_texture_threshold", 0.08))
        )
        img = image.detach().float().clamp(0.0, 1.0)
        _, H, _ = img.shape
        rows = torch.arange(H, device=img.device, dtype=img.dtype).view(H, 1)
        upper = rows < float(H) * top_ratio
        r, g, b = img[0], img[1], img[2]
        max_rgb = img.max(dim=0).values
        min_rgb = img.min(dim=0).values
        saturation = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-6)
        blue_sky = (b >= min_blue) & (b >= r + blue_margin) & (b >= g + 0.5 * blue_margin)
        gray = img.mean(dim=0)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
        dy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
        low_texture = (dx + dy) <= texture_threshold
        cloud_sky = (max_rgb >= cloud_brightness) & (saturation <= cloud_saturation) & low_texture
        return upper & (blue_sky | cloud_sky)

    def save_ply(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self.get_xyz.detach().cpu().float().numpy()
        n = int(xyz.shape[0])
        normals = np.zeros_like(xyz, dtype=np.float32)
        rgb = self.get_features.detach().cpu().float().clamp(0.0, 1.0).numpy()
        f_dc = (rgb - 0.5) / 0.28209479177387814
        f_rest = np.zeros((n, 24), dtype=np.float32)
        opacity = self.opacity_logit.detach().cpu().float().numpy()
        scale = self.get_scaling.detach().cpu().float().clamp_min(1.0e-8).log().numpy()
        rot = self.get_rotation.detach().cpu().float().numpy()
        attributes = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            *(f"f_rest_{idx}" for idx in range(24)),
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        dtype = np.dtype([(name, "<f4") for name in attributes])
        elements = np.empty(n, dtype=dtype)
        values = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rot), axis=1).astype(np.float32, copy=False)
        for idx, name in enumerate(attributes):
            elements[name] = values[:, idx]
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header.extend(f"property float {name}" for name in attributes)
        header.append("end_header")
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            elements.tofile(f)
        return str(path)

    def save_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "anchor_level": self._anchor_level,
                "anchor_voxel_size": self._anchor_voxel_size,
                "anchor_grid_coord": self._anchor_grid_coord,
                "anchor_obs_count": self._anchor_obs_count,
                "anchor_conf_accum": self._anchor_conf_accum,
                "anchor_birth_frame": self._anchor_birth_frame,
                "anchor_last_seen_kf": self._anchor_last_seen_kf,
                "anchor_last_update_kf_ord": self._anchor_last_update_kf_ord,
                "anchor_inlier_obs": self._anchor_inlier_obs,
                "anchor_outlier_obs": self._anchor_outlier_obs,
                "config": self.config,
            },
            path,
        )
        return str(path)


class PanoGaussianMapper:
    """Keyframe-driven map insertion and optional render refinement."""

    def __init__(
        self,
        gaussian_map: PanoGaussianMap,
        *,
        renderer: PFGS360Renderer | None = None,
        lr: float = 2e-3,
        loss_weights: BackendLossWeights | None = None,
    ) -> None:
        self.map = gaussian_map
        self.renderer = renderer or PFGS360Renderer(config=gaussian_map.config)
        self.optimizer = gaussian_map.make_optimizer(lr=lr)
        if loss_weights is None:
            sky_cfg = gaussian_map.config.get("SkyBox", {}) if isinstance(gaussian_map.config, dict) else {}
            self.loss_weights = BackendLossWeights(
                sky_alpha=float(sky_cfg.get("sky_alpha_loss_weight", 0.0)),
            )
        else:
            self.loss_weights = loss_weights
        self.stats = MapperStats()
        self.optim_cfg = gaussian_map.config.get("BackendOptimization", {}) if isinstance(gaussian_map.config, dict) else {}
        mapping_cfg = gaussian_map.config.get("Mapping", {}) if isinstance(gaussian_map.config, dict) else {}
        novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
        pano_cfg = gaussian_map.config.get("PanoVGGT", {}) if isinstance(gaussian_map.config, dict) else {}
        m3_enabled = bool((pano_cfg.get("M3Sphere", {}) or {}).get("enabled", False)) if isinstance(pano_cfg, dict) else False
        self.novel_insertion_enabled = bool(novel_cfg.get("enabled", m3_enabled))
        self.novel_insertion_strategy = str(novel_cfg.get("strategy", "legacy") or "legacy").lower()
        self.pfgs360_replace_fuse_enabled = bool(
            self.novel_insertion_enabled and self.novel_insertion_strategy == "pfgs360_replace_fuse"
        )
        self.pfgs360_insertion_enabled = bool(
            self.novel_insertion_enabled and self.novel_insertion_strategy in {"pfgs360", "pfgs360_replace_fuse"}
        )
        self.pfgs360_voxel_size = max(float(novel_cfg.get("voxel_size", 0.12)), 1.0e-6)
        self.replace_fuse_delete_rel_min = float(novel_cfg.get("replace_delete_rel_min", 0.10))
        self.replace_fuse_delete_rel_max = float(novel_cfg.get("replace_delete_rel_max", 0.20))
        self.replace_fuse_insert_rel_min = float(novel_cfg.get("replace_insert_rel_min", self.replace_fuse_delete_rel_min))
        self.replace_fuse_front_depth_abs_tol = float(novel_cfg.get("replace_front_depth_abs_tol", 0.03))
        self.replace_fuse_front_depth_rel_tol = float(novel_cfg.get("replace_front_depth_rel_tol", 0.02))
        self.replace_fuse_max_delete_per_keyframe = max(0, int(novel_cfg.get("max_replace_delete_per_keyframe", 30000)))
        self.replace_fuse_compact_voxels = bool(novel_cfg.get("compact_voxels", True))
        self.pfgs360_render_alpha_min = float(novel_cfg.get("render_alpha_min", 0.20))
        self.pfgs360_missing_alpha_min = float(novel_cfg.get("missing_alpha_min", self.pfgs360_render_alpha_min))
        self.pfgs360_render_depth_rel_threshold = float(novel_cfg.get("render_depth_rel_threshold", 0.10))
        self.pfgs360_foreground_rel_threshold = float(novel_cfg.get("foreground_rel_threshold", 0.10))
        self.pfgs360_photometric_error_threshold = float(novel_cfg.get("photometric_error_threshold", 0.08))
        self.pfgs360_near_grid_radius = max(0, int(novel_cfg.get("near_grid_radius", 1)))
        self.pfgs360_near_distance_factor = max(0.0, float(novel_cfg.get("near_distance_factor", 1.0)))
        default_reset = 0 if self.pfgs360_replace_fuse_enabled else 3
        default_prune = 0 if self.pfgs360_replace_fuse_enabled else 6
        self.pfgs360_reset_after_outliers = max(0, int(novel_cfg.get("reset_after_outlier_observations", default_reset)))
        self.pfgs360_prune_after_outliers = max(0, int(novel_cfg.get("prune_after_outlier_observations", default_prune)))
        self.pfgs360_protect_recent_keyframes = max(0, int(novel_cfg.get("protect_recent_keyframes", 8)))
        self.pfgs360_max_prune_per_keyframe = max(0, int(novel_cfg.get("max_prune_per_keyframe", 500)))
        self.first_keyframe_max_seeds = int(novel_cfg.get("first_keyframe_max_seeds", 80000))
        self.keyframe_max_seeds = int(novel_cfg.get("keyframe_max_seeds", 30000))
        self.max_missing_seeds_per_keyframe = int(novel_cfg.get("max_missing_seeds_per_keyframe", 0))
        self.max_depth_mismatch_seeds_per_keyframe = int(
            novel_cfg.get("max_depth_mismatch_seeds_per_keyframe", 0)
        )
        self.prioritize_depth_mismatch = bool(novel_cfg.get("prioritize_depth_mismatch", True))
        self.global_anchor_budget = int(novel_cfg.get("global_anchor_budget", 1500000))
        self.first_keyframe_voxel_neighbor_radius = max(
            0,
            int(novel_cfg.get("first_keyframe_voxel_neighbor_radius", novel_cfg.get("voxel_neighbor_radius", 0))),
        )
        self.voxel_neighbor_radius = max(0, int(novel_cfg.get("voxel_neighbor_radius", 0)))
        self.keyframes: list[MapperKeyframe] = []
        self.observations: dict[int, MapperObservation] = {}
        self.pose_deltas: dict[int, PoseDelta] = {}
        self.last_inserted_range: tuple[int, int] = (0, 0)
        self.last_requested_source_flat_idx: torch.Tensor | None = None
        self.last_inserted_source_flat_idx: torch.Tensor | None = None
        self.last_source_hw: tuple[int, int] | None = None
        self.last_depth_insertion_diagnostic: DepthInsertionDiagnostic | None = None
        self.frontend_graph_window_ids: tuple[int, ...] = ()
        if loss_weights is None and self.pfgs360_replace_fuse_enabled:
            sky_cfg = gaussian_map.config.get("SkyBox", {}) if isinstance(gaussian_map.config, dict) else {}
            self.loss_weights = BackendLossWeights(
                depth=float(self.optim_cfg.get("depth_loss_weight", 0.03)),
                opacity=float(self.optim_cfg.get("opacity_loss_weight", 0.0)),
                sky_alpha=float(sky_cfg.get("sky_alpha_loss_weight", 0.0)),
                photometric_mode=str(self.optim_cfg.get("photometric_loss_mode", "l1_dssim")),
                rgb_l1_weight=float(self.optim_cfg.get("rgb_l1_weight", 0.8)),
                dssim_weight=float(self.optim_cfg.get("dssim_weight", 0.2)),
                depth_loss_mode=str(self.optim_cfg.get("depth_loss_mode", "relative_clamped")),
                depth_residual_clamp=float(self.optim_cfg.get("depth_residual_clamp", 0.20)),
            )

    @property
    def feedforward_window_enabled(self) -> bool:
        cfg = self._feedforward_window_cfg()
        return bool(cfg.get("enabled", False)) or bool(self.optim_cfg.get("optimize_after_every_chunk", False))

    def _feedforward_window_cfg(self) -> dict:
        cfg = self.optim_cfg.get("FeedForwardWindow", {}) if isinstance(self.optim_cfg, dict) else {}
        return cfg if isinstance(cfg, dict) else {}

    @property
    def uses_joint_optimization(self) -> bool:
        cfg = self.optim_cfg
        return bool(cfg.get("enabled", False)) or bool(cfg.get("pose_refine_enable", False))

    def insert_keyframe(
        self,
        seeds: GaussianSeedBatch,
        frontend_output: FrontendOutput,
        image: torch.Tensor | None = None,
    ) -> int:
        requested = len(seeds)
        current_kf_ord = int(self.stats.n_keyframes)
        self.last_requested_source_flat_idx = (
            None if seeds.source_flat_idx is None else seeds.source_flat_idx.detach().cpu().long()
        )
        self.last_source_hw = seeds.source_hw
        self.last_depth_insertion_diagnostic = None
        sky_mask = self._skybox_mask_from_image(image) if image is not None else None
        if image is not None and self.map.has_skybox:
            self.map.initialize_skybox_from_image(image, frontend_output.pose_c2w, sky_mask=sky_mask)
        seeds, filter_stats = self._filter_novel_seeds(
            seeds,
            frontend_output=frontend_output,
            image=image,
            sky_mask=sky_mask,
        )
        self.last_inserted_source_flat_idx = (
            None if seeds.source_flat_idx is None else seeds.source_flat_idx.detach().cpu().long()
        )
        if self.last_source_hw is None:
            self.last_source_hw = seeds.source_hw
        start = self.map.anchor_count()
        n = self.map.add_seeds(
            seeds,
            voxel_size=self.pfgs360_voxel_size if self.pfgs360_replace_fuse_enabled else None,
            last_update_kf_ord=current_kf_ord if self.pfgs360_replace_fuse_enabled else None,
        )
        end = start + int(n)
        self.last_inserted_range = (start, end)
        if self.pfgs360_replace_fuse_enabled:
            compacted = self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
            if compacted > 0:
                filter_stats["compacted"] = int(filter_stats.get("compacted", 0)) + int(compacted)
                self._rebuild_keyframe_ranges_from_birth_frames()
                self.last_inserted_range = self._range_for_birth_frame(int(frontend_output.frame_id))
        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        self.stats.n_keyframes += 1
        self.stats.n_anchors = self.map.anchor_count()
        self.stats.last_inserted_anchors = int(n)
        self.stats.last_skipped_voxel = int(filter_stats.get("skipped_voxel", 0))
        self.stats.last_skipped_budget = int(filter_stats.get("skipped_budget", 0))
        self.stats.last_hash_hits = int(filter_stats.get("hash_hits", 0))
        self.stats.last_hash_near_hits = int(filter_stats.get("hash_near_hits", 0))
        self.stats.last_suppressed_insert = int(filter_stats.get("suppressed_insert", 0))
        self.stats.last_outlier_resets = int(filter_stats.get("outlier_resets", 0))
        self.stats.last_outlier_pruned = int(filter_stats.get("outlier_pruned", 0))
        self.stats.last_render_missing_pixels = int(filter_stats.get("missing_pixels", 0))
        self.stats.last_render_depth_mismatch_pixels = int(filter_stats.get("depth_mismatch_pixels", 0))
        self.stats.last_render_bad_pixels = int(filter_stats.get("render_bad_pixels", 0))
        self.stats.last_missing_seed_candidates = int(filter_stats.get("missing_seed_candidates", 0))
        self.stats.last_depth_mismatch_seed_candidates = int(
            filter_stats.get("depth_mismatch_seed_candidates", 0)
        )
        self.stats.last_skipped_missing_budget = int(filter_stats.get("skipped_missing_budget", 0))
        self.stats.last_skipped_depth_mismatch_budget = int(
            filter_stats.get("skipped_depth_mismatch_budget", 0)
        )
        self.stats.last_replace_deleted = int(filter_stats.get("replace_deleted", 0))
        self.stats.last_replace_fused = int(filter_stats.get("fused", 0))
        self.stats.last_replace_compacted = int(filter_stats.get("compacted", 0))
        if image is not None:
            register_start, register_end = self.last_inserted_range
            self._register_keyframe(frontend_output, image, start=register_start, end=register_end, sky_mask=sky_mask)
            self.register_observation(frontend_output, image, is_keyframe=True, sky_mask=sky_mask)
        if n == 0:
            self.stats.notes.append(f"frame {frontend_output.frame_id}: no seeds inserted")
        if self.novel_insertion_enabled and requested != n:
            self.stats.notes.append(
                (
                    f"frame {frontend_output.frame_id}: novel insertion kept {n}/{requested} "
                    f"seeds, skipped_voxel={self.stats.last_skipped_voxel}, "
                    f"skipped_budget={self.stats.last_skipped_budget}"
                )
            )
        if self.pfgs360_insertion_enabled and (
            self.stats.last_suppressed_insert or self.stats.last_outlier_resets or self.stats.last_outlier_pruned
        ):
            self.stats.notes.append(
                (
                    f"frame {frontend_output.frame_id}: pfgs360 insertion "
                    f"hits={self.stats.last_hash_hits}, near_hits={self.stats.last_hash_near_hits}, "
                    f"suppressed={self.stats.last_suppressed_insert}, "
                    f"resets={self.stats.last_outlier_resets}, pruned={self.stats.last_outlier_pruned}"
                )
            )
        return n

    def set_frontend_graph_window_ids(self, frame_ids) -> None:
        ids: list[int] = []
        if frame_ids is None:
            self.frontend_graph_window_ids = ()
            return
        for frame_id in frame_ids:
            try:
                value = int(frame_id)
            except (TypeError, ValueError):
                continue
            if value not in ids:
                ids.append(value)
        self.frontend_graph_window_ids = tuple(ids)

    def register_observation(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        is_keyframe: bool | None = None,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        self.register_observation_values(
            frame_id=int(frontend_output.frame_id),
            image=image,
            c2w=frontend_output.pose_c2w,
            inverse_depth=frontend_output.inverse_depth,
            depth_confidence=frontend_output.depth_confidence,
            world_points=frontend_output.world_points,
            world_points_confidence=frontend_output.world_points_confidence,
            is_keyframe=bool(frontend_output.is_keyframe) if is_keyframe is None else bool(is_keyframe),
            sky_mask=sky_mask,
        )

    def register_observation_values(
        self,
        *,
        frame_id: int,
        image: torch.Tensor,
        c2w: torch.Tensor,
        inverse_depth: torch.Tensor | None = None,
        depth_confidence: torch.Tensor | None = None,
        world_points: torch.Tensor | None = None,
        world_points_confidence: torch.Tensor | None = None,
        is_keyframe: bool = False,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        img = image.detach().cpu().float()
        if img.ndim == 4 and int(img.shape[0]) == 1:
            img = img[0]
        if img.ndim != 3:
            return
        frame_id = int(frame_id)
        sky = sky_mask if sky_mask is not None else self._skybox_mask_from_image(img)
        depth, conf = self._target_depth_from_tensors(
            inverse_depth=inverse_depth,
            world_points=world_points,
            pose_c2w=c2w,
            confidence=depth_confidence if depth_confidence is not None else world_points_confidence,
            size=(int(img.shape[-2]), int(img.shape[-1])),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.observations[frame_id] = MapperObservation(
            frame_id=frame_id,
            image=img,
            pose_c2w=c2w.detach().cpu().float(),
            is_keyframe=bool(is_keyframe),
            sky_mask=sky.detach().cpu().bool() if torch.is_tensor(sky) else None,
            target_depth=None if depth is None else depth.detach().cpu().float(),
            depth_confidence=None if conf is None else conf.detach().cpu().float(),
        )

    def _filter_novel_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None = None,
        image: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        if not self.novel_insertion_enabled or len(seeds) == 0:
            return seeds, {"skipped_voxel": 0, "skipped_budget": 0}
        if self.pfgs360_insertion_enabled:
            return self._filter_pfgs360_seeds(
                seeds,
                frontend_output=frontend_output,
                image=image,
                sky_mask=sky_mask,
            )
        per_keyframe_budget = self.first_keyframe_max_seeds if self.stats.n_keyframes == 0 else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))

        xyz_cpu = seeds.xyz.detach().cpu().float()
        scale_cpu = seeds.scale.detach().cpu().float().clamp_min(1.0e-6)
        level_cpu = seeds.level.detach().cpu()
        conf_cpu = seeds.confidence.detach().cpu().float()
        order = torch.argsort(conf_cpu, descending=True)
        occupied = self._build_voxel_index()
        neighbor_radius = (
            self.first_keyframe_voxel_neighbor_radius if self.stats.n_keyframes == 0 else self.voxel_neighbor_radius
        )
        kept: list[int] = []
        skipped_voxel = 0
        skipped_budget = 0
        for seed_idx in order.tolist():
            key = self._seed_voxel_key_from_cpu(xyz_cpu, scale_cpu, level_cpu, int(seed_idx))
            hit = self._find_voxel_hit(occupied, key, radius=neighbor_radius)
            if hit is not None:
                self._accumulate_existing_observation(hit, float(conf_cpu[seed_idx]))
                skipped_voxel += 1
                continue
            if len(kept) >= budget:
                skipped_budget += 1
                continue
            kept.append(int(seed_idx))
            occupied[key] = -1
        if not kept:
            return self._empty_seed_like(seeds), {"skipped_voxel": skipped_voxel, "skipped_budget": skipped_budget}
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), {"skipped_voxel": skipped_voxel, "skipped_budget": skipped_budget}

    def _filter_pfgs360_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None,
        image: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        if self.pfgs360_replace_fuse_enabled:
            return self._filter_replace_fuse_seeds(
                seeds,
                frontend_output=frontend_output,
                image=image,
                sky_mask=sky_mask,
            )
        seeds = self._with_pfgs360_seed_metadata(seeds)
        per_keyframe_budget = self.first_keyframe_max_seeds if self.stats.n_keyframes == 0 else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))

        stats = {
            "skipped_voxel": 0,
            "skipped_budget": 0,
            "hash_hits": 0,
            "hash_near_hits": 0,
            "suppressed_insert": 0,
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
            "missing_seed_candidates": 0,
            "depth_mismatch_seed_candidates": 0,
            "skipped_missing_budget": 0,
            "skipped_depth_mismatch_budget": 0,
        }
        insert_enabled = (
            torch.ones(len(seeds), dtype=torch.bool)
            if seeds.insert_enabled is None
            else seeds.insert_enabled.detach().cpu().bool()
        )
        depth_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        missing_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        first_keyframe = self.stats.n_keyframes == 0 or self.map.anchor_count() == 0
        if first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            self.last_depth_insertion_diagnostic = self._prediction_depth_insertion_diagnostic(
                frontend_output,
                image_hw,
            )
        if not first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            temporal_ok = self._seed_pixel_mask(seeds, insert_enabled, image_hw)
            render_masks, evidence_stats = self._pfgs360_render_bad_mask(
                frontend_output,
                image,
                sky_mask=sky_mask,
                temporal_ok=temporal_ok,
            )
            stats.update(evidence_stats)
            if render_masks is not None:
                depth_seed_mask = (
                    self._sample_seed_mask(render_masks["depth_mismatch"], seeds).detach().cpu().bool()
                    & insert_enabled
                )
                missing_seed_mask = (
                    self._sample_seed_mask(render_masks["missing"], seeds).detach().cpu().bool()
                    & insert_enabled
                )
                if self.prioritize_depth_mismatch:
                    missing_seed_mask &= ~depth_seed_mask
                render_bad_seed_mask = depth_seed_mask | missing_seed_mask
                stats["depth_mismatch_seed_candidates"] = int(depth_seed_mask.sum().item())
                stats["missing_seed_candidates"] = int(missing_seed_mask.sum().item())
                insert_enabled &= render_bad_seed_mask

        score = seeds.insert_score if seeds.insert_score is not None else seeds.confidence
        score_cpu = score.detach().cpu().float()
        conf_cpu = seeds.confidence.detach().cpu().float()
        xyz_cpu = seeds.xyz.detach().cpu().float()
        grid_cpu = (
            seeds.grid_coord.detach().cpu().to(torch.int32)
            if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == len(seeds)
            else torch.floor(xyz_cpu / float(self.pfgs360_voxel_size)).to(torch.int32)
        )
        order = torch.argsort(score_cpu, descending=True)
        occupied = self._build_voxel_index()
        anchor_xyz_cpu = self.map.get_xyz.detach().cpu().float() if self.map.anchor_count() > 0 else torch.zeros(0, 3)
        kept: list[int] = []
        kept_missing = 0
        kept_depth_mismatch = 0
        missing_budget = int(self.max_missing_seeds_per_keyframe)
        depth_budget = int(self.max_depth_mismatch_seeds_per_keyframe)
        for seed_idx in order.tolist():
            key = (0, int(grid_cpu[seed_idx, 0]), int(grid_cpu[seed_idx, 1]), int(grid_cpu[seed_idx, 2]))
            hit, near_hit = self._find_pfgs360_hash_hit(
                occupied,
                key,
                xyz_cpu[seed_idx],
                anchor_xyz_cpu,
            )
            if hit is not None:
                if int(hit) >= 0:
                    self._accumulate_existing_observation(
                        int(hit),
                        float(conf_cpu[seed_idx]),
                        frame_id=int(seeds.frame_id),
                    )
                    if near_hit:
                        stats["hash_near_hits"] += 1
                    else:
                        stats["hash_hits"] += 1
                stats["skipped_voxel"] += 1
                continue
            if not bool(insert_enabled[seed_idx]):
                stats["suppressed_insert"] += 1
                continue
            is_depth_mismatch_seed = bool(depth_seed_mask[seed_idx])
            is_missing_seed = bool(missing_seed_mask[seed_idx]) and not is_depth_mismatch_seed
            if is_depth_mismatch_seed and depth_budget > 0 and kept_depth_mismatch >= depth_budget:
                stats["skipped_depth_mismatch_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if is_missing_seed and missing_budget > 0 and kept_missing >= missing_budget:
                stats["skipped_missing_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if len(kept) >= budget:
                stats["skipped_budget"] += 1
                continue
            kept.append(int(seed_idx))
            if is_depth_mismatch_seed:
                kept_depth_mismatch += 1
            elif is_missing_seed:
                kept_missing += 1
            occupied[key] = -1
        if not kept:
            return self._empty_seed_like(seeds), stats
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), stats

    def _filter_replace_fuse_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        frontend_output: FrontendOutput | None,
        image: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
    ) -> tuple[GaussianSeedBatch, dict[str, int]]:
        seeds = self._with_pfgs360_seed_metadata(seeds)
        per_keyframe_budget = self.first_keyframe_max_seeds if self.stats.n_keyframes == 0 else self.keyframe_max_seeds
        budget = len(seeds) if per_keyframe_budget <= 0 else min(len(seeds), int(per_keyframe_budget))
        if self.global_anchor_budget > 0:
            budget = min(budget, max(0, int(self.global_anchor_budget) - self.map.anchor_count()))
        budget = max(0, int(budget))

        stats = {
            "skipped_voxel": 0,
            "skipped_budget": 0,
            "hash_hits": 0,
            "hash_near_hits": 0,
            "suppressed_insert": 0,
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "replace_deleted": 0,
            "fused": 0,
            "compacted": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
            "missing_seed_candidates": 0,
            "depth_mismatch_seed_candidates": 0,
            "skipped_missing_budget": 0,
            "skipped_depth_mismatch_budget": 0,
        }
        current_kf_ord = int(self.stats.n_keyframes)
        if self.map.anchor_count() > 0:
            stats["compacted"] += self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
        insert_enabled = (
            torch.ones(len(seeds), dtype=torch.bool)
            if seeds.insert_enabled is None
            else seeds.insert_enabled.detach().cpu().bool()
        )
        first_keyframe = self.stats.n_keyframes == 0 or self.map.anchor_count() == 0
        insert_seed_mask = torch.ones(len(seeds), dtype=torch.bool)
        depth_seed_mask = torch.zeros(len(seeds), dtype=torch.bool)
        if first_keyframe and frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            self.last_depth_insertion_diagnostic = self._prediction_depth_insertion_diagnostic(
                frontend_output,
                image_hw,
            )
        elif frontend_output is not None and image is not None:
            image_hw = tuple(int(v) for v in image.shape[-2:])
            render_masks, evidence_stats = self._pfgs360_replace_fuse_masks_and_delete(
                frontend_output,
                image,
                sky_mask=sky_mask,
            )
            stats.update({key: int(stats.get(key, 0)) + int(value) for key, value in evidence_stats.items()})
            if render_masks is not None:
                insert_seed_mask = self._sample_seed_mask(render_masks["insert"], seeds).detach().cpu().bool()
                depth_seed_mask = self._sample_seed_mask(render_masks["delete"], seeds).detach().cpu().bool()
                insert_enabled &= insert_seed_mask
                stats["depth_mismatch_seed_candidates"] = int(depth_seed_mask.sum().item())
                stats["missing_seed_candidates"] = int((insert_seed_mask & ~depth_seed_mask).sum().item())

        score = seeds.insert_score if seeds.insert_score is not None else seeds.confidence
        score_cpu = score.detach().cpu().float()
        conf_cpu = seeds.confidence.detach().cpu().float()
        xyz_cpu = seeds.xyz.detach().cpu().float()
        grid_cpu = torch.floor(xyz_cpu / float(self.pfgs360_voxel_size)).to(torch.int32)
        order = torch.argsort(score_cpu, descending=True)
        occupied = self._build_replace_fuse_voxel_index()
        anchor_xyz_cpu = self.map.get_xyz.detach().cpu().float() if self.map.anchor_count() > 0 else torch.zeros(0, 3)
        kept: list[int] = []
        kept_depth_mismatch = 0
        kept_insert_only = 0
        depth_budget = int(self.max_depth_mismatch_seeds_per_keyframe)
        insert_only_budget = 0 if first_keyframe else int(self.max_missing_seeds_per_keyframe)
        for seed_idx in order.tolist():
            key = (0, int(grid_cpu[seed_idx, 0]), int(grid_cpu[seed_idx, 1]), int(grid_cpu[seed_idx, 2]))
            hits = occupied.get(key, [])
            valid_hits = [int(hit) for hit in hits if int(hit) >= 0]
            if valid_hits:
                hit = self._select_replace_fuse_hit(valid_hits, xyz_cpu[seed_idx], anchor_xyz_cpu)
                self._fuse_seed_into_anchor(hit, seeds, int(seed_idx), current_kf_ord=current_kf_ord)
                stats["hash_hits"] += 1
                stats["skipped_voxel"] += 1
                stats["fused"] += 1
                if int(hit) < int(anchor_xyz_cpu.shape[0]):
                    anchor_xyz_cpu[hit] = self.map.get_xyz.detach().cpu().float()[hit]
                continue
            if hits:
                stats["skipped_voxel"] += 1
                continue
            if not bool(insert_enabled[seed_idx]):
                stats["suppressed_insert"] += 1
                continue
            is_delete_band_seed = bool(depth_seed_mask[seed_idx])
            if is_delete_band_seed and depth_budget > 0 and kept_depth_mismatch >= depth_budget:
                stats["skipped_depth_mismatch_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if (not is_delete_band_seed) and insert_only_budget > 0 and kept_insert_only >= insert_only_budget:
                stats["skipped_missing_budget"] += 1
                stats["skipped_budget"] += 1
                continue
            if len(kept) >= budget:
                stats["skipped_budget"] += 1
                continue
            kept.append(int(seed_idx))
            if is_delete_band_seed:
                kept_depth_mismatch += 1
            else:
                kept_insert_only += 1
            occupied[key] = [-1]
        if not kept:
            return self._empty_seed_like(seeds), stats
        keep_idx = torch.tensor(kept, dtype=torch.long, device=seeds.xyz.device)
        return self._subset_seeds(seeds, keep_idx), stats

    def _with_pfgs360_seed_metadata(self, seeds: GaussianSeedBatch) -> GaussianSeedBatch:
        n = len(seeds)
        device = seeds.xyz.device
        dtype = seeds.xyz.dtype
        grid = (
            seeds.grid_coord.to(device=device, dtype=torch.int32)
            if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == n
            else torch.floor(seeds.xyz.detach() / float(self.pfgs360_voxel_size)).to(device=device, dtype=torch.int32)
        )
        return GaussianSeedBatch(
            xyz=seeds.xyz,
            rgb=seeds.rgb,
            confidence=seeds.confidence,
            scale=(
                seeds.scale.to(device=device, dtype=dtype)
                if self.pfgs360_replace_fuse_enabled
                else torch.full((n,), float(self.pfgs360_voxel_size), device=device, dtype=dtype)
            ),
            level=torch.zeros(n, dtype=torch.int8, device=device),
            frame_id=int(seeds.frame_id),
            source_flat_idx=seeds.source_flat_idx,
            source_hw=seeds.source_hw,
            insert_enabled=(
                torch.ones(n, dtype=torch.bool, device=device)
                if seeds.insert_enabled is None
                else seeds.insert_enabled.to(device=device, dtype=torch.bool)
            ),
            insert_score=(
                seeds.confidence.to(device=device, dtype=dtype).clamp(0.0, 1.0)
                if seeds.insert_score is None
                else seeds.insert_score.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            ),
            grid_coord=grid,
        )

    def _find_pfgs360_hash_hit(
        self,
        occupied: dict[tuple[int, int, int, int], int],
        key: tuple[int, int, int, int],
        candidate_xyz: torch.Tensor,
        anchor_xyz_cpu: torch.Tensor,
    ) -> tuple[int | None, bool]:
        hit = occupied.get(key)
        if hit is not None:
            return int(hit), False
        radius = int(self.pfgs360_near_grid_radius)
        if radius <= 0 or anchor_xyz_cpu.numel() == 0 or self.pfgs360_near_distance_factor <= 0.0:
            return None, False
        level, x, y, z = key
        best_row: int | None = None
        best_dist = float("inf")
        thresh = float(self.pfgs360_near_distance_factor) * float(self.pfgs360_voxel_size)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    row = occupied.get((level, x + dx, y + dy, z + dz))
                    if row is None or int(row) < 0 or int(row) >= int(anchor_xyz_cpu.shape[0]):
                        continue
                    dist = float(torch.linalg.norm(anchor_xyz_cpu[int(row)] - candidate_xyz).item())
                    if dist <= thresh and dist < best_dist:
                        best_dist = dist
                        best_row = int(row)
        return best_row, best_row is not None

    def _pfgs360_replace_fuse_masks_and_delete(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
        stats = {
            "replace_deleted": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
        }
        if self.map.anchor_count() == 0:
            return None, stats
        target = image.detach().to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        if target.ndim == 4:
            target = target[0]
        H, W = int(target.shape[-2]), int(target.shape[-1])
        target_depth, depth_conf = self._target_depth_from_output(frontend_output, (H, W), target.device, target.dtype)
        if target_depth is None:
            return None, stats
        with torch.no_grad():
            camera = PanoRenderCamera(
                image_height=H,
                image_width=W,
                c2w=frontend_output.pose_c2w.detach().to(device=target.device, dtype=target.dtype),
            )
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, sky_mask))
            render_depth = pkg.get("depth")
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not torch.is_tensor(render_depth):
                return None, stats
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(render_depth)
            render_depth = render_depth.to(device=target.device, dtype=target.dtype)
            alpha = alpha.to(device=target.device, dtype=target.dtype)
            valid = torch.isfinite(target_depth) & torch.isfinite(render_depth) & (target_depth > 1.0e-6) & (render_depth > 1.0e-6)
            if depth_conf is not None:
                valid = valid & (depth_conf.to(device=target.device, dtype=target.dtype) > 1.0e-6)
            scale, shift = self._robust_depth_scale_shift(target_depth, render_depth, valid & (alpha >= self.pfgs360_render_alpha_min))
            aligned_target = (target_depth * scale + shift).clamp_min(1.0e-6)
            valid_aligned = torch.isfinite(aligned_target) & torch.isfinite(render_depth) & (render_depth > 1.0e-6)
            rel = (aligned_target - render_depth).abs() / torch.maximum(aligned_target, render_depth).clamp_min(1.0e-6)
            missing = ~valid_aligned
            insert_mask = (valid_aligned & (rel >= float(self.replace_fuse_insert_rel_min))) | missing
            delete_mask = (
                valid_aligned
                & (rel >= float(self.replace_fuse_delete_rel_min))
                & (rel <= float(self.replace_fuse_delete_rel_max))
            )
            if sky_mask is not None:
                non_sky = ~self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)
                missing = missing & non_sky
                insert_mask = insert_mask & non_sky
                delete_mask = delete_mask & non_sky
            stats["missing_pixels"] = int(missing.sum().detach().cpu())
            stats["depth_mismatch_pixels"] = int(delete_mask.sum().detach().cpu())
            stats["render_bad_pixels"] = int(insert_mask.sum().detach().cpu())
            stats["replace_deleted"] = self._delete_responsible_replace_fuse_anchors(
                delete_mask.detach(),
                pkg,
                frontend_output,
                H,
                W,
            )
            masks = {
                "insert": insert_mask.detach().cpu().bool(),
                "delete": delete_mask.detach().cpu().bool(),
                "missing": missing.detach().cpu().bool(),
                "depth_mismatch": delete_mask.detach().cpu().bool(),
                "render_bad": insert_mask.detach().cpu().bool(),
            }
            self.last_depth_insertion_diagnostic = DepthInsertionDiagnostic(
                frame_id=int(frontend_output.frame_id),
                render_depth=render_depth.detach().cpu().float(),
                predicted_depth=aligned_target.detach().cpu().float(),
                rel_depth_error=rel.detach().cpu().float(),
                missing_mask=masks["missing"],
                depth_mismatch_mask=masks["depth_mismatch"],
                render_bad_mask=masks["render_bad"],
                depth_scale=float(scale.detach().cpu()),
                depth_shift=float(shift.detach().cpu()),
            )
            return masks, stats

    def _delete_responsible_replace_fuse_anchors(
        self,
        delete_mask: torch.Tensor,
        render_pkg: dict,
        frontend_output: FrontendOutput,
        H: int,
        W: int,
    ) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            return 0
        render_depth = render_pkg.get("depth")
        if not torch.is_tensor(render_depth):
            return 0
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        xyz = self.map.get_xyz.detach()
        c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        w2c = torch.linalg.inv(c2w)
        xyz_h = torch.cat([xyz, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        visibility = render_pkg.get("visibility_filter")
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            valid = valid & visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() == 0:
            return 0
        cam_rows = cam.index_select(0, rows)
        pixels = bearing_to_erp_pixel(cam_rows, int(H), int(W))
        ui = pixels[:, 0].round().long().remainder(int(W))
        vi = pixels[:, 1].round().long().clamp(0, int(H) - 1)
        delete_at_pixel = delete_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
        rd = render_depth.to(device=device, dtype=dtype)[0, vi, ui].clamp_min(1.0e-6)
        anchor_depth = dist.index_select(0, rows)
        tol = torch.maximum(
            rd.new_full(rd.shape, float(self.replace_fuse_front_depth_abs_tol)),
            rd * float(self.replace_fuse_front_depth_rel_tol),
        )
        front_surface = (anchor_depth - rd).abs() <= tol
        prune_rows = rows[delete_at_pixel & front_surface]
        if prune_rows.numel() == 0:
            return 0
        max_delete = int(self.replace_fuse_max_delete_per_keyframe)
        if max_delete > 0 and prune_rows.numel() > max_delete:
            prune_rows = prune_rows[:max_delete]
        prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
        prune_mask[prune_rows] = True
        deleted = self.map.prune_anchors(prune_mask)
        if deleted > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            self._refresh_pfgs360_voxel_cache(compact=False)
        return int(deleted)

    def _build_replace_fuse_voxel_index(self) -> dict[tuple[int, int, int, int], list[int]]:
        occupied: dict[tuple[int, int, int, int], list[int]] = {}
        if self.map.anchor_count() <= 0:
            return occupied
        self._refresh_pfgs360_voxel_cache(compact=False)
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            key = (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            occupied.setdefault(key, []).append(int(idx))
        return occupied

    def _refresh_pfgs360_voxel_cache(self, *, compact: bool) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            self.map._anchor_grid_coord = torch.zeros(0, 3, dtype=torch.int32)
            return 0
        grid = torch.floor(self.map.get_xyz.detach().cpu().float() / float(self.pfgs360_voxel_size)).to(torch.int32)
        self.map._anchor_grid_coord = grid
        if not compact:
            return 0
        return self._compact_replace_fuse_voxels()

    def _compact_replace_fuse_voxels(self) -> int:
        n = self.map.anchor_count()
        if n <= 1:
            return 0
        index = self._build_replace_fuse_voxel_index_no_refresh()
        duplicate_rows: list[int] = []
        opacity = self.map.get_opacity.detach().cpu().view(-1)
        obs = self.map._anchor_obs_count[:n].detach().cpu().float().clamp_min(1.0)
        score = opacity * obs
        for rows in index.values():
            if len(rows) <= 1:
                continue
            rows_t = torch.tensor(rows, dtype=torch.long)
            keeper = int(rows_t[torch.argmax(score.index_select(0, rows_t))])
            for row in rows:
                if int(row) == keeper:
                    continue
                self._merge_anchor_into_anchor(keeper, int(row))
                duplicate_rows.append(int(row))
        if not duplicate_rows:
            return 0
        prune_mask = torch.zeros(n, dtype=torch.bool, device=self.map.get_xyz.device)
        prune_mask[torch.tensor(duplicate_rows, dtype=torch.long, device=prune_mask.device)] = True
        pruned = self.map.prune_anchors(prune_mask)
        if pruned > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            self._refresh_pfgs360_voxel_cache(compact=False)
        return int(pruned)

    def _build_replace_fuse_voxel_index_no_refresh(self) -> dict[tuple[int, int, int, int], list[int]]:
        occupied: dict[tuple[int, int, int, int], list[int]] = {}
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            key = (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            occupied.setdefault(key, []).append(int(idx))
        return occupied

    @staticmethod
    def _select_replace_fuse_hit(
        hits: list[int],
        seed_xyz: torch.Tensor,
        anchor_xyz_cpu: torch.Tensor,
    ) -> int:
        if len(hits) == 1:
            return int(hits[0])
        rows = torch.tensor(hits, dtype=torch.long)
        dist = torch.linalg.norm(anchor_xyz_cpu.index_select(0, rows) - seed_xyz.view(1, 3), dim=-1)
        return int(rows[torch.argmin(dist)])

    def _fuse_seed_into_anchor(
        self,
        anchor_idx: int,
        seeds: GaussianSeedBatch,
        seed_idx: int,
        *,
        current_kf_ord: int,
    ) -> None:
        if int(anchor_idx) < 0 or int(anchor_idx) >= self.map.anchor_count():
            return
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        idx = int(anchor_idx)
        seed_xyz = seeds.xyz[int(seed_idx)].detach().to(device=device, dtype=dtype)
        seed_rgb = seeds.rgb[int(seed_idx)].detach().to(device=device, dtype=dtype).clamp(0.0, 1.0)
        seed_conf = float(seeds.confidence[int(seed_idx)].detach().cpu().clamp(1.0e-4, 1.0))
        seed_scale = seeds.scale[int(seed_idx)].detach().to(device=device, dtype=dtype).clamp_min(1.0e-5)
        old_w = float(self.map._anchor_conf_accum[idx]) if idx < int(self.map._anchor_conf_accum.shape[0]) else 1.0
        new_w = max(seed_conf, 1.0e-4)
        denom = max(old_w + new_w, 1.0e-6)
        with torch.no_grad():
            self.map.xyz.data[idx] = (self.map.xyz.data[idx] * old_w + seed_xyz * new_w) / denom
            old_rgb = self.map.get_features.detach()[idx]
            fused_rgb = (old_rgb * old_w + seed_rgb * new_w) / denom
            self.map.features.data[idx] = self.map._inv_sigmoid(fused_rgb.view(1, 3)).view(3)
            old_opacity = self.map.get_opacity.detach()[idx, 0]
            fused_opacity = torch.maximum(old_opacity, seed_conf * old_opacity.new_tensor(1.0))
            self.map.opacity_logit.data[idx, 0] = self.map._inv_sigmoid(fused_opacity.view(1, 1)).view(())
            old_scale = self.map.get_scaling.detach()[idx]
            fused_scale = torch.minimum(old_scale, seed_scale.expand_as(old_scale))
            self.map.scaling.data[idx] = torch.log(torch.expm1(fused_scale.clamp_min(1.0e-5)))
        self._accumulate_existing_observation(idx, seed_conf, frame_id=int(seeds.frame_id))
        if idx < int(self.map._anchor_last_update_kf_ord.shape[0]):
            self.map._anchor_last_update_kf_ord[idx] = int(current_kf_ord)
        if idx < int(self.map._anchor_voxel_size.shape[0]):
            self.map._anchor_voxel_size[idx] = float(seed_scale.detach().cpu())

    def _merge_anchor_into_anchor(self, keeper: int, other: int) -> None:
        if keeper == other or keeper < 0 or other < 0:
            return
        if keeper >= self.map.anchor_count() or other >= self.map.anchor_count():
            return
        device = self.map.get_xyz.device
        with torch.no_grad():
            keep_w = float(self.map._anchor_conf_accum[keeper].clamp_min(1.0e-4))
            other_w = float(self.map._anchor_conf_accum[other].clamp_min(1.0e-4))
            denom = max(keep_w + other_w, 1.0e-6)
            self.map.xyz.data[keeper] = (self.map.xyz.data[keeper] * keep_w + self.map.xyz.data[other] * other_w) / denom
            rgb = (self.map.get_features.detach()[keeper] * keep_w + self.map.get_features.detach()[other] * other_w) / denom
            self.map.features.data[keeper] = self.map._inv_sigmoid(rgb.view(1, 3)).view(3)
            opacity = torch.maximum(self.map.get_opacity.detach()[keeper], self.map.get_opacity.detach()[other])
            self.map.opacity_logit.data[keeper] = self.map._inv_sigmoid(opacity.view(1, 1)).view(1)
            scale = torch.minimum(self.map.get_scaling.detach()[keeper], self.map.get_scaling.detach()[other])
            self.map.scaling.data[keeper] = torch.log(torch.expm1(scale.clamp_min(1.0e-5)))
        self.map._anchor_obs_count[keeper] += self.map._anchor_obs_count[other]
        self.map._anchor_conf_accum[keeper] += self.map._anchor_conf_accum[other]
        self.map._anchor_last_seen_kf[keeper] = max(
            int(self.map._anchor_last_seen_kf[keeper]),
            int(self.map._anchor_last_seen_kf[other]),
        )
        if self.map._anchor_last_update_kf_ord.shape[0] > max(keeper, other):
            self.map._anchor_last_update_kf_ord[keeper] = max(
                int(self.map._anchor_last_update_kf_ord[keeper]),
                int(self.map._anchor_last_update_kf_ord[other]),
            )
        if self.map._anchor_voxel_size.shape[0] > max(keeper, other):
            self.map._anchor_voxel_size[keeper] = min(
                float(self.map._anchor_voxel_size[keeper]),
                float(self.map._anchor_voxel_size[other]),
            )

    def _pfgs360_render_bad_mask(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
        temporal_ok: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
        stats = {
            "outlier_resets": 0,
            "outlier_pruned": 0,
            "missing_pixels": 0,
            "depth_mismatch_pixels": 0,
            "render_bad_pixels": 0,
        }
        if self.map.anchor_count() == 0:
            return None, stats
        target = image.detach().to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        if target.ndim == 4:
            target = target[0]
        H, W = int(target.shape[-2]), int(target.shape[-1])
        target_depth, depth_conf = self._target_depth_from_output(frontend_output, (H, W), target.device, target.dtype)
        if target_depth is None:
            return None, stats
        with torch.no_grad():
            camera = PanoRenderCamera(
                image_height=H,
                image_width=W,
                c2w=frontend_output.pose_c2w.detach().to(device=target.device, dtype=target.dtype),
            )
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, sky_mask))
            render_depth = pkg.get("depth")
            render_rgb = pkg.get("render")
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not (torch.is_tensor(render_depth) and torch.is_tensor(render_rgb)):
                return None, stats
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(render_depth)
            render_depth = render_depth.to(device=target.device, dtype=target.dtype)
            alpha = alpha.to(device=target.device, dtype=target.dtype)
            valid = torch.isfinite(target_depth) & torch.isfinite(render_depth) & (target_depth > 1.0e-6) & (render_depth > 1.0e-6)
            if depth_conf is not None:
                valid = valid & (depth_conf.to(device=target.device, dtype=target.dtype) > 1.0e-6)
            scale, shift = self._robust_depth_scale_shift(target_depth, render_depth, valid & (alpha >= self.pfgs360_render_alpha_min))
            aligned_target = (target_depth * scale + shift).clamp_min(1.0e-6)
            valid_aligned = torch.isfinite(aligned_target) & torch.isfinite(render_depth) & (render_depth > 1.0e-6)
            missing_alpha_min = max(0.0, min(1.0, float(self.pfgs360_missing_alpha_min)))
            missing = (alpha < missing_alpha_min) | (~valid_aligned)
            rel = (aligned_target - render_depth).abs() / torch.maximum(aligned_target, render_depth).clamp_min(1.0e-6)
            depth_mismatch = valid_aligned & (rel > self.pfgs360_render_depth_rel_threshold)
            render_bad = missing | depth_mismatch
            if sky_mask is not None:
                non_sky = ~self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)
                missing = missing & non_sky
                depth_mismatch = depth_mismatch & non_sky
                render_bad = render_bad & non_sky
            stats["missing_pixels"] = int(missing.sum().detach().cpu())
            stats["depth_mismatch_pixels"] = int(depth_mismatch.sum().detach().cpu())
            stats["render_bad_pixels"] = int(render_bad.sum().detach().cpu())
            evidence_stats = self._update_pfgs360_outlier_evidence(
                render_bad.detach(),
                pkg,
                frontend_output,
                H,
                W,
                temporal_ok=temporal_ok,
            )
            stats.update(evidence_stats)
            masks = {
                "render_bad": render_bad.detach().cpu().bool(),
                "missing": missing.detach().cpu().bool(),
                "depth_mismatch": depth_mismatch.detach().cpu().bool(),
            }
            self.last_depth_insertion_diagnostic = DepthInsertionDiagnostic(
                frame_id=int(frontend_output.frame_id),
                render_depth=render_depth.detach().cpu().float(),
                predicted_depth=aligned_target.detach().cpu().float(),
                rel_depth_error=rel.detach().cpu().float(),
                missing_mask=masks["missing"],
                depth_mismatch_mask=masks["depth_mismatch"],
                render_bad_mask=masks["render_bad"],
                depth_scale=float(scale.detach().cpu()),
                depth_shift=float(shift.detach().cpu()),
            )
            return masks, stats

    def _prediction_depth_insertion_diagnostic(
        self,
        frontend_output: FrontendOutput,
        image_hw: tuple[int, int],
    ) -> DepthInsertionDiagnostic | None:
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        target_depth, _ = self._target_depth_from_output(frontend_output, image_hw, device, dtype)
        if target_depth is None:
            return None
        return DepthInsertionDiagnostic(
            frame_id=int(frontend_output.frame_id),
            render_depth=None,
            predicted_depth=target_depth.detach().cpu().float(),
            rel_depth_error=None,
            missing_mask=None,
            depth_mismatch_mask=None,
            render_bad_mask=None,
            depth_scale=1.0,
            depth_shift=0.0,
        )

    def _target_depth_from_output(
        self,
        frontend_output: FrontendOutput,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._target_depth_from_tensors(
            inverse_depth=frontend_output.inverse_depth,
            world_points=frontend_output.world_points,
            pose_c2w=frontend_output.pose_c2w,
            confidence=(
                frontend_output.depth_confidence
                if frontend_output.depth_confidence is not None
                else frontend_output.world_points_confidence
            ),
            size=size,
            device=device,
            dtype=dtype,
        )

    def _target_depth_from_tensors(
        self,
        *,
        inverse_depth: torch.Tensor | None,
        world_points: torch.Tensor | None,
        pose_c2w: torch.Tensor,
        confidence: torch.Tensor | None,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        H, W = int(size[0]), int(size[1])
        target_depth: torch.Tensor | None = None
        if inverse_depth is not None:
            inv = inverse_depth.detach().float()
            if inv.ndim == 2:
                inv = inv.unsqueeze(0)
            target_depth = inv.clamp_min(1.0e-6).reciprocal()
        elif world_points is not None:
            pts = world_points.detach().float()
            if pts.ndim == 4 and int(pts.shape[0]) == 1:
                pts = pts[0]
            if pts.ndim == 3 and int(pts.shape[-1]) == 3:
                c2w = pose_c2w.detach().float()
                center = c2w[:3, 3].view(1, 1, 3)
                target_depth = torch.linalg.norm(pts - center, dim=-1, keepdim=False).unsqueeze(0)
        if target_depth is None:
            return None, None
        if tuple(target_depth.shape[-2:]) != (H, W):
            target_depth = F.interpolate(target_depth.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
        target_depth = target_depth.to(device=device, dtype=dtype)
        conf_t = None
        if confidence is not None:
            conf_t = confidence.detach().float()
            if conf_t.ndim == 2:
                conf_t = conf_t.unsqueeze(0)
            if tuple(conf_t.shape[-2:]) != (H, W):
                conf_t = F.interpolate(conf_t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            conf_t = conf_t.to(device=device, dtype=dtype)
        return target_depth, conf_t

    @staticmethod
    def _robust_depth_scale_shift(
        target_depth: torch.Tensor,
        render_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        if values.numel() < 16:
            one = render_depth.new_tensor(1.0)
            zero = render_depth.new_tensor(0.0)
            return one, zero
        td = target_depth.reshape(-1).index_select(0, values).clamp_min(1.0e-6)
        rd = render_depth.reshape(-1).index_select(0, values).clamp_min(1.0e-6)
        scale = torch.median(rd / td).clamp(0.05, 20.0)
        shift = torch.median(rd - scale * td)
        return scale, shift

    @staticmethod
    def _sample_seed_mask(mask: torch.Tensor, seeds: GaussianSeedBatch) -> torch.Tensor:
        if seeds.source_flat_idx is None or seeds.source_hw is None:
            return torch.ones(len(seeds), dtype=torch.bool)
        flat = mask.detach().bool().reshape(-1)
        idx = seeds.source_flat_idx.detach().cpu().long()
        valid = (idx >= 0) & (idx < int(flat.shape[0]))
        out = torch.ones(len(seeds), dtype=torch.bool)
        if bool(valid.any()):
            out[valid] = flat.index_select(0, idx[valid])
        return out

    @staticmethod
    def _seed_pixel_mask(
        seeds: GaussianSeedBatch,
        flags: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if seeds.source_flat_idx is None or seeds.source_hw is None:
            return None
        H, W = int(image_hw[0]), int(image_hw[1])
        if tuple(seeds.source_hw) != (H, W):
            return None
        idx = seeds.source_flat_idx.detach().cpu().long()
        valid = (idx >= 0) & (idx < H * W)
        if not bool(valid.any()):
            return None
        out = torch.zeros(1, H, W, dtype=torch.bool)
        out.view(-1)[idx[valid]] = flags.detach().cpu().bool()[valid]
        return out

    def _update_pfgs360_outlier_evidence(
        self,
        render_bad: torch.Tensor,
        render_pkg: dict,
        frontend_output: FrontendOutput,
        H: int,
        W: int,
        *,
        temporal_ok: torch.Tensor | None = None,
    ) -> dict[str, int]:
        stats = {"outlier_resets": 0, "outlier_pruned": 0}
        n = self.map.anchor_count()
        if n <= 0 or self.map._anchor_outlier_obs.shape[0] != n:
            return stats
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        xyz = self.map.get_xyz.detach()
        c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        w2c = torch.linalg.inv(c2w)
        xyz_h = torch.cat([xyz, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        visibility = render_pkg.get("visibility_filter")
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            valid = valid & visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() == 0:
            return stats
        pixels = bearing_to_erp_pixel(cam.index_select(0, rows), H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        bad = render_bad.to(device=device, dtype=torch.bool)[0, vi, ui]
        if temporal_ok is not None:
            temporal = temporal_ok.to(device=device, dtype=torch.bool)[0, vi, ui]
            rows = rows.index_select(0, torch.nonzero(temporal, as_tuple=False).flatten())
            bad = bad[temporal]
            if rows.numel() == 0:
                return stats
        rows_cpu = rows.detach().cpu()
        bad_cpu = bad.detach().cpu()
        if bool((~bad_cpu).any()):
            self.map._anchor_inlier_obs[rows_cpu[~bad_cpu]] += 1
        if bool(bad_cpu.any()):
            outlier_rows = rows_cpu[bad_cpu]
            self.map._anchor_outlier_obs[outlier_rows] += 1
            self.map._anchor_last_seen_kf[outlier_rows] = int(frontend_output.frame_id)
        reset_rows = torch.zeros(0, dtype=torch.long)
        if self.pfgs360_reset_after_outliers > 0:
            reset_rows = torch.nonzero(
                self.map._anchor_outlier_obs >= int(self.pfgs360_reset_after_outliers),
                as_tuple=False,
            ).flatten()
            if reset_rows.numel() > 0:
                rows_dev = reset_rows.to(device=device)
                low_opacity = torch.full((int(rows_dev.numel()), 1), 0.01, device=device, dtype=dtype)
                with torch.no_grad():
                    self.map.opacity_logit.data[rows_dev] = torch.minimum(
                        self.map.opacity_logit.data[rows_dev],
                        self.map._inv_sigmoid(low_opacity),
                    )
                stats["outlier_resets"] = int(reset_rows.numel())
        if self.pfgs360_prune_after_outliers > 0 and self.pfgs360_max_prune_per_keyframe > 0:
            old_enough = self.map._anchor_birth_frame <= int(frontend_output.frame_id) - int(self.pfgs360_protect_recent_keyframes)
            prune_rows = torch.nonzero(
                (self.map._anchor_outlier_obs >= int(self.pfgs360_prune_after_outliers)) & old_enough,
                as_tuple=False,
            ).flatten()
            if prune_rows.numel() > self.pfgs360_max_prune_per_keyframe:
                prune_rows = prune_rows[: self.pfgs360_max_prune_per_keyframe]
            if prune_rows.numel() > 0:
                prune_mask = torch.zeros(n, dtype=torch.bool)
                prune_mask[prune_rows] = True
                stats["outlier_pruned"] = self.map.prune_anchors(prune_mask.to(device=device))
        return stats

    def _build_voxel_index(self) -> dict[tuple[int, int, int, int], int]:
        occupied: dict[tuple[int, int, int, int], int] = {}
        if self.map._anchor_grid_coord.numel() == 0:
            return occupied
        levels = self.map._anchor_level.detach().cpu().tolist()
        coords = self.map._anchor_grid_coord.detach().cpu().tolist()
        for idx, (level, coord) in enumerate(zip(levels, coords)):
            occupied.setdefault((int(level), int(coord[0]), int(coord[1]), int(coord[2])), int(idx))
        return occupied

    @staticmethod
    def _seed_voxel_key_from_cpu(
        xyz_cpu: torch.Tensor,
        scale_cpu: torch.Tensor,
        level_cpu: torch.Tensor,
        seed_idx: int,
    ) -> tuple[int, int, int, int]:
        level = int(level_cpu[seed_idx])
        scale = float(scale_cpu[seed_idx])
        coord = torch.floor(xyz_cpu[seed_idx] / scale).to(torch.int32)
        return (level, int(coord[0]), int(coord[1]), int(coord[2]))

    def _find_voxel_hit(
        self,
        occupied: dict[tuple[int, int, int, int], int],
        key: tuple[int, int, int, int],
        *,
        radius: int | None = None,
    ) -> int | None:
        level, x, y, z = key
        radius = int(self.voxel_neighbor_radius if radius is None else radius)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    hit = occupied.get((level, x + dx, y + dy, z + dz))
                    if hit is not None:
                        return int(hit)
        return None

    def _accumulate_existing_observation(self, anchor_idx: int, confidence: float, *, frame_id: int | None = None) -> None:
        if int(anchor_idx) < 0:
            return
        if int(anchor_idx) >= int(self.map._anchor_obs_count.shape[0]):
            return
        self.map._anchor_obs_count[int(anchor_idx)] += 1
        self.map._anchor_conf_accum[int(anchor_idx)] += float(confidence)
        if frame_id is not None and int(anchor_idx) < int(self.map._anchor_last_seen_kf.shape[0]):
            self.map._anchor_last_seen_kf[int(anchor_idx)] = int(frame_id)

    @staticmethod
    def _subset_seeds(seeds: GaussianSeedBatch, keep_idx: torch.Tensor) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=seeds.xyz.index_select(0, keep_idx.to(device=seeds.xyz.device)),
            rgb=seeds.rgb.index_select(0, keep_idx.to(device=seeds.rgb.device)),
            confidence=seeds.confidence.index_select(0, keep_idx.to(device=seeds.confidence.device)),
            scale=seeds.scale.index_select(0, keep_idx.to(device=seeds.scale.device)),
            level=seeds.level.index_select(0, keep_idx.to(device=seeds.level.device)),
            frame_id=int(seeds.frame_id),
            source_flat_idx=(
                None
                if seeds.source_flat_idx is None
                else seeds.source_flat_idx.index_select(0, keep_idx.to(device=seeds.source_flat_idx.device))
            ),
            source_hw=seeds.source_hw,
            insert_enabled=(
                None
                if seeds.insert_enabled is None
                else seeds.insert_enabled.index_select(0, keep_idx.to(device=seeds.insert_enabled.device))
            ),
            insert_score=(
                None
                if seeds.insert_score is None
                else seeds.insert_score.index_select(0, keep_idx.to(device=seeds.insert_score.device))
            ),
            grid_coord=(
                None
                if seeds.grid_coord is None
                else seeds.grid_coord.index_select(0, keep_idx.to(device=seeds.grid_coord.device))
            ),
        )

    @staticmethod
    def _empty_seed_like(seeds: GaussianSeedBatch) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=seeds.xyz[:0],
            rgb=seeds.rgb[:0],
            confidence=seeds.confidence[:0],
            scale=seeds.scale[:0],
            level=seeds.level[:0],
            frame_id=int(seeds.frame_id),
            source_flat_idx=None if seeds.source_flat_idx is None else seeds.source_flat_idx[:0],
            source_hw=seeds.source_hw,
            insert_enabled=None if seeds.insert_enabled is None else seeds.insert_enabled[:0],
            insert_score=None if seeds.insert_score is None else seeds.insert_score[:0],
            grid_coord=None if seeds.grid_coord is None else seeds.grid_coord[:0],
        )

    def _register_keyframe(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        start: int,
        end: int,
        sky_mask: torch.Tensor | None = None,
    ) -> None:
        frame_id = int(frontend_output.frame_id)
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        base_c2w = frontend_output.pose_c2w.detach().to(device=device, dtype=dtype)
        self.pose_deltas[frame_id] = PoseDelta(base_c2w).to(device=device)
        record = MapperKeyframe(
            frame_id=frame_id,
            image=image.detach().cpu().float(),
            gaussian_start=int(start),
            gaussian_end=int(end),
            sky_mask=sky_mask.detach().cpu().bool() if torch.is_tensor(sky_mask) else None,
            target_depth=self._keyframe_target_depth(frontend_output, image, sky_mask=sky_mask),
            depth_confidence=self._keyframe_depth_confidence(frontend_output, image, sky_mask=sky_mask),
        )
        self.keyframes = [kf for kf in self.keyframes if kf.frame_id != frame_id]
        self.keyframes.append(record)

    def _keyframe_target_depth(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.pfgs360_insertion_enabled:
            return None
        img = image.detach()
        if img.ndim == 4:
            img = img[0]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        depth, _ = self._target_depth_from_output(frontend_output, (H, W), torch.device("cpu"), torch.float32)
        if depth is None:
            return None
        if sky_mask is not None:
            mask = self._normalize_skybox_mask(sky_mask, height=H, width=W, device=torch.device("cpu"))
            depth = depth.masked_fill(mask.bool(), 0.0)
        return depth.detach().cpu().float()

    def _keyframe_depth_confidence(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.pfgs360_insertion_enabled:
            return None
        img = image.detach()
        if img.ndim == 4:
            img = img[0]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        _, conf = self._target_depth_from_output(frontend_output, (H, W), torch.device("cpu"), torch.float32)
        if conf is None:
            conf = torch.ones(1, H, W, dtype=torch.float32)
        if sky_mask is not None:
            mask = self._normalize_skybox_mask(sky_mask, height=H, width=W, device=torch.device("cpu"))
            conf = conf.masked_fill(mask.bool(), 0.0)
        return conf.detach().cpu().float()

    def refined_pose_c2w(self, frame_id: int) -> torch.Tensor | None:
        pose_delta = self.pose_deltas.get(int(frame_id))
        if pose_delta is None:
            return None
        return pose_delta().detach().cpu()

    def refined_keyframe_poses(self) -> list[tuple[int, torch.Tensor]]:
        out = []
        for keyframe in self.keyframes:
            pose = self.refined_pose_c2w(keyframe.frame_id)
            if pose is not None:
                out.append((int(keyframe.frame_id), pose))
        return out

    def apply_frontend_pose_updates(self, updates: dict[int, torch.Tensor]) -> int:
        """Replace registered keyframe pose bases with frontend graph updates."""

        if not updates:
            return 0
        device = self.map.get_xyz.device
        registered = {int(keyframe.frame_id) for keyframe in self.keyframes}
        applied = 0
        for frame_id, pose in updates.items():
            fid = int(frame_id)
            if fid not in registered:
                continue
            pose_t = pose.detach().to(device=device, dtype=self.map.get_xyz.dtype)
            if tuple(pose_t.shape) != (4, 4) or not torch.isfinite(pose_t).all():
                continue
            self.pose_deltas[fid] = PoseDelta(pose_t).to(device=device)
            applied += 1
        if applied > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        return applied

    def render_view(self, *, image: torch.Tensor, c2w: torch.Tensor) -> dict | None:
        if self.map.anchor_count() == 0 and not self.map.has_skybox:
            return None
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        with torch.no_grad():
            pkg = self.renderer.render(camera, self.map)
            return self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target))

    def _skybox_optimization_mask_enabled(self) -> bool:
        return bool(self._skybox_mask_enabled() and getattr(self.map, "skybox_optimize", False))

    def _skybox_mask_enabled(self) -> bool:
        return bool(
            self.map.has_skybox
            and getattr(self.map, "skybox_optimization_mask_enable", True)
        )

    def _skybox_mask_from_image(self, image: torch.Tensor | None) -> torch.Tensor | None:
        if image is None or not self._skybox_mask_enabled():
            return None
        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.ndim != 3 or int(img.shape[0]) != 3:
            return None
        mask = self.map._sky_mask_from_image(img.clamp(0.0, 1.0))
        return mask.detach().bool().view(1, int(img.shape[-2]), int(img.shape[-1]))

    @staticmethod
    def _normalize_skybox_mask(
        sky_mask: torch.Tensor,
        *,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = sky_mask.detach().bool()
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected sky mask as HxW, 1xHxW, or Bx1xHxW, got {tuple(sky_mask.shape)}")
        if tuple(mask.shape[-2:]) != (int(height), int(width)):
            mask = (
                F.interpolate(
                    mask.float().view(1, 1, *mask.shape[-2:]),
                    size=(int(height), int(width)),
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
        return mask.to(device=device).view(1, int(height), int(width))

    def _skybox_mask_for_target(
        self,
        target: torch.Tensor,
        sky_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._skybox_mask_enabled():
            return None
        H, W = int(target.shape[-2]), int(target.shape[-1])
        if sky_mask is None:
            sky_mask = self._skybox_mask_from_image(target)
        if sky_mask is None:
            return None
        return self._normalize_skybox_mask(sky_mask, height=H, width=W, device=target.device)

    def _apply_skybox_optimization_mask(
        self,
        render_pkg: dict,
        sky_mask: torch.Tensor | None,
    ) -> dict:
        if sky_mask is None or not self._skybox_mask_enabled():
            return render_pkg
        gs_rgb = render_pkg.get("gs_only")
        sky_rgb = render_pkg.get("sky_bg_only")
        trans = render_pkg.get("sky_bg_alpha")
        if not (torch.is_tensor(gs_rgb) and torch.is_tensor(sky_rgb) and torch.is_tensor(trans)):
            return render_pkg
        if sky_rgb.ndim != 3 or trans.ndim != 3:
            return render_pkg
        mask = self._normalize_skybox_mask(
            sky_mask,
            height=int(sky_rgb.shape[-2]),
            width=int(sky_rgb.shape[-1]),
            device=sky_rgb.device,
        ).to(dtype=sky_rgb.dtype)
        sky_rgb_masked = sky_rgb * mask
        out = dict(render_pkg)
        if bool(getattr(self.map, "skybox_force_sky_render", False)):
            out["render"] = (gs_rgb * (1.0 - mask) + sky_rgb_masked).clamp(0.0, 1.0)
            out["skybox_force_sky_render"] = True
        else:
            out["render"] = (gs_rgb + trans.to(sky_rgb) * sky_rgb_masked).clamp(0.0, 1.0)
            out["skybox_force_sky_render"] = False
        out["skybox_optimization_mask"] = mask
        return out

    def render_keyframe_diagnostic(self, frame_id: int) -> KeyframeRenderDiagnostic | None:
        """Render an optimized keyframe for post-optimization diagnostics."""

        if self.map.anchor_count() == 0 and not self.map.has_skybox:
            return None
        frame_id = int(frame_id)
        keyframe = next((kf for kf in self.keyframes if int(kf.frame_id) == frame_id), None)
        pose_delta = self.pose_deltas.get(frame_id)
        if keyframe is None or pose_delta is None:
            return None
        target = keyframe.image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        with torch.no_grad():
            camera = PanoRenderCamera(image_height=H, image_width=W, c2w=pose_delta().detach())
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, self._skybox_mask_for_target(target, keyframe.sky_mask))
            loss, _ = backend_render_loss(pkg, target, weights=self.loss_weights)
            render = pkg["render"].detach()
            mse = torch.mean((render - target).square()).clamp_min(1e-12)
            psnr = -10.0 * torch.log10(mse)
            depth = pkg.get("depth")
            return KeyframeRenderDiagnostic(
                frame_id=frame_id,
                target=target.detach().cpu(),
                render=render.cpu(),
                depth=depth.detach().cpu() if torch.is_tensor(depth) else None,
                loss=float(loss.detach().cpu()),
                psnr=float(psnr.detach().cpu()),
                anchor_count=self.map.anchor_count(),
                phase=self.stats.last_phase,
            )

    def refine_on_keyframe(
        self,
        *,
        image: torch.Tensor,
        c2w: torch.Tensor,
        steps: int = 1,
    ) -> dict[str, float]:
        if (self.map.anchor_count() == 0 and not self.map.has_skybox) or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        sky_mask = self._skybox_mask_for_target(target)
        last = {"loss": 0.0}
        for _ in range(int(steps)):
            self.optimizer.zero_grad(set_to_none=True)
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
            loss, metrics = backend_render_loss(pkg, target, weights=self.loss_weights)
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = "legacy_keyframe"
        self.stats.optimization_steps += int(steps)
        return last

    def optimize_feedforward_window(
        self,
        *,
        current_frame_ids,
        history_frame_ids=None,
    ) -> dict[str, float]:
        if not self.uses_joint_optimization or not self.feedforward_window_enabled:
            return {}
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            steps = int(self.optim_cfg.get("steps_per_chunk", cfg.get("steps", 200)))
        else:
            steps = int(cfg.get("steps", self.optim_cfg.get("sliding_window_steps", 0)))
        window_ids = self._feedforward_window_ids(current_frame_ids, history_frame_ids)
        observations = self._selected_observations_for_ids(window_ids)
        if not observations:
            return {"loss": 0.0, "steps": 0.0, "window_size": 0.0}
        if not bool(cfg.get("optimize_non_keyframe_observations", True)):
            observations = [obs for obs in observations if bool(obs.is_keyframe)]
        if not observations:
            return {"loss": 0.0, "steps": 0.0, "window_size": 0.0}
        total_start = time.perf_counter()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        selected_keyframe_ids = self._feedforward_keyframe_ids(window_ids)
        current_keyframe_ids = self._feedforward_keyframe_ids(current_frame_ids)
        gaussian_scales = self._feedforward_gaussian_scales(selected_keyframe_ids)
        gaussian_enabled = (
            gaussian_scales is not None
            and self.map.anchor_count() > 0
            and bool((gaussian_scales > 0).any().detach().cpu())
        )
        pose_enabled = bool(self.optim_cfg.get("pose_refine_enable", False))
        trainable_pose_ids = self._feedforward_trainable_pose_ids(current_keyframe_ids, pose_enabled=pose_enabled)
        pose_params = [
            self.pose_deltas[fid].delta
            for fid in trainable_pose_ids
            if fid in self.pose_deltas
        ]
        param_groups = self._map_param_groups(gaussian_enabled=gaussian_enabled, phase="feedforward_window")
        if pose_params:
            param_groups.append({"params": pose_params, "lr": float(self.optim_cfg.get("pose_lr", 1e-3))})
        if not param_groups:
            sky_pruned = self._prune_sky_observations(observations) if self.pfgs360_replace_fuse_enabled else 0
            self.stats.last_sky_pruned = int(sky_pruned)
            return {
                "loss": 0.0,
                "steps": 0.0,
                "window_size": float(len(observations)),
                "feedforward_window_size": float(len(observations)),
                "sky_pruned": float(sky_pruned),
                "profile_backend_feedforward_window_sec": float(time.perf_counter() - total_start),
            }
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        pose_prior_weight = float(self.optim_cfg.get("pose_prior_weight", 1e-3))
        min_delta, patience = self._early_stop_options()
        best = float("inf")
        stale = 0
        actual_steps = 0
        last: dict[str, float] = {"loss": 0.0}
        last_sampled_ids: list[int] = []
        sample_sec = 0.0
        render_loss_sec = 0.0
        backward_step_sec = 0.0
        sky_pruned_total = 0
        for step_idx in range(max(0, steps)):
            optimizer.zero_grad(set_to_none=True)
            render_losses = []
            metric_accum: dict[str, list[torch.Tensor]] = {}
            sky_prune_mask = None
            section_start = time.perf_counter()
            sampled = self._sample_observations_for_step(observations)
            sample_sec += time.perf_counter() - section_start
            last_sampled_ids = [int(obs.frame_id) for obs in sampled]
            section_start = time.perf_counter()
            for obs in sampled:
                target = obs.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                c2w = self._observation_pose(obs, trainable_pose_ids=trainable_pose_ids).to(device=device, dtype=dtype)
                camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
                pkg = self.renderer.render(camera, self.map)
                sky_mask = self._skybox_mask_for_target(target, obs.sky_mask)
                pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
                if self.pfgs360_replace_fuse_enabled:
                    mask_i = self._sky_prune_mask_from_render_pkg(
                        obs,
                        pkg,
                        sky_mask,
                        c2w,
                        H,
                        W,
                    )
                    if mask_i is not None:
                        sky_prune_mask = mask_i if sky_prune_mask is None else (sky_prune_mask | mask_i)
                target_depth = None if obs.target_depth is None else obs.target_depth.to(device=device, dtype=dtype)
                depth_confidence = None if obs.depth_confidence is None else obs.depth_confidence.to(device=device, dtype=dtype)
                loss_i, metrics_i = backend_render_loss(
                    pkg,
                    target,
                    target_depth=target_depth,
                    depth_confidence=depth_confidence,
                    weights=self.loss_weights,
                )
                if sky_mask is not None:
                    metrics_i = dict(metrics_i)
                    metrics_i["skybox_mask_ratio"] = sky_mask.to(device=device, dtype=dtype).mean().detach()
                render_losses.append(loss_i)
                for key, value in metrics_i.items():
                    metric_accum.setdefault(key, []).append(value.detach())
            if not render_losses:
                break
            render_loss_sec += time.perf_counter() - section_start
            loss = torch.stack(render_losses).mean()
            if pose_params and pose_prior_weight > 0.0:
                prior = torch.stack([param.square().mean() for param in pose_params]).mean()
                loss = loss + pose_prior_weight * prior
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                if gaussian_enabled and gaussian_scales is not None:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            if self.pfgs360_replace_fuse_enabled and sky_prune_mask is not None and bool(sky_prune_mask.any().detach().cpu()):
                pruned = self._apply_sky_prune_mask(sky_prune_mask)
                if pruned > 0:
                    sky_pruned_total += int(pruned)
                    gaussian_scales = self._feedforward_gaussian_scales(selected_keyframe_ids)
                    gaussian_enabled = (
                        gaussian_scales is not None
                        and self.map.anchor_count() > 0
                        and bool((gaussian_scales > 0).any().detach().cpu())
                    )
                    param_groups = self._map_param_groups(gaussian_enabled=gaussian_enabled, phase="feedforward_window")
                    if pose_params:
                        param_groups.append({"params": pose_params, "lr": float(self.optim_cfg.get("pose_lr", 1e-3))})
                    if param_groups:
                        optimizer = torch.optim.AdamW(
                            param_groups,
                            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
                        )
            actual_steps = step_idx + 1
            last = {
                key: float(torch.stack(values).mean().detach().cpu())
                for key, values in metric_accum.items()
                if values
            }
            last["loss"] = float(loss.detach().cpu())
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(actual_steps)
                        break
        active_mask = None
        if self.map.anchor_count() > 0:
            active_mask = self._active_anchor_mask_for_keyframes(selected_keyframe_ids)
        prune_stats = (
            {"opacity_resets": 0, "pruned": 0}
            if self.pfgs360_replace_fuse_enabled
            else self._maybe_prune_feedforward_window(
                observations,
                active_mask=active_mask,
                selected_keyframe_ids=selected_keyframe_ids,
            )
        )
        pose_norm = self._pose_delta_norm(trainable_pose_ids)
        total_sec = float(time.perf_counter() - total_start)
        last["steps"] = float(actual_steps)
        last["pose_delta_norm"] = pose_norm
        last["window_size"] = float(len(observations))
        last["feedforward_window_size"] = float(len(observations))
        last["feedforward_keyframe_count"] = float(len(selected_keyframe_ids))
        last["sampled_window_size"] = float(len(last_sampled_ids))
        last["last_sampled_keyframe"] = float(last_sampled_ids[0]) if last_sampled_ids else -1.0
        last["trainable_pose_count"] = float(len(trainable_pose_ids))
        last["frontend_graph_window_hint_count"] = float(len(self.frontend_graph_window_ids))
        last["feedforward_opacity_resets"] = float(prune_stats.get("opacity_resets", 0))
        last["feedforward_pruned"] = float(prune_stats.get("pruned", 0))
        last["sky_pruned"] = float(sky_pruned_total)
        last["profile_backend_feedforward_window_sec"] = total_sec
        last["profile_backend_feedforward_window_step_avg_sec"] = total_sec / max(1, actual_steps)
        last["profile_backend_feedforward_window_sample_sec"] = float(sample_sec)
        last["profile_backend_feedforward_window_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_feedforward_window_backward_step_sec"] = float(backward_step_sec)
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = "feedforward_window"
        self.stats.last_pose_delta_norm = pose_norm
        self.stats.last_window_size = int(len(observations))
        self.stats.last_window_observations = [int(obs.frame_id) for obs in observations]
        self.stats.last_window_keyframes = list(selected_keyframe_ids)
        self.stats.last_feedforward_current_frames = [int(fid) for fid in current_frame_ids or []]
        hist_limit = max(0, int(cfg.get("history_keyframes", 2)))
        self.stats.last_feedforward_history_frames = [int(fid) for fid in (history_frame_ids or [])][-hist_limit:]
        self.stats.last_sampled_keyframes = list(last_sampled_ids)
        self.stats.last_trainable_pose_count = int(len(trainable_pose_ids))
        self.stats.last_feedforward_opacity_resets = int(prune_stats.get("opacity_resets", 0))
        self.stats.last_feedforward_pruned = int(prune_stats.get("pruned", 0))
        self.stats.last_sky_pruned = int(sky_pruned_total)
        self.stats.optimization_steps += int(actual_steps)
        return last

    def _feedforward_window_ids(self, current_frame_ids, history_frame_ids=None) -> list[int]:
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            current_limit = max(1, int(self.optim_cfg.get("current_chunk_observation_frames", cfg.get("current_chunk_observation_frames", 4))))
            recent_keyframes = max(
                0,
                int(self.optim_cfg.get("recent_keyframe_observation_frames", cfg.get("recent_keyframe_observation_frames", 2))),
            )
            ids: list[int] = []
            current = [] if current_frame_ids is None else [int(fid) for fid in current_frame_ids]
            for fid in current[-current_limit:]:
                if fid not in ids:
                    ids.append(fid)
            if recent_keyframes > 0:
                for keyframe in self.keyframes[-recent_keyframes:]:
                    fid = int(keyframe.frame_id)
                    if fid not in ids:
                        ids.append(fid)
            return ids
        hist_limit = max(0, int(cfg.get("history_keyframes", 2)))
        ids: list[int] = []
        history = [] if history_frame_ids is None else [int(fid) for fid in history_frame_ids]
        current = [] if current_frame_ids is None else [int(fid) for fid in current_frame_ids]
        for fid in history[-hist_limit:] if hist_limit > 0 else []:
            if fid not in ids:
                ids.append(fid)
        for fid in current:
            if fid not in ids:
                ids.append(fid)
        return ids

    def _selected_observations_for_ids(self, frame_ids: list[int]) -> list[MapperObservation]:
        selected: list[MapperObservation] = []
        for fid in frame_ids:
            obs = self.observations.get(int(fid))
            if obs is not None:
                selected.append(obs)
        return selected

    def _feedforward_keyframe_ids(self, frame_ids) -> list[int]:
        registered = {int(kf.frame_id) for kf in self.keyframes}
        ids: list[int] = []
        for fid in frame_ids or []:
            value = int(fid)
            if value in registered and value not in ids:
                ids.append(value)
        return ids

    def _feedforward_trainable_pose_ids(self, current_keyframe_ids: list[int], *, pose_enabled: bool) -> set[int]:
        if not pose_enabled:
            return set()
        return {int(fid) for fid in current_keyframe_ids if int(fid) in self.pose_deltas}

    def _active_anchor_mask_for_keyframes(self, keyframe_ids: list[int]) -> torch.Tensor:
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        if n <= 0:
            return torch.zeros(n, dtype=torch.bool, device=device)
        if self.pfgs360_replace_fuse_enabled:
            return self._active_anchor_mask_for_recent_updates()
        if not keyframe_ids:
            return torch.zeros(n, dtype=torch.bool, device=device)
        birth = self.map._anchor_birth_frame.to(device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for fid in keyframe_ids:
            mask |= birth == int(fid)
        return mask

    def _active_anchor_mask_for_recent_updates(self) -> torch.Tensor:
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        if n <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        recent_kfs = max(1, int(self.optim_cfg.get("recent_insert_keyframes", 2)))
        latest_ord = max(0, int(self.stats.n_keyframes) - 1)
        min_ord = max(0, latest_ord - recent_kfs + 1)
        if int(self.map._anchor_last_update_kf_ord.shape[0]) != n:
            return torch.zeros(n, dtype=torch.bool, device=device)
        return self.map._anchor_last_update_kf_ord.to(device=device) >= int(min_ord)

    def _feedforward_gaussian_scales(self, selected_keyframe_ids: list[int]) -> torch.Tensor | None:
        if not bool(self.optim_cfg.get("gaussian_refine_enable", True)):
            return None
        n = self.map.anchor_count()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        scales = torch.zeros(n, device=device, dtype=dtype)
        if n <= 0:
            return scales
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled:
            active = self._active_anchor_mask_for_recent_updates()
            scales[active] = float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("new_gaussian_lr_scale", 1.0)))
            return scales
        scope = str(cfg.get("gaussian_scope", "selected_birth_keyframes")).lower()
        if scope == "all":
            scales.fill_(float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("existing_gaussian_lr_scale", 0.1))))
            return scales
        active = self._active_anchor_mask_for_keyframes(selected_keyframe_ids)
        scales[active] = float(cfg.get("gaussian_lr_scale", self.optim_cfg.get("new_gaussian_lr_scale", 1.0)))
        return scales

    def _sample_observations_for_step(self, observations: list[MapperObservation]) -> list[MapperObservation]:
        cfg = self._feedforward_window_cfg()
        if self.pfgs360_replace_fuse_enabled or bool(self.optim_cfg.get("optimize_after_every_chunk", False)):
            sample_n = max(
                1,
                int(self.optim_cfg.get("sample_frames_per_step", cfg.get("sample_observations_per_step", 1))),
            )
            return random.sample(observations, min(sample_n, len(observations))) if len(observations) > 1 else list(observations)
        random_window = bool(cfg.get("random_observation_per_iter", False))
        if random_window and len(observations) > 1:
            sample_n = max(1, int(cfg.get("sample_observations_per_step", self.optim_cfg.get("sample_keyframes_per_step", 1))))
            return random.sample(observations, min(sample_n, len(observations)))
        return list(observations)

    def _observation_pose(self, obs: MapperObservation, *, trainable_pose_ids: set[int]) -> torch.Tensor:
        pose_delta = self.pose_deltas.get(int(obs.frame_id))
        if pose_delta is None:
            return obs.pose_c2w.detach()
        if int(obs.frame_id) in trainable_pose_ids:
            return pose_delta()
        return pose_delta().detach()

    def _sky_prune_mask_from_render_pkg(
        self,
        obs: MapperObservation,
        render_pkg: dict,
        sky_mask: torch.Tensor | None,
        c2w: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor | None:
        n = self.map.anchor_count()
        if n <= 0:
            return None
        if sky_mask is None and obs.sky_mask is not None:
            sky_mask = self._normalize_skybox_mask(
                obs.sky_mask,
                height=int(H),
                width=int(W),
                device=self.map.get_xyz.device,
            )
        if sky_mask is None:
            return None
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        visibility = render_pkg.get("visibility_filter")
        visible = torch.ones(n, dtype=torch.bool, device=device)
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            visible = visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(visible, as_tuple=False).flatten()
        if rows.numel() == 0:
            return None
        xyz = self.map.get_xyz.detach()
        w2c = torch.linalg.inv(c2w.detach().to(device=device, dtype=dtype))
        xyz_h = torch.cat([xyz.index_select(0, rows), torch.ones(rows.numel(), 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        if not bool(valid.any()):
            return None
        rows = rows.index_select(0, torch.nonzero(valid, as_tuple=False).flatten())
        cam = cam[valid]
        pixels = bearing_to_erp_pixel(cam, int(H), int(W))
        ui = pixels[:, 0].round().long().remainder(int(W))
        vi = pixels[:, 1].round().long().clamp(0, int(H) - 1)
        sky = sky_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
        if not bool(sky.any()):
            return None
        prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
        prune_mask[rows[sky]] = True
        return prune_mask

    def _apply_sky_prune_mask(self, prune_mask: torch.Tensor) -> int:
        n = self.map.anchor_count()
        if n <= 0:
            return 0
        mask = prune_mask.detach().to(device=self.map.get_xyz.device, dtype=torch.bool).view(-1)
        if int(mask.numel()) != n or not bool(mask.any().detach().cpu()):
            return 0
        pruned = self.map.prune_anchors(mask)
        if pruned > 0:
            self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
            self.stats.n_anchors = self.map.anchor_count()
            self._rebuild_keyframe_ranges_from_birth_frames()
            if self.pfgs360_replace_fuse_enabled:
                self._refresh_pfgs360_voxel_cache(compact=self.replace_fuse_compact_voxels)
        return int(pruned)

    def _prune_sky_observations(self, observations: list[MapperObservation]) -> int:
        if not observations or self.map.anchor_count() <= 0:
            return 0
        total = 0
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        with torch.no_grad():
            for obs in observations:
                if self.map.anchor_count() <= 0:
                    break
                target = obs.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                c2w = self._observation_pose(obs, trainable_pose_ids=set()).to(device=device, dtype=dtype)
                pkg = self.renderer.render(PanoRenderCamera(image_height=H, image_width=W, c2w=c2w), self.map)
                sky_mask = self._skybox_mask_for_target(target, obs.sky_mask)
                mask = self._sky_prune_mask_from_render_pkg(obs, pkg, sky_mask, c2w, H, W)
                if mask is not None:
                    total += self._apply_sky_prune_mask(mask)
        return int(total)

    def _clamp_replace_fuse_scaling(self) -> None:
        if self.map.anchor_count() <= 0:
            return
        min_scale = max(1.0e-5, float(self.optim_cfg.get("gaussian_scale_min", 0.008)))
        max_scale = max(min_scale, float(self.optim_cfg.get("gaussian_scale_max", 0.08)))
        with torch.no_grad():
            scale = self.map.get_scaling.detach().clamp(min=min_scale, max=max_scale)
            self.map.scaling.data.copy_(torch.log(torch.expm1(scale.clamp_min(1.0e-5))))

    def _maybe_prune_feedforward_window(
        self,
        observations: list[MapperObservation],
        *,
        active_mask: torch.Tensor | None,
        selected_keyframe_ids: list[int],
    ) -> dict[str, int]:
        cfg = self._feedforward_window_cfg()
        prune_cfg = cfg.get("prune", {}) if isinstance(cfg, dict) else {}
        prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
        stats = {"opacity_resets": 0, "pruned": 0}
        if not bool(prune_cfg.get("enabled", False)):
            return stats
        n = self.map.anchor_count()
        if n <= 0 or active_mask is None or int(active_mask.numel()) != n or not bool(active_mask.any()):
            return stats
        with torch.no_grad():
            for obs in observations:
                self._accumulate_feedforward_prune_evidence(obs, active_mask=active_mask)
            device = self.map.get_xyz.device
            active_cpu = active_mask.detach().cpu().bool()
            outlier = self.map._anchor_outlier_obs[:n].detach().cpu().float()
            inlier = self.map._anchor_inlier_obs[:n].detach().cpu().float()
            seen = outlier + inlier
            bad_ratio = outlier / seen.clamp_min(1.0)
            reset_after = max(0, int(prune_cfg.get("reset_after_bad", 3)))
            prune_after = max(0, int(prune_cfg.get("prune_after_bad", 6)))
            min_seen = max(1, int(prune_cfg.get("min_seen", 2)))
            min_bad_ratio = float(prune_cfg.get("min_bad_ratio", 0.7))
            max_inlier_count = max(0, int(prune_cfg.get("max_inlier_count", 1)))
            opacity_after_reset = max(1.0e-5, min(1.0 - 1.0e-5, float(prune_cfg.get("opacity_after_reset", 0.01))))
            current_opacity = self.map.get_opacity.detach().cpu().view(-1)
            reset_rows = torch.zeros(0, dtype=torch.long)
            if reset_after > 0:
                reset_rows = torch.nonzero(
                    active_cpu
                    & (outlier >= float(reset_after))
                    & (bad_ratio >= min_bad_ratio)
                    & (current_opacity > opacity_after_reset * 1.5),
                    as_tuple=False,
                ).flatten()
                if reset_rows.numel() > 0:
                    rows_dev = reset_rows.to(device=device)
                    low_opacity = torch.full(
                        (int(rows_dev.numel()), 1),
                        opacity_after_reset,
                        device=device,
                        dtype=self.map.get_xyz.dtype,
                    )
                    self.map.opacity_logit.data[rows_dev] = torch.minimum(
                        self.map.opacity_logit.data[rows_dev],
                        self.map._inv_sigmoid(low_opacity),
                    )
                    stats["opacity_resets"] = int(reset_rows.numel())
            if prune_after > 0:
                reset_set = set(int(row) for row in reset_rows.tolist())
                prune_rows = torch.nonzero(
                    active_cpu
                    & (outlier >= float(prune_after))
                    & (seen >= float(min_seen))
                    & (bad_ratio >= min_bad_ratio)
                    & (inlier <= float(max_inlier_count)),
                    as_tuple=False,
                ).flatten()
                if reset_set:
                    keep_rows = [int(row) for row in prune_rows.tolist() if int(row) not in reset_set]
                    prune_rows = torch.tensor(keep_rows, dtype=torch.long)
                max_prune = max(0, int(prune_cfg.get("max_prune_per_window", cfg.get("max_prune_per_window", 500))))
                if max_prune > 0 and prune_rows.numel() > max_prune:
                    order = torch.argsort(outlier[prune_rows], descending=True)
                    prune_rows = prune_rows.index_select(0, order[:max_prune])
                if prune_rows.numel() > 0:
                    prune_mask = torch.zeros(n, dtype=torch.bool)
                    prune_mask[prune_rows] = True
                    stats["pruned"] = self.map.prune_anchors(prune_mask.to(device=device))
                    if stats["pruned"] > 0:
                        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
                        self.stats.n_anchors = self.map.anchor_count()
                        self._rebuild_keyframe_ranges_from_birth_frames()
                        self.last_inserted_range = self._range_for_latest_keyframe()
        if stats["opacity_resets"] or stats["pruned"]:
            self.stats.notes.append(
                (
                    "feedforward prune: "
                    f"window_keyframes={list(int(fid) for fid in selected_keyframe_ids)}, "
                    f"opacity_resets={stats['opacity_resets']}, pruned={stats['pruned']}"
                )
            )
        return stats

    def _accumulate_feedforward_prune_evidence(
        self,
        obs: MapperObservation,
        *,
        active_mask: torch.Tensor,
    ) -> None:
        n = self.map.anchor_count()
        if n <= 0 or int(active_mask.numel()) != n:
            return
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        target = obs.image.to(device=device, dtype=dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        c2w = self._observation_pose(obs, trainable_pose_ids=set()).to(device=device, dtype=dtype)
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
        pkg = self.renderer.render(camera, self.map)
        sky_mask = self._skybox_mask_for_target(target, obs.sky_mask)
        pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
        visibility = pkg.get("visibility_filter")
        visible = active_mask.detach().to(device=device, dtype=torch.bool)
        if torch.is_tensor(visibility) and int(visibility.numel()) == n:
            visible = visible & visibility.to(device=device, dtype=torch.bool).view(-1)
        rows = torch.nonzero(visible, as_tuple=False).flatten()
        if rows.numel() == 0:
            return
        xyz = self.map.get_xyz.detach()
        w2c = torch.linalg.inv(c2w)
        xyz_h = torch.cat([xyz.index_select(0, rows), torch.ones(rows.numel(), 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        dist = torch.linalg.norm(cam, dim=-1)
        valid = dist > 1.0e-6
        if not bool(valid.any()):
            return
        rows = rows.index_select(0, torch.nonzero(valid, as_tuple=False).flatten())
        cam = cam[valid]
        pixels = bearing_to_erp_pixel(cam, H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        sky_bad = torch.zeros(rows.shape[0], device=device, dtype=torch.bool)
        non_sky = torch.ones(rows.shape[0], device=device, dtype=torch.bool)
        if sky_mask is not None:
            sky = sky_mask.to(device=device, dtype=torch.bool)[0, vi, ui]
            sky_bad = sky
            non_sky = ~sky
        depth_bad = torch.zeros_like(sky_bad)
        render_depth = pkg.get("depth")
        if torch.is_tensor(render_depth) and obs.target_depth is not None:
            prune_cfg = self._feedforward_window_cfg().get("prune", {})
            prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
            target_depth = obs.target_depth.to(device=device, dtype=dtype)
            if tuple(target_depth.shape[-2:]) != (H, W):
                target_depth = F.interpolate(target_depth.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            rd = render_depth.to(device=device, dtype=dtype)
            td = target_depth.to(device=device, dtype=dtype)
            alpha = pkg.get("alpha", pkg.get("opacity"))
            if not torch.is_tensor(alpha):
                alpha = torch.ones_like(rd)
            valid_depth = torch.isfinite(td) & torch.isfinite(rd) & (td > 1.0e-6) & (rd > 1.0e-6)
            scale, shift = self._robust_depth_scale_shift(
                td,
                rd,
                valid_depth & (alpha.to(device=device, dtype=dtype) >= self.pfgs360_render_alpha_min),
            )
            aligned = (td * scale + shift).clamp_min(1.0e-6)
            rel = (aligned - rd).abs() / torch.maximum(aligned, rd).clamp_min(1.0e-6)
            depth_threshold = float(prune_cfg.get("depth_rel_threshold", self.pfgs360_render_depth_rel_threshold))
            depth_bad = (rel[0, vi, ui] > depth_threshold) & non_sky
        render = pkg.get("render")
        photo_bad = torch.zeros_like(sky_bad)
        if torch.is_tensor(render):
            prune_cfg = self._feedforward_window_cfg().get("prune", {})
            prune_cfg = prune_cfg if isinstance(prune_cfg, dict) else {}
            photo_error = (render.to(device=device, dtype=dtype) - target).abs().mean(dim=0)
            photo_threshold = float(prune_cfg.get("photo_error_threshold", self.pfgs360_photometric_error_threshold))
            photo_bad = (photo_error[vi, ui] > photo_threshold) & non_sky
        bad = sky_bad | depth_bad | photo_bad
        inlier = (~bad) & non_sky
        rows_cpu = rows.detach().cpu()
        bad_cpu = bad.detach().cpu()
        inlier_cpu = inlier.detach().cpu()
        if bool(bad_cpu.any()):
            self.map._anchor_outlier_obs[rows_cpu[bad_cpu]] += 1
            self.map._anchor_last_seen_kf[rows_cpu[bad_cpu]] = int(obs.frame_id)
        if bool(inlier_cpu.any()):
            self.map._anchor_inlier_obs[rows_cpu[inlier_cpu]] += 1

    def _rebuild_keyframe_ranges_from_birth_frames(self) -> None:
        birth = self.map._anchor_birth_frame.detach().cpu().long()
        for keyframe in self.keyframes:
            rows = torch.nonzero(birth == int(keyframe.frame_id), as_tuple=False).flatten()
            if rows.numel() == 0:
                keyframe.gaussian_start = 0
                keyframe.gaussian_end = 0
            else:
                keyframe.gaussian_start = int(rows.min().item())
                keyframe.gaussian_end = int(rows.max().item()) + 1

    def _range_for_latest_keyframe(self) -> tuple[int, int]:
        if not self.keyframes:
            return (0, 0)
        latest = self.keyframes[-1]
        return (int(latest.gaussian_start), int(latest.gaussian_end))

    def _range_for_birth_frame(self, frame_id: int) -> tuple[int, int]:
        birth = self.map._anchor_birth_frame.detach().cpu().long()
        rows = torch.nonzero(birth == int(frame_id), as_tuple=False).flatten()
        if rows.numel() == 0:
            return (0, 0)
        return (int(rows.min().item()), int(rows.max().item()) + 1)

    def optimize_after_keyframe(self) -> dict[str, float]:
        """Run local and sliding-window joint Gaussian/pose optimization."""
        if self.feedforward_window_enabled:
            return {}
        if not self.uses_joint_optimization or not self.keyframes:
            return {}
        total_start = time.perf_counter()
        metrics: dict[str, float] = {}
        keyframe_steps = int(self.optim_cfg.get("keyframe_steps", 0))
        if keyframe_steps > 0:
            optimize_pose = bool(self.optim_cfg.get("keyframe_optimize_pose", self.optim_cfg.get("pose_refine_enable", False)))
            section_start = time.perf_counter()
            keyframe_metrics = self._optimize_keyframe_set(
                [self.keyframes[-1]],
                steps=keyframe_steps,
                phase="keyframe",
                gaussian_scales=self._gaussian_scales_for_phase("keyframe", [self.keyframes[-1]]),
                pose_enabled=optimize_pose,
            )
            metrics["profile_backend_keyframe_sec"] = float(time.perf_counter() - section_start)
            metrics.update(keyframe_metrics)
            if "loss" in keyframe_metrics:
                metrics["keyframe_loss"] = keyframe_metrics["loss"]
        local_steps = int(self.optim_cfg.get("local_submap_steps", 0))
        if local_steps > 0:
            local_window = int(self.optim_cfg.get("local_window_keyframes", 2))
            selected = self._select_backend_keyframe_window(max(1, local_window))
            section_start = time.perf_counter()
            local_metrics = self._optimize_keyframe_set(
                selected,
                steps=local_steps,
                phase="local_submap",
                gaussian_scales=self._gaussian_scales_for_phase("local_submap", selected),
            )
            metrics["profile_backend_local_submap_sec"] = float(time.perf_counter() - section_start)
            metrics.update(local_metrics)
            if "loss" in local_metrics:
                metrics["local_loss"] = local_metrics["loss"]

        sliding_steps = int(self.optim_cfg.get("sliding_window_steps", 0))
        if sliding_steps > 0:
            window = int(self.optim_cfg.get("window_keyframes", 8))
            selected = self._select_backend_keyframe_window(max(1, window))
            section_start = time.perf_counter()
            sliding_metrics = self._optimize_keyframe_set(
                selected,
                steps=sliding_steps,
                phase="sliding_window",
                gaussian_scales=self._gaussian_scales_for_phase("sliding_window", selected),
            )
            metrics["profile_backend_sliding_window_sec"] = float(time.perf_counter() - section_start)
            metrics.update(sliding_metrics)
            if "loss" in sliding_metrics:
                metrics["sliding_loss"] = sliding_metrics["loss"]
        metrics["profile_backend_optimize_after_keyframe_sec"] = float(time.perf_counter() - total_start)
        return metrics

    def _select_backend_keyframe_window(self, latest_count: int) -> list[MapperKeyframe]:
        latest_count = max(1, int(latest_count))
        if not bool(self.optim_cfg.get("use_frontend_graph_window", False)):
            return self.keyframes[-latest_count:]
        by_id = {int(kf.frame_id): kf for kf in self.keyframes}
        latest_ids = [int(kf.frame_id) for kf in self.keyframes[-latest_count:]]
        hint_ids = [int(fid) for fid in self.frontend_graph_window_ids if int(fid) in by_id]
        selected_ids: set[int] = set(hint_ids) | set(latest_ids)
        cap = max(0, int(self.optim_cfg.get("frontend_graph_window_max_keyframes", 0)))
        if cap > 0 and len(selected_ids) > cap:
            ordered: list[int] = []
            for fid in reversed(hint_ids):
                if fid not in ordered:
                    ordered.append(fid)
                if len(ordered) >= cap:
                    break
            if len(ordered) < cap:
                for kf in reversed(self.keyframes):
                    fid = int(kf.frame_id)
                    if fid not in ordered:
                        ordered.append(fid)
                    if len(ordered) >= cap:
                        break
            selected_ids = set(ordered)
        return [kf for kf in self.keyframes if int(kf.frame_id) in selected_ids]

    def bootstrap_latest_keyframe(self, *, steps: int, diagnostic_callback=None, diagnostic_every: int = 0) -> dict[str, float]:
        if not self.keyframes or int(steps) <= 0:
            return {}
        selected = [self.keyframes[-1]]
        total_start = time.perf_counter()
        metrics = self._optimize_keyframe_set(
            selected,
            steps=int(steps),
            phase="bootstrap",
            gaussian_scales=self._gaussian_scales_for_phase("bootstrap", selected),
            pose_enabled=False,
            diagnostic_callback=diagnostic_callback,
            diagnostic_every=diagnostic_every,
        )
        metrics["profile_backend_bootstrap_sec"] = float(time.perf_counter() - total_start)
        return metrics

    def optimize_frame_observation(
        self,
        *,
        image: torch.Tensor,
        c2w: torch.Tensor,
        steps: int,
        phase: str = "non_keyframe",
    ) -> dict[str, float]:
        if self.feedforward_window_enabled:
            return {"loss": 0.0, "steps": 0.0}
        if (self.map.anchor_count() == 0 and not self.map.has_skybox) or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        total_start = time.perf_counter()
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        sky_mask = self._skybox_mask_for_target(target)
        gaussian_scales = self._gaussian_scales_for_phase(phase, [])
        param_groups = self._map_param_groups(
            gaussian_enabled=gaussian_scales is not None and self.map.anchor_count() > 0,
            phase=phase,
        )
        if not param_groups:
            return {"loss": 0.0, "steps": 0.0, "profile_backend_non_keyframe_sec": float(time.perf_counter() - total_start)}
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)))
        last: dict[str, float] = {"loss": 0.0}
        best = float("inf")
        stale = 0
        min_delta, patience = self._early_stop_options()
        render_loss_sec = 0.0
        backward_step_sec = 0.0
        for step_idx in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            section_start = time.perf_counter()
            pkg = self.renderer.render(camera, self.map)
            pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
            loss, metrics = backend_render_loss(pkg, target, weights=self.loss_weights)
            render_loss_sec += time.perf_counter() - section_start
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                if gaussian_scales is not None:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
            last["loss"] = float(loss.detach().cpu())
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(step_idx + 1)
                        break
        last["steps"] = float(steps if "early_stop_step" not in last else int(last["early_stop_step"]))
        last["pose_delta_norm"] = 0.0
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = phase
        self.stats.last_pose_delta_norm = 0.0
        self.stats.optimization_steps += int(last["steps"])
        total_sec = float(time.perf_counter() - total_start)
        last["profile_backend_non_keyframe_sec"] = total_sec
        last["profile_backend_non_keyframe_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_non_keyframe_backward_step_sec"] = float(backward_step_sec)
        last["profile_backend_non_keyframe_step_avg_sec"] = total_sec / max(1.0, float(last["steps"]))
        return last

    def finalize_optimization(self) -> dict[str, float]:
        """Run low-frequency global polish after the sequence/block is complete."""
        if self.feedforward_window_enabled and not bool(self._feedforward_window_cfg().get("allow_final_global", False)):
            return {}
        if not self.uses_joint_optimization or not self.keyframes:
            return {}
        steps = int(self.optim_cfg.get("final_global_steps", 0))
        if steps <= 0:
            return {}
        max_kfs = int(self.optim_cfg.get("final_global_max_keyframes", 0))
        selected = self.keyframes if max_kfs <= 0 else self.keyframes[-max(1, max_kfs) :]
        return self._optimize_keyframe_set(
            selected,
            steps=steps,
            phase="final_global",
            gaussian_scales=self._gaussian_scales_for_phase("final_global", selected),
        )

    def _gaussian_scales_for_phase(self, phase: str, selected: list[MapperKeyframe]) -> torch.Tensor | None:
        if not bool(self.optim_cfg.get("gaussian_refine_enable", True)):
            return None
        n = self.map.anchor_count()
        if n <= 0:
            return torch.zeros(0, device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        scales = torch.zeros(n, device=device, dtype=dtype)
        if phase == "final_global":
            scales.fill_(float(self.optim_cfg.get("global_gaussian_lr_scale", 1.0)))
            return scales
        if phase == "bootstrap":
            scales.fill_(1.0)
            return scales
        if phase == "keyframe":
            scales.fill_(float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.15)))
            new_start, new_end = self.last_inserted_range
            if new_end > new_start:
                scales[new_start:new_end] = float(self.optim_cfg.get("new_gaussian_lr_scale", 1.0))
            return scales
        if phase == "non_keyframe":
            scales.fill_(float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.15)))
            return scales

        new_start, new_end = self.last_inserted_range
        mode = str(self.optim_cfg.get("optimize_existing_gaussians", "visible_recent")).lower()
        existing_scale = float(self.optim_cfg.get("existing_gaussian_lr_scale", 0.1))
        if phase == "sliding_window" and mode == "all":
            scales.fill_(existing_scale)
        elif phase == "sliding_window" and mode in {"visible_recent", "window", "recent"}:
            for kf in selected:
                if kf.gaussian_end > kf.gaussian_start:
                    scales[kf.gaussian_start : kf.gaussian_end] = existing_scale
        elif phase == "sliding_window" and mode in {"none", "frozen"}:
            pass

        if new_end > new_start:
            scales[new_start:new_end] = 1.0
        return scales

    def _map_param_groups(self, *, gaussian_enabled: bool, phase: str) -> list[dict]:
        groups: list[dict] = []
        if gaussian_enabled:
            if self.pfgs360_replace_fuse_enabled:
                groups.extend(
                    [
                        {
                            "params": [self.map.xyz],
                            "lr": float(self.optim_cfg.get("xyz_lr", 5.0e-4)),
                            "name": "xyz",
                        },
                        {
                            "params": [self.map.features],
                            "lr": float(self.optim_cfg.get("feature_lr", 2.0e-3)),
                            "name": "features",
                        },
                        {
                            "params": [self.map.opacity_logit],
                            "lr": float(self.optim_cfg.get("opacity_lr", 1.0e-3)),
                            "name": "opacity",
                        },
                        {
                            "params": [self.map.scaling],
                            "lr": float(self.optim_cfg.get("scaling_lr", 1.0e-4)),
                            "name": "scaling",
                        },
                        {
                            "params": [self.map.rotation],
                            "lr": float(self.optim_cfg.get("rotation_lr", 1.0e-4)),
                            "name": "rotation",
                        },
                    ]
                )
            else:
                groups.append(
                    {
                        "params": self.map.gaussian_parameters(),
                        "lr": float(self.optim_cfg.get("gaussian_lr", self.optimizer.param_groups[0]["lr"])),
                        "name": "gaussians",
                    }
                )
        if bool(self.optim_cfg.get("optimize_skybox", True)):
            sky_params = self.map.skybox_parameters()
            if sky_params:
                groups.append(
                    {
                        "params": sky_params,
                        "lr": float(self.optim_cfg.get("skybox_lr", getattr(self.map, "skybox_lr", 1.0e-2))),
                        "name": "skybox",
                    }
                )
        return groups

    def _early_stop_options(self) -> tuple[float, int]:
        cfg = self.optim_cfg.get("early_stop", {}) if isinstance(self.optim_cfg, dict) else {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return 0.0, 0
        return float(cfg.get("min_delta", 0.0)), max(0, int(cfg.get("patience", 0)))

    def _pose_trainable_keyframes(
        self,
        keyframes: list[MapperKeyframe],
        *,
        fixed: int,
        pose_enabled: bool,
    ) -> list[MapperKeyframe]:
        if not pose_enabled:
            return []
        candidates = keyframes[max(0, int(fixed)) :]
        pose_window = int(
            self.optim_cfg.get(
                "pose_window_keyframes",
                self.optim_cfg.get("pose_window", 0),
            )
        )
        if pose_window > 0:
            candidates = candidates[-pose_window:]
        return candidates

    def _sample_keyframes_for_step(
        self,
        keyframes: list[MapperKeyframe],
        *,
        selected_ids: set[int],
    ) -> tuple[list[MapperKeyframe], set[int]]:
        random_window = bool(self.optim_cfg.get("random_window_frame_per_iter", False))
        if random_window and len(keyframes) > 1:
            sample_n = max(1, int(self.optim_cfg.get("sample_keyframes_per_step", 1)))
            sample_n = min(sample_n, len(keyframes))
            sampled = random.sample(keyframes, sample_n)
        else:
            sampled = list(keyframes)

        replay_ids: set[int] = set()
        replay_n = max(0, int(self.optim_cfg.get("replay_random_keyframes", 0)))
        if replay_n > 0:
            outside = [kf for kf in self.keyframes if int(kf.frame_id) not in selected_ids]
            if outside:
                replay = random.sample(outside, min(replay_n, len(outside)))
                sampled.extend(replay)
                replay_ids = {int(kf.frame_id) for kf in replay}
        return sampled, replay_ids

    def _optimize_keyframe_set(
        self,
        keyframes: list[MapperKeyframe],
        *,
        steps: int,
        phase: str,
        gaussian_scales: torch.Tensor | None,
        pose_enabled: bool | None = None,
        diagnostic_callback=None,
        diagnostic_every: int = 0,
    ) -> dict[str, float]:
        if not keyframes or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        total_start = time.perf_counter()
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        gaussian_enabled = gaussian_scales is not None and self.map.anchor_count() > 0
        pose_enabled = bool(self.optim_cfg.get("pose_refine_enable", False)) if pose_enabled is None else bool(pose_enabled)
        fixed = max(0, int(self.optim_cfg.get("fixed_window_frames", 1)))
        if phase == "final_global":
            fixed = max(1, fixed)
        trainable_pose_ids = {
            int(kf.frame_id)
            for kf in self._pose_trainable_keyframes(
                keyframes,
                fixed=fixed,
                pose_enabled=pose_enabled,
            )
        }
        pose_params = [
            self.pose_deltas[fid].delta
            for fid in trainable_pose_ids
            if fid in self.pose_deltas
        ]
        param_groups = self._map_param_groups(gaussian_enabled=gaussian_enabled, phase=phase)
        if pose_params:
            param_groups.append(
                {
                    "params": pose_params,
                    "lr": float(self.optim_cfg.get("pose_lr", 1e-3)),
                }
            )
        if not param_groups:
            return {"loss": 0.0, "steps": 0.0, f"profile_backend_{str(phase)}_optimize_set_sec": float(time.perf_counter() - total_start)}

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        pose_prior_weight = float(self.optim_cfg.get("pose_prior_weight", 1e-3))
        last: dict[str, float] = {"loss": 0.0}
        best = float("inf")
        stale = 0
        min_delta, patience = self._early_stop_options()
        actual_steps = 0
        selected_ids = {int(kf.frame_id) for kf in keyframes}
        last_sampled_ids: list[int] = []
        sample_sec = 0.0
        render_loss_sec = 0.0
        backward_step_sec = 0.0
        diagnostic_sec = 0.0
        for step_idx in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            render_losses = []
            metric_accum: dict[str, list[torch.Tensor]] = {}
            section_start = time.perf_counter()
            sampled_keyframes, replay_ids = self._sample_keyframes_for_step(
                keyframes,
                selected_ids=selected_ids,
            )
            sample_sec += time.perf_counter() - section_start
            last_sampled_ids = [int(kf.frame_id) for kf in sampled_keyframes]
            section_start = time.perf_counter()
            for kf in sampled_keyframes:
                target = kf.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                pose_delta = self.pose_deltas.get(kf.frame_id)
                if pose_delta is None:
                    continue
                frame_id = int(kf.frame_id)
                if frame_id in trainable_pose_ids and frame_id not in replay_ids:
                    c2w = pose_delta()
                else:
                    c2w = pose_delta().detach()
                camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
                pkg = self.renderer.render(camera, self.map)
                sky_mask = self._skybox_mask_for_target(target, kf.sky_mask)
                pkg = self._apply_skybox_optimization_mask(pkg, sky_mask)
                target_depth = None if kf.target_depth is None else kf.target_depth.to(device=device, dtype=dtype)
                depth_confidence = None if kf.depth_confidence is None else kf.depth_confidence.to(device=device, dtype=dtype)
                loss_i, metrics_i = backend_render_loss(
                    pkg,
                    target,
                    target_depth=target_depth,
                    depth_confidence=depth_confidence,
                    weights=self.loss_weights,
                )
                if sky_mask is not None:
                    metrics_i = dict(metrics_i)
                    metrics_i["skybox_mask_ratio"] = sky_mask.to(device=device, dtype=dtype).mean().detach()
                render_losses.append(loss_i)
                for key, value in metrics_i.items():
                    metric_accum.setdefault(key, []).append(value.detach())
            if not render_losses:
                elapsed = float(time.perf_counter() - total_start)
                return {"loss": 0.0, "steps": 0.0, f"profile_backend_{str(phase)}_optimize_set_sec": elapsed}
            render_loss_sec += time.perf_counter() - section_start
            loss = torch.stack(render_losses).mean()
            if pose_params and pose_prior_weight > 0.0:
                prior = torch.stack([param.square().mean() for param in pose_params]).mean()
                loss = loss + pose_prior_weight * prior
            if loss.requires_grad:
                section_start = time.perf_counter()
                loss.backward()
                if gaussian_enabled:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
                if self.pfgs360_replace_fuse_enabled:
                    self._clamp_replace_fuse_scaling()
                backward_step_sec += time.perf_counter() - section_start
            actual_steps = step_idx + 1
            last = {
                key: float(torch.stack(values).mean().detach().cpu())
                for key, values in metric_accum.items()
                if values
            }
            last["loss"] = float(loss.detach().cpu())
            if diagnostic_callback is not None and len(keyframes) == 1:
                section_start = time.perf_counter()
                every = max(1, int(diagnostic_every))
                if actual_steps == 1 or actual_steps % every == 0 or actual_steps == int(steps):
                    diagnostic = self.render_keyframe_diagnostic(int(keyframes[0].frame_id))
                    if diagnostic is not None:
                        diagnostic_callback(actual_steps, diagnostic)
                diagnostic_sec += time.perf_counter() - section_start
            current = float(last["loss"])
            if min_delta > 0.0 and patience > 0:
                if current < best - min_delta:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        last["early_stop_step"] = float(actual_steps)
                        break
        pose_norm = self._pose_delta_norm(trainable_pose_ids)
        last["steps"] = float(actual_steps)
        last["pose_delta_norm"] = pose_norm
        last["window_size"] = float(len(keyframes))
        last["sampled_window_size"] = float(len(last_sampled_ids))
        last["last_sampled_keyframe"] = float(last_sampled_ids[0]) if last_sampled_ids else -1.0
        last["trainable_pose_count"] = float(len(trainable_pose_ids))
        last["frontend_graph_window_hint_count"] = float(len(self.frontend_graph_window_ids))
        phase_key = str(phase).replace("/", "_")
        total_sec = float(time.perf_counter() - total_start)
        last["profile_backend_optimize_set_sec"] = total_sec
        last["profile_backend_step_avg_sec"] = total_sec / max(1, actual_steps)
        last["profile_backend_sample_sec"] = float(sample_sec)
        last["profile_backend_render_loss_sec"] = float(render_loss_sec)
        last["profile_backend_backward_step_sec"] = float(backward_step_sec)
        last["profile_backend_diagnostic_sec"] = float(diagnostic_sec)
        last[f"profile_backend_{phase_key}_optimize_set_sec"] = total_sec
        last[f"profile_backend_{phase_key}_step_avg_sec"] = total_sec / max(1, actual_steps)
        last[f"profile_backend_{phase_key}_sample_sec"] = float(sample_sec)
        last[f"profile_backend_{phase_key}_render_loss_sec"] = float(render_loss_sec)
        last[f"profile_backend_{phase_key}_backward_step_sec"] = float(backward_step_sec)
        last[f"profile_backend_{phase_key}_diagnostic_sec"] = float(diagnostic_sec)
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = phase
        self.stats.last_pose_delta_norm = pose_norm
        self.stats.last_window_size = int(len(keyframes))
        self.stats.last_window_keyframes = [int(kf.frame_id) for kf in keyframes]
        self.stats.last_sampled_keyframes = list(last_sampled_ids)
        self.stats.last_trainable_pose_count = int(len(trainable_pose_ids))
        self.stats.optimization_steps += int(actual_steps)
        return last

    def _apply_gaussian_grad_scales(self, scales: torch.Tensor) -> None:
        if scales.numel() != self.map.anchor_count():
            return
        for param in self.map.gaussian_parameters():
            if param.grad is None or param.grad.shape[0] != scales.shape[0]:
                continue
            view_shape = (scales.shape[0],) + (1,) * (param.grad.ndim - 1)
            param.grad.mul_(scales.view(view_shape).to(device=param.grad.device, dtype=param.grad.dtype))

    def _pose_delta_norm(self, frame_ids: set[int] | None = None) -> float:
        deltas = []
        ids = frame_ids if frame_ids is not None else set(self.pose_deltas)
        for fid in ids:
            pose_delta = self.pose_deltas.get(int(fid))
            if pose_delta is not None:
                deltas.append(pose_delta.delta.detach().norm())
        if not deltas:
            return 0.0
        return float(torch.stack(deltas).mean().cpu())
