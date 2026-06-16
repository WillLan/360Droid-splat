"""Neural anchor-scaffold panorama Gaussian map.

This module implements a Scaffold-GS-style anchor representation for the
panoramic backend. Anchors store learnable positions, features, scales and
local offsets; render-time materialization decodes explicit Gaussians for the
current view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing
from mapping.gaussian_initializer import GaussianSeedBatch


SH_C0 = 0.28209479177387814


@dataclass
class MaterializedGaussians:
    """Renderer-compatible explicit Gaussians generated from neural anchors."""

    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    anchor_indices: torch.Tensor
    offset_indices: torch.Tensor
    source_anchor_count: int
    config: dict[str, Any]
    active_sh_degree: int = 0
    has_skybox: bool = False

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    @property
    def get_features(self) -> torch.Tensor:
        return self.features

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        return ((self.features - 0.5) / SH_C0).unsqueeze(1)


@dataclass
class NeuralAnchorCandidateBatch:
    """Raster-ordered neural anchor candidates built directly from frontend output."""

    xyz: torch.Tensor
    rgb: torch.Tensor
    frame_id: int
    source_flat_idx: torch.Tensor | None = None
    source_hw: tuple[int, int] | None = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


class NeuralGaussianDecoder(nn.Module):
    """Scaffold-GS-style decoder for opacity, color and covariance attributes."""

    def __init__(
        self,
        *,
        feat_dim: int = 32,
        hidden_dim: int = 32,
        k_offsets: int = 10,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_offsets = int(k_offsets)
        input_dim = self.feat_dim + 3 + 1
        self.input_dim = input_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.k_offsets),
            nn.Tanh(),
        )
        self.mlp_color = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.k_offsets * 3),
            nn.Sigmoid(),
        )
        self.mlp_cov = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.k_offsets * 7),
        )

        self._init_scaffold_outputs()

    def _init_scaffold_outputs(self) -> None:
        # Keep the first materialization visible and numerically stable.
        with torch.no_grad():
            opacity_linear = self.mlp_opacity[-2]
            if isinstance(opacity_linear, nn.Linear):
                opacity_linear.bias.fill_(0.1)
            cov_linear = self.mlp_cov[-1]
            if isinstance(cov_linear, nn.Linear):
                cov_linear.bias.zero_()
                for offset_idx in range(self.k_offsets):
                    cov_linear.bias[offset_idx * 7 + 3] = 1.0

    def forward(
        self,
        *,
        anchor_feat: torch.Tensor,
        view_dir: torch.Tensor,
        ob_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([anchor_feat, view_dir, ob_dist], dim=-1)
        opacity_raw = self.mlp_opacity(x).view(-1, self.k_offsets)
        color = self.mlp_color(x).view(-1, self.k_offsets, 3)
        cov_raw = self.mlp_cov(x).view(-1, self.k_offsets, 7)
        return (
            torch.nan_to_num(opacity_raw, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0),
            torch.nan_to_num(color, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            torch.nan_to_num(cov_raw, nan=0.0, posinf=20.0, neginf=-20.0),
        )


class NeuralScaffoldPanoMap(nn.Module):
    """Scaffold-GS-style neural anchor map for panoramic rendering."""

    map_mode = "neural_anchor_scaffold_panorama"

    def __init__(
        self,
        *,
        config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.neural_cfg = self._neural_cfg(self.config)
        self.device_hint = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self.feat_dim = int(self.neural_cfg.get("feat_dim", 32))
        self.hidden_dim = int(self.neural_cfg.get("hidden_dim", self.feat_dim))
        self.k_offsets = max(1, int(self.neural_cfg.get("k_offsets", 10)))
        self.voxel_size = max(float(self.neural_cfg.get("voxel_size", 0.05)), 1.0e-8)
        self.insert_radius_factor = max(0.0, float(self.neural_cfg.get("insert_radius_factor", 2.0)))
        self.insert_radius = self.voxel_size * self.insert_radius_factor
        self.max_anchors = max(0, int(self.neural_cfg.get("max_anchors", 200000)))
        self.max_materialized_gaussians = max(0, int(self.neural_cfg.get("max_materialized_gaussians", 800000)))
        self.opacity_mask_threshold = float(self.neural_cfg.get("opacity_mask_threshold", 0.0))
        self.init_opacity = min(1.0 - 1.0e-5, max(1.0e-5, float(self.neural_cfg.get("init_opacity", 0.15))))
        self.max_scale = max(1.0e-5, float(self.neural_cfg.get("max_scale", 1.0)))
        self.aggregate_render_stats = bool(self.neural_cfg.get("aggregate_render_stats", False))
        self.aggregate_viewspace_points = bool(self.neural_cfg.get("aggregate_viewspace_points", False))

        self.decoder = NeuralGaussianDecoder(
            feat_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            k_offsets=self.k_offsets,
        )
        self.mlp_frozen = False
        self.last_inserted_source_flat_idx: torch.Tensor | None = None
        self.last_source_hw: tuple[int, int] | None = None
        self.last_candidate_count = 0
        self.last_compacted_anchors = 0
        self.last_insert_total_sec = 0.0
        self.last_insert_accept_sec = 0.0
        self.last_insert_append_sec = 0.0
        self.last_insert_compact_sec = 0.0
        self.to(device=self.device_hint, dtype=self.dtype)
        self._reset_anchor_parameters()
        self._reset_anchor_metadata()

    @staticmethod
    def _neural_cfg(config: dict | None) -> dict:
        if not isinstance(config, dict):
            return {}
        cfg = config.get("NeuralScaffold", {})
        return cfg if isinstance(cfg, dict) else {}

    def _reset_anchor_parameters(self) -> None:
        device = self.device_hint
        dtype = self.dtype
        self.anchor_xyz = nn.Parameter(torch.zeros(0, 3, device=device, dtype=dtype))
        self.anchor_feat = nn.Parameter(torch.zeros(0, self.feat_dim, device=device, dtype=dtype))
        self.anchor_log_scale = nn.Parameter(torch.zeros(0, 6, device=device, dtype=dtype))
        self.local_offsets = nn.Parameter(torch.zeros(0, self.k_offsets, 3, device=device, dtype=dtype))
        self.register_buffer("anchor_rgb_prior", torch.zeros(0, 3, device=device, dtype=dtype))
        self.register_buffer("anchor_confidence", torch.zeros(0, device=device, dtype=dtype))
        self.register_buffer("anchor_level", torch.zeros(0, device=device, dtype=torch.long))
        self.register_buffer("anchor_grid_coord", torch.zeros(0, 3, device=device, dtype=torch.int32))
        self.register_buffer("anchor_insert_score", torch.zeros(0, device=device, dtype=dtype))

    def _reset_anchor_metadata(self) -> None:
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

    @property
    def has_skybox(self) -> bool:
        return False

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.anchor_xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        if self.anchor_log_scale.numel() == 0:
            return self.anchor_log_scale.new_zeros((0, 3))
        return self.anchor_log_scale[:, 3:6].exp().clamp(1.0e-5, self.max_scale)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._identity_quats(self.anchor_count(), device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.anchor_xyz.new_full((self.anchor_count(), 1), self.init_opacity)

    @property
    def get_features(self) -> torch.Tensor:
        return self.anchor_rgb_prior.clamp(0.0, 1.0)

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        return ((self.get_features - 0.5) / SH_C0).unsqueeze(1)

    @staticmethod
    def _identity_quats(n: int, *, device, dtype) -> torch.Tensor:
        quat = torch.zeros(int(n), 4, device=device, dtype=dtype)
        if int(n) > 0:
            quat[:, 0] = 1.0
        return quat

    @staticmethod
    def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1.0e-5, 1.0 - 1.0e-5)
        return torch.log(x / (1.0 - x))

    def anchor_count(self) -> int:
        return int(self.anchor_xyz.shape[0])

    def gaussian_parameters(self) -> list[nn.Parameter]:
        return [self.anchor_xyz, self.anchor_feat, self.anchor_log_scale, self.local_offsets]

    def skybox_parameters(self) -> list[nn.Parameter]:
        return []

    def get_optimizer_param_groups(self) -> list[dict]:
        cfg = self.neural_cfg
        groups = [
            {"params": [self.anchor_xyz], "lr": float(cfg.get("lr_anchor_xyz", 5.0e-4)), "name": "anchor_xyz"},
            {"params": [self.anchor_feat], "lr": float(cfg.get("lr_anchor_feat", 2.0e-3)), "name": "anchor_feat"},
            {
                "params": [self.anchor_log_scale],
                "lr": float(cfg.get("lr_anchor_scale", 5.0e-4)),
                "name": "anchor_log_scale",
            },
            {
                "params": [self.local_offsets],
                "lr": float(cfg.get("lr_local_offsets", 5.0e-4)),
                "name": "local_offsets",
            },
        ]
        if not self.mlp_frozen:
            groups.extend(
                [
                    {
                        "params": [p for p in self.decoder.mlp_opacity.parameters() if p.requires_grad],
                        "lr": float(cfg.get("lr_mlp_opacity", 1.0e-3)),
                        "name": "mlp_opacity",
                    },
                    {
                        "params": [p for p in self.decoder.mlp_color.parameters() if p.requires_grad],
                        "lr": float(cfg.get("lr_mlp_color", 1.0e-3)),
                        "name": "mlp_color",
                    },
                    {
                        "params": [p for p in self.decoder.mlp_cov.parameters() if p.requires_grad],
                        "lr": float(cfg.get("lr_mlp_cov", 1.0e-3)),
                        "name": "mlp_cov",
                    },
                ]
            )
        return [group for group in groups if list(group.get("params", []))]

    def freeze_mlp(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad_(False)
        self.decoder.eval()
        self.mlp_frozen = True

    def unfreeze_mlp(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad_(True)
        self.decoder.train()
        self.mlp_frozen = False

    def make_optimizer(self, *, lr: float = 2e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.get_optimizer_param_groups(), weight_decay=float(weight_decay))

    def add_seeds(
        self,
        seeds: GaussianSeedBatch,
        *,
        voxel_size: float | None = None,
        last_update_kf_ord: int | None = None,
    ) -> int:
        return self.insert_from_seed_batch(seeds, last_update_kf_ord=last_update_kf_ord)

    def insert_from_seed_batch(self, seed_batch: GaussianSeedBatch, *, last_update_kf_ord: int | None = None) -> int:
        if len(seed_batch) == 0:
            return 0
        enabled = seed_batch.insert_enabled
        if enabled is not None and int(enabled.numel()) == len(seed_batch):
            idx = torch.nonzero(enabled.detach().view(-1).bool(), as_tuple=False).flatten()
        else:
            idx = torch.arange(len(seed_batch), device=seed_batch.xyz.device, dtype=torch.long)
        if idx.numel() == 0:
            return 0
        candidates = NeuralAnchorCandidateBatch(
            xyz=seed_batch.xyz.index_select(0, idx),
            rgb=seed_batch.rgb.index_select(0, idx),
            frame_id=int(seed_batch.frame_id),
            source_flat_idx=None
            if seed_batch.source_flat_idx is None
            else seed_batch.source_flat_idx.index_select(0, idx.to(seed_batch.source_flat_idx.device)),
            source_hw=seed_batch.source_hw,
        )
        return self.insert_from_candidates(candidates, last_update_kf_ord=last_update_kf_ord)

    def insert_from_frontend_output(
        self,
        frontend_output: Any,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None = None,
        last_update_kf_ord: int | None = None,
    ) -> int:
        candidates = self._candidates_from_frontend_output(frontend_output, image, sky_mask=sky_mask)
        return self.insert_from_candidates(candidates, last_update_kf_ord=last_update_kf_ord)

    def _candidates_from_frontend_output(
        self,
        frontend_output: Any,
        image: torch.Tensor,
        *,
        sky_mask: torch.Tensor | None,
    ) -> NeuralAnchorCandidateBatch:
        points = getattr(frontend_output, "world_points", None)
        if points is None:
            points = self._world_points_from_inverse_depth(frontend_output)
            if points is None:
                return NeuralAnchorCandidateBatch(
                    xyz=torch.zeros(0, 3, device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype),
                    rgb=torch.zeros(0, 3, device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype),
                    frame_id=int(getattr(frontend_output, "frame_id", 0)),
                )
        pts = points.detach().float()
        if pts.ndim == 4 and int(pts.shape[0]) == 1:
            pts = pts[0]
        if pts.ndim != 3 or int(pts.shape[-1]) != 3:
            raise ValueError(f"Expected frontend world_points as HxWx3, got {tuple(pts.shape)}")
        H, W = int(pts.shape[0]), int(pts.shape[1])
        img = self._image_chw_for_size(image, H, W, device=pts.device, dtype=pts.dtype)
        finite = torch.isfinite(pts).all(dim=-1)
        valid = self._valid_world_mask(getattr(frontend_output, "valid_world_points_mask", None), (H, W), device=pts.device)
        sky = self._mask_for_size(sky_mask, (H, W), device=pts.device)
        keep = finite & valid
        if sky is not None:
            keep &= ~sky
        flat_keep = keep.reshape(-1)
        rows = torch.nonzero(flat_keep, as_tuple=False).flatten()
        if rows.numel() == 0:
            empty = torch.zeros(0, 3, device=pts.device, dtype=pts.dtype)
            return NeuralAnchorCandidateBatch(
                xyz=empty,
                rgb=empty.clone(),
                frame_id=int(getattr(frontend_output, "frame_id", 0)),
                source_flat_idx=rows.to(torch.long),
                source_hw=(H, W),
            )
        rgb_hw = img.permute(1, 2, 0).contiguous().view(-1, 3)
        return NeuralAnchorCandidateBatch(
            xyz=pts.reshape(-1, 3).index_select(0, rows),
            rgb=rgb_hw.index_select(0, rows).clamp(0.0, 1.0),
            frame_id=int(getattr(frontend_output, "frame_id", 0)),
            source_flat_idx=rows.detach().to(torch.long),
            source_hw=(H, W),
        )

    def _world_points_from_inverse_depth(self, frontend_output: Any) -> torch.Tensor | None:
        inv = getattr(frontend_output, "inverse_depth", None)
        pose = getattr(frontend_output, "pose_c2w", None)
        if inv is None or pose is None:
            return None
        inv_t = inv.detach().float()
        if inv_t.ndim == 4 and int(inv_t.shape[0]) == 1:
            inv_t = inv_t[0]
        if inv_t.ndim == 3 and int(inv_t.shape[0]) == 1:
            inv_t = inv_t[0]
        if inv_t.ndim != 2:
            return None
        H, W = int(inv_t.shape[0]), int(inv_t.shape[1])
        device = inv_t.device
        dtype = inv_t.dtype
        rows, cols = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        pixel = torch.stack([cols + 0.5, rows + 0.5], dim=-1)
        bearing = erp_pixel_to_bearing(pixel, H, W).to(device=device, dtype=dtype)
        valid = torch.isfinite(inv_t) & (inv_t > 1.0e-8)
        depth = torch.where(valid, 1.0 / inv_t.clamp_min(1.0e-8), torch.full_like(inv_t, float("nan")))
        cam = bearing * depth.unsqueeze(-1)
        c2w = pose.detach().to(device=device, dtype=dtype)
        if c2w.shape != (4, 4):
            return None
        world = torch.einsum("ij,hwj->hwi", c2w[:3, :3], cam) + c2w[:3, 3].view(1, 1, 3)
        return world

    @staticmethod
    def _image_chw_for_size(image: torch.Tensor, H: int, W: int, *, device, dtype) -> torch.Tensor:
        raw = image.detach()
        img = raw.to(device=device, dtype=dtype)
        if not raw.is_floating_point():
            img = img / 255.0
        if img.ndim == 4 and int(img.shape[0]) == 1:
            img = img[0]
        if img.ndim != 3:
            raise ValueError(f"Expected image as CxHxW, got {tuple(img.shape)}")
        if int(img.shape[0]) == 1:
            img = img.expand(3, -1, -1)
        if int(img.shape[0]) != 3:
            raise ValueError(f"Expected image with 1 or 3 channels, got {int(img.shape[0])}")
        if (int(img.shape[-2]), int(img.shape[-1])) != (H, W):
            img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
        return img.clamp(0.0, 1.0)

    @staticmethod
    def _mask_for_size(mask: torch.Tensor | None, size: tuple[int, int], *, device) -> torch.Tensor | None:
        if mask is None:
            return None
        H, W = int(size[0]), int(size[1])
        m = mask.detach().to(device=device)
        if m.ndim == 4 and int(m.shape[0]) == 1:
            m = m[0]
        if m.ndim == 3 and int(m.shape[0]) == 1:
            m = m[0]
        if m.ndim != 2:
            raise ValueError(f"Expected mask as HxW or 1xHxW, got {tuple(m.shape)}")
        if (int(m.shape[-2]), int(m.shape[-1])) != (H, W):
            m = F.interpolate(m.float().view(1, 1, *m.shape[-2:]), size=(H, W), mode="nearest")[0, 0]
        return m.bool()

    def _valid_world_mask(self, valid_mask: torch.Tensor | None, size: tuple[int, int], *, device) -> torch.Tensor:
        valid = self._mask_for_size(valid_mask, size, device=device)
        if valid is None:
            return torch.ones(int(size[0]), int(size[1]), device=device, dtype=torch.bool)
        return valid

    def insert_from_candidates(
        self,
        candidates: NeuralAnchorCandidateBatch,
        *,
        last_update_kf_ord: int | None = None,
    ) -> int:
        total_start = time.perf_counter()
        self.last_insert_total_sec = 0.0
        self.last_insert_accept_sec = 0.0
        self.last_insert_append_sec = 0.0
        self.last_insert_compact_sec = 0.0
        self.last_candidate_count = int(len(candidates))
        self.last_inserted_source_flat_idx = None
        self.last_source_hw = candidates.source_hw
        self.last_compacted_anchors = 0
        if len(candidates) == 0:
            self.last_insert_total_sec = float(time.perf_counter() - total_start)
            return 0
        section_start = time.perf_counter()
        accepted = self._accepted_candidate_indices(candidates)
        self.last_insert_accept_sec = float(time.perf_counter() - section_start)
        if not accepted:
            self.last_insert_total_sec = float(time.perf_counter() - total_start)
            return 0
        if self.max_anchors > 0:
            room = max(0, self.max_anchors - self.anchor_count())
            accepted = accepted[:room]
            if not accepted:
                self.last_insert_total_sec = float(time.perf_counter() - total_start)
                return 0
        idx = torch.tensor(accepted, device=candidates.xyz.device, dtype=torch.long)
        before = self.anchor_count()
        section_start = time.perf_counter()
        self._append_candidate_indices(candidates, idx, last_update_kf_ord=last_update_kf_ord)
        self.last_insert_append_sec = float(time.perf_counter() - section_start)
        section_start = time.perf_counter()
        self.last_compacted_anchors = self.compact_voxels()
        self.last_insert_compact_sec = float(time.perf_counter() - section_start)
        after = self.anchor_count()
        self.last_insert_total_sec = float(time.perf_counter() - total_start)
        return max(0, int(after - before))

    def _accepted_candidate_indices(self, candidates: NeuralAnchorCandidateBatch) -> list[int]:
        xyz_cpu = candidates.xyz.detach().cpu().float()
        n = int(xyz_cpu.shape[0])
        if n <= 0:
            return []
        finite = torch.isfinite(xyz_cpu).all(dim=1)
        existing_index = self._existing_spatial_hash()
        accepted: list[int] = []
        for row in range(n):
            if not bool(finite[row]):
                continue
            if self._has_near_existing_anchor(xyz_cpu[row], existing_index):
                continue
            accepted.append(int(row))
        return accepted

    def _append_candidate_indices(
        self,
        candidates: NeuralAnchorCandidateBatch,
        idx: torch.Tensor,
        *,
        last_update_kf_ord: int | None,
    ) -> None:
        device = self.anchor_xyz.device
        dtype = self.anchor_xyz.dtype
        xyz = candidates.xyz.index_select(0, idx).to(device=device, dtype=dtype)
        rgb = candidates.rgb.index_select(0, idx).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        confidence = torch.ones(xyz.shape[0], device=device, dtype=dtype)
        level = torch.zeros(xyz.shape[0], device=device, dtype=torch.long)
        insert_score = torch.zeros(xyz.shape[0], device=device, dtype=dtype)
        scale = self._initial_scale_for_xyz(xyz).view(-1, 1).to(device=device, dtype=dtype)
        log_scale = scale.clamp(1.0e-5, self.max_scale).log().expand(-1, 6).contiguous()

        feat = torch.zeros(xyz.shape[0], self.feat_dim, device=device, dtype=dtype)
        offsets = torch.zeros(xyz.shape[0], self.k_offsets, 3, device=device, dtype=dtype)

        self.anchor_xyz = nn.Parameter(torch.cat([self.anchor_xyz.detach(), xyz], dim=0))
        self.anchor_feat = nn.Parameter(torch.cat([self.anchor_feat.detach(), feat], dim=0))
        self.anchor_log_scale = nn.Parameter(torch.cat([self.anchor_log_scale.detach(), log_scale], dim=0))
        self.local_offsets = nn.Parameter(torch.cat([self.local_offsets.detach(), offsets], dim=0))
        self.anchor_rgb_prior = torch.cat([self.anchor_rgb_prior.detach(), rgb.detach()], dim=0)
        self.anchor_confidence = torch.cat([self.anchor_confidence.detach(), confidence.detach()], dim=0)
        self.anchor_level = torch.cat([self.anchor_level.detach(), level.detach()], dim=0)
        self.anchor_insert_score = torch.cat([self.anchor_insert_score.detach(), insert_score.detach()], dim=0)

        grid = torch.floor(xyz.detach().cpu().float() / self.voxel_size).to(torch.int32).to(device=device)
        self.anchor_grid_coord = torch.cat([self.anchor_grid_coord.detach(), grid.to(device=device)], dim=0)
        if candidates.source_flat_idx is not None:
            self.last_inserted_source_flat_idx = candidates.source_flat_idx.index_select(
                0,
                idx.to(candidates.source_flat_idx.device),
            ).detach().cpu().long()
        self._append_cpu_metadata(
            frame_id=int(candidates.frame_id),
            count=int(idx.numel()),
            grid_cpu=grid.detach().cpu(),
            last_update_kf_ord=last_update_kf_ord,
        )

    def _initial_scale_for_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        n = int(xyz.shape[0])
        fallback = xyz.new_full((n,), float(self.voxel_size))
        total = int(self.anchor_xyz.shape[0]) + n
        if n <= 1 or total <= 1 or n > 8192 or total > 65536:
            return fallback
        reference = torch.cat([self.anchor_xyz.detach(), xyz.detach()], dim=0)
        dist = torch.cdist(xyz.detach(), reference)
        if self.anchor_count() > 0:
            row = torch.arange(n, device=xyz.device)
            dist[row, self.anchor_count() + row] = float("inf")
        else:
            eye = torch.eye(n, device=xyz.device, dtype=torch.bool)
            dist[:, :n] = dist[:, :n].masked_fill(eye, float("inf"))
        nearest = dist.min(dim=1).values.clamp(1.0e-5, self.max_scale)
        nearest = torch.where(torch.isfinite(nearest), nearest, fallback)
        return nearest

    def _append_cpu_metadata(
        self,
        *,
        frame_id: int,
        count: int,
        grid_cpu: torch.Tensor,
        last_update_kf_ord: int | None,
    ) -> None:
        count = int(count)
        self._anchor_level = torch.cat([self._anchor_level, torch.zeros(count, dtype=torch.int8)], dim=0)
        self._anchor_voxel_size = torch.cat([self._anchor_voxel_size, torch.full((count,), self.voxel_size, dtype=torch.float32)], dim=0)
        self._anchor_grid_coord = torch.cat([self._anchor_grid_coord, grid_cpu.to(torch.int32)], dim=0)
        self._anchor_obs_count = torch.cat([self._anchor_obs_count, torch.ones(count, dtype=torch.int32)], dim=0)
        self._anchor_conf_accum = torch.cat([self._anchor_conf_accum, torch.ones(count, dtype=torch.float32)], dim=0)
        frame_ids = torch.full((count,), int(frame_id), dtype=torch.int32)
        self._anchor_birth_frame = torch.cat([self._anchor_birth_frame, frame_ids], dim=0)
        self._anchor_last_seen_kf = torch.cat([self._anchor_last_seen_kf, frame_ids], dim=0)
        update_ord = int(frame_id) if last_update_kf_ord is None else int(last_update_kf_ord)
        self._anchor_last_update_kf_ord = torch.cat(
            [self._anchor_last_update_kf_ord, torch.full((count,), update_ord, dtype=torch.int32)],
            dim=0,
        )
        self._anchor_inlier_obs = torch.cat([self._anchor_inlier_obs, torch.zeros(count, dtype=torch.int32)], dim=0)
        self._anchor_outlier_obs = torch.cat([self._anchor_outlier_obs, torch.zeros(count, dtype=torch.int32)], dim=0)

    def _cell_key(self, point: torch.Tensor) -> tuple[int, int, int]:
        cell = torch.floor(point / self.voxel_size).to(torch.int64)
        return int(cell[0]), int(cell[1]), int(cell[2])

    def _neighbor_cells(self, point: torch.Tensor):
        cx, cy, cz = self._cell_key(point)
        radius_cells = max(0, int(torch.ceil(torch.tensor(self.insert_radius / self.voxel_size)).item()))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    yield (cx + dx, cy + dy, cz + dz)

    def _existing_spatial_hash(self) -> dict[tuple[int, int, int], list[int]]:
        index: dict[tuple[int, int, int], list[int]] = {}
        xyz = self.anchor_xyz.detach().cpu().float()
        for row in range(int(xyz.shape[0])):
            index.setdefault(self._cell_key(xyz[row]), []).append(row)
        return index

    def _has_near_existing_anchor(self, point: torch.Tensor, index: dict[tuple[int, int, int], list[int]]) -> bool:
        if self.anchor_count() <= 0:
            return False
        xyz = self.anchor_xyz.detach().cpu().float()
        r2 = float(self.insert_radius) ** 2
        for cell in self._neighbor_cells(point):
            rows = index.get(cell)
            if not rows:
                continue
            pts = xyz[torch.tensor(rows, dtype=torch.long)]
            if bool(((pts - point.view(1, 3)).square().sum(dim=1) <= r2).any()):
                return True
        return False

    def _has_near_accepted_seed(
        self,
        point: torch.Tensor,
        accepted_xyz: list[torch.Tensor],
        accepted_index: dict[tuple[int, int, int], list[int]],
    ) -> bool:
        if not accepted_xyz:
            return False
        r2 = float(self.insert_radius) ** 2
        for cell in self._neighbor_cells(point):
            rows = accepted_index.get(cell)
            if not rows:
                continue
            pts = torch.stack([accepted_xyz[row] for row in rows], dim=0)
            if bool(((pts - point.view(1, 3)).square().sum(dim=1) <= r2).any()):
                return True
        return False

    def materialize(self, camera) -> MaterializedGaussians:
        n = self.anchor_count()
        device = self.anchor_xyz.device
        dtype = self.anchor_xyz.dtype
        if n <= 0:
            empty = torch.zeros(0, device=device, dtype=dtype)
            empty_long = torch.zeros(0, device=device, dtype=torch.long)
            return MaterializedGaussians(
                xyz=empty.view(0, 3),
                scaling=empty.view(0, 3),
                rotation=empty.view(0, 4),
                opacity=empty.view(0, 1),
                features=empty.view(0, 3),
                anchor_indices=empty_long,
                offset_indices=empty_long,
                source_anchor_count=0,
                config=self.config,
            )

        anchor_rows = self._materialized_anchor_rows()
        xyz_anchor = self.anchor_xyz.index_select(0, anchor_rows)
        feat = self.anchor_feat.index_select(0, anchor_rows)
        camera_center = camera.c2w.to(device=device, dtype=dtype)[:3, 3]
        delta = xyz_anchor - camera_center.view(1, 3)
        ob_dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        view_dir = delta / ob_dist
        opacity_raw, color_raw, cov_raw = self.decoder(
            anchor_feat=feat,
            view_dir=view_dir,
            ob_dist=ob_dist,
        )

        opacity = opacity_raw.clamp_min(0.0).view(-1, 1)
        color = color_raw.view(-1, 3)
        cov = cov_raw.view(-1, 7)
        scale_residual = torch.sigmoid(cov[:, :3])
        quat_raw = cov[:, 3:7]
        quat_norm = torch.linalg.norm(quat_raw, dim=-1, keepdim=True)
        identity = self._identity_quats(quat_raw.shape[0], device=device, dtype=dtype)
        rotation = torch.where(quat_norm > 1.0e-6, quat_raw / quat_norm.clamp_min(1.0e-6), identity)

        anchor_log_scale = self.anchor_log_scale.index_select(0, anchor_rows)
        offset_scale = anchor_log_scale[:, :3].exp().clamp(1.0e-5, self.max_scale)
        base_scale = anchor_log_scale[:, 3:6].exp().clamp(1.0e-5, self.max_scale)
        offsets = self.local_offsets.index_select(0, anchor_rows)
        gaussian_xyz = (xyz_anchor.unsqueeze(1) + offsets * offset_scale.unsqueeze(1)).reshape(-1, 3)
        gaussian_scale = (base_scale.unsqueeze(1) * scale_residual.view(-1, self.k_offsets, 3)).reshape(-1, 3)
        gaussian_scale = gaussian_scale.clamp(1.0e-5, self.max_scale)
        anchor_indices = anchor_rows.view(-1, 1).expand(-1, self.k_offsets).reshape(-1)
        offset_indices = (
            torch.arange(self.k_offsets, device=device, dtype=torch.long).view(1, -1).expand(anchor_rows.numel(), -1).reshape(-1)
        )
        mask = opacity_raw.reshape(-1) > self.opacity_mask_threshold
        if not bool(mask.any()):
            best = torch.argmax(opacity.reshape(-1))
            mask = torch.zeros_like(opacity.reshape(-1), dtype=torch.bool)
            mask[best] = True
        return MaterializedGaussians(
            xyz=torch.nan_to_num(gaussian_xyz[mask], nan=0.0, posinf=0.0, neginf=0.0),
            scaling=torch.nan_to_num(gaussian_scale[mask], nan=1.0e-5, posinf=self.max_scale, neginf=1.0e-5),
            rotation=torch.nan_to_num(rotation[mask], nan=0.0, posinf=0.0, neginf=0.0),
            opacity=torch.nan_to_num(opacity[mask], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            features=torch.nan_to_num(color[mask], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            anchor_indices=anchor_indices[mask],
            offset_indices=offset_indices[mask],
            source_anchor_count=n,
            config=self.config,
        )

    def _materialized_anchor_rows(self) -> torch.Tensor:
        n = self.anchor_count()
        device = self.anchor_xyz.device
        if self.max_materialized_gaussians <= 0 or n * self.k_offsets <= self.max_materialized_gaussians:
            return torch.arange(n, device=device, dtype=torch.long)
        max_anchor_count = max(1, self.max_materialized_gaussians // self.k_offsets)
        return torch.arange(min(max_anchor_count, n), device=device, dtype=torch.long)

    def postprocess_render_package(self, pkg: dict, materialized: MaterializedGaussians) -> dict:
        if not self.aggregate_render_stats:
            return pkg
        n = int(materialized.source_anchor_count)
        if n <= 0:
            return pkg
        out = dict(pkg)
        anchor_indices = materialized.anchor_indices
        if anchor_indices.numel() == 0:
            return out
        device = anchor_indices.device
        visibility = out.get("visibility_filter")
        if torch.is_tensor(visibility) and int(visibility.numel()) == int(anchor_indices.numel()):
            src = visibility.to(device=device, dtype=torch.int32).view(-1)
            dst = torch.zeros(n, device=device, dtype=torch.int32)
            dst.scatter_reduce_(0, anchor_indices, src, reduce="amax", include_self=True)
            out["visibility_filter"] = dst > 0
        radii = out.get("radii")
        if torch.is_tensor(radii) and int(radii.numel()) == int(anchor_indices.numel()):
            src = radii.to(device=device, dtype=torch.int32).view(-1)
            dst = torch.zeros(n, device=device, dtype=torch.int32)
            dst.scatter_reduce_(0, anchor_indices, src, reduce="amax", include_self=True)
            out["radii"] = dst
        n_touched = out.get("n_touched")
        if torch.is_tensor(n_touched) and int(n_touched.numel()) == int(anchor_indices.numel()):
            src = n_touched.to(device=device, dtype=torch.int32).view(-1)
            dst = torch.zeros(n, device=device, dtype=torch.int32)
            dst.scatter_reduce_(0, anchor_indices, src, reduce="amax", include_self=True)
            out["n_touched"] = dst
        viewspace = out.get("viewspace_points")
        if (
            self.aggregate_viewspace_points
            and torch.is_tensor(viewspace)
            and viewspace.ndim >= 2
            and int(viewspace.shape[0]) == int(anchor_indices.numel())
        ):
            dst = torch.zeros(n, int(viewspace.shape[-1]), device=viewspace.device, dtype=viewspace.dtype)
            seen = torch.zeros(n, device=viewspace.device, dtype=torch.bool)
            for row in range(int(anchor_indices.numel())):
                anchor = int(anchor_indices[row].detach().cpu())
                if not bool(seen[anchor]):
                    dst[anchor] = viewspace[row]
                    seen[anchor] = True
            out["viewspace_points"] = dst
        return out

    def compact_voxels(self) -> int:
        n = self.anchor_count()
        if n <= 1:
            return 0
        xyz = self.anchor_xyz.detach().cpu().float()
        keep = torch.zeros(n, dtype=torch.bool)
        seen: set[tuple[int, int, int]] = set()
        for row in range(n):
            if not bool(torch.isfinite(xyz[row]).all()):
                continue
            key = self._cell_key(xyz[row])
            if key in seen:
                continue
            seen.add(key)
            keep[row] = True
        removed = int((~keep).sum().item())
        if removed <= 0:
            return 0
        self._apply_anchor_keep_mask(keep.to(device=self.anchor_xyz.device))
        return removed

    def prune_anchors(self, prune_mask: torch.Tensor) -> int:
        n = self.anchor_count()
        if n <= 0:
            return 0
        mask = prune_mask.detach().to(device=self.anchor_xyz.device, dtype=torch.bool).view(-1)
        if int(mask.shape[0]) != n:
            raise ValueError(f"Prune mask length {int(mask.shape[0])} does not match anchor count {n}")
        keep = ~mask
        pruned = int(mask.sum().item())
        if pruned <= 0:
            return 0
        self._apply_anchor_keep_mask(keep)
        return pruned

    def _apply_anchor_keep_mask(self, keep: torch.Tensor) -> None:
        keep = keep.detach().to(device=self.anchor_xyz.device, dtype=torch.bool).view(-1)
        self.anchor_xyz = nn.Parameter(self.anchor_xyz.detach()[keep])
        self.anchor_feat = nn.Parameter(self.anchor_feat.detach()[keep])
        self.anchor_log_scale = nn.Parameter(self.anchor_log_scale.detach()[keep])
        self.local_offsets = nn.Parameter(self.local_offsets.detach()[keep])
        self.anchor_rgb_prior = self.anchor_rgb_prior.detach()[keep]
        self.anchor_confidence = self.anchor_confidence.detach()[keep]
        self.anchor_level = self.anchor_level.detach()[keep]
        self.anchor_grid_coord = self.anchor_grid_coord.detach()[keep]
        self.anchor_insert_score = self.anchor_insert_score.detach()[keep]
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

    def initialize_skybox_from_image(self, *args, **kwargs) -> bool:
        return False

    def stats(self) -> dict[str, float | int]:
        return {
            "anchors": self.anchor_count(),
            "k_offsets": self.k_offsets,
            "voxel_size": float(self.voxel_size),
            "insert_radius": float(self.insert_radius),
            "feat_dim": self.feat_dim,
            "mlp_frozen": int(bool(self.mlp_frozen)),
            "last_candidate_count": int(self.last_candidate_count),
            "last_compacted_anchors": int(self.last_compacted_anchors),
            "aggregate_render_stats": int(bool(self.aggregate_render_stats)),
            "last_insert_total_sec": float(self.last_insert_total_sec),
            "last_insert_accept_sec": float(self.last_insert_accept_sec),
            "last_insert_append_sec": float(self.last_insert_append_sec),
            "last_insert_compact_sec": float(self.last_insert_compact_sec),
        }

    def mlp_state_payload(self) -> dict[str, Any]:
        return {
            "mlp_opacity": self.decoder.mlp_opacity.state_dict(),
            "mlp_color": self.decoder.mlp_color.state_dict(),
            "mlp_cov": self.decoder.mlp_cov.state_dict(),
            "feat_dim": int(self.feat_dim),
            "hidden_dim": int(self.hidden_dim),
            "k_offsets": int(self.k_offsets),
            "input_mode": "anchor_feat_view_dir_ob_dist",
            "mlp_frozen": bool(self.mlp_frozen),
        }

    def save_mlp_state(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.mlp_state_payload(), path)
        return str(path)

    def save_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "mlp_state": self.mlp_state_payload(),
                "mlp_frozen": bool(self.mlp_frozen),
                "map_mode": self.map_mode,
                "config": self.config,
                "stats": self.stats(),
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
            },
            path,
        )
        if bool(self.neural_cfg.get("save_mlp", False)):
            self.save_mlp_state(path.parent / "mlp_state.pth")
        return str(path)

    def save_ply(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self.get_xyz.detach().cpu().float().numpy()
        n = int(xyz.shape[0])
        normals = np.zeros_like(xyz, dtype=np.float32)
        rgb = self.get_features.detach().cpu().float().clamp(0.0, 1.0).numpy()
        f_dc = (rgb - 0.5) / SH_C0
        f_rest = np.zeros((n, 24), dtype=np.float32)
        opacity = np.full((n, 1), float(self._inv_sigmoid(torch.tensor(self.init_opacity)).item()), dtype=np.float32)
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
