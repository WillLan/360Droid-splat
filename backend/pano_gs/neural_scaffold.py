"""Neural anchor-scaffold panorama Gaussian map.

This module implements a Scaffold-GS-style anchor representation for the
panoramic backend. Anchors store learnable positions, features, scales and
local offsets; render-time materialization decodes explicit Gaussians for the
current view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


class NeuralGaussianDecoder(nn.Module):
    """Small MLP decoder for opacity, color and covariance attributes."""

    def __init__(
        self,
        *,
        feat_dim: int = 24,
        hidden_dim: int = 64,
        k_offsets: int = 4,
        latitude_aware: bool = True,
        distance_aware: bool = True,
        include_rgb_prior: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_offsets = int(k_offsets)
        self.latitude_aware = bool(latitude_aware)
        self.distance_aware = bool(distance_aware)
        self.include_rgb_prior = bool(include_rgb_prior)

        input_dim = self.feat_dim + 3 + 1
        if self.distance_aware:
            input_dim += 1
        if self.latitude_aware:
            input_dim += 1
        if self.include_rgb_prior:
            input_dim += 3
        self.input_dim = input_dim

        self.mlp_opacity = self._make_mlp(input_dim, self.k_offsets)
        self.mlp_color = self._make_mlp(input_dim, self.k_offsets * 3)
        self.mlp_cov = self._make_mlp(input_dim, self.k_offsets * 7)

    def _make_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(
        self,
        *,
        anchor_feat: torch.Tensor,
        rgb_prior: torch.Tensor,
        view_dir: torch.Tensor,
        log_distance: torch.Tensor,
        level_scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = [anchor_feat, view_dir, level_scalar]
        if self.distance_aware:
            parts.append(log_distance)
        if self.latitude_aware:
            parts.append(view_dir[:, 2:3].clamp(-1.0, 1.0))
        if self.include_rgb_prior:
            parts.append(rgb_prior)
        x = torch.cat(parts, dim=-1)
        opacity_raw = self.mlp_opacity(x).view(-1, self.k_offsets)
        color_raw = self.mlp_color(x).view(-1, self.k_offsets, 3)
        cov_raw = self.mlp_cov(x).view(-1, self.k_offsets, 7)
        return (
            torch.nan_to_num(opacity_raw, nan=-20.0, posinf=20.0, neginf=-20.0),
            torch.nan_to_num(color_raw, nan=0.0, posinf=20.0, neginf=-20.0),
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

        self.feat_dim = int(self.neural_cfg.get("feat_dim", 24))
        self.hidden_dim = int(self.neural_cfg.get("hidden_dim", 64))
        self.k_offsets = max(1, int(self.neural_cfg.get("k_offsets", 4)))
        self.voxel_size = max(float(self.neural_cfg.get("voxel_size", 0.1)), 1.0e-8)
        self.insert_radius_factor = max(0.0, float(self.neural_cfg.get("insert_radius_factor", 2.0)))
        self.insert_radius = self.voxel_size * self.insert_radius_factor
        self.max_anchors = max(0, int(self.neural_cfg.get("max_anchors", 200000)))
        self.max_materialized_gaussians = max(0, int(self.neural_cfg.get("max_materialized_gaussians", 800000)))
        self.opacity_mask_threshold = float(self.neural_cfg.get("opacity_mask_threshold", 0.0))
        self.init_opacity = min(1.0 - 1.0e-5, max(1.0e-5, float(self.neural_cfg.get("init_opacity", 0.15))))
        self.max_scale = max(1.0e-5, float(self.neural_cfg.get("max_scale", 1.0)))

        self.decoder = NeuralGaussianDecoder(
            feat_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            k_offsets=self.k_offsets,
            latitude_aware=bool(self.neural_cfg.get("latitude_aware", True)),
            distance_aware=bool(self.neural_cfg.get("distance_aware", True)),
            include_rgb_prior=True,
        )
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
        return [
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
            {
                "params": list(self.decoder.mlp_opacity.parameters()),
                "lr": float(cfg.get("lr_mlp_opacity", 1.0e-3)),
                "name": "mlp_opacity",
            },
            {
                "params": list(self.decoder.mlp_color.parameters()),
                "lr": float(cfg.get("lr_mlp_color", 1.0e-3)),
                "name": "mlp_color",
            },
            {
                "params": list(self.decoder.mlp_cov.parameters()),
                "lr": float(cfg.get("lr_mlp_cov", 1.0e-3)),
                "name": "mlp_cov",
            },
        ]

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
        accepted = self._accepted_seed_indices(seed_batch)
        if not accepted:
            return 0
        if self.max_anchors > 0:
            room = max(0, self.max_anchors - self.anchor_count())
            accepted = accepted[:room]
            if not accepted:
                return 0
        idx = torch.tensor(accepted, device=seed_batch.xyz.device, dtype=torch.long)
        self._append_seed_indices(seed_batch, idx, last_update_kf_ord=last_update_kf_ord)
        return int(idx.numel())

    def _accepted_seed_indices(self, seed_batch: GaussianSeedBatch) -> list[int]:
        xyz_cpu = seed_batch.xyz.detach().cpu().float()
        n = int(xyz_cpu.shape[0])
        if n <= 0:
            return []
        if seed_batch.insert_enabled is not None:
            enabled = seed_batch.insert_enabled.detach().cpu().bool().view(-1)
        else:
            enabled = torch.ones(n, dtype=torch.bool)
        if seed_batch.insert_score is not None and int(seed_batch.insert_score.numel()) == n:
            score = seed_batch.insert_score.detach().cpu().float().view(-1)
        else:
            score = seed_batch.confidence.detach().cpu().float().view(-1)
        finite = torch.isfinite(xyz_cpu).all(dim=1) & torch.isfinite(score) & enabled
        order = torch.argsort(score, descending=True).tolist()
        existing_index = self._existing_spatial_hash()
        accepted_index: dict[tuple[int, int, int], list[int]] = {}
        accepted_xyz: list[torch.Tensor] = []
        accepted: list[int] = []
        for row in order:
            if not bool(finite[row]):
                continue
            point = xyz_cpu[row]
            if self._has_near_existing_anchor(point, existing_index):
                continue
            if self._has_near_accepted_seed(point, accepted_xyz, accepted_index):
                continue
            cell = self._cell_key(point)
            accepted_index.setdefault(cell, []).append(len(accepted_xyz))
            accepted_xyz.append(point)
            accepted.append(int(row))
        return accepted

    def _append_seed_indices(
        self,
        seeds: GaussianSeedBatch,
        idx: torch.Tensor,
        *,
        last_update_kf_ord: int | None,
    ) -> None:
        device = self.anchor_xyz.device
        dtype = self.anchor_xyz.dtype
        xyz = seeds.xyz.index_select(0, idx).to(device=device, dtype=dtype)
        rgb = seeds.rgb.index_select(0, idx).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        confidence = seeds.confidence.index_select(0, idx).to(device=device, dtype=dtype).view(-1).clamp(0.0, 1.0)
        level = seeds.level.index_select(0, idx).to(device=device, dtype=torch.long).view(-1)
        if seeds.insert_score is not None and int(seeds.insert_score.numel()) == int(len(seeds)):
            insert_score = seeds.insert_score.index_select(0, idx).to(device=device, dtype=dtype).view(-1)
        else:
            insert_score = confidence
        scale = seeds.scale.index_select(0, idx).to(device=device, dtype=dtype).view(-1, 1)
        valid_scale = torch.isfinite(scale) & (scale > 1.0e-8)
        scale = torch.where(valid_scale, scale, scale.new_full(scale.shape, self.voxel_size)).clamp(1.0e-5, self.max_scale)
        log_scale = scale.log().expand(-1, 6).contiguous()

        feat = torch.randn(xyz.shape[0], self.feat_dim, device=device, dtype=dtype) * 0.01
        if bool(self.neural_cfg.get("init_feat_from_rgb", True)):
            feat[:, : min(3, self.feat_dim)] = rgb[:, : min(3, self.feat_dim)]
        offsets = self._initial_offsets(xyz.shape[0], device=device, dtype=dtype)

        self.anchor_xyz = nn.Parameter(torch.cat([self.anchor_xyz.detach(), xyz], dim=0))
        self.anchor_feat = nn.Parameter(torch.cat([self.anchor_feat.detach(), feat], dim=0))
        self.anchor_log_scale = nn.Parameter(torch.cat([self.anchor_log_scale.detach(), log_scale], dim=0))
        self.local_offsets = nn.Parameter(torch.cat([self.local_offsets.detach(), offsets], dim=0))
        self.anchor_rgb_prior = torch.cat([self.anchor_rgb_prior.detach(), rgb.detach()], dim=0)
        self.anchor_confidence = torch.cat([self.anchor_confidence.detach(), confidence.detach()], dim=0)
        self.anchor_level = torch.cat([self.anchor_level.detach(), level.detach()], dim=0)
        self.anchor_insert_score = torch.cat([self.anchor_insert_score.detach(), insert_score.detach()], dim=0)

        grid = self._seed_grid(seeds, idx)
        self.anchor_grid_coord = torch.cat([self.anchor_grid_coord.detach(), grid.to(device=device)], dim=0)
        self._append_cpu_metadata(seeds, idx.detach().cpu(), grid.detach().cpu(), last_update_kf_ord=last_update_kf_ord)

    def _initial_offsets(self, n: int, *, device, dtype) -> torch.Tensor:
        base = torch.zeros(self.k_offsets, 3, device=device, dtype=dtype)
        pattern = [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (-0.5, 0.0, 0.0),
            (0.0, -0.5, 0.0),
            (0.0, 0.0, -0.5),
        ]
        for idx, value in enumerate(pattern[: self.k_offsets]):
            base[idx] = torch.tensor(value, device=device, dtype=dtype)
        return base.view(1, self.k_offsets, 3).expand(n, -1, -1).contiguous()

    def _seed_grid(self, seeds: GaussianSeedBatch, idx: torch.Tensor) -> torch.Tensor:
        if seeds.grid_coord is not None and int(seeds.grid_coord.shape[0]) == int(len(seeds)):
            return seeds.grid_coord.index_select(0, idx).to(dtype=torch.int32)
        grid = torch.floor(seeds.xyz.detach().index_select(0, idx).cpu().float() / self.voxel_size).to(torch.int32)
        return grid.to(device=idx.device)

    def _append_cpu_metadata(
        self,
        seeds: GaussianSeedBatch,
        idx_cpu: torch.Tensor,
        grid_cpu: torch.Tensor,
        *,
        last_update_kf_ord: int | None,
    ) -> None:
        count = int(idx_cpu.numel())
        self._anchor_level = torch.cat([self._anchor_level, seeds.level.detach().cpu().index_select(0, idx_cpu).to(torch.int8)], dim=0)
        self._anchor_voxel_size = torch.cat([self._anchor_voxel_size, torch.full((count,), self.voxel_size, dtype=torch.float32)], dim=0)
        self._anchor_grid_coord = torch.cat([self._anchor_grid_coord, grid_cpu.to(torch.int32)], dim=0)
        self._anchor_obs_count = torch.cat([self._anchor_obs_count, torch.ones(count, dtype=torch.int32)], dim=0)
        self._anchor_conf_accum = torch.cat(
            [self._anchor_conf_accum, seeds.confidence.detach().cpu().index_select(0, idx_cpu).to(torch.float32)],
            dim=0,
        )
        frame_ids = torch.full((count,), int(seeds.frame_id), dtype=torch.int32)
        self._anchor_birth_frame = torch.cat([self._anchor_birth_frame, frame_ids], dim=0)
        self._anchor_last_seen_kf = torch.cat([self._anchor_last_seen_kf, frame_ids], dim=0)
        update_ord = int(seeds.frame_id) if last_update_kf_ord is None else int(last_update_kf_ord)
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
        rgb_prior = self.anchor_rgb_prior.index_select(0, anchor_rows)
        level = self.anchor_level.index_select(0, anchor_rows).to(device=device, dtype=dtype).view(-1, 1)
        level_scalar = (level / 16.0).clamp(0.0, 1.0)
        camera_center = camera.c2w.to(device=device, dtype=dtype)[:3, 3]
        delta = xyz_anchor - camera_center.view(1, 3)
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        view_dir = delta / distance
        log_distance = torch.log(distance)
        opacity_raw, color_raw, cov_raw = self.decoder(
            anchor_feat=feat,
            rgb_prior=rgb_prior,
            view_dir=view_dir,
            log_distance=log_distance,
            level_scalar=level_scalar,
        )

        opacity = torch.sigmoid(opacity_raw.clamp(-10.0, 10.0)).view(-1, 1)
        color = torch.sigmoid(color_raw).view(-1, 3)
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
        score = self.anchor_insert_score
        if int(score.numel()) != n or not bool(torch.isfinite(score).any()):
            score = self.anchor_confidence
        return torch.topk(score.detach(), k=min(max_anchor_count, n), largest=True).indices.to(device=device)

    def postprocess_render_package(self, pkg: dict, materialized: MaterializedGaussians) -> dict:
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
            src = visibility.to(device=device, dtype=torch.bool).view(-1)
            dst = torch.zeros(n, device=device, dtype=torch.bool)
            dst.scatter_reduce_(0, anchor_indices, src, reduce="amax", include_self=True)
            out["visibility_filter"] = dst
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
        if torch.is_tensor(viewspace) and viewspace.ndim >= 2 and int(viewspace.shape[0]) == int(anchor_indices.numel()):
            dst = torch.zeros(n, int(viewspace.shape[-1]), device=viewspace.device, dtype=viewspace.dtype)
            seen = torch.zeros(n, device=viewspace.device, dtype=torch.bool)
            for row in range(int(anchor_indices.numel())):
                anchor = int(anchor_indices[row].detach().cpu())
                if not bool(seen[anchor]):
                    dst[anchor] = viewspace[row]
                    seen[anchor] = True
            out["viewspace_points"] = dst
        return out

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
        return pruned

    def initialize_skybox_from_image(self, *args, **kwargs) -> bool:
        return False

    def stats(self) -> dict[str, float | int]:
        return {
            "anchors": self.anchor_count(),
            "k_offsets": self.k_offsets,
            "voxel_size": float(self.voxel_size),
            "insert_radius": float(self.insert_radius),
            "feat_dim": self.feat_dim,
        }

    def save_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
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
