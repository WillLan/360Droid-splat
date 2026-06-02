from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class AnchorHashIndex:
    """Backend-local global voxel hash for structure anchors."""

    def __init__(self, n_levels: int = 0):
        self.structure_hash: list[dict[Tuple[int, int, int], int]] = []
        self._row_count = 0
        self._structure_count = 0
        self.last_rebuild_stats: dict = {}
        self.reset(n_levels=n_levels)

    def reset(self, n_levels: int = 0) -> None:
        n_levels = max(int(n_levels), 0)
        self.structure_hash = [dict() for _ in range(n_levels)]
        self._row_count = 0
        self._structure_count = 0
        self.last_rebuild_stats = {
            "n_hash_rebuild_collisions": 0,
            "n_duplicate_pruned": 0,
        }

    def ensure_num_levels(self, n_levels: int) -> None:
        n_levels = max(int(n_levels), 0)
        if len(self.structure_hash) == n_levels:
            return
        if len(self.structure_hash) < n_levels:
            self.structure_hash.extend(dict() for _ in range(n_levels - len(self.structure_hash)))
            return
        self.structure_hash = self.structure_hash[:n_levels]

    def lookup(self, level: int, key: Tuple[int, int, int]) -> int | None:
        level = int(level)
        if level < 0 or level >= len(self.structure_hash):
            return None
        return self.structure_hash[level].get(tuple(int(v) for v in key))

    def insert(self, level: int, key: Tuple[int, int, int], row_idx: int) -> None:
        level = int(level)
        self.ensure_num_levels(level + 1)
        self.structure_hash[level][tuple(int(v) for v in key)] = int(row_idx)

    def is_stale(self, gaussians) -> bool:
        total = int(getattr(gaussians.get_xyz, "shape", [0])[0])
        return total != self._row_count

    def _n_levels_from_model(self, gaussians) -> int:
        voxel_sizes = getattr(gaussians, "voxel_size_lis", None)
        if voxel_sizes is not None and len(voxel_sizes) > 0:
            return int(len(voxel_sizes))
        anchor_level = getattr(gaussians, "_anchor_level", None)
        if anchor_level is not None and anchor_level.numel() > 0:
            return int(anchor_level.max().item()) + 1
        return 1

    def _select_survivor(self, gaussians, incumbent: int, challenger: int) -> tuple[int, int]:
        obs_count = getattr(gaussians, "_anchor_obs_count", None)
        conf_accum = getattr(gaussians, "_anchor_conf_accum", None)

        incumbent_obs = (
            int(obs_count[incumbent].item())
            if obs_count is not None and obs_count.shape[0] > max(incumbent, challenger)
            else 0
        )
        challenger_obs = (
            int(obs_count[challenger].item())
            if obs_count is not None and obs_count.shape[0] > max(incumbent, challenger)
            else 0
        )
        if challenger_obs > incumbent_obs:
            return challenger, incumbent
        if challenger_obs < incumbent_obs:
            return incumbent, challenger

        incumbent_conf = (
            float(conf_accum[incumbent].item())
            if conf_accum is not None and conf_accum.shape[0] > max(incumbent, challenger)
            else 0.0
        )
        challenger_conf = (
            float(conf_accum[challenger].item())
            if conf_accum is not None and conf_accum.shape[0] > max(incumbent, challenger)
            else 0.0
        )
        if challenger_conf > incumbent_conf:
            return challenger, incumbent
        if challenger_conf < incumbent_conf:
            return incumbent, challenger
        if challenger < incumbent:
            return challenger, incumbent
        return incumbent, challenger

    def _build_hash_tables(
        self,
        gaussians,
        *,
        snap_xyz_to_voxel: bool,
    ) -> tuple[list[dict[Tuple[int, int, int], int]], torch.Tensor, torch.Tensor | None, list[int], int]:
        n_levels = self._n_levels_from_model(gaussians)
        structure_hash = [dict() for _ in range(n_levels)]

        xyz = gaussians.get_xyz.detach().cpu().to(dtype=torch.float32)
        total = int(xyz.shape[0])

        anchor_level = getattr(gaussians, "_anchor_level", None)
        if anchor_level is None or anchor_level.shape[0] != total:
            anchor_level = torch.zeros((total,), dtype=torch.int8)
        else:
            anchor_level = anchor_level.to(dtype=torch.int8, device="cpu")

        voxel_size_cpu = getattr(gaussians, "_anchor_voxel_size", None)
        if voxel_size_cpu is None or voxel_size_cpu.shape[0] != total:
            voxel_size_cpu = torch.zeros((total,), dtype=torch.float32)
        else:
            voxel_size_cpu = voxel_size_cpu.to(dtype=torch.float32, device="cpu")

        is_sky = getattr(gaussians, "_is_sky_anchor", None)
        if is_sky is None or is_sky.shape[0] != total:
            fallback_sky = getattr(gaussians, "_is_sky", None)
            if fallback_sky is not None and fallback_sky.shape[0] == total:
                is_sky = fallback_sky.to(dtype=torch.bool, device="cpu")
            else:
                is_sky = torch.zeros((total,), dtype=torch.bool)
        else:
            is_sky = is_sky.to(dtype=torch.bool, device="cpu")

        grid_coord = torch.zeros((total, 3), dtype=torch.int32)
        snapped_xyz = xyz.clone() if snap_xyz_to_voxel else None
        duplicate_rows: set[int] = set()
        collisions = 0

        default_voxel_sizes = getattr(gaussians, "voxel_size_lis", None) or [1.0] * n_levels
        for row_idx in range(total):
            if bool(is_sky[row_idx].item()):
                continue
            level = int(anchor_level[row_idx].item())
            level = min(max(level, 0), n_levels - 1)
            voxel_size = float(voxel_size_cpu[row_idx].item())
            if voxel_size <= 1e-6 or not np.isfinite(voxel_size):
                voxel_size = float(default_voxel_sizes[level])
            grid = np.floor(xyz[row_idx].numpy() / max(voxel_size, 1e-6)).astype(np.int32)
            grid_coord[row_idx] = torch.from_numpy(grid)
            if snapped_xyz is not None:
                snapped_xyz[row_idx] = (torch.from_numpy(grid.astype(np.float32)) + 0.5) * voxel_size
            key = (int(grid[0]), int(grid[1]), int(grid[2]))
            incumbent = structure_hash[level].get(key)
            if incumbent is None:
                structure_hash[level][key] = row_idx
                continue
            collisions += 1
            winner, loser = self._select_survivor(gaussians, incumbent, row_idx)
            structure_hash[level][key] = winner
            duplicate_rows.add(loser)

        return structure_hash, grid_coord, snapped_xyz, sorted(duplicate_rows), collisions

    def rebuild_from_model(
        self,
        gaussians,
        *,
        resolve_collisions: bool = False,
        snap_xyz_to_voxel: bool = False,
    ) -> dict:
        n_levels = self._n_levels_from_model(gaussians)
        self.ensure_num_levels(n_levels)

        total_duplicate_pruned = 0
        total_collisions = 0

        while True:
            structure_hash, grid_coord, snapped_xyz, duplicate_rows, collisions = self._build_hash_tables(
                gaussians,
                snap_xyz_to_voxel=snap_xyz_to_voxel,
            )
            total_collisions += int(collisions)

            if hasattr(gaussians, "_anchor_grid_coord"):
                gaussians._anchor_grid_coord = grid_coord
            if snapped_xyz is not None and snapped_xyz.shape == gaussians._xyz.shape:
                structure_mask = torch.ones((snapped_xyz.shape[0],), dtype=torch.bool)
                is_sky = getattr(gaussians, "_is_sky_anchor", None)
                if is_sky is not None and is_sky.shape[0] == snapped_xyz.shape[0]:
                    structure_mask = ~is_sky.to(dtype=torch.bool, device="cpu")
                gaussians._xyz.data[structure_mask.to(device=gaussians._xyz.device)] = snapped_xyz[
                    structure_mask
                ].to(device=gaussians._xyz.device, dtype=gaussians._xyz.dtype)

            if resolve_collisions and duplicate_rows:
                prune_mask = torch.zeros(
                    (gaussians.get_xyz.shape[0],),
                    device=gaussians.get_xyz.device,
                    dtype=torch.bool,
                )
                prune_mask[torch.as_tensor(duplicate_rows, device=prune_mask.device)] = True
                gaussians.prune_points(prune_mask)
                total_duplicate_pruned += len(duplicate_rows)
                continue

            self.structure_hash = structure_hash
            self._row_count = int(gaussians.get_xyz.shape[0])
            is_sky = getattr(gaussians, "_is_sky_anchor", None)
            if is_sky is not None and is_sky.shape[0] == self._row_count:
                self._structure_count = int((~is_sky).sum().item())
            else:
                self._structure_count = self._row_count
            break

        self.last_rebuild_stats = {
            "n_hash_rebuild_collisions": int(total_collisions),
            "n_duplicate_pruned": int(total_duplicate_pruned),
        }
        return dict(self.last_rebuild_stats)
