"""Minimal panoramic Gaussian map and mapper.

The map is intentionally compact: it exposes the attributes expected by the
PFGS360 adapter while keeping anchor-scaffold metadata local to this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from backend.pano_gs.adapter import PFGS360Renderer, PanoRenderCamera
from backend.pano_gs.losses import BackendLossWeights, backend_render_loss
from backend.pano_gs.pose_param import PoseDelta
from frontend.pano_droid.interfaces import FrontendOutput
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
    notes: list[str] = field(default_factory=list)


@dataclass
class MapperKeyframe:
    frame_id: int
    image: torch.Tensor
    gaussian_start: int
    gaussian_end: int


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
        self._anchor_level = torch.zeros(0, dtype=torch.int8)
        self._anchor_voxel_size = torch.zeros(0, dtype=torch.float32)
        self._anchor_grid_coord = torch.zeros(0, 3, dtype=torch.int32)
        self._anchor_obs_count = torch.zeros(0, dtype=torch.int32)
        self._anchor_conf_accum = torch.zeros(0, dtype=torch.float32)

    def _reset_parameters(self) -> None:
        device = self.device_hint
        self.xyz = nn.Parameter(torch.zeros(0, 3, device=device))
        self.rotation = nn.Parameter(torch.zeros(0, 4, device=device))
        self.scaling = nn.Parameter(torch.zeros(0, 3, device=device))
        self.opacity_logit = nn.Parameter(torch.zeros(0, 1, device=device))
        self.features = nn.Parameter(torch.zeros(0, 3, device=device))

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

    def anchor_count(self) -> int:
        return int(self.xyz.shape[0])

    def add_seeds(self, seeds: GaussianSeedBatch) -> int:
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

        def inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
            x = x.clamp(1e-5, 1.0 - 1e-5)
            return torch.log(x / (1.0 - x))

        new_xyz = torch.cat([self.xyz.detach(), xyz], dim=0)
        new_rot = torch.cat([self.rotation.detach(), quat], dim=0)
        new_scaling = torch.cat([self.scaling.detach(), torch.log(torch.expm1(scale.clamp_min(1e-5)))], dim=0)
        new_opacity = torch.cat([self.opacity_logit.detach(), inv_sigmoid(conf)], dim=0)
        new_features = torch.cat([self.features.detach(), inv_sigmoid(rgb)], dim=0)

        self.xyz = nn.Parameter(new_xyz)
        self.rotation = nn.Parameter(new_rot)
        self.scaling = nn.Parameter(new_scaling)
        self.opacity_logit = nn.Parameter(new_opacity)
        self.features = nn.Parameter(new_features)

        self._anchor_level = torch.cat([self._anchor_level, seeds.level.detach().cpu().to(torch.int8)], dim=0)
        self._anchor_voxel_size = torch.cat(
            [self._anchor_voxel_size, seeds.scale.detach().cpu().to(torch.float32)], dim=0
        )
        grid = torch.floor(seeds.xyz.detach().cpu() / seeds.scale.detach().cpu().view(-1, 1).clamp_min(1e-6))
        self._anchor_grid_coord = torch.cat([self._anchor_grid_coord, grid.to(torch.int32)], dim=0)
        self._anchor_obs_count = torch.cat(
            [self._anchor_obs_count, torch.ones(len(seeds), dtype=torch.int32)], dim=0
        )
        self._anchor_conf_accum = torch.cat(
            [self._anchor_conf_accum, seeds.confidence.detach().cpu().to(torch.float32)], dim=0
        )
        return int(xyz.shape[0])

    def make_optimizer(self, *, lr: float = 2e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=float(lr), weight_decay=float(weight_decay))


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
        self.loss_weights = loss_weights or BackendLossWeights()
        self.stats = MapperStats()
        self.optim_cfg = gaussian_map.config.get("BackendOptimization", {}) if isinstance(gaussian_map.config, dict) else {}
        mapping_cfg = gaussian_map.config.get("Mapping", {}) if isinstance(gaussian_map.config, dict) else {}
        novel_cfg = mapping_cfg.get("NovelGaussianInsertion", {}) if isinstance(mapping_cfg, dict) else {}
        self.novel_insertion_enabled = bool(novel_cfg.get("enabled", False))
        self.first_keyframe_max_seeds = int(novel_cfg.get("first_keyframe_max_seeds", 80000))
        self.keyframe_max_seeds = int(novel_cfg.get("keyframe_max_seeds", 20000))
        self.global_anchor_budget = int(novel_cfg.get("global_anchor_budget", 1500000))
        self.voxel_neighbor_radius = max(0, int(novel_cfg.get("voxel_neighbor_radius", 1)))
        self.keyframes: list[MapperKeyframe] = []
        self.pose_deltas: dict[int, PoseDelta] = {}
        self.last_inserted_range: tuple[int, int] = (0, 0)

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
        seeds, filter_stats = self._filter_novel_seeds(seeds)
        start = self.map.anchor_count()
        n = self.map.add_seeds(seeds)
        end = start + int(n)
        self.last_inserted_range = (start, end)
        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        self.stats.n_keyframes += 1
        self.stats.n_anchors = self.map.anchor_count()
        self.stats.last_inserted_anchors = int(n)
        self.stats.last_skipped_voxel = int(filter_stats.get("skipped_voxel", 0))
        self.stats.last_skipped_budget = int(filter_stats.get("skipped_budget", 0))
        if image is not None:
            self._register_keyframe(frontend_output, image, start=start, end=end)
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
        return n

    def _filter_novel_seeds(self, seeds: GaussianSeedBatch) -> tuple[GaussianSeedBatch, dict[str, int]]:
        if not self.novel_insertion_enabled or len(seeds) == 0:
            return seeds, {"skipped_voxel": 0, "skipped_budget": 0}
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
        kept: list[int] = []
        skipped_voxel = 0
        skipped_budget = 0
        for seed_idx in order.tolist():
            key = self._seed_voxel_key_from_cpu(xyz_cpu, scale_cpu, level_cpu, int(seed_idx))
            hit = self._find_voxel_hit(occupied, key)
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
    ) -> int | None:
        level, x, y, z = key
        radius = int(self.voxel_neighbor_radius)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    hit = occupied.get((level, x + dx, y + dy, z + dz))
                    if hit is not None:
                        return int(hit)
        return None

    def _accumulate_existing_observation(self, anchor_idx: int, confidence: float) -> None:
        if int(anchor_idx) < 0:
            return
        if int(anchor_idx) >= int(self.map._anchor_obs_count.shape[0]):
            return
        self.map._anchor_obs_count[int(anchor_idx)] += 1
        self.map._anchor_conf_accum[int(anchor_idx)] += float(confidence)

    @staticmethod
    def _subset_seeds(seeds: GaussianSeedBatch, keep_idx: torch.Tensor) -> GaussianSeedBatch:
        return GaussianSeedBatch(
            xyz=seeds.xyz.index_select(0, keep_idx.to(device=seeds.xyz.device)),
            rgb=seeds.rgb.index_select(0, keep_idx.to(device=seeds.rgb.device)),
            confidence=seeds.confidence.index_select(0, keep_idx.to(device=seeds.confidence.device)),
            scale=seeds.scale.index_select(0, keep_idx.to(device=seeds.scale.device)),
            level=seeds.level.index_select(0, keep_idx.to(device=seeds.level.device)),
            frame_id=int(seeds.frame_id),
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
        )

    def _register_keyframe(
        self,
        frontend_output: FrontendOutput,
        image: torch.Tensor,
        *,
        start: int,
        end: int,
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
        )
        self.keyframes = [kf for kf in self.keyframes if kf.frame_id != frame_id]
        self.keyframes.append(record)

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

    def render_view(self, *, image: torch.Tensor, c2w: torch.Tensor) -> dict | None:
        if self.map.anchor_count() == 0:
            return None
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        with torch.no_grad():
            return self.renderer.render(camera, self.map)

    def render_keyframe_diagnostic(self, frame_id: int) -> KeyframeRenderDiagnostic | None:
        """Render an optimized keyframe for post-optimization diagnostics."""

        if self.map.anchor_count() == 0:
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
        if self.map.anchor_count() == 0 or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        target = image.to(device=self.map.get_xyz.device, dtype=self.map.get_xyz.dtype)
        H, W = int(target.shape[-2]), int(target.shape[-1])
        camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w.to(target))
        last = {"loss": 0.0}
        for _ in range(int(steps)):
            self.optimizer.zero_grad(set_to_none=True)
            pkg = self.renderer.render(camera, self.map)
            loss, metrics = backend_render_loss(pkg, target, weights=self.loss_weights)
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = "legacy_keyframe"
        self.stats.optimization_steps += int(steps)
        return last

    def optimize_after_keyframe(self) -> dict[str, float]:
        """Run local and sliding-window joint Gaussian/pose optimization."""
        if not self.uses_joint_optimization or not self.keyframes:
            return {}
        metrics: dict[str, float] = {}
        local_steps = int(self.optim_cfg.get("local_submap_steps", 0))
        if local_steps > 0:
            local_window = int(self.optim_cfg.get("local_window_keyframes", 2))
            selected = self.keyframes[-max(1, local_window) :]
            local_metrics = self._optimize_keyframe_set(
                selected,
                steps=local_steps,
                phase="local_submap",
                gaussian_scales=self._gaussian_scales_for_phase("local_submap", selected),
            )
            metrics.update(local_metrics)
            if "loss" in local_metrics:
                metrics["local_loss"] = local_metrics["loss"]

        sliding_steps = int(self.optim_cfg.get("sliding_window_steps", 0))
        if sliding_steps > 0:
            window = int(self.optim_cfg.get("window_keyframes", 8))
            selected = self.keyframes[-max(1, window) :]
            sliding_metrics = self._optimize_keyframe_set(
                selected,
                steps=sliding_steps,
                phase="sliding_window",
                gaussian_scales=self._gaussian_scales_for_phase("sliding_window", selected),
            )
            metrics.update(sliding_metrics)
            if "loss" in sliding_metrics:
                metrics["sliding_loss"] = sliding_metrics["loss"]
        return metrics

    def finalize_optimization(self) -> dict[str, float]:
        """Run low-frequency global polish after the sequence/block is complete."""
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

    def _optimize_keyframe_set(
        self,
        keyframes: list[MapperKeyframe],
        *,
        steps: int,
        phase: str,
        gaussian_scales: torch.Tensor | None,
    ) -> dict[str, float]:
        if not keyframes or int(steps) <= 0:
            return {"loss": 0.0, "steps": 0.0}
        device = self.map.get_xyz.device
        dtype = self.map.get_xyz.dtype
        gaussian_enabled = gaussian_scales is not None and self.map.anchor_count() > 0
        pose_enabled = bool(self.optim_cfg.get("pose_refine_enable", False))
        fixed = max(0, int(self.optim_cfg.get("fixed_window_frames", 1)))
        if phase == "final_global":
            fixed = max(1, fixed)
        trainable_pose_ids = {kf.frame_id for kf in keyframes[fixed:]} if pose_enabled else set()
        pose_params = [
            self.pose_deltas[fid].delta
            for fid in trainable_pose_ids
            if fid in self.pose_deltas
        ]
        param_groups = []
        if gaussian_enabled:
            param_groups.append(
                {
                    "params": list(self.map.parameters()),
                    "lr": float(self.optim_cfg.get("gaussian_lr", self.optimizer.param_groups[0]["lr"])),
                }
            )
        if pose_params:
            param_groups.append(
                {
                    "params": pose_params,
                    "lr": float(self.optim_cfg.get("pose_lr", 1e-3)),
                }
            )
        if not param_groups:
            return {"loss": 0.0, "steps": 0.0}

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        pose_prior_weight = float(self.optim_cfg.get("pose_prior_weight", 1e-3))
        last: dict[str, float] = {"loss": 0.0}
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            render_losses = []
            metric_accum: dict[str, list[torch.Tensor]] = {}
            for kf in keyframes:
                target = kf.image.to(device=device, dtype=dtype)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                pose_delta = self.pose_deltas.get(kf.frame_id)
                if pose_delta is None:
                    continue
                c2w = pose_delta() if kf.frame_id in trainable_pose_ids else pose_delta().detach()
                camera = PanoRenderCamera(image_height=H, image_width=W, c2w=c2w)
                pkg = self.renderer.render(camera, self.map)
                loss_i, metrics_i = backend_render_loss(pkg, target, weights=self.loss_weights)
                render_losses.append(loss_i)
                for key, value in metrics_i.items():
                    metric_accum.setdefault(key, []).append(value.detach())
            if not render_losses:
                return {"loss": 0.0, "steps": 0.0}
            loss = torch.stack(render_losses).mean()
            if pose_params and pose_prior_weight > 0.0:
                prior = torch.stack([param.square().mean() for param in pose_params]).mean()
                loss = loss + pose_prior_weight * prior
            if loss.requires_grad:
                loss.backward()
                if gaussian_enabled:
                    self._apply_gaussian_grad_scales(gaussian_scales)
                optimizer.step()
            last = {
                key: float(torch.stack(values).mean().detach().cpu())
                for key, values in metric_accum.items()
                if values
            }
            last["loss"] = float(loss.detach().cpu())
        pose_norm = self._pose_delta_norm(trainable_pose_ids)
        last["steps"] = float(steps)
        last["pose_delta_norm"] = pose_norm
        self.stats.last_loss = float(last.get("loss", 0.0))
        self.stats.last_phase = phase
        self.stats.last_pose_delta_norm = pose_norm
        self.stats.optimization_steps += int(steps)
        return last

    def _apply_gaussian_grad_scales(self, scales: torch.Tensor) -> None:
        if scales.numel() != self.map.anchor_count():
            return
        for param in (self.map.xyz, self.map.rotation, self.map.scaling, self.map.opacity_logit, self.map.features):
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
