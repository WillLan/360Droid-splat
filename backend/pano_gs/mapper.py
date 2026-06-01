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
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch


@dataclass
class MapperStats:
    n_keyframes: int = 0
    n_anchors: int = 0
    last_loss: float | None = None
    last_backend: str = "pfgs360_gsplat"
    fallback_renderer: bool = False
    notes: list[str] = field(default_factory=list)


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

    def insert_keyframe(self, seeds: GaussianSeedBatch, frontend_output: FrontendOutput) -> int:
        n = self.map.add_seeds(seeds)
        self.optimizer = self.map.make_optimizer(lr=self.optimizer.param_groups[0]["lr"])
        self.stats.n_keyframes += 1
        self.stats.n_anchors = self.map.anchor_count()
        if n == 0:
            self.stats.notes.append(f"frame {frontend_output.frame_id}: no seeds inserted")
        return n

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
            loss.backward()
            self.optimizer.step()
            last = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        self.stats.last_loss = float(last.get("loss", 0.0))
        return last

