"""Initialize anchor-scaffold Gaussian seeds from PanoDROID outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from frontend.pano_droid.interfaces import FrontendOutput
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid


@dataclass
class GaussianSeedBatch:
    xyz: torch.Tensor
    rgb: torch.Tensor
    confidence: torch.Tensor
    scale: torch.Tensor
    level: torch.Tensor
    frame_id: int

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def to(self, device: torch.device | str) -> "GaussianSeedBatch":
        return GaussianSeedBatch(
            xyz=self.xyz.to(device),
            rgb=self.rgb.to(device),
            confidence=self.confidence.to(device),
            scale=self.scale.to(device),
            level=self.level.to(device),
            frame_id=self.frame_id,
        )


class GaussianInitializer:
    """Back-project ERP pixels into anchor-scaffold Gaussian seeds."""

    def __init__(
        self,
        *,
        max_seeds_per_keyframe: int = 2048,
        min_confidence: float = 0.15,
        depth_min: float = 0.05,
        depth_max: float = 1.0e4,
        voxel_sizes: tuple[float, ...] = (0.12, 0.45, 1.8),
    ) -> None:
        self.max_seeds_per_keyframe = int(max_seeds_per_keyframe)
        self.min_confidence = float(min_confidence)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)

    def from_frontend_output(
        self,
        output: FrontendOutput,
        image: torch.Tensor,
    ) -> GaussianSeedBatch:
        if output.inverse_depth is None:
            return self._empty(output.frame_id, image)
        inv = output.inverse_depth.detach().float()
        if inv.ndim == 2:
            inv = inv.unsqueeze(0)
        conf = output.depth_confidence
        if conf is None:
            conf_t = torch.ones_like(inv)
        else:
            conf_t = conf.detach().float()
            if conf_t.ndim == 2:
                conf_t = conf_t.unsqueeze(0)
        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB image, got {tuple(img.shape)}")
        _, H, W = img.shape
        if inv.shape[-2:] != (H, W):
            raise ValueError(
                f"Inverse-depth shape {tuple(inv.shape[-2:])} does not match image {(H, W)}"
            )

        depth = inv.clamp_min(1e-6).reciprocal().clamp(self.depth_min, self.depth_max)
        mask = torch.isfinite(depth) & torch.isfinite(conf_t) & (conf_t >= self.min_confidence)
        flat_idx = torch.nonzero(mask.view(-1), as_tuple=False).flatten()
        if flat_idx.numel() == 0:
            return self._empty(output.frame_id, image)
        if flat_idx.numel() > self.max_seeds_per_keyframe:
            scores = conf_t.view(-1)[flat_idx]
            _, order = torch.topk(scores, k=self.max_seeds_per_keyframe, largest=True)
            flat_idx = flat_idx[order]

        grid = pixel_grid(H, W, device=img.device, dtype=img.dtype).view(-1, 2)
        pixels = grid[flat_idx]
        bearing = erp_pixel_to_bearing(pixels, H, W)
        depth_sel = depth.view(-1)[flat_idx].to(bearing)
        pts_cam = bearing * depth_sel.unsqueeze(-1)
        c2w = output.pose_c2w.detach().to(device=pts_cam.device, dtype=pts_cam.dtype)
        pts_h = torch.cat([pts_cam, torch.ones(pts_cam.shape[0], 1, device=pts_cam.device)], dim=1)
        xyz = (c2w @ pts_h.T).T[:, :3]
        rgb = img.permute(1, 2, 0).reshape(-1, 3)[flat_idx].clamp(0.0, 1.0)
        conf_sel = conf_t.view(-1)[flat_idx].clamp(0.0, 1.0)
        levels = self._levels_from_depth(depth_sel)
        scale = torch.tensor(self.voxel_sizes, device=xyz.device, dtype=xyz.dtype)[levels.long()]
        return GaussianSeedBatch(
            xyz=xyz,
            rgb=rgb.to(xyz),
            confidence=conf_sel.to(xyz),
            scale=scale,
            level=levels.to(device=xyz.device),
            frame_id=int(output.frame_id),
        )

    def _levels_from_depth(self, depth: torch.Tensor) -> torch.Tensor:
        n_levels = len(self.voxel_sizes)
        if n_levels <= 1:
            return torch.zeros_like(depth, dtype=torch.int8)
        log_depth = torch.log10(depth.clamp_min(1e-6))
        cuts = torch.linspace(
            math.log10(max(self.depth_min, 1e-6)),
            math.log10(max(self.depth_max, self.depth_min + 1e-6)),
            n_levels + 1,
            device=depth.device,
            dtype=depth.dtype,
        )[1:-1]
        return torch.bucketize(log_depth, cuts).to(torch.int8)

    @staticmethod
    def _empty(frame_id: int, image: torch.Tensor) -> GaussianSeedBatch:
        device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
        return GaussianSeedBatch(
            xyz=torch.zeros(0, 3, device=device),
            rgb=torch.zeros(0, 3, device=device),
            confidence=torch.zeros(0, device=device),
            scale=torch.zeros(0, device=device),
            level=torch.zeros(0, dtype=torch.int8, device=device),
            frame_id=int(frame_id),
        )

