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
    source_flat_idx: torch.Tensor | None = None
    source_hw: tuple[int, int] | None = None

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
            source_flat_idx=None if self.source_flat_idx is None else self.source_flat_idx.to(device),
            source_hw=self.source_hw,
        )


class GaussianInitializer:
    """Back-project ERP pixels into anchor-scaffold Gaussian seeds.

    ``max_seeds_per_keyframe <= 0`` disables top-k sampling and keeps every
    valid pixel, which is useful for dense PanoVGGT initialization checks.
    """

    def __init__(
        self,
        *,
        max_seeds_per_keyframe: int = 2048,
        min_confidence: float = 0.15,
        depth_min: float = 0.05,
        depth_max: float = 1.0e4,
        voxel_sizes: tuple[float, ...] = (0.12, 0.45, 1.8),
        sky_mask_enable: bool = False,
        sky_mask_top_ratio: float = 0.58,
        sky_mask_min_blue: float = 0.35,
        sky_mask_blue_margin: float = 0.05,
        sky_mask_cloud_brightness: float = 0.72,
        sky_mask_cloud_saturation: float = 0.22,
        sky_mask_texture_threshold: float = 0.08,
        seed_source: str = "depth_pose",
    ) -> None:
        seed_source = str(seed_source).lower()
        if seed_source not in {"depth_pose", "world_points_only"}:
            raise ValueError(f"Unsupported Gaussian seed_source: {seed_source}")
        self.max_seeds_per_keyframe = int(max_seeds_per_keyframe)
        self.min_confidence = float(min_confidence)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)
        self.sky_mask_enable = bool(sky_mask_enable)
        self.sky_mask_top_ratio = float(sky_mask_top_ratio)
        self.sky_mask_min_blue = float(sky_mask_min_blue)
        self.sky_mask_blue_margin = float(sky_mask_blue_margin)
        self.sky_mask_cloud_brightness = float(sky_mask_cloud_brightness)
        self.sky_mask_cloud_saturation = float(sky_mask_cloud_saturation)
        self.sky_mask_texture_threshold = float(sky_mask_texture_threshold)
        self.seed_source = seed_source

    def from_frontend_output(
        self,
        output: FrontendOutput,
        image: torch.Tensor,
    ) -> GaussianSeedBatch:
        if self.seed_source == "world_points_only":
            return self.from_world_points_only(output, image)
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

        raw_depth = inv.clamp_min(1e-6).reciprocal()
        mask = (
            torch.isfinite(raw_depth)
            & (raw_depth >= self.depth_min)
            & (raw_depth <= self.depth_max)
            & torch.isfinite(conf_t)
            & (conf_t >= self.min_confidence)
        )
        if self.sky_mask_enable:
            sky = self._sky_mask_from_image(img).to(device=mask.device)
            mask = mask & ~sky
        flat_idx = torch.nonzero(mask.view(-1), as_tuple=False).flatten()
        if flat_idx.numel() == 0:
            return self._empty(output.frame_id, image)
        if self.max_seeds_per_keyframe > 0 and flat_idx.numel() > self.max_seeds_per_keyframe:
            scores = conf_t.view(-1)[flat_idx]
            _, order = torch.topk(scores, k=self.max_seeds_per_keyframe, largest=True)
            flat_idx = flat_idx[order]

        grid = pixel_grid(H, W, device=img.device, dtype=img.dtype).view(-1, 2)
        pixels = grid[flat_idx]
        bearing = erp_pixel_to_bearing(pixels, H, W)
        depth_sel = raw_depth.view(-1)[flat_idx].to(bearing)
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
            source_flat_idx=flat_idx.detach().to(device=xyz.device, dtype=torch.long),
            source_hw=(int(H), int(W)),
        )

    def from_world_points_only(
        self,
        output: FrontendOutput,
        image: torch.Tensor,
    ) -> GaussianSeedBatch:
        """Create seeds from already-global point maps without depth backprojection."""

        if output.world_points is None:
            raise ValueError(
                "Mapping.seed_source=world_points_only requires FrontendOutput.world_points."
            )
        points = output.world_points.detach().float()
        if points.ndim == 4 and points.shape[0] == 1:
            points = points[0]
        if points.ndim == 3 and points.shape[-1] != 3 and points.shape[0] == 3:
            points = points.permute(1, 2, 0)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected world_points as HxWx3, got {tuple(points.shape)}")

        img = image.detach().float()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB image, got {tuple(img.shape)}")
        _, H, W = img.shape
        if tuple(points.shape[:2]) != (H, W):
            raise ValueError(f"World-points shape {tuple(points.shape[:2])} does not match image {(H, W)}")

        conf = output.world_points_confidence
        if conf is None:
            conf = output.depth_confidence
        if conf is None:
            conf_t = torch.ones((1, H, W), dtype=points.dtype, device=points.device)
        else:
            conf_t = conf.detach().float()
            if conf_t.ndim == 2:
                conf_t = conf_t.unsqueeze(0)
            if conf_t.ndim == 3 and conf_t.shape[0] != 1:
                raise ValueError(f"Expected confidence as 1xHxW, got {tuple(conf_t.shape)}")
            if tuple(conf_t.shape[-2:]) != (H, W):
                raise ValueError(f"World-points confidence shape {tuple(conf_t.shape[-2:])} does not match image {(H, W)}")

        valid_mask = output.valid_world_points_mask
        if valid_mask is None:
            valid_t = torch.ones((1, H, W), dtype=torch.bool, device=points.device)
        else:
            valid_t = valid_mask.detach().bool()
            if valid_t.ndim == 2:
                valid_t = valid_t.unsqueeze(0)
            if valid_t.ndim == 3 and valid_t.shape[0] != 1:
                raise ValueError(f"Expected valid_world_points_mask as 1xHxW, got {tuple(valid_t.shape)}")
            if tuple(valid_t.shape[-2:]) != (H, W):
                raise ValueError(f"World-points mask shape {tuple(valid_t.shape[-2:])} does not match image {(H, W)}")

        conf_t = conf_t.to(device=points.device, dtype=points.dtype)
        valid_t = valid_t.to(device=points.device)
        mask = (
            torch.isfinite(points).all(dim=-1, keepdim=False).unsqueeze(0)
            & valid_t
            & torch.isfinite(conf_t)
            & (conf_t >= self.min_confidence)
        )
        if self.sky_mask_enable:
            sky = self._sky_mask_from_image(img).to(device=mask.device)
            mask = mask & ~sky
        flat_idx = torch.nonzero(mask.view(-1), as_tuple=False).flatten()
        if flat_idx.numel() == 0:
            return self._empty(output.frame_id, image)
        if self.max_seeds_per_keyframe > 0 and flat_idx.numel() > self.max_seeds_per_keyframe:
            scores = conf_t.reshape(-1)[flat_idx]
            _, order = torch.topk(scores, k=self.max_seeds_per_keyframe, largest=True)
            flat_idx = flat_idx[order]

        xyz = points.reshape(-1, 3)[flat_idx]
        rgb = img.to(device=xyz.device, dtype=xyz.dtype).permute(1, 2, 0).reshape(-1, 3)[flat_idx].clamp(0.0, 1.0)
        conf_sel = conf_t.reshape(-1)[flat_idx].clamp(0.0, 1.0)
        c2w = output.pose_c2w.detach().to(device=xyz.device, dtype=xyz.dtype)
        camera_center = c2w[:3, 3] if c2w.shape == (4, 4) else torch.zeros(3, device=xyz.device, dtype=xyz.dtype)
        distance = torch.linalg.norm(xyz - camera_center.view(1, 3), dim=-1).clamp_min(self.depth_min)
        levels = self._levels_from_depth(distance)
        scale = torch.tensor(self.voxel_sizes, device=xyz.device, dtype=xyz.dtype)[levels.long()]
        return GaussianSeedBatch(
            xyz=xyz,
            rgb=rgb,
            confidence=conf_sel.to(xyz),
            scale=scale,
            level=levels.to(device=xyz.device),
            frame_id=int(output.frame_id),
            source_flat_idx=flat_idx.detach().to(device=xyz.device, dtype=torch.long),
            source_hw=(int(H), int(W)),
        )

    def _sky_mask_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return a conservative sky-like mask with shape ``1xHxW``."""

        img = image.detach().float().clamp(0.0, 1.0)
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB image, got {tuple(img.shape)}")
        _, H, W = img.shape
        rows = torch.arange(H, device=img.device, dtype=img.dtype).view(1, H, 1)
        upper = rows < float(H) * self.sky_mask_top_ratio
        r, g, b = img[0:1], img[1:2], img[2:3]
        max_rgb = img.max(dim=0, keepdim=True).values
        min_rgb = img.min(dim=0, keepdim=True).values
        brightness = max_rgb
        saturation = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-6)

        blue_sky = (
            (b >= self.sky_mask_min_blue)
            & (b >= r + self.sky_mask_blue_margin)
            & (b >= g + 0.5 * self.sky_mask_blue_margin)
        )

        gray = img.mean(dim=0, keepdim=True)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, :, 1:] = (gray[:, :, 1:] - gray[:, :, :-1]).abs()
        dy[:, 1:, :] = (gray[:, 1:, :] - gray[:, :-1, :]).abs()
        low_texture = (dx + dy) <= self.sky_mask_texture_threshold
        cloud_sky = (
            (brightness >= self.sky_mask_cloud_brightness)
            & (saturation <= self.sky_mask_cloud_saturation)
            & low_texture
        )
        return upper & (blue_sky | cloud_sky)

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
            source_flat_idx=torch.zeros(0, dtype=torch.long, device=device),
            source_hw=None,
        )
