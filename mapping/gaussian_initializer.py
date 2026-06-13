"""Initialize anchor-scaffold Gaussian seeds from PanoDROID outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

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
    insert_enabled: torch.Tensor | None = None
    insert_score: torch.Tensor | None = None
    grid_coord: torch.Tensor | None = None

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
            insert_enabled=None if self.insert_enabled is None else self.insert_enabled.to(device),
            insert_score=None if self.insert_score is None else self.insert_score.to(device),
            grid_coord=None if self.grid_coord is None else self.grid_coord.to(device),
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
        sky_mask_source: str = "heuristic",
        seed_source: str = "depth_pose",
        insertion_strategy: str = "legacy",
        pfgs360_voxel_size: float = 0.12,
        pfgs360_gaussian_scale_mode: str = "voxel",
        pfgs360_gaussian_scale_factor: float = 1.25,
        pfgs360_gaussian_scale_min: float = 0.008,
        pfgs360_gaussian_scale_max: float = 0.08,
        pfgs360_gaussian_scale_lat_cos_min: float = 0.25,
        temporal_pair_conf_min: float = 0.70,
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
        self.sky_mask_source = str(sky_mask_source or "heuristic").lower()
        self.seed_source = seed_source
        self.insertion_strategy = str(insertion_strategy or "legacy").lower()
        self.pfgs360_voxel_size = max(float(pfgs360_voxel_size), 1.0e-6)
        self.pfgs360_gaussian_scale_mode = str(pfgs360_gaussian_scale_mode or "voxel").lower()
        self.pfgs360_gaussian_scale_factor = float(pfgs360_gaussian_scale_factor)
        self.pfgs360_gaussian_scale_min = max(float(pfgs360_gaussian_scale_min), 1.0e-8)
        self.pfgs360_gaussian_scale_max = max(float(pfgs360_gaussian_scale_max), self.pfgs360_gaussian_scale_min)
        self.pfgs360_gaussian_scale_lat_cos_min = min(
            1.0,
            max(0.0, float(pfgs360_gaussian_scale_lat_cos_min)),
        )
        self.temporal_pair_conf_min = float(temporal_pair_conf_min)

    def from_frontend_output(
        self,
        output: FrontendOutput,
        image: torch.Tensor,
        *,
        insertion_hints: dict | None = None,
        first_keyframe: bool = False,
    ) -> GaussianSeedBatch:
        if self.seed_source == "world_points_only":
            return self.from_world_points_only(
                output,
                image,
                insertion_hints=insertion_hints,
                first_keyframe=first_keyframe,
            )
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
            sky = self._configured_sky_mask(img, (H, W), device=mask.device, insertion_hints=insertion_hints)
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
        *,
        insertion_hints: dict | None = None,
        first_keyframe: bool = False,
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

        if self.insertion_strategy in {"pfgs360", "pfgs360_replace_fuse"}:
            return self._from_world_points_pfgs360(
                output,
                img,
                points,
                insertion_hints=insertion_hints,
                first_keyframe=first_keyframe,
            )

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
            sky = self._configured_sky_mask(img, (H, W), device=mask.device, insertion_hints=insertion_hints)
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

    def _from_world_points_pfgs360(
        self,
        output: FrontendOutput,
        img: torch.Tensor,
        points: torch.Tensor,
        *,
        insertion_hints: dict | None,
        first_keyframe: bool,
    ) -> GaussianSeedBatch:
        _, H, W = img.shape
        conf_t = self._world_confidence(output, points, H, W)
        valid_t = self._world_valid_mask(output, points, H, W)
        finite = torch.isfinite(points).all(dim=-1, keepdim=False).unsqueeze(0)
        mask = finite & valid_t & torch.isfinite(conf_t) & (conf_t >= self.min_confidence)

        if self.sky_mask_enable:
            sky = self._configured_sky_mask(img, (H, W), device=mask.device, insertion_hints=insertion_hints)
            mask = mask & ~sky

        hints = insertion_hints or {}
        non_sky = self._hint_mask(hints.get("non_sky"), (H, W), device=points.device)
        if non_sky is not None:
            mask = mask & non_sky

        replace_fuse = self.insertion_strategy == "pfgs360_replace_fuse"
        pair_conf = self._hint_field(hints.get("pair_confidence"), (H, W), device=points.device, dtype=points.dtype)
        if replace_fuse:
            temporal_ok = torch.ones_like(mask, dtype=torch.bool)
        elif bool(first_keyframe):
            temporal_ok = torch.ones_like(mask, dtype=torch.bool)
        elif pair_conf is None:
            temporal_ok = torch.ones_like(mask, dtype=torch.bool)
        else:
            temporal_ok = pair_conf >= float(self.temporal_pair_conf_min)

        flat_idx = torch.nonzero(mask.view(-1), as_tuple=False).flatten()
        if flat_idx.numel() == 0:
            return self._empty(output.frame_id, img)

        xyz_flat = points.reshape(-1, 3)
        rgb_flat = img.to(device=points.device, dtype=points.dtype).permute(1, 2, 0).reshape(-1, 3)
        conf_flat = conf_t.reshape(-1).to(device=points.device, dtype=points.dtype)
        temporal_flat = temporal_ok.reshape(-1).to(device=points.device)
        pair_flat = None if pair_conf is None else pair_conf.reshape(-1).to(device=points.device, dtype=points.dtype)

        xyz_sel = xyz_flat[flat_idx]
        rgb_sel = rgb_flat[flat_idx].clamp(0.0, 1.0)
        conf_sel = conf_flat[flat_idx].clamp(0.0, 1.0)
        insert_pixel = temporal_flat[flat_idx].bool()
        score_sel = (
            conf_sel
            if replace_fuse or pair_flat is None
            else (conf_sel * pair_flat[flat_idx].clamp(0.0, 1.0))
        )

        if replace_fuse:
            n = int(xyz_sel.shape[0])
            grid = torch.floor(xyz_sel / float(self.pfgs360_voxel_size)).to(torch.int32)
            if self.pfgs360_gaussian_scale_mode in {"erp_depth_latitude", "depth_latitude"}:
                scale = self._pfgs360_depth_latitude_scale(
                    output,
                    xyz_sel,
                    flat_idx,
                    H,
                    W,
                    device=points.device,
                    dtype=points.dtype,
                )
            else:
                scale = torch.full((n,), float(self.pfgs360_voxel_size), device=points.device, dtype=points.dtype)
            return GaussianSeedBatch(
                xyz=xyz_sel,
                rgb=rgb_sel,
                confidence=conf_sel,
                scale=scale,
                level=torch.zeros(n, dtype=torch.int8, device=points.device),
                frame_id=int(output.frame_id),
                source_flat_idx=flat_idx.detach().to(device=points.device, dtype=torch.long),
                source_hw=(int(H), int(W)),
                insert_enabled=insert_pixel.detach().to(device=points.device, dtype=torch.bool),
                insert_score=score_sel.detach().to(device=points.device, dtype=points.dtype),
                grid_coord=grid.detach().to(device=points.device, dtype=torch.int32),
            )

        grid = torch.floor(xyz_sel / float(self.pfgs360_voxel_size)).to(torch.int32)
        unique_grid, inverse = torch.unique(grid, dim=0, return_inverse=True)
        n_voxels = int(unique_grid.shape[0])
        weights = conf_sel.clamp_min(1.0e-4)
        sum_w = torch.zeros(n_voxels, device=points.device, dtype=points.dtype)
        sum_w.index_add_(0, inverse, weights)
        xyz_sum = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
        rgb_sum = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
        xyz_sum.index_add_(0, inverse, xyz_sel * weights.unsqueeze(-1))
        rgb_sum.index_add_(0, inverse, rgb_sel * weights.unsqueeze(-1))
        xyz = xyz_sum / sum_w.clamp_min(1.0e-8).unsqueeze(-1)
        rgb = rgb_sum / sum_w.clamp_min(1.0e-8).unsqueeze(-1)

        count = torch.zeros(n_voxels, device=points.device, dtype=points.dtype)
        count.index_add_(0, inverse, torch.ones_like(weights))
        conf_sum = torch.zeros(n_voxels, device=points.device, dtype=points.dtype)
        conf_sum.index_add_(0, inverse, conf_sel)
        conf = (conf_sum / count.clamp_min(1.0)).clamp(0.0, 1.0)

        insert_count = torch.zeros(n_voxels, device=points.device, dtype=points.dtype)
        insert_count.index_add_(0, inverse, insert_pixel.to(dtype=points.dtype))
        insert_enabled = insert_count > 0
        best_pos = self._best_positions_by_score(score_sel, inverse, n_voxels)
        source_flat_idx = flat_idx[best_pos]
        insert_score = score_sel[best_pos].clamp(0.0, 1.0)

        if self.max_seeds_per_keyframe > 0 and n_voxels > self.max_seeds_per_keyframe:
            _, order = torch.topk(insert_score, k=self.max_seeds_per_keyframe, largest=True)
            xyz = xyz.index_select(0, order)
            rgb = rgb.index_select(0, order)
            conf = conf.index_select(0, order)
            unique_grid = unique_grid.index_select(0, order)
            insert_enabled = insert_enabled.index_select(0, order)
            insert_score = insert_score.index_select(0, order)
            source_flat_idx = source_flat_idx.index_select(0, order)

        n = int(xyz.shape[0])
        if self.insertion_strategy == "pfgs360_replace_fuse" and self.pfgs360_gaussian_scale_mode in {
            "erp_depth_latitude",
            "depth_latitude",
        }:
            scale = self._pfgs360_depth_latitude_scale(
                output,
                xyz,
                source_flat_idx,
                H,
                W,
                device=points.device,
                dtype=points.dtype,
            )
        else:
            scale = torch.full((n,), float(self.pfgs360_voxel_size), device=points.device, dtype=points.dtype)
        return GaussianSeedBatch(
            xyz=xyz,
            rgb=rgb,
            confidence=conf,
            scale=scale,
            level=torch.zeros(n, dtype=torch.int8, device=points.device),
            frame_id=int(output.frame_id),
            source_flat_idx=source_flat_idx.detach().to(device=points.device, dtype=torch.long),
            source_hw=(int(H), int(W)),
            insert_enabled=insert_enabled.detach().to(device=points.device, dtype=torch.bool),
            insert_score=insert_score.detach().to(device=points.device, dtype=points.dtype),
            grid_coord=unique_grid.detach().to(device=points.device, dtype=torch.int32),
        )

    def _pfgs360_depth_latitude_scale(
        self,
        output: FrontendOutput,
        xyz: torch.Tensor,
        source_flat_idx: torch.Tensor,
        H: int,
        W: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        c2w = output.pose_c2w.detach().to(device=device, dtype=dtype)
        center = c2w[:3, 3] if tuple(c2w.shape) == (4, 4) else torch.zeros(3, device=device, dtype=dtype)
        depth = torch.linalg.norm(xyz.to(device=device, dtype=dtype) - center.view(1, 3), dim=-1).clamp_min(self.depth_min)
        rows = torch.div(source_flat_idx.to(device=device, dtype=torch.long), int(W), rounding_mode="floor").to(dtype=dtype)
        lat = math.pi * 0.5 - (rows + 0.5) * math.pi / float(H)
        cos_lat = torch.cos(lat).clamp(float(self.pfgs360_gaussian_scale_lat_cos_min), 1.0)
        angular = math.sqrt((2.0 * math.pi / float(W)) * (math.pi / float(H))) * torch.sqrt(cos_lat)
        scale = float(self.pfgs360_gaussian_scale_factor) * depth * angular
        return scale.clamp(float(self.pfgs360_gaussian_scale_min), float(self.pfgs360_gaussian_scale_max))

    def _world_confidence(self, output: FrontendOutput, points: torch.Tensor, H: int, W: int) -> torch.Tensor:
        conf = output.world_points_confidence
        if conf is None:
            conf = output.depth_confidence
        if conf is None:
            return torch.ones((1, H, W), dtype=points.dtype, device=points.device)
        conf_t = conf.detach().float()
        if conf_t.ndim == 2:
            conf_t = conf_t.unsqueeze(0)
        if conf_t.ndim == 3 and conf_t.shape[0] != 1:
            raise ValueError(f"Expected confidence as 1xHxW, got {tuple(conf_t.shape)}")
        if tuple(conf_t.shape[-2:]) != (H, W):
            raise ValueError(f"World-points confidence shape {tuple(conf_t.shape[-2:])} does not match image {(H, W)}")
        return conf_t.to(device=points.device, dtype=points.dtype)

    def _world_valid_mask(self, output: FrontendOutput, points: torch.Tensor, H: int, W: int) -> torch.Tensor:
        valid_mask = output.valid_world_points_mask
        if valid_mask is None:
            return torch.ones((1, H, W), dtype=torch.bool, device=points.device)
        valid_t = valid_mask.detach().bool()
        if valid_t.ndim == 2:
            valid_t = valid_t.unsqueeze(0)
        if valid_t.ndim == 3 and valid_t.shape[0] != 1:
            raise ValueError(f"Expected valid_world_points_mask as 1xHxW, got {tuple(valid_t.shape)}")
        if tuple(valid_t.shape[-2:]) != (H, W):
            raise ValueError(f"World-points mask shape {tuple(valid_t.shape[-2:])} does not match image {(H, W)}")
        return valid_t.to(device=points.device)

    def _configured_sky_mask(
        self,
        image: torch.Tensor,
        size: tuple[int, int],
        *,
        device: torch.device,
        insertion_hints: dict | None,
    ) -> torch.Tensor:
        hints = insertion_hints or {}
        hint_sky = self._hint_mask(hints.get("sky_mask"), size, device=device)
        if self.sky_mask_source in {"panovggt", "panovggt_head", "pano_vggt", "m3", "m3_head"}:
            if hint_sky is None:
                raise ValueError(
                    "Mapping.sky_mask_source=panovggt_head requires insertion_hints['sky_mask']."
                )
            return hint_sky
        if self.sky_mask_source in {"hint", "frontend", "frontend_hint"} and hint_sky is not None:
            return hint_sky
        return self._sky_mask_from_image(image).to(device=device)

    @staticmethod
    def _hint_field(
        value: torch.Tensor | None,
        size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        field = value.detach().float()
        if field.ndim == 2:
            field = field.unsqueeze(0)
        if field.ndim != 3 or int(field.shape[0]) != 1:
            raise ValueError(f"Insertion hint must have shape HxW or 1xHxW, got {tuple(value.shape)}")
        if tuple(field.shape[-2:]) != tuple(size):
            field = F.interpolate(field.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
        return field.to(device=device, dtype=dtype)

    @classmethod
    def _hint_mask(
        cls,
        value: torch.Tensor | None,
        size: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        field = cls._hint_field(value, size, device=device, dtype=torch.float32)
        if field is None:
            return None
        return (field > 0.5).to(device=device)

    @staticmethod
    def _best_positions_by_score(score: torch.Tensor, inverse: torch.Tensor, n_voxels: int) -> torch.Tensor:
        positions = torch.arange(score.shape[0], device=score.device, dtype=torch.long)
        if hasattr(torch.Tensor, "scatter_reduce_"):
            best_score = torch.full((n_voxels,), -float("inf"), device=score.device, dtype=score.dtype)
            best_score.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
            is_best = score >= best_score[inverse]
            sentinel = torch.full_like(positions, int(score.shape[0]))
            pos_candidates = torch.where(is_best, positions, sentinel)
            best_pos = torch.full((n_voxels,), int(score.shape[0]), device=score.device, dtype=torch.long)
            best_pos.scatter_reduce_(0, inverse, pos_candidates, reduce="amin", include_self=True)
            return best_pos.clamp_max(max(0, int(score.shape[0]) - 1))
        best = []
        for idx in range(n_voxels):
            rows = torch.nonzero(inverse == idx, as_tuple=False).flatten()
            if rows.numel() == 0:
                best.append(0)
            else:
                best.append(int(rows[torch.argmax(score.index_select(0, rows))]))
        return torch.tensor(best, dtype=torch.long, device=score.device)

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
            insert_enabled=torch.zeros(0, dtype=torch.bool, device=device),
            insert_score=torch.zeros(0, device=device),
            grid_coord=torch.zeros(0, 3, dtype=torch.int32, device=device),
        )
