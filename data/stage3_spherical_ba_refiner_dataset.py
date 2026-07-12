"""Omni360 four-source clips for Stage 3 BA/refiner training."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from frontend.pano_vggt.matching_dataset import (
    Omni360SceneTrainingDataset,
    _convert_depth_to_euclidean_range,
    _read_h5_depth,
    load_erp_image,
    validate_pose_rotation,
)


class Stage3Omni360Dataset(Omni360SceneTrainingDataset):
    """Reuse the audited Omni360 pose/depth loader with random temporal stride."""

    def __init__(
        self,
        root: str,
        *,
        pose_root: str,
        scenes: list[str] | tuple[str, ...] = ("DTW", "NYC"),
        split: str = "train",
        views_per_sample: int = 4,
        stride_min: int = 2,
        stride_max: int = 6,
        resize: tuple[int, int] = (504, 1008),
        validation_fraction: float = 0.1,
        depth_format: str = "euclidean_range",
        depth_scale: float = 1.0,
        depth_invalid_value: float | None = 1000.0,
        pose_coordinate_system: str = "ue_airsim",
        pose_translation_scale: float = 0.01,
        seed: int = 1234,
        max_clips: int | None = None,
    ) -> None:
        self.stride_min = max(1, int(stride_min))
        self.stride_max = max(self.stride_min, int(stride_max))
        self.seed = int(seed)
        self.epoch = 0
        super().__init__(
            root,
            pose_root=pose_root,
            scenes=scenes,
            mode="matching_only",
            frames_per_sample=int(views_per_sample),
            clip_stride=self.stride_min,
            resize=resize,
            include_supervision=False,
            depth_format=depth_format,
            depth_scale=depth_scale,
            depth_invalid_value=depth_invalid_value,
            pose_coordinate_system=pose_coordinate_system,
            pose_translation_scale=pose_translation_scale,
            split=split,
            validation_fraction=validation_fraction,
            allow_fallback_mode=False,
            max_clips=max_clips,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sequence_bounds(self, scene: str) -> tuple[int, int]:
        length = len(self.frames_by_scene[scene])
        if self.validation_fraction <= 0.0 or self.split == "all":
            return 0, length - 1
        val_start = int(math.floor(float(length) * (1.0 - self.validation_fraction)))
        if self.split == "val":
            return val_start, length - 1
        return 0, max(0, val_start - 1)

    def _stride(self, index: int, scene: str, start: int) -> int:
        _, last = self._sequence_bounds(scene)
        maximum = (last - int(start)) // max(1, self.frames_per_sample - 1)
        upper = min(self.stride_max, maximum)
        if upper < self.stride_min:
            raise RuntimeError("Stage 3 clip cannot satisfy the configured minimum stride.")
        if self.split != "train" or upper == self.stride_min:
            return self.stride_min
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + int(index))
        return int(torch.randint(self.stride_min, upper + 1, (1,), generator=generator).item())

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene, start = self.clips[int(index)]
        stride = self._stride(int(index), scene, start)
        frames = self.frames_by_scene[scene]
        selected = [frames[start + view * stride] for view in range(self.frames_per_sample)]
        images = torch.stack([load_erp_image(str(frame.image_path), self.resize) for frame in selected], dim=0)

        if not all(frame.depth_path is not None for frame in selected):
            raise ValueError("Stage 3 Omni360 clips require GT depth for diagnostics.")
        depth_list: list[torch.Tensor] = []
        format_valid_list: list[torch.Tensor] = []
        for frame in selected:
            depth = _read_h5_depth(frame.depth_path) * self.depth_scale  # type: ignore[arg-type]
            if self.resize is not None and tuple(depth.shape[-2:]) != tuple(self.resize):
                depth = F.interpolate(depth.unsqueeze(0), size=self.resize, mode="nearest").squeeze(0)
            depth, format_valid = _convert_depth_to_euclidean_range(depth, self.depth_format)
            depth_list.append(depth)
            format_valid_list.append(format_valid)
        depths = torch.stack(depth_list, dim=0)
        valid_depth = torch.isfinite(depths) & (depths > 0.0) & torch.stack(format_valid_list, dim=0)
        if self.depth_invalid_value is not None:
            valid_depth &= depths < float(self.depth_invalid_value)
        depths = torch.where(valid_depth, depths, torch.zeros_like(depths))

        if not all(frame.c2w is not None for frame in selected):
            raise ValueError("Stage 3 Omni360 clips require GT camera poses for diagnostics.")
        poses = torch.stack([frame.c2w for frame in selected if frame.c2w is not None], dim=0).float()
        validate_pose_rotation(poses)
        return {
            "images": images,
            "gt_depths": depths,
            "gt_valid_depth": valid_depth,
            "gt_poses_c2w": poses,
            "frame_ids": torch.tensor([frame.frame_id for frame in selected], dtype=torch.long),
            "frame_id_strings": [str(frame.frame_id) for frame in selected],
            "scene_id": scene,
            "sequence_id": scene,
            "stride": stride,
            "split": self.split,
        }


class SyntheticStage3Dataset(Dataset):
    def __init__(self, *, length: int = 4, views: int = 4, height: int = 16, width: int = 32) -> None:
        self.length, self.views, self.height, self.width = int(length), int(views), int(height), int(width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(3907 + int(index))
        base = torch.rand(3, self.height, self.width, generator=generator)
        images = torch.stack([torch.roll(base, -view, dims=-1) for view in range(self.views)], dim=0)
        depth = torch.ones(self.views, 1, self.height, self.width)
        poses = torch.eye(4).view(1, 4, 4).repeat(self.views, 1, 1)
        poses[:, 0, 3] = torch.arange(self.views).float() * 0.05
        return {
            "images": images,
            "gt_depths": depth,
            "gt_valid_depth": torch.ones_like(depth, dtype=torch.bool),
            "gt_poses_c2w": poses,
            "frame_ids": torch.arange(self.views, dtype=torch.long) + int(index) * 100,
            "frame_id_strings": [f"synthetic_{index}_{view}" for view in range(self.views)],
            "scene_id": "synthetic",
            "sequence_id": f"synthetic_{index}",
            "stride": 1,
            "split": "train",
        }


def stage3_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage 3 batch.")
    tensor_keys = ("images", "gt_depths", "gt_valid_depth", "gt_poses_c2w", "frame_ids")
    out: dict[str, Any] = {key: torch.stack([sample[key] for sample in batch], dim=0) for key in tensor_keys}
    out.update(
        {
            "frame_id_strings": [sample["frame_id_strings"] for sample in batch],
            "scene_ids": [sample["scene_id"] for sample in batch],
            "sequence_ids": [sample["sequence_id"] for sample in batch],
            "strides": torch.tensor([sample["stride"] for sample in batch], dtype=torch.long),
            "split": [sample["split"] for sample in batch],
        }
    )
    return out

