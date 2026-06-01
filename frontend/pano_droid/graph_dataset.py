"""Multi-frame datasets for DROID-style PanoDROID training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .dataset import load_erp_image


@dataclass(frozen=True)
class PanoCityFrame:
    image_path: Path
    depth_path: Path
    c2w: np.ndarray
    frame_name: str
    block_name: str
    local_index: int


def _natural_block_key(path: Path) -> tuple[str, int]:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return path.name, int(digits or 0)


def _load_depth_png(
    path: Path,
    *,
    resize: Optional[tuple[int, int]] = None,
    depth_scale: float = 1.0,
    invalid_value: Optional[float] = 65535.0,
) -> torch.Tensor:
    image = Image.open(path)
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.NEAREST)
    arr = np.asarray(image, dtype=np.float32)
    depth = torch.from_numpy(arr).unsqueeze(0).contiguous() * float(depth_scale)
    if invalid_value is not None:
        invalid = arr >= float(invalid_value)
        if invalid.any():
            depth[:, torch.from_numpy(invalid)] = 0.0
    return depth


def _pose_json_for_block(block_dir: Path) -> Path:
    pose_files = sorted(block_dir.glob("*poses*.json"))
    if not pose_files:
        raise FileNotFoundError(f"No pose json found in {block_dir}")
    return next((p for p in pose_files if ".1." not in p.name), pose_files[0])


def _load_block_frames(
    block_dir: Path,
    *,
    depth_scale: float,
    depth_invalid_value: Optional[float],
) -> list[PanoCityFrame]:
    image_dir = block_dir / "pano_images"
    depth_dir = block_dir / "panodepth_images"
    if not image_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing pano_images or panodepth_images in {block_dir}")
    pose_path = _pose_json_for_block(block_dir)
    with open(pose_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", payload)
    if not isinstance(frames, list):
        raise ValueError(f"Pose json must contain a frame list: {pose_path}")

    out: list[PanoCityFrame] = []
    for idx, frame in enumerate(frames):
        image_path = image_dir / str(frame["name"])
        depth_path = depth_dir / str(frame["depth"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing PanoCity RGB image: {image_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing PanoCity depth image: {depth_path}")
        c2w = np.asarray(frame["transformation_matrix"], dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"Bad c2w shape in {pose_path} frame {idx}: {c2w.shape}")
        out.append(
            PanoCityFrame(
                image_path=image_path,
                depth_path=depth_path,
                c2w=c2w,
                frame_name=str(frame["name"]),
                block_name=block_dir.name,
                local_index=idx,
            )
        )
    # The arguments are intentionally part of this helper signature so the
    # dataset records the depth policy at construction time.
    _ = depth_scale, depth_invalid_value
    return out


class PanoCityGraphDataset(Dataset):
    """PanoCity block dataset returning contiguous multi-frame ERP clips."""

    def __init__(
        self,
        root: str,
        *,
        n_frames: int = 7,
        resize: Optional[tuple[int, int]] = (512, 1024),
        stride: int = 1,
        depth_scale: float = 1.0,
        depth_invalid_value: Optional[float] = 65535.0,
        blocks: Optional[list[str]] = None,
        max_clips: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.n_frames = int(n_frames)
        self.resize = resize
        self.stride = max(1, int(stride))
        self.depth_scale = float(depth_scale)
        self.depth_invalid_value = depth_invalid_value
        if self.n_frames < 2:
            raise ValueError("PanoCityGraphDataset requires n_frames >= 2.")
        if not self.root.is_dir():
            raise FileNotFoundError(f"PanoCity root does not exist: {self.root}")

        block_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("beijing_block")],
            key=_natural_block_key,
        )
        if blocks:
            wanted = set(blocks)
            block_dirs = [p for p in block_dirs if p.name in wanted]
        if not block_dirs:
            raise FileNotFoundError(f"No PanoCity block directories found under {self.root}")

        self.blocks: dict[str, list[PanoCityFrame]] = {}
        self.clips: list[tuple[str, int]] = []
        span = (self.n_frames - 1) * self.stride + 1
        for block_dir in block_dirs:
            frames = _load_block_frames(
                block_dir,
                depth_scale=self.depth_scale,
                depth_invalid_value=self.depth_invalid_value,
            )
            if len(frames) < span:
                continue
            self.blocks[block_dir.name] = frames
            for start in range(0, len(frames) - span + 1):
                self.clips.append((block_dir.name, start))
        if max_clips is not None:
            self.clips = self.clips[: int(max_clips)]
        if not self.clips:
            raise ValueError("No valid PanoCity clips were built.")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        block_name, start = self.clips[int(index)]
        frames = self.blocks[block_name]
        selected = [frames[start + i * self.stride] for i in range(self.n_frames)]
        images = torch.stack([load_erp_image(str(f.image_path), self.resize) for f in selected], dim=0)
        depths = torch.stack(
            [
                _load_depth_png(
                    f.depth_path,
                    resize=self.resize,
                    depth_scale=self.depth_scale,
                    invalid_value=self.depth_invalid_value,
                )
                for f in selected
            ],
            dim=0,
        )
        poses = torch.from_numpy(np.stack([f.c2w for f in selected], axis=0)).float()
        valid = depths > 0.0
        inverse_depths = torch.zeros_like(depths)
        inverse_depths[valid] = 1.0 / depths[valid].clamp_min(1e-6)
        return {
            "images": images,
            "depths": depths,
            "inverse_depths": inverse_depths,
            "poses_c2w": poses,
            "block_name": block_name,
            "frame_names": [f.frame_name for f in selected],
            "frame_indices": torch.tensor([f.local_index for f in selected], dtype=torch.long),
        }


class SyntheticPanoGraphDataset(Dataset):
    """Small graph dataset for train_graph smoke tests."""

    def __init__(
        self,
        *,
        length: int = 4,
        n_frames: int = 3,
        height: int = 32,
        width: int = 64,
    ) -> None:
        self.length = int(length)
        self.n_frames = int(n_frames)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        gen = torch.Generator().manual_seed(int(index))
        base = torch.rand(3, self.height, self.width, generator=gen)
        base = torch.nn.functional.avg_pool2d(base.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        images = []
        depths = []
        poses = []
        for i in range(self.n_frames):
            images.append(torch.roll(base, shifts=(0, -i), dims=(1, 2)))
            depths.append(torch.full((1, self.height, self.width), 5.0 + 0.1 * i))
            T = torch.eye(4)
            T[0, 3] = 0.05 * i
            poses.append(T)
        depths_t = torch.stack(depths, dim=0)
        return {
            "images": torch.stack(images, dim=0),
            "depths": depths_t,
            "inverse_depths": depths_t.reciprocal(),
            "poses_c2w": torch.stack(poses, dim=0),
            "block_name": "synthetic",
            "frame_names": [f"synthetic_{i:04d}.png" for i in range(self.n_frames)],
            "frame_indices": torch.arange(self.n_frames, dtype=torch.long),
        }


def build_graph_dataset_from_config(config: dict, *, train: bool = True) -> Dataset:
    ds_cfg = config.get("Dataset", {})
    if ds_cfg.get("synthetic", False):
        return SyntheticPanoGraphDataset(
            length=int(ds_cfg.get("synthetic_length", 4)),
            n_frames=int(ds_cfg.get("n_frames", 3)),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
        )
    root = ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.dataset_path is required unless Dataset.synthetic=true.")
    resize = None
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    if h is not None and w is not None:
        resize = (int(h), int(w))
    blocks = ds_cfg.get("blocks")
    if isinstance(blocks, str):
        blocks = [x.strip() for x in blocks.split(",") if x.strip()]
    return PanoCityGraphDataset(
        root,
        n_frames=int(ds_cfg.get("n_frames", 7)),
        resize=resize,
        stride=int(ds_cfg.get("clip_stride", 1)),
        depth_scale=float(ds_cfg.get("depth_scale", 1.0)),
        depth_invalid_value=ds_cfg.get("depth_invalid_value", 65535.0),
        blocks=blocks,
        max_clips=ds_cfg.get("max_clips"),
    )

