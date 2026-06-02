"""ERP sequence datasets for PanoDROID training."""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .spherical_ba import se3_exp


def load_erp_image(path: str, resize: Optional[tuple[int, int]] = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def discover_erp_images(root: str, sequence: Optional[str] = None) -> list[str]:
    root_path = Path(root)
    candidates = []
    if sequence:
        candidates.append(root_path / "Sequences" / sequence)
        candidates.append(root_path / sequence)
    candidates.extend([root_path / "pano_images", root_path / "images", root_path / "rgb", root_path])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    for folder in candidates:
        if not folder.is_dir():
            continue
        files: list[str] = []
        for ext in exts:
            files.extend(glob.glob(str(folder / ext)))
        files = sorted(dict.fromkeys(files))
        if files:
            return files
    raise FileNotFoundError(f"No ERP images found under {root}")


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_optional_c2w_poses(root: str, image_count: int, sequence: Optional[str] = None) -> Optional[list[np.ndarray]]:
    root_path = Path(root)
    pose_candidates = [
        root_path / "poses.txt",
        root_path / "gt.txt",
        root_path / "pose.txt",
    ]
    if sequence:
        pose_candidates.append(root_path / "GroundTruth" / f"{sequence}.txt")
    pose_path = next((p for p in pose_candidates if p.is_file()), None)
    if pose_path is None:
        return None
    rows = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    if not rows:
        return None
    poses = []
    for row in rows[:image_count]:
        numeric = []
        for token in row:
            try:
                numeric.append(float(token))
            except ValueError:
                pass
        if len(numeric) == 16:
            T = np.asarray(numeric, dtype=np.float32).reshape(4, 4)
        elif len(numeric) >= 7:
            vals = numeric[-7:]
            tx, ty, tz, qx, qy, qz, qw = vals
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        else:
            return None
        poses.append(T)
    if len(poses) < image_count:
        return None
    origin = poses[0][:3, 3].copy()
    for T in poses:
        T[:3, 3] -= origin
    return poses


class PanoSequenceDataset(Dataset):
    """Image-pair dataset for ERP sequences."""

    def __init__(
        self,
        root: str,
        *,
        sequence: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        stride: int = 1,
        begin: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.root = str(root)
        self.sequence = sequence
        self.resize = resize
        self.stride = max(1, int(stride))
        images = discover_erp_images(root, sequence=sequence)
        images = images[int(begin) : end]
        if len(images) <= self.stride:
            raise ValueError("Need at least two images for pair training.")
        self.images = images
        self.poses_c2w = load_optional_c2w_poses(root, len(images), sequence=sequence)

    def __len__(self) -> int:
        return len(self.images) - self.stride

    def __getitem__(self, idx: int) -> dict:
        j = idx + self.stride
        image0 = load_erp_image(self.images[idx], self.resize)
        image1 = load_erp_image(self.images[j], self.resize)
        sample = {
            "image0": image0,
            "image1": image1,
            "frame_id0": torch.tensor(idx, dtype=torch.long),
            "frame_id1": torch.tensor(j, dtype=torch.long),
        }
        if self.poses_c2w is not None:
            c2w0 = torch.from_numpy(self.poses_c2w[idx])
            c2w1 = torch.from_numpy(self.poses_c2w[j])
            rel = torch.linalg.inv(c2w1) @ c2w0
            sample["gt_relative_pose"] = rel.float()
        return sample


class SyntheticPanoPairDataset(Dataset):
    """Deterministic tiny dataset for smoke training and tests."""

    def __init__(self, length: int = 16, height: int = 32, width: int = 64) -> None:
        self.length = int(length)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        gen = torch.Generator().manual_seed(int(idx))
        base = torch.rand(3, self.height, self.width, generator=gen)
        base = torch.nn.functional.avg_pool2d(base.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        shift_u = int(idx % 5) - 2
        shift_v = int(idx % 3) - 1
        image1 = torch.roll(base, shifts=(-shift_v, -shift_u), dims=(1, 2))
        flow = torch.zeros(2, self.height, self.width)
        flow[0].fill_(float(shift_u))
        flow[1].fill_(float(shift_v))
        inv = torch.full((1, self.height, self.width), 0.2 + 0.01 * (idx % 4))
        xi = torch.tensor(
            [0.01 * shift_u, 0.002 * shift_v, 0.0, 0.0, 0.0, 0.001 * shift_u],
            dtype=torch.float32,
        )
        return {
            "image0": base,
            "image1": image1,
            "gt_flow": flow,
            "gt_inverse_depth": inv,
            "gt_relative_pose": se3_exp(xi),
            "frame_id0": torch.tensor(idx, dtype=torch.long),
            "frame_id1": torch.tensor(idx + 1, dtype=torch.long),
        }


def build_dataset_from_config(config: dict, *, train: bool = True) -> Dataset:
    ds_cfg = config.get("Dataset", {})
    if ds_cfg.get("synthetic", False):
        return SyntheticPanoPairDataset(
            length=int(ds_cfg.get("synthetic_length", 16)),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
        )
    resize = None
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    if h is not None and w is not None:
        resize = (int(h), int(w))
    root = ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.dataset_path is required unless Dataset.synthetic=true.")
    return PanoSequenceDataset(
        root,
        sequence=ds_cfg.get("sequence"),
        resize=resize,
        stride=int(ds_cfg.get("pair_stride", 1)),
        begin=int(ds_cfg.get("begin", 0)),
        end=ds_cfg.get("end"),
    )
