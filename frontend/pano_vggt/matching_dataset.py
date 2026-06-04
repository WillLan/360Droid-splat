"""Datasets for PanoVGGT-M3-Sphere staged head training."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from frontend.pano_droid.dataset import load_erp_image
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid

TrainingMode = Literal["sky_only", "matching_only", "head_joint_calibration"]
TrainingSplit = Literal["train", "val", "all"]


@dataclass(frozen=True)
class Omni360Frame:
    """One indexed Omni360-Scene ERP frame and its optional supervision."""

    frame_id: int
    image_path: Path
    depth_path: Path | None
    semantic_path: Path | None
    c2w: torch.Tensor | None


_SCENE_LAYOUTS: dict[str, dict[str, Any]] = {
    "DTW": {
        "rgb_dirs": ("dtw_Raw",),
        "depth_dir": "dtw_Depth",
        "semantic_dirs": ("dtw_seg_panorama", "dtw_instance_panorama"),
        "pose_file": "DowntownWest_record.csv",
    },
    "NYC": {
        "rgb_dirs": ("nyc_Raw",),
        "depth_dir": "nyc_Depth",
        "semantic_dirs": ("nyc_seg_panorama", "nyc_instance_panorama"),
        "pose_file": "NYC_record.csv",
    },
    "CityPark": {
        "rgb_dirs": ("citypark_Raw_Part1", "citypark_Raw_Part2", "citypark_Raw_Part3"),
        "depth_dir": "citypark_Depth",
        "semantic_dirs": ("cpk_seg_panorama", "cpk_instance_panorama"),
        "pose_file": "CityPark_record.csv",
    },
}
_IMAGE_ID_RE = re.compile(r"panorama_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
_DEPTH_ID_RE = re.compile(r"Depth_(\d+)\.h5$", re.IGNORECASE)


def normalize_training_mode(mode: str) -> TrainingMode:
    """Normalize and validate a staged training mode string."""

    value = str(mode).lower()
    if value not in {"sky_only", "matching_only", "head_joint_calibration"}:
        raise ValueError(f"Unsupported PanoVGGT-M3-Sphere training mode: {mode!r}")
    return value  # type: ignore[return-value]


def build_temporal_pair_indices(
    n_frames: int,
    *,
    radius: int = 1,
    bidirectional: bool = False,
    max_pairs: int | None = None,
    sampling: Literal["all", "random", "linspace"] = "all",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create temporal training edges for a clip of ``n_frames`` frames.

    ``radius`` defines the candidate temporal neighborhood. ``max_pairs`` can
    then sub-sample that candidate set so larger windows do not force all
    pairwise edges into every optimization step.
    """

    edges: list[tuple[int, int]] = []
    for i in range(int(n_frames)):
        for d in range(1, int(radius) + 1):
            j = i + d
            if j >= int(n_frames):
                continue
            edges.append((i, j))
            if bidirectional:
                edges.append((j, i))
    if not edges:
        raise ValueError("No temporal pair indices could be built.")
    pairs = torch.tensor(edges, dtype=torch.long)
    if max_pairs is None or int(max_pairs) <= 0 or int(max_pairs) >= int(pairs.shape[0]):
        return pairs
    count = int(max_pairs)
    mode = str(sampling).lower()
    if mode == "all":
        return pairs[:count]
    if mode == "random":
        keep = torch.randperm(int(pairs.shape[0]), generator=generator)[:count]
        return pairs[keep]
    if mode == "linspace":
        keep = torch.linspace(0, int(pairs.shape[0]) - 1, steps=count).round().long()
        return pairs[keep]
    raise ValueError(f"Unsupported pair sampling mode: {sampling!r}")


def validate_training_sample(sample: dict[str, Any], mode: str, *, allow_fallback_mode: bool = False) -> None:
    """Validate that a sample contains supervision required by ``mode``."""

    training_mode = normalize_training_mode(mode)
    has_pose_value = sample.get("has_pose", False)
    has_sky_value = sample.get("has_sky", False)
    has_pose_flag = bool(has_pose_value.all()) if torch.is_tensor(has_pose_value) else bool(has_pose_value)
    has_sky_flag = bool(has_sky_value.all()) if torch.is_tensor(has_sky_value) else bool(has_sky_value)
    has_pose = has_pose_flag and torch.is_tensor(sample.get("poses_c2w"))
    has_depth = torch.is_tensor(sample.get("depths")) and torch.is_tensor(sample.get("valid_depth"))
    has_sky = has_sky_flag and torch.is_tensor(sample.get("sky_mask"))
    if training_mode in ("matching_only", "head_joint_calibration") and not (has_pose and has_depth):
        if allow_fallback_mode:
            return
        raise ValueError(f"{training_mode} requires RGB, depth, and pose supervision; got has_pose={has_pose}, has_depth={has_depth}.")
    if training_mode in ("sky_only", "head_joint_calibration") and not has_sky:
        if allow_fallback_mode:
            return
        raise ValueError(f"{training_mode} requires semantic sky supervision; got has_sky={has_sky}.")


def _collect_id_paths(folder: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not folder.is_dir():
        return out
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in out:
            raise ValueError(f"Duplicate frame id {frame_id} in {folder}.")
        out[frame_id] = path
    return out


def _collect_images(scene_dir: Path, rgb_dirs: tuple[str, ...]) -> dict[int, Path]:
    merged: dict[int, Path] = {}
    for name in rgb_dirs:
        for frame_id, path in _collect_id_paths(scene_dir / name, _IMAGE_ID_RE).items():
            if frame_id in merged:
                raise ValueError(f"Duplicate RGB frame id {frame_id} in scene {scene_dir.name}.")
            merged[frame_id] = path
    return merged


def _read_h5_depth(path: Path) -> torch.Tensor:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on real-data environment
        raise ImportError("Omni360SceneTrainingDataset requires h5py to read .h5 depth files.") from exc
    with h5py.File(path, "r") as handle:
        key = "depth" if "depth" in handle else next(iter(handle.keys()))
        array = np.asarray(handle[key], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Depth file must contain a 2D array, got {array.shape}: {path}")
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _normalize_depth_format(depth_format: str) -> str:
    value = str(depth_format).lower()
    aliases = {
        "range": "euclidean_range",
        "spherical_range": "euclidean_range",
        "euclidean": "euclidean_range",
        "euclidean_range": "euclidean_range",
        "z": "z_depth",
        "z_depth": "z_depth",
        "planar_depth": "z_depth",
        "orthogonal_z": "z_depth",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported Dataset.depth_format. Expected euclidean_range or z_depth, "
            f"got {depth_format!r}."
        )
    return aliases[value]


def _convert_depth_to_euclidean_range(depth: torch.Tensor, depth_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an ERP depth map to spherical euclidean range depth."""

    if depth.ndim != 3 or int(depth.shape[0]) != 1:
        raise ValueError(f"depth must have shape 1xHxW, got {tuple(depth.shape)}")
    fmt = _normalize_depth_format(depth_format)
    valid = torch.ones_like(depth, dtype=torch.bool)
    if fmt == "euclidean_range":
        return depth, valid
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    uv = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    bearing_z = erp_pixel_to_bearing(uv, height, width)[..., 2].to(device=depth.device, dtype=depth.dtype)
    front = bearing_z > 1.0e-6
    converted = depth / bearing_z.clamp_min(1.0e-6).unsqueeze(0)
    return converted, front.unsqueeze(0)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _normalize_pose_coordinate_system(value: str) -> str:
    system = str(value).lower()
    aliases = {
        "project": "project_erp",
        "project_erp": "project_erp",
        "erp": "project_erp",
        "raw": "project_erp",
        "ue": "ue_airsim",
        "ue5": "ue_airsim",
        "airsim": "ue_airsim",
        "ue_airsim": "ue_airsim",
    }
    if system not in aliases:
        raise ValueError(
            "Unsupported Dataset.pose_coordinate_system. Expected project_erp or ue_airsim, "
            f"got {value!r}."
        )
    return aliases[system]


def _ue_to_project_axis_matrix() -> np.ndarray:
    """Return matrix mapping project ERP coordinates into UE/AirSim axes."""

    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _euler_degrees_to_c2w(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    *,
    scale: float,
    coordinate_system: str = "ue_airsim",
) -> torch.Tensor:
    """Convert pose CSV Euler fields into a project-convention c2w matrix."""

    r = math.radians(float(roll))
    p = math.radians(float(pitch))
    yw = math.radians(float(yaw))
    rotation = _rotation_z(yw) @ _rotation_y(p) @ _rotation_x(r)
    translation = np.asarray([x, y, z], dtype=np.float32) * float(scale)
    system = _normalize_pose_coordinate_system(coordinate_system)
    if system == "ue_airsim":
        axes = _ue_to_project_axis_matrix()
        rotation = axes.T @ rotation @ axes
        translation = axes.T @ translation
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return torch.from_numpy(pose)


def _load_pose_csv(path: Path, *, translation_scale: float, coordinate_system: str = "ue_airsim") -> dict[int, torch.Tensor]:
    if not path.is_file():
        return {}
    poses: dict[int, torch.Tensor] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            lower = {str(k).lower(): float(v) for k, v in row.items() if k is not None and str(v) != ""}
            if {"x", "y", "z", "roll", "pitch", "yaw"}.issubset(lower):
                roll = lower["roll"]
                pitch = lower["pitch"]
            elif {"x", "y", "z", "pitch", "roll", "yaw"}.issubset(lower):
                roll = lower["roll"]
                pitch = lower["pitch"]
            else:
                raise ValueError(f"Unsupported pose CSV header in {path}: {reader.fieldnames}")
            poses[idx] = _euler_degrees_to_c2w(
                lower["x"],
                lower["y"],
                lower["z"],
                roll,
                pitch,
                lower["yaw"],
                scale=translation_scale,
                coordinate_system=coordinate_system,
            )
    return poses


def validate_pose_rotation(poses_c2w: torch.Tensor, *, atol: float = 1.0e-3) -> None:
    """Raise if a pose tensor does not contain valid rotation matrices."""

    if poses_c2w.ndim < 3 or poses_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"poses_c2w must end with 4x4, got {tuple(poses_c2w.shape)}")
    rotations = poses_c2w[..., :3, :3].float()
    eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    err = rotations.transpose(-1, -2) @ rotations - eye
    det = torch.linalg.det(rotations)
    if float(err.abs().max()) > float(atol) or not torch.allclose(det, torch.ones_like(det), atol=atol, rtol=0.0):
        raise ValueError("poses_c2w contains invalid rotation matrices.")


def _load_semantic(path: Path, resize: tuple[int, int] | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image = Image.open(path)
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.NEAREST)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return torch.from_numpy(arr.astype(np.int64)).contiguous(), None
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3]
        alpha_values = np.unique(alpha)
        if alpha_values.size > 1 or int(alpha_values[0]) != 255:
            return torch.from_numpy(alpha.astype(np.int64)).contiguous(), None
    if arr.ndim == 3:
        rgb = torch.from_numpy(arr[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return None, rgb
    raise ValueError(f"Unsupported semantic image shape {arr.shape}: {path}")


def _sky_ids_from_class_map(class_map: dict[str, Any]) -> set[int]:
    ids = {int(v) for v in class_map.get("sky_ids", [])}
    sky_names = {str(v).lower() for v in class_map.get("sky_names", [])}
    classes = class_map.get("classes", {})
    if isinstance(classes, dict):
        for name, idx in classes.items():
            if str(name).lower() in sky_names:
                ids.add(int(idx))
    elif isinstance(classes, list):
        for item in classes:
            if isinstance(item, dict) and str(item.get("name", "")).lower() in sky_names and "id" in item:
                ids.add(int(item["id"]))
    return ids


def sky_mask_from_semantic(
    semantic_labels: torch.Tensor | None,
    semantic_rgb: torch.Tensor | None,
    class_map: dict[str, Any],
) -> torch.Tensor:
    """Build a sky mask from semantic labels or RGB colors using ``class_map``."""

    masks: list[torch.Tensor] = []
    if semantic_labels is not None:
        sky_ids = _sky_ids_from_class_map(class_map)
        if sky_ids:
            label_mask = torch.zeros_like(semantic_labels, dtype=torch.bool)
            for sky_id in sky_ids:
                label_mask |= semantic_labels.long() == int(sky_id)
            masks.append(label_mask)
    if semantic_rgb is not None:
        colors = class_map.get("sky_colors", [])
        if colors:
            rgb = semantic_rgb.float()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            color_mask = torch.zeros(rgb.shape[-2:], dtype=torch.bool, device=rgb.device)
            for color in colors:
                value = torch.tensor(color, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
                if float(value.max()) > 1.5:
                    value = value / 255.0
                color_mask |= (rgb - value).abs().amax(dim=0) <= 1.0 / 255.0
            masks.append(color_mask)
    if not masks:
        raise ValueError("Could not build sky_mask: configure Dataset.class_map.sky_ids or sky_colors.")
    out = masks[0]
    for mask in masks[1:]:
        out = out | mask
    return out.unsqueeze(0)


class Omni360SceneTrainingDataset(Dataset):
    """Omni360-Scene training clips for staged PanoVGGT-M3-Sphere heads."""

    def __init__(
        self,
        root: str,
        *,
        pose_root: str | None = None,
        scenes: list[str] | tuple[str, ...] = ("DTW", "NYC"),
        mode: str = "matching_only",
        frames_per_sample: int = 4,
        clip_stride: int = 1,
        temporal_radius: int = 1,
        bidirectional_pairs: bool = False,
        resize: tuple[int, int] | None = None,
        depth_format: str = "euclidean_range",
        depth_scale: float = 1.0,
        depth_invalid_value: float | None = 1000.0,
        pose_coordinate_system: str = "ue_airsim",
        pose_translation_scale: float = 0.01,
        split: TrainingSplit = "train",
        validation_fraction: float = 0.0,
        validation_split: Literal["tail"] = "tail",
        class_map: dict[str, Any] | None = None,
        allow_fallback_mode: bool = False,
        max_clips: int | None = None,
        pairs_per_sample: int | None = None,
        pair_sampling: Literal["all", "random", "linspace"] = "all",
    ) -> None:
        self.root = Path(root)
        self.pose_root = Path(pose_root) if pose_root is not None else None
        self.scenes = [str(scene) for scene in scenes]
        self.mode = normalize_training_mode(mode)
        self.split = str(split).lower()
        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported training split: {split!r}")
        self.frames_per_sample = int(frames_per_sample)
        self.clip_stride = max(1, int(clip_stride))
        self.temporal_radius = max(1, int(temporal_radius))
        self.bidirectional_pairs = bool(bidirectional_pairs)
        self.resize = resize
        self.depth_format = _normalize_depth_format(depth_format)
        self.depth_scale = float(depth_scale)
        self.depth_invalid_value = depth_invalid_value
        self.pose_coordinate_system = _normalize_pose_coordinate_system(pose_coordinate_system)
        self.pose_translation_scale = float(pose_translation_scale)
        self.validation_fraction = max(0.0, min(0.95, float(validation_fraction)))
        if str(validation_split).lower() != "tail":
            raise ValueError(f"Unsupported validation split mode: {validation_split!r}")
        self.class_map = class_map or {}
        self.allow_fallback_mode = bool(allow_fallback_mode)
        self.pairs_per_sample = None if pairs_per_sample is None else max(1, int(pairs_per_sample))
        self.pair_sampling = str(pair_sampling).lower()
        if self.pair_sampling not in {"all", "random", "linspace"}:
            raise ValueError(f"Unsupported pair_sampling mode: {pair_sampling!r}")
        if self.frames_per_sample < 2:
            raise ValueError("frames_per_sample must be at least 2.")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Omni360-Scene root does not exist: {self.root}")

        self.frames_by_scene: dict[str, list[Omni360Frame]] = {}
        self.clips: list[tuple[str, int]] = []
        span = (self.frames_per_sample - 1) * self.clip_stride + 1
        for scene in self.scenes:
            frames = self._load_scene(scene)
            if len(frames) < span:
                continue
            self.frames_by_scene[scene] = frames
            starts = self._split_starts(len(frames), span)
            for start in starts:
                self.clips.append((scene, start))
        if max_clips is not None:
            self.clips = self.clips[: int(max_clips)]
        if not self.clips:
            raise ValueError("No Omni360 training clips were built.")

    def _split_starts(self, n_frames: int, span: int) -> range:
        if self.validation_fraction <= 0.0 or self.split == "all":
            return range(0, int(n_frames) - int(span) + 1)
        val_start = int(math.floor(float(n_frames) * (1.0 - self.validation_fraction)))
        val_start = min(max(0, val_start), int(n_frames))
        if self.split == "val":
            last = int(n_frames) - int(span)
            if val_start > last:
                return range(0, 0)
            return range(val_start, last + 1)
        train_last_exclusive = max(0, val_start - int(span) + 1)
        return range(0, train_last_exclusive)

    def _load_scene(self, scene: str) -> list[Omni360Frame]:
        if scene not in _SCENE_LAYOUTS:
            raise ValueError(f"Unsupported Omni360 scene {scene!r}.")
        layout = _SCENE_LAYOUTS[scene]
        scene_dir = self.root / scene
        images = _collect_images(scene_dir, tuple(layout["rgb_dirs"]))
        depth_paths = _collect_id_paths(scene_dir / str(layout["depth_dir"]), _DEPTH_ID_RE)
        semantic_paths: dict[int, Path] = {}
        for semantic_dir in layout["semantic_dirs"]:
            for frame_id, path in _collect_id_paths(scene_dir / str(semantic_dir), _IMAGE_ID_RE).items():
                semantic_paths.setdefault(frame_id, path)
        pose_file = self.pose_root / str(layout["pose_file"]) if self.pose_root is not None else Path("")
        poses = (
            _load_pose_csv(
                pose_file,
                translation_scale=self.pose_translation_scale,
                coordinate_system=self.pose_coordinate_system,
            )
            if self.pose_root is not None
            else {}
        )

        frame_ids = sorted(images)
        frames: list[Omni360Frame] = []
        for frame_id in frame_ids:
            frames.append(
                Omni360Frame(
                    frame_id=frame_id,
                    image_path=images[frame_id],
                    depth_path=depth_paths.get(frame_id),
                    semantic_path=semantic_paths.get(frame_id),
                    c2w=poses.get(frame_id),
                )
            )
        return frames

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene, start = self.clips[int(index)]
        frames = self.frames_by_scene[scene]
        selected = [frames[start + i * self.clip_stride] for i in range(self.frames_per_sample)]
        images = torch.stack([load_erp_image(str(frame.image_path), self.resize) for frame in selected], dim=0)
        depths: torch.Tensor | None = None
        valid_depth: torch.Tensor | None = None
        if all(frame.depth_path is not None for frame in selected):
            depth_list = []
            depth_format_valid_list = []
            for frame in selected:
                depth = _read_h5_depth(frame.depth_path) * self.depth_scale  # type: ignore[arg-type]
                if self.resize is not None:
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),
                        size=self.resize,
                        mode="nearest",
                    ).squeeze(0)
                depth, format_valid = _convert_depth_to_euclidean_range(depth, self.depth_format)
                depth_list.append(depth)
                depth_format_valid_list.append(format_valid)
            depths = torch.stack(depth_list, dim=0)
            depth_format_valid = torch.stack(depth_format_valid_list, dim=0)
            valid_depth = torch.isfinite(depths) & (depths > 0.0) & depth_format_valid
            if self.depth_invalid_value is not None:
                valid_depth &= depths < float(self.depth_invalid_value)
                depths = torch.where(valid_depth, depths, torch.zeros_like(depths))

        poses = None
        if all(frame.c2w is not None for frame in selected):
            poses = torch.stack([frame.c2w for frame in selected if frame.c2w is not None], dim=0).float()
            validate_pose_rotation(poses)

        sem_labels_list: list[torch.Tensor] = []
        sem_rgb_list: list[torch.Tensor] = []
        sky_list: list[torch.Tensor] = []
        for frame in selected:
            if frame.semantic_path is None:
                continue
            labels, rgb = _load_semantic(frame.semantic_path, self.resize)
            if labels is not None:
                sem_labels_list.append(labels)
            if rgb is not None:
                sem_rgb_list.append(rgb)
            try:
                sky_list.append(sky_mask_from_semantic(labels, rgb, self.class_map))
            except ValueError:
                pass
        semantic_labels = torch.stack(sem_labels_list, dim=0) if len(sem_labels_list) == len(selected) else None
        semantic_rgb = torch.stack(sem_rgb_list, dim=0) if len(sem_rgb_list) == len(selected) else None
        sky_mask = torch.stack(sky_list, dim=0) if len(sky_list) == len(selected) else None

        sample: dict[str, Any] = {
            "images": images,
            "depths": depths,
            "valid_depth": valid_depth,
            "poses_c2w": poses,
            "semantic_labels": semantic_labels,
            "semantic_rgb": semantic_rgb,
            "sky_mask": sky_mask,
            "pair_indices": build_temporal_pair_indices(
                self.frames_per_sample,
                radius=self.temporal_radius,
                bidirectional=self.bidirectional_pairs,
                max_pairs=self.pairs_per_sample,
                sampling="linspace" if self.split == "val" and self.pair_sampling == "random" else self.pair_sampling,
            ),
            "frame_ids": [str(frame.frame_id) for frame in selected],
            "sequence_id": scene,
            "dataset_name": "omni360_scene",
            "split": self.split,
            "has_pose": poses is not None,
            "has_sky": sky_mask is not None,
        }
        validate_training_sample(sample, self.mode, allow_fallback_mode=self.allow_fallback_mode)
        return sample


class SyntheticOmni360TrainingDataset(Dataset):
    """Deterministic synthetic dataset for PanoVGGT-M3-Sphere tests."""

    def __init__(
        self,
        *,
        variant: Literal["complete", "no_pose", "no_sky"] = "complete",
        length: int = 4,
        n_frames: int = 3,
        height: int = 32,
        width: int = 64,
        mode: str = "matching_only",
        class_map: dict[str, Any] | None = None,
    ) -> None:
        self.variant = variant
        self.length = int(length)
        self.n_frames = int(n_frames)
        self.height = int(height)
        self.width = int(width)
        self.mode = normalize_training_mode(mode)
        self.class_map = class_map or {"sky_ids": [1], "classes": {"sky": 1}}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(int(index) + 17)
        base = torch.rand(3, self.height, self.width, generator=gen)
        base = torch.nn.functional.avg_pool2d(base.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        images = []
        depths = []
        poses = []
        labels = []
        for frame_idx in range(self.n_frames):
            images.append(torch.roll(base, shifts=-frame_idx, dims=2))
            depths.append(torch.full((1, self.height, self.width), 2.0 + 0.01 * frame_idx))
            pose = torch.eye(4)
            pose[0, 3] = 0.01 * frame_idx
            poses.append(pose)
            label = torch.zeros(self.height, self.width, dtype=torch.long)
            label[: max(1, self.height // 4)] = 1
            labels.append(label)
        images_t = torch.stack(images, dim=0)
        depths_t = torch.stack(depths, dim=0)
        labels_t = torch.stack(labels, dim=0)
        sky_mask = (labels_t == 1).unsqueeze(1)
        sample: dict[str, Any] = {
            "images": images_t,
            "depths": depths_t,
            "valid_depth": depths_t > 0.0,
            "poses_c2w": torch.stack(poses, dim=0),
            "semantic_labels": labels_t,
            "semantic_rgb": None,
            "sky_mask": sky_mask,
            "pair_indices": build_temporal_pair_indices(self.n_frames, radius=1, bidirectional=False),
            "frame_ids": [f"synthetic_{index}_{i}" for i in range(self.n_frames)],
            "sequence_id": "synthetic",
            "dataset_name": "synthetic_omni360",
            "has_pose": True,
            "has_sky": True,
        }
        if self.variant == "no_pose":
            sample["poses_c2w"] = None
            sample["has_pose"] = False
        if self.variant == "no_sky":
            sample["semantic_labels"] = None
            sample["sky_mask"] = None
            sample["has_sky"] = False
        validate_training_sample(sample, self.mode, allow_fallback_mode=False)
        return sample


def build_matching_dataset_from_config(config: dict[str, Any], *, split: TrainingSplit = "train") -> Dataset:
    """Build a real or synthetic staged matching/sky training dataset."""

    ds_cfg = config.get("Dataset", {})
    tr_cfg = config.get("Training", {})
    mode = normalize_training_mode(str(tr_cfg.get("mode", "matching_only")))
    if bool(ds_cfg.get("synthetic", False)):
        return SyntheticOmni360TrainingDataset(
            variant=ds_cfg.get("synthetic_variant", "complete"),
            length=int(ds_cfg.get("synthetic_length", 4)),
            n_frames=int(tr_cfg.get("frames_per_sample", ds_cfg.get("n_frames", 3))),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
            mode=mode,
            class_map=dict(ds_cfg.get("class_map", {})),
        )
    resize = None
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    if h is not None and w is not None:
        resize = (int(h), int(w))
    root = ds_cfg.get("root") or ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.root is required unless Dataset.synthetic=true.")
    return Omni360SceneTrainingDataset(
        root,
        pose_root=ds_cfg.get("pose_root"),
        scenes=list(ds_cfg.get("scenes", ["DTW", "NYC"])),
        mode=mode,
        frames_per_sample=int(tr_cfg.get("frames_per_sample", ds_cfg.get("n_frames", 4))),
        clip_stride=int(ds_cfg.get("clip_stride", 1)),
        temporal_radius=int(ds_cfg.get("temporal_radius", ds_cfg.get("pair_radius", 1))),
        bidirectional_pairs=bool(ds_cfg.get("bidirectional_pairs", False)),
        resize=resize,
        depth_format=str(ds_cfg.get("depth_format", "euclidean_range")),
        depth_scale=float(ds_cfg.get("depth_scale", 1.0)),
        depth_invalid_value=ds_cfg.get("depth_invalid_value", 1000.0),
        pose_coordinate_system=str(ds_cfg.get("pose_coordinate_system", "ue_airsim")),
        pose_translation_scale=float(ds_cfg.get("pose_translation_scale", 0.01)),
        split=split,
        validation_fraction=float(ds_cfg.get("validation_fraction", config.get("Validation", {}).get("fraction", 0.0))),
        validation_split=str(ds_cfg.get("validation_split", config.get("Validation", {}).get("split", "tail"))),
        class_map=dict(ds_cfg.get("class_map", {})),
        allow_fallback_mode=bool(ds_cfg.get("allow_fallback_mode", False)),
        max_clips=ds_cfg.get("max_clips"),
        pairs_per_sample=ds_cfg.get("pairs_per_sample", config.get("Pairs", {}).get("pairs_per_sample")),
        pair_sampling=str(ds_cfg.get("pair_sampling", config.get("Pairs", {}).get("pair_sampling", "all"))),
    )
