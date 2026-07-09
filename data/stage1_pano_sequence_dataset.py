"""Manifest dataset for Stage 1 spherical Selfi adapter training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("scene_id", "sequence_id", "frame_id", "rgb_path", "split", "domain")
VALID_DOMAINS = {"indoor", "outdoor"}


@dataclass(frozen=True)
class Stage1ManifestRecord:
    """One ERP frame entry from the Stage 1 manifest."""

    scene_id: str
    sequence_id: str
    frame_id: int | str
    rgb_path: str
    depth_path: str | None = None
    pose_path: str | None = None
    timestamp: float | None = None
    split: str = "train"
    domain: str = "indoor"


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_from_mapping(raw: dict[str, Any]) -> Stage1ManifestRecord:
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"Stage 1 manifest record is missing required field(s): {missing}.")
    domain = str(raw["domain"]).lower()
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Stage 1 manifest domain must be indoor/outdoor, got {raw['domain']!r}.")
    return Stage1ManifestRecord(
        scene_id=str(raw["scene_id"]),
        sequence_id=str(raw["sequence_id"]),
        frame_id=raw["frame_id"],
        rgb_path=str(raw["rgb_path"]),
        depth_path=None if raw.get("depth_path") in (None, "") else str(raw.get("depth_path")),
        pose_path=None if raw.get("pose_path") in (None, "") else str(raw.get("pose_path")),
        timestamp=None if raw.get("timestamp") is None else float(raw["timestamp"]),
        split=str(raw.get("split", "train")),
        domain=domain,
    )


def load_stage1_manifest(path: str | Path) -> list[Stage1ManifestRecord]:
    """Load and validate a Stage 1 manifest JSON file."""

    raw = _read_json(path)
    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("Stage 1 manifest must be a list or a mapping with a 'records' list.")
    return [_record_from_mapping(record) for record in records]


def summarize_stage1_manifest(records: list[Stage1ManifestRecord]) -> dict[str, Any]:
    """Return small manifest summary stats for tests and tooling."""

    domains: dict[str, int] = {}
    splits: dict[str, int] = {}
    sequences: set[tuple[str, str]] = set()
    for record in records:
        domains[record.domain] = domains.get(record.domain, 0) + 1
        splits[record.split] = splits.get(record.split, 0) + 1
        sequences.add((record.scene_id, record.sequence_id))
    return {
        "num_records": len(records),
        "domains": domains,
        "splits": splits,
        "num_sequences": len(sequences),
    }


def _frame_sort_key(record: Stage1ManifestRecord) -> tuple[float, str]:
    if record.timestamp is not None:
        return float(record.timestamp), str(record.frame_id)
    try:
        return float(record.frame_id), str(record.frame_id)
    except (TypeError, ValueError):
        return 0.0, str(record.frame_id)


def _frame_gap_ok(a: Stage1ManifestRecord, b: Stage1ManifestRecord, max_gap: int | None) -> bool:
    if max_gap is None:
        return True
    try:
        return abs(int(b.frame_id) - int(a.frame_id)) <= int(max_gap)
    except (TypeError, ValueError):
        return True


def build_stage1_windows(
    records: list[Stage1ManifestRecord],
    *,
    views_per_sample: int = 4,
    split: str = "train",
    domains: list[str] | tuple[str, ...] | None = None,
    max_temporal_gap: int | None = 10,
) -> list[list[Stage1ManifestRecord]]:
    """Build deterministic same-sequence windows from manifest records."""

    domain_set = {str(domain).lower() for domain in domains} if domains else None
    selected = [
        record
        for record in records
        if str(record.split) == str(split) and (domain_set is None or record.domain in domain_set)
    ]
    grouped: dict[tuple[str, str], list[Stage1ManifestRecord]] = {}
    for record in selected:
        grouped.setdefault((record.scene_id, record.sequence_id), []).append(record)
    windows: list[list[Stage1ManifestRecord]] = []
    v = int(views_per_sample)
    if v < 2:
        raise ValueError("views_per_sample must be at least 2.")
    for group in grouped.values():
        ordered = sorted(group, key=_frame_sort_key)
        if len(ordered) < v:
            continue
        for start in range(0, len(ordered) - v + 1):
            window = ordered[start : start + v]
            if all(_frame_gap_ok(a, b, max_temporal_gap) for a, b in zip(window[:-1], window[1:])):
                windows.append(window)
    return windows


def build_adjacent_and_skip_pairs(views_per_sample: int) -> torch.Tensor:
    """Build adjacent plus one-skip forward view pairs."""

    v = int(views_per_sample)
    pairs: list[tuple[int, int]] = []
    for idx in range(v - 1):
        pairs.append((idx, idx + 1))
    for idx in range(v - 2):
        pairs.append((idx, idx + 2))
    return torch.tensor(pairs, dtype=torch.long)


def _resolve_path(path: str | None, manifest_dir: Path) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else manifest_dir / value


def load_erp_image(path: Path, resize: tuple[int, int] | None) -> torch.Tensor:
    """Load an ERP RGB image as ``3 x H x W`` float tensor in [0, 1]."""

    image = Image.open(path).convert("RGB")
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_optional_tensor(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu")
        return value.float() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix == ".json":
        return torch.as_tensor(_read_json(path), dtype=torch.float32)
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - exercised only without h5py installed.
            raise RuntimeError("Reading .h5 Stage 1 tensors requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            key = "depth" if "depth" in handle else next(iter(handle.keys()))
            return torch.from_numpy(np.asarray(handle[key], dtype=np.float32)).contiguous()
    image = Image.open(path)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return torch.from_numpy(arr).unsqueeze(0).contiguous()
    return torch.from_numpy(arr).float()


def _resize_depth_tensor(value: torch.Tensor, resize: tuple[int, int]) -> torch.Tensor:
    if value.ndim == 2:
        tensor = value.unsqueeze(0).unsqueeze(0)
        squeeze = "hw"
    elif value.ndim == 3 and int(value.shape[0]) == 1:
        tensor = value.unsqueeze(0)
        squeeze = "chw"
    else:
        return value
    if tuple(tensor.shape[-2:]) == tuple(resize):
        return value
    out = F.interpolate(tensor.float(), size=resize, mode="nearest")
    if squeeze == "hw":
        return out[0, 0]
    return out[0]


class Stage1PanoSequenceDataset(Dataset):
    """Windowed ERP sequence dataset backed by a Stage 1 manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        domains: list[str] | tuple[str, ...] | None = None,
        views_per_sample: int = 4,
        image_height: int = 504,
        image_width: int = 1008,
        pair_mode: str = "adjacent_and_skip",
        max_temporal_gap: int | None = 10,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.records = load_stage1_manifest(self.manifest_path)
        self.views_per_sample = int(views_per_sample)
        self.resize = (int(image_height), int(image_width))
        self.pair_mode = str(pair_mode)
        self.windows = build_stage1_windows(
            self.records,
            views_per_sample=self.views_per_sample,
            split=split,
            domains=domains,
            max_temporal_gap=max_temporal_gap,
        )
        if not self.windows:
            raise ValueError("Stage 1 dataset built zero windows from the provided manifest.")

    def __len__(self) -> int:
        return len(self.windows)

    def _pair_indices(self) -> torch.Tensor:
        if self.pair_mode != "adjacent_and_skip":
            raise ValueError(f"Unsupported Stage 1 pair_mode: {self.pair_mode!r}.")
        return build_adjacent_and_skip_pairs(self.views_per_sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[int(index)]
        images = torch.stack(
            [
                load_erp_image(_resolve_path(record.rgb_path, self.manifest_dir), self.resize)  # type: ignore[arg-type]
                for record in window
            ],
            dim=0,
        )
        depth_values = [
            None if (value := _load_optional_tensor(_resolve_path(record.depth_path, self.manifest_dir))) is None
            else _resize_depth_tensor(value, self.resize)
            for record in window
        ]
        pose_values = [_load_optional_tensor(_resolve_path(record.pose_path, self.manifest_dir)) for record in window]
        depths = torch.stack([value if value.ndim == 3 else value.unsqueeze(0) for value in depth_values], dim=0) if all(value is not None for value in depth_values) else None
        poses = torch.stack([value.reshape(4, 4) for value in pose_values], dim=0) if all(value is not None for value in pose_values) else None
        return {
            "images": images,
            "depths": depths,
            "poses_c2w": poses,
            "pair_indices": self._pair_indices(),
            "frame_ids": [record.frame_id for record in window],
            "scene_id": window[0].scene_id,
            "sequence_id": window[0].sequence_id,
            "domain": window[0].domain,
            "split": window[0].split,
        }


def stage1_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Stage 1 samples while preserving optional tensors."""

    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if all(torch.is_tensor(value) for value in values):
            out[key] = torch.stack(values, dim=0)
        elif all(value is None for value in values):
            out[key] = None
        else:
            out[key] = values
    return out
