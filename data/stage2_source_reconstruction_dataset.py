"""Source-only ERP clip dataset for Stage 2 Gaussian-head training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .stage1_pano_sequence_dataset import Stage1ManifestRecord, load_erp_image, load_stage1_manifest


def _frame_sort_key(record: Stage1ManifestRecord) -> tuple[float, str]:
    if record.timestamp is not None:
        return float(record.timestamp), str(record.frame_id)
    try:
        return float(record.frame_id), str(record.frame_id)
    except (TypeError, ValueError):
        return 0.0, str(record.frame_id)


@dataclass(frozen=True)
class Stage2SourceSampleIndex:
    sequence_key: tuple[str, str]
    start: int


class Stage2SourceReconstructionDataset(Dataset):
    """Build source-only clips with deterministic per-epoch random stride."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str = "train",
        domains: list[str] | tuple[str, ...] | None = None,
        views_per_sample: int = 4,
        stride_min: int = 2,
        stride_max: int = 6,
        image_height: int = 504,
        image_width: int = 1008,
        seed: int = 1234,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.manifest_dir = self.manifest_path.resolve().parent
        self.split = str(split)
        self.views_per_sample = int(views_per_sample)
        self.stride_min = max(1, int(stride_min))
        self.stride_max = max(self.stride_min, int(stride_max))
        self.resize = (int(image_height), int(image_width))
        self.seed = int(seed)
        self.epoch = 0
        if self.views_per_sample < 2:
            raise ValueError("views_per_sample must be at least 2.")
        domain_set = {str(value).lower() for value in domains} if domains else None
        records = load_stage1_manifest(self.manifest_path)
        selected = [
            record
            for record in records
            if str(record.split) == self.split and (domain_set is None or record.domain in domain_set)
        ]
        grouped: dict[tuple[str, str], list[Stage1ManifestRecord]] = {}
        for record in selected:
            grouped.setdefault((record.scene_id, record.sequence_id), []).append(record)
        self.sequences = {key: sorted(values, key=_frame_sort_key) for key, values in grouped.items()}
        minimum_span = (self.views_per_sample - 1) * self.stride_min + 1
        self.sample_indices: list[Stage2SourceSampleIndex] = []
        for key, sequence in sorted(self.sequences.items()):
            for start in range(max(0, len(sequence) - minimum_span + 1)):
                self.sample_indices.append(Stage2SourceSampleIndex(sequence_key=key, start=start))
        if max_samples is not None:
            self.sample_indices = self.sample_indices[: max(0, int(max_samples))]
        if not self.sample_indices:
            raise ValueError(f"No Stage 2 source clips were built for split={self.split!r}.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _stride_for_index(self, index: int, sequence_length: int, start: int) -> int:
        maximum_allowed = (sequence_length - 1 - int(start)) // (self.views_per_sample - 1)
        upper = min(self.stride_max, maximum_allowed)
        if upper < self.stride_min:
            raise RuntimeError("Stage 2 sample index no longer has enough source frames.")
        if self.split != "train" or upper == self.stride_min:
            return self.stride_min
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + int(index))
        return int(torch.randint(self.stride_min, upper + 1, (1,), generator=generator).item())

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = self.sample_indices[int(index)]
        sequence = self.sequences[sample_index.sequence_key]
        stride = self._stride_for_index(int(index), len(sequence), sample_index.start)
        records = [
            sequence[sample_index.start + view * stride]
            for view in range(self.views_per_sample)
        ]
        images = torch.stack(
            [
                load_erp_image(
                    Path(record.rgb_path) if Path(record.rgb_path).is_absolute() else self.manifest_dir / record.rgb_path,
                    self.resize,
                )
                for record in records
            ],
            dim=0,
        )
        numeric_ids: list[int] = []
        for offset, record in enumerate(records):
            try:
                numeric_ids.append(int(record.frame_id))
            except (TypeError, ValueError):
                numeric_ids.append(int(sample_index.start + offset * stride))
        return {
            "images": images,
            "frame_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "frame_id_strings": [str(record.frame_id) for record in records],
            "scene_id": records[0].scene_id,
            "sequence_id": records[0].sequence_id,
            "stride": int(stride),
            "split": self.split,
        }


class SyntheticStage2SourceDataset(Dataset):
    """Small deterministic source-only clips used by unit and smoke tests."""

    def __init__(
        self,
        *,
        length: int = 4,
        views_per_sample: int = 3,
        height: int = 16,
        width: int = 32,
    ) -> None:
        self.length = int(length)
        self.views_per_sample = int(views_per_sample)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(1729 + int(index))
        base = torch.rand(3, self.height, self.width, generator=generator)
        images = torch.stack(
            [torch.roll(base, shifts=-view, dims=-1) for view in range(self.views_per_sample)],
            dim=0,
        )
        return {
            "images": images,
            "frame_ids": torch.arange(self.views_per_sample, dtype=torch.long) + int(index) * 100,
            "frame_id_strings": [f"synthetic_{index}_{view}" for view in range(self.views_per_sample)],
            "scene_id": "synthetic",
            "sequence_id": f"synthetic_{index}",
            "stride": 1,
            "split": "train",
        }


def stage2_source_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage 2 batch.")
    return {
        "images": torch.stack([sample["images"] for sample in batch], dim=0),
        "frame_ids": torch.stack([sample["frame_ids"] for sample in batch], dim=0),
        "frame_id_strings": [sample["frame_id_strings"] for sample in batch],
        "scene_ids": [sample["scene_id"] for sample in batch],
        "sequence_ids": [sample["sequence_id"] for sample in batch],
        "strides": torch.tensor([sample["stride"] for sample in batch], dtype=torch.long),
        "split": [sample["split"] for sample in batch],
    }
