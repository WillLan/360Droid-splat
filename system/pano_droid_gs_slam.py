"""PanoDROID front-end plus panoramic Gaussian backend SLAM runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper
from frontend.pano_droid.adapter import build_frontend_from_config
from frontend.pano_droid.dataset import discover_erp_images, load_erp_image
from frontend.pano_droid.interfaces import PanoFrame
from mapping.gaussian_initializer import GaussianInitializer


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_sequence_frames(config: dict) -> Iterable[PanoFrame]:
    ds_cfg = config.get("Dataset", {})
    if ds_cfg.get("synthetic", False):
        from frontend.pano_droid.dataset import SyntheticPanoPairDataset

        ds = SyntheticPanoPairDataset(
            length=int(ds_cfg.get("synthetic_length", 4)),
            height=int(ds_cfg.get("height", ds_cfg.get("erp_resize_height", 32))),
            width=int(ds_cfg.get("width", ds_cfg.get("erp_resize_width", 64))),
        )
        yielded_first = False
        for idx in range(len(ds)):
            sample = ds[idx]
            if not yielded_first:
                yielded_first = True
                yield PanoFrame(image=sample["image0"], timestamp=float(idx), frame_id=idx)
            yield PanoFrame(image=sample["image1"], timestamp=float(idx + 1), frame_id=idx + 1)
        return

    root = ds_cfg.get("dataset_path")
    if root is None:
        raise ValueError("Dataset.dataset_path is required unless Dataset.synthetic=true.")
    files = discover_erp_images(root, sequence=ds_cfg.get("sequence"))
    begin = int(ds_cfg.get("begin", 0))
    end = ds_cfg.get("end")
    files = files[begin:end]
    h = ds_cfg.get("erp_resize_height")
    w = ds_cfg.get("erp_resize_width")
    resize = (int(h), int(w)) if h is not None and w is not None else None
    for local_idx, path in enumerate(files):
        frame_id = begin + local_idx
        yield PanoFrame(
            image=load_erp_image(path, resize=resize),
            timestamp=float(frame_id),
            frame_id=frame_id,
            meta={"path": path},
        )


class PanoDroidGSSlamSystem:
    """Small orchestration layer matching the original SLAM staging."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.frontend = build_frontend_from_config(config)
        self.initializer = GaussianInitializer(
            max_seeds_per_keyframe=int(config.get("Mapping", {}).get("max_seeds_per_keyframe", 2048)),
            min_confidence=float(config.get("Mapping", {}).get("min_depth_confidence", 0.15)),
            voxel_sizes=tuple(config.get("Hierarchical", {}).get("voxel_size_lis", [0.12, 0.45, 1.8])),
        )
        self.map = PanoGaussianMap(config=config)
        render_cfg = config.get("Renderer", {})
        self.renderer = PFGS360Renderer(
            config=config,
            extra_gsplat360_roots=list(render_cfg.get("extra_gsplat360_roots", [])),
            allow_fallback=bool(render_cfg.get("allow_smoke_fallback", True)),
        )
        self.mapper = PanoGaussianMapper(
            self.map,
            renderer=self.renderer,
            lr=float(config.get("Mapping", {}).get("lr", 2e-3)),
        )

    def run(self, *, max_frames: int | None = None) -> dict:
        self.frontend.initialize({"config": self.config})
        output_dir = Path(self.config.get("Results", {}).get("save_dir", "outputs/pano_droid_gs_slam"))
        output_dir.mkdir(parents=True, exist_ok=True)
        refine_steps = int(self.config.get("Mapping", {}).get("refine_steps_per_keyframe", 0))
        frame_count = 0
        keyframes = 0
        last_status = None
        for frame in iter_sequence_frames(self.config):
            if max_frames is not None and frame_count >= int(max_frames):
                break
            out = self.frontend.track(frame)
            last_status = out.tracking_status
            if out.is_keyframe and out.inverse_depth is not None:
                seeds = self.initializer.from_frontend_output(out, frame.image)
                self.mapper.insert_keyframe(seeds, out)
                keyframes += 1
                if refine_steps > 0:
                    self.mapper.refine_on_keyframe(
                        image=frame.image,
                        c2w=out.pose_c2w,
                        steps=refine_steps,
                    )
            frame_count += 1

        summary = {
            "frames": frame_count,
            "keyframes": keyframes,
            "anchors": self.map.anchor_count(),
            "last_tracking_status": last_status,
            "map_mode": self.map.map_mode,
            "renderer": self.config.get("Training", {}).get("panorama_render_mode", "pfgs360_gsplat"),
            "notes": self.mapper.stats.notes,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pano_droid_gs_slam.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    system = PanoDroidGSSlamSystem(load_config(args.config))
    print(json.dumps(system.run(max_frames=args.max_frames), indent=2))


if __name__ == "__main__":
    main()

