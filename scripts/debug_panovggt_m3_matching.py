"""Debug PanoVGGT-M3 matching inference and dense spherical factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

from frontend.pano_vggt.dense_matcher import PoseGuidedDenseMatcher
from frontend.pano_vggt.engine import build_panovggt_engine
from frontend.pano_vggt.factor_graph import DenseSphereFactor, DenseSphereFactorGraph
from frontend.pano_vggt.grid_utils import feature_uv_to_image_uv
from frontend.pano_vggt.m3_config import parse_m3_sphere_config


def _synthetic_images(num_frames: int, height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, steps=height).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, steps=width).view(1, 1, 1, width)
    frames = []
    for idx in range(int(num_frames)):
        shift = float(idx) / max(1, int(num_frames))
        rgb = torch.cat(
            [
                torch.remainder(x + shift, 1.0).expand(1, 1, height, width),
                y.expand(1, 1, height, width),
                (1.0 - 0.5 * y + 0.2 * torch.sin(6.28318 * (x + shift))).clamp(0.0, 1.0).expand(1, 1, height, width),
            ],
            dim=1,
        )
        frames.append(rgb[0])
    return torch.stack(frames, dim=0)


def _load_image_stack(paths: list[str], *, height: int | None, width: int | None) -> torch.Tensor:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        images.append(tensor)
    stack = torch.stack(images, dim=0)
    if height is not None and width is not None:
        stack = F.interpolate(stack, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return stack


def _default_fake_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [int(args.height), int(args.width)],
            "fake_translation_step": float(args.fake_translation_step),
            "M3Sphere": {"enabled": True, "descriptor_dim": int(args.descriptor_dim)},
            "MatchingHead": {
                "enabled": True,
                "descriptor_dim": int(args.descriptor_dim),
                "allow_fake_matching": True,
                "fake_feature_stride": int(args.fake_feature_stride),
            },
            "DenseMatching": {
                "enabled": True,
                "search_radius": int(args.search_radius),
                "max_factors": int(args.max_factors),
                "max_samples_per_edge": int(args.max_samples_per_edge),
            },
            "InferenceWindow": {"temporal_radius": int(args.temporal_radius), "max_edges": int(args.max_edges)},
        }
    }


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    else:
        config = _default_fake_config(args)
    if args.fake:
        config.setdefault("PanoVGGT", {})
        config["PanoVGGT"].update(_default_fake_config(args)["PanoVGGT"])
    if args.checkpoint:
        pano = config.setdefault("PanoVGGT", {})
        pano.setdefault("M3Sphere", {})["enabled"] = True
        pano.setdefault("MatchingHead", {})["enabled"] = True
        pano["MatchingHead"]["checkpoint"] = args.checkpoint
    return config


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    value = tensor.detach().cpu().float()
    if value.ndim == 3:
        value = value.mean(dim=0)
    value = value - value.min()
    value = value / value.max().clamp_min(1.0e-6)
    array = (value.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="L")


def _rgb_to_pil(image: torch.Tensor) -> Image.Image:
    value = image.detach().cpu().float().clamp(0.0, 1.0)
    array = (value.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _scatter_factor_values(factor: DenseSphereFactor, values: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor:
    grid = torch.zeros(feature_hw, dtype=torch.float32)
    counts = torch.zeros(feature_hw, dtype=torch.float32)
    uv = factor.src_uv.detach().cpu()
    vals = values.detach().cpu().float().reshape(-1)
    x = uv[:, 0].floor().long().clamp(0, feature_hw[1] - 1)
    y = uv[:, 1].floor().long().clamp(0, feature_hw[0] - 1)
    grid[y, x] += vals
    counts[y, x] += 1.0
    return grid / counts.clamp_min(1.0)


def _save_factor_masks(graph: DenseSphereFactorGraph, output_dir: Path, feature_hw: tuple[int, int]) -> None:
    if not graph.factors:
        return
    factor = graph.factors[0]
    masks = {
        "valid_factor_mask": factor.valid_mask.float(),
        "depth_consistency_mask": factor.metadata.get("depth_consistency_mask", torch.zeros_like(factor.valid_mask)).float(),
        "forward_backward_mask": factor.metadata.get("fb_pass_mask", torch.zeros_like(factor.valid_mask)).float(),
    }
    for name, values in masks.items():
        _tensor_to_pil(_scatter_factor_values(factor, values, feature_hw)).save(output_dir / f"{name}.png")


def _save_match_lines(
    images: torch.Tensor,
    graph: DenseSphereFactorGraph,
    output_dir: Path,
    *,
    image_hw: tuple[int, int],
    feature_hw: tuple[int, int],
    max_matches: int,
) -> None:
    if not graph.factors:
        return
    factor = graph.factors[0]
    src_image = _rgb_to_pil(images[factor.src])
    tgt_image = _rgb_to_pil(images[factor.tgt])
    canvas = Image.new("RGB", (src_image.width + tgt_image.width, max(src_image.height, tgt_image.height)))
    canvas.paste(src_image, (0, 0))
    canvas.paste(tgt_image, (src_image.width, 0))
    draw = ImageDraw.Draw(canvas)
    valid_idx = torch.nonzero(factor.valid_mask, as_tuple=False).flatten()[: int(max_matches)]
    if valid_idx.numel() == 0:
        valid_idx = torch.arange(min(int(max_matches), factor.src_uv.shape[0]))
    src_uv = feature_uv_to_image_uv(factor.src_uv[valid_idx].cpu(), feature_hw, image_hw)
    tgt_uv = feature_uv_to_image_uv(factor.tgt_uv[valid_idx].cpu(), feature_hw, image_hw)
    for s, t in zip(src_uv.tolist(), tgt_uv.tolist()):
        color = (64, 220, 120)
        draw.line([(float(s[0]), float(s[1])), (src_image.width + float(t[0]), float(t[1]))], fill=color, width=1)
        draw.ellipse((float(s[0]) - 2, float(s[1]) - 2, float(s[0]) + 2, float(s[1]) + 2), fill=(255, 80, 80))
        draw.ellipse((src_image.width + float(t[0]) - 2, float(t[1]) - 2, src_image.width + float(t[0]) + 2, float(t[1]) + 2), fill=(80, 180, 255))
    canvas.save(output_dir / "match_lines.png")


def _build_graph_if_needed(engine: Any, pred: Any, config: dict[str, Any]) -> DenseSphereFactorGraph | None:
    graph = getattr(engine, "last_dense_factor_graph", None)
    if graph is not None:
        return graph
    if pred.dense_descriptors is None or pred.match_confidence is None or pred.sky_prob is None:
        return None
    m3 = parse_m3_sphere_config(config)
    matcher = PoseGuidedDenseMatcher(
        search_radius=m3.dense_matching.search_radius,
        max_factors=m3.dense_matching.max_factors,
        max_samples_per_edge=m3.dense_matching.max_samples_per_edge,
        use_wraparound=m3.dense_matching.use_wraparound,
    )
    edges = DenseSphereFactorGraph.build_edges(
        int(pred.poses_c2w.shape[0]),
        temporal_radius=m3.inference_window.temporal_radius,
        max_edges=m3.inference_window.max_edges,
        device=pred.depth.device,
    )
    return matcher.match(
        poses_c2w=pred.poses_c2w,
        depth=pred.depth,
        dense_descriptors=pred.dense_descriptors,
        match_confidence=pred.match_confidence,
        sky_prob=pred.sky_prob,
        image_hw=pred.image_hw or tuple(pred.depth.shape[-2:]),
        feature_hw=pred.feature_hw,
        edge_pairs=edges,
        static_confidence=pred.static_confidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--output-dir", default="outputs/panovggt_m3_matching_debug")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--descriptor-dim", type=int, default=24)
    parser.add_argument("--fake-feature-stride", type=int, default=4)
    parser.add_argument("--fake-translation-step", type=float, default=0.08)
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--max-factors", type=int, default=8192)
    parser.add_argument("--max-samples-per-edge", type=int, default=512)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument("--max-matches", type=int, default=80)
    args = parser.parse_args()

    config = _load_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        images = _load_image_stack(args.images, height=args.height, width=args.width)
    else:
        images = _synthetic_images(args.num_frames, args.height, args.width)

    engine = build_panovggt_engine(config.get("PanoVGGT", config))
    pred = engine.infer(images)
    graph = _build_graph_if_needed(engine, pred, config)

    image_hw = pred.image_hw or tuple(pred.depth.shape[-2:])
    vis_images = F.interpolate(images.float(), size=image_hw, mode="bilinear", align_corners=False)
    if pred.sky_prob is not None:
        _tensor_to_pil(pred.sky_prob[0, 0]).save(output_dir / "sky_prob.png")
    if pred.match_confidence is not None:
        _tensor_to_pil(pred.match_confidence[0, 0]).save(output_dir / "match_confidence.png")

    metrics = dict(pred.matching_debug or {})
    if graph is not None:
        metrics.update(graph.metrics())
        if pred.feature_hw is not None:
            _save_match_lines(vis_images, graph, output_dir, image_hw=image_hw, feature_hw=pred.feature_hw, max_matches=args.max_matches)
            _save_factor_masks(graph, output_dir, pred.feature_hw)

    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
