# Stage 1 Dataset Plan

## Purpose

Stage 1 needs ERP image windows, optional initial ray-depth maps, and optional
camera-to-world poses for spherical pseudo-correspondence generation. The data
loader is manifest based so local smoke tests, curated real data, and future
remote training can share the same schema.

## Manifest Schema

The manifest is a JSON object:

```json
{
  "records": [
    {
      "scene_id": "scene_0001",
      "sequence_id": "scene_0001",
      "frame_id": 0,
      "split": "train",
      "domain": "indoor",
      "rgb_path": "images/000000.png",
      "depth_path": "depths/000000.npy",
      "pose_path": "poses/000000.npy",
      "timestamp": 0.0
    }
  ]
}
```

Required fields:

- `scene_id`
- `sequence_id`
- `frame_id`
- `rgb_path`
- `split`
- `domain`, one of `indoor` or `outdoor`

Optional fields:

- `depth_path`
- `pose_path`
- `timestamp`

Paths are resolved relative to the manifest file unless already absolute.

The loader also accepts a top-level list of records for compatibility with the
manifest builder:

```json
[
  {
    "scene_id": "scene_0001",
    "sequence_id": "seq_000",
    "frame_id": 0,
    "rgb_path": "images/000000.png",
    "depth_path": null,
    "pose_path": null,
    "timestamp": 0.0,
    "split": "train",
    "domain": "indoor"
  }
]
```

## Sampling

`Stage1PanoSequenceDataset` builds fixed-size sequence windows:

- default `views_per_sample=4`
- windows do not cross sequence boundaries
- optional `max_temporal_gap` rejects windows with large timestamp gaps
- pair indices include adjacent and skip pairs

The training script uses pseudo correspondences only when both depth and pose are
available. Missing geometry is allowed for manifest validation, but those samples
cannot produce Stage 1 alignment supervision.

## Suggested Real Data Mix

Recommended first real training set:

- 8k-20k ERP frames total.
- Balanced indoor/outdoor if both are available.
- Keep train/val/test sequence-disjoint.
- Prefer windows with useful overlap and non-degenerate motion.
- Include a small validation subset with stable depth/pose quality for fast
  smoke checks.

## Overlap Check

Use:

```bash
python tools/check_stage1_overlap.py --manifest data/stage1_dataset_manifest.json
```

When depth and pose are present, the tool reports:

- valid correspondence ratio
- mean spherical pseudo reprojection consistency in degrees

The reprojection consistency is computed by converting target ERP pixels back to
unit rays and measuring great-circle angular distance to the pseudo target rays.

## Feature Shape Check

Use this after configuring a real PanoVGGT model:

```bash
python tools/dump_panovggt_feature_shapes.py --config configs/stage1_spherical_selfi_adapter.yaml
```

The output should confirm:

- exactly four hook names
- four feature tensors with shape `B x V x C x H x W`
- adapter output shape `B x V x 24 x 504 x 1008`
- adapter channel norm near `1.0`

## Visualization

Use:

```bash
python tools/visualize_spherical_adapter_matches.py \
  --manifest data/stage1_dataset_manifest.json \
  --output outputs/stage1_match_preview.png \
  --metrics-json outputs/stage1_match_metrics.json
```

The metrics JSON reports angular error in degrees and PCK at 1, 3, and 5
degrees. If predicted target coordinates are not provided, the tool visualizes
pseudo correspondences and reports a zero-error pseudo-vs-pseudo baseline.

## Current Placeholder

`data/stage1_dataset_manifest.json` is an empty schema-valid placeholder:

```json
{"records": []}
```

It is intentionally not populated with local machine-specific dataset paths.
