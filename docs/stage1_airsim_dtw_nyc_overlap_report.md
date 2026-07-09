# Stage 1.5 Airsim DTW/NYC Overlap Report

## Current Manifest Window Check

The generated Airsim manifest contains only DTW and NYC records. With
`views_per_sample=4`, `pair_mode=adjacent_and_skip`, and `max_temporal_gap=10`,
the contiguous train/val blocks are expected to produce valid sample windows.

Expected train windows before runtime filtering:

- DTW train: `5582 - 4 + 1 = 5579`
- NYC train: `4661 - 4 + 1 = 4658`
- total train windows: `10237`

Expected val windows before runtime filtering:

- DTW val: `621 - 4 + 1 = 618`
- NYC val: `518 - 4 + 1 = 515`
- total val windows: `1133`

## Geometry Overlap

No GT pose was found in the Airsim DTW/NYC directories. Therefore
`tools/check_stage1_overlap.py` can validate manifest windows and summarize
depth availability, but pseudo correspondence validity must be measured during
the PanoVGGT-backed smoke run.

Default smoke thresholds:

- `min_valid_corr_ratio`: `0.03`
- `num_query_per_pair`: `2048`
- `visibility_rel_thresh`: `0.05`
- `max_temporal_gap`: `10`

## Commands

```bash
python tools/check_stage1_overlap.py \
  --manifest data/stage1_airsim_dtw_nyc_manifest.json \
  --image-height 504 \
  --image-width 1008

python tools/check_stage1_overlap.py \
  --manifest data/stage1_airsim_dtw_nyc_debug_manifest.json \
  --image-height 504 \
  --image-width 1008
```

## Bad Sequences To Skip

None identified from static audit. Revisit this section after the 200-step
PanoVGGT-backed smoke training reports valid correspondence ratios.
