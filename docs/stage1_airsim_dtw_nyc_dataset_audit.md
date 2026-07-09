# Stage 1.5 Airsim DTW/NYC Dataset Audit

- dataset root: `/mnt/disk1/lanboyang/Datasets/Airsim360/Omni360-Scene`
- used scenes: `DTW`, `NYC`
- domain: `outdoor`
- source data policy: original Airsim files are not copied, moved, or renamed
- manifest files:
  - `data/stage1_airsim_dtw_nyc_manifest.json`
  - `data/stage1_airsim_dtw_nyc_debug_manifest.json`

## Scene Summary

| scene | RGB frames | depth frames | train | val | RGB path | depth path |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| DTW | 6203 | 6203 | 5582 | 621 | `DTW/dtw_Raw/panorama_N.png` | `DTW/dtw_Depth/Depth_N.h5` |
| NYC | 5179 | 5179 | 4661 | 518 | `NYC/nyc_Raw/panorama_N.png` | `NYC/nyc_Depth/Depth_N.h5` |

## Format

- RGB images are `2048x1024` ERP PNG files and are resized to `504x1008` by the Stage 1 dataset loader.
- Depth files are HDF5 files with key `depth`, shape `1024x2048`, dtype `float32`.
- Sampled depth values showed finite depth maps with far-plane values at `1000.0`.
- No GT pose or trajectory file was found under the DTW/NYC scene directories during audit.

## Split

The manifest uses contiguous per-scene splits to avoid validation leakage:

- first 90% of each scene: `train`
- last 10% of each scene: `val`

Because each scene has a single continuous frame stream and no sequence metadata was found, `sequence_id` is set to `DTW_train`, `DTW_val`, `NYC_train`, or `NYC_val`.

## Known Issues

- `pose_path` is `null` for all records. Stage 1.5 training must use frozen PanoVGGT init pose/depth for pseudo correspondence.
- Airsim depth is available for sanity checks, but it is not used as training geometry fallback in the default Airsim config.
- Values at or near `1000.0` should be treated as far-plane/invalid depth in reports.
