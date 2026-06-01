# PanoDROID-GS-SLAM

This repository is the independent PanoDROID-MVP project derived from the
read-only `360GS-SLAM` codebase.  It keeps the original project rules and uses
the same panorama backend convention: +X right, +Y down, +Z forward.

The default backend path is:

- `MapRepresentation.mode: anchor_scaffold_panorama`
- `Training.panorama_render_mode: pfgs360_gsplat`
- optional production renderer: `gsplat360.rasterization`

The legacy `diff-gaussian-rasterization` CUDA path is not vendored here.  If a
future experiment proves it is still required, add it explicitly with its
license/dependency boundary documented.

## Quick Start

```bash
python -m frontend.pano_droid.train --config configs/pano_droid_train.yaml
python -m system.pano_droid_gs_slam --config configs/pano_droid_gs_slam.yaml --max-frames 4
pytest
```

The default configs use synthetic tiny data so the training and integration
loops are runnable before real 360UAV/ERP data is attached.

## Real Data

Set `Dataset.synthetic: false` and provide `Dataset.dataset_path`.  The loader
searches common ERP folders such as `images`, `rgb`, `Sequences/<sequence>`,
and the dataset root.  Optional `poses.txt`, `gt.txt`, or `GroundTruth/*.txt`
pose files are read when available.

## Renderer Note

`backend/pano_gs/adapter.py` tries to import `gsplat360` first.  When the CUDA
extension is unavailable, tests and smoke runs can use the explicit
`Renderer.allow_smoke_fallback: true` point renderer.  That fallback is not a
production-quality 3DGS renderer.

