# PanoDROID-GS-SLAM Pipeline

Run:

```bash
python -m system.pano_droid_gs_slam --config configs/pano_droid_gs_slam.yaml
```

Pipeline:

1. Load ERP frames from the configured dataset.
2. Track frames with the configured frontend. The default graph mode uses
   `PanoDroidGraphTracker`; `Frontend.mode: panovggt_long` switches to the
   chunked PanoVGGT frontend described in `docs/pano_vggt_long_frontend.md`.
3. Select keyframes using frontend score and forced interval.
4. Convert keyframe inverse depth and confidence into Gaussian anchor seeds.
5. Insert seeds into `PanoGaussianMap`.
6. Render/refine through `PFGS360Renderer` when enabled and available.

The backend target is `anchor_scaffold_panorama + pfgs360_gsplat`.  The frontend
does not use the old pairwise tracker for SLAM; `PanoDroidGraphTracker` delegates
to `PanoFactorGraph`, which keeps active/inactive factors, recurrent edge hidden
state, refined pose/depth state, factor ages, and keyframe removal bookkeeping.
Runtime factor creation is GT-free: temporal factors are always added, while
proximity factors are ranked by current spherical projection distance when
refined depth is available and by pose distance only as the cold-start fallback.

Each graph update runs the DROID-style update module followed by
`SphericalDenseBA`, so `FrontendOutput.ba_residual` is the real graph BA residual
from the refined state. The PanoVGGT frontend keeps the same `FrontendOutput`
contract but may emit delayed stabilized outputs after chunk alignment.
Production rendering expects the PFGS360 `gsplat360` package and CUDA extension.
The fallback point renderer exists only for tests and early integration smoke
runs.
