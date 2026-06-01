# PanoDROID-GS-SLAM Pipeline

Run:

```bash
python -m system.pano_droid_gs_slam --config configs/pano_droid_gs_slam.yaml
```

Pipeline:

1. Load ERP frames from the configured dataset.
2. Track frames with the graph-backed `PanoDroidGraphTracker`
   (`PanoDROIDFrontendAdapter` remains a compatibility name).
3. Select keyframes using frontend score and forced interval.
4. Convert keyframe inverse depth and confidence into Gaussian anchor seeds.
5. Insert seeds into `PanoGaussianMap`.
6. Render/refine through `PFGS360Renderer` when enabled and available.

The backend target is `anchor_scaffold_panorama + pfgs360_gsplat`.  The frontend
does not use the old pairwise tracker for SLAM; it maintains a DROID-style
sliding graph and reports real graph BA residuals in `FrontendOutput.ba_residual`.
Production rendering expects the PFGS360 `gsplat360` package and CUDA extension.
The fallback point renderer exists only for tests and early integration smoke
runs.
