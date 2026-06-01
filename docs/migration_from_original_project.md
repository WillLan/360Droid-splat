# Migration From 360GS-SLAM

The source project is treated as read-only.  This project root contains the new
independent implementation and migrated project rules:

- `AGENTS.md`
- `.cursor/rules/360uav-experiment-rules.mdc`
- `.cursor/rules/server-resource-limits.mdc`
- `.cursor/mcp.json`

The run used as the backend target had these effective choices:

- frontend mode: `360dvo` in the original run, replaced here by PanoDROID-MVP
- map mode: `anchor_scaffold_panorama`
- panorama renderer: `pfgs360_gsplat`
- pose convention: original backend convention, +Y down

Copied code is intentionally limited.  The old frontend, unrelated renderer
experiments, old run scripts, and legacy CUDA rasterizer are not vendored.

## Kept Boundaries

- `frontend/pano_droid`: trainable PanoDROID-MVP frontend.
- `mapping/gaussian_initializer.py`: converts frontend pose/depth/confidence
  into anchor-scaffold Gaussian seeds.
- `backend/pano_gs`: PFGS360 renderer adapter and compact anchor map.
- `system/pano_droid_gs_slam.py`: frontend/tracking/mapping orchestration.

## Deferred Items

- DROID-style fnet/cnet/correlation are implemented locally.  They follow the
  DROID module boundaries but do not vendor DROID-SLAM CUDA/lietorch code.
- Full parity with the original `PanoScaffoldModel` growth/prune heuristics is
  deferred.  This project currently keeps a smaller anchor-scaffold map with
  compatible renderer attributes.
- Production rendering requires `gsplat360` and its CUDA extension.  The local
  fallback renderer is only for smoke tests.
