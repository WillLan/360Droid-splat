# Stage 0 Project Audit

Status: conservative cleanup executed after user confirmation. Confirmed ReSplat/AnchorSplat source, config, and test files were removed; old untracked ReSplat outputs were moved to backup.

## Task Interpretation

The requested Stage 0 work is a safe cleanup before a future Selfi-style panoramic per-pixel Gaussian frontend. The cleanup target is old experimental frontend Gaussian prediction code:

- ReSplat-style recurrent/feed-forward Gaussian frontend code.
- AnchorSplat-style anchor/voxel Gaussian frontend code.
- Old frontend-only configs, tests, training entrypoints, and experiment outputs.

The cleanup must not implement new adapter, Gaussian head, BA, loop closure, or backend fusion work. It must preserve the current disabled/default PanoVGGT and PanoDROID behavior.

## Cleanup Mode

The user selected option 1, conservative cleanup.

Recommended option:

- Conservative cleanup: delete the clearly isolated old ReSplat/AnchorSplat source, configs, and tests; remove their public exports and frontend mode dispatch; move old untracked outputs to backup; leave PanoVGGT core feature hooks and backend direct-fusion hooks for a second confirmed pass. Tradeoff: safest first pass, but some `resplat` keywords remain documented in core files.

Other reasonable options:

- Full cleanup: also remove `resplat_feature_hook` from PanoVGGT engine/tracker and `ReSplatFusion` direct-fusion hooks from system/backend. Tradeoff: cleaner grep result, higher risk because it changes core launch behavior and backend stats.
- Audit-only: keep code unchanged and use this report as the planning artifact. Tradeoff: zero code risk, but old frontend code remains importable.

## Directory Tree Summary

Tracked project structure:

```text
backend/
  legacy_360gs/              read-only migrated/reference backend and renderer code
  pano_gs/                   current Gaussian map, mapper, losses, renderer adapter
configs/                     SLAM, training, PanoVGGT, ReSplat, AnchorSplat configs
docs/                        project design and migration documents
frontend/
  pano_droid/                PanoDROID frontend, training, graph tracker, interfaces
  pano_vggt/                 PanoVGGT long frontend, M3 matching/BA, old Gaussian frontend experiments
mapping/                     Gaussian seed initialization from FrontendOutput
scripts/                     run/debug shell and Python entrypoints
system/                      online SLAM orchestration entrypoints
tests/                       unit/smoke tests
```

Untracked project outputs currently present:

```text
.codex_tmp/                  local scratch artifacts, includes old ReSplat scratch files
outputs/                     local outputs, includes outputs/pano_resplat
pic/                         local visualization snapshots
panovggt_m3_ffw_forceaccept_block1_300_online_20260611_022957/
```

## Main Pipeline Flow

1. `system/pano_droid_gs_slam.py` loads config with `load_config()` and starts from `main()`.
2. `PanoDroidGSSlamSystem.__init__()` builds the frontend through `frontend.pano_droid.adapter.build_frontend_from_config()`.
3. For `Frontend.mode=panovggt_long`, the adapter builds `frontend.pano_vggt.tracker.PanoVGGTLongTracker`.
4. `PanoVGGTLongTracker` batches frames into chunks and calls `PanoVGGTInferenceEngine.infer()`.
5. The engine normalizes PanoVGGT outputs into `PanoVGGTLocalPrediction`.
6. The tracker emits standard `frontend.pano_droid.interfaces.FrontendOutput`.
7. `mapping.gaussian_initializer.GaussianInitializer` creates map seed batches from `FrontendOutput.inverse_depth` or `FrontendOutput.world_points`.
8. `backend.pano_gs.mapper.PanoGaussianMapper` registers keyframes/observations, inserts/fuses Gaussian anchors, optimizes, and calls the renderer.
9. `backend.pano_gs.adapter.PFGS360Renderer` adapts the map to the PFGS360/gsplat360 renderer or fallback renderer.

## PanoVGGT Output Production

Primary code path:

- `frontend/pano_vggt/engine.py`: `normalize_panovggt_output()` converts external/fake PanoVGGT outputs into `PanoVGGTLocalPrediction`.
- `frontend/pano_vggt/types.py`: `PanoVGGTLocalPrediction` carries `poses_c2w`, `depth`, `confidence`, `chunk_world_points`, optional `local_points/global_points`, dense descriptors, match confidence, sky outputs, and M3 debug metrics.
- `frontend/pano_vggt/tracker.py`: `PanoVGGTLongTracker._process_chunk()` aligns chunk predictions, resizes depth/confidence/world points to image resolution, stores per-frame caches, and emits `FrontendOutput`.

Important contract:

- `FrontendOutput` remains the public frontend/backend boundary. It already contains `pose_c2w`, `inverse_depth`, `depth_confidence`, `world_points`, `world_points_confidence`, and `valid_world_points_mask`.

## Gaussian Initialization

Current map seeding happens in `mapping/gaussian_initializer.py`.

- `GaussianInitializer.from_frontend_output()` backprojects inverse depth and pose.
- `GaussianInitializer.from_world_points_only()` consumes already-global point maps from PanoVGGT.
- PFGS360-style dense insertion and replace/fuse policies are configured under `Mapping.NovelGaussianInsertion`.

This file is not an old frontend experiment and should be kept.

## Renderer Calls

Renderer integration is in:

- `backend/pano_gs/adapter.py`: `PFGS360Renderer`, `PanoRenderCamera`.
- `backend/pano_gs/mapper.py`: `render_view()`, `render_keyframe_diagnostic()`, optimization render calls.
- `system/pano_droid_gs_slam.py`: runtime visualization and final render export.

Old frontend-specific renderer adapter:

- `frontend/pano_vggt/pano_resplat_renderer.py` is scoped to `PanoGaussianState` from ReSplat/AnchorSplat and is a delete candidate with the old frontend cluster.

## Backend Map Update

Current backend map update is in `backend/pano_gs/mapper.py`.

- General map insertion and optimization are used by current SLAM and should be kept.
- ReSplat-specific direct fusion methods exist:
  - `PanoGaussianMap.add_or_fuse_resplat_gaussians()`
  - `PanoGaussianMapper.fuse_resplat_state()`
  - `PanoGaussianMapper.optimize_resplat_global_window()`
- `system/pano_droid_gs_slam.py` has a `ReSplatFusion` side path that consumes `PanoReSplatOnline` artifacts.

These backend/system hooks are old-frontend-specific, but they sit in core files. They are marked `need_human_review` before removal.

## Training Entrypoints

Current/kept training entrypoints:

- `frontend/pano_droid/train.py`
- `frontend/pano_droid/train_graph.py`
- `frontend/pano_vggt/train_matching.py`
- `frontend/pano_vggt/train_panovggt_geometry.py`

Old frontend training entrypoints:

- `frontend/pano_vggt/train_resplat_gaussian.py`
- `frontend/pano_vggt/train_anchor_splat_gaussian.py`

Human-review training entrypoint:

- `frontend/pano_vggt/train_gaussian.py` trains anchor/scaffold Gaussian prediction heads. It is not named ReSplat/AnchorSplat, but matches the requested "old initial Gaussian head / old refiner head" category.

## Config Classification

Keep:

- PanoDROID configs: `configs/pano_droid_*`
- PanoVGGT long configs: `configs/pano_vggt_long_*`, `configs/pano_vggt_legacy_online_*`
- PanoVGGT-M3-Sphere configs: `configs/pano_vggt_m3_sphere_*`, `configs/panovggt_m3_sphere_omni360_train.yaml`
- PanoVGGT geometry fine-tune config: `configs/panovggt_geometry_finetune_omni360_256x512.yaml`

Delete candidates:

- `configs/pano_resplat_*`
- `configs/pano_anchor_splat_*`

Need human review:

- `configs/panovggt_m3_sphere_omni360_gaussian_train.yaml` because it trains anchor/scaffold Gaussian heads rather than M3 matching/dense BA.

## Suspected Old Files And Dependencies

Confirmed old frontend source cluster:

- `frontend/pano_vggt/pano_anchor_splat_*.py`
- `frontend/pano_vggt/pano_resplat_*.py`
- `frontend/pano_vggt/resplat_types.py`
- `frontend/pano_vggt/pano_point_transformer.py`
- `frontend/pano_vggt/train_anchor_splat_gaussian.py`
- `frontend/pano_vggt/train_resplat_gaussian.py`

External importers outside the old cluster:

- `frontend/pano_vggt/__init__.py` exports old AnchorSplat/ReSplat public symbols.
- `frontend/pano_droid/adapter.py` still dispatches `Frontend.mode=pano_anchor_splat` and `Frontend.mode=pano_resplat_online`.
- Old tests import these modules directly.
- `system/pano_droid_gs_slam.py` consumes old online ReSplat artifacts through duck-typed methods when `ReSplatFusion.enabled=true`.
- `backend/pano_gs/mapper.py` has old ReSplat direct-fusion methods but avoids frontend imports at import time.

Review finding:

Symptom: Old frontend experiments are wired through public package exports, frontend mode dispatch, system runtime branches, backend mapper methods, configs, and tests.
Source: Clean Architecture - Dependency Inversion / Acyclic Dependencies; Ousterhout - Information Leakage.
Consequence: Removing or replacing one frontend concept requires coordinated edits across frontend, system, backend, tests, and configs, increasing regression risk.
Remedy: First delete the isolated old module/config/test cluster, then separately confirm removal of core dispatch and backend hooks.
