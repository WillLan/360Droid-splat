# Stage 0 Cleanup Summary

Status: conservative cleanup executed after user confirmation.

## Deleted Files

Tracked files removed with `git rm`:

```text
configs/pano_anchor_splat_decoder_smoke.yaml
configs/pano_anchor_splat_refiner_smoke.yaml
configs/pano_resplat_dense_world_init_256x512.yaml
configs/pano_resplat_dense_world_refine_256x512.yaml
configs/pano_resplat_init.yaml
configs/pano_resplat_joint.yaml
configs/pano_resplat_online_ob3d_archiviz_flat_quality_gsplat360_real.yaml
configs/pano_resplat_online_ob3d_barbershop_quality_gsplat360_real.yaml
configs/pano_resplat_online_ob3d_smoke.yaml
configs/pano_resplat_overfit.yaml
configs/pano_resplat_panovggt_aligned_init_128x256_render_512x1024.yaml
configs/pano_resplat_panovggt_aligned_refine_resplat_update_128x256_render_512x1024.yaml
configs/pano_resplat_point_decoder_gaussian_init_128x256_render_512x1024.yaml
configs/pano_resplat_point_decoder_gaussian_refine_ghosting_error_decoder_128x256_render_512x1024.yaml
configs/pano_resplat_point_decoder_gaussian_refine_resplat_update_128x256_render_512x1024.yaml
configs/pano_resplat_point_decoder_gaussian_refine_voxel_anchor_128x256_render_512x1024.yaml
configs/pano_resplat_refine.yaml
frontend/pano_vggt/pano_anchor_splat_decoder.py
frontend/pano_vggt/pano_anchor_splat_encoder.py
frontend/pano_vggt/pano_anchor_splat_error.py
frontend/pano_vggt/pano_anchor_splat_frontend.py
frontend/pano_vggt/pano_anchor_splat_prior.py
frontend/pano_vggt/pano_anchor_splat_refiner.py
frontend/pano_vggt/pano_anchor_splat_types.py
frontend/pano_vggt/pano_anchor_splat_voxel.py
frontend/pano_vggt/pano_point_transformer.py
frontend/pano_vggt/pano_resplat_feedback.py
frontend/pano_vggt/pano_resplat_frontend.py
frontend/pano_vggt/pano_resplat_geometry.py
frontend/pano_vggt/pano_resplat_init.py
frontend/pano_vggt/pano_resplat_online_frontend.py
frontend/pano_vggt/pano_resplat_point_decoder_init.py
frontend/pano_vggt/pano_resplat_refiner.py
frontend/pano_vggt/pano_resplat_renderer.py
frontend/pano_vggt/pano_resplat_voxel.py
frontend/pano_vggt/resplat_types.py
frontend/pano_vggt/train_anchor_splat_gaussian.py
frontend/pano_vggt/train_resplat_gaussian.py
tests/test_pano_anchor_splat.py
tests/test_pano_resplat_dense_world_training.py
tests/test_pano_resplat_initializer.py
tests/test_pano_resplat_no_target_feedback.py
tests/test_pano_resplat_online_frontend.py
tests/test_pano_resplat_panovggt_aligned.py
tests/test_pano_resplat_resplat_update.py
tests/test_pano_resplat_shapes.py
tests/test_pano_resplat_state_renderer.py
tests/test_pano_resplat_training_step.py
tests/test_pano_resplat_voxel_anchor.py
tests/test_pano_resplat_zero_init_refiner.py
tests/test_resplat_state_fusion.py
```

## Moved To Backup

Untracked old outputs moved instead of permanently deleted:

```text
outputs/pano_resplat -> _deprecated_frontend_cleanup_backup/outputs/pano_resplat
outputs/pytest-resplat-basetemp -> _deprecated_frontend_cleanup_backup/outputs/pytest-resplat-basetemp
```

## Updated Surviving Source

```text
frontend/pano_droid/adapter.py
frontend/pano_vggt/__init__.py
```

Changes:

- Removed old `pano_anchor_splat` / `pano_resplat_online` frontend dispatch.
- Removed public exports for deleted AnchorSplat/ReSplat classes and builders.

## Preserved Content

Preserved because it is current or general-purpose:

- PanoVGGT long frontend and config-gated M3-Sphere matching/dense BA code.
- PanoDROID graph frontend and training code.
- `FrontendOutput` public API.
- General Gaussian initializer in `mapping/gaussian_initializer.py`.
- General backend map, mapper, renderer, and PFGS360 adapter.
- PanoVGGT-M3-Sphere configs that do not train the old Gaussian head.
- Dataset loaders and general training utilities.

## Needs Human Confirmation

Retained for manual review:

```text
frontend/pano_vggt/gaussian_head.py
frontend/pano_vggt/train_gaussian.py
configs/panovggt_m3_sphere_omni360_gaussian_train.yaml
tests/test_panovggt_gaussian_training.py
frontend/pano_vggt/engine.py                 # resplat_feature_hook side-channel
frontend/pano_vggt/tracker.py                # resplat_features_by_frame cache
system/pano_droid_gs_slam.py                 # ReSplatFusion direct-fusion branch
backend/pano_gs/mapper.py                    # ReSplat direct-fusion backend methods/stats
.codex_tmp/*resplat* and .codex_tmp/*gaussian*
```

## Main Pipeline Import Status

The conservative cleanup removes the deleted old modules from public package exports and frontend mode dispatch, so the normal `graph`, `pano_droid`, and `panovggt_long` frontend modes should continue to import through the existing paths.

Residual `resplat` keywords are expected in:

- Stage 0 documentation and backup paths.
- The retained human-review Gaussian-head and direct-fusion boundary files listed above.
- Untracked scratch artifacts under `.codex_tmp/`.

## Suggested Selfi-Style Entry Points

Recommended future integration points:

- `frontend/pano_vggt/engine.py`: keep PanoVGGT frozen output normalization and feature extraction here.
- `frontend/pano_vggt/matching_head.py` and `matching_adapter.py`: extend or replace with Selfi-style spherical feature adapter outputs if confirmed.
- `frontend/pano_vggt/spherical_correspondence.py`: build spherical pseudo-correspondences.
- `frontend/pano_vggt/spherical_dense_ba.py`: keep main dense BA residual as S2 tangent residual.
- `frontend/pano_vggt/tracker.py`: place config-gated per-frame/chunk refinement before existing alignment emission.
- `mapping/gaussian_initializer.py`: consume refined pose/depth/world points through existing `FrontendOutput` fields.

## Verification

Commands run after cleanup:

```bash
python -m compileall .
pytest -q
python -c "import frontend.pano_vggt; import frontend.pano_droid.adapter; import system.pano_droid_gs_slam; print('import smoke ok')"
rg -n "frontend\\.pano_vggt\\.(pano_anchor_splat|pano_resplat|resplat_types|pano_point_transformer|train_resplat_gaussian|train_anchor_splat_gaussian)|from \\.pano_anchor_splat|from \\.pano_resplat|from \\.resplat_types|from \\.pano_point_transformer|PanoAnchorSplat|PanoReSplatOnline|build_pano_anchor_splat|build_pano_resplat_online" frontend mapping backend system tests configs scripts pyproject.toml README.md
```

Results:

- `python -m compileall .`: passed. It traversed untracked scratch directories and emitted only SyntaxWarnings from `.codex_tmp/Scaffold-GS`.
- `pytest -q`: passed.
- Import smoke: passed.
- Stale deleted-module reference scan: no matches in active source/config/test paths.
