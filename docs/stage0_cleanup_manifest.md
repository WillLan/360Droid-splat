# Stage 0 Cleanup Manifest

Status: conservative cleanup executed after user confirmation. Rows marked `delete` were removed with `git rm`; rows marked `move_to_backup` were moved under `_deprecated_frontend_cleanup_backup/`.

Action meanings:

- `delete`: tracked source/config/test file can be removed with `git rm` after confirmation.
- `move_to_backup`: untracked output/log/checkpoint directory should be moved under `_deprecated_frontend_cleanup_backup/` after confirmation.
- `keep`: leave unchanged.
- `need_human_review`: do not delete or edit until explicitly confirmed.

## Confirmed Delete Candidates

| Path | Type | Matched keywords | Why old frontend experiment | Imported by other modules | Recommended action |
|---|---|---|---|---|---|
| `frontend/pano_vggt/pano_anchor_splat_decoder.py` | source | `anchor_splat`, `gaussian_decoder` | PanoAnchorSplat local-window Gaussian decoder | old cluster, `tests/test_pano_anchor_splat.py` | delete |
| `frontend/pano_vggt/pano_anchor_splat_encoder.py` | source | `anchor_splat` | PanoAnchorSplat feature encoder | old cluster, `tests/test_pano_anchor_splat.py` | delete |
| `frontend/pano_vggt/pano_anchor_splat_error.py` | source | `anchor_splat`, render error | PanoAnchorSplat render-error encoder | old cluster only | delete |
| `frontend/pano_vggt/pano_anchor_splat_frontend.py` | source | `anchor_splat`, frontend | AnchorSplat frontend entrypoint | `frontend/pano_droid/adapter.py`, `frontend/pano_vggt/__init__.py`, old tests | delete |
| `frontend/pano_vggt/pano_anchor_splat_prior.py` | source | `anchor_splat`, prior | Offline PanoVGGT prior adapter for AnchorSplat | old cluster, `tests/test_pano_anchor_splat.py` | delete |
| `frontend/pano_vggt/pano_anchor_splat_refiner.py` | source | `anchor_splat`, refiner | AnchorSplat Gaussian refiner-lite | old cluster only | delete |
| `frontend/pano_vggt/pano_anchor_splat_types.py` | source | `anchor_splat`, config | AnchorSplat config and state containers | `frontend/pano_vggt/__init__.py`, old cluster, old tests | delete |
| `frontend/pano_vggt/pano_anchor_splat_voxel.py` | source | `anchor_splat`, voxel | Anchor/voxel construction for old frontend | old cluster, `tests/test_pano_anchor_splat.py` | delete |
| `frontend/pano_vggt/pano_point_transformer.py` | source | `point_transformer`, `knn_attention`, ReSplat | Point-neighborhood transformer for Pano-ReSplat states | old cluster, `tests/test_pano_resplat_resplat_update.py` | delete |
| `frontend/pano_vggt/pano_resplat_feedback.py` | source | `resplat`, feedback, render error | ReSplat render feedback and error decoder | old cluster, old tests | delete |
| `frontend/pano_vggt/pano_resplat_frontend.py` | source | `resplat`, frontend | Pano-ReSplat frontend orchestration | old cluster, old tests | delete |
| `frontend/pano_vggt/pano_resplat_geometry.py` | source | `resplat`, ERP geometry | Geometry helpers scoped to old ReSplat state/render pipeline | old cluster, `tests/test_pano_resplat_state_renderer.py` | delete |
| `frontend/pano_vggt/pano_resplat_init.py` | source | `resplat`, Gaussian initialization | Legacy compact feed-forward Gaussian initialization | old cluster only | delete |
| `frontend/pano_vggt/pano_resplat_online_frontend.py` | source | `resplat_online`, online frontend | Online wrapper emitting ReSplat artifacts | `frontend/pano_droid/adapter.py`, `frontend/pano_vggt/__init__.py`, old tests | delete |
| `frontend/pano_vggt/pano_resplat_point_decoder_init.py` | source | `resplat`, point decoder | Old PanoVGGT point-decoder Gaussian initializer | `frontend/pano_vggt/__init__.py`, old tests | delete |
| `frontend/pano_vggt/pano_resplat_refiner.py` | source | `resplat`, recurrent updater | Recurrent Gaussian update block | old cluster, old tests | delete |
| `frontend/pano_vggt/pano_resplat_renderer.py` | source | `resplat`, renderer adapter | Renderer adapter for old `PanoGaussianState` frontend outputs | old cluster, old tests | delete |
| `frontend/pano_vggt/pano_resplat_voxel.py` | source | `resplat`, voxel compactor | Voxel compaction for dense ReSplat Gaussian states | old cluster, old tests | delete |
| `frontend/pano_vggt/resplat_types.py` | source | `resplat`, Gaussian state | Core old ReSplat Gaussian state containers | old cluster, old tests | delete |
| `frontend/pano_vggt/train_anchor_splat_gaussian.py` | source | `anchor_splat`, training | AnchorSplat frontend trainer | `tests/test_pano_anchor_splat.py` | delete |
| `frontend/pano_vggt/train_resplat_gaussian.py` | source | `resplat`, training | ReSplat Gaussian initializer/refiner trainer | old cluster, old tests | delete |
| `configs/pano_anchor_splat_decoder_smoke.yaml` | config | `anchor_splat` | AnchorSplat decoder smoke config | not imported | delete |
| `configs/pano_anchor_splat_refiner_smoke.yaml` | config | `anchor_splat` | AnchorSplat refiner smoke config | not imported | delete |
| `configs/pano_resplat_dense_world_init_256x512.yaml` | config | `resplat` | ReSplat dense-world init training config | not imported | delete |
| `configs/pano_resplat_dense_world_refine_256x512.yaml` | config | `resplat` | ReSplat dense-world refine training config | not imported | delete |
| `configs/pano_resplat_init.yaml` | config | `resplat` | ReSplat init training config | not imported | delete |
| `configs/pano_resplat_joint.yaml` | config | `resplat` | ReSplat joint training config | not imported | delete |
| `configs/pano_resplat_online_ob3d_archiviz_flat_quality_gsplat360_real.yaml` | config | `resplat_online`, `ReSplatFusion` | Old online ReSplat experiment launch config | not imported | delete |
| `configs/pano_resplat_online_ob3d_barbershop_quality_gsplat360_real.yaml` | config | `resplat_online`, `ReSplatFusion` | Old online ReSplat experiment launch config | not imported | delete |
| `configs/pano_resplat_online_ob3d_smoke.yaml` | config | `resplat_online`, `ReSplatFusion` | Old online ReSplat smoke config | not imported | delete |
| `configs/pano_resplat_overfit.yaml` | config | `resplat` | ReSplat overfit config | not imported | delete |
| `configs/pano_resplat_panovggt_aligned_init_128x256_render_512x1024.yaml` | config | `resplat`, PanoVGGT aligned init | Old ReSplat aligned init training config | not imported | delete |
| `configs/pano_resplat_panovggt_aligned_refine_resplat_update_128x256_render_512x1024.yaml` | config | `resplat`, refine update | Old ReSplat aligned refine config | not imported | delete |
| `configs/pano_resplat_point_decoder_gaussian_init_128x256_render_512x1024.yaml` | config | `resplat`, point decoder | Old point-decoder Gaussian init config | not imported | delete |
| `configs/pano_resplat_point_decoder_gaussian_refine_ghosting_error_decoder_128x256_render_512x1024.yaml` | config | `resplat`, error decoder | Old ReSplat ghosting/error-decoder refine config | not imported | delete |
| `configs/pano_resplat_point_decoder_gaussian_refine_resplat_update_128x256_render_512x1024.yaml` | config | `resplat`, recurrent update | Old ReSplat recurrent update config | not imported | delete |
| `configs/pano_resplat_point_decoder_gaussian_refine_voxel_anchor_128x256_render_512x1024.yaml` | config | `resplat`, voxel anchor | Old ReSplat voxel-anchor refine config | not imported | delete |
| `configs/pano_resplat_refine.yaml` | config | `resplat` | ReSplat refine training config | not imported | delete |
| `tests/test_pano_anchor_splat.py` | source | `anchor_splat` | Tests old AnchorSplat modules/training | pytest only | delete |
| `tests/test_pano_resplat_dense_world_training.py` | source | `resplat` | Tests old ReSplat training utilities | pytest only | delete |
| `tests/test_pano_resplat_initializer.py` | source | `resplat` | Tests old ReSplat initializer | pytest only | delete |
| `tests/test_pano_resplat_no_target_feedback.py` | source | `resplat`, feedback | Tests old feedback behavior | pytest only | delete |
| `tests/test_pano_resplat_online_frontend.py` | source | `resplat_online`, direct fusion | Tests old online ReSplat artifact path | pytest only | delete |
| `tests/test_pano_resplat_panovggt_aligned.py` | source | `resplat`, PanoVGGT aligned | Tests old aligned ReSplat frontend | pytest only | delete |
| `tests/test_pano_resplat_resplat_update.py` | source | `resplat`, recurrent update | Tests old recurrent update/refiner path | pytest only | delete |
| `tests/test_pano_resplat_shapes.py` | source | `resplat` | Shape smoke tests for old frontend | pytest only | delete |
| `tests/test_pano_resplat_state_renderer.py` | source | `resplat`, renderer | Tests old ReSplat renderer adapter/state | pytest only | delete |
| `tests/test_pano_resplat_training_step.py` | source | `resplat`, training | Tests old ReSplat training step | pytest only | delete |
| `tests/test_pano_resplat_voxel_anchor.py` | source | `resplat`, voxel | Tests old voxel/anchor compaction | pytest only | delete |
| `tests/test_pano_resplat_zero_init_refiner.py` | source | `resplat`, refiner | Tests old zero-init ReSplat refiner | pytest only | delete |
| `tests/test_resplat_state_fusion.py` | source | `resplat`, state fusion | Tests old backend direct fusion path | pytest only | delete |

## Move To Backup Candidates

| Path | Type | Matched keywords | Why old frontend experiment | Imported by other modules | Recommended action |
|---|---|---|---|---|---|
| `outputs/pano_resplat/` | output | `pano_resplat` | Untracked local ReSplat training/output tree, 2305 files, about 74 MB | not imported | move_to_backup |
| `outputs/pytest-resplat-basetemp/` | output | `resplat` | Untracked empty pytest temp directory for old ReSplat tests | not imported | move_to_backup |

## Need Human Review

| Path | Type | Matched keywords | Why uncertain | Imported by other modules | Recommended action |
|---|---|---|---|---|---|
| `frontend/pano_vggt/gaussian_head.py` | source | `AnchorGaussian`, `GaussianHead`, `IterativeGaussianRefiner` | Not named ReSplat/AnchorSplat, but implements anchor/scaffold Gaussian prediction and old refiner behavior | `frontend/pano_vggt/__init__.py`, `frontend/pano_vggt/train_gaussian.py`, tests | need_human_review |
| `frontend/pano_vggt/train_gaussian.py` | source | `gaussian_head`, `anchor_gaussian_head`, `gaussian_refiner` | Trains anchor/scaffold Gaussian head; may be old initial/refiner head | `tests/test_panovggt_gaussian_training.py`, old trainers use helper functions | need_human_review |
| `configs/panovggt_m3_sphere_omni360_gaussian_train.yaml` | config | `GaussianHead`, `panovggt-gaussian-head` | M3-named config but trains anchor/scaffold Gaussian head, not matching/dense BA | not imported | need_human_review |
| `tests/test_panovggt_gaussian_training.py` | source | `AnchorGaussian`, `gaussian_head` | Tests the uncertain Gaussian-head cluster | pytest only | need_human_review |
| `frontend/pano_vggt/__init__.py` | source | `PanoAnchorSplat`, `PanoReSplat`, old exports | Must be edited if old modules are deleted; file itself should stay | public package import surface | need_human_review |
| `frontend/pano_droid/adapter.py` | source | `PanoAnchorSplat`, `PanoReSplatOnline`, old modes | Must be edited if old frontend modes are removed; affects launch behavior | system frontend factory | need_human_review |
| `frontend/pano_vggt/engine.py` | source | `resplat_feature_hook`, `last_resplat_dense_features` | Old ReSplat feature side-channel inside core PanoVGGT engine | tracker, config | need_human_review |
| `frontend/pano_vggt/tracker.py` | source | `resplat_features_by_frame` | Old ReSplat feature cache inside core PanoVGGT tracker | system frontend | need_human_review |
| `system/pano_droid_gs_slam.py` | source | `ReSplatFusion`, `resplat_direct_fusion_enabled` | Old direct-fusion runtime branch inside main SLAM system | main system entrypoint | need_human_review |
| `backend/pano_gs/mapper.py` | source | `add_or_fuse_resplat_gaussians`, `fuse_resplat_state`, `ReSplatGlobal` | Old direct-fusion methods in core backend map/mapper | system direct fusion path, old test | need_human_review |
| `.codex_tmp/*resplat*`, `.codex_tmp/*gaussian*` | output | `resplat`, `gaussian` | Scratch artifacts outside requested project output directories; may be useful for comparison | not imported | need_human_review |

## Keep

| Path | Type | Matched keywords | Reason | Imported by other modules | Recommended action |
|---|---|---|---|---|---|
| `frontend/pano_vggt/matching_head.py` | source | M3 matching | Current PanoVGGT-M3-Sphere matching head | current M3 training/inference | keep |
| `frontend/pano_vggt/matching_adapter.py` | source | M3 matching | Current matching/sky checkpoint adapter | current M3 engine/training | keep |
| `frontend/pano_vggt/matching_losses.py` | source | M3 matching | Current matching/sky losses | current M3 training | keep |
| `frontend/pano_vggt/matching_dataset.py` | source | M3 training data | Shared current training dataset code | current M3/PanoVGGT training | keep |
| `frontend/pano_vggt/dense_matcher.py` | source | dense matching | Current spherical dense matching | current M3 path | keep |
| `frontend/pano_vggt/factor_graph.py` | source | dense factors | Current dense factor graph | current M3 path | keep |
| `frontend/pano_vggt/dense_ba_refiner.py` | source | dense BA | Current M3 dense BA wrapper | current PanoVGGT long tracker | keep |
| `frontend/pano_vggt/spherical_correspondence.py` | source | spherical residual | Current spherical correspondence logic | current M3 training/BA | keep |
| `frontend/pano_vggt/spherical_dense_ba.py` | source | spherical BA | Current S2 tangent dense BA | current M3 tests/tracker | keep |
| `frontend/pano_vggt/m3_config.py` | source | M3 config | Current config-gated M3 path | current engine/tracker/tests | keep |
| `mapping/gaussian_initializer.py` | source | Gaussian initializer | General map seed initialization from `FrontendOutput` | current SLAM pipeline | keep |
| `backend/pano_gs/adapter.py` | source | renderer adapter | General PFGS360/gsplat360 renderer adapter | current backend | keep |
| `backend/pano_gs/mapper.py` | source | backend map | General backend map/mapper must stay; only old ReSplat methods need review | current SLAM pipeline | keep |
