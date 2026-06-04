# PanoVGGT-M3-Sphere Implementation Plan

This document is the high-level plan for adding a config-gated
PanoVGGT-M3-Sphere extension to the existing PanoVGGT long-sequence frontend.
It is intentionally a plan only: no matching head, dataset, matcher, BA, engine
flow, or tracker flow is implemented in this stage.

## 1. Current PanoVGGT Frontend Call Graph

Current `Frontend.mode: panovggt_long` path:

```text
system.pano_droid_gs_slam.PanoDroidGSSlamSystem.__init__
  -> frontend.pano_droid.adapter.build_frontend_from_config
    -> frontend.pano_vggt.build_panovggt_frontend_from_config
      -> PanoVGGTLongTracker(...)
        -> build_panovggt_engine(...)
          -> FakePanoVGGTInferenceEngine or ExternalPanoVGGTInferenceEngine

PanoDroidGSSlamSystem.run
  -> iter_sequence_frames(...)
  -> frontend.track(PanoFrame)
    -> PanoVGGTLongTracker.track
      -> ensure_chw_image
      -> append frame record
      -> _run_ready_chunks(final=False)
        -> _process_chunk(start, end)
          -> stack chunk images
          -> engine.infer(images)
          -> _align_chunk(pred, frame_ids)
          -> _backend_feedback_correction(...)
          -> _chunk_descriptor(...)
          -> loop bookkeeping
          -> per-frame pose/depth/conf/world_points resize
          -> _make_output(...)
            -> FrontendOutput
  -> frontend.pop_ready_outputs()
  -> GaussianInitializer.from_frontend_output(...)
  -> PanoGaussianMapper.insert_keyframe(...)
```

Flush path:

```text
PanoDroidGSSlamSystem.run end
  -> frontend.flush()
    -> _run_ready_chunks(final=True)
    -> pop_ready_outputs()
```

Legacy online path:

```text
PanoDroidGSSlamSystem.__init__
  -> Runtime.mode == legacy_online
  -> PanoVGGTLegacyOnlineSlamSystem
    -> build_frontend_from_config
    -> LegacyViewpointAdapter.build(PanoFrame, FrontendOutput)
    -> LegacyOnlineBackendClient
```

## 2. Current Prediction, FrontendOutput, and Gaussian Initializer Contracts

`PanoVGGTLocalPrediction` currently carries:

- `poses_c2w`: local chunk camera-to-world poses, shape `N x 4 x 4`.
- `depth`: local metric depth, shape `N x 1 x H x W`.
- `confidence`: depth/geometry confidence, shape `N x 1 x H x W`.
- `chunk_world_points`: local chunk point maps, shape `N x H x W x 3`.
- Optional `local_points`, `global_points`, and `descriptors`.

`PanoVGGTLongTracker` converts prediction fields into `FrontendOutput`:

- `pose_c2w`: aligned or corrected camera pose.
- `inverse_depth`: resized inverse depth at source frame image resolution.
- `depth_confidence`: resized confidence.
- `world_points`: aligned world point map at source frame image resolution.
- `world_points_confidence`: same confidence map.
- `valid_world_points_mask`: finite and positive-confidence world point mask.
- `ba_residual`: currently the chunk alignment residual.

`GaussianInitializer` consumes only `FrontendOutput` plus source image:

- In default graph mode it can back-project `inverse_depth` with `pose_c2w`.
- In PanoVGGT mode configs use `Mapping.seed_source: world_points_only`, so it
  requires `FrontendOutput.world_points`.
- Therefore PanoVGGT-M3-Sphere must write refined geometry back into existing
  `pose_c2w`, `inverse_depth`, `world_points`, confidence, and mask fields.
- The public `FrontendOutput` dataclass must not be expanded or broken.

## 3. Matching Head Insertion Position

The matching head belongs after external PanoVGGT feature extraction and before
the tracker consumes the local prediction:

```text
External/Fake PanoVGGT geometry inference
  -> expose or hook PanoVGGT feature tensor
  -> PanoVGGT-M3-Sphere matching head
    -> dense_descriptor
    -> match_confidence
    -> static_confidence
  -> PanoVGGTLocalPrediction plus internal M3-Sphere sidecar
```

The matching head must not change the old `panovggt_long` path when disabled.
Real inference must fail clearly if the feature hook or matching head is missing
and no explicit fallback is enabled.

## 4. Dense Matcher and Factor Graph Insertion Position

Dense matching should run inside the PanoVGGT-M3-Sphere gated branch after the
chunk prediction is available and before `_align_chunk()`:

```text
pred = engine.infer(images)
if PanoVGGT.m3_sphere.enabled:
  match_outputs = matching_head(features)
  correspondences = pose_guided_dense_match(pred, match_outputs)
  factors = dense_factor_graph.build(correspondences)
  refined = spherical_dense_ba(pred, factors)
  pred = refined.to_local_prediction()
transform = _align_chunk(pred, frame_ids)
```

The factor graph is local to the chunk or local window in early stages. It must
produce factors whose residual target is a target bearing, not only a target ERP
pixel.

## 5. Spherical Dense BA Insertion Position

Spherical dense BA runs after dense correspondences are built and before chunk
alignment:

```text
PanoVGGT initial pose/depth/world_points
  -> pose-guided dense matching
  -> dense correspondence factor graph
  -> spherical tangent dense BA
  -> refined local pose/depth/world_points
  -> existing _align_chunk()
  -> existing FrontendOutput emission
```

The BA output must stay in the local PanoVGGT chunk coordinate system until
`_align_chunk()` applies the existing Sim(3)/SE(3) alignment.

## 6. Refined Geometry Entering FrontendOutput

The refined state should be converted back into a normal `PanoVGGTLocalPrediction`:

- `poses_c2w`: refined local chunk poses.
- `depth`: refined local depth.
- `chunk_world_points`: rebuilt from refined pose/depth and ERP bearings.
- `confidence`: combined or updated confidence from PanoVGGT confidence,
  matching confidence, static confidence, and BA validity.
- Optional internal debug sidecar: residual statistics and factor counts.

Then existing tracker code can proceed:

- `_align_chunk()` aligns refined local world points into global coordinates.
- Per-frame emission resizes inverse depth and world points to source image
  resolution.
- `_make_output()` fills the existing `FrontendOutput` fields.

## 7. New Module List

Planned modules, all behind config gates:

- `frontend/pano_vggt/m3_sphere_types.py`
  - Dataclasses for matching head outputs, dense correspondences, factor graph
    edges, BA outputs, debug stats, and config normalization.
- `frontend/pano_vggt/m3_sphere_head.py`
  - Dense matching head attached to PanoVGGT features.
- `frontend/pano_vggt/m3_sphere_hooks.py`
  - Feature extraction/hook utilities for external PanoVGGT models.
- `frontend/pano_vggt/m3_sphere_correspondence.py`
  - GT spherical correspondence generation and pose-guided correspondence
    helpers.
- `frontend/pano_vggt/m3_sphere_matcher.py`
  - Pose-guided local dense matcher.
- `frontend/pano_vggt/m3_sphere_factor_graph.py`
  - Dense correspondence factor graph.
- `frontend/pano_vggt/m3_sphere_ba.py`
  - Spherical tangent dense BA using `Log_b_star(b_hat)` residuals.
- `frontend/pano_vggt/m3_sphere_refinement.py`
  - Chunk-level orchestration that refines `PanoVGGTLocalPrediction`.
- `frontend/pano_vggt/m3_sphere_train.py`
  - Training entry point for the matching head.
- `frontend/pano_vggt/m3_sphere_dataset.py`
  - Dataset adapters for PanoCity and AirSim360-Scene.
- `tests/test_panovggt_m3_sphere_*.py`
  - Synthetic-only tests for geometry, contracts, matcher, BA, and integration.

## 8. Configuration Schema

Default config must keep the feature disabled:

```yaml
PanoVGGT:
  m3_sphere:
    enabled: false
    descriptor_dim: 24
    feature_hook:
      enabled: false
      layer: null
      require_real_features: true
      fallback: error
    matching_head:
      checkpoint: null
      freeze_panovggt: true
      input_dim: null
      descriptor_dim: 24
      hidden_dim: 128
      norm: l2
    dense_matcher:
      enabled: true
      search_radius: 4
      topk: 1
      min_match_confidence: 0.2
      min_static_confidence: 0.2
      pose_guided: true
      max_factors: 65536
    factor_graph:
      temporal_radius: 2
      bidirectional: true
      max_edges: 24
      min_valid_factors_per_edge: 128
    ba:
      enabled: true
      residual_mode: tangent
      debug_pixel_residual: false
      iters: 3
      lm: 0.0001
      fixed_frames: 1
      sample_stride: 1
      max_pose_step: 0.05
      max_depth_step: 0.10
    fake:
      enabled: false
      descriptors: false
    debug:
      save_stats: true
      save_correspondence_preview: false
```

Rules:

- `descriptor_dim` defaults to `24`.
- Matching head output spatial size equals its input feature spatial size.
- Image resolution and feature resolution are not hard-coded.
- Fake descriptors are legal only when `PanoVGGT.engine: fake`,
  `Dataset.synthetic: true`, or an explicit fake flag is set for tests.

## 9. Dataset and Training Design

Training data:

- PanoCity RGB, depth, and pose ground truth.
- AirSim360-Scene RGB, depth, and pose ground truth.
- Dataset adapters should normalize into a shared sample schema:
  - `images`: `N x 3 x H x W`.
  - `depths`: `N x 1 x H x W`.
  - `poses_c2w`: `N x 4 x 4`.
  - `valid_depth_mask`: `N x 1 x H x W`.
  - Optional semantic/static masks when available.

Training loop:

- Freeze external PanoVGGT by default.
- Run PanoVGGT feature extraction.
- Feed features to matching head.
- Generate GT spherical correspondences from GT depth and pose.
- Train descriptors and confidences from GT correspondences.
- Keep W&B/visualization enabled for remote experiments according to project
  rules.

Unit tests:

- Use synthetic samples only.
- Do not require real PanoCity/AirSim360 paths.
- Do not require real external PanoVGGT checkpoint.

## 10. GT Spherical Correspondence Generation Design

For source frame `i` and target frame `j`:

1. Build source ERP pixel centers `p_i = (u + 0.5, v + 0.5)` at the chosen
   supervision grid.
2. Convert source pixels to unit bearings `b_i`.
3. Use GT depth `d_i` to get source camera points:
   `x_i = d_i * b_i`.
4. Transform into world:
   `X_w = T_i_c2w * [x_i, 1]`.
5. Transform into target camera:
   `x_j = T_j_w2c * X_w`.
6. Target bearing:
   `b_star = normalize(x_j)`.
7. Optional debug target pixel:
   `p_j = bearing_to_erp_pixel(b_star)`.
8. Validity:
   - finite depth and pose;
   - positive depth in source;
   - optional target depth consistency check;
   - optional static/object mask.

The training target for BA and factor graph is `b_star`, not ERP pixel delta.

## 11. Matching Head Structure

Inputs:

- PanoVGGT feature tensor at feature resolution.
- Shape should be inferred from the tensor, for example `N x C x Hf x Wf`.

Outputs:

- `dense_descriptor`: `N x descriptor_dim x Hf x Wf`.
- `match_confidence`: `N x 1 x Hf x Wf`.
- `static_confidence`: `N x 1 x Hf x Wf`.

Design constraints:

- Output spatial size `Hf x Wf` must equal input feature spatial size.
- `descriptor_dim` defaults to `24`.
- Descriptors should be L2-normalized unless explicitly disabled for ablation.
- Confidence heads use bounded activations such as sigmoid.

## 12. Losses

Descriptor losses:

- Positive correspondence contrastive loss on GT spherical correspondences.
- Local hard negative loss within a pose-guided search neighborhood.
- Optional InfoNCE over sampled correspondences.

Confidence losses:

- Match confidence supervised by valid GT correspondence and descriptor match
  correctness.
- Static confidence supervised by static masks when available, otherwise by
  geometry consistency heuristics only when explicitly enabled.
- Calibration loss so confidence predicts match success probability.

Geometry losses:

- Main BA residual loss:
  `rho(||Log_b_star(b_hat)||_2)`.
- Depth consistency or inverse-depth regularization as auxiliary terms.
- ERP pixel residual only as debug/ablation metric.

## 13. Dense Matcher

Inference matcher:

- Use PanoVGGT initial pose/depth to project source feature-grid samples into
  target feature-grid neighborhoods.
- Search locally around the projected target location.
- Score descriptor similarity with match confidence and static confidence.
- Return dense correspondences:
  - source frame id;
  - target frame id;
  - source feature-grid coordinate or ERP pixel coordinate;
  - matched target bearing `b_star`;
  - match confidence weight;
  - static confidence weight;
  - validity mask.

ERP seam handling:

- Horizontal wrap must be explicit.
- Feature-grid search must wrap horizontally when the feature map represents
  ERP longitude.

## 14. Factor Graph

Graph nodes:

- Chunk/window frame poses.
- Source-frame inverse-depth map values.
- Optional world-point map view derived from pose/depth.

Graph factors:

- One dense correspondence factor per valid source sample.
- Edge connects source frame `i` and target frame `j`.
- Factor residual is the `S^2` tangent residual at matched target bearing.

Factor selection:

- Temporal edges first.
- Optional proximity/overlap edges from PanoVGGT initial geometry.
- Cap factor count per edge and per chunk to avoid CPU/RAM pressure.

## 15. Spherical BA Formula and Solver Design

For a source sample in frame `i` matched to target frame `j`:

```text
b_i = erp_pixel_to_bearing(p_i)
x_i = b_i / inverse_depth_i(p_i)
X_w = T_i_c2w * [x_i, 1]
x_j_hat = T_j_w2c * X_w
b_hat = normalize(x_j_hat)
r = Log_b_star(b_hat) in R^2
```

Residual:

```text
r = E(b_star)^T * (theta / sin(theta)) * (b_hat - cos(theta) * b_star)
theta = acos(dot(b_star, b_hat))
E(b_star) = tangent basis at b_star
```

Objective:

```text
min_{poses, inverse_depth}
  sum_k rho(||r_k||_2) * w_match_k * w_static_k * w_area_k
  + damping / priors
```

Solver:

- Start with a correctness-first PyTorch solver.
- Use SE(3) left retraction for poses.
- Use log inverse-depth updates for depth positivity.
- Fix one or more anchor frames.
- Use LM damping and bounded update norms.
- Initial version may use autograd Jacobians or finite subset normal equations;
  later version can add closed-form tangent Jacobians.
- Pixel residuals may be logged as debug stats but must not drive the main
  PanoVGGT-M3-Sphere BA objective.

## 16. Tracker Integration Route

Integration steps:

1. Parse and validate `PanoVGGT.m3_sphere`.
2. Keep disabled path byte-for-byte behaviorally compatible with current
   `panovggt_long`.
3. Add an internal `refine_prediction_if_enabled(...)` call after
   `engine.infer(images)` and before `_align_chunk(...)`.
4. Return a normal `PanoVGGTLocalPrediction` for existing alignment/emission.
5. Route debug stats to logs or output directories without changing
   `FrontendOutput`.
6. Add tests that compare old disabled path outputs before and after the gated
   integration.

## 17. Final/Global BA and Gaussian Map Consistency

The PanoVGGT-M3-Sphere local BA refines frontend geometry before Gaussian seed
creation. The Gaussian backend may still refine poses and map anchors later.

Consistency rules:

- Frontend refined local geometry enters Gaussian map through existing
  `FrontendOutput`.
- Global chunk alignment remains responsible for putting local PanoVGGT chunks
  into the SLAM world coordinate system.
- Backend pose feedback can continue to update tracker pose cache through
  `apply_backend_pose_updates`.
- Final/global backend BA should be treated as a downstream map optimization,
  not as a replacement for frontend dense matching BA.
- If backend pose feedback changes poses, future chunk alignment should consume
  that feedback as it does today.

## 18. Test Plan

Contract tests:

- Config defaults keep `PanoVGGT.m3_sphere.enabled == false`.
- Disabled `panovggt_long` fake smoke behavior is unchanged.
- Matching output spatial size equals input feature spatial size.
- `descriptor_dim` defaults to `24`.
- Fake descriptors are rejected outside explicit fake/synthetic mode.
- Missing feature hook/head in real inference raises a clear error.

Geometry tests:

- GT spherical correspondence generation works on synthetic pose/depth.
- ERP seam wrapping is handled.
- `Log_b_star(b_hat)` residual has shape `... x 2`.
- Tangent residual is near zero for identical bearings.
- Pixel residual is available only under debug/ablation configuration.

Training tests:

- Tiny synthetic training step runs without real data/checkpoints.
- Losses are finite and backpropagate to matching head parameters.
- Dataset adapters can be mocked with tiny temporary folders.

Inference tests:

- Pose-guided matcher returns valid dense correspondences on synthetic tensors.
- Matcher handles missing/low confidence by dropping factors.
- Factor graph caps factor counts.

BA tests:

- Spherical tangent BA reduces synthetic angular residual.
- Fixed frame remains fixed.
- Depth updates remain positive and bounded.
- Solver handles weak/empty factor cases gracefully.

SLAM integration tests:

- Gated enabled fake/synthetic smoke produces `FrontendOutput` through existing
  fields.
- Disabled path `panovggt_long` smoke remains unchanged.
- Gaussian initializer receives refined `world_points` without API changes.

## 19. Five Implementation Phases and Acceptance Criteria

### Phase 1: Interface Contract and Spherical Geometry Foundation

Scope:

- Add internal dataclasses and config validation plan.
- Add GT spherical correspondence generation.
- Add tangent residual tests.
- Do not touch tracker or engine main flow.

Acceptance:

- Unit tests use only synthetic tensors.
- `FrontendOutput` API remains unchanged.
- Disabled `panovggt_long` path is unaffected.

### Phase 2: Training Loop

Scope:

- Add matching head module.
- Add synthetic dataset smoke path plus real dataset adapter interfaces.
- Add descriptor/confidence losses.
- Add tiny training smoke test without real checkpoint.

Acceptance:

- Matching head output spatial size equals input feature size.
- `descriptor_dim` default is `24`.
- Training smoke backpropagates through the head.
- Fake descriptors are limited to explicit fake/synthetic mode.

### Phase 3: Inference Loop

Scope:

- Add feature hook validation.
- Add pose-guided local dense matcher.
- Add dense correspondence sidecar outputs.
- Keep all enabled behavior under config gate.

Acceptance:

- Missing real feature hook/head errors clearly unless explicit fallback is set.
- Synthetic inference matcher returns finite correspondences.
- Disabled `panovggt_long` path remains unchanged.

### Phase 4: Optimization Loop

Scope:

- Add dense correspondence factor graph.
- Add spherical tangent dense BA.
- Rebuild refined depth/world_points from optimized state.

Acceptance:

- Main BA residual is `Log_b_star(b_hat)` in `R^2`.
- Pixel residual appears only in debug/ablation stats.
- BA reduces synthetic angular residual and keeps updates bounded.

### Phase 5: SLAM Integration, Experiments, and Hardening

Scope:

- Insert gated refinement before `_align_chunk()`.
- Pass refined pose/depth/world_points through existing `FrontendOutput`.
- Run fake/synthetic smoke tests, then remote experiments under project safety
  rules.
- Add debug stats, fallback policy, and hardening tests.

Acceptance:

- Existing disabled path tests pass.
- Enabled synthetic SLAM smoke passes.
- Gaussian initializer consumes refined `world_points`.
- Remote experiment launch keeps W&B and visualization enabled.

## 20. Fallback and Debug Stats Design

Fallback policy:

- Default real inference fallback is `error`.
- Explicit fallback options may include:
  - `skip_refinement`: use raw PanoVGGT prediction and log reason.
  - `debug_pixel_only`: compute debug pixel residuals without BA.
  - `fake_descriptors`: allowed only in fake/synthetic mode.

Debug stats:

- Feature hook availability.
- Matching head input/output shapes.
- Descriptor dimension.
- Number of candidate matches and accepted matches.
- Mean/min/max match confidence.
- Mean/min/max static confidence.
- Factor counts per edge and per chunk.
- Tangent residual mean, median, and max angular norm.
- Optional debug ERP pixel residual mean.
- BA iterations, damping, pose update norm, depth update norm.
- Refined world point finite ratio.
- Fallback reason when refinement is skipped.

Stats should be written to local diagnostics and W&B when running remote
training/experiments, without changing `FrontendOutput`.
