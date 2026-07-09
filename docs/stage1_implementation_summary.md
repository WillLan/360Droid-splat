# Stage 1 Implementation Summary

## Completed Scope

Stage 1 is implemented as a standalone spherical Selfi adapter training stack.
It does not modify the existing SLAM frontend, mapper, backend, or
`FrontendOutput` API.

Implemented components:

- Stage 1A spherical ERP geometry and pseudo correspondence.
- Stage 1B frozen PanoVGGT feature wrapper and Spherical DPT adapter.
- Stage 1C spherical Selfi alignment loss, manifest dataset, training entry,
  config, visualization, and dataset/overlap tools.
- Stage 1 completion docs and manifest placeholder.

## Files Added Or Updated

Geometry:

- `geometry/spherical_erp.py`
- `geometry/spherical_projection.py`
- `geometry/spherical_pseudo_correspondence.py`

Model and loss:

- `models/panovggt_feature_wrapper.py`
- `models/spherical_selfi_dpt_adapter.py`
- `losses/spherical_selfi_alignment_loss.py`

Data, config, and training:

- `data/stage1_pano_sequence_dataset.py`
- `data/stage1_dataset_manifest.json`
- `configs/stage1_spherical_selfi_adapter.yaml`
- `training/train_spherical_selfi_adapter.py`

Tools:

- `tools/build_stage1_dataset_manifest.py`
- `tools/check_stage1_overlap.py`
- `tools/dump_panovggt_feature_shapes.py`
- `tools/visualize_spherical_adapter_matches.py`

Tests:

- `tests/test_spherical_erp_geometry.py`
- `tests/test_spherical_pseudo_correspondence.py`
- `tests/test_spherical_selfi_dpt_adapter.py`
- `tests/test_spherical_selfi_alignment_loss.py`
- `tests/test_stage1_dataset_manifest.py`

Docs:

- `docs/stage1_selfi_adapter_design.md`
- `docs/stage1_dataset_plan.md`
- `docs/stage1_implementation_summary.md`

Packaging:

- `pyproject.toml` package discovery now includes `geometry`, `models`,
  `losses`, and `data`.

## Geometry Convention

The ERP convention matches the project panoramic camera convention:

- `+X` points right.
- `+Y` points down.
- `+Z` points forward.
- ERP pixel coordinates are floating point `[u, v]`.
- Pixel centers are represented as `col + 0.5, row + 0.5`.
- Horizontal longitude wraps into `[0, W)`.
- Vertical latitude does not wrap; image sampling clamps vertically to the
  nearest valid pixel center.
- Default ERP size is `H=504, W=1008`, while public functions accept explicit
  `height` and `width`.

Angular distances are great-circle distances on the unit sphere:

```text
theta = acos(clamp(dot(normalize(ray_a), normalize(ray_b)), -1, 1))
```

## Pose, Depth, And Correspondence Convention

Projection uses camera-to-world poses `T_c2w`.

Depth is interpreted as Euclidean range along an ERP unit ray:

```text
X_cam = depth * ray
X_world = T_src_c2w * X_cam
X_tgt = inverse(T_tgt_c2w) * X_world
target_ray = normalize(X_tgt)
target_uv = ray_to_erp(target_ray)
```

The implementation does not infer or convert z-depth. If a caller has z-depth,
it must convert it to Euclidean ray depth before calling the pseudo
correspondence utility.

Pseudo correspondence visibility uses target range consistency:

```text
visible = abs(range_pred - range_tgt) / clamp(range_tgt, min=eps)
          < visibility_rel_thresh
```

## Adapter Contract

`PanoVGGTFeatureWrapper`:

- Wraps a caller-provided PanoVGGT-like `nn.Module`.
- Requires exactly four hook names.
- Sets the wrapped model to `eval()` and disables all PanoVGGT parameter grads.
- Runs the wrapped model under `torch.no_grad()` by default.
- Captures each hook output and normalizes map/token features to
  `B x V x C x H x W`.
- Preserves initial depth, c2w poses, and optional world points when returned by
  the wrapped model.

`SphericalSelfiDPTAdapter`:

- Accepts four normalized map/token feature stages.
- Projects each stage to `hidden_dim`.
- Uses top-down DPT-style fusion.
- Uses horizontal circular padding in ERP convolution blocks and vertical
  replicate padding.
- Upsamples to full ERP resolution, defaulting to `H=504, W=1008`.
- Outputs `B x V x 24 x H x W` by default.
- Applies L2 normalization along the feature channel dimension.

## Reprojection / Alignment Error

The default Stage 1 training loss is full-resolution spherical soft-label CE,
not ERP pixel distance. The target descriptor map keeps the full adapter output
resolution, and the target distribution is built from unit-sphere geodesic
distance:

```python
score_i = dot(f_src, f_tgt_i) / temperature
P_i = softmax(score_i + log(area_i))
Q_i = softmax(-geodesic(ray_i, target_ray) ** 2 / (2 * sigma ** 2) + log(area_i))
loss = CE(Q, P) + expected_geodesic_weight * sum_i P_i * geodesic(ray_i, target_ray)
```

The geodesic distance uses the stable great-circle `atan2(||cross||, dot)` form.
`erp_aux` is a seam-aware pixel diagnostic for predicted argmax matches; it is
not the main reprojection residual.

The overlap and visualization tools also report angular metrics by converting
ERP pixels back to unit rays and measuring great-circle distance.

## Explicitly Not Implemented

Stage 1 intentionally does not implement:

- SLAM frontend integration.
- Dense BA integration.
- Mapping/backend changes.
- `FrontendOutput` API changes.
- Remote experiment launch.

Real PanoVGGT checkpoint validation is not bundled with the local tests. The
repository now provides `tools/dump_panovggt_feature_shapes.py` to verify real
hook names and feature shapes once a real checkpoint/config is available.

## Verification Commands

Core tests:

```bash
python -m pytest -q tests/test_spherical_erp_geometry.py tests/test_spherical_pseudo_correspondence.py tests/test_spherical_selfi_dpt_adapter.py tests/test_spherical_selfi_alignment_loss.py tests/test_stage1_dataset_manifest.py
```

Compile check:

```bash
python -m compileall geometry models losses data training tools tests/test_spherical_selfi_alignment_loss.py tests/test_stage1_dataset_manifest.py
```

Synthetic training smoke:

```bash
python training/train_spherical_selfi_adapter.py --config .codex_tmp/stage1c_smoke/config.yaml --max-steps 1 --wandb-mode disabled
```

Feature-shape smoke:

```bash
python tools/dump_panovggt_feature_shapes.py --config .codex_tmp/stage1c_smoke/config.yaml --synthetic --device cpu --views 4
```

Tool smokes:

```bash
python tools/check_stage1_overlap.py --manifest .codex_tmp/stage1c_smoke/manifest.json --image-height 16 --image-width 32 --num-query-per-pair 8 --max-windows 1
python tools/visualize_spherical_adapter_matches.py --manifest .codex_tmp/stage1c_smoke/manifest.json --output .codex_tmp/stage1c_smoke/matches.png --height 16 --width 32 --max-matches 8 --metrics-json .codex_tmp/stage1c_smoke/metrics.json
```

Latest local result:

```text
21 tests passed.
compileall succeeded.
synthetic training smoke completed 1 step and wrote adapter_latest.pt.
synthetic feature dump output shape: [1, 4, 24, 16, 32], channel norm mean: 1.0.
geometry overlap smoke: mean_valid_corr_ratio=1.0,
  mean_angular_pseudo_reprojection_error_deg=0.0.
match visualization smoke: mean_angular_error_deg=0.0,
  PCK@1/3/5deg=1.0.
```
