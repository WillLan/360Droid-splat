# Stage 3 BA stabilization and 200-step pilot summary

## Outcome

The original Stage 3 BA reduced its spherical correspondence objective but
usually moved the cameras farther from GT. The stabilized formal BA is now:

```text
BA0 only
+ dense_depth_mode = none
+ gauge_mode = initial_baseline
+ solver_mode = standard_lm
+ pose_update_side = right
+ pose_dof_mode = rotation_only
+ reliability_keep_fraction = 0.10
+ max_pose_update_deg = 0.02 per LM iteration
```

Camera centers and dense depth are invariant by construction. Sparse query
depths remain Schur-eliminated nuisance variables. Full SE(3), translation-only,
and rotation-then-translation modes are retained only as gated research
ablations because their translation gains did not transfer to a second
holdout.

## What was wrong

Three independent problems were measured:

1. Dense affine propagation transferred noisy sparse depth corrections to the
   full depth image. Disabling it made all dense depth metrics invariant.
2. Bearing-only BA had a global-scale gauge. The hard initial-baseline KKT
   constraint eliminated scale drift without changing bearing geometry.
3. The original `5 degree / 0.05 translation` limits were about two orders of
   magnitude too large for an already accurate PanoVGGT initialization. LM
   followed ambiguous, spatially smooth Adapter matches to a lower feature
   objective but a worse GT pose.

The standard gain-ratio LM, right-local perturbation, exact reduced-DOF Schur
solve, and strict rotation trust region were all tested independently. Tight
forward-backward tolerances, parallax gates, local subpixel soft-argmax, and
independent-peak margins did not improve both absolute and relative rotation
consistently and remain disabled.

## Original 200-step controlled ablation

The five original BA0-only DDP runs were saved under:

```text
outputs/stage3_ba0_ablation_gpu67_20260712_r1
```

The required axes were isolated:

- no dense affine: dense depth delta became exactly zero;
- fixed scale gauge: alignment-scale drift was nearly eliminated;
- standard LM: produced valid gain ratios and damping updates, but could not
  correct a biased correspondence optimum by itself.

The best old E4 path still changed validation rotation by `+0.14341 deg`, RPE
rotation by `+0.07312 deg`, and ATE by `+0.00329`, so it was not accepted as a
useful BA.

## Frozen BA-only validation

The final rotation-only configuration was selected on two disjoint validation
ranges that were not used by the initial 32-window factor sweeps:

| Validation batches | Windows | Absolute rotation delta | RPE rotation delta |
|---|---:|---:|---:|
| 32-95 | 64 | -0.01176 deg | -0.00755 deg |
| 96-223 | 128 | -0.00523 deg | -0.00396 deg |
| Weighted aggregate | 192 | -0.00741 deg | -0.00515 deg |

Translation, ATE, and translation-direction metrics are exactly invariant in
this mode. GT was read only after BA for diagnostics; it is not used by the
matcher, solver, acceptance rule, loss, or Refiner.

## Full-resolution 200-step DDP pilot

The final pilot used physical GPUs 6 and 7, effective batch size four, BF16,
W&B online, and visualization enabled:

```text
outputs/stage3_ba_rotation_only_trust002_gpu67_20260713_pilot200_r1
```

W&B run: `qzbamwdp`

Runtime was 15 minutes 51 seconds including initialization, step-200
diagnostics, eight-batch validation, checkpointing, and W&B upload. Peak
observed memory was approximately 26.0 GiB on GPU 6 and 24.4 GiB on GPU 7.
No OOM, non-finite loss/gradient, W&B error, or opacity collapse occurred.

The step-200 training window had:

- BA residual: `0.85209 deg -> 0.53909 deg`;
- objective: `0.08692 -> 0.06271`;
- 3/3 LM steps accepted, mean gain ratio `1.4132`;
- gauge scale `1.0`, dense depth scale/shift `1.0 / 0.0`.

The step-200 diagnostic window had:

| Metric | Initial | BA0 | Refine3 |
|---|---:|---:|---:|
| pose rotation mean (deg) | 0.49141 | 0.44802 | 0.44802 |
| RPE rotation mean (deg) | 0.31329 | 0.29242 | 0.29242 |
| LOO PSNR (dB) | 11.62728 | 11.62833 | 12.51856 |
| LOO SSIM | 0.18468 | 0.18550 | 0.23435 |
| confidence p50 | 0.25272 | 0.25272 | 0.39232 |

On the fixed eight-batch validation aggregate, BA0 improved RPE rotation by
`0.00078 deg`, kept all translation metrics exactly fixed, and worsened mean
absolute rotation by `0.00555 deg` (3.5%, inside the predeclared 5% safety
gate). Across the larger 192-window frozen audit, both rotation metrics
improved.

Refine3 improved validation PSNR by `2.2410 dB`, SSIM by `0.05399`, and L1 by
`0.09354`. Raw depth AbsRel improved slightly, while scale-aligned AbsRel
worsened from `0.06508` to `0.06688`; depth quality is therefore mixed and must
remain a formal-training guardrail. Gaussian rotation updates were close to
their per-round bounds at step 200, so their saturation should also be tracked
during a longer run.

## Artifacts and next launch

The pilot produced:

```text
run/checkpoints/latest.pt
run/checkpoints/best_val_loo_psnr.pt
run/checkpoints/best_val_pose_ate.pt
run/visualizations/step_0000200.png
run/visualizations/val_0000200.png
```

Checkpoint format is `spherical_ba_recurrent_gaussian_refiner_v1`, global step
is 200, all 103 saved model tensors are finite, and the recorded Adapter and
Stage 2 SHA256 values match the configured frozen checkpoints.

A 20k launch is not automatic. If authorized separately, use the formal config
and keep diagnostics every 200 steps, validation/checkpoint every 1000 steps,
W&B online, and the following stop conditions:

- validation rotation or RPE rotation exceeds the initial value by more than
  5%;
- opacity/confidence collapses toward zero;
- scale-aligned depth continues to worsen without geometry improvement;
- Gaussian rotation updates remain saturated without rendering gains;
- non-finite values, rising memory, or insufficient GPU headroom.

At the measured pilot throughput, 20k steps are expected to take about 26-28
hours on two RTX 5090 GPUs, including periodic diagnostics and validation.

## GT-correspondence oracle protocol

`tools/evaluate_stage3_ba_oracle.py` isolates solver correctness from Adapter
matching quality.  It preserves each real validation window's Stage 3 source
queries, selected factor positions, and confidence weights, while replacing
the target bearing with a continuous projection computed from GT Euclidean-ray
depth and GT c2w poses.  GT target depth consistency removes invalid and
occluded projections.

The evaluator runs three matched arms:

1. Adapter correspondence plus Stage 2 depth;
2. GT correspondence plus Stage 2 depth;
3. GT correspondence plus GT depth.

All arms use both the formal solver and a diagnostic rotation-only solver with
eight LM iterations and a `0.5 degree` per-iteration rotation limit.  The
formal arm measures deployed behavior; the diagnostic arm prevents the formal
`3 x 0.02 degree` trust limit from hiding the solver's capture range.  GT is
strictly diagnostic and is not imported by the training or runtime BA path.
