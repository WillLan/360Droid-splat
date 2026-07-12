# Stage 3: Spherical Dense BA and Recurrent Per-Pixel Gaussian Refiner

## Boundary and frozen baseline

Stage 3 is a standalone, config-gated training stack. It freezes PanoVGGT, the
Stage 1 24D spherical adapter, the trained Stage 2 Gaussian Head, and the
ResNet18 error backbone. It trains only the error projections/router and the
full-resolution recurrent Refiner. It does not alter `FrontendOutput`, SLAM
tracker dispatch, backend fusion, or the canonical one-Gaussian-per-valid-pixel
representation.

The fixed schedule is:

```text
Stage 2 observation
-> BA0 -> Refine1
-> BA1 -> Refine2
-> BA2 -> Refine3
```

There is no final BA, RAE, point/voxel attention, free XYZ update, or depth
probability volume.

## Adapter matches

Each four-frame window caches matches once. Every source frame samples 2048
query pixels with the Stage 1 `fibonacci_depth_filtered` sampler in the
`[0.05, 20] m` range. A query is matched independently into all three other
views with the exact Stage 1 prediction score:

```text
cosine(normalized_query, normalized_target) / 0.07 + log(cos(target_latitude))
```

The target is the global argmax over the full `504 x 1008` ERP. Computation is
chunked by 32 queries, but does not use pose windows or approximate candidate
search. Four frames produce at most 8192 shared sparse-depth variables and
24576 directed factors. The same cache is reused by all three BA calls.

## Spherical BA and dense depth shift

For source bearing `b_i`, sparse range `d_i`, and c2w poses, BA predicts the
target bearing by transforming `d_i b_i` through world coordinates. Its main
residual is always:

```text
Log_{matched_target_bearing}(predicted_target_bearing) in R^2
```

The first pose is fixed. The other poses use left-multiplicative SE(3) updates,
and each source query owns one log-inverse-depth variable shared by its three
target factors. Huber-weighted LM/GN builds factor-local Jacobians and uses a
Schur complement to eliminate the diagonal sparse-depth block before solving
the at-most `18 x 18` pose system.

After each accepted solve, a robust affine fit relates the incoming query
depths to the optimized sparse depths for every frame. The fitted
`D_BA = a D_in + b` is applied to all finite pixels in `[0.05, 20] m`; farther
pixels remain unchanged. Solver outputs and `a/b` are detached, while the
affine application keeps the linear gradient to `D_in`. A failed, non-finite,
under-constrained, or more-than-1.05x-worse solve returns the input pose/depth.

## ReSplat-style feedback and Refiner

Every feedback group renders each target from the other three source
observations. The target image and detached render are compared by RGB
pixel-unshuffle and frozen ERP-padded ResNet18 features at 1/2, 1/4, and 1/8
resolution. Their trainable projection produces a 32D quarter-resolution error
map.

Current Gaussian centers are projected into the three other cameras. Target
error is sampled with periodic longitude and accepted only when the Gaussian
survived target-conditioned opacity pruning, was rasterizer-visible, has alpha
above 0.05, and agrees with rendered Euclidean depth. Signed mean, absolute
mean, coverage, and an area-weighted global token are projected back to a 32D
full-resolution source-pixel error feature. This is error transport only; it is
not RAE or precise per-source compositing responsibility.

The Refiner encodes:

- static `24D adapter + RGB` to 32D;
- the normalized 39D current Gaussian state to 32D;
- routed rendering error to 32D.

The concatenated 96D input passes through `1x1 96->64`, one depthwise
ERP-aware context block, and a 32D spherical ConvGRU. There is no BA Encoder or
iteration embedding. The Geometry Head updates depth, SO(3) local rotation,
and log-scale multiplier. The Appearance Head updates degree-2 RGB SH and
degree-1 density SH. All output layers are zero initialized. Per-iteration
bounds shrink from coarse to fine. Pixels deeper than 20 m keep geometry fixed
but may update appearance.

## Training and gradient path

Four render groups are used per step: feedback after BA0, loss/feedback after
BA1, loss/feedback after BA2, and final loss after Refine3. This is 16 target
rasterizations for a four-frame window. The error branch consumes detached
renders, while the render loss uses the non-detached tensors. Refine1 and
Refine2 observations and hidden states are detached after their stage loss;
the shared network accumulates gradients from all three stages and performs
one optimizer step.

The stage weights are `0.64, 0.80, 1.0`. Every stage combines ERP-area L1,
periodic DSSIM, cached-match S2 geometry, a BA-depth anchor, and normalized
update regularization. GT pose/depth never enter the loss.

Every 200 steps, diagnostics render Initial, BA0, Refine1, BA1, Refine2, BA2,
and Refine3. They log leave-one-out rendering metrics, scale-aligned pose error,
raw/scale-aligned depth metrics, BA residuals and affine parameters, and update
statistics. Refiner snapshots are asserted to preserve the preceding BA pose.
Validation runs every 1000 steps and writes latest, best final LOO PSNR, and
best final pose ATE checkpoints.

## Formal configuration and launch gate

The formal config is
`configs/stage3_spherical_ba_recurrent_refiner_omni360.yaml`. It uses the
audited Omni360 DTW/NYC image, H5 depth, and UE/AirSim pose CSV loader, four
views with random stride 2-6, full `504 x 1008` Refiner resolution, 20k steps,
BF16, two-rank DDP, and effective batch size four.

The configured Stage 2 best checkpoint was verified on server 50902 at the
nested joint-training output path. Its SHA256 is
`ab9aaa0a301fe8d601dde43ed7b0cedb37fbf1e3cf1b58c75f7bcc76227be850`;
the Stage 3 loader rejects a mismatch before training.

Before a real launch, verify the configured Stage 2 checkpoint exists on the
server, then run CUDA smoke tiers at `126x252`, `252x504`, and `504x1008`.
The tier driver is `tools/smoke_stage3_cuda.py`; it enables synchronized
component profiling and writes `smoke_summary.json` without changing the
formal training config.
The real config refuses the CPU fallback renderer. A full-resolution gate must
record peak allocated/reserved GPU memory, RAM/swap, matcher/BA/rasterizer
times, 16-render step time, and leave safe GPU memory headroom. Remote launch
must separately follow the project tmux, W&B, visualization, GPU ownership,
and resource-safety rules.
