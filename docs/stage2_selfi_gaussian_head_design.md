# Stage 2: Selfi-style Spherical Per-Pixel Gaussian Head

## Status and boundary

Stage 2 is a standalone, config-gated prediction and training stack. It does
not alter `FrontendOutput`, tracker dispatch, mapping, or the backend map. The
canonical frontend representation remains exactly one Gaussian observation per
valid source ERP pixel. No anchor, voxel, recurrent updater, point transformer,
or old `frontend/pano_vggt/gaussian_head.py` code is imported.

The implementation follows the Gaussian Head description in Selfi where the
paper is explicit: the network consumes a full-resolution aligned 24D feature
and RGB; a shared U-Net feeds separate quaternion/depth convolutional heads and
point-wise scale, RGB-SH, and density-SH heads. Initial depth is used after the
network to turn the predicted residual into geometry, not concatenated into the
Head input. Selfi does not publish its exact U-Net channel table, SH degrees,
activations, or parameter bounds, so the concrete choices below are project
decisions recorded in config.

Unlike Selfi's six-source/five-held-out-target training, this project uses all
selected frames as sources and reconstructs every selected view from the union
of all source Gaussians, including its own observations. This is an intentional
experiment choice. It introduces a copy-through risk, so validation always
supports all-source, self-only, and leave-one-out diagnostics.

## Frozen inputs

The formal input is `B x S x 3 x 504 x 1008`. A reusable PanoVGGT factory
captures the four configured stages and returns initial camera-to-world poses,
Euclidean ERP ray depth, and four feature maps. The formal Stage 1 adapter
checkpoint is loaded through a strict `spherical_selfi_adapter_v1` loader. The
loader checks:

- file SHA256 and checkpoint format;
- descriptor dimension and output image size;
- four hook names and token grids;
- `c2w` pose and `euclidean_ray_depth` conventions;
- strict adapter state-dict compatibility.

PanoVGGT and adapter stay in `eval()`, have `requires_grad=False`, and run under
`no_grad`. Formal Stage 2 input uses the checkpoint:

```text
/mnt/disk1/lanboyang/Project/360Droid-splat/outputs/
stage1_selfi_adapter_airsim_dtw_nyc_fullres_spherical_ce_depth20_ddp2/
checkpoints/best_val_angular_error.pt
```

Its expected SHA256 is
`95cdbb5404acba8654e868335ede6a00c281451d7ef7c104d3a10f63fc3730b9`.

For memory-tier smoke tests, PanoVGGT and the adapter still run at
`504 x 1008`; only their dense output, RGB, and initial depth are explicitly
resized afterward with periodic-longitude interpolation.

## Network

`SphericalSelfiGaussianHead` receives only:

```text
adapter feature  B x S x 24 x Hg x Wg
RGB              B x S x  3 x Hg x Wg
```

The tensors are concatenated into `[B*S, 27, Hg, Wg]`. The four-level U-Net has
channels `[32, 64, 128, 256]`; each level uses two `3x3 Conv + GroupNorm +
GELU` blocks. Stride-2 convolution downsamples. The decoder uses periodic ERP
bilinear interpolation to each exact skip size, concatenation, and two further
convolutions. All spatial convolution pads longitude circularly and latitude by
replication. The final shared decoder feature has 32 channels.

Prediction branches are:

```text
Conv_q: 32 -> 4
Conv_D: 32 -> 1
MLP:    32 -> 64 -> scale(3), RGB-SH(27), density-SH(4)
```

RGB SH degree is 2 and density SH degree is 1. Total raw channels are 39.
Initialization gives zero depth residual, identity local quaternion, unit scale
multiplier, neutral RGB, and source-view opacity/confidence 0.1.

## Observation contract and geometry

The dense output shapes are:

| Field | Shape | Meaning |
|---|---:|---|
| `initial_depth` | `B x S x 1 x H x W` | frozen Euclidean ray range |
| `depth_residual` | `B x S x 1 x H x W` | `0.25 D0 tanh(raw)` |
| `refined_depth` | `B x S x 1 x H x W` | `D0 + delta D` |
| `local_quaternion` | `B x S x 4 x H x W` | unit `wxyz`, source-camera local |
| `log_scale_multiplier` | `B x S x 3 x H x W` | anisotropic relative footprint |
| `rgb_sh` | `B x S x 9 x 3 x H x W` | source-camera RGB SH |
| `density_sh` | `B x S x 4 x H x W` | view-conditioned opacity logits |
| `confidence` | `B x S x 1 x H x W` | source-ray density SH after sigmoid |
| `valid_mask` | `B x S x 1 x H x W` | finite positive-depth validity |
| `source_uv/ray` | `H x W x 2/3` | immutable pixel-center provenance |
| `frame_ids` | `B x S` | source provenance |

The center is derived, never freely predicted or duplicated as canonical
storage:

```text
X_cam   = refined_depth * source_ray
X_world = R_c2w * X_cam + t_c2w
```

`centers_camera()`, `centers_world()`, and `scales()` materialize geometry
lazily. Scale starts from depth and the runtime ERP pixel solid angle,
including `cos(latitude)`, then applies the predicted anisotropic exponential
multiplier. `with_geometry()` returns a new observation using updated poses or
depth, which is the Stage 3 BA boundary. Consequently changing depth from `d`
to `d'` naturally rescales the footprint by approximately `d'/d`.

## Renderer mapping

For each target camera, `materialize_batch()` evaluates source-camera RGB and
density SH using the target viewing direction and produces the existing
PFGS360 contract:

| PFGS360 field | Stage 2 value |
|---|---|
| `get_xyz` | derived world centers, `N x 3` |
| `get_rotation` | `q_c2w * q_local`, `N x 4 wxyz` |
| `get_scaling` | positive world scale, `N x 3` |
| `get_opacity` | evaluated density SH then sigmoid, `N x 1` |
| `get_features` | evaluated RGB SH in `[0,1]`, `N x 3` |

The target-conditioned materialization removes the lowest 30% opacity values
before that render. It never changes the canonical observation or persists a
compacted representation. At `S=4`, `504 x 1008` there are 2,032,128 canonical
observations and approximately 1,422,490 materialized Gaussians per target
render.

Only the real CUDA `gsplat360` path is supported for Stage 2 renderer and
training tests. There is no Stage 2 CPU renderer. The backend's existing
detached, front-hemisphere-only fallback is rejected by constructing
`PFGS360Renderer(allow_fallback=False)`.

## Dataset and optimization

`Stage2SourceReconstructionDataset` reuses the Stage 1 manifest and ERP image
loader, groups records by scene/sequence, and constructs source-only windows.
Training stride is deterministically randomized per epoch/index in `[2,6]`;
validation uses the minimum stride. There is no target field or target split in
a sample, and manifest train/validation records remain isolated.

For every batch item and every source camera, all canonical source observations
are materialized for that target and rendered sequentially. Default loss is:

```text
L = latitude_weighted_L1(render(all sources), source RGB)
    + 1e-3 SmoothL1(delta D / D0, 0)
```

Optional default-zero terms are periodic-longitude DSSIM, rendered initial
depth consistency, and a Stage 1 pseudo-correspondence geometry consistency
loss. The geometry term generates pseudo matches from frozen initial
pose/depth, samples refined depth at both ends, converts both points to world
space, and penalizes normalized disagreement. LPIPS is optional validation
only and is imported only when enabled.

The optimizer is AdamW with peak LR `2e-4`, 1,000-step linear warmup, cosine
decay, BF16 autocast, gradient clipping, and finite parameter/gradient/loss
guards. During training the Head is recomputed for each target view and that
view is backpropagated immediately. This costs additional U-Net compute but
keeps only one CUDA rasterizer graph live at a time. Checkpoints use
`spherical_selfi_gaussian_head_v1` and contain Head,
config, optimizer, scheduler, step, metrics, adapter SHA, and PanoVGGT config;
frozen weights are not copied.

## Diagnostics and acceptance

Validation reports latitude-weighted L1/PSNR/SSIM for:

- all-source rendering;
- self-only rendering;
- leave-one-out rendering.

It also records per-source mean opacity and depth-residual, scale, confidence
quantiles. If all-source improves without leave-one-out improvement, the run is
reported as source-copy degeneration rather than cross-view reconstruction.
Local PNG panels and the same W&B image include target, all-source render, RGB
error, depth residual, and confidence.

Tests cover dynamic/odd shapes, the 39-channel contract, positive scales,
quaternion composition, SH basis/direction, pruning without canonical mutation,
ERP seam equivariance, latitude footprint, geometry updates, source-only
sampling, checkpoint resume/SHA guards, periodic losses, and unchanged
`FrontendOutput`. The real renderer one-step smoke is CUDA-only.

Remote resource gates are:

1. run pure unit tests;
2. run CUDA `126 x 252` Head/renderer smoke;
3. run CUDA `252 x 504` smoke;
4. run `504 x 1008`, `S=4` forward/backward and record peak VRAM, Gaussian
   count, and rasterizer time.

All remote commands must use the `pfgs360` Python, run long jobs inside `tmux`,
and retain W&B plus visualization. Before any run, recheck RAM, swap, Python
jobs, and `nvidia-smi`. Full resolution is accepted only if the renderer GPU
retains about 2 GiB free; OOM must be recorded and must not be hidden by
silently reducing views or resolution.

Example smoke commands (after the normal server preflight) are:

```bash
/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python \
  training/train_spherical_selfi_gaussian_head.py \
  --config configs/stage2_spherical_selfi_gaussian_head_airsim.yaml \
  --manifest data/stage1_airsim_dtw_nyc_debug_manifest.json \
  --head-height 126 --head-width 252 --max-steps 1 \
  --output-dir outputs/stage2_smoke_126x252 --wandb-mode online

/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python \
  training/train_spherical_selfi_gaussian_head.py \
  --config configs/stage2_spherical_selfi_gaussian_head_airsim.yaml \
  --manifest data/stage1_airsim_dtw_nyc_debug_manifest.json \
  --head-height 252 --head-width 504 --max-steps 1 \
  --output-dir outputs/stage2_smoke_252x504 --wandb-mode online
```

The formal full-resolution command omits the head-size overrides. These are
long-running GPU jobs and must be started inside `tmux`; the trainer records
peak VRAM, materialized count, materialization time, and rasterizer time.

## Stage 3 boundary

The observation retains frame ID, UV, ray, initial/refined depth, camera-local
rotation/SH, confidence, and validity. Stage 3 may call `with_geometry()` after
optimizing pose/depth and reuse the same 24D adapter feature for matching. The
feature itself is not stored as a Gaussian parameter. Future runtime work must
write refined pose, inverse depth, world points, confidence, and mask only into
the existing `FrontendOutput` fields; observations remain a tracker-internal
sidecar indexed by frame ID.
