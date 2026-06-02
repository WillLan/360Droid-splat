# PanoDROID Training

Legacy pairwise smoke training:

```bash
python -m frontend.pano_droid.train --config configs/pano_droid_train.yaml
```

The pairwise trainer uses `SyntheticPanoPairDataset`, which creates shifted ERP
pairs with supervised flow, inverse depth, and relative pose targets.  This is
for smoke and tiny overfit validation only.  The primary DROID-style path is
`frontend.pano_droid.train_graph`.

For real data, set:

```yaml
Dataset:
  synthetic: false
  dataset_path: /path/to/erp_dataset
  sequence: null
  erp_resize_height: 512
  erp_resize_width: 1024
```

Checkpoints are saved under `Training.output_dir/checkpoints` as `latest.pt`
and `best.pt`.  Inference defaults to the graph tracker:

```bash
python -m frontend.pano_droid.infer \
  --checkpoint outputs/pano_droid_train/checkpoints/latest.pt \
  --image0 frame_000.png \
  --image1 frame_001.png \
  --output outputs/pano_droid_infer.npz
```

## Recommended Training Flow

1. Set `Dataset.synthetic: false`, `Dataset.dataset_path`, optional
   `Dataset.sequence`, and ERP resize to the training resolution.
2. Train the graph frontend:

   ```bash
   python -m frontend.pano_droid.train_graph --config configs/pano_droid_train_panocity_beijing.yaml
   ```

3. Verify `Training.output_dir/checkpoints/latest.pt` and `best.pt`.
4. Put the checkpoint path into `Frontend.checkpoint` in
   `configs/pano_droid_gs_slam.yaml`.
5. Run the full SLAM system:

   ```bash
   python -m system.pano_droid_gs_slam --config configs/pano_droid_gs_slam.yaml
   ```

## PanoCity Beijing Graph Training

The DROID-style multi-frame trainer is available as:

```bash
python -m frontend.pano_droid.train_graph \
  --config configs/pano_droid_train_panocity_beijing.yaml \
  --max-steps 100
```

On server `50902`, run it inside `tmux` and use the `pfgs360` Python:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 /mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python \
  -m frontend.pano_droid.train_graph \
  --config configs/pano_droid_train_panocity_beijing.yaml \
  --max-steps 100
```

The PanoCity Beijing config expects:

- dataset root: `/mnt/disk1/zwh/Dataset/PanoCity/beijing`
- block layout: `beijing_block*/pano_images`, `panodepth_images`,
  `*_poses.json`
- training resize: height `512`, width `1024`
- graph clip length: `n_frames=7`

`train_graph` is now the primary DROID-style trainer. It extracts all frame
features once, keeps per-edge recurrent hidden state, feeds encoded correlation
plus encoded motion into the update block, and predicts DROID-style
`delta/weight/eta/upmask` heads.  The edge weight is 2-channel per-coordinate
weight, not the legacy scalar confidence map.  After every update step it runs
`SphericalDenseBA` on the feature-grid ERP residual to refine pose and inverse
depth. The default PanoCity model profile is `droid_base` (`128/128/128`
feature/context/hidden channels with image normalization); tiny profiles remain
for smoke tests. The runtime SLAM frontend now uses the same graph path through
`PanoFactorGraph` instead of the legacy pairwise pose head.

The PyTorch `SphericalDenseBA` implementation is correctness-first: it uses
closed-form ERP pixel Jacobians, shared projection ops, wrapped pixel residuals,
valid/depth masks, LM damping, fixed-frame handling, bounded updates, and a
pose-depth Schur complement.  It is the default graph training/inference path,
while a future CUDA backend can replace the same interface for speed.

Important graph options:

- `Graph.edge_strategy: mixed`: alternates temporal and spherical
  projection-distance graph construction when depth is available.
- `Graph.edge_pose_source: init`: builds proximity edges from the same warm-start
  poses used by training, reducing the train/inference graph-selection gap.
- `Graph.max_edges_per_step`: randomly samples edges per batch instead of
  prefix truncating them.  The PanoCity default is `24`, close to the DROID
  training factor count for short clips.
- `Graph.ba_iters_per_update`, `Graph.ba_sample_stride`, `Graph.fixed_frames`:
  control the feature-resolution BA loop.
- `Graph.init_mode`, `Graph.init_noise_prob`, `Graph.init_identity_prob`, and
  `Graph.init_noise_std`: keep DROID GT-anchor training while mixing off-GT
  starts for inference robustness.
- `Graph.loss_gamma`: gamma weighting across recurrent refined states.
- `Training.scheduler: onecycle`, `Training.restart_prob`, and
  `Training.resume_checkpoint`: match the DROID-SLAM long-training cadence more
  closely.
- `Training.freeze_legacy_pairwise: true`: freezes `pose_head` and the old
  damping head so DDP can use `find_unused_parameters: false` in graph-only
  training.

Default graph losses follow the DROID-style main supervision:
`geodesic_loss + residual_loss + flow_loss` with gamma weighting over recurrent
refined states.  A sampled full-resolution flow term supervises
`refined_inverse_depth_full`, so the convex upsampling mask is trained by
geometry instead of only reported as a diagnostic.  Depth L1, smoothness, and
residual-aware confidence calibration remain available as auxiliary terms, but
the PanoCity config keeps them disabled by default so they do not dominate the
graph/BA objective.

At startup, rank 0 records dataset sanity statistics in
`Training.output_dir/data_stats.json`: image range, depth range, invalid-depth
ratio, and pose translation scale.  The same stats are stored in checkpoints
alongside config/git metadata to make long runs reproducible.  Training metrics
also include per-module grad norms and unused trainable parameter counts for
catching accidental legacy-path participation.

During graph training, diagnostics are written under
`Training.output_dir/visualizations`:

- `step_XXXXXXX_trajectory.png`: 3D predicted trajectory against GT
  trajectory.  GT uses circle markers, prediction uses triangle markers, and
  the colorbar shows frame index.
- `step_XXXXXXX_depth.png`: three panels from left to right: predicted depth,
  GT depth, and absolute depth error.
- `step_XXXXXXX_metrics.json`: trajectory RMSE, depth MAE, and BA residual for
  the visualized batch

For remote experiments, keep both `Visualization.enabled=true` and
`WeightsAndBiases.enabled=true`.  Every time graph diagnostics are generated,
rank 0 must log the same images to W&B:

- `train/*`: scalar losses and training metrics
- `diagnostics/trajectory_3d`: 3D GT-vs-pred trajectory image
- `diagnostics/depth_pred_gt_error`: predicted depth, GT depth, and absolute
  error image
- `diagnostics/trajectory_rmse` and `diagnostics/depth_mae`

W&B authentication uses the normal `wandb login` flow on the training server.
The configured account owner is `zb2302106@buaa.edu.cn`, but W&B requires an API
key or an existing login session rather than an email-only login.
If online sync is unavailable, launch with W&B `offline` mode and keep the
offline run directory for later `wandb sync`; do not disable W&B for remote
training runs.

For 4-GPU training on 50902 GPUs 4, 5, 6, and 7, launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python \
  -m torch.distributed.run --nproc_per_node=4 \
  -m frontend.pano_droid.train_graph \
  --config configs/pano_droid_train_panocity_beijing.yaml \
  --wandb \
  --run-name pano_droid_panocity_beijing_gpus4567
```
