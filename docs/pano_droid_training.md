# PanoDROID Training

Run:

```bash
python -m frontend.pano_droid.train --config configs/pano_droid_train.yaml
```

The default config uses `SyntheticPanoPairDataset`, which creates shifted ERP
pairs with supervised flow, inverse depth, and relative pose targets.  This is
for smoke and tiny overfit validation.  It is not evidence of performance on a
real sequence.

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
and `best.pt`.  Inference uses:

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
2. Train the frontend only:

   ```bash
   python -m frontend.pano_droid.train --config configs/pano_droid_train.yaml
   ```

3. Verify `outputs/pano_droid_train/checkpoints/latest.pt` and `best.pt`.
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

The first graph trainer uses supervised RGB+pose+depth losses with sampled
spherical projection residuals.  It is designed as a correctness-first
PyTorch implementation; large-scale CUDA BA acceleration is a later step.

During graph training, diagnostics are written under
`Training.output_dir/visualizations`:

- `step_XXXXXXX_trajectory.png`: predicted trajectory against GT trajectory
- `step_XXXXXXX_depth.png`: predicted depth, GT depth, and absolute error
- `step_XXXXXXX_metrics.json`: trajectory RMSE and depth MAE for the visualized
  batch
