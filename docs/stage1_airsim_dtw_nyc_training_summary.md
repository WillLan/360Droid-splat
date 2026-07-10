# Stage 1.5 Airsim DTW/NYC Training Summary

Status: formal training completed; the best validated checkpoint is the Stage 2 baseline.

## Data

- dataset root: `/mnt/disk1/lanboyang/Datasets/Airsim360/Omni360-Scene`
- scenes: `DTW`, `NYC`
- manifest: `data/stage1_airsim_dtw_nyc_manifest.json`
- debug manifest: `data/stage1_airsim_dtw_nyc_debug_manifest.json`
- full records: `11382`
- debug records: `256`
- input resolution: `504x1008`

## Training Config

- config: `configs/stage1_spherical_selfi_adapter_airsim_dtw_nyc.yaml`
- PanoVGGT: frozen
- adapter output: `B x V x 24 x 504 x 1008`
- formal training steps: `50000`
- W&B: online
- visualization: enabled every `1000` steps

## Required Preflight

```bash
python tools/dump_panovggt_feature_shapes.py \
  --config configs/stage1_spherical_selfi_adapter_airsim_dtw_nyc.yaml \
  --manifest data/stage1_airsim_dtw_nyc_debug_manifest.json \
  --num-samples 2 \
  --device cuda
```

Expected:

- input image shape: `B,V,3,504,1008`
- PanoVGGT requires grad: `false`
- adapter requires grad: `true`
- adapter output shape: `B,V,24,504,1008`
- adapter channel norm mean: near `1.0`

## Smoke Training Command

Run inside `tmux` on server `50902`:

```bash
cd /mnt/disk1/lanboyang/Project/360Droid-splat
export PYTHONPATH=/mnt/disk1/lanboyang/Project/PanoVGGT:$PYTHONPATH

/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python training/train_spherical_selfi_adapter.py \
  --config configs/stage1_spherical_selfi_adapter_airsim_dtw_nyc.yaml \
  --manifest data/stage1_airsim_dtw_nyc_debug_manifest.json \
  --max_steps 200 \
  --log_interval 10 \
  --val_interval 50 \
  --save_interval 100 \
  --wandb-mode online
```

## Formal Training Command

Run only after smoke training passes:

```bash
/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python training/train_spherical_selfi_adapter.py \
  --config configs/stage1_spherical_selfi_adapter_airsim_dtw_nyc.yaml \
  --wandb-mode online
```

## Results

- completed step: `39,000`
- best validation mean angular error: `0.2859647 deg`
- recommended checkpoint:
  `/mnt/disk1/lanboyang/Project/360Droid-splat/outputs/stage1_selfi_adapter_airsim_dtw_nyc_fullres_spherical_ce_depth20_ddp2/checkpoints/best_val_angular_error.pt`
- checkpoint format: `spherical_selfi_adapter_v1`
- SHA256: `95cdbb5404acba8654e868335ede6a00c281451d7ef7c104d3a10f63fc3730b9`
- Stage 2 policy: load this checkpoint strictly, keep the adapter in `eval()` with
  `requires_grad=False`, and do not copy its weights into Stage 2 checkpoints.
