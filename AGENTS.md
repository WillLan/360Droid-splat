# Project Operating Rules

## Remote experiment safety

- For experiments on server `50902` at `lanboyang@172.19.53.39`, always run jobs inside `tmux` so the experiment survives terminal disconnects.
- Do not run more than 2 experiment groups at the same time.
- Run these experiments in the `pfgs360` conda environment:
  `/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python`
- Each concurrent experiment group should use one GPU; at most 2 GPUs may be used by this experiment queue at once.
- CPU and system memory are the primary safety constraints. Monitor CPU load, RAM, and swap while experiments are running. If the server is under heavy CPU load, close to memory pressure, or swap begins growing, pause launching new runs and report the status.
- Avoid commands or job layouts that can occupy excessive CPU/RAM and make the server unresponsive or prevent SSH login, even if GPU memory appears available.
- Preserve existing experiment outputs. Do not delete or overwrite result directories unless the user explicitly asks.

## Training visualization and W&B logging

- Every remote training run for this project must keep diagnostic visualization enabled and must log the generated visualizations to Weights & Biases.
- PanoDROID graph training should save the local PNG diagnostics under
  `Training.output_dir/visualizations` and log the same images to W&B as
  `diagnostics/trajectory_3d` and `diagnostics/depth_pred_gt_error`.
- Do not disable W&B just because online sync is unavailable. If the server
  cannot reach W&B, run W&B in `offline` mode and preserve the offline run
  directory so it can be synced later.
- Local smoke tests and unit tests may keep W&B disabled to avoid requiring
  external credentials, but any launched experiment on `50902` should use
  `WeightsAndBiases.enabled=true` and `Visualization.enabled=true`.

## 360UAV experiment baseline

- When the user asks to repeat this experiment setup, use the configuration corresponding to:
  `/mnt/disk1/lanboyang/Project/360GS-SLAM/results/debug_pfgs360_full200_neural_sky_fix123_dia_mv_global_prune_v2d_pfgs360_far_guard`
- The 360UAV sequence root for the 5-sequence experiment is:
  `/mnt/disk1/lanboyang/Datasets/360uav/seqs`
- After all sequences finish, summarize the experiment results in a concise table and include failed or incomplete runs explicitly.

## Code update deployment workflow

When code is modified for this project, use the following deployment workflow by default:

Project mapping:

- Local project root: `E:\Project\360Droid-splat`
- GitHub repository: `https://github.com/WillLan/360Droid-splat.git`
- Git remote name: `origin`
- Default branch: `main`
- Remote server: `50902` via `lanboyang@172.19.53.39`
- Remote project root: `/mnt/disk1/lanboyang/Project/360Droid-splat`
- Read-only reference source: `/mnt/disk1/lanboyang/Project/360GS-SLAM`

Do not push, pull, clone, or deploy this project into the old
`/mnt/disk1/lanboyang/Project/360GS-SLAM` directory.  That directory remains a
read-only reference for migration and comparison.

1. Make the local code change and run an appropriate lightweight check, such as:

   ```powershell
   git status --short --branch
   python -m compileall frontend mapping backend system tests
   python -m pytest
   ```

2. Commit the local change with a clear message.
3. Before uploading, ask the user to confirm whether to push the commit.
4. Only if the user confirms upload, push the commit to `origin main`.
5. After a successful push, update server `50902` only under the mapped remote project root.
   If the remote project directory does not exist, clone it with:

   ```bash
   ssh lanboyang@172.19.53.39 'git clone https://github.com/WillLan/360Droid-splat.git /mnt/disk1/lanboyang/Project/360Droid-splat'
   ```

6. If the remote project directory already exists, first inspect the worktree.
   Only fast-forward pull when the remote worktree is clean:

   ```bash
   ssh lanboyang@172.19.53.39 'cd /mnt/disk1/lanboyang/Project/360Droid-splat && git status --short && git pull --ff-only origin main'
   ```

7. If the remote repository has local uncommitted or untracked files that block
   `git pull`, report the blocking files and ask before moving, deleting, or
   overwriting anything.
8. Do not automatically start or restart experiments after pulling code on the
   server. Experiments must only start when the user gives a separate explicit
   command.
9. Training launch still follows the remote experiment safety rules: check
   `free -h`, active Python jobs, and `nvidia-smi`; run long jobs inside
   `tmux`; and use
   `/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python`.
