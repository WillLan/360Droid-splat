# Project Operating Rules

## Interactive research coding workflow

- For research design, algorithm changes, and non-trivial code improvements,
  use a clarification-first workflow.
- Before editing code for a non-trivial change, read the relevant files and
  report the current task interpretation, uncertain design points, 2-3 concrete
  options for each important uncertainty, and the recommended option with its
  tradeoff.
- Do not edit code until the user confirms the design when the task affects
  model architecture, loss functions or residual definitions, training data,
  sampling, metrics, validation protocol, config schema or defaults, public
  APIs, saved output formats, or experiment launch behavior.
- If there are multiple reasonable implementations, stop and ask. Do not
  silently choose one for algorithmic, data, config, output, or experiment
  behavior.
- If the user request is underspecified, ask clarification questions instead of
  filling important gaps by assumption.
- For simple local fixes, typo fixes, formatting, clearly reversible
  implementation details, or edits that only apply a user-confirmed design,
  proceed without asking, but state the assumption briefly.
- When asking, provide 2-3 options, mark one as recommended, and explain the
  tradeoff in one short sentence.
- If the user says "先讨论", "先给方案", "不要直接改", "让我确认",
  "多问我", or equivalent wording, stop after the design/options and wait for
  confirmation before editing code.

## PanoVGGT-M3-Sphere implementation guardrails

- Current task goal: plan and implement a config-gated PanoVGGT-M3-Sphere
  extension for ERP Gaussian-SLAM that adds M3-style dense matching, dense
  correspondence factors, and spherical dense BA before existing PanoVGGT chunk
  alignment.
- The new functionality must be config-gated. Default behavior must remain off
  unless the config explicitly enables the PanoVGGT-M3-Sphere path.
- The disabled path must preserve the existing `panovggt_long` behavior.
  Existing configs and tests that do not opt into the new path should keep the
  same tracker, engine, alignment, and output behavior.
- Do not break or expand the public `FrontendOutput` API. Refined
  pose/depth/world-points must enter the existing fields already consumed by
  mapping and backend code.
- The matching head output spatial size must equal the matching head input
  feature spatial size. Any resize to ERP image resolution must be explicit and
  local to a caller that needs it.
- `descriptor_dim` defaults to `24`.
- Do not hard-code any input image resolution. Infer height, width, and feature
  grid size from tensors or config values at runtime.
- The dense BA main residual must be an `S^2` tangent residual,
  `r = Log_b_star(b_hat)` in `R^2`, where `b_star` is the matched target
  bearing and `b_hat` is the predicted target bearing from current pose/depth.
- ERP pixel residuals are allowed only for debug, diagnostics, or explicit
  ablations. They must not be the main dense BA residual for
  PanoVGGT-M3-Sphere.
- Fake descriptors are allowed only in explicit fake or synthetic test modes.
  Real inference must raise a clear error or use an explicitly configured
  fallback when matching features or the matching head are unavailable.
- Unit tests must not depend on real PanoCity data, AirSim360-Scene data, or a
  real external PanoVGGT checkpoint.

## Remote experiment safety

- For experiments on server `50902` at `lanboyang@172.19.53.39`, always run jobs inside `tmux` so the experiment survives terminal disconnects.
- Do not run more than 2 experiment groups at the same time.
- Run these experiments in the `pfgs360` conda environment:
  `/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python`
- Full SLAM / end-to-end mapping experiment groups should use one GPU per group;
  at most 2 GPUs may be used by those full-pipeline experiment queues at once.
- PanoVGGT-M3-Sphere head training is the exception: it may use up to 2 GPUs
  per head-training experiment group, and at most 4 GPUs total across the
  concurrent head-training queue. Keep this to at most 2 concurrent groups
  such as one `sky_only` group and one `matching_only` group.
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
