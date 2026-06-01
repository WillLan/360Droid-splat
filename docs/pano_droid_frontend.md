# PanoDROID Frontend

The frontend accepts ERP frame streams and exposes the requested SLAM interface:

- `PanoFrame`
- `FrontendOutput`
- `PanoDROIDFrontend`
- `PanoDroidGraphTracker`
- `PanoFactorGraph`
- `PanoDROIDFrontendAdapter` as the backward-compatible graph wrapper

The MVP keeps DROID-style module boundaries:

- DROID-style feature encoder (`fnet`) with ordinary 2D convolutions
- DROID-style context encoder (`cnet`) that returns `hidden, context`
- spherical correlation pyramid with seam-aware ERP sampling
- `SphereConvGRU` update block using BlueHorn/SphereNet-style spherical convolution
- update heads for spherical flow target deltas, confidence/edge weights,
  inverse depth, graph damping/upmask, and keyframe score
- a runtime `PanoFactorGraph` that keeps active/inactive factors, edge hidden
  state, refined pose/depth state, and graph BA diagnostics during SLAM inference

ERP geometry follows the original backend convention in `utils/erp_geometry.py`:
+X right, +Y down, +Z forward.

## Spherical Pieces

- `spherical_camera.py` implements ERP/bearing conversion, tangent bases,
  log residuals, area weights, horizontal wrap, and seam-aware deltas.
- `sphere_conv.py` implements BlueHorn/SphereNet-style ERP spherical kernel
  sampling with `grid_sample`; the previous padding-only convolution has been
  removed.
- `sphere_gru.py` replaces 3x3/5x5 convolutions in ConvGRU gates with the new
  `SphereConv2d`; 1x1 gates remain ordinary Conv2d.
- `projective_ops.py` is the single shared ERP projection/residual path used by
  model BA, losses, and standalone BA utilities.
- `dense_ba.py` provides the default PyTorch `SphericalDenseBA` normal-equation
  layer used by graph training and inference.
- `spherical_ba.py` provides standalone spherical BA/loss utilities.

The pairwise `PanoDroidModel.forward(image0, image1)` path is legacy smoke-test
compatibility.  SLAM inference should use `PanoDroidGraphTracker`, which delegates
optimization to `PanoFactorGraph` and outputs refined pose, depth, confidence,
and BA residual.
