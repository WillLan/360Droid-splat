# PanoDROID Frontend

The frontend accepts ERP image pairs and exposes the requested SLAM interface:

- `PanoFrame`
- `FrontendOutput`
- `PanoDROIDFrontend`
- `PanoDROIDFrontendAdapter`

The MVP keeps DROID-style module boundaries:

- DROID-style feature encoder (`fnet`) with ordinary 2D convolutions
- DROID-style context encoder (`cnet`) that returns `hidden, context`
- spherical correlation pyramid with seam-aware ERP sampling
- `SphereConvGRU` update block using BlueHorn/SphereNet-style spherical convolution
- update heads for spherical flow, confidence, inverse depth, damping, pose,
  and keyframe score

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
- `spherical_ba.py` provides differentiable PyTorch spherical BA utilities.

The first version is trainable but intentionally compact.  Replacing the small
encoders with original DROID modules should happen behind the current class
interfaces.
