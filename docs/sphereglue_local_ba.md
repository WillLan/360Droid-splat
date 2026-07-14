# SphereGlue local-BA experiment dependency

The `superpoint_sphereglue` local-BA matcher is an optional, research-only
experiment path. This repository does not vendor SphereGlue, the SuperPoint
implementation, or either project's pretrained weights.

Configure three external paths under
`SphericalSelfiRuntime.local_ba.matching`:

```yaml
type: superpoint_sphereglue
lightglue_repo: /path/to/LightGlue
sphereglue_repo: /path/to/SphereGlue
superpoint_checkpoint: /path/to/superpoint_v1.pth
sphereglue_checkpoint: /path/to/SphereGlue/checkpoint.pt  # or .safetensors
```

At startup the adapter checks that these files exist and records SHA-256
digests for the LightGlue SuperPoint source, SuperPoint checkpoint, SphereGlue
source, and SphereGlue checkpoint in the per-window matching metadata. External code and weights must
be obtained and used under their own licenses. In particular, do not copy or
redistribute the SuperPoint pretrained weights through this repository.

The rest of the BA pipeline is unchanged: SphereGlue correspondences are
converted to the existing `Stage3MatchCache`, then the same spherical
`BlockSparseSphericalBA` solver is used. The global graph continues to use the
Adapter/Fibonacci matcher in all experiment arms.
